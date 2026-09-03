import re
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .aggregate import ContextState
from ..engine.base import Engine
from ..groups.base import Group
from ..groups.filter import FilterRule, passes_filters, passes_output_filters, passes_feedback_filters
from ..groups.map import node_contribution
from .attractor import context_attract_ids
from .puller import Puller
from .store import Store

if TYPE_CHECKING:
    from ..prompts import PromptSet

_STEP_RE = re.compile(r"element:\s*(\S[^\n]*)", re.IGNORECASE)
_NOTE_RE = re.compile(r"narrative:\s*([^\n]+)", re.IGNORECASE)


def _parse_step(text: str) -> dict:
    element = _STEP_RE.search(text)
    note = _NOTE_RE.search(text)
    return {
        "element": element.group(1).strip() if element else "",
        "note": note.group(1).strip() if note else "",
    }


def _build_llamacpp_logit_chain(filters: list[FilterRule], engine):
    """Build LogitsProcessorChain for LlamaCppEngine from logit-level filter rules."""
    from ..engine.logits_chain import LogitsProcessorChain
    from ..engine.processors import LanguageRedirect

    logit_rules = [f for f in filters if f.enabled and f.mode == "logit" and f.pipeline == "output"]
    if not logit_rules:
        return None

    from ..groups.token_group import TokenGroup

    k = logit_rules[0].group_dim
    tg = TokenGroup(k=k)
    vocab_size = engine.vocab_size()

    chain = LogitsProcessorChain()
    for rule in logit_rules:
        if rule.type == "language":
            try:
                lang_ids = set(tg.build_lang_token_ids_from_tokenizer(
                    rule.value, vocab_size, engine.tokenize, engine.detokenize))
            except Exception:
                lang_ids = tg.token_ids_for_lang(rule.value, vocab_size)

            mask = tg.build_exclude_mask_from_tokens(lang_ids, vocab_size)

            if rule.action == "exclude":
                chain.add(LanguageRedirect(exclude_mask=mask))
            else:
                chain.add(LanguageRedirect(boost_mask=mask))

    return chain if len(chain) > 0 else None


class AutoPullLoop:
    def __init__(self, engine: Engine, store: Store, puller, group: Group,
                 prompts: "PromptSet", max_steps: int = 6, pull_every: int = 2,
                 max_active: int = 16, filters=None, vocab_index=None,
                 context_attract_max_nodes: int = 4, context_attract_top_k: int = 20,
                 context_attract_min_similarity: float = 0.6):
        self.engine = engine
        self.store = store
        self.puller = puller
        self.group = group
        self.prompts = prompts
        self.max_steps = max_steps
        self.pull_every = pull_every
        self.max_active = max_active
        self.filters = filters or []
        # Context -> attract linking (ARCHITECTURE.md §10): when a VocabIndex
        # is supplied, generation is dynamically pulled toward the vocabulary
        # of whatever's currently pulled into the active window, recomputed
        # after every pull cycle. Opt-in and None by default -- unrelated to
        # correctness of the retrieval loop itself, purely an extra nudge.
        self.vocab_index = vocab_index
        self.context_attract_max_nodes = context_attract_max_nodes
        self.context_attract_top_k = context_attract_top_k
        self.context_attract_min_similarity = context_attract_min_similarity

    def _step_grammar(self) -> str | None:
        return self.prompts.step_grammar(self.group)

    def _make_logits_processor(self):
        """Создать LogitsProcessorChain из logit-level фильтров (pre-sampling,
        нативно через LlamaCppEngine — единственный поддерживаемый движок,
        см. ARCHITECTURE.md §11 про отказ от ServerEngine)."""
        return _build_llamacpp_logit_chain(self.filters, self.engine)

    def _blocked_ranges(self):
        """Вернуть диапазоны заблокированных символов для runtime-фильтра.

        Предвычисленной маски токенов недостаточно: BPE-декодирование в
        контексте может превращать «мусорные» токены в валидные CJK-символы.
        Поэтому при исключении языка накладываем ещё посимвольный фильтр.
        """
        from ..engine.llamacpp import DEFAULT_BLOCKED_RANGES
        excludes = [f for f in self.filters
                    if f.enabled and f.mode == "logit" and f.pipeline == "output"
                    and f.type == "language" and f.action == "exclude"]
        if excludes:
            return DEFAULT_BLOCKED_RANGES
        return None

    def run(self, task: str, state: ContextState | None = None) -> dict:
        if state is None:
            state = ContextState(self.group)
            order: deque[str] = deque()
            fragments: list[str] = []
        else:
            order = deque(state.active_ids())
            fragments = list(order)
        log: list[dict] = []
        tail = task
        steps: list[dict] = []
        logits_processor = self._make_logits_processor()
        blocked_ranges = self._blocked_ranges()
        attract_ids: set[int] = set()

        for n in range(1, self.max_steps + 1):
            prompt = self.prompts.step(self.group, task, state, tail, self.store, fragments, filters=self.filters)
            out = self.engine.generate(prompt, grammar=self._step_grammar(),
                                       logits_processor=logits_processor,
                                       blocked_ranges=blocked_ranges,
                                       attract_ids=attract_ids or None)
            parsed = _parse_step(out)
            steps.append({"n": n, **parsed})
            tail = out[-1200:]

            if n % self.pull_every == 0:
                query = self.engine.embed(tail)
                hits = self.puller.pull(query, exclude=state.active_ids(), filters=self.filters)
                for score, node in hits:
                    if node.embedding is None and self.group.name == "vector":
                        continue
                    contrib = node_contribution(node.node_id, node.embedding, self.group, score)
                    state.add(node.node_id, contrib)
                    order.append(node.node_id)
                    if node.node_id not in fragments:
                        fragments.append(node.node_id)
                    log.append({"step": n, "node": node.node_id, "score": round(score, 4)})
                while len(order) > self.max_active:
                    old = order.popleft()
                    state.remove(old)
                    if old in fragments:
                        fragments.remove(old)
                    log.append({"step": n, "node": old, "event": "forgot"})

                # §10: re-derive the attract target from whatever's active now,
                # so it tracks the window as Puller swaps material in and out.
                if self.vocab_index is not None:
                    attract_ids = context_attract_ids(
                        self.store, order, self.vocab_index,
                        max_nodes=self.context_attract_max_nodes,
                        top_k=self.context_attract_top_k,
                        min_similarity=self.context_attract_min_similarity,
                    )

            if not passes_output_filters(out, self.filters, self.filters):
                log.append({"step": n, "event": "output_filtered", "text": out[:80]})
                tail = ""

            if not passes_feedback_filters(tail, self.filters, self.filters):
                log.append({"step": n, "event": "feedback_filtered"})
                tail = ""

        final_prompt = self.prompts.final(self.group, task, state, filters=self.filters)
        answer = self.engine.generate(final_prompt, grammar=self.prompts.final_grammar(),
                                      logits_processor=logits_processor,
                                      blocked_ranges=blocked_ranges,
                                      attract_ids=attract_ids or None)

        if not passes_output_filters(answer, self.filters, self.filters):
            log.append({"event": "final_output_filtered", "text": answer[:80]})

        return {
            "answer": answer,
            "steps": steps,
            "context": {
                "h": self.group.to_text(state.h),
                "pulled": log,
                "active_count": len(state),
            },
        }

import re
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .aggregate import ContextState
from ..engine.base import Engine
from ..groups.base import Group
from ..groups.filter import FilterRule, passes_filters, passes_output_filters, passes_feedback_filters
from ..groups.logit_filter import LogitFilter
from ..groups.map import node_contribution
from ..groups.token_group import TokenGroup
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


def _build_logit_filters(filters: list[FilterRule], vocab_size: int = 150000,
                         engine=None):
    """Build LogitFilter for ServerEngine (logit_bias based)."""
    logit_rules = [f for f in filters if f.enabled and f.mode == "logit" and f.pipeline == "output"]
    if not logit_rules:
        return None, None

    from ..engine.server import ServerEngine
    from ..groups.logit_filter import LogitFilter
    from ..groups.token_group import TokenGroup

    k = logit_rules[0].group_dim
    tg = TokenGroup(k=k)
    lf = LogitFilter(tg, vocab_size=vocab_size)

    exclude_masks = []
    allow_masks = []

    use_server = isinstance(engine, ServerEngine)

    for rule in logit_rules:
        if rule.type == "language":
            mask = None

            if use_server:
                try:
                    lang_ids = engine.build_lang_token_ids(rule.value, vocab_size)
                    mask = tg.build_exclude_mask_from_tokens(lang_ids, vocab_size)
                except Exception:
                    pass

            if mask is None:
                mask = tg.lang_to_exclude_set(rule.value, vocab_size)

            if rule.action == "exclude":
                exclude_masks.append(mask)
            else:
                allow_masks.append(mask)

    return lf, {"exclude_masks": exclude_masks or None, "allow_masks": allow_masks or None}


class AutoPullLoop:
    def __init__(self, engine: Engine, store: Store, puller, group: Group,
                 prompts: "PromptSet", max_steps: int = 6, pull_every: int = 2,
                 max_active: int = 16, filters=None):
        self.engine = engine
        self.store = store
        self.puller = puller
        self.group = group
        self.prompts = prompts
        self.max_steps = max_steps
        self.pull_every = pull_every
        self.max_active = max_active
        self.filters = filters or []

    def _step_grammar(self) -> str | None:
        return self.prompts.step_grammar(self.group)

    def _make_logits_processor(self):
        """Создать logits_processor из logit-level фильтров.

        LlamaCppEngine → LogitsProcessorChain (pre-sampling, native).
        ServerEngine → None (server не поддерживает logits_processor).
        """
        from ..engine.llamacpp import LlamaCppEngine
        if isinstance(self.engine, LlamaCppEngine):
            return _build_llamacpp_logit_chain(self.filters, self.engine)
        lf, filter_kwargs = _build_logit_filters(self.filters, engine=self.engine)
        if lf is None:
            return None
        return lf.to_logits_processor(**filter_kwargs)

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

    def _make_logit_bias(self) -> dict[int, float] | None:
        """Создать logit_bias dict для ServerEngine."""
        from ..engine.server import ServerEngine
        from ..engine.llamacpp import LlamaCppEngine
        if isinstance(self.engine, LlamaCppEngine):
            return None
        if not isinstance(self.engine, ServerEngine):
            return None
        logit_rules = [f for f in self.filters if f.enabled and f.mode == "logit" and f.pipeline == "output"]
        if not logit_rules:
            return None

        from ..groups.token_group import TokenGroup
        tg = TokenGroup(k=logit_rules[0].group_dim)
        vocab_size = 152064
        logit_bias: dict[int, float] = {}

        for rule in logit_rules:
            if rule.type == "language":
                try:
                    lang_ids = self.engine.build_lang_token_ids(rule.value, vocab_size)
                except Exception:
                    lang_ids = tg.token_ids_for_lang(rule.value, vocab_size)

                bias = -100.0 if rule.action == "exclude" else 0.0
                for tid in lang_ids:
                    if tid < vocab_size:
                        logit_bias[tid] = bias

        return logit_bias if logit_bias else None

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
        logit_bias = self._make_logit_bias()
        blocked_ranges = self._blocked_ranges()

        for n in range(1, self.max_steps + 1):
            prompt = self.prompts.step(self.group, task, state, tail, self.store, fragments, filters=self.filters)
            out = self.engine.generate(prompt, grammar=self._step_grammar(),
                                       logits_processor=logits_processor,
                                       logit_bias=logit_bias,
                                       blocked_ranges=blocked_ranges)
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

            if not passes_output_filters(out, self.filters, self.filters):
                log.append({"step": n, "event": "output_filtered", "text": out[:80]})
                tail = ""

            if not passes_feedback_filters(tail, self.filters, self.filters):
                log.append({"step": n, "event": "feedback_filtered"})
                tail = ""

        final_prompt = self.prompts.final(self.group, task, state, filters=self.filters)
        answer = self.engine.generate(final_prompt, grammar=self.prompts.final_grammar(),
                                      logits_processor=logits_processor,
                                      logit_bias=logit_bias,
                                      blocked_ranges=blocked_ranges)

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

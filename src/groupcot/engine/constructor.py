"""Semantic concept constructor (ARCHITECTURE.md §5 / §3.3).

The constructor turns a natural-language *intent* ("forbid wrong tensor layer
sizes like [768, 758]") into a concrete set of token IDs that a logits
processor can suppress or attract. It does this via a **self-query**: the model
itself is asked to emit a structured specification of the forbidden/desired
concept, which is then compiled (lightweight variant, V3a) by tokenizing the
listed lexicon phrases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Set

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_META_PROMPT = (
    "You are a JSON-only semantic mask constructor.\n"
    "Output EXACTLY one JSON object and nothing else: no prose, no markdown, "
    "no code fences, no example lists.\n"
    'Schema: {"concept": str, "mode": "exclude"|"include"|"attract"|"constrain", '
    '"weight": float, "lexicon": [short verbatim phrases to block/boost], '
    '"prototypes": [str], "allowed": [str]}\n'
    "Keep 'lexicon' to at most 8 short phrases (a few tokens each).\n"
    "Example input: forbid the word 'cat'\n"
    'Example output: {"concept": "cat", "mode": "exclude", "weight": 1.0, '
    '"lexicon": ["cat", "cats"], "prototypes": [], "allowed": []}\n'
    "Now respond for this constraint only:\n"
)

_SYSTEM_PROMPT = (
    "You are a JSON-only semantic mask constructor. Output exactly one JSON "
    'object and nothing else: no prose, no markdown, no code fences. '
    'Schema: {"concept": str, "mode": "exclude"|"include"|"attract"|"constrain", '
    '"weight": float, "lexicon": [short verbatim phrases to block/boost], '
    '"prototypes": [str], "allowed": [str]}. '
    "Keep 'lexicon' to at most 8 short phrases (a few tokens each)."
)


@dataclass
class ConceptSpec:
    """Structured description of a concept, produced by the self-query."""

    concept: str = ""
    mode: str = "exclude"  # exclude | include | attract | constrain
    weight: float = 1.0
    lexicon: list[str] = field(default_factory=list)
    prototypes: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)

    def to_token_ids(self, tokenize_fn) -> Set[int]:
        """Compile the lexicon into a set of token IDs via ``tokenize_fn``.

        For each phrase we also consider leading-space and capitalized
        variants, because tokenizers encode a mid-sentence word as ``" word"``
        and sentence-initial words as ``"Word"``; excluding these maximizes
        suppression coverage for a verbatim-phrase block.
        """
        ids: Set[int] = set()
        for phrase in self.lexicon:
            variants = {phrase, " " + phrase}
            if phrase:
                variants.add(phrase.capitalize())
                variants.add(" " + phrase.capitalize())
            for variant in variants:
                try:
                    ids.update(int(t) for t in tokenize_fn(variant))
                except Exception:
                    continue
        return ids


class ConceptConstructor:
    """Build ``ConceptSpec`` from an intent via model self-query, then compile.

    Lightweight variant (V3a): the self-query yields an explicit lexicon which
    we tokenize directly. The full variant (V3b, embed + Puller over the group
    space) is a future extension.
    """

    def __init__(self, engine, group_dim: int = 64):
        self.engine = engine
        self.group_dim = group_dim

    def construct(self, intent: str, mode: str = "exclude") -> ConceptSpec:
        """Run the self-query and parse the resulting ``ConceptSpec``.

        Prefers the chat interface (reliable instruction-following on
        instruct models) and falls back to a completion prompt otherwise.
        """
        raw = None
        if hasattr(self.engine, "chat"):
            try:
                raw = self.engine.chat(
                    [{"role": "system", "content": _SYSTEM_PROMPT},
                     {"role": "user", "content": intent}],
                    max_tokens=512, temperature=0.0,
                )
            except Exception:
                raw = None
        if not raw:
            raw = self.engine.generate(_META_PROMPT + intent, max_tokens=512, temperature=0.0)
        spec = self._parse(raw)
        if not spec.mode or spec.mode not in ("exclude", "include", "attract", "constrain"):
            spec.mode = mode
        return spec

    @staticmethod
    def _parse(raw: str | None) -> ConceptSpec:
        if not raw:
            return ConceptSpec()
        m = _JSON_RE.search(raw)
        if not m:
            return ConceptSpec()
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return ConceptSpec()
        if not isinstance(data, dict):
            return ConceptSpec()
        return ConceptSpec(
            concept=str(data.get("concept", "")),
            mode=str(data.get("mode", "exclude")),
            weight=float(data.get("weight", 1.0)),
            lexicon=list(data.get("lexicon", []) or []),
            prototypes=list(data.get("prototypes", []) or []),
            allowed=list(data.get("allowed", []) or []),
        )

    def compile(self, spec: ConceptSpec, vocab_index=None, top_k: int = 40,
                min_similarity: float = 0.55) -> Set[int]:
        """Compile a spec into a set of token IDs (exclusion or attraction).

        Lightweight (V3a, always on): tokenizes the lexicon phrases directly.
        Full (V3b, opt-in via ``vocab_index``): additionally embeds the
        lexicon/prototypes and pulls in the nearest tokens from a prebuilt
        ``VocabIndex`` (ARCHITECTURE.md §6.3) — this is what catches
        morphological/semantic variants a literal lexicon misses (e.g. "cats",
        "kitten" for a "cat" concept). ``spec.allowed`` is subtracted from the
        semantic expansion so it stays a safety net, not a wider net.
        """
        ids: Set[int] = set()
        if hasattr(self.engine, "tokenize"):
            ids |= spec.to_token_ids(self.engine.tokenize)
        if vocab_index is not None:
            ids |= self._compile_semantic(spec, vocab_index, top_k, min_similarity)
        return ids

    def _compile_semantic(self, spec: ConceptSpec, vocab_index, top_k: int,
                           min_similarity: float) -> Set[int]:
        """V3b: embed(prototypes/lexicon) → nearest tokens in ``vocab_index``.

        Uses ``vocab_index.embed_query`` (not ``self.engine.embed``) so the
        seed lands in the same vector space the index was built in — those
        can be different engines (a dedicated embedding model vs. the
        generation model whose tokens are actually being masked).
        """
        seeds = list(spec.prototypes) + list(spec.lexicon)
        if not seeds and spec.concept:
            seeds = [spec.concept]
        ids: Set[int] = set()
        for seed in seeds:
            if not seed:
                continue
            try:
                vec = vocab_index.embed_query(seed)
            except Exception:
                continue
            for tid, _text, _sim in vocab_index.nearest(
                    vec, top_k=top_k, min_similarity=min_similarity):
                ids.add(tid)
        if spec.allowed and ids and hasattr(self.engine, "detokenize"):
            allowed = {a.strip().lower() for a in spec.allowed if a.strip()}
            ids = {tid for tid in ids if self._decode(tid).strip().lower() not in allowed}
        return ids

    def _decode(self, token_id: int) -> str:
        try:
            return self.engine.detokenize([token_id])
        except Exception:
            return ""

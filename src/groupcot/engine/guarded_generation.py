"""Sentence-level concept guard + reject/widen/regenerate loop.

ARCHITECTURE.md §5.1, "Фаза 4": token-level masking (ConceptSuppress/V3a/V3b)
can only ever forbid *words*. A concept like "поехать на море" isn't a word —
it's a whole-sentence meaning that "поехать на океан" or "поехать на рыбалку"
also carry despite sharing almost no tokens. Catching that requires embedding
and classifying complete sentences, which a token-by-token logits processor
cannot do (it never sees more than the next token).

The mechanism here therefore works one generated sentence at a time, and is
dual (§3.3 / §10): a concept spec is either **exclude** (forbid a meaning) or
**include**/**attract** (require staying near a meaning) —

    generate a sentence -> embed it -> compare to every concept's prototype
    sentences -> if it matches an exclude concept, or fails to match *any*
    include concept closely enough, don't keep it: retry the same sentence
    position with a wider/stronger pull toward the right answer.

The two directions widen differently, because they start from different
amounts of information:

- **exclude**: we don't know in advance what unanticipated phrasing will
  carry the forbidden meaning, so each rejection extracts the tokens that
  made *this specific leaked sentence* carry it (self-query seeded with the
  leak) and folds them into the exclusion mask — the mask can only ever grow.
- **include**: the desired concept is already fully specified by the spec's
  own prototypes/lexicon, so there's nothing new to learn from a drifted
  sentence — instead each rejection just pushes the existing attraction
  *harder*. On a ``LlamaCppEngine`` this escalates ``mix_alpha`` (probability
  mixing, ARCHITECTURE.md §12.1 — naturally bounded to [0,1], no per-model
  calibration needed); on any other engine it falls back to escalating the
  older additive ``attract_weight`` (§10.3 found this needs a hand-tuned cap
  per model and can degenerate into repeating one token — mixing exists
  specifically to fix that, but needs raw pre-sampling logits that only
  ``LlamaCppEngine`` exposes, §12.4).

Critically, in both directions each rejection changes something about the
next attempt before retrying — naively regenerating with an unchanged
mask/weight and non-zero temperature does not reliably escape (exclude) or
reach (include) the same semantic cluster, since the underlying distribution
hasn't changed. Changing it is what makes each retry provably move the
reachable space instead of resampling the same one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

from .constructor import ConceptConstructor, ConceptSpec
from ..context.puller import cosine

_COMPLETE_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?\n]+")

_INCLUDE_MODES = ("include", "attract")
_EXCLUDE_MODES = ("exclude", "constrain")


def _complete_sentences(text: str) -> list[str]:
    """Sentences in `text` that end on a terminator; a trailing fragment with
    no terminator yet is not returned (caller decides whether to wait for
    more text or cut its losses)."""
    return [m.group(0).strip() for m in _COMPLETE_SENTENCE_RE.finditer(text)
            if m.group(0).strip()]


class GuardViolation(NamedTuple):
    """A sentence failed the guard. ``kind="exclude"`` — too close to a
    forbidden concept; ``kind="include"`` — not close enough to *any*
    required concept (drifted off-topic). Field order keeps ``violation[0]``/
    ``violation[1]`` as ``(spec, similarity)`` for exclude-only callers."""
    spec: ConceptSpec
    similarity: float
    kind: str = "exclude"


class SentenceConceptGuard:
    """Classifies whole sentences against a set of concepts by embedding
    similarity to each concept's example sentences (``spec.prototypes``).

    This is deliberately *not* the token-lexicon path (V3a/V3b) — prototypes
    here are full example sentences ("поехать на море", "поехать на океан",
    "поехать на рыбалку"), and a match is "this new sentence means roughly
    the same thing as one of those", not "this sentence contains a blocked
    word".

    Concepts are dual (ARCHITECTURE.md §3.3/§10), split by ``spec.mode``:

    - **exclude** (default; ``"exclude"``/``"constrain"``): reject a sentence
      whose similarity to this concept's prototypes clears ``threshold`` —
      "too close to a forbidden meaning".
    - **include** (``"include"``/``"attract"``): reject a sentence whose
      similarity to *every* include concept's prototypes falls *below*
      ``include_threshold`` — "not close enough to any required meaning"
      (drifted off-topic). Multiple include concepts are OR'd: matching any
      one of them is enough to pass.

    ``engine`` is whatever provides ``embed()`` for this classifier — pass a
    dedicated embedding model here, not the generation model (verified
    empirically: a generative decoder's own embeddings have no usable
    separation between related and unrelated sentences, see
    ARCHITECTURE.md §5.1.1). It need not be the same engine ``generate_guarded``
    uses to generate text.

    ``aggregation="mean"`` (default) averages similarity across all of a
    concept's prototypes rather than taking the single closest one. This
    matters because short prototypes sharing a sentence template (e.g. every
    prototype phrased "поехать в X") otherwise let template similarity alone
    push an unrelated "X" over threshold; several *differently phrased*
    prototypes per concept average that out. Use ``"max"`` for the more
    permissive one-hit classifier if that trade-off is preferred instead.
    """

    def __init__(self, engine, concepts: list[ConceptSpec], threshold: float = 0.75,
                 aggregation: str = "mean", include_threshold: float | None = None):
        if not hasattr(engine, "embed"):
            raise ValueError("SentenceConceptGuard requires an engine with embed()")
        if aggregation not in ("mean", "max"):
            raise ValueError('aggregation must be "mean" or "max"')
        self.engine = engine
        self.concepts = concepts
        self.threshold = threshold
        self.include_threshold = threshold if include_threshold is None else include_threshold
        self.aggregation = aggregation
        self._proto_cache: dict[int, list] = {}

    def _prototype_vectors(self, idx: int, spec: ConceptSpec) -> list:
        if idx not in self._proto_cache:
            self._proto_cache[idx] = [self.engine.embed(p) for p in spec.prototypes if p.strip()]
        return self._proto_cache[idx]

    def _similarity(self, vec, idx: int, spec: ConceptSpec) -> float | None:
        protos = self._prototype_vectors(idx, spec)
        if not protos:
            return None
        sims = [cosine(vec, p) for p in protos]
        return (sum(sims) / len(sims)) if self.aggregation == "mean" else max(sims)

    def classify(self, text: str) -> GuardViolation | None:
        """Return the first violation found, or ``None`` if ``text`` is fine.

        Exclude concepts are checked first (any match is an immediate
        violation); only if none fire are include concepts checked (a
        violation there means the text matched *none* of them well enough).
        """
        text = text.strip()
        if not text:
            return None
        vec = self.engine.embed(text)

        best_exclude: GuardViolation | None = None
        include_specs: list[tuple[int, ConceptSpec]] = []
        for idx, spec in enumerate(self.concepts):
            if spec.mode in _INCLUDE_MODES:
                include_specs.append((idx, spec))
                continue
            sim = self._similarity(vec, idx, spec)
            if sim is None:
                continue
            if sim >= self.threshold and (best_exclude is None or sim > best_exclude.similarity):
                best_exclude = GuardViolation(spec, sim, kind="exclude")
        if best_exclude is not None:
            return best_exclude

        if not include_specs:
            return None
        best_include_sim = -1.0
        best_include_spec = None
        for idx, spec in include_specs:
            sim = self._similarity(vec, idx, spec)
            if sim is None:
                continue
            if sim > best_include_sim:
                best_include_sim = sim
                best_include_spec = spec
        if best_include_spec is None:
            return None  # no include concept had any usable prototypes
        if best_include_sim < self.include_threshold:
            return GuardViolation(best_include_spec, best_include_sim, kind="include")
        return None


@dataclass
class GuardResult:
    text: str
    rejected_sentences: list[str] = field(default_factory=list)
    widened_ids: set[int] = field(default_factory=set)     # exclude side: concept_ids grown over the run
    attract_ids: set[int] = field(default_factory=set)      # include side, non-mixing fallback (static)
    mix_ids: set[int] = field(default_factory=set)          # include side, mixing path (static), §12.5
    gave_up: bool = False


def _filter_whole_word_tokens(engine, token_ids) -> set[int]:
    """Keep only tokens whose raw surface form starts with a literal leading
    space — a BPE word-boundary marker meaning "start of a new word", not a
    mid-word continuation fragment.

    Mixing (§12.1) injects a token directly into the stream at every
    position with probability ~alpha; a fragment like "ца"/"ive"/"ord"
    (perfectly normal *inside* a word, meaningless standalone) shatters
    syntax when injected on its own — verified empirically on a real model:
    an unfiltered lexicon of 5 Russian words compiled down to 33 raw BPE
    pieces, mostly single-character fragments, and mixing them produced
    Cyrillic word salad at any alpha above ~0.1. This filter is a real fix,
    not a nice-to-have — exclude/attract (additive, §3.3/§10.1) don't need
    it (an excluded/boosted fragment only ever competes at its own position,
    it doesn't itself become injected as a word), only mixing does.
    """
    out: set[int] = set()
    for tid in token_ids:
        try:
            text = engine.detokenize([tid])
        except Exception:
            continue
        if text.startswith(" ") and text.strip().isalpha() and len(text.strip()) >= 3:
            out.add(tid)
    return out


def _leak_intent(spec: ConceptSpec, leaked_text: str) -> str:
    return (
        f'The text below expresses the forbidden concept "{spec.concept}" in a '
        "way that slipped past the current filter. List the specific words or "
        "short phrases from THIS text (not general synonyms) that should be "
        "added to the exclusion lexicon.\n"
        f"Text: {leaked_text}"
    )


def generate_guarded(
    engine, prompt: str, guard: SentenceConceptGuard, *,
    max_tokens: int = 256, chunk_tokens: int = 24, max_chunk_tokens: int = 128,
    max_rejections: int = 6,
    concept_ids: set[int] | None = None, attract_ids: set[int] | None = None,
    attract_weight: float = 5.0, attract_weight_step: float = 2.0,
    max_attract_weight: float | None = None,
    mix_alpha: float = 0.3, mix_alpha_step: float = 0.2,
    vocab_index=None,
    temperature: float = 0.7,
    **generate_kwargs,
) -> GuardResult:
    """Generate sentence-by-sentence, rejecting and regenerating any sentence
    that the guard flags — either too close to an exclude concept, or not
    close enough to any include concept (§5.1/§10 Фаза 4).

    The two violation kinds widen differently before a retry:

    - **exclude**: the rejected sentence itself is fed to a self-query
      (§5.1) to extract the phrasing that carried the forbidden meaning,
      which is folded into ``concept_ids`` — so retries narrow the space
      rather than resampling the same distribution. This mask only grows.
    - **include**: the target concept is already fully known from its own
      spec (compiled once, up front — no self-query needed), so a repeated
      drift-violation instead pushes the existing attraction *harder*. On a
      ``LlamaCppEngine`` (checked via ``isinstance``) this escalates
      ``mix_alpha`` — probability mixing, ARCHITECTURE.md §12.1:
      ``p_final = (1-α)·p_natural + α·p_concept``, naturally bounded to
      [0,1], no per-model calibration. On any other engine (mixing needs raw
      pre-sampling logits that only ``LlamaCppEngine`` exposes, §12.4) it
      falls back to escalating the older additive ``attract_weight``,
      capped at ``max_attract_weight`` (default ``3 * attract_weight``) —
      §10.3 found an uncapped weight on a narrow token set drives the model
      into repeating a single token indefinitely, since an additive bias
      applies at *every* position in the chunk, not just where it's
      topically appropriate; mixing exists specifically because it doesn't
      have that failure mode. Whichever path is active resets to its base
      value after each accepted sentence.

    Gives up (returns with ``gave_up=True``) after ``max_rejections``
    *consecutive* rejections at the same sentence position — the counter
    resets on every accepted sentence, alongside the escalated weight/alpha.
    This matters for include mode in particular: if a position typically
    needs a couple of retries before enough push lands it on-topic, that
    pattern repeating over many sentences in a long response is normal,
    expected behavior, not a reason to abandon the whole generation — only a
    position that *never* resolves should trigger that.

    Only kwargs from the base ``Engine.generate`` contract (``base.py``) are
    forwarded by default — ``top_p``/``top_k``/``seed``/``mix_ids``/
    ``mix_alpha`` are a ``LlamaCppEngine``-only extension, not part of that
    contract, and ``MockEngine`` rejects unknown kwargs. Pass extras via
    ``**generate_kwargs`` only when the caller knows ``engine`` supports them
    — ``mix_ids``/``mix_alpha`` themselves are handled internally and must
    not be passed this way.
    """
    from .llamacpp import LlamaCppEngine

    ctor = ConceptConstructor(engine)
    concept_ids = set(concept_ids or [])
    attract_ids = set(attract_ids or [])
    mix_ids: set[int] = set()
    if max_attract_weight is None:
        max_attract_weight = attract_weight * 3

    # Mixing needs raw pre-sampling logits (§12.4) -- only LlamaCppEngine has
    # them. Everything else (MockEngine, and generate_guarded's own fake-engine
    # tests) uses the older additive attract path instead.
    supports_mix = isinstance(engine, LlamaCppEngine)

    # Include concepts are fully specified up front -- compile their token
    # ids once rather than waiting for a drift violation to "discover" them.
    for spec in guard.concepts:
        if spec.mode in _INCLUDE_MODES:
            try:
                ids = ctor.compile(spec, vocab_index=vocab_index)
            except Exception:
                continue
            if supports_mix:
                mix_ids |= _filter_whole_word_tokens(engine, ids)
            else:
                attract_ids |= ids

    accepted = ""
    rejected_sentences: list[str] = []
    rejections = 0
    budget = chunk_tokens
    current_attract_weight = attract_weight
    current_mix_alpha = mix_alpha

    def used_tokens() -> int:
        if hasattr(engine, "tokenize"):
            try:
                return len(engine.tokenize(accepted))
            except Exception:
                pass
        return len(accepted.split())

    while used_tokens() < max_tokens:
        remaining = max_tokens - used_tokens()
        step = min(budget, remaining)
        mix_kwargs = {}
        if supports_mix and mix_ids:
            mix_kwargs = {"mix_ids": mix_ids, "mix_alpha": current_mix_alpha}
        piece = engine.generate(
            prompt + accepted, max_tokens=step, temperature=temperature,
            concept_ids=concept_ids or None,
            attract_ids=attract_ids or None, attract_weight=current_attract_weight,
            **mix_kwargs, **generate_kwargs,
        )
        if not piece:
            break  # model has nothing more to say (e.g. immediate EOS)

        sentences = _complete_sentences(piece)
        if not sentences:
            # No sentence boundary yet in this window. Give it more room
            # rather than discarding progress and stalling forever.
            if step >= remaining or budget >= max_chunk_tokens:
                accepted += piece
                break
            budget = min(budget * 2, max_chunk_tokens)
            continue

        budget = chunk_tokens
        first = sentences[0]
        violation = guard.classify(first)
        if violation is None:
            accepted += first
            # Reset escalation (whichever path is active) and the
            # per-position give-up counter (see docstring) on success.
            current_attract_weight = attract_weight
            current_mix_alpha = mix_alpha
            rejections = 0
            continue

        rejected_sentences.append(first)
        rejections += 1
        if rejections > max_rejections:
            return GuardResult(accepted, rejected_sentences, concept_ids,
                               attract_ids, mix_ids, gave_up=True)

        if violation.kind == "include":
            if supports_mix and mix_ids:
                current_mix_alpha = min(current_mix_alpha + mix_alpha_step, 1.0)
            else:
                current_attract_weight = min(
                    current_attract_weight + attract_weight_step, max_attract_weight)
        else:
            try:
                leak_spec = ctor.construct(_leak_intent(violation.spec, first), mode="exclude")
            except Exception:
                leak_spec = ConceptSpec(concept=violation.spec.concept, mode="exclude", lexicon=[first])
            concept_ids |= ctor.compile(leak_spec, vocab_index=vocab_index)
        # loop retries the same `accepted` position with the updated mask/weight/alpha

    return GuardResult(accepted, rejected_sentences, concept_ids, attract_ids, mix_ids, gave_up=False)

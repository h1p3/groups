from groupcot.engine.constructor import ConceptSpec
from groupcot.engine.guarded_generation import (
    SentenceConceptGuard, generate_guarded, _complete_sentences,
    _filter_whole_word_tokens,
)
from groupcot.engine.llamacpp import LlamaCppEngine


class _FakeSubwordEngine:
    """Fake engine with a small fixed vocab mimicking real BPE fragmentation:
    whole words carry a leading space (word-boundary marker), mid-word
    continuation pieces don't."""

    _VOCAB = {
        1: " recipe", 2: " cook", 3: "ing", 4: "e", 5: " a", 6: " the",
        7: " ", 8: " x", 9: "!!",
    }

    def detokenize(self, ids):
        return self._VOCAB[ids[0]]


def test_filter_whole_word_tokens_keeps_only_leading_space_words():
    engine = _FakeSubwordEngine()
    kept = _filter_whole_word_tokens(engine, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # " recipe", " cook", " the" -- real whole words, len(stripped) >= 3.
    assert kept == {1, 2, 6}
    # 3 ("ing"), 4 ("e") -- no leading space, mid-word fragments.
    # 5 (" a"), 8 (" x") -- leading space but len(stripped) < 3.
    # 7 (" ") -- strips to empty, isalpha() is False.
    # 9 ("!!") -- not alphabetic.
    assert kept.isdisjoint({3, 4, 5, 7, 8, 9})


def test_filter_whole_word_tokens_handles_detokenize_errors():
    class _Flaky:
        def detokenize(self, ids):
            raise ValueError("boom")
    assert _filter_whole_word_tokens(_Flaky(), [1, 2, 3]) == set()


def test_complete_sentences_splits_and_drops_trailing_fragment():
    assert _complete_sentences("Hello world. How are you?") == [
        "Hello world.", "How are you?",
    ]
    assert _complete_sentences("Hello world. and then it trails off") == ["Hello world."]
    assert _complete_sentences("no terminator at all") == []


class _FakeEmbedEngine:
    """Toy 2D embedding space: sea-related text -> (1, 0), everything else -> (0, 1)."""

    _SEA_MARKERS = ("море", "океан", "рыбалк")

    def embed(self, text):
        t = text.lower()
        if any(m in t for m in self._SEA_MARKERS):
            return [1.0, 0.0]
        return [0.0, 1.0]


class _FakeTemplateEmbedEngine:
    """Mimics the real e5 finding: prototypes sharing a sentence template
    ('поехать в X') score deceptively high on a differently-conceptual X
    under max-aggregation, but mean-aggregation over several differently
    phrased prototypes separates them."""

    # Constructed so cos(candidate, proto1) = 0.95 (shared "поехать в X"
    # template) and cos(candidate, proto2) = 0.70 (proto2 has no shared
    # template, so its similarity reflects meaning, not phrasing) — mean
    # (0.825) falls below a 0.9 threshold while max (0.95) clears it.
    _VECS = {
        "поехать в лес": [1.0, 0.0],
        "поехать на море": [0.95, 0.3122],
        "хочу искупаться в море": [0.70, 0.7141],
    }

    def embed(self, text):
        return self._VECS.get(text.strip(), [0.0, 1.0])


def test_sentence_concept_guard_mean_vs_max_aggregation():
    engine = _FakeTemplateEmbedEngine()
    spec = ConceptSpec(
        concept="sea", mode="exclude",
        prototypes=["поехать на море", "хочу искупаться в море"],
    )
    max_guard = SentenceConceptGuard(engine, [spec], threshold=0.9, aggregation="max")
    mean_guard = SentenceConceptGuard(engine, [spec], threshold=0.9, aggregation="mean")

    # Against the single template-sharing prototype, max-aggregation over-triggers...
    assert max_guard.classify("поехать в лес") is not None
    # ...while mean-aggregation (averaging in the non-template prototype) does not.
    assert mean_guard.classify("поехать в лес") is None


def test_sentence_concept_guard_classify():
    engine = _FakeEmbedEngine()
    sea = ConceptSpec(
        concept="поездка к морю", mode="exclude",
        prototypes=["поехать на море", "поехать на океан", "поехать на рыбалку"],
    )
    guard = SentenceConceptGuard(engine, [sea], threshold=0.75)

    match = guard.classify("Давай съездим на море позагорать.")
    assert match is not None
    assert match[0] is sea
    assert match[1] >= 0.75

    assert guard.classify("Давай сходим в лес за грибами.") is None
    assert guard.classify("") is None


def test_sentence_concept_guard_include_flags_drift():
    engine = _FakeEmbedEngine()  # sea-markers -> (1,0), else -> (0,1)
    forest = ConceptSpec(
        concept="лес", mode="include",
        prototypes=["поехать в лес", "прогулка по лесу"],
    )
    guard = SentenceConceptGuard(engine, [forest], threshold=0.75)

    # Forest prototypes embed to (0,1) under this fake (no sea markers) --
    # a sentence that also embeds to (0,1) stays "close enough", no violation.
    assert guard.classify("Идём гулять в парк.") is None

    # A sea-flavored sentence embeds to (1,0): far from the include concept's
    # (0,1) prototypes -> flagged as drift, not as a forbidden match.
    violation = guard.classify("Давай съездим на море.")
    assert violation is not None
    assert violation.kind == "include"
    assert violation.spec is forest


def test_sentence_concept_guard_include_specs_are_ored():
    engine = _FakeEmbedEngine()
    sea = ConceptSpec(concept="море", mode="include", prototypes=["поехать на море"])
    generic = ConceptSpec(concept="прочее", mode="include", prototypes=["сходить в парк"])
    guard = SentenceConceptGuard(engine, [sea, generic], threshold=0.75)

    # Sea-flavored text matches the "sea" include-concept even though it
    # fails the unrelated "generic" one -> OR semantics, no violation.
    assert guard.classify("Давай съездим на море позагорать.") is None

    # Against just the "generic" concept alone, the same sea-flavored text
    # matches neither -> violation.
    generic_only_guard = SentenceConceptGuard(engine, [generic], threshold=0.75)
    violation = generic_only_guard.classify("Давай съездим на море.")
    assert violation is not None and violation.kind == "include"


def test_sentence_concept_guard_exclude_checked_before_include():
    engine = _FakeEmbedEngine()
    forbidden_sea = ConceptSpec(concept="море", mode="exclude", prototypes=["поехать на море"])
    required_sea = ConceptSpec(concept="море-тема", mode="include", prototypes=["поехать на море"])
    # Contradictory setup on purpose: a sea sentence matches BOTH the forbidden
    # exclude concept and the required include concept. Exclude must win --
    # forbidding a meaning takes priority over merely failing to require it.
    guard = SentenceConceptGuard(engine, [forbidden_sea, required_sea], threshold=0.75)
    violation = guard.classify("Давай съездим на море.")
    assert violation is not None
    assert violation.kind == "exclude"
    assert violation.spec is forbidden_sea


class _FakeGuardEngine:
    """Fake generative+embedding engine for the reject/widen/regenerate loop.

    Ignores the specific token IDs in `concept_ids` (that machinery is tested
    separately for ConceptConstructor/ConceptSuppress) and instead switches
    its canned output once *any* mask has been applied — modeling "the
    widened exclusion mask changed what the model can produce".
    """

    _SEA_SENTENCE = "Давай поедем на море загорать. "
    _FOREST_SENTENCE = "Давай поедем в лес гулять. "

    def __init__(self, always_sea: bool = False):
        self.calls = 0
        self.always_sea = always_sea

    def tokenize(self, text):
        return [abs(hash(w.lower())) % 100_000 for w in text.split()]

    def embed(self, text):
        t = text.lower()
        if any(m in t for m in ("море", "океан", "рыбалк")):
            return [1.0, 0.0]
        return [0.0, 1.0]

    def chat(self, messages, **kwargs):
        return ('{"concept": "sea", "mode": "exclude", "lexicon": ["море"]}')

    def generate(self, prompt, max_tokens=24, temperature=0.7, concept_ids=None, **kwargs):
        self.calls += 1
        if self.always_sea or not concept_ids:
            return self._SEA_SENTENCE
        return self._FOREST_SENTENCE


def _sea_spec():
    return ConceptSpec(
        concept="поездка к морю", mode="exclude",
        prototypes=["поехать на море", "поехать на океан", "поехать на рыбалку"],
    )


def test_generate_guarded_rejects_then_widens_and_accepts():
    engine = _FakeGuardEngine()
    guard = SentenceConceptGuard(engine, [_sea_spec()], threshold=0.5)

    result = generate_guarded(
        engine, "Что мне сегодня сделать? ", guard,
        max_tokens=40, chunk_tokens=10,
    )

    assert "море" not in result.text.lower()
    assert "океан" not in result.text.lower()
    assert len(result.rejected_sentences) == 1
    assert "море" in result.rejected_sentences[0].lower()
    assert result.widened_ids  # mask grew from the rejected sentence
    assert not result.gave_up
    assert engine.calls >= 2  # at least one reject + one accepted retry


def test_generate_guarded_gives_up_after_max_rejections():
    engine = _FakeGuardEngine(always_sea=True)  # can never escape the forbidden concept
    guard = SentenceConceptGuard(engine, [_sea_spec()], threshold=0.5)

    result = generate_guarded(
        engine, "Что мне сегодня сделать? ", guard,
        max_tokens=100, chunk_tokens=10, max_rejections=2,
    )

    assert result.gave_up
    assert len(result.rejected_sentences) == 3  # max_rejections + 1
    assert result.text == ""  # nothing was ever accepted


def test_generate_guarded_passthrough_when_nothing_matches():
    engine = _FakeGuardEngine()
    forest_only_guard = SentenceConceptGuard(
        engine,
        [ConceptSpec(concept="x", mode="exclude", prototypes=["something unrelated entirely"])],
        threshold=0.99,
    )
    result = generate_guarded(
        engine, "Prompt: ", forest_only_guard, max_tokens=20, chunk_tokens=10,
    )
    assert result.rejected_sentences == []
    assert not result.gave_up
    assert "море" in result.text.lower()  # accepted as-is, nothing forbidden matched


class _FakeIncludeGuardEngine:
    """Fake engine for the include-mode escalation path: stays off-topic
    until attract_weight climbs high enough -- models "a stronger pull
    toward the desired field eventually works", the include-side analogue of
    _FakeGuardEngine's "a wider exclusion mask eventually works"."""

    _OFF_TOPIC = "Сегодня хорошая погода. "
    _ON_TOPIC = "Давай поедем в лес гулять. "

    def __init__(self, weight_needed: float = 7.0):
        self.calls = 0
        self.weight_needed = weight_needed
        self.seen_weights: list[float] = []

    def tokenize(self, text):
        return [abs(hash(w.lower())) % 100_000 for w in text.split()]

    def embed(self, text):
        t = text.lower()
        if any(m in t for m in ("лес", "гулять")):
            return [1.0, 0.0]
        return [0.0, 1.0]

    def generate(self, prompt, max_tokens=24, temperature=0.7, attract_weight=5.0, **kwargs):
        self.calls += 1
        self.seen_weights.append(attract_weight)
        return self._ON_TOPIC if attract_weight >= self.weight_needed else self._OFF_TOPIC


def _forest_include_spec():
    return ConceptSpec(
        concept="лес", mode="include", lexicon=["лес"],
        prototypes=["поехать в лес", "прогулка по лесу", "погулять в лесу"],
    )


def test_generate_guarded_include_escalates_attract_weight_not_mask():
    engine = _FakeIncludeGuardEngine(weight_needed=7.0)
    guard = SentenceConceptGuard(engine, [_forest_include_spec()], threshold=0.5)

    result = generate_guarded(
        engine, "О чём поговорим? ", guard,
        max_tokens=40, chunk_tokens=10, attract_weight=5.0, attract_weight_step=1.5,
    )

    assert "лес" in result.text.lower()
    assert len(result.rejected_sentences) >= 1
    assert not result.widened_ids  # include violations never touch concept_ids
    assert result.attract_ids       # precompiled from the include spec up front, not from a leak
    assert max(engine.seen_weights) >= 7.0
    assert not result.gave_up


class _FakeLlamaCppIncludeEngine(LlamaCppEngine):
    """Subclasses the real LlamaCppEngine (so `isinstance(engine,
    LlamaCppEngine)` in generate_guarded picks the mix_alpha path) without
    needing llama_cpp installed or a real model -- __init__ is overridden to
    skip all of that."""

    _OFF_TOPIC = "Сегодня хорошая погода. "
    _ON_TOPIC = "Давай поедем в лес гулять. "

    def __init__(self, alpha_needed: float = 0.7):
        self.calls = 0
        self.alpha_needed = alpha_needed
        self.seen_alphas: list[float] = []
        self._vocab: dict[int, str] = {}  # tid -> " word", for detokenize

    def tokenize(self, text):
        ids = []
        for w in text.split():
            tid = abs(hash(w.lower())) % 100_000
            self._vocab[tid] = " " + w.lower()  # simulate a BPE word-boundary token
            ids.append(tid)
        return ids

    def detokenize(self, ids):
        return "".join(self._vocab.get(tid, "") for tid in ids)

    def embed(self, text):
        t = text.lower()
        if any(m in t for m in ("лес", "гулять")):
            return [1.0, 0.0]
        return [0.0, 1.0]

    def generate(self, prompt, max_tokens=24, temperature=0.7, mix_ids=None,
                mix_alpha=0.0, **kwargs):
        self.calls += 1
        self.seen_alphas.append(mix_alpha)
        return self._ON_TOPIC if mix_alpha >= self.alpha_needed else self._OFF_TOPIC


def test_generate_guarded_uses_mix_alpha_for_llamacpp_engine():
    engine = _FakeLlamaCppIncludeEngine(alpha_needed=0.7)
    guard = SentenceConceptGuard(engine, [_forest_include_spec()], threshold=0.5)

    result = generate_guarded(
        engine, "О чём поговорим? ", guard,
        max_tokens=40, chunk_tokens=10, mix_alpha=0.3, mix_alpha_step=0.2,
    )

    assert "лес" in result.text.lower()
    assert result.mix_ids                      # mixing path used...
    assert not result.attract_ids               # ...not the additive fallback
    assert not result.widened_ids
    assert max(engine.seen_alphas) >= 0.7
    assert all(0.0 <= a <= 1.0 for a in engine.seen_alphas)
    assert not result.gave_up


def test_generate_guarded_mix_alpha_naturally_capped_at_one():
    engine = _FakeLlamaCppIncludeEngine(alpha_needed=999.0)  # unreachable
    guard = SentenceConceptGuard(engine, [_forest_include_spec()], threshold=0.5)

    result = generate_guarded(
        engine, "О чём поговорим? ", guard,
        max_tokens=200, chunk_tokens=10, max_rejections=50,
        mix_alpha=0.3, mix_alpha_step=0.2,
    )

    assert max(engine.seen_alphas) <= 1.0
    assert result.gave_up


def test_generate_guarded_include_weight_escalation_is_capped():
    # weight_needed is unreachable within the default cap (3x base=5.0 -> 15.0),
    # so this must give up rather than escalate the weight without bound --
    # verified empirically that unbounded weight on a narrow attract set drives
    # the model into repeating a single boosted token forever (see docstring).
    engine = _FakeIncludeGuardEngine(weight_needed=20.0)
    guard = SentenceConceptGuard(engine, [_forest_include_spec()], threshold=0.5)

    result = generate_guarded(
        engine, "О чём поговорим? ", guard,
        max_tokens=200, chunk_tokens=10, max_rejections=50,
        attract_weight=5.0, attract_weight_step=2.0,  # default cap = 15.0
    )

    assert max(engine.seen_weights) <= 15.0
    assert "лес" not in result.text.lower()  # never actually got there

import os

import numpy as np
import pytest

from groupcot.engine.processors import ConceptSuppress, ConceptAttract
from groupcot.engine.constructor import ConceptConstructor, ConceptSpec
from groupcot.engine.vocab_index import VocabIndex


def test_concept_suppress():
    proc = ConceptSuppress({5, 10})
    scores = np.arange(20, dtype=np.float32)
    out = proc(None, scores)
    assert np.isneginf(out[5])
    assert np.isneginf(out[10])
    assert out[0] == 0.0  # untouched
    assert out[11] == 11.0


def test_concept_attract():
    proc = ConceptAttract({3}, weight=2.0)
    scores = np.arange(10, dtype=np.float32)
    out = proc(None, scores)
    assert out[3] == 3.0 + 2.0
    assert out[0] == 0.0
    assert out[4] == 4.0


def test_spec_to_token_ids():
    spec = ConceptSpec(lexicon=["ab", "cd"])
    ids = spec.to_token_ids(lambda t: [ord(c) for c in t])
    # Base phrase characters must be present...
    for ch in "abcd":
        assert ord(ch) in ids
    # ...and the space / capitalized variants add extra IDs.
    assert ord(" ") in ids
    assert ord("A") in ids
    assert ord("C") in ids


def test_constructor_parse_strips_prose():
    raw = 'prefix {"concept":"x","mode":"exclude","lexicon":["foo","bar"]} suffix'
    spec = ConceptConstructor._parse(raw)
    assert spec.concept == "x"
    assert spec.mode == "exclude"
    assert spec.lexicon == ["foo", "bar"]


def test_constructor_parse_invalid():
    assert ConceptConstructor._parse("no json here").concept == ""
    assert ConceptConstructor._parse("{bad json").lexicon == []


def test_constructor_compile():
    class FakeEngine:
        def tokenize(self, text):
            return [ord(c) for c in text]

    ctor = ConceptConstructor(FakeEngine())
    spec = ConceptSpec(lexicon=["ab"])
    ids = ctor.compile(spec)
    assert ord("a") in ids and ord("b") in ids
    # Leading-space / capitalized variants should also be included.
    assert ord(" ") in ids and ord("A") in ids


def test_constructor_construct_self_query():
    class FakeEngine:
        def generate(self, prompt, max_tokens=400, temperature=0.0):
            return '{"concept":"tensors","mode":"exclude","lexicon":["768","758"]}'

    ctor = ConceptConstructor(FakeEngine())
    spec = ctor.construct("forbid wrong tensor layer sizes")
    assert spec.mode == "exclude"
    assert "768" in spec.lexicon
    assert "758" in spec.lexicon


class _FakeSemanticEngine:
    """Fake engine with a toy 2D embedding space: cat-like tokens near (1, 0),
    dog-like tokens near (0, 1). Used to test VocabIndex / V3b compilation
    without a real model."""

    _VOCAB = {
        0: "cat", 1: " cat", 2: "cats", 3: " cats", 4: " kitten",
        5: "dog", 6: " dog", 7: " puppy",
        8: "!!", 9: "x",  # filtered out: non-alpha / too short
        # ids 0/2/5 ("cat"/"cats"/"dog", no leading space) are mid-word-shaped
        # and get filtered out of VocabIndex candidates by require_word_boundary
        # (default True) -- kept here only so V3a's tokenize() can still find
        # them (V3a doesn't go through VocabIndex at all, unaffected).
    }
    _EMB = {
        "cat": [1.0, 0.0], "cats": [0.95, 0.05], "kitten": [0.9, 0.1],
        "dog": [0.0, 1.0], "puppy": [0.05, 0.9],
    }

    def vocab_size(self) -> int:
        return len(self._VOCAB)

    def detokenize(self, ids):
        return self._VOCAB[ids[0]]

    def tokenize(self, text):
        stripped = text.strip()
        return [tid for tid, tok in self._VOCAB.items() if tok.strip() == stripped]

    def embed(self, text):
        return self._EMB.get(text.strip(), [0.0, 0.0])

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def test_vocab_index_require_word_boundary_can_be_disabled(tmp_path):
    """require_word_boundary=False restores the old, fragment-inclusive
    candidate pool -- an explicit opt-out, not the default."""
    idx = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path,
                      require_word_boundary=False)
    idx.build()
    # Now id 0 ("cat", no leading space) is eligible too.
    assert 0 in idx.token_ids


def test_vocab_index_word_boundary_flag_changes_cache_key(tmp_path):
    idx_default = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path)
    idx_no_boundary = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path,
                                 require_word_boundary=False)
    assert idx_default._cache_key() != idx_no_boundary._cache_key()


def test_vocab_index_build_dedupes_and_filters(tmp_path):
    idx = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path)
    idx.build()
    # "!!" (non-alpha), "x" (too short), and the no-leading-space variants
    # ("cat" id 0, "cats" id 2, "dog" id 5 -- mid-word-shaped) are filtered;
    # only whole-word (leading-space) tokens survive.
    assert sorted(t.strip() for t in idx.texts) == ["cat", "cats", "dog", "kitten", "puppy"]
    assert idx.embeddings.shape[0] == len(idx.token_ids)
    assert not ({0, 2, 5} & set(idx.token_ids))


def test_vocab_index_nearest_finds_semantic_neighbors(tmp_path):
    idx = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path)
    idx.build()
    results = idx.nearest([1.0, 0.0], top_k=10, min_similarity=0.5)
    texts = {text.strip() for _tid, text, _sim in results}
    assert {"cat", "cats", "kitten"} <= texts
    assert "dog" not in texts and "puppy" not in texts


def test_vocab_index_cache_roundtrip(tmp_path):
    idx = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path)
    idx.build()
    cached = VocabIndex(_FakeSemanticEngine(), max_candidates=100, cache_dir=tmp_path)
    cached.build()  # should load from disk, not re-embed
    assert cached.token_ids == idx.token_ids
    assert np.allclose(cached.embeddings, idx.embeddings)


def test_compile_semantic_catches_morphological_variants(tmp_path):
    engine = _FakeSemanticEngine()
    idx = VocabIndex(engine, max_candidates=100, cache_dir=tmp_path)
    ctor = ConceptConstructor(engine)
    spec = ConceptSpec(concept="cat", mode="exclude", lexicon=["cat"])
    # V3a alone only tokenizes the literal lexicon ("cat" / " cat").
    lexicon_only = ctor.compile(spec)
    assert {0, 1} <= lexicon_only
    assert 2 not in lexicon_only and 4 not in lexicon_only  # cats / kitten missing

    # V3b (vocab_index passed) additionally pulls in the semantic neighborhood.
    # Note: the VocabIndex candidate for "cats" is id 3 (" cats", leading
    # space) now, not id 2 -- id 2 ("cats", no space) never enters the
    # candidate pool at all (require_word_boundary filters it out).
    expanded = ctor.compile(spec, vocab_index=idx, top_k=10, min_similarity=0.5)
    assert lexicon_only <= expanded
    assert 3 in expanded  # " cats"
    assert 4 in expanded  # " kitten"
    assert not ({5, 7} & expanded)  # dog / puppy stay out (and id 5 was never a candidate anyway)


def test_compile_semantic_respects_allowed(tmp_path):
    engine = _FakeSemanticEngine()
    idx = VocabIndex(engine, max_candidates=100, cache_dir=tmp_path)
    idx.build()
    ctor = ConceptConstructor(engine)
    spec = ConceptSpec(concept="cat", mode="exclude", lexicon=["cat"], allowed=["kitten"])
    expanded = ctor.compile(spec, vocab_index=idx, top_k=10, min_similarity=0.5)
    assert 3 in expanded       # " cats" still excluded
    assert 4 not in expanded  # " kitten" explicitly allowed


class _FakeTokenOnlyEngine:
    """Generation-side engine: tokenize/detokenize/vocab_size, deliberately NO
    embed() at all — proves the dual-engine path never needs the generation
    engine to embed anything when a separate embed_engine is supplied.

    Each word has both an unspaced id (e.g. sentence-initial usage) and a
    leading-space id (mid-sentence usage, the realistic BPE norm) -- only the
    latter survive as VocabIndex candidates (require_word_boundary)."""

    _VOCAB = {0: "cat", 1: " cat", 2: "cats", 3: " cats", 4: "dog", 5: " dog"}

    def vocab_size(self):
        return len(self._VOCAB)

    def detokenize(self, ids):
        return self._VOCAB[ids[0]]

    def tokenize(self, text):
        stripped = text.strip()
        return [tid for tid, tok in self._VOCAB.items() if tok.strip() == stripped]


class _FakeEmbedOnlyEngine:
    """Embedding-side engine: only embed(), no tokenize/vocab_size/detokenize."""

    _EMB = {"cat": [1.0, 0.0], "cats": [0.95, 0.05], "dog": [0.0, 1.0]}

    def embed(self, text):
        return self._EMB.get(text.strip(), [0.0, 0.0])


def test_vocab_index_dual_engine_routes_embeddings_to_embed_engine(tmp_path):
    gen_engine = _FakeTokenOnlyEngine()
    embed_engine = _FakeEmbedOnlyEngine()
    idx = VocabIndex(gen_engine, max_candidates=100, cache_dir=tmp_path,
                      embed_engine=embed_engine)
    idx.build()  # would raise if it tried gen_engine.embed() (doesn't exist)
    assert sorted(t.strip() for t in idx.texts) == ["cat", "cats", "dog"]
    assert not ({0, 2, 4} & set(idx.token_ids))  # unspaced ids never became candidates

    query = idx.embed_query("cat")
    assert query == [1.0, 0.0]
    results = idx.nearest(query, top_k=10, min_similarity=0.5)
    assert {tid for tid, _text, _sim in results} == {1, 3}  # " cat", " cats" — not dog


def test_vocab_index_cache_key_differs_by_embed_engine(tmp_path):
    gen_engine = _FakeTokenOnlyEngine()
    idx_a = VocabIndex(gen_engine, max_candidates=100, cache_dir=tmp_path,
                        embed_engine=_FakeEmbedOnlyEngine())
    idx_b = VocabIndex(gen_engine, max_candidates=100, cache_dir=tmp_path,
                        embed_engine=_FakeEmbedOnlyEngine())
    # Distinct instances of the same class collapse to the same identity
    # string (no model_path to distinguish them), which is the
    # correct/expected fallback — but the cache key must still differ from a
    # single-engine (no dual-engine) VocabIndex over the same generation engine.
    idx_single = VocabIndex(gen_engine, max_candidates=100, cache_dir=tmp_path)
    assert idx_a._cache_key() == idx_b._cache_key()
    assert idx_a._cache_key() != idx_single._cache_key()


def test_compile_semantic_uses_embed_engine_not_generation_engine(tmp_path):
    gen_engine = _FakeTokenOnlyEngine()
    embed_engine = _FakeEmbedOnlyEngine()
    idx = VocabIndex(gen_engine, max_candidates=100, cache_dir=tmp_path,
                      embed_engine=embed_engine)
    idx.build()
    ctor = ConceptConstructor(gen_engine)  # gen_engine has no embed() at all
    spec = ConceptSpec(concept="cat", mode="exclude", lexicon=["cat"])
    ids = ctor.compile(spec, vocab_index=idx, top_k=10, min_similarity=0.5)
    assert 3 in ids  # " cats" pulled in via embed_engine, not gen_engine


@pytest.mark.skipif(
    not os.environ.get("GROUPCOC_RUN_MODEL"),
    reason="requires a local GGUF model (set GROUPCOC_RUN_MODEL=1)",
)
def test_llamacpp_concept_exclusion():
    from groupcot.engine.llamacpp import LlamaCppEngine

    eng = LlamaCppEngine(
        model_path="models/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        n_ctx=2048, n_gpu_layers=0,
    )
    # Choose a concrete digit token and forbid it; it must never appear.
    digit_id = eng.tokenize("7")[0]
    out = eng.generate(
        "List numbers: 1 2 3 4 5 6 7 8 9",
        max_tokens=40, temperature=0.0, concept_ids={digit_id},
    )
    assert digit_id not in eng.tokenize(out)

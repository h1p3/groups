import numpy as np

from groupcot.groups.token_group import TokenGroup
from groupcot.groups.semantic_field import build_concept_field


class _FakeVocabIndexForField:
    """Minimal VocabIndex stand-in: pre-set token_ids/embeddings, and a fixed
    text->vector map for embed_query (no real embedding model involved)."""

    def __init__(self, token_ids, embeddings, query_vecs):
        self.token_ids = token_ids
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self._query_vecs = query_vecs

    def build(self):
        pass  # already "built"

    def embed_query(self, text):
        return self._query_vecs[text]


def test_build_concept_field_includes_identical_embedding_at_distance_zero():
    tg = TokenGroup(k=8)
    rng = np.random.RandomState(5)
    seed_vec = rng.randn(32).astype(np.float32)
    other_vec = rng.randn(32).astype(np.float32)
    idx = _FakeVocabIndexForField(
        token_ids=[10, 20],
        embeddings=[seed_vec, other_vec],
        query_vecs={"seed text": seed_vec},
    )
    field = build_concept_field(tg, idx, ["seed text"], max_distance=0)
    # project_embedding is deterministic, so the seed's own embedding always
    # lands at Hamming distance 0 from itself, regardless of the random
    # projection matrix -- this must hold no matter what.
    assert 10 in field


def test_build_concept_field_larger_radius_is_superset():
    tg = TokenGroup(k=16)
    rng = np.random.RandomState(6)
    embeddings = rng.randn(30, 32).astype(np.float32)
    seed_vec = rng.randn(32).astype(np.float32)
    idx = _FakeVocabIndexForField(
        token_ids=list(range(30)), embeddings=embeddings, query_vecs={"seed": seed_vec})
    small = build_concept_field(tg, idx, ["seed"], max_distance=1)
    large = build_concept_field(tg, idx, ["seed"], max_distance=16)
    assert small <= large


def test_build_concept_field_max_distance_equal_to_k_includes_everything():
    tg = TokenGroup(k=6)
    rng = np.random.RandomState(7)
    embeddings = rng.randn(15, 10).astype(np.float32)
    seed_vec = rng.randn(10).astype(np.float32)
    idx = _FakeVocabIndexForField(
        token_ids=list(range(15)), embeddings=embeddings, query_vecs={"seed": seed_vec})
    # max_distance == k is the largest possible Hamming distance in (Z/2Z)^k,
    # so every candidate must be included regardless of the projection.
    field = build_concept_field(tg, idx, ["seed"], max_distance=6)
    assert field == set(range(15))


def test_build_concept_field_empty_seeds_returns_empty():
    tg = TokenGroup(k=4)
    idx = _FakeVocabIndexForField(
        token_ids=[1, 2], embeddings=[[1.0, 0.0], [0.0, 1.0]], query_vecs={})
    assert build_concept_field(tg, idx, [], max_distance=4) == set()


def test_build_concept_field_unions_across_multiple_seeds():
    tg = TokenGroup(k=8)
    rng = np.random.RandomState(8)
    embeddings = rng.randn(10, 20).astype(np.float32)
    seed_a = embeddings[0]  # identical to token_id 0's embedding
    seed_b = embeddings[5]  # identical to token_id 5's embedding
    idx = _FakeVocabIndexForField(
        token_ids=list(range(10)), embeddings=embeddings,
        query_vecs={"a": seed_a, "b": seed_b})
    field_a = build_concept_field(tg, idx, ["a"], max_distance=0)
    field_b = build_concept_field(tg, idx, ["b"], max_distance=0)
    field_both = build_concept_field(tg, idx, ["a", "b"], max_distance=0)
    assert field_both == field_a | field_b
    assert 0 in field_both and 5 in field_both

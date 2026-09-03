from groupcot.context.attractor import (
    active_node_texts, context_attract_ids, context_include_spec,
)
from groupcot.context.store import Store
from groupcot.groups import Cyclic


def _store_with(*texts: str) -> Store:
    store = Store(group=Cyclic(64))
    for i, text in enumerate(texts):
        store.add(f"n{i}", text)
    return store


def test_active_node_texts_respects_given_order_and_max_nodes():
    store = _store_with("first", "second", "third", "fourth")
    order = ["n0", "n1", "n2", "n3"]  # oldest-first, as AutoPullLoop's own order deque would be

    assert active_node_texts(store, order) == ["first", "second", "third", "fourth"]
    # max_nodes keeps the *last* (most recent) ones, not the first.
    assert active_node_texts(store, order, max_nodes=2) == ["third", "fourth"]


def test_active_node_texts_skips_missing_or_removed_nodes():
    store = _store_with("first", "second")
    store.remove("n0")
    assert active_node_texts(store, ["n0", "n1", "n_missing"]) == ["second"]


class _FakeVocabIndex:
    """Toy embedding space: 'forest'-flavored text -> tokens {1, 2}; anything
    else -> tokens {9}."""

    def embed_query(self, text):
        return [1.0, 0.0] if "лес" in text.lower() else [0.0, 1.0]

    def nearest(self, vec, top_k=20, min_similarity=0.6):
        if vec == [1.0, 0.0]:
            return [(1, "лес", 1.0), (2, "деревья", 0.9)]
        return [(9, "прочее", 1.0)]


def test_context_attract_ids_unions_nearest_over_active_nodes():
    store = _store_with("прогулка по лесу", "что-то ещё")
    ids = context_attract_ids(store, ["n0", "n1"], _FakeVocabIndex())
    assert ids == {1, 2, 9}


def test_context_attract_ids_respects_max_nodes():
    store = _store_with("прогулка по лесу", "что-то ещё")
    # Only the most recent node ("n1", off-topic) is considered.
    ids = context_attract_ids(store, ["n0", "n1"], _FakeVocabIndex(), max_nodes=1)
    assert ids == {9}


def test_context_include_spec_builds_prototypes_from_active_text():
    store = _store_with("прогулка по лесу", "поход в горы")
    spec = context_include_spec(store, ["n0", "n1"])
    assert spec is not None
    assert spec.mode == "include"
    assert spec.prototypes == ["прогулка по лесу", "поход в горы"]


def test_context_include_spec_truncates_long_text():
    store = _store_with("x" * 500)
    spec = context_include_spec(store, ["n0"], max_chars=50)
    assert len(spec.prototypes[0]) == 50


def test_context_include_spec_none_when_no_active_context():
    store = _store_with("something")
    assert context_include_spec(store, []) is None

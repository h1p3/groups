import pytest

from groupcot.context.puller import cosine, Puller
from groupcot.context.store import Store


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_pull_top_k():
    store = Store(group=None)
    store.add("a", "alpha", embedding=[1.0, 0.0])
    store.add("b", "beta", embedding=[0.0, 1.0])
    store.add("c", "gamma", embedding=[0.9, 0.1])
    puller = Puller(store, top_k=2, threshold=0.5)
    hits = puller.pull([1.0, 0.0])
    assert [h[1].node_id for h in hits] == ["a", "c"]
    assert hits[0][0] >= 0.5


def test_pull_excludes():
    store = Store(group=None)
    store.add("a", "alpha", embedding=[1.0, 0.0])
    store.add("b", "beta", embedding=[0.9, 0.1])
    puller = Puller(store, top_k=2, threshold=0.0)
    hits = puller.pull([1.0, 0.0], exclude={"a"})
    assert [h[1].node_id for h in hits] == ["b"]

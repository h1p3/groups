from pathlib import Path

from groupcot.context import AutoPullLoop, Puller, Store
from groupcot.engine.mock import MockEngine
from groupcot.groups.cyclic import Cyclic
from groupcot.groups.vector import VectorAdd
from groupcot.prompts import PromptSet

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def test_loop_end_to_end_cyclic():
    engine = MockEngine(embed_dim=4)
    group = Cyclic(64)
    store = Store(group=group)
    for i in range(3):
        text = f"фрагмент {i}"
        store.add(f"n{i}", text, embedding=engine.embed(text))
    puller = Puller(store, top_k=2, threshold=-1.0)
    prompts = PromptSet(PROMPTS)
    loop = AutoPullLoop(engine, store, puller, group, prompts=prompts,
                        max_steps=4, pull_every=1, max_active=2)
    res = loop.run("найти ответ")
    assert res["answer"]
    assert res["steps"][0]["element"]
    assert res["context"]["pulled"]
    assert res["context"]["active_count"] <= 2
    assert res["context"]["h"].isdigit()


def test_loop_forgetting_keeps_budget():
    engine = MockEngine(embed_dim=4)
    group = Cyclic(64)
    store = Store(group=group)
    for i in range(5):
        store.add(f"n{i}", f"фрагмент {i}", embedding=engine.embed(f"фрагмент {i}"))
    puller = Puller(store, top_k=2, threshold=-1.0)
    prompts = PromptSet(PROMPTS)
    loop = AutoPullLoop(engine, store, puller, group, prompts=prompts,
                        max_steps=6, pull_every=1, max_active=2)
    res = loop.run("задача")
    assert res["context"]["active_count"] <= 2
    assert any(e.get("event") == "forgot" for e in res["context"]["pulled"])


class _FakeVocabIndex:
    """Always returns the same couple of tokens, regardless of the query --
    only used to prove AutoPullLoop calls nearest() at all and forwards the
    result, not to test VocabIndex's own matching logic (covered elsewhere)."""

    def embed_query(self, text):
        return [1.0, 0.0]

    def nearest(self, vec, top_k=20, min_similarity=0.6):
        return [(101, "tok_a", 0.9), (102, "tok_b", 0.8)]


class _RecordingEngine(MockEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_attract_ids: list[set | None] = []

    def generate(self, prompt, attract_ids=None, **kwargs):
        self.seen_attract_ids.append(set(attract_ids) if attract_ids else None)
        return super().generate(prompt, **kwargs)


def test_loop_wires_context_attract_ids_from_vocab_index():
    engine = _RecordingEngine(embed_dim=4)
    group = Cyclic(64)
    store = Store(group=group)
    for i in range(3):
        text = f"фрагмент {i}"
        store.add(f"n{i}", text, embedding=engine.embed(text))
    puller = Puller(store, top_k=2, threshold=-1.0)
    prompts = PromptSet(PROMPTS)
    loop = AutoPullLoop(engine, store, puller, group, prompts=prompts,
                        max_steps=3, pull_every=1, max_active=2,
                        vocab_index=_FakeVocabIndex())
    loop.run("найти ответ")

    # Step 1 runs before any pull cycle has happened yet -> no attract target.
    assert engine.seen_attract_ids[0] is None
    # After the first pull cycle (pull_every=1), later steps and the final
    # answer are pulled toward the active context's vocabulary.
    assert {101, 102} in engine.seen_attract_ids


def test_loop_without_vocab_index_never_sets_attract_ids():
    engine = _RecordingEngine(embed_dim=4)
    group = Cyclic(64)
    store = Store(group=group)
    for i in range(3):
        text = f"фрагмент {i}"
        store.add(f"n{i}", text, embedding=engine.embed(text))
    puller = Puller(store, top_k=2, threshold=-1.0)
    prompts = PromptSet(PROMPTS)
    loop = AutoPullLoop(engine, store, puller, group, prompts=prompts,
                        max_steps=3, pull_every=1, max_active=2)  # no vocab_index
    loop.run("найти ответ")
    assert all(a is None for a in engine.seen_attract_ids)


def test_loop_vector_group():
    engine = MockEngine(embed_dim=6)
    group = VectorAdd(6)
    store = Store(group=group)
    for i in range(3):
        text = f"фрагмент {i}"
        store.add(f"n{i}", text, embedding=engine.embed(text))
    puller = Puller(store, top_k=2, threshold=-1.0)
    prompts = PromptSet(PROMPTS)
    loop = AutoPullLoop(engine, store, puller, group, prompts=prompts,
                        max_steps=3, pull_every=1, max_active=2)
    res = loop.run("задача")
    assert res["context"]["h"]
    assert res["context"]["active_count"] <= 2

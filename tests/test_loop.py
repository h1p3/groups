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

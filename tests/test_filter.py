from groupcot.context.puller import Puller, cosine
from groupcot.context.store import Store, detect_meta
from groupcot.groups import Cyclic, FilterRule, VectorAdd


def test_detect_meta_chinese():
    meta = detect_meta("这是一段中文文本")
    assert meta["language"] == "zh"


def test_detect_meta_english():
    meta = detect_meta("This is English text")
    assert meta["language"] == "en"


def test_detect_meta_russian():
    meta = detect_meta("Это русский текст")
    assert meta["language"] == "ru"


def test_detect_meta_mixed_cjk_priority():
    meta = detect_meta("Hello 你好")
    assert meta["language"] == "zh"


def test_store_add_auto_meta():
    g = VectorAdd(8)
    store = Store(g)
    n = store.add("n1", "Привет мир")
    assert n.meta.get("language") == "ru"
    n2 = store.add("n2", "你好世界")
    assert n2.meta.get("language") == "zh"


def test_store_add_explicit_meta_overrides():
    g = VectorAdd(8)
    store = Store(g)
    n = store.add("n1", "Hello", meta={"language": "de"})
    assert n.meta["language"] == "de"


def test_filter_exclude_language():
    g = VectorAdd(8)
    store = Store(g)
    import numpy as np
    e1 = np.ones(8).tolist()
    e2 = np.ones(8) * 2
    e2 = e2.tolist()
    store.add("ru1", "Русский текст", embedding=e1)
    store.add("zh1", "中文文本", embedding=e2)
    rule = FilterRule(type="language", action="exclude", value="zh")
    puller = Puller(store, top_k=10)
    query = np.ones(8).tolist()
    hits = puller.pull(query, filters=[rule])
    ids = [n.node_id for _, n in hits]
    assert "ru1" in ids
    assert "zh1" not in ids


def test_filter_allow_language():
    g = VectorAdd(8)
    store = Store(g)
    import numpy as np
    e1 = np.ones(8).tolist()
    e2 = np.ones(8) * 2
    e2 = e2.tolist()
    store.add("ru1", "Русский текст", embedding=e1)
    store.add("en1", "English text", embedding=e2)
    rule = FilterRule(type="language", action="allow", value="ru")
    puller = Puller(store, top_k=10)
    query = np.ones(8).tolist()
    hits = puller.pull(query, filters=[rule])
    ids = [n.node_id for _, n in hits]
    assert "ru1" in ids
    assert "en1" not in ids


def test_filter_exclude_pattern():
    g = VectorAdd(8)
    store = Store(g)
    import numpy as np
    e1 = np.ones(8).tolist()
    e2 = np.ones(8).tolist()
    store.add("n1", "Играем в шахматы", embedding=e1)
    store.add("n2", "Играем в футбол", embedding=e2)
    rule = FilterRule(type="pattern", action="exclude", value="футбол")
    puller = Puller(store, top_k=10)
    query = np.ones(8).tolist()
    hits = puller.pull(query, filters=[rule])
    ids = [n.node_id for _, n in hits]
    assert "n1" in ids
    assert "n2" not in ids


def test_filter_combined():
    g = VectorAdd(8)
    store = Store(g)
    import numpy as np
    e = np.ones(8).tolist()
    store.add("n1", "Русский текст про шахматы", embedding=e)
    store.add("n2", "中文文本", embedding=e)
    store.add("n3", "English about football", embedding=e)
    rules = [
        FilterRule(type="language", action="exclude", value="zh"),
        FilterRule(type="language", action="exclude", value="en"),
    ]
    puller = Puller(store, top_k=10)
    query = np.ones(8).tolist()
    hits = puller.pull(query, filters=rules)
    ids = [n.node_id for _, n in hits]
    assert ids == ["n1"]


def test_filter_empty_rules_pass_all():
    g = VectorAdd(8)
    store = Store(g)
    import numpy as np
    e = np.ones(8).tolist()
    store.add("n1", "text", embedding=e)
    puller = Puller(store, top_k=10)
    hits = puller.pull(np.ones(8).tolist(), filters=[])
    assert len(hits) == 1


def test_filter_topic():
    g = VectorAdd(8)
    store = Store(g)
    import numpy as np
    e = np.ones(8).tolist()
    store.add("n1", "Шахматы — интересная игра", embedding=e, meta={"topics": ["chess", "board_games"]})
    store.add("n2", "Футбол — популярный вид спорта", embedding=e, meta={"topics": ["football", "sports"]})
    rule = FilterRule(type="topic", action="exclude", value="board_games")
    puller = Puller(store, top_k=10)
    hits = puller.pull(np.ones(8).tolist(), filters=[rule])
    ids = [n.node_id for _, n in hits]
    assert "n1" not in ids
    assert "n2" in ids


def test_cyclic_group_still_works():
    g = Cyclic(64)
    from groupcot.context.aggregate import ContextState
    from groupcot.groups.map import node_contribution
    state = ContextState(g)
    contrib = node_contribution("n1", None, g, 0.5)
    state.add("n1", contrib)
    assert state.h != g.identity()
    state.remove("n1")
    assert state.h == g.identity()


# --- output/feedback filter tests ---


def test_output_filter_exclude_language():
    from groupcot.groups.filter import passes_output_filters
    rule = FilterRule(pipeline="output", type="language", action="exclude", value="zh")
    assert passes_output_filters("Привет мир", [rule]) is True
    assert passes_output_filters("你好世界", [rule]) is False


def test_output_filter_allow_language():
    from groupcot.groups.filter import passes_output_filters
    rule = FilterRule(pipeline="output", type="language", action="allow", value="ru")
    assert passes_output_filters("Привет мир", [rule]) is True
    assert passes_output_filters("Hello world", [rule]) is False


def test_output_filter_pattern():
    from groupcot.groups.filter import passes_output_filters
    rule = FilterRule(pipeline="output", type="pattern", action="exclude", value=r"forbidden")
    assert passes_output_filters("safe text", [rule]) is True
    assert passes_output_filters("this is forbidden", [rule]) is False


def test_output_filter_length():
    from groupcot.groups.filter import passes_output_filters
    rule = FilterRule(pipeline="output", type="length", action="exclude", value="50")
    assert passes_output_filters("short", [rule]) is True
    assert passes_output_filters("x" * 51, [rule]) is False


def test_feedback_filter():
    from groupcot.groups.filter import passes_feedback_filters
    rule = FilterRule(pipeline="feedback", type="pattern", action="exclude", value=r"error")
    assert passes_feedback_filters("good tail", [rule]) is True
    assert passes_feedback_filters("error in output", [rule]) is False


def test_filter_independent_pipelines():
    from groupcot.groups.filter import passes_output_filters, passes_feedback_filters
    rules = [
        FilterRule(pipeline="input", type="language", action="exclude", value="zh"),
        FilterRule(pipeline="output", type="language", action="exclude", value="en"),
        FilterRule(pipeline="feedback", type="pattern", action="exclude", value=r"bad"),
    ]
    assert passes_output_filters("Привет", rules) is True
    assert passes_output_filters("Hello", rules) is False
    assert passes_feedback_filters("good text", rules) is True
    assert passes_feedback_filters("bad text", rules) is False


def test_filter_depends_on():
    from groupcot.groups.filter import passes_output_filters
    output_rule = FilterRule(
        pipeline="output", type="language", action="exclude", value="zh",
        depends_on=["input"],
    )
    input_rule = FilterRule(pipeline="input", type="language", action="exclude", value="zh")
    all_rules = [output_rule, input_rule]
    assert passes_output_filters("你好", all_rules, all_rules) is False
    assert passes_output_filters("Hello", all_rules, all_rules) is True


def test_filter_depends_on_inactive():
    from groupcot.groups.filter import passes_output_filters
    output_rule = FilterRule(
        pipeline="output", type="language", action="exclude", value="zh",
        depends_on=["input"],
    )
    all_rules = [output_rule]
    assert passes_output_filters("你好", all_rules, all_rules) is True


def test_disabled_rule_skipped():
    from groupcot.groups.filter import passes_output_filters
    rule = FilterRule(pipeline="output", type="language", action="exclude", value="zh", enabled=False)
    assert passes_output_filters("你好", [rule]) is True


def test_lang_directive_respects_pipeline():
    from groupcot.prompts import _lang_directive_from_filters
    rules = [
        FilterRule(pipeline="input", type="language", action="exclude", value="zh"),
        FilterRule(pipeline="output", type="language", action="exclude", value="en"),
    ]
    inp = _lang_directive_from_filters(rules, "input")
    out = _lang_directive_from_filters(rules, "output")
    assert "китайском" in inp
    assert "английском" in out
    assert "китайском" not in out


# --- TokenGroup tests ---


def test_token_group_op():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=8)
    a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    b = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int32)
    c = tg.op(a, b)
    assert np.array_equal(c, np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int32))
    assert np.array_equal(tg.inverse(c), c)
    assert np.array_equal(tg.op(a, tg.inverse(a)), tg.identity())


def test_token_group_distance():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=8)
    a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    b = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int32)
    assert tg.distance(a, b) == 1
    assert tg.distance(a, a) == 0


def test_token_group_project():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=8)
    logits = np.random.randn(1000)
    elem = tg.project(logits)
    assert elem.shape == (8,)
    assert set(np.unique(elem)).issubset({0, 1})


def test_token_group_lang_exclude():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=8)
    mask = tg.lang_to_exclude_set("zh", vocab_size=65536)
    assert mask[0x4F60] == True  # 你
    assert mask[0x0041] == False  # A
    zh_chars = [0x4F60, 0x597D, 0x4E16, 0x754C]
    for cid in zh_chars:
        assert mask[cid] == True


def test_logit_filter_exclude():
    from groupcot.groups.token_group import TokenGroup
    from groupcot.groups.logit_filter import LogitFilter
    import numpy as np
    tg = TokenGroup(k=8)
    lf = LogitFilter(tg, vocab_size=1000)
    logits = np.ones(1000)
    mask = tg.lang_to_exclude_set("zh", vocab_size=1000)
    masked = lf.apply(logits, exclude_masks=[mask])
    zh_ids = [cid for cid in range(1000) if mask[cid]]
    non_zh_ids = [cid for cid in range(1000) if not mask[cid]]
    for tid in zh_ids:
        assert masked[tid] == -np.inf
    for tid in non_zh_ids[:5]:
        assert masked[tid] == 1.0


def test_logit_filter_allow():
    from groupcot.groups.token_group import TokenGroup
    from groupcot.groups.logit_filter import LogitFilter
    import numpy as np
    tg = TokenGroup(k=8)
    lf = LogitFilter(tg, vocab_size=1000)
    logits = np.ones(1000)
    mask = tg.lang_to_exclude_set("ru", vocab_size=1000)
    masked = lf.apply(logits, allow_masks=[mask])
    ru_ids = [cid for cid in range(1000) if mask[cid]]
    non_ru_ids = [cid for cid in range(1000) if not mask[cid]]
    for tid in ru_ids[:5]:
        assert masked[tid] == 1.0
    for tid in non_ru_ids:
        assert masked[tid] == -np.inf


def test_logit_filter_lang_exclude():
    from groupcot.groups.token_group import TokenGroup
    from groupcot.groups.logit_filter import LogitFilter
    import numpy as np
    tg = TokenGroup(k=8)
    lf = LogitFilter(tg, vocab_size=1000)
    logits = np.ones(1000)
    masked = lf.apply_lang_exclude(logits, ["zh"])
    zh_ids = [cid for cid in range(1000) if tg.lang_to_exclude_set("zh", 1000)[cid]]
    for tid in zh_ids:
        assert masked[tid] == -np.inf


def test_filter_rule_mode():
    rule = FilterRule(
        pipeline="output", type="language", action="exclude", value="zh",
        mode="logit", group_dim=32,
    )
    assert rule.mode == "logit"
    assert rule.group_dim == 32


def test_logit_filter_blocks_chinese_tokens():
    """Logit filter с mode=logit должен блокировать токены китайского языка.

    Ранее: lang_to_exclude_set использовал Unicode codepoints как token IDs,
    что НЕ совпадает с реальными token IDs токенизатора Qwen.
    Теперь: TokenGroup загружает токенизатор для получения реальных token IDs.
    """
    from groupcot.groups.token_group import TokenGroup
    from groupcot.groups.logit_filter import LogitFilter
    import numpy as np

    tg = TokenGroup(k=64)
    lf = LogitFilter(tg, vocab_size=152064)

    mask_zh = tg.lang_to_exclude_set("zh", vocab_size=152064)
    mask_ru = tg.lang_to_exclude_set("ru", vocab_size=152064)

    logits = np.zeros(152064)
    masked = lf.apply_lang_exclude(logits, ["zh"])

    zh_indices = np.where(mask_zh)[0]
    ru_indices = np.where(mask_ru)[0]

    if len(zh_indices) > 0:
        assert np.all(masked[zh_indices] == -np.inf), \
            f"Chinese tokens should be -inf but got {masked[zh_indices[:5]]}"
    if len(ru_indices) > 0:
        assert np.any(masked[ru_indices] != -np.inf), \
            "Russian tokens should NOT be filtered"


def test_chat_worker_applies_logit_filters():
    """_chat_worker должен применять logit-level фильтры из self.filters (через
    LogitsProcessorChain — единственный путь теперь, что LlamaCppEngine
    единственный поддерживаемый бэкенд, см. ARCHITECTURE.md §11)."""
    from groupcot.gui import GroupGUI
    import inspect
    src = inspect.getsource(GroupGUI._chat_worker)
    assert "logits_processor" in src, "_chat_worker должен передавать logits_processor"


def test_exclude_zh_all_pipelines():
    """Exclude zh filter должен работать и в input, и в output, и в feedback."""
    from groupcot.groups.filter import passes_filters, passes_output_filters, passes_feedback_filters

    node_zh = type("Node", (), {"meta": {"language": "zh"}, "text": "中文"})()
    node_ru = type("Node", (), {"meta": {"language": "ru"}, "text": "Русский"})()

    rules = [
        FilterRule(pipeline="input", type="language", action="exclude", value="zh"),
        FilterRule(pipeline="output", type="language", action="exclude", value="zh"),
        FilterRule(pipeline="feedback", type="language", action="exclude", value="zh"),
    ]

    assert passes_filters(node_zh, rules) == False
    assert passes_filters(node_ru, rules) == True
    assert passes_output_filters("你好世界", rules) == False
    assert passes_output_filters("Привет мир", rules) == True
    assert passes_feedback_filters("你好世界", rules) == False
    assert passes_feedback_filters("Привет мир", rules) == True


def test_mock_engine_chinese_blocked_by_text_filter():
    """Mock engine отвечает по-китайски — post-hoc текстовый фильтр должен заблокировать."""
    from groupcot.engine.mock import MockEngine
    from groupcot.groups.filter import FilterRule, passes_output_filters

    zh_responses = [
        "element: 1\nnarrative: 这是一段中文回复\n",
        "element: 2\nnarrative: 你好世界\n",
    ]
    engine = MockEngine(responses=zh_responses)
    rules = [
        FilterRule(pipeline="output", type="language", action="exclude", value="zh", mode="text"),
    ]

    for resp in zh_responses:
        assert passes_output_filters(resp, rules, rules) == False, \
            f"Chinese response should be blocked: {resp[:50]}"

    ru_responses = [
        "element: 1\nnarrative: Привет мир\n",
        "element: 2\nnarrative: Это русский ответ\n",
    ]
    for resp in ru_responses:
        assert passes_output_filters(resp, rules, rules) == True, \
            f"Russian response should pass: {resp[:50]}"


def test_build_lang_token_ids_from_tokenizer():
    """build_lang_token_ids_from_tokenizer должен находить токены через decode."""
    from groupcot.groups.token_group import TokenGroup

    fake_vocab = {
        0: "<|pad|>",
        1: "Hello",
        2: "你好",
        3: "Привет",
        4: "world",
        5: "мир",
        100: "中文",
        200: "русский",
    }
    vocab_size = 256

    def fake_decode(token_ids):
        return " ".join(fake_vocab.get(tid, "") for tid in token_ids)

    def fake_tokenize(text):
        return [tid for tid, s in fake_vocab.items() if s in text]

    tg = TokenGroup(k=64)
    zh_ids = tg.build_lang_token_ids_from_tokenizer("zh", vocab_size, fake_tokenize, fake_decode)
    assert 2 in zh_ids, f"Token 2 ('你好') should be in zh_ids, got {zh_ids}"
    assert 100 in zh_ids, f"Token 100 ('中文') should be in zh_ids"
    assert 3 not in zh_ids, f"Token 3 ('Привет') should NOT be in zh_ids"

    ru_ids = tg.build_lang_token_ids_from_tokenizer("ru", vocab_size, fake_tokenize, fake_decode)
    assert 3 in ru_ids, f"Token 3 ('Привет') should be in ru_ids"
    assert 200 in ru_ids, f"Token 200 ('русский') should be in ru_ids"
    assert 2 not in ru_ids, f"Token 2 ('你好') should NOT be in ru_ids"

    mask = tg.build_lang_exclude_mask("zh", vocab_size, fake_tokenize, fake_decode)
    assert mask[2] == True
    assert mask[100] == True
    assert mask[3] == False


def test_logit_bias_zh_tokens_are_negative():
    """logit_bias для китайских токенов должен быть -100."""
    from groupcot.groups.token_group import TokenGroup
    tg = TokenGroup(k=64)
    vocab_size = 152064

    zh_ids = list(tg.token_ids_for_lang("zh", vocab_size))
    logit_bias = {tid: -100.0 for tid in zh_ids if tid < vocab_size}

    for tid in zh_ids[:10]:
        assert logit_bias.get(tid) == -100.0, f"Token {tid} should have bias -100"

    ru_ids = list(tg.token_ids_for_lang("ru", vocab_size))
    for tid in ru_ids[:10]:
        assert tid not in logit_bias, f"Russian token {tid} should NOT be in logit_bias"


def test_token_group_distance_matrix():
    from groupcot.groups.token_group import TokenGroup
    tg = TokenGroup(k=4)
    elems = [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (1, 1, 1, 1),
    ]
    mat = tg.distance_matrix(elems)
    assert mat[0, 0] == 0
    assert mat[0, 1] == 1
    assert mat[0, 2] == 4
    assert mat[1, 2] == 3


def test_token_group_logit_accumulate():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=8)
    logits = np.random.randn(1000)
    accum = tg.logit_accumulate(logits)
    assert accum.shape == (8,)
    assert np.all(accum >= 0)
    assert np.all(accum <= 1)


def test_token_group_tokens_in_coset():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=4)
    elements = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.int32)
    center = np.array([0, 0, 0, 0], dtype=np.int32)
    mask = tg.tokens_in_coset(center, elements, max_distance=1)
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == False
    assert mask[3] == False


def test_token_group_project_embedding_is_stable_across_calls():
    """Unlike project()/project_per_token() (keyed off the current logits,
    which change every generation step), project_embedding() must give the
    exact same group element for the exact same embedding every time --
    that's what makes a concept field usable as a fixed coset."""
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=16)
    rng = np.random.RandomState(0)
    e = rng.randn(384).astype(np.float32)
    first = tg.project_embedding(e)
    second = tg.project_embedding(e)
    assert np.array_equal(first, second)
    assert first.shape == (16,)
    assert set(np.unique(first)).issubset({0, 1})


def test_token_group_project_embedding_differs_for_different_embeddings():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=32)
    rng = np.random.RandomState(1)
    e1 = rng.randn(384).astype(np.float32)
    e2 = rng.randn(384).astype(np.float32)
    # Vanishingly unlikely to collide at k=32 for unrelated random vectors --
    # a real regression (e.g. _We collapsing to zero) would make them equal.
    assert not np.array_equal(tg.project_embedding(e1), tg.project_embedding(e2))


def test_token_group_project_embeddings_batch_matches_scalar():
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=12)
    rng = np.random.RandomState(2)
    embeddings = rng.randn(20, 64).astype(np.float32)
    batch = tg.project_embeddings_batch(embeddings)
    assert batch.shape == (20, 12)
    for i in range(20):
        assert np.array_equal(batch[i], tg.project_embedding(embeddings[i]))


def test_token_group_project_embedding_uses_separate_matrix_from_project():
    """project_embedding must not silently reuse _W/_b (the logits-keyed
    matrix) -- that would make it just as unstable as project_per_token."""
    from groupcot.groups.token_group import TokenGroup
    import numpy as np
    tg = TokenGroup(k=8, seed=42)
    rng = np.random.RandomState(3)
    e = rng.randn(50).astype(np.float32)
    tg.project_embedding(e)
    assert tg._We is not None
    assert tg._We.shape == (8, 50)
    # _W (logits-keyed) must be untouched by an embedding-only call.
    assert tg._W is None

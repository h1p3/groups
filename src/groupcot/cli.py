"""CLI entry point for groupcot.

Usage:
    python -m groupcot --help
    python -m groupcot chat --backend llamacpp --model path/to/model.gguf
    python -m groupcot chat --backend mock

Note: the "server" backend (a remote llama-server over HTTP) was removed —
see ARCHITECTURE.md §11. It couldn't support raw logits access, the manual
sampling loop, or the concept/guard/attract machinery this project actually
needs; llamacpp is the only real backend now, mock the only other one.
"""
from __future__ import annotations

import argparse


TAG_SYS = '<im/system>'
TAG_USER = '<im/user>'
TAG_ASST = '<im/assistant>'
TAG_END = '/end\n'
NL = '\n'


def cmd_chat(args):
    engine = _create_engine(args)
    print("Engine: " + args.backend)
    print("Type 'quit' or 'exit' to stop.\n")
    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        history.append({"role": "user", "content": user_input})
        prompt = _build_prompt(history)
        processors = _build_processors(args, engine)
        concept_ids, attract_ids = _build_concept_ids(args, engine)
        kwargs = {"max_tokens": args.max_tokens, "temperature": args.temperature}
        if processors is not None:
            kwargs["logits_processor"] = processors
        blocked = _blocked_ranges_for_args(args)
        if blocked is not None:
            kwargs["blocked_ranges"] = blocked
        if concept_ids is not None:
            kwargs["concept_ids"] = concept_ids
        if attract_ids is not None:
            kwargs["attract_ids"] = attract_ids
        answer = engine.generate(prompt, **kwargs)
        history.append({"role": "assistant", "content": answer})
        print("Assistant: " + answer + "\n")


def cmd_info(args):
    engine = _create_engine(args)
    print("Backend: " + args.backend)
    if hasattr(engine, "vocab_size"):
        print("Vocab size: " + str(engine.vocab_size()))
    if hasattr(engine, "tokenize"):
        tokens = engine.tokenize("Hello world")
        print("Tokenize test: " + str(tokens[:10]) + "...")
    if hasattr(engine, "embed"):
        emb = engine.embed("Hello world")
        print("Embedding dim: " + str(len(emb)))


def cmd_benchmark(args):
    import time
    engine = _create_engine(args)
    prompt = "The capital of France is"
    print("Backend: " + args.backend)
    print("Prompt: " + repr(prompt))
    processors = _build_processors(args, engine)
    concept_ids, attract_ids = _build_concept_ids(args, engine)
    kwargs = {"max_tokens": 50, "temperature": 0.0}
    if processors is not None:
        kwargs["logits_processor"] = processors
    blocked = _blocked_ranges_for_args(args)
    if blocked is not None:
        kwargs["blocked_ranges"] = blocked
    if concept_ids is not None:
        kwargs["concept_ids"] = concept_ids
    if attract_ids is not None:
        kwargs["attract_ids"] = attract_ids
    times = []
    for i in range(args.iterations):
        t0 = time.time()
        out = engine.generate(prompt, **kwargs)
        elapsed = time.time() - t0
        times.append(elapsed)
        print("  run %d: %.2fs -> %s" % (i + 1, elapsed, repr(out[:60])))
    avg = sum(times) / len(times)
    print("\nAverage: %.2fs per generation (50 tokens)" % avg)


def _create_engine(args):
    from .engine import create_engine
    if args.backend == "llamacpp":
        return create_engine(
            "llamacpp",
            model_path=args.model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
        )
    elif args.backend == "mock":
        return create_engine("mock", embed_dim=8)
    else:
        raise ValueError("Unknown backend: " + args.backend)


def _build_processors(args, engine=None):
    from .engine.logits_chain import LogitsProcessorChain
    from .engine.processors import LanguageRedirect
    if not args.exclude_lang:
        return None
    from .engine.llamacpp import LlamaCppEngine
    from .groups.token_group import TokenGroup
    if engine is None:
        engine = _create_engine(args)
    if not isinstance(engine, LlamaCppEngine):
        print("Warning: --exclude-lang only works with llamacpp backend")
        return None
    vocab_size = engine.vocab_size()
    tg = TokenGroup(k=64)
    chain = LogitsProcessorChain()
    for lang in args.exclude_lang:
        lang_ids = set(tg.build_lang_token_ids_from_tokenizer(
            lang, vocab_size, engine.tokenize, engine.detokenize))
        mask = tg.build_exclude_mask_from_tokens(lang_ids, vocab_size)
        chain.add(LanguageRedirect(exclude_mask=mask))
        print("  Exclude lang '%s': %d token IDs" % (lang, len(lang_ids)))
    return chain if len(chain) > 0 else None


def _blocked_ranges_for_args(args):
    """Вернуть диапазоны заблокированных символов для runtime-фильтра.

    Предвычисленной маски токенов недостаточно (BPE-контекст может
    превращать «мусорные» токены в валидные CJK-символы), поэтому при
    исключении языка накладываем ещё посимвольный фильтр.
    """
    from .engine.llamacpp import DEFAULT_BLOCKED_RANGES
    if args.exclude_lang:
        return DEFAULT_BLOCKED_RANGES
    return None


def _build_concept_ids(args, engine):
    """Построить (concept_ids, attract_ids) из --concept через конструктор.

    Возвращает два множества token-ID: для исключения (exclude/constrain) и
    для притяжения (include/attract). Если --concept не задан — (None, None).

    С ``--semantic-concept`` дополнительно строится/кэшируется
    ``VocabIndex`` и компиляция расширяется семантическим поиском ближайших
    токенов по эмбеддингам (V3b, ARCHITECTURE.md §6.3) — это ловит формы,
    которые лексикон V3a пропускает (плюрал, парафраз).
    """
    if not args.concept:
        return None, None
    from .engine.constructor import ConceptConstructor
    ctor = ConceptConstructor(engine)
    vocab_index = None
    if getattr(args, "semantic_concept", False):
        from .engine.vocab_index import VocabIndex
        vocab_index = VocabIndex(engine, max_candidates=args.concept_vocab_size)
        print("  Building/loading vocab embedding index (V3b, %d candidates; "
              "one model call per new candidate, cached after first run)..."
              % args.concept_vocab_size)
        vocab_index.build()
        print("  Vocab index ready: %d candidate tokens" % len(vocab_index.token_ids))
    concept_ids: set[int] = set()
    attract_ids: set[int] = set()
    for intent in args.concept:
        try:
            spec = ctor.construct(intent)
        except Exception as exc:
            print("  ConceptConstructor failed for %r: %s" % (intent, exc))
            continue
        ids = ctor.compile(spec, vocab_index=vocab_index,
                            top_k=args.concept_topk, min_similarity=args.concept_min_sim)
        print("  Concept '%s' (mode=%s): %d token IDs"
              % (spec.concept or intent, spec.mode, len(ids)))
        if spec.mode in ("include", "attract"):
            attract_ids |= ids
        else:
            concept_ids |= ids
    return (concept_ids or None, attract_ids or None)


def _build_prompt(history):
    lines = []
    lines.append(TAG_SYS)
    lines.append("You are a helpful assistant. Answer briefly.")
    lines.append(TAG_END)
    for msg in history:
        lines.append(TAG_USER if msg["role"] == "user" else TAG_ASST)
        lines.append(msg["content"])
        lines.append(TAG_END)
    lines.append(TAG_ASST)
    return NL.join(lines)


def main():
    p = argparse.ArgumentParser(
        prog="groupcot", description="GroupCoT: hypergraph context on groups")
    p.add_argument("--backend", choices=["llamacpp", "mock"],
                   default="mock")
    p.add_argument("--model", default=None, help="GGUF model path (llamacpp)")
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=99,
                   help="Layers to offload to GPU (llamacpp backend); matches "
                        "config.yaml's convention. llama.cpp falls back to CPU "
                        "automatically if no CUDA build/device is present, so "
                        "this is safe on CPU-only machines too. Use 0 to force CPU.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)

    filter_parent = argparse.ArgumentParser(add_help=False)
    filter_parent.add_argument("--exclude-lang", action="append", default=[],
                               help="Exclude language tokens (repeatable)")
    filter_parent.add_argument("--concept", action="append", default=[],
                               help="Semantic concept intent to exclude/include "
                                    "(repeatable); built via the self-query constructor")
    filter_parent.add_argument("--semantic-concept", action="store_true",
                               help="Expand --concept via an embedding-based semantic "
                                    "neighborhood, not just the literal lexicon (V3b, "
                                    "ARCHITECTURE.md §6.3); builds+caches a vocab index "
                                    "on first use, requires an engine with embed()")
    filter_parent.add_argument("--concept-topk", type=int, default=40,
                               help="Nearest tokens per seed to pull in with --semantic-concept")
    filter_parent.add_argument("--concept-min-sim", type=float, default=0.55,
                               help="Minimum cosine similarity for --semantic-concept expansion")
    filter_parent.add_argument("--concept-vocab-size", type=int, default=1500,
                               help="Max candidate tokens embedded into the vocab index "
                                    "(--semantic-concept); each is one model call on first "
                                    "build (cached after), so larger is slower to warm up")

    sub = p.add_subparsers(dest="command")
    sub.add_parser("chat", parents=[filter_parent], help="Interactive chat")
    sub.add_parser("info", help="Show engine info")
    bp = sub.add_parser("benchmark", parents=[filter_parent],
                        help="Benchmark generation")
    bp.add_argument("-n", "--iterations", type=int, default=3)

    args = p.parse_args()
    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
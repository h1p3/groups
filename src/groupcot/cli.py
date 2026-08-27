"""CLI entry point for groupcot.

Usage:
    python -m groupcot --help
    python -m groupcot chat --backend llamacpp --model path/to/model.gguf
    python -m groupcot chat --backend server --base-url http://127.0.0.1:8090
    python -m groupcot chat --backend mock
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
        processors = _build_processors(args)
        kwargs = {"max_tokens": args.max_tokens, "temperature": args.temperature}
        if processors is not None:
            kwargs["logits_processor"] = processors
        blocked = _blocked_ranges_for_args(args)
        if blocked is not None:
            kwargs["blocked_ranges"] = blocked
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
    processors = _build_processors(args)
    kwargs = {"max_tokens": 50, "temperature": 0.0}
    if processors is not None:
        kwargs["logits_processor"] = processors
    blocked = _blocked_ranges_for_args(args)
    if blocked is not None:
        kwargs["blocked_ranges"] = blocked
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
    elif args.backend == "server":
        return create_engine("server", base_url=args.base_url)
    elif args.backend == "mock":
        return create_engine("mock", embed_dim=8)
    else:
        raise ValueError("Unknown backend: " + args.backend)


def _build_processors(args):
    from .engine.logits_chain import LogitsProcessorChain
    from .engine.processors import LanguageRedirect
    if not args.exclude_lang:
        return None
    from .engine.llamacpp import LlamaCppEngine
    from .groups.token_group import TokenGroup
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
    p.add_argument("--backend", choices=["server", "llamacpp", "mock"],
                   default="mock")
    p.add_argument("--model", default=None, help="GGUF model path (llamacpp)")
    p.add_argument("--base-url", default="http://127.0.0.1:8090",
                   help="Server base URL")
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--exclude-lang", action="append", default=[],
                   help="Exclude language tokens (repeatable)")

    sub = p.add_subparsers(dest="command")
    sub.add_parser("chat", help="Interactive chat")
    sub.add_parser("info", help="Show engine info")
    bp = sub.add_parser("benchmark", help="Benchmark generation")
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
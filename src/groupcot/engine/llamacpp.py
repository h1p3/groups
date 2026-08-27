from typing import Callable

import numpy as np

from .base import Engine

try:
    from llama_cpp import Llama, LogitsProcessorList
except ImportError:
    Llama = None
    LogitsProcessorList = None

# Unicode ranges for Chinese / CJK scripts used by the runtime character filter.
# When excluding a language we cannot rely solely on a precomputed token mask
# (BPE contextual decoding can turn a "garbage" token into a valid CJK char), so
# we additionally reject any generated token whose decoded text introduces a
# blocked codepoint and roll back the offending token sequence.
DEFAULT_BLOCKED_RANGES = (
    (0x3400, 0x4DBF),   # CJK Ext A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility
    (0x20000, 0x2A6DF), # CJK Ext B
    (0x2A700, 0x2EBEF), # CJK Ext C-F
    (0x3000, 0x303F),   # CJK symbols and punctuation
    (0xFF00, 0xFFEF),   # Fullwidth forms
)


class LlamaCppEngine(Engine):
    def __init__(self, model_path, n_ctx: int = 8192, n_gpu_layers: int = 0,
                 seed: int = -1, verbose: bool = False, **kwargs):
        if Llama is None:
            raise RuntimeError("llama-cpp-python is not installed (pip install groupcot[llamacpp])")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
            logits_all=True,
            embedding=True,
            **kwargs,
        )

    def generate(self, prompt, grammar=None, max_tokens: int = 128,
                 temperature: float = 0.7, logits_processor: Callable | None = None,
                 logit_bias: dict[int, float] | None = None,
                 blocked_ranges: tuple | None = None,
                 top_p: float = 0.95, top_k: int = 40, seed=None) -> str:
        # When a runtime character filter is requested we must sample manually
        # because llama-cpp-python's built-in logits_processor hook is unreliable
        # at temperature > 0 (it only touches top-K candidates).
        if blocked_ranges is not None:
            return self._generate_filtered(
                prompt, max_tokens=max_tokens, temperature=temperature,
                logits_processor=logits_processor, blocked_ranges=blocked_ranges,
                top_p=top_p, top_k=top_k, seed=seed,
            )

        kwargs: dict = dict(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if grammar:
            kwargs["grammar"] = grammar
        if logits_processor:
            if LogitsProcessorList is not None:
                llp = LogitsProcessorList()
                llp.append(logits_processor)
                kwargs["logits_processor"] = llp
            else:
                kwargs["logits_processor"] = logits_processor
        if logit_bias:
            kwargs["logit_bias"] = logit_bias
        out = self.llm.create_completion(**kwargs)
        return out["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Manual generation with runtime character-level filtering
    # ------------------------------------------------------------------
    def _is_blocked_char(self, ch: str) -> bool:
        o = ord(ch)
        for lo, hi in self._blocked_ranges:
            if lo <= o <= hi:
                return True
        return False

    def _count_blocked(self, text: str) -> int:
        return sum(1 for c in text if self._is_blocked_char(c))

    def _find_rollback(self, generated: list[int], candidate: int) -> int | None:
        """Return index in `generated` to roll back to (exclusive) if adding
        `candidate` introduces a blocked codepoint, else None."""
        seq = generated + [candidate]
        prev_text = self.detokenize(generated)
        full_text = self.detokenize(seq)
        prev_n = self._count_blocked(prev_text)
        full_n = self._count_blocked(full_text)
        if full_n <= prev_n:
            return None
        # Search backwards over the trailing window that could form the char.
        lo = max(0, len(generated) - 8)
        for i in range(len(seq), lo - 1, -1):
            if self._count_blocked(self.detokenize(seq[:i])) == prev_n:
                return i
        return max(0, lo)

    def _sample_token(self, logits: np.ndarray, temperature: float,
                      top_p: float, top_k: int, seed) -> int:
        logits = np.asarray(logits, dtype=np.float32)
        if temperature == 0:
            return int(np.argmax(logits))
        if top_k and top_k > 0:
            k = min(top_k, len(logits))
            idx = np.argpartition(logits, -k)[-k:]
            mask = np.full(len(logits), -np.inf, dtype=np.float32)
            mask[idx] = logits[idx]
            logits = mask
        logits = logits / temperature
        mx = np.max(logits)
        if not np.isfinite(mx):
            return int(np.argmax(logits))
        e = np.exp(logits - mx)
        p = e / np.sum(e)
        if top_p and top_p < 1.0:
            order = np.argsort(p)[::-1]
            cum = np.cumsum(p[order])
            keep = order[cum <= top_p]
            if len(keep) == 0:
                keep = order[:1]
            p2 = np.zeros_like(p)
            p2[keep] = p[keep]
            p2 /= p2.sum()
            p = p2
        return int(np.random.choice(len(p), p=p))

    def _generate_filtered(self, prompt, max_tokens, temperature,
                           logits_processor, blocked_ranges,
                           top_p, top_k, seed) -> str:
        self._blocked_ranges = blocked_ranges
        tokens = self.tokenize(prompt)
        self.llm.reset()
        self.llm.eval(tokens)
        generated: list[int] = []
        for _ in range(max_tokens):
            logits = np.array(self.llm.eval_logits[-1], dtype=np.float32)
            if logits_processor is not None:
                logits = logits_processor(
                    self.llm.input_ids[: self.llm.n_tokens], logits)
            candidate = self._sample_token(logits, temperature, top_p, top_k, seed)
            rollback = self._find_rollback(generated, candidate)
            tries = 0
            while rollback is not None and tries < 200:
                logits[candidate] = -np.inf
                candidate = self._sample_token(logits, temperature, top_p, top_k, seed)
                rollback = self._find_rollback(generated, candidate)
                tries += 1
            if rollback is not None:
                candidate = int(np.argmax(logits))
                rollback = self._find_rollback(generated, candidate)
            if rollback is not None:
                generated = generated[:rollback]
                self.llm.reset()
                self.llm.eval(tokens + generated)
            generated.append(candidate)
            self.llm.eval([candidate])
            if candidate == self.llm.token_eos():
                break
        return self.detokenize(generated)

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text using the llama.cpp tokenizer."""
        return list(self.llm.tokenize(text.encode("utf-8")))

    def detokenize(self, token_ids: list[int]) -> str:
        """Detokenize token IDs using the llama.cpp tokenizer."""
        return self.llm.detokenize(token_ids).decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        """Return the model's vocabulary size."""
        return self.llm.n_vocab()

    def embed(self, text) -> list[float]:
        out = self.llm.create_embedding(text)
        return out["data"][0]["embedding"]

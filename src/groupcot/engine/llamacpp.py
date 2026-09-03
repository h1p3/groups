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
                 concept_ids: set[int] | None = None,
                 attract_ids: set[int] | None = None,
                 attract_weight: float = 5.0,
                 mix_ids: set[int] | None = None,
                 mix_alpha: float = 0.0,
                 mix_weights: dict[int, float] | None = None,
                 top_p: float = 0.95, top_k: int = 40, seed=None) -> str:
        # Manual sampling is required whenever a runtime filter is active,
        # because llama-cpp-python's built-in logits_processor hook is unreliable
        # at temperature > 0 (it only touches top-K candidates). This also covers
        # semantic concept suppression / attraction (concept_ids / attract_ids)
        # and probability mixing (mix_ids/mix_alpha, ARCHITECTURE.md §12) --
        # the latter needs p_natural = softmax(logits) before sampling, which
        # only this manual path has access to.
        if (blocked_ranges is not None or concept_ids is not None
                or attract_ids is not None or mix_ids is not None):
            return self._generate_filtered(
                prompt, max_tokens=max_tokens, temperature=temperature,
                logits_processor=logits_processor, blocked_ranges=blocked_ranges,
                concept_ids=concept_ids, attract_ids=attract_ids,
                attract_weight=attract_weight,
                mix_ids=mix_ids, mix_alpha=mix_alpha, mix_weights=mix_weights,
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

    @staticmethod
    def _build_mix_probs(vocab_size: int, mix_ids: set[int],
                         mix_weights: dict[int, float] | None) -> np.ndarray:
        """p_concept: a fixed distribution over mix_ids -- uniform, unless
        mix_weights gives relative weights (e.g. from VocabIndex similarity
        scores or coset distance), in which case those are renormalized to
        sum to 1. Built once per generate() call (§12.1) since the field
        itself doesn't change token to token."""
        probs = np.zeros(vocab_size, dtype=np.float64)
        ids = [tid for tid in mix_ids if 0 <= tid < vocab_size]
        if not ids:
            return probs
        if mix_weights:
            w = np.array([max(mix_weights.get(tid, 0.0), 0.0) for tid in ids], dtype=np.float64)
            if w.sum() <= 0:
                w = np.ones(len(ids), dtype=np.float64)
        else:
            w = np.ones(len(ids), dtype=np.float64)
        probs[ids] = w / w.sum()
        return probs

    @staticmethod
    def _mix_and_sample(logits: np.ndarray, mix_probs: np.ndarray, alpha: float,
                        temperature: float, top_p: float, top_k: int) -> int:
        """Sample from p_final = (1-alpha)*p_natural + alpha*p_concept
        (ARCHITECTURE.md §12.1) instead of shifting logits additively.
        Unlike an additive bias, alpha in [0,1] can't blow up the
        distribution regardless of the model's logit scale -- both terms are
        already normalized probabilities, so the convex combination always
        is too.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if top_k and top_k > 0:
            k = min(top_k, len(logits))
            idx = np.argpartition(logits, -k)[-k:]
            masked = np.full(len(logits), -np.inf, dtype=np.float32)
            masked[idx] = logits[idx]
            logits = masked
        t = temperature if temperature > 0 else 1.0
        scaled = logits / t
        mx = np.max(scaled)
        if not np.isfinite(mx):
            p_natural = np.zeros(len(logits), dtype=np.float64)
            p_natural[int(np.argmax(logits))] = 1.0
        else:
            e = np.exp(scaled - mx)
            p_natural = (e / np.sum(e)).astype(np.float64)
        if top_p and top_p < 1.0:
            order = np.argsort(p_natural)[::-1]
            cum = np.cumsum(p_natural[order])
            keep = order[cum <= top_p]
            if len(keep) == 0:
                keep = order[:1]
            p2 = np.zeros_like(p_natural)
            p2[keep] = p_natural[keep]
            p2 /= p2.sum()
            p_natural = p2

        alpha = min(max(alpha, 0.0), 1.0)
        p_final = (1.0 - alpha) * p_natural + alpha * mix_probs
        total = p_final.sum()
        if total <= 0:
            return int(np.argmax(p_natural))
        p_final = p_final / total
        if temperature == 0:
            return int(np.argmax(p_final))
        return int(np.random.choice(len(p_final), p=p_final))

    def _generate_filtered(self, prompt, max_tokens, temperature,
                            logits_processor, blocked_ranges,
                            concept_ids, attract_ids, attract_weight,
                            top_p, top_k, seed,
                            mix_ids=None, mix_alpha: float = 0.0,
                            mix_weights: dict[int, float] | None = None) -> str:
        self._blocked_ranges = blocked_ranges or ()
        cids = set(concept_ids) if concept_ids else set()
        aids = set(attract_ids) if attract_ids else set()
        mids = set(mix_ids) if mix_ids else set()
        tokens = self.tokenize(prompt)

        # p_concept (§12.1) is fixed for the whole call -- the field doesn't
        # change token to token, only vocab_size does (once we know it).
        mix_probs = None
        if mids and mix_alpha > 0:
            mix_probs = self._build_mix_probs(self.vocab_size(), mids, mix_weights)

        def pick(logits_arr):
            if mix_probs is not None:
                return self._mix_and_sample(logits_arr, mix_probs, mix_alpha,
                                            temperature, top_p, top_k)
            return self._sample_token(logits_arr, temperature, top_p, top_k, seed)

        self.llm.reset()
        self.llm.eval(tokens)
        generated: list[int] = []
        for _ in range(max_tokens):
            # NOTE: `self.llm.eval_logits[-1]` (llama-cpp-python's own API) is
            # ~160x slower than this for no reason worth paying: its property
            # getter does `self.scores[:self.n_tokens, :].tolist()` -- converts
            # the WHOLE history-so-far to a Python list, growing every step,
            # just to hand back the last row. `self.llm.scores` is already a
            # numpy array; indexing the one row we want and copying it (to
            # avoid aliasing the engine's internal buffer, since we mutate
            # `logits` below) is a single vocab_size-sized copy, not
            # n_tokens*vocab_size. Verified bit-identical output, measured
            # ~160x faster in isolation -- this was the actual bottleneck
            # behind "attract is 15x slower than exclude" (both paths pay this
            # per token; chunked/guarded generation just pays it more times).
            logits = np.array(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float32)
            if logits_processor is not None:
                logits = logits_processor(
                    self.llm.input_ids[: self.llm.n_tokens], logits)
            # Semantic concept suppression / attraction (dual operation, §3.3).
            # Exclusion always applies to the raw logits first, mixing or not
            # -- a token forbidden by concept_ids must stay unreachable
            # regardless of whether p_concept would otherwise favor it.
            if cids:
                for tid in cids:
                    if tid < len(logits):
                        logits[tid] = -np.inf
            if aids:
                for tid in aids:
                    if tid < len(logits):
                        logits[tid] += attract_weight
            candidate = pick(logits)
            # Reject excluded concept tokens (numerical safety after sampling).
            tries = 0
            while candidate in cids and tries < 50:
                logits[candidate] = -np.inf
                candidate = pick(logits)
                tries += 1
            # Character-level blocked-ranges rollback (language exclusion).
            rollback = self._find_rollback(generated, candidate)
            while rollback is not None and tries < 200:
                logits[candidate] = -np.inf
                candidate = pick(logits)
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

    @staticmethod
    def _pool_embedding(vecs):
        """Mean-pool a per-token embedding matrix down to one vector, if needed."""
        if vecs and isinstance(vecs[0], list):
            n = len(vecs)
            dim = len(vecs[0])
            pooled = [0.0] * dim
            for v in vecs:
                for i, x in enumerate(v):
                    pooled[i] += x
            return [x / n for x in pooled]
        return vecs

    def embed(self, text) -> list[float]:
        out = self.llm.create_embedding(text)
        return self._pool_embedding(out["data"][0]["embedding"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (used by VocabIndex, §6.3/V3b).

        NOTE: this deliberately does NOT pass a list to
        ``llm.create_embedding`` in one call — that path was found to return
        corrupted per-item vectors (each item beyond the first is the
        previous item's raw float buffer read at a shifted offset, not an
        independent embedding) on llama-cpp-python 0.3.35. One call per text
        is slower but correct.
        """
        return [self.embed(t) for t in texts]

    def chat(self, messages, grammar=None, max_tokens: int = 128,
             temperature: float = 0.7) -> str:
        kwargs: dict = dict(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
        if grammar:
            kwargs["grammar"] = grammar
        out = self.llm.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"]

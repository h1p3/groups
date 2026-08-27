from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from .logits_chain import LogitsProcessor


_REGISTRY: dict[str, type] = {}


def register_processor(name: str, cls: type | None = None):
    """Register a processor class by name.

    As a decorator:
        @register_processor("my_proc")
        class MyProcessor:
            ...

    As a direct call:
        register_processor("my_proc", MyProcessor)
    """
    def _wrap(c: type) -> type:
        _REGISTRY[c.__name__] = c
        if name != c.__name__:
            _REGISTRY[name] = c
        return c

    if cls is not None:
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_processor_class(name: str) -> type | None:
    """Look up a registered processor class by name."""
    return _REGISTRY.get(name)


def list_processors() -> dict[str, type]:
    """Return all registered processor classes."""
    return dict(_REGISTRY)


def build_chain(specs: list[dict[str, Any]], engine=None) -> "LogitsProcessorChain | None":
    """Build a LogitsProcessorChain from a list of dicts.

    Each dict must have a "type" key matching a registered processor name.
    Extra keys are passed as kwargs to the constructor.

    Example:
        build_chain([
            {"type": "LanguageRedirect", "exclude_ids": [100, 200], "boost_strength": 5.0},
            {"type": "TokenBias", "biases": {10: 3.0}},
        ])
    """
    from .logits_chain import LogitsProcessorChain

    if not specs:
        return None

    chain = LogitsProcessorChain()
    for spec in specs:
        proc_type = spec.get("type")
        if proc_type is None:
            raise ValueError("Each processor spec must have a 'type' key")
        cls = get_processor_class(proc_type)
        if cls is None:
            raise ValueError(f"Unknown processor type: {proc_type!r}. "
                             f"Available: {list(_REGISTRY.keys())}")
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        chain.add(cls(**kwargs))

    return chain if len(chain) > 0 else None


class LanguageRedirect:
    """Exclude tokens of one language and boost tokens of another.

    Works with both pre-computed token id sets and token-group based filtering.
    """

    def __init__(
        self,
        exclude_ids: set[int] | None = None,
        boost_ids: set[int] | None = None,
        exclude_mask: np.ndarray | None = None,
        boost_mask: np.ndarray | None = None,
        boost_strength: float = 5.0,
    ) -> None:
        self.exclude_ids = exclude_ids or set()
        self.boost_ids = boost_ids or set()
        self.exclude_mask = exclude_mask
        self.boost_mask = boost_mask
        self.boost_strength = boost_strength

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        result = scores.copy()

        if self.exclude_mask is not None:
            mask = self.exclude_mask[: len(result)]
            result[mask] = -np.inf
        elif self.exclude_ids:
            for tid in self.exclude_ids:
                if tid < len(result):
                    result[tid] = -np.inf

        if self.boost_mask is not None:
            mask = self.boost_mask[: len(result)]
            result[mask] += self.boost_strength
        elif self.boost_ids:
            for tid in self.boost_ids:
                if tid < len(result):
                    result[tid] += self.boost_strength

        return result


class PatternBlock:
    """Block tokens whose decoded text matches a regex pattern.

    Builds a mask by scanning the vocabulary with the tokenizer.
    """

    def __init__(
        self,
        block_pattern: str | None = None,
        block_ids: set[int] | None = None,
        block_mask: np.ndarray | None = None,
    ) -> None:
        self.block_pattern = re.compile(block_pattern) if block_pattern else None
        self.block_ids = block_ids or set()
        self.block_mask = block_mask

    @classmethod
    def from_tokenizer(
        cls,
        pattern: str,
        vocab_size: int,
        tokenize_fn,
        decode_fn,
    ) -> PatternBlock:
        """Build a PatternBlock by scanning the tokenizer vocabulary."""
        compiled = re.compile(pattern)
        blocked_ids: set[int] = set()
        for token_id in range(vocab_size):
            try:
                text = decode_fn([token_id])
            except Exception:
                continue
            if compiled.search(text):
                blocked_ids.add(token_id)
        return cls(block_ids=blocked_ids)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        result = scores.copy()

        if self.block_mask is not None:
            mask = self.block_mask[: len(result)]
            result[mask] = -np.inf
        elif self.block_ids:
            for tid in self.block_ids:
                if tid < len(result):
                    result[tid] = -np.inf

        return result


class SemanticShift:
    """Shift logits toward tokens whose group projection is close to a target element.

    Uses TokenGroup to compute Hamming distance and boost/penalize tokens.
    """

    def __init__(
        self,
        target_element: np.ndarray,
        token_group,  # TokenGroup
        boost_strength: float = 2.0,
        max_distance: int | None = None,
    ) -> None:
        self.target = np.asarray(target_element, dtype=np.int32)
        self.tg = token_group
        self.boost_strength = boost_strength
        self.max_distance = max_distance or token_group.k // 2

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        result = scores.copy()
        vocab_size = len(result)
        self.tg._ensure_W(vocab_size)

        # Vectorized: compute distance from each token's projected element to target
        # Project: sign(W[tid] + b) -> {0,1}^k
        projections = (self.tg._W[:vocab_size] + self.tg._b[None, :] > 0).astype(np.int32)
        dists = np.sum(projections != self.target[None, :], axis=1)
        close_mask = dists <= self.max_distance
        result[close_mask] += self.boost_strength

        return result


class TokenBias:
    """Apply additive bias to specific token IDs.

    Unlike logit_bias which only excludes, this can add arbitrary offsets.
    """

    def __init__(self, biases: dict[int, float] | None = None,
                 bias_mask: np.ndarray | None = None) -> None:
        self.biases = biases or {}
        self.bias_mask = bias_mask

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        result = scores.copy()

        if self.bias_mask is not None:
            mask = self.bias_mask[: len(result)]
            np.add(result, self.bias_mask[: len(result)], out=result,
                   where=mask.astype(bool))
        elif self.biases:
            for tid, bias in self.biases.items():
                if tid < len(result):
                    result[tid] += bias

        return result


# Register built-in processors
register_processor("LanguageRedirect", LanguageRedirect)
register_processor("PatternBlock", PatternBlock)
register_processor("SemanticShift", SemanticShift)
register_processor("TokenBias", TokenBias)

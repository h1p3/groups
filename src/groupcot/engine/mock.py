from typing import Callable

import numpy as np

from .base import Engine


class MockEngine(Engine):
    def __init__(self, responses=None, embed_dim: int = 8):
        self.responses = list(responses or [])
        self.embed_dim = embed_dim
        self.call_count = 0

    def generate(self, prompt, grammar=None, max_tokens: int = 128,
                 temperature: float = 0.7, logits_processor: Callable | None = None,
                 logit_bias: dict[int, float] | None = None,
                 blocked_ranges=None) -> str:
        idx = self.call_count
        self.call_count += 1
        if self.responses:
            return self.responses[idx % len(self.responses)]
        return "element: 7\nnarrative: (mock) генерация шага\n"

    def embed(self, text) -> list[float]:
        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.normal(size=self.embed_dim).tolist()

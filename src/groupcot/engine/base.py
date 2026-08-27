from abc import ABC, abstractmethod
from typing import Callable


class Engine(ABC):
    @abstractmethod
    def generate(self, prompt, grammar=None, max_tokens: int = 128,
                 temperature: float = 0.7, logits_processor: Callable | None = None,
                 logit_bias: dict[int, float] | None = None,
                 blocked_ranges=None) -> str: ...

    @abstractmethod
    def embed(self, text) -> list[float]: ...

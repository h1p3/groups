from abc import ABC, abstractmethod
from typing import Callable


class Engine(ABC):
    @abstractmethod
    def generate(self, prompt, grammar=None, max_tokens: int = 128,
                 temperature: float = 0.7, logits_processor: Callable | None = None,
                 logit_bias: dict[int, float] | None = None,
                 blocked_ranges=None,
                 concept_ids: set[int] | None = None,
                 attract_ids: set[int] | None = None,
                 attract_weight: float = 5.0) -> str: ...

    @abstractmethod
    def chat(self, messages, grammar=None, max_tokens: int = 128,
             temperature: float = 0.7) -> str: ...

    @abstractmethod
    def embed(self, text) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: sequential ``embed`` calls; engines
        that can batch (e.g. a local llama.cpp model) should override this."""
        return [self.embed(t) for t in texts]

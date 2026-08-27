from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LogitsProcessor(Protocol):
    """Protocol for logits processors.

    Each processor receives raw input token ids and scores (vocab_size,),
    and returns modified scores of the same shape.
    """

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray: ...


class LogitsProcessorChain:
    """Chains multiple logits processors sequentially.

    Compatible with llama-cpp-python's logits_processor parameter.
    """

    def __init__(self) -> None:
        self.processors: list[LogitsProcessor] = []

    def add(self, processor: LogitsProcessor) -> None:
        self.processors.append(processor)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        for proc in self.processors:
            scores = proc(input_ids, scores)
        return scores

    def __len__(self) -> int:
        return len(self.processors)

    def __repr__(self) -> str:
        names = [type(p).__name__ for p in self.processors]
        return f"LogitsProcessorChain({', '.join(names)})"

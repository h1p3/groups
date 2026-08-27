from __future__ import annotations

from typing import Callable

import numpy as np


class RawLogitsAccessor:
    """Captures raw logits at each generation step for analysis.

    Usage as a LogitsProcessor (passthrough — does not modify logits):
        accessor = RawLogitsAccessor()
        engine.generate(prompt, logits_processor=accessor)
        print(accessor.last_logits)     # most recent raw scores
        print(accessor.history)         # all captured scores

    Usage as a callback:
        def on_logits(step, input_ids, scores):
            print(f"step={step}, shape={scores.shape}")

        accessor = RawLogitsAccessor(callback=on_logits)
    """

    def __init__(self, callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
                 max_history: int = 100) -> None:
        self._step = 0
        self._last_logits: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._max_history = max_history
        self._callback = callback

    @property
    def last_logits(self) -> np.ndarray | None:
        """Most recently captured raw scores (vocab_size,)."""
        return self._last_logits

    @property
    def history(self) -> list[np.ndarray]:
        """List of all captured score vectors."""
        return list(self._history)

    @property
    def step(self) -> int:
        """Number of times __call__ has been invoked."""
        return self._step

    def reset(self) -> None:
        """Clear all captured history."""
        self._step = 0
        self._last_logits = None
        self._history.clear()

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Passthrough processor — captures scores without modifying them.

        Compatible with llama-cpp-python logits_processor API.
        """
        self._last_logits = scores.copy()
        if len(self._history) < self._max_history:
            self._history.append(self._last_logits)
        self._step += 1

        if self._callback is not None:
            self._callback(self._step, input_ids, scores)

        return scores

    def top_k_tokens(self, k: int = 10) -> list[tuple[int, float]]:
        """Get top-k token IDs and scores from the last captured logits."""
        if self._last_logits is None:
            return []
        indices = np.argsort(self._last_logits)[::-1][:k]
        return [(int(idx), float(self._last_logits[idx])) for idx in indices]

    def save(self, path: str) -> None:
        """Save all captured history to a .npz file."""
        if not self._history:
            return
        arrays = {f"step_{i}": arr for i, arr in enumerate(self._history)}
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load history from a .npz file."""
        data = np.load(path)
        self._history = [data[key] for key in sorted(data.keys())]
        self._last_logits = self._history[-1] if self._history else None
        self._step = len(self._history)

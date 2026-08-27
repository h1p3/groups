import json

import numpy as np

from .base import Group


class VectorAdd(Group):
    name = "vector"

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def op(self, a, b):
        return np.asarray(a, dtype=float) + np.asarray(b, dtype=float)

    def inverse(self, a):
        return -np.asarray(a, dtype=float)

    def identity(self):
        return np.zeros(self.dim, dtype=float)

    def parse(self, text: str):
        return np.asarray(json.loads(text), dtype=float)

    def to_text(self, a):
        return json.dumps(np.asarray(a, dtype=float).round(6).tolist())

    def compact(self, a):
        v = np.asarray(a, dtype=float).round(4)
        return f"{json.dumps(v[:8].tolist())} …(dim={self.dim})"

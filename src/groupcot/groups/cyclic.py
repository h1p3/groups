from .base import Group


class Cyclic(Group):
    name = "cyclic"

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def _norm(self, a) -> int:
        return int(a) % self.n

    def op(self, a, b):
        return self._norm(int(a) + int(b))

    def inverse(self, a):
        return self._norm(-int(a))

    def identity(self):
        return 0

    def parse(self, text: str):
        return self._norm(text.strip())

    def to_text(self, a):
        return str(self._norm(a))

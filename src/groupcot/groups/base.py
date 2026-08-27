from abc import ABC, abstractmethod
from typing import Any


class Group(ABC):
    name: str = "group"

    @abstractmethod
    def op(self, a: Any, b: Any) -> Any: ...

    @abstractmethod
    def inverse(self, a: Any) -> Any: ...

    @abstractmethod
    def identity(self) -> Any: ...

    @abstractmethod
    def parse(self, text: str) -> Any: ...

    @abstractmethod
    def to_text(self, a: Any) -> str: ...

    def compact(self, a: Any) -> str:
        return self.to_text(a)

from ..groups.base import Group


class ContextState:
    def __init__(self, group: Group):
        self.group = group
        self.h = group.identity()
        self._contrib: dict[str, object] = {}

    def add(self, node_id: str, element) -> None:
        if node_id in self._contrib:
            self.h = self.group.op(self.h, self.group.inverse(self._contrib[node_id]))
        self.h = self.group.op(self.h, element)
        self._contrib[node_id] = element

    def remove(self, node_id: str) -> None:
        if node_id in self._contrib:
            self.h = self.group.op(self.h, self.group.inverse(self._contrib[node_id]))
            del self._contrib[node_id]

    def active_ids(self):
        return set(self._contrib.keys())

    def __len__(self) -> int:
        return len(self._contrib)

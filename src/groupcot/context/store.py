import json
import re
from pathlib import Path

import numpy as np

from ..groups.base import Group


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")


def detect_meta(text: str) -> dict:
    """Авто-детект метаданных узла по тексту."""
    meta: dict = {}
    if _CJK_RE.search(text):
        meta["language"] = "zh"
    elif _LATIN_RE.search(text):
        meta["language"] = "en"
    elif _CYRILLIC_RE.search(text):
        meta["language"] = "ru"
    else:
        meta["language"] = "unknown"
    return meta


class Node:
    def __init__(self, node_id, text, embedding=None, edges=None, meta=None):
        self.node_id = node_id
        self.text = text
        self.embedding = embedding
        self.edges = list(edges or [])
        self.meta = dict(meta or {})


class Store:
    def __init__(self, group, path: Path | None = None):
        self.group = group
        self.path = path
        self._nodes: dict[str, Node] = {}

    def add(self, node_id, text, embedding=None, edges=None, meta=None) -> Node:
        merged = detect_meta(text)
        if meta:
            merged.update(meta)
        node = Node(node_id, text, embedding, edges, merged)
        self._nodes[node_id] = node
        return node

    def get(self, node_id) -> Node | None:
        return self._nodes.get(node_id)

    def remove(self, node_id) -> None:
        self._nodes.pop(node_id, None)

    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def neighbors(self, node_id) -> list[Node]:
        node = self.get(node_id)
        if node is None:
            return []
        return [self._nodes[nid] for nid in node.edges if nid in self._nodes]

    def save(self, path: Path | None = None):
        path = Path(path or self.path)
        if path is None:
            raise ValueError("no path given")
        data = [
            {
                "id": node.node_id,
                "text": node.text,
                "embedding": None if node.embedding is None else np.asarray(node.embedding, dtype=float).tolist(),
                "edges": node.edges,
                "meta": node.meta,
            }
            for node in self._nodes.values()
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, group: Group) -> "Store":
        store = cls(group, path=Path(path))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in data:
            emb = None
            if row.get("embedding") is not None:
                emb = np.asarray(row["embedding"], dtype=float)
            store.add(row["id"], row["text"], emb, row.get("edges") or [], row.get("meta") or {})
        return store

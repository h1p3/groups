import numpy as np

from ..groups.filter import passes_filters


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Puller:
    def __init__(self, store, top_k: int = 2, threshold: float = 0.0):
        self.store = store
        self.top_k = top_k
        self.threshold = threshold

    def pull(self, query_embedding, exclude=(), filters=None):
        exclude = set(exclude)
        scored = []
        for node in self.store.nodes():
            if node.embedding is None or node.node_id in exclude:
                continue
            if filters and not passes_filters(node, filters):
                continue
            score = cosine(query_embedding, node.embedding)
            if score >= self.threshold:
                scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[: self.top_k]

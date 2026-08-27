import hashlib

import numpy as np

from .base import Group
from .cyclic import Cyclic
from .vector import VectorAdd


def node_element(node_id: str, group: Group):
    if isinstance(group, Cyclic):
        digest = hashlib.sha256(node_id.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        return group.parse(str(value))
    raise TypeError(f"node_element is not defined for {type(group).__name__}")


def node_contribution(node_id: str, embedding, group: Group, score: float = 1.0):
    if isinstance(group, VectorAdd):
        if embedding is None:
            raise ValueError(f"embedding is required for node {node_id!r} in a vector group")
        return np.asarray(embedding, dtype=float) * float(score)
    if isinstance(group, Cyclic):
        return node_element(node_id, group)
    raise TypeError(f"node_contribution is not defined for {type(group).__name__}")

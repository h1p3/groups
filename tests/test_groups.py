import numpy as np
import pytest

from groupcot.groups.cyclic import Cyclic
from groupcot.groups.vector import VectorAdd


def test_cyclic_ops():
    g = Cyclic(12)
    assert g.op(10, 5) == 3
    assert g.inverse(7) == 5
    assert g.op(g.inverse(7), 7) == g.identity()
    assert g.op(7, 8) == g.op(8, 7)
    assert g.parse(g.to_text(11)) == 11


def test_cyclic_associativity():
    g = Cyclic(12)
    for a in range(12):
        for b in range(12):
            for c in range(12):
                assert g.op(g.op(a, b), c) == g.op(a, g.op(b, c))


def test_cyclic_negative_n():
    with pytest.raises(ValueError):
        Cyclic(0)


def test_vector_ops():
    g = VectorAdd(4)
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert np.allclose(g.op(a, b), [5.0, 5.0, 5.0, 5.0])
    assert np.allclose(g.op(a, g.inverse(a)), g.identity())
    assert np.allclose(g.op(a, b), g.op(b, a))


def test_vector_parse_roundtrip():
    g = VectorAdd(3)
    a = [1.5, -2.0, 0.25]
    assert np.allclose(g.parse(g.to_text(a)), a)

import numpy as np

from groupcot.context.aggregate import ContextState
from groupcot.groups.cyclic import Cyclic
from groupcot.groups.map import node_element
from groupcot.groups.vector import VectorAdd


def test_add_remove_cyclic():
    g = Cyclic(64)
    s = ContextState(g)
    s.add("a", 5)
    s.add("b", 9)
    assert s.h == 14
    s.remove("a")
    assert s.h == 9
    assert len(s) == 1
    assert s.active_ids() == {"b"}


def test_commutativity_cyclic():
    g = Cyclic(64)
    s1 = ContextState(g)
    s1.add("a", 5)
    s1.add("b", 9)
    s2 = ContextState(g)
    s2.add("b", 9)
    s2.add("a", 5)
    assert s1.h == s2.h


def test_add_remove_vector():
    g = VectorAdd(4)
    s = ContextState(g)
    s.add("a", [1.0, 2.0, 3.0, 4.0])
    s.add("b", [4.0, 3.0, 2.0, 1.0])
    assert np.allclose(s.h, [5.0, 5.0, 5.0, 5.0])
    s.remove("a")
    assert np.allclose(s.h, [4.0, 3.0, 2.0, 1.0])


def test_reread_same_node():
    g = Cyclic(64)
    s = ContextState(g)
    s.add("a", 5)
    s.add("a", 7)
    assert s.h == 7
    assert len(s) == 1
    s.remove("a")
    assert s.h == g.identity()


def test_node_element_deterministic():
    g = Cyclic(64)
    assert node_element("n1", g) == node_element("n1", g)
    assert 0 <= node_element("n1", g) < 64

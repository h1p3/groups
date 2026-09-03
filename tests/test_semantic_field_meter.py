import numpy as np

from groupcot.engine.semantic_field_meter import SemanticFieldMeter


def test_adherence_sums_probability_in_field():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    assert SemanticFieldMeter.adherence(probs, {1, 3}) == 0.2 + 0.4


def test_adherence_empty_field_is_zero():
    probs = np.array([0.5, 0.5])
    assert SemanticFieldMeter.adherence(probs, set()) == 0.0


def test_adherence_ignores_out_of_range_ids():
    probs = np.array([0.5, 0.5])
    assert SemanticFieldMeter.adherence(probs, {0, 99}) == 0.5


def test_adherence_full_field_sums_to_one():
    rng = np.random.RandomState(0)
    logits = rng.randn(50)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    assert abs(SemanticFieldMeter.adherence(probs, set(range(50))) - 1.0) < 1e-9


def test_coverage_exact_percentage():
    field = {1, 2, 3, 4}
    masked = {2, 3, 5, 6}
    result = SemanticFieldMeter.coverage(field, masked)
    assert result["field_size"] == 4
    assert result["masked_in_field"] == 2  # {2, 3}
    assert result["coverage_pct"] == 50.0


def test_coverage_full_and_none():
    field = {1, 2, 3}
    assert SemanticFieldMeter.coverage(field, {1, 2, 3, 9})["coverage_pct"] == 100.0
    assert SemanticFieldMeter.coverage(field, {9, 10})["coverage_pct"] == 0.0


def test_coverage_empty_field():
    assert SemanticFieldMeter.coverage(set(), {1, 2}) == {
        "field_size": 0, "masked_in_field": 0, "coverage_pct": 0.0,
    }

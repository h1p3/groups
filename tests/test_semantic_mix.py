import numpy as np
import pytest

from groupcot.engine.llamacpp import LlamaCppEngine


def test_build_mix_probs_uniform_over_ids():
    probs = LlamaCppEngine._build_mix_probs(10, {2, 5, 7}, None)
    assert probs.shape == (10,)
    assert abs(probs.sum() - 1.0) < 1e-9
    for i in (2, 5, 7):
        assert abs(probs[i] - 1.0 / 3) < 1e-9
    for i in set(range(10)) - {2, 5, 7}:
        assert probs[i] == 0.0


def test_build_mix_probs_weighted():
    probs = LlamaCppEngine._build_mix_probs(5, {0, 1}, {0: 3.0, 1: 1.0})
    assert abs(probs[0] - 0.75) < 1e-9
    assert abs(probs[1] - 0.25) < 1e-9
    assert probs[2] == 0.0


def test_build_mix_probs_negative_weights_fall_back_to_uniform():
    # A caller passing garbage (all-zero or negative weights) shouldn't
    # produce a degenerate all-zero mix -- fall back to uniform.
    probs = LlamaCppEngine._build_mix_probs(4, {0, 1}, {0: -1.0, 1: -1.0})
    assert abs(probs[0] - 0.5) < 1e-9
    assert abs(probs[1] - 0.5) < 1e-9


def test_build_mix_probs_empty_ids_is_all_zero():
    probs = LlamaCppEngine._build_mix_probs(5, set(), None)
    assert np.all(probs == 0.0)


def test_build_mix_probs_filters_out_of_range_ids():
    probs = LlamaCppEngine._build_mix_probs(3, {0, 1, 99}, None)
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-9  # only {0, 1} counted


def test_mix_and_sample_alpha_zero_matches_natural_argmax_at_temp_zero():
    logits = np.array([1.0, 5.0, 2.0, 0.0], dtype=np.float32)
    mix_probs = LlamaCppEngine._build_mix_probs(4, {0}, None)  # would pick token 0 alone
    candidate = LlamaCppEngine._mix_and_sample(
        logits, mix_probs, alpha=0.0, temperature=0.0, top_p=1.0, top_k=0)
    assert candidate == 1  # argmax(logits), mix_probs had zero effect


def test_mix_and_sample_alpha_one_ignores_logits_entirely():
    # Logits strongly favor token 1, but alpha=1 means p_final == mix_probs,
    # which only has support on {0} -- must return 0 regardless of logits.
    logits = np.array([-100.0, 100.0, -100.0, -100.0], dtype=np.float32)
    mix_probs = LlamaCppEngine._build_mix_probs(4, {0}, None)
    candidate = LlamaCppEngine._mix_and_sample(
        logits, mix_probs, alpha=1.0, temperature=0.0, top_p=1.0, top_k=0)
    assert candidate == 0


def test_mix_and_sample_partial_alpha_stays_within_support():
    """At alpha=0.5 the candidate must come from natural-top-k union
    mix_probs's support -- specifically, it must never be a token with
    probability 0 in *both* p_natural and p_concept."""
    logits = np.array([5.0, -5.0, -5.0, -5.0], dtype=np.float32)  # token 0 dominant naturally
    mix_probs = LlamaCppEngine._build_mix_probs(4, {2}, None)  # concept wants token 2
    seen = set()
    for _ in range(200):
        c = LlamaCppEngine._mix_and_sample(
            logits, mix_probs, alpha=0.5, temperature=1.0, top_p=1.0, top_k=0)
        seen.add(c)
    assert seen <= {0, 2}  # only tokens with nonzero mass in either component
    assert 0 in seen and 2 in seen  # both should show up over 200 draws


def test_mix_and_sample_does_not_blow_up_on_extreme_logit_scale():
    """The whole point of §12.1: unlike additive attract_weight (broke on
    this model's real logit std~2.9, see ARCHITECTURE.md §10.3), mixing must
    stay well-behaved regardless of how extreme the raw logits are."""
    rng = np.random.RandomState(0)
    logits = (rng.randn(50) * 1000).astype(np.float32)  # deliberately extreme
    mix_probs = LlamaCppEngine._build_mix_probs(50, {1, 2, 3}, None)
    for alpha in (0.0, 0.3, 0.7, 1.0):
        c = LlamaCppEngine._mix_and_sample(
            logits, mix_probs, alpha=alpha, temperature=0.8, top_p=1.0, top_k=0)
        assert 0 <= c < 50


def test_mix_and_sample_clamps_alpha_outside_unit_interval():
    logits = np.array([5.0, -5.0], dtype=np.float32)
    mix_probs = LlamaCppEngine._build_mix_probs(2, {1}, None)
    # alpha > 1 should behave like alpha == 1 (clamped), not overshoot.
    c = LlamaCppEngine._mix_and_sample(
        logits, mix_probs, alpha=5.0, temperature=0.0, top_p=1.0, top_k=0)
    assert c == 1
    # alpha < 0 should behave like alpha == 0.
    c = LlamaCppEngine._mix_and_sample(
        logits, mix_probs, alpha=-5.0, temperature=0.0, top_p=1.0, top_k=0)
    assert c == 0

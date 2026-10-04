# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Tests for the asymmetric fitness function (paper Eq. 5-9)."""

import pytest

from mergekit_cabs.fitness import AsymmetricFitness


def test_penalty_asymmetry():
    """Delta > 0 is multiplied by alpha=100; improvement only by beta=1."""
    f = AsymmetricFitness(alpha=100.0, beta=1.0)
    breakdown = f.compute({"a": 1.1}, {"a": 1.0})
    assert breakdown.deltas["a"] == pytest.approx(0.1)
    assert breakdown.penalties["a"] == pytest.approx(100.0 * 0.1)

    breakdown = f.compute({"a": 0.9}, {"a": 1.0})
    assert breakdown.penalties["a"] == pytest.approx(-0.1)
    assert breakdown.fitness == pytest.approx(-0.1)


def test_fitness_sum_over_tasks():
    f = AsymmetricFitness(alpha=100.0, beta=1.0)
    breakdown = f.compute(
        {"t1": 2.0, "t2": 0.5}, {"t1": 1.0, "t2": 1.0}
    )
    # t1: +1.0 -> alpha * 1.0 ; t2: -0.5 -> beta * (-0.5)
    assert breakdown.fitness == pytest.approx(100.0 - 0.5)
    assert set(breakdown.losses) == {"t1", "t2"}


def test_task_mismatch_raises():
    f = AsymmetricFitness()
    with pytest.raises(ValueError):
        f.compute({"a": 1.0, "b": 1.0}, {"a": 1.0})
    with pytest.raises(ValueError):
        f.compute({"a": 1.0}, {"b": 1.0})


def test_default_coefficients_match_paper():
    f = AsymmetricFitness()
    assert f.alpha == 100.0
    assert f.beta == 1.0


def test_zero_base_loss_guarded():
    """A zero baseline must not produce NaN (eps guard)."""
    f = AsymmetricFitness()
    breakdown = f.compute({"a": 0.5}, {"a": 0.0})
    assert breakdown.penalties["a"] == pytest.approx(100.0 * 0.5 / 1e-12)
    assert breakdown.fitness == breakdown.fitness  # finite

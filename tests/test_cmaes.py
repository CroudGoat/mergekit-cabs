# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Tests for the CMA-ES core of Adaptive Weight Allocation (AWA)."""

import math

import torch

from mergekit_cabs.cmaes import CMAES, CMAESOptions, optimize


def test_paper_defaults():
    """sigma0=0.05, C=I, x0=1, log-weights ln(mu+0.5)-ln(i)."""
    opt = CMAES([1.0, 1.0], CMAESOptions(popsize=6, n_generations=1, seed=0))
    assert opt.sigma == 0.05
    assert torch.equal(opt.C, torch.eye(2, dtype=torch.float64))
    assert torch.equal(opt.m, torch.ones(2, dtype=torch.float64))
    assert opt.mu == 3
    expected_w = torch.tensor(
        [math.log(3.5) - math.log(1), math.log(3.5) - math.log(2), math.log(3.5) - math.log(3)],
        dtype=torch.float64,
    )
    expected_w = expected_w / expected_w.sum()
    assert torch.allclose(opt.w, expected_w)


def test_bounds_projection():
    """Samples are projected onto [l, u] = [0.1, 2] (paper Eq. 3-4)."""
    opt = CMAES([1.0, 1.0], CMAESOptions(popsize=8, n_generations=1, seed=1, sigma0=100.0))
    pop = opt.ask()
    assert pop.min() >= 0.1 - 1e-12
    assert pop.max() <= 2.0 + 1e-12


def test_optimize_convex_quadratic():
    """F(lambda) = sum_t c_t (lambda_t - target_t)^2 recovers the target."""
    target = [1.4, 0.6]
    scale = [10.0, 1.0]

    def fitness(lam):
        return sum(c * (l - t) ** 2 for c, l, t in zip(scale, lam, target))

    result = optimize(
        fitness,
        dim=2,
        options=CMAESOptions(popsize=10, n_generations=60, seed=2),
    )
    assert result.best_fitness < 1e-6
    for got, want in zip(result.best_lambda, target):
        assert abs(got - want) < 1e-2


def test_asymmetric_landscape_favours_no_degradation():
    """With alpha >> beta the search must avoid regions where any term grows."""
    # lambda_1 controls a well, lambda_2 controls a penalty cliff at >0.9.
    def fitness(lam):
        well = (lam[0] - 1.2) ** 2
        cliff = max(0.0, lam[1] - 0.9) ** 2
        return well + 100.0 * cliff  # strongly penalises lambda_2 > 0.9

    result = optimize(
        fitness, dim=2, options=CMAESOptions(popsize=10, n_generations=50, seed=3)
    )
    assert result.best_lambda[1] <= 0.95
    assert abs(result.best_lambda[0] - 1.2) < 0.1


def test_step_size_and_covariance_stay_valid():
    opt = CMAES([1.0] * 3, CMAESOptions(popsize=6, n_generations=20, seed=4))

    def fitness(lam):
        return sum((l - 0.7 * (i + 1)) ** 2 for i, l in enumerate(lam))

    for _ in range(20):
        pop = opt.ask()
        fits = [fitness(x.tolist()) for x in pop]
        opt.tell(pop, fits)
        assert opt.sigma > 0
        evals = torch.linalg.eigvalsh(opt.C)
        assert evals.min() > 0
        assert torch.allclose(opt.C, opt.C.T, atol=1e-12)


def test_history_recorded():
    def fitness(lam):
        return (lam[0] - 1.0) ** 2

    result = optimize(fitness, dim=1, options=CMAESOptions(popsize=4, n_generations=5, seed=5))
    assert len(result.history) == 5
    assert result.history[0]["generation"] == 0
    assert result.n_evaluations == 20

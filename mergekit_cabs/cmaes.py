# mergekit-cabs
# SPDX-License-Identifier: MIT
"""CMA-ES implementation for CABS+ Adaptive Weight Allocation (AWA).

Faithful to arXiv:2608.12842, Sec. III-C and Algorithm 1 (lines 7-21):

- K candidate coefficients sampled per generation:
      lambda_k ~ m + sigma * N(0, C),  projected onto [l, u] (l=0.1, u=2)
- asymmetric fitness  F(lambda) = sum_t f_t(lambda)   (see fitness.py)
- top mu = K/2 individuals selected (minimisation);
- log-weighted mean update  w_i' = ln(mu + 0.5) - ln(i),  normalised;
- step-size adaptation through the conjugate evolution path p_sigma
  (paper Eq. 13-15) and covariance adaptation through p_c and the
  population (paper Eq. 16-17);
- initial state C^(0) = I, sigma^(0) = 0.05, m^(0) = 1 (lambda_0 = 1).

Standard CMA-ES constants (Hansen) are used for c_sigma, d_sigma, c_c,
c_1, c_mu;  E||N(0, I)|| uses the classical approximation.

The implementation is intentionally dependency-free (torch only) because
the number of merged tasks D is tiny (typically 2-10), so eigendecomposing
C every generation is negligible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class CMAESOptions:
    """Knobs of the AWA search (paper defaults where given)."""

    bounds: Tuple[float, float] = (0.1, 2.0)  # paper: l = 0.1, u = 2
    sigma0: float = 0.05  # paper: sigma^(0) = 0.05
    popsize: Optional[int] = None  # K; default 4 + floor(3 ln D) (CMA default)
    n_generations: int = 30  # G
    seed: int = 0
    # Standard CMA-ES constants; exposed for completeness/testing.
    active_cma: bool = False


@dataclass
class CMAESResult:
    """Outcome of an AWA run."""

    best_lambda: List[float]
    best_fitness: float
    mean: List[float]
    sigma: float
    n_evaluations: int
    history: List[dict] = field(default_factory=list)


def _default_popsize(D: int) -> int:
    return 4 + int(math.floor(3 * math.log(D)))


class CMAES:
    """Minimise F(lambda) over box-constrained R^D with (mu, lambda)-CMA-ES."""

    def __init__(self, x0: Sequence[float], options: CMAESOptions):
        self.D = len(x0)
        if self.D < 1:
            raise ValueError("x0 must be non-empty")
        self.options = options
        self.l, self.u = options.bounds
        if not (self.l < self.u):
            raise ValueError("bounds must satisfy l < u")

        gen = torch.Generator().manual_seed(options.seed)
        self._gen = gen

        self.m = torch.tensor(list(x0), dtype=torch.float64)
        self.sigma = float(options.sigma0)
        self.C = torch.eye(self.D, dtype=torch.float64)
        self.p_sigma = torch.zeros(self.D, dtype=torch.float64)
        self.p_c = torch.zeros(self.D, dtype=torch.float64)

        K = options.popsize or _default_popsize(self.D)
        self.K = K
        self.mu = K // 2
        if self.mu < 1:
            raise ValueError("popsize must be >= 2")

        # Paper Eq. 11-12: logarithmically ranked weights.
        w = torch.tensor(
            [math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)],
            dtype=torch.float64,
        )
        self.w = w / w.sum()
        self.mu_eff = 1.0 / (self.w**2).sum().item()

        D_dim = self.D
        self.c_sigma = (self.mu_eff + 2.0) / (D_dim + self.mu_eff + 5.0)
        self.d_sigma = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (D_dim + 1.0)) - 1.0)
            + self.c_sigma
        )
        self.c_c = 4.0 / (D_dim + 4.0)
        self.c_1 = 2.0 / ((D_dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((D_dim + 2.0) ** 2 + self.mu_eff),
        )
        # E||N(0, I)|| for x ~ N(0, I_D)
        self.chi_n = math.sqrt(D_dim) * (
            1.0 - 1.0 / (4.0 * D_dim) + 1.0 / (21.0 * D_dim**2)
        )

        self._best_f = math.inf
        self._best_x: Optional[torch.Tensor] = None
        self._n_evals = 0
        self.history: List[dict] = []

    # ------------------------------------------------------------------ API
    def ask(self) -> torch.Tensor:
        """Sample the next population, projected onto [l, u] (paper Eq. 3-4)."""
        chol = torch.linalg.cholesky(self.C)
        normals = torch.randn(self.K, self.D, dtype=torch.float64, generator=self._gen)
        arz = self.m.unsqueeze(0) + self.sigma * (normals @ chol.T)
        return torch.clamp(arz, self.l, self.u)

    def tell(self, population: torch.Tensor, fitnesses: Sequence[float]) -> None:
        """Update m, sigma, C from evaluated candidates (paper Eq. 10-17)."""
        fitnesses_t = torch.tensor(list(fitnesses), dtype=torch.float64)
        if fitnesses_t.numel() != self.K:
            raise ValueError(f"expected {self.K} fitness values, got {fitnesses_t.numel()}")
        self._n_evals += self.K

        order = torch.argsort(fitnesses_t)  # ascending: minimise
        best_idx = order[0].item()
        if fitnesses_t[best_idx].item() < self._best_f:
            self._best_f = fitnesses_t[best_idx].item()
            self._best_x = population[best_idx].clone()

        top = order[: self.mu]
        selected = population[top]  # (mu, D)

        m_old = self.m.clone()
        sigma_old = self.sigma
        self.m = (self.w.unsqueeze(1) * selected).sum(dim=0)

        y = (selected - m_old.unsqueeze(0)) / sigma_old  # y_i (paper Eq. 17)
        y_w = (self.w.unsqueeze(1) * y).sum(dim=0)

        # C^(-1/2) via eigendecomposition (D is tiny).
        evals, evecs = torch.linalg.eigh(self.C)
        c_inv_half = evecs @ torch.diag(evals.clamp_min(1e-30) ** -0.5) @ evecs.T

        # Step-size path (paper Eq. 13).
        d_vec = (self.m - m_old) / sigma_old
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * (c_inv_half @ d_vec)

        # Step-size update (paper Eq. 15).
        ps_norm = self.p_sigma.norm().item()
        self.sigma = float(
            sigma_old
            * math.exp((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.0))
        )
        self.sigma = max(min(self.sigma, 1e3), 1e-12)

        # Covariance path (paper Eq. 16).
        self.p_c = (1 - self.c_c) * self.p_c + math.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * d_vec

        # Covariance update (paper Eq. 17).
        rank_one = torch.outer(self.p_c, self.p_c)
        rank_mu = (self.w.unsqueeze(1) * y).T @ y
        self.C = (
            (1 - self.c_1 - self.c_mu) * self.C
            + self.c_1 * rank_one
            + self.c_mu * rank_mu
        )
        # Keep symmetric / PSD.
        self.C = 0.5 * (self.C + self.C.T)
        evals_min = torch.linalg.eigvalsh(self.C).min().item()
        if evals_min <= 0:
            self.C += torch.eye(self.D, dtype=torch.float64) * (1e-10 - evals_min)

    @property
    def best(self) -> Tuple[Optional[torch.Tensor], float]:
        return self._best_x, self._best_f

    def result(self) -> CMAESResult:
        mean = self.m.tolist()
        best_x = self._best_x.tolist() if self._best_x is not None else list(mean)
        best_f = self._best_f if self._best_x is not None else math.inf
        return CMAESResult(
            best_lambda=best_x,
            best_fitness=best_f,
            mean=mean,
            sigma=self.sigma,
            n_evaluations=self._n_evals,
            history=self.history,
        )


def optimize(
    fitness_fn,
    dim: int,
    x0: Optional[Sequence[float]] = None,
    options: Optional[CMAESOptions] = None,
    callback=None,
) -> CMAESResult:
    """Run AWA/CMA-ES to minimise ``fitness_fn(lambda: Sequence[float]) -> float``.

    ``callback`` (optional) is called each generation with
    ``(generation, mean, best_fitness)``.
    """
    options = options or CMAESOptions()
    x0 = list(x0) if x0 is not None else [1.0] * dim  # paper: lambda_0 = 1
    if len(x0) != dim:
        raise ValueError("x0 dimension mismatch")
    opt = CMAES(x0, options)
    for g in range(options.n_generations):
        pop = opt.ask()
        fits = [float(fitness_fn(pop[k].tolist())) for k in range(opt.K)]
        opt.tell(pop, fits)
        _, best_f = opt.best
        opt.history.append(
            {
                "generation": g,
                "mean": [round(v, 6) for v in opt.m.tolist()],
                "sigma": opt.sigma,
                "best_fitness": None if math.isinf(best_f) else best_f,
            }
        )
        if callback is not None:
            callback(g, opt.m.tolist(), best_f)
    return opt.result()

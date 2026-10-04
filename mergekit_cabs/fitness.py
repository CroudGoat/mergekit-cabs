# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Asymmetric fitness function of CABS+ AWA (arXiv:2608.12842, Eq. 5-9).

For each task t the loss of a candidate merge L_t(theta(lambda)) is
compared against the baseline loss L_base^(t) evaluated at lambda = 1:

    Delta_t(lambda) = (L_t(lambda) - L_base^(t)) / L_base^(t)
    f_t(lambda)     = alpha * Delta_t   if Delta_t > 0   (performance drop)
                    = beta  * Delta_t   if Delta_t <= 0  (improvement)
    F(lambda)       = sum_t f_t(lambda)

with alpha = 100 and beta = 1 by default.  This penalises degrading any
single task far more strongly than it rewards lowering already-small
losses, which prevents high-loss tasks from dominating the optimisation
(the failure mode of the plain-sum objective, Eq. 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple


@dataclass
class FitnessBreakdown:
    """Per-task details of one fitness evaluation."""

    fitness: float
    losses: Dict[str, float]
    deltas: Dict[str, float]
    penalties: Dict[str, float]


class AsymmetricFitness:
    """Vectorised bookkeeping of the paper's asymmetric penalty."""

    def __init__(self, alpha: float = 100.0, beta: float = 1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def compute(
        self,
        losses: Mapping[str, float],
        base_losses: Mapping[str, float],
        eps: float = 1e-12,
    ) -> FitnessBreakdown:
        if set(losses) != set(base_losses):
            missing = set(base_losses) - set(losses)
            extra = set(losses) - set(base_losses)
            raise ValueError(f"task mismatch: missing={missing}, extra={extra}")
        deltas: Dict[str, float] = {}
        penalties: Dict[str, float] = {}
        total = 0.0
        for task, loss in losses.items():
            base = base_losses[task]
            delta = (loss - base) / max(abs(base), eps)
            deltas[task] = delta
            penalty = self.alpha * delta if delta > 0 else self.beta * delta
            penalties[task] = penalty
            total += penalty
        return FitnessBreakdown(
            fitness=total,
            losses=dict(losses),
            deltas=deltas,
            penalties=penalties,
        )

    def aggregate(self, per_task_penalties: Sequence[float]) -> float:
        return float(sum(per_task_penalties))


def base_losses_from_evaluator(evaluator) -> Dict[str, float]:
    """Evaluate the baseline L_base^(t) at lambda = 1 (paper Eq. 6)."""
    n_tasks = evaluator.num_tasks
    return evaluator.losses([1.0] * n_tasks)

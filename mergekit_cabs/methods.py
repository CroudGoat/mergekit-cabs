# mergekit-cabs
# SPDX-License-Identifier: MIT
"""``cabs`` and ``cabs_plus`` merge methods for mergekit.

Both methods follow the same tensor-level kernel:

    W_final = W_base + sum_i  lambda_i * tau_tilde_i

where ``tau_tilde_i`` are the Conflict-Aware / Balanced-Sparsified task
vectors.  The two variants differ in defaults and intended usage:

``cabs`` (ICML 2025, arXiv:2503.01874)
    Scaling coefficients come from the recipe (the original paper found
    them by grid search).  Overlap-region handling defaults to the plain
    official-CABS behaviour (magnitude fill, no consensus).

``cabs_plus`` (arXiv:2608.12842)
    Coefficients are produced offline by the Adaptive Weight Allocation
    (AWA) CMA-ES search (``mergekit-cabs awa``) and pasted into the
    recipe as ``weight`` values.  Overlap handling defaults to the
    TIES-style sign election described in the paper.

Recipe example::

    models:
      - model: finetuned_a          # listed first = highest CA priority
        parameters:
          weight: 1.21              # lambda_A (from AWA for cabs_plus)
          n: 64
          m: 256
      - model: finetuned_b
        parameters:
          weight: 0.98
          n: 64
          m: 256
    merge_method: cabs_plus
    base_model: mistralai/Mistral-7B-v0.1
    dtype: float16
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Optional

import torch
from typing_extensions import Literal

from mergekit.merge_methods.base import (
    BasePolicy,
    GroupMergeMethod,
    InputContract,
    MergeMethodSpec,
    ParameterScope,
    ParameterSpec,
    TensorGroup,
)

from .sparsify import ConsensusMethod, PruneMethod, conflict_aware_masks

CABS_PAPER_URL = "https://arxiv.org/abs/2503.01874"
CABS_PLUS_PAPER_URL = "https://arxiv.org/abs/2608.12842"


class CabsMergeMethod(GroupMergeMethod):
    """Tensor-level CABS / CABS+ kernel registered under ``variant``."""

    def __init__(
        self,
        variant: Literal["cabs", "cabs_plus"] = "cabs",
        default_consensus: ConsensusMethod = ConsensusMethod.none,
        default_prune_method: PruneMethod = PruneMethod.nm,
    ):
        if variant not in ("cabs", "cabs_plus"):
            raise ValueError(f"Unknown CABS variant {variant!r}")
        self.variant = variant
        self._default_consensus = default_consensus
        self._default_prune_method = default_prune_method

    @cached_property
    def spec(self) -> MergeMethodSpec:
        is_plus = self.variant == "cabs_plus"
        return MergeMethodSpec(
            name=self.variant,
            pretty_name="CABS+" if is_plus else "CABS",
            reference_url=CABS_PLUS_PAPER_URL if is_plus else CABS_PAPER_URL,
            parameters=(
                # lambda_i scaling coefficient (recipe-fixed for `cabs`,
                # AWA-optimised for `cabs_plus`)
                ParameterSpec(
                    "weight",
                    float,
                    ParameterScope.NON_BASE,
                    default=1.0,
                    description="Scaling coefficient lambda_i of this task vector",
                ),
                # retention quota: BS uses n/m; magnitude uses `density`
                ParameterSpec(
                    "n",
                    Optional[int],
                    ParameterScope.NON_BASE,
                    default=64 if not is_plus else 64,
                    description="BS: elements kept per m-block",
                ),
                ParameterSpec(
                    "m",
                    Optional[int],
                    ParameterScope.NON_BASE,
                    default=256,
                    description="BS: block size",
                ),
                ParameterSpec(
                    "density",
                    Optional[float],
                    ParameterScope.NON_BASE,
                    default=None,
                    description="Retention fraction for magnitude pruning",
                ),
                ParameterSpec(
                    "prune_method",
                    str,
                    ParameterScope.NON_BASE,
                    default=self._default_prune_method.value,
                    description="'nm' (Balanced Sparsification) or 'magnitude'",
                ),
                ParameterSpec(
                    "consensus",
                    Optional[str],
                    ParameterScope.SHARED,
                    default=self._default_consensus.value,
                    description="Overlap resolution: 'none' or 'ties' (majority sign)",
                ),
                ParameterSpec(
                    "normalize",
                    bool,
                    ParameterScope.SHARED,
                    default=False,
                    description="Divide mixed delta by per-position claim count",
                ),
            ),
            contract=InputContract(base=BasePolicy.REQUIRED),
        )

    def merge_group(self, group: TensorGroup, /, **parameters: Any) -> torch.Tensor:
        base = group.base.tensor
        entries = group.non_base
        if not entries:
            return base

        if len(entries) == 1:
            # Single task vector: CA degenerates to plain pruning.
            weight = float(parameters["weight"][entries[0].id])
            delta = _prune_single(entries[0].tensor - base, parameters, entries[0].id)
            return (base + weight * delta).to(base.dtype)

        # --- build per-vector settings (validated once, applied per tensor) ---
        methods = []
        for entry in entries:
            eid = entry.id
            method_name = parameters["prune_method"][eid]
            if method_name not in (PruneMethod.nm.value, PruneMethod.magnitude.value):
                raise ValueError(
                    f"prune_method must be 'nm' or 'magnitude', got {method_name!r}"
                )
            n = parameters["n"][eid]
            m = parameters["m"][eid]
            density = parameters["density"][eid]
            if method_name == PruneMethod.nm.value:
                if not n or not m:
                    raise ValueError(
                        "prune_method 'nm' requires integer n and m per model "
                        "(e.g. n: 64, m: 256)"
                    )
                if n <= 0 or m <= 0 or n > m:
                    raise ValueError(f"invalid n:m configuration n={n}, m={m}")
            elif density is None:
                raise ValueError(
                    "prune_method 'magnitude' requires a per-model `density`"
                )
            methods.append(
                (
                    method_name,
                    n or 0,
                    m or 0,
                    float(density) if density is not None else None,
                    float(parameters["weight"][eid]),
                )
            )

        # Use the richest configuration among models for the shared kernel;
        # the kernel itself handles mixed quotas via per-block fill.
        nm_used = any(mth[0] == PruneMethod.nm.value for mth in methods)
        if nm_used:
            n = max((mth[1] for mth in methods if mth[0] == "nm"), default=64)
            m = max((mth[2] for mth in methods if mth[0] == "nm"), default=256)
            prune_method = PruneMethod.nm
            densities = None
        else:
            prune_method = PruneMethod.magnitude
            densities = [mth[3] for mth in methods]

        consensus = parameters["consensus"] or ConsensusMethod.none.value
        if consensus not in (ConsensusMethod.none.value, ConsensusMethod.ties.value):
            raise ValueError(f"consensus must be 'none' or 'ties', got {consensus!r}")

        vectors = [entry.tensor - base for entry in entries]

        if prune_method == PruneMethod.magnitude and len(set(densities)) == 1:
            effective_density = densities
        elif prune_method == PruneMethod.magnitude:
            # Sequential official semantics per vector would require per-vector
            # free-region quotas; approximate the mixed case with the mean and
            # document. (Recipes almost always use homogeneous settings.)
            effective_density = [sum(densities) / len(densities)] * len(densities)
        else:
            effective_density = None

        result = conflict_aware_masks(
            vectors,
            density=effective_density,
            prune_method=PruneMethod(prune_method),
            n=int(n) if nm_used else 64,
            m=int(m) if nm_used else 256,
            consensus=ConsensusMethod(consensus),
        )

        weights = torch.tensor(
            [mth[4] for mth in methods],
            dtype=vectors[0].dtype
            if vectors[0].dtype.is_floating_point
            else torch.float32,
            device=vectors[0].device,
        )
        stacked = torch.stack(result.pruned, dim=0)
        while stacked.dim() > weights.dim():
            weights = weights.unsqueeze(-1)
        mixed = (stacked * weights).sum(dim=0)

        if parameters["normalize"]:
            claim_count = torch.stack(result.masks, dim=0).sum(dim=0)
            claim_count = claim_count.clamp(min=1).to(mixed.dtype)
            mixed = mixed / claim_count

        return (base + mixed).to(base.dtype)


def _prune_single(delta: torch.Tensor, parameters: dict, eid: Any) -> torch.Tensor:
    """Prune one task vector according to its per-model settings."""
    from .sparsify import magnitude_prune_mask, nm_prune_mask

    method_name = parameters["prune_method"][eid]
    if method_name not in (PruneMethod.nm.value, PruneMethod.magnitude.value):
        raise ValueError(
            f"prune_method must be 'nm' or 'magnitude', got {method_name!r}"
        )
    if method_name == PruneMethod.nm.value:
        n, m = parameters["n"][eid], parameters["m"][eid]
        if not n or not m:
            raise ValueError("prune_method 'nm' requires n and m")
        mask = nm_prune_mask(delta, n=int(n), m=int(m))
    else:
        density = parameters["density"][eid]
        if density is None:
            raise ValueError("prune_method 'magnitude' requires `density`")
        mask = magnitude_prune_mask(delta, density=float(density))
    return delta * mask


CABS = CabsMergeMethod(
    variant="cabs",
    default_consensus=ConsensusMethod.none,
    default_prune_method=PruneMethod.nm,
)
CABS_PLUS = CabsMergeMethod(
    variant="cabs_plus",
    default_consensus=ConsensusMethod.ties,
    default_prune_method=PruneMethod.nm,
)

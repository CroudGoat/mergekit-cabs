# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Conflict-Aware (CA) and Balanced (BS) sparsification kernels for task vectors.

Reference implementations:
- CABS  : "CABS: Conflict-Aware and Balanced Sparsification for Enhancing Model
           Merging" (ICML 2025), arXiv:2503.01874, https://github.com/zongzhenyang/CABS
- CABS+ : "CABS+: Efficient and Scalable Model Merging via Conflict-Aware
           Sparsification and Adaptive Weight Allocation", arXiv:2608.12842.

The kernels here generalise the official two-vector code to an arbitrary
number of task vectors while keeping parity with the two-vector code paths
(see ``conflict_aware_masks`` docstring for details).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import torch


class PruneMethod(str, Enum):
    """Per-block structured pruning (BS) or plain magnitude pruning."""

    nm = "nm"  # n:m block pruning == Balanced Sparsification (BS)
    magnitude = "magnitude"  # unstructured magnitude pruning


class ConsensusMethod(str, Enum):
    """Resolution strategy for positions claimed by more than one vector.

    CA guarantees non-overlap only while the per-block retention quotas sum
    to <= 1.  When quotas exceed the free space, overlap is unavoidable and
    CABS+ (Sec. III-A) resolves it with TIES-style sign selection.
    """

    none = "none"
    ties = "ties"


def _work_abs_flat(tensor: torch.Tensor) -> torch.Tensor:
    """abs() in a dtype safe for topk/sort on the current device."""
    w = tensor.abs().reshape(-1)
    if w.device.type == "cpu" and w.dtype in (torch.float16, torch.bfloat16):
        w = w.float()
    return w


def magnitude_prune_mask(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Return a 0/1 mask keeping the ``density`` fraction of largest |values|."""
    numel = tensor.numel()
    k = int(round(density * numel))
    k = max(0, min(numel, k))
    mask = torch.zeros(numel, dtype=torch.uint8, device=tensor.device)
    if k > 0:
        w = _work_abs_flat(tensor)
        topk = torch.argsort(w, descending=True)[:k]
        mask[topk] = 1
    return mask.reshape_as(tensor)


def nm_prune_mask(tensor: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Balanced Sparsification (BS): keep top-``n`` of every ``m``-block.

    The tensor is flattened row-major and split into consecutive blocks of
    ``m`` elements.  Within each block the ``n`` elements with the largest
    absolute value are retained, giving an even spatial distribution of the
    retained weights.  The merged model stays dense; BS is a merging-time
    structuring device, not an inference-acceleration technique.

    Trailing elements beyond the last full block are zeroed, matching the
    official reference implementation (``n_m_block_pruning`` in
    zongzhenyang/CABS).
    """
    flat = tensor.reshape(-1)
    mask = torch.zeros_like(flat, dtype=torch.uint8)
    n_full = (flat.numel() // m) * m
    if n_full > 0:
        blocks = flat[:n_full].view(-1, m).float().abs()
        topk = blocks.topk(n, dim=1).indices
        block_mask = torch.zeros_like(blocks, dtype=torch.uint8).scatter_(1, topk, 1)
        mask[:n_full] = block_mask.reshape(-1)
    # Trailing partial block: zeroed (faithful to the official code).
    return mask.reshape_as(tensor)


def _per_block_quota_fill(
    vector: torch.Tensor,
    free: torch.Tensor,
    n: int,
    m: int,
) -> torch.Tensor:
    """Per-block selection of ``n`` positions, preferring free positions.

    1. Take as many as possible from ``free`` (ranked by |vector|).
    2. Fill the remaining quota from claimed positions (ranked by |vector|).

    With a fully-free block this reduces exactly to BS (top-n in block);
    with an exhausted free region it reproduces the official n >= m//2
    branch (keep all free + magnitude fill from the claimed region).
    """
    flat = vector.reshape(-1)
    free_flat = free.reshape(-1)
    mask = torch.zeros_like(flat, dtype=torch.uint8)
    n_full = (flat.numel() // m) * m
    if n_full > 0:
        w = flat[:n_full].view(-1, m).float().abs()
        f = free_flat[:n_full].view(-1, m)
        n_free = f.sum(dim=1)
        n_from_free = torch.clamp(n_free, max=n)
        n_from_claimed = n - n_from_free

        order = torch.argsort(w, dim=1, descending=True)
        is_free_sorted = torch.gather(f, 1, order)

        free_rank = torch.cumsum(is_free_sorted, dim=1) - is_free_sorted.long()
        claimed_rank = (
            torch.cumsum(1 - is_free_sorted, dim=1) - (1 - is_free_sorted).long()
        )
        take_free = is_free_sorted.bool() & (
            free_rank < n_from_free.unsqueeze(1)
        )
        take_claimed = (~is_free_sorted.bool()) & (
            claimed_rank < n_from_claimed.unsqueeze(1)
        )
        sel = torch.zeros_like(w, dtype=torch.bool)
        sel.scatter_(1, order, take_free | take_claimed)
        mask[:n_full] = sel.reshape(-1).to(mask.dtype)
    return mask.reshape_as(vector)


@dataclass
class SparsifyResult:
    """Pruned task vectors together with their retention masks."""

    pruned: List[torch.Tensor]
    masks: List[torch.Tensor]
    overlap_fraction: float  # fraction of positions claimed by >1 mask
    claimed_fraction: float  # fraction of positions claimed by >=1 mask


def conflict_aware_masks(
    vectors: Sequence[torch.Tensor],
    density: Optional[Sequence[float]] = None,
    prune_method: PruneMethod = PruneMethod.nm,
    n: int = 64,
    m: int = 256,
    consensus: ConsensusMethod = ConsensusMethod.none,
) -> SparsifyResult:
    """Conflict-Aware sequential pruning over k task vectors.

    Implements CA (arXiv:2503.01874 / arXiv:2608.12842, Eq. 1 and
    Algorithm 1 lines 2-5): each vector is pruned *sequentially*, and the
    positions claimed by earlier vectors are excluded from later ones so
    that each task vector retains distinct, non-overlapping parameters.

    For two vectors and either pruning method the result matches the
    official CABS scripts:

    - magnitude: ``tau2_tilde = MP(tau2 * (1 - mask1), s)``  (the second
      vector's density becomes (1-s)^2, exactly like
      ``block_random_pruning`` in the official ``prune_task_vector.py``);
    - n:m with n < m/2: ``tau2_tilde = nm(tau2 * (1 - mask1), n, m)``;
    - n:m with n >= m/2: all free positions are kept and the rest of the
      per-block quota is filled from the claimed region (the official
      ``mask_half`` branch).

    When the retention quotas exceed the available free space (e.g. three
    vectors at n/m = 0.5), the quota is filled from the claimed region as
    described in CABS+ Sec. III-A ("Minimizing overlap under low sparsity");
    with ``consensus=ties`` the multiply-claimed positions are additionally
    filtered by majority-sign selection (TIES-like), as prescribed by the
    CABS+ paper.

    Args:
        vectors: task vectors in pruning-priority order (first = highest
            priority; recipes should list models in the intended order).
        density: retention fraction per vector (magnitude method).
        prune_method: ``nm`` (BS) or ``magnitude``.
        n: per-block retention for BS.
        m: block size for BS.
        consensus: overlap resolution for multiply-claimed positions.

    Returns:
        SparsifyResult with pruned vectors (= vector * mask) and masks.
    """
    if len(vectors) == 0:
        raise ValueError("conflict_aware_masks requires at least one task vector")
    if prune_method == PruneMethod.magnitude and density is None:
        raise ValueError("magnitude pruning requires `density`")
    for i, vec in enumerate(vectors):
        if vec.shape != vectors[0].shape:
            raise ValueError(
                f"Task vector {i} shape {tuple(vec.shape)} does not match "
                f"vector 0 shape {tuple(vectors[0].shape)}"
            )

    if density is None:
        density = [1.0] * len(vectors)

    claimed = torch.zeros_like(vectors[0], dtype=torch.uint8)
    masks: List[torch.Tensor] = []
    pruned: List[torch.Tensor] = []

    for idx, vec in enumerate(vectors):
        free = (claimed == 0).to(torch.uint8)
        if prune_method == PruneMethod.nm:
            mask = _per_block_quota_fill(vec, free, n=n, m=m)
        else:
            # Official sequential magnitude behaviour: rank |vec| over the
            # free region and keep the vector's own quota of it.
            d = float(density[idx])
            w = _work_abs_flat(vec)
            free_flat = free.reshape(-1)
            k = int(round(d * vec.numel()))
            scores = torch.where(
                free_flat > 0, w, torch.full_like(w, -1.0)
            )
            topk = torch.argsort(scores, descending=True)[:k]
            mask = torch.zeros_like(free_flat)
            mask[topk] = 1
            mask = mask.reshape_as(vec)
        masks.append(mask)
        pruned.append(vec * mask)
        claimed = ((claimed + mask) > 0).to(claimed.dtype)

    if consensus == ConsensusMethod.ties and len(vectors) > 1:
        stacked = torch.stack(pruned, dim=0)
        claim_count = torch.stack(masks, dim=0).sum(dim=0)
        overlap = claim_count > 1
        if overlap.any():
            # Majority sign across the vectors claiming the position,
            # weighted by the retained magnitude (TIES-style election).
            signed = torch.where(stacked != 0, torch.sign(stacked), torch.zeros_like(stacked))
            sign_sum = signed.sum(dim=0)
            majority = (sign_sum >= 0).to(stacked.dtype) * 2 - 1
            keep = (torch.sign(stacked) == majority.unsqueeze(0)) | (stacked == 0)
            stacked = torch.where(
                overlap.unsqueeze(0) & keep, stacked, torch.zeros_like(stacked)
            )
            pruned = list(stacked.unbind(dim=0))

    return SparsifyResult(
        pruned=pruned,
        masks=masks,
        overlap_fraction=_overlap_fraction(masks),
        claimed_fraction=(claimed > 0).float().mean().item(),
    )


def _overlap_fraction(masks: Sequence[torch.Tensor]) -> float:
    count = torch.stack(masks, dim=0).sum(dim=0)
    return (count > 1).float().mean().item()

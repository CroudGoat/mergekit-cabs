# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Unit tests for CA/BS sparsification kernels."""

import pytest
import torch

from mergekit_cabs.sparsify import (
    ConsensusMethod,
    PruneMethod,
    conflict_aware_masks,
    magnitude_prune_mask,
    nm_prune_mask,
)


def test_nm_mask_per_block_topn():
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    mask = nm_prune_mask(x, n=64, m=256)
    flat = mask.reshape(-1, 256)
    # exactly n kept per block
    assert torch.all(flat.sum(dim=1) == 64)
    # kept elements are the block's top-n by magnitude
    for row in range(4):
        block = x[row]
        top64 = set(block.abs().topk(64).indices.tolist())
        kept = set((flat[row] == 1).nonzero().flatten().tolist())
        assert top64 == kept


def test_nm_trailing_block_zeroed():
    # 600 elements with m=256 -> 2 full blocks (512) + 88 trailing (zeroed)
    x = torch.randn(600)
    mask = nm_prune_mask(x, n=64, m=256)
    assert mask[:512].sum() == 128
    assert mask[512:].sum() == 0


def test_magnitude_mask_density():
    x = torch.randn(1000)
    mask = magnitude_prune_mask(x, density=0.3)
    assert int(mask.sum()) == 300
    assert set(mask.unique().tolist()) == {0, 1}


def test_ca_two_vectors_nm_low_sparsity_disjoint():
    """n=64, m=256 (n < m/2): the second vector must avoid mask1 positions."""
    torch.manual_seed(1)
    v1, v2 = torch.randn(2, 512), torch.randn(2, 512)
    res = conflict_aware_masks([v1, v2], prune_method=PruneMethod.nm, n=64, m=256)
    m1, m2 = res.masks
    assert torch.all((m1 * m2) == 0), "masks must be disjoint when n < m/2"
    assert torch.all(m1.reshape(-1, 256).sum(dim=1) == 64)
    # second vector keeps n per block, from the free region only
    assert torch.all(m2.reshape(-1, 256).sum(dim=1) == 64)
    assert res.overlap_fraction == 0.0


def test_ca_two_vectors_nm_high_sparsity_matches_official():
    """n=96, m=128 (n >= m/2): keep all free + fill from claimed = n per block.

    Reproduces the official 7B-code branch:
        final_mask2 = clamp((1 - mask1) + mask_half, 0, 1)
    so vector 2 retains (m-n) + m/2 = 32 + 64 = 96 positions per block.
    """
    torch.manual_seed(2)
    n, m = 96, 128
    v1, v2 = torch.randn(3, 128), torch.randn(3, 128)
    res = conflict_aware_masks([v1, v2], prune_method=PruneMethod.nm, n=n, m=m)
    m1, m2 = res.masks
    per_block2 = m2.reshape(-1, m).sum(dim=1)
    assert torch.all(per_block2 == n)
    # official branch keeps ALL free positions (1 - mask1)
    free = (m1 == 0).to(m2.dtype)
    assert torch.all(m2[free == 1] == 1)


def test_ca_two_vectors_magnitude_matches_official():
    """magnitude: tau2_tilde = MP(tau2 * (1 - mask1), s) — official semantics."""
    torch.manual_seed(3)
    s = 0.5
    v1, v2 = torch.randn(1000), torch.randn(1000)
    res = conflict_aware_masks(
        [v1, v2], density=[1 - s, 1 - s], prune_method=PruneMethod.magnitude
    )
    m1, m2 = res.masks
    # reference: official block_random_pruning
    ref_m1 = magnitude_prune_mask(v1, density=1 - s)
    free = (ref_m1 == 0).to(v2.dtype)
    ref_m2 = magnitude_prune_mask(v2 * free, density=1 - s)
    assert torch.equal(m1, ref_m1)
    assert torch.equal(m2, ref_m2)


def test_ca_three_vectors_overlap_fill_and_ties_consensus():
    """Quota sum = 1.5 > 1 -> overlap is unavoidable; ties consensus must
    keep only majority-sign contributions on overlapped positions."""
    torch.manual_seed(4)
    n, m = 128, 256  # density 0.5 each, 3 vectors
    vs = [torch.randn(8, 256) for _ in range(3)]
    res_none = conflict_aware_masks(vs, prune_method=PruneMethod.nm, n=n, m=m,
                                    consensus=ConsensusMethod.none)
    assert res_none.overlap_fraction > 0
    # every block still meets its quota of n
    for mk in res_none.masks:
        assert torch.all(mk.reshape(-1, m).sum(dim=1) == n)

    res_ties = conflict_aware_masks(vs, prune_method=PruneMethod.nm, n=n, m=m,
                                    consensus=ConsensusMethod.ties)
    stacked = torch.stack(res_ties.pruned)
    overlap = torch.stack(res_ties.masks).sum(dim=0) > 1
    # on overlap, no two retained values may disagree in sign
    nz = (stacked != 0).float()
    signs = torch.sign(stacked) * nz
    pos = signs.max(dim=0).values
    neg = signs.min(dim=0).values
    conflict = ((pos > 0) & (neg < 0)) & overlap
    assert not conflict.any(), "ties consensus must eliminate sign conflicts"


def test_pruned_equals_vector_times_mask():
    torch.manual_seed(5)
    vs = [torch.randn(64, 64) for _ in range(2)]
    res = conflict_aware_masks(vs, prune_method=PruneMethod.nm, n=16, m=64)
    for v, p, mk in zip(vs, res.pruned, res.masks):
        assert torch.equal(p, v * mk.to(v.dtype))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        conflict_aware_masks([torch.randn(4, 4), torch.randn(5, 5)])

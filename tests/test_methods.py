# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Tests for the mergekit-integrated cabs / cabs_plus merge methods."""

import pytest
import torch

import mergekit_cabs  # noqa: F401  (registers methods on import)
from mergekit.merge_methods.api import merge_tensors
from mergekit.merge_methods.registry import get, registered_methods


def test_registration():
    names = [m.spec.name for m in registered_methods()]
    assert "cabs" in names
    assert "cabs_plus" in names


def _merge(base, deltas, weights=None, method="cabs", **params):
    tensors = [base] + [base + d for d in deltas]
    per_input = {}
    if weights is not None:
        per_input["weight"] = weights
    for key, value in params.items():
        per_input[key] = value
    return merge_tensors(
        tensors,
        method=method,
        base_index=0,
        parameters=per_input or None,
    )


def test_cabs_single_vector_task_arithmetic_equivalence():
    """One vector + weight => base + w * nm(tau)."""
    torch.manual_seed(10)
    base = torch.randn(32, 64)
    delta = torch.randn(32, 64)
    out = _merge(base, [delta], weights=[1.2], n=16, m=64)
    from mergekit_cabs.sparsify import nm_prune_mask

    expected = base + 1.2 * (delta * nm_prune_mask(delta, 16, 64))
    assert torch.allclose(out, expected, atol=1e-6)


def test_cabs_two_vectors_no_overlap_sum():
    torch.manual_seed(11)
    base = torch.randn(64, 256)
    d1, d2 = torch.randn(64, 256), torch.randn(64, 256)
    out = _merge(base, [d1, d2], weights=[1.0, 1.0], n=64, m=256)
    from mergekit_cabs.sparsify import conflict_aware_masks, PruneMethod

    res = conflict_aware_masks([d1, d2], prune_method=PruneMethod.nm, n=64, m=256)
    expected = base + res.pruned[0] + res.pruned[1]
    assert torch.allclose(out, expected, atol=1e-6)
    # CA guarantee: disjoint support
    assert torch.all((res.masks[0] * res.masks[1]) == 0)


def test_cabs_weights_affect_output():
    torch.manual_seed(12)
    base = torch.randn(32, 256)
    d1, d2 = torch.randn(32, 256), torch.randn(32, 256)
    out_w = _merge(base, [d1, d2], weights=[1.5, 0.5], n=32, m=256)
    out_1 = _merge(base, [d1, d2], weights=[1.0, 1.0], n=32, m=256)
    assert not torch.allclose(out_w, out_1)


def test_cabs_plus_default_consensus_ties():
    """cabs_plus defaults to TIES-style majority-sign overlap handling."""
    from mergekit_cabs.methods import CABS, CABS_PLUS

    assert CABS_PLUS.spec.parameters and any(
        p.name == "consensus" and p.default == "ties" for p in CABS_PLUS.spec.parameters
    )
    assert any(p.name == "consensus" and p.default == "none" for p in CABS.spec.parameters)


def test_cabs_single_vector_all_methods():
    torch.manual_seed(13)
    base = torch.randn(32, 64)
    delta = torch.randn(32, 64)
    out_mag = _merge(base, [delta], prune_method="magnitude", density=[0.5])
    # magnitude keeps exactly half of the elements
    kept = (out_mag - base) != 0
    assert kept.float().mean().item() == pytest.approx(0.5, abs=1e-6)
    # nm keeps n/m of the elements
    out_nm = _merge(base, [delta], n=16, m=64, prune_method="nm")
    kept_nm = (out_nm - base) != 0
    assert kept_nm.float().mean().item() == pytest.approx(16 / 64, abs=1e-6)


def test_method_rejects_bad_prune_method():
    torch.manual_seed(14)
    base = torch.randn(8, 8)
    delta = torch.randn(8, 8)
    with pytest.raises(ValueError):
        _merge(base, [delta], prune_method="gibberish", density=[0.5])


def test_cabs_normalize_option():
    """normalize divides the mixed delta by the per-position claim count."""
    torch.manual_seed(15)
    base = torch.randn(64, 256)
    d1, d2 = torch.randn(64, 256), torch.randn(64, 256)
    out = _merge(base, [d1, d2], n=64, m=256, normalize=True)
    from mergekit_cabs.sparsify import conflict_aware_masks, PruneMethod

    res = conflict_aware_masks([d1, d2], prune_method=PruneMethod.nm, n=64, m=256)
    stacked = torch.stack(res.pruned)
    counts = torch.stack(res.masks).sum(dim=0).clamp(min=1)
    expected = base + (stacked.sum(dim=0) / counts)
    assert torch.allclose(out, expected, atol=1e-6)

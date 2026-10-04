# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Task vector extraction / application utilities.

Follows the official CABS scripts (``extract_task_vector_7b.py``,
``apply_task_vectors``): the task vector of a finetuned model w.r.t. a
base model is the parameter-wise difference

    tau = W_finetuned - W_base

restricted to floating-point parameters that exist in both models.
Transformers / safetensors are imported lazily so that the pure-tensor
merge methods work without them installed.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for model-loading utilities "
            "(pip install transformers)"
        ) from exc
    return transformers


def load_model(path: str, dtype: torch.dtype = torch.float16, device: str = "cpu"):
    """Load a causal LM (or classifier) with dtype fallbacks across versions."""
    transformers = _import_transformers()
    kwargs = {"device_map": None}
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, **kwargs
        )
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, **kwargs
        )
    return model.to(device)


def float_param_state_dict(model) -> Dict[str, torch.Tensor]:
    """state_dict entries that participate in merging (floating point only)."""
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if tensor.dtype.is_floating_point
    }


def extract_task_vectors(
    base_model_path: str,
    finetuned_model_paths: Sequence[str],
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (base_state_dict, {model_label: task_vector_state_dict}).

    Labels are the basenames of the finetuned paths.  Keys are aligned to
    the intersection of floating-point keys across all models; buffers such
    as ``inv_freq`` are skipped automatically.
    """
    base_model = load_model(base_model_path, dtype=dtype, device=device)
    base_sd = float_param_state_dict(base_model)
    base_keys = set(base_sd.keys())

    vectors: Dict[str, Dict[str, torch.Tensor]] = {}
    common_keys: Optional[set] = None
    for path in finetuned_model_paths:
        model = load_model(path, dtype=dtype, device=device)
        sd = float_param_state_dict(model)
        keys = set(sd.keys()) & base_keys
        common_keys = keys if common_keys is None else (common_keys & keys)
        label = os.path.basename(os.path.normpath(path))
        vectors[label] = sd
        del model
    if common_keys is None:
        raise ValueError("no finetuned models given")

    base_out: Dict[str, torch.Tensor] = {}
    task_vectors: Dict[str, Dict[str, torch.Tensor]] = {
        label: {} for label in vectors
    }
    for key in sorted(common_keys):
        base_t = base_sd[key]
        base_out[key] = base_t.clone()
        for label, sd in vectors.items():
            task_vectors[label][key] = sd[key] - base_t
    del base_model
    return base_out, task_vectors


def apply_task_vectors(
    base_state_dict: Mapping[str, torch.Tensor],
    pruned_task_vectors: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """W_final = W_base + sum_i lambda_i * tau_tilde_i (Algorithm 1, line 21)."""
    if len(pruned_task_vectors) != len(weights):
        raise ValueError("one weight per task vector required")
    merged: Dict[str, torch.Tensor] = {}
    for key, base_t in base_state_dict.items():
        out = base_t.to(dtype).clone()
        for tv, w in zip(pruned_task_vectors, weights):
            if key in tv:
                out += tv[key].to(dtype) * float(w)
        merged[key] = out
    return merged


def save_state_dict_safetensors(
    state_dict: Mapping[str, torch.Tensor], out_dir: str, max_shard_size: int = 4
) -> None:
    """Save tensors as sharded safetensors + index json (HF layout)."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(state_dict.keys())
    single: Dict[str, torch.Tensor] = {}
    total = sum(state_dict[k].numel() * state_dict[k].element_size() for k in keys)
    limit = max_shard_size * 1024**3
    if total <= limit:
        single = {k: state_dict[k].contiguous() for k in keys}
        save_file(single, os.path.join(out_dir, "model.safetensors"))
        return
    shards: List[Dict[str, torch.Tensor]] = [{}]
    current = 0
    index: Dict[str, str] = {}
    for k in keys:
        t = state_dict[k].contiguous()
        size = t.numel() * t.element_size()
        if current + size > limit and shards[-1]:
            shards.append({})
            current = 0
        shards[-1][k] = t
        current += size
    weight_map = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, fname))
        for k in shard:
            index[k] = fname
            weight_map[k] = fname
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def load_state_dict_safetensors(directory: str) -> Dict[str, torch.Tensor]:
    """Load a (sharded) safetensors directory written by save_state_dict_safetensors."""
    from safetensors.torch import load_file

    index_path = os.path.join(directory, "model.safetensors.index.json")
    tensors: Dict[str, torch.Tensor] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for fname in sorted(set(index["weight_map"].values())):
            tensors.update(load_file(os.path.join(directory, fname)))
    else:
        tensors.update(load_file(os.path.join(directory, "model.safetensors")))
    return tensors

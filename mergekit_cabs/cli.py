# mergekit-cabs
# SPDX-License-Identifier: MIT
"""Command line interface for mergekit-cabs.

Subcommands
-----------
extract     Extract task vectors (tau = W_finetuned - W_base) to safetensors.
prune       Conflict-Aware + Balanced sparsification of task vectors.
merge       Apply pruned task vectors to a base model with fixed weights.
awa         Full CABS+ Adaptive Weight Allocation search (CMA-ES).
yaml        mergekit-yaml with cabs / cabs_plus registered.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", required=True, help="Base model path or HF id")
    parser.add_argument(
        "--models", nargs="+", required=True, dest="finetuned_models",
        help="Finetuned model paths, in CA pruning-priority order",
    )


def _add_prune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prune-method", default="nm", choices=["nm", "magnitude"],
        help="'nm' = Balanced Sparsification (BS), 'magnitude' = unstructured",
    )
    parser.add_argument("--n", type=int, default=64, help="BS: keep n per block")
    parser.add_argument("--m", type=int, default=256, help="BS: block size")
    parser.add_argument(
        "--density", type=float, nargs="+", default=None,
        help="Retention fraction(s) for magnitude pruning (per model)",
    )
    parser.add_argument(
        "--consensus", default="none", choices=["none", "ties"],
        help="Overlap resolution (ties = CABS+ majority-sign election)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mergekit-cabs", description="CABS / CABS+ merging for mergekit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract task vectors")
    _add_model_args(p_extract)
    p_extract.add_argument("--output-dir", required=True)
    p_extract.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p_extract.add_argument("--device", default="cpu")

    p_prune = sub.add_parser("prune", help="CA+BS sparsify pre-extracted task vectors")
    p_prune.add_argument("--task-vectors", nargs="+", required=True,
                         help="Directories of extracted task vectors (priority order)")
    p_prune.add_argument("--output-dir", required=True)
    _add_prune_args(p_prune)

    p_merge = sub.add_parser("merge", help="Merge pruned task vectors into the base model")
    p_merge.add_argument("--base-model", required=True)
    p_merge.add_argument("--task-vectors", nargs="+", required=True)
    p_merge.add_argument("--weights", type=float, nargs="+", required=True)
    p_merge.add_argument("--output-dir", required=True)
    p_merge.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    p_awa = sub.add_parser("awa", help="CABS+ AWA coefficient search (CMA-ES)")
    p_awa.add_argument("--config", required=True, help="AWA config JSON (see recipes/)")

    p_yaml = sub.add_parser(
        "yaml",
        help="mergekit-yaml with cabs/cabs_plus registered "
             "(extra flags are forwarded to mergekit-yaml)",
    )
    p_yaml.add_argument("recipe")
    p_yaml.add_argument("output")
    p_yaml.add_argument("extra", nargs=argparse.REMAINDER, default=[],
                        metavar="...",
                        help="forwarded to mergekit-yaml (e.g. --cuda --lazy-unpickle)")

    return parser


def cmd_extract(args) -> int:
    from .sparsify import PruneMethod  # noqa: F401  (import check)
    from .task_vector import extract_task_vectors, save_state_dict_safetensors

    dtype = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}[args.dtype]
    import torch

    torch_dtype = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32
    }[dtype]
    base_sd, task_vectors = extract_task_vectors(
        args.base_model, args.finetuned_models, dtype=torch_dtype, device=args.device
    )
    save_state_dict_safetensors(base_sd, f"{args.output_dir}/base")
    for label, tv in task_vectors.items():
        safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        save_state_dict_safetensors(tv, f"{args.output_dir}/tv_{safe_label}")
        print(f"[cabs] extracted task vector: {label} -> tv_{safe_label}")
    return 0


def cmd_prune(args) -> int:
    import torch

    from .sparsify import ConsensusMethod, PruneMethod, conflict_aware_masks
    from .task_vector import load_state_dict_safetensors, save_state_dict_safetensors

    tvs = [load_state_dict_safetensors(d) for d in args.task_vectors]
    keys = set(tvs[0].keys())
    for tv in tvs[1:]:
        keys &= set(tv.keys())
    keys = sorted(keys)

    pruned = [{} for _ in tvs]
    n_tensors = 0
    for key in keys:
        res = conflict_aware_masks(
            [tv[key] for tv in tvs],
            density=args.density,
            prune_method=PruneMethod(args.prune_method),
            n=args.n,
            m=args.m,
            consensus=ConsensusMethod(args.consensus),
        )
        for i, p in enumerate(res.pruned):
            pruned[i][key] = p
        n_tensors += 1
    for i, (src, out) in enumerate(zip(args.task_vectors, pruned)):
        name = src.rstrip("/").replace("/", "_")
        save_state_dict_safetensors(out, f"{args.output_dir}/pruned_{i}")
        print(f"[cabs] pruned vector {i} ({src}) -> {args.output_dir}/pruned_{i}")
    print(f"[cabs] pruned {n_tensors} tensors")
    return 0


def cmd_merge(args) -> int:
    import torch

    from .task_vector import (
        apply_task_vectors,
        load_state_dict_safetensors,
        load_model,
        save_state_dict_safetensors,
    )

    torch_dtype = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32
    }[args.dtype]
    tvs = [load_state_dict_safetensors(d) for d in args.task_vectors]
    if len(tvs) != len(args.weights):
        raise SystemExit("number of --weights must match number of --task-vectors")
    base_model = load_model(args.base_model, dtype=torch_dtype, device="cpu")
    base_sd = {k: v for k, v in base_model.state_dict().items() if v.dtype.is_floating_point}
    merged = apply_task_vectors(base_sd, tvs, args.weights, dtype=torch_dtype)
    save_state_dict_safetensors(merged, args.output_dir)
    try:
        base_model.config.save_pretrained(args.output_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[cabs] warning: config save failed: {exc}")
    print(f"[cabs] merged model saved to {args.output_dir}")
    return 0


def cmd_awa(args) -> int:
    from .awa_search import AWAConfig, run_awa

    config = AWAConfig.from_json(args.config)
    summary = run_awa(config)
    print(
        f"[cabs+] lambda* = {summary['lambda_star']}  "
        f"best F = {summary['best_fitness']:.6f}"
    )
    return 0


def cmd_yaml(args) -> int:
    # Importing the package registers cabs / cabs_plus.
    import click

    import mergekit_cabs  # noqa: F401  (registers the merge methods)
    from mergekit.scripts.run_yaml import main as run_yaml_main

    forwarded = [args.recipe, args.output] + list(args.extra)
    try:
        run_yaml_main.main(args=forwarded, standalone_mode=False)
    except click.exceptions.Abort:  # pragma: no cover - Ctrl-C
        print("[cabs] aborted", file=sys.stderr)
        return 130
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "extract": cmd_extract,
        "prune": cmd_prune,
        "merge": cmd_merge,
        "awa": cmd_awa,
        "yaml": cmd_yaml,
    }
    return handlers[args.command](args)


def _yaml_entry() -> int:
    """Console entry point: mergekit-cabs-yaml RECIPE OUTPUT [mergekit flags...]."""
    return main(["yaml"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

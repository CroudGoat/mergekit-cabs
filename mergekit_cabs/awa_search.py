# mergekit-cabs
# SPDX-License-Identifier: MIT
"""End-to-end CABS+ AWA driver (arXiv:2608.12842, Algorithm 1).

Pipeline:
  1. extract task vectors tau_i = W_i - W_base
  2. Phase 1 of Algorithm 1: CA + BS sparsification -> tau_tilde_i
  3. Phase 2 (AWA): CMA-ES search over lambda with the asymmetric fitness
     computed from per-task losses on small calibration sets
  4. emit the optimal coefficients lambda*, a ready-to-run mergekit YAML
     recipe (merge_method: cabs_plus), and optionally the merged model.

Memory strategy: the live model stays on the accelerator; the base weights
and the (sparse) pruned task vectors live on the CPU and are streamed
per-tensor into the model for every candidate, so peak GPU memory stays at
inference level (the paper's headline property vs. AdaMerging).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .cmaes import CMAESOptions, optimize
from .fitness import AsymmetricFitness
from .sparsify import ConsensusMethod, PruneMethod, conflict_aware_masks
from .task_vector import (
    extract_task_vectors,
    load_model,
    save_state_dict_safetensors,
)


@dataclass
class AWAConfig:
    base_model: str
    models: Sequence[str]
    tasks: Sequence[dict]  # [{name, data, max_examples, max_len}]
    pruning_method: str = "nm"
    n: int = 64
    m: int = 256
    consensus: str = "ties"
    popsize: Optional[int] = None
    n_generations: int = 30
    bounds: Sequence[float] = (0.1, 2.0)
    sigma0: float = 0.05
    alpha: float = 100.0
    beta: float = 1.0
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 2
    max_examples: int = 48
    max_len: int = 1024
    output_dir: str = "cabs_plus_output"
    save_merged: bool = False

    @classmethod
    def from_json(cls, path: str) -> "AWAConfig":
        with open(path) as f:
            raw = json.load(f)
        awa = raw.get("awa", {})
        pruning = raw.get("pruning", {})
        return cls(
            base_model=raw["base_model"],
            models=raw["models"],
            tasks=raw["tasks"],
            pruning_method=pruning.get("method", "nm"),
            n=int(pruning.get("n", 64)),
            m=int(pruning.get("m", 256)),
            consensus=pruning.get("consensus", "ties"),
            popsize=awa.get("popsize"),
            n_generations=int(awa.get("n_generations", 30)),
            bounds=awa.get("bounds", [0.1, 2.0]),
            sigma0=float(awa.get("sigma0", 0.05)),
            alpha=float(awa.get("alpha", 100.0)),
            beta=float(awa.get("beta", 1.0)),
            seed=int(awa.get("seed", 0)),
            device=raw.get("device", "cuda"),
            dtype=raw.get("dtype", "float16"),
            batch_size=int(raw.get("batch_size", 2)),
            max_examples=int(raw.get("max_examples", 48)),
            max_len=int(raw.get("max_len", 1024)),
            output_dir=raw.get("output_dir", "cabs_plus_output"),
            save_merged=bool(raw.get("save_merged", False)),
        )


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_calibration_batches(
    task_cfgs: Sequence[dict],
    tokenizer,
    batch_size: int,
    default_max_examples: int,
    default_max_len: int,
    device: str,
) -> Dict[str, List[dict]]:
    """Tokenise per-task JSONL calibration data into eval batches."""
    batches: Dict[str, List[dict]] = {}
    for task in task_cfgs:
        name = task.get("name") or os.path.basename(task["data"])
        max_examples = int(task.get("max_examples", default_max_examples))
        max_len = int(task.get("max_len", default_max_len))
        texts: List[str] = []
        with open(task["data"]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj.get("text") or obj.get("prompt", ""))
                if len(texts) >= max_examples:
                    break
        if not texts:
            raise ValueError(f"task {name!r}: no calibration examples found")
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        )
        input_ids_all = enc["input_ids"]
        attn_all = enc["attention_mask"]
        task_batches = []
        for i in range(0, len(texts), batch_size):
            window = input_ids_all[i : i + batch_size]
            window_mask = attn_all[i : i + batch_size]
            width = max(len(x) for x in window)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            ids = torch.full((len(window), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(window), width), dtype=torch.long)
            for j, (seq, m) in enumerate(zip(window, window_mask)):
                ids[j, : len(seq)] = torch.tensor(seq)
                mask[j, : len(m)] = torch.tensor(m)
            task_batches.append(
                {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
            )
        batches[name] = task_batches
    return batches


class CandidateEvaluator:
    """Applies lambda to the live model and measures per-task NLL losses."""

    def __init__(
        self,
        model,
        base_params: Dict[str, torch.Tensor],
        pruned_vectors: List[Dict[str, torch.Tensor]],
        task_batches: Dict[str, List[dict]],
    ):
        self.model = model
        self.base_params = base_params
        self.pruned_vectors = pruned_vectors
        self.task_batches = task_batches
        self._keys = None

    @property
    def num_tasks(self) -> int:
        return len(self.task_batches)

    def apply(self, lambdas: Sequence[float]) -> None:
        if len(lambdas) != len(self.pruned_vectors):
            raise ValueError("need one lambda per task vector")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                base_t = self.base_params.get(name)
                if base_t is None or name not in self.pruned_vectors[0]:
                    continue
                param.copy_(base_t.to(param.dtype))
                for lam, tv in zip(lambdas, self.pruned_vectors):
                    delta = tv.get(name)
                    if delta is not None:
                        param.add_(delta.to(param.dtype), alpha=float(lam))

    @torch.no_grad()
    def losses(self, lambdas: Sequence[float]) -> Dict[str, float]:
        self.apply(lambdas)
        self.model.eval()
        out: Dict[str, float] = {}
        for task_name, batches in self.task_batches.items():
            total_nll = 0.0
            total_tokens = 0
            for batch in batches:
                input_ids = batch["input_ids"]
                attn = batch["attention_mask"]
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                label_mask = attn[:, 1:].to(torch.bool)
                nll = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).view(shift_labels.shape)
                sel = label_mask & (shift_labels != -100)
                total_nll += nll[sel].sum().item()
                total_tokens += int(sel.sum().item())
            out[task_name] = total_nll / max(1, total_tokens)
        return out


def run_awa(config: AWAConfig, log=print) -> dict:
    """Execute the full CABS+ AWA search; returns the result dictionary."""
    if config.device == "cuda" and not torch.cuda.is_available():  # pragma: no cover
        log("[cabs+] cuda unavailable, falling back to cpu")
        config.device = "cpu"
    dtype = _DTYPE_MAP[config.dtype]

    t0 = time.time()
    log(f"[cabs+] extracting task vectors for {len(config.models)} model(s) ...")
    base_sd, task_vectors = extract_task_vectors(
        config.base_model, config.models, dtype=dtype, device="cpu"
    )

    log("[cabs+] Phase 1: Conflict-Aware + Balanced sparsification ...")
    pruned_vectors: List[Dict[str, torch.Tensor]] = []
    keys = sorted(base_sd.keys())
    per_tensor = [{} for _ in task_vectors]
    common = set(keys)
    for tv in task_vectors:
        common &= set(tv.keys())
    overlap_stats = []
    n_done = 0
    for key in sorted(common):
        vecs = [tv[key] for tv in task_vectors]
        res = conflict_aware_masks(
            vecs,
            prune_method=PruneMethod(config.pruning_method),
            n=config.n,
            m=config.m,
            consensus=ConsensusMethod(config.consensus),
        )
        for i, p in enumerate(res.pruned):
            per_tensor[i][key] = p
        overlap_stats.append(res.overlap_fraction)
        n_done += 1
        if n_done % 50 == 0:
            log(f"[cabs+]   pruned {n_done}/{len(common)} tensors")
    pruned_vectors = per_tensor
    mean_overlap = sum(overlap_stats) / max(1, len(overlap_stats))
    log(f"[cabs+] Phase 1 done (mean overlap fraction: {mean_overlap:.4f})")

    log("[cabs+] loading base model + tokenizer ...")
    from transformers import AutoTokenizer

    model = load_model(config.base_model, dtype=dtype, device=config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_params = {k: v for k, v in base_sd.items()}
    task_batches = _load_calibration_batches(
        config.tasks,
        tokenizer,
        config.batch_size,
        config.max_examples,
        config.max_len,
        config.device,
    )
    evaluator = CandidateEvaluator(model, base_params, pruned_vectors, task_batches)

    log("[cabs+] Phase 2: AWA - computing baseline losses at lambda = 1 ...")
    base_losses = evaluator.losses([1.0] * len(config.models))
    log(f"[cabs+] baseline losses: {base_losses}")

    fitness = AsymmetricFitness(alpha=config.alpha, beta=config.beta)

    def score(lambdas: Sequence[float]) -> float:
        losses = evaluator.losses(lambdas)
        breakdown = fitness.compute(losses, base_losses)
        return breakdown.fitness

    options = CMAESOptions(
        bounds=(float(config.bounds[0]), float(config.bounds[1])),
        sigma0=config.sigma0,
        popsize=config.popsize,
        n_generations=config.n_generations,
        seed=config.seed,
    )

    def on_gen(g, mean, best):
        log(
            f"[cabs+] gen {g:03d}  best F={best:.6f}  "
            f"mean={[round(v, 4) for v in mean]}"
        )

    log(f"[cabs+] running CMA-ES (K={options.popsize or 'auto'}, G={options.n_generations}) ...")
    result = optimize(
        score,
        dim=len(config.models),
        options=options,
        callback=on_gen,
    )
    elapsed = time.time() - t0
    log(f"[cabs+] optimal lambda* = {result.best_lambda}  (F = {result.best_fitness:.6f})")
    log(f"[cabs+] total time: {elapsed / 60:.1f} min, evals: {result.n_evaluations}")

    final_losses = evaluator.losses(result.best_lambda)

    os.makedirs(config.output_dir, exist_ok=True)
    summary = {
        "base_model": config.base_model,
        "models": list(config.models),
        "lambda_star": result.best_lambda,
        "best_fitness": result.best_fitness,
        "base_losses": base_losses,
        "final_losses": final_losses,
        "pruning": {
            "method": config.pruning_method,
            "n": config.n,
            "m": config.m,
            "consensus": config.consensus,
            "mean_overlap_fraction": mean_overlap,
        },
        "awa": {
            "popsize": options.popsize,
            "n_generations": options.n_generations,
            "bounds": list(options.bounds),
            "sigma0": options.sigma0,
            "alpha": config.alpha,
            "beta": config.beta,
            "seed": config.seed,
            "n_evaluations": result.n_evaluations,
            "elapsed_seconds": elapsed,
        },
        "history": result.history,
    }
    with open(os.path.join(config.output_dir, "awa_result.json"), "w") as f:
        json.dump(summary, f, indent=2)

    recipe = render_recipe(
        base_model=config.base_model,
        models=list(config.models),
        weights=result.best_lambda,
        method="cabs_plus",
        n=config.n,
        m=config.m,
        consensus=config.consensus,
    )
    recipe_path = os.path.join(config.output_dir, "cabs_plus_recipe.yml")
    with open(recipe_path, "w") as f:
        f.write(recipe)
    log(f"[cabs+] wrote {recipe_path} and awa_result.json")

    if config.save_merged:
        log("[cabs+] merging final model with lambda* ...")
        merged = {}
        for key in sorted(common):
            out = base_sd[key].to(dtype).clone()
            for tv, lam in zip(pruned_vectors, result.best_lambda):
                out += tv[key].to(dtype) * float(lam)
            merged[key] = out
        merged_dir = os.path.join(config.output_dir, "merged_model")
        save_state_dict_safetensors(merged, merged_dir)
        # copy config/tokenizer files for a loadable directory
        from transformers import AutoConfig

        try:
            AutoConfig.from_pretrained(config.base_model).save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
        except Exception as exc:  # pragma: no cover
            log(f"[cabs+] warning: could not save config/tokenizer ({exc})")
        log(f"[cabs+] merged model saved to {merged_dir}")

    return summary


def render_recipe(
    base_model: str,
    models: Sequence[str],
    weights: Sequence[float],
    method: str = "cabs_plus",
    n: int = 64,
    m: int = 256,
    consensus: str = "ties",
) -> str:
    """Render a mergekit YAML recipe with the AWA-optimal coefficients."""
    lines = ["models:"]
    for model, weight in zip(models, weights):
        lines.append(f"  - model: {model}")
        lines.append("    parameters:")
        lines.append(f"      weight: {weight:.6f}")
        lines.append(f"      n: {n}")
        lines.append(f"      m: {m}")
    lines.append(f"merge_method: {method}")
    lines.append(f"base_model: {base_model}")
    lines.append("dtype: float16")
    if consensus and method == "cabs_plus":
        lines.append("parameters:")
        lines.append(f"  consensus: {consensus}")
    return "\n".join(lines) + "\n"

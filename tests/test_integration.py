# mergekit-cabs
# SPDX-License-Identifier: MIT
"""End-to-end integration test: real mergekit merge run with cabs / cabs_plus.

Requires transformers; skipped automatically when unavailable.
Builds three tiny random Llama models on disk, then runs mergekit's own
merge machinery with the registered ``cabs`` and ``cabs_plus`` methods.
"""

import shutil

import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

import mergekit_cabs  # noqa: E402,F401  (registers methods)
from mergekit.config import MergeConfiguration  # noqa: E402
from mergekit.merge import run_merge  # noqa: E402
from mergekit.options import MergeOptions  # noqa: E402


def _make_tiny_models(tmp_path, hidden=64, layers=1, vocab=128):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "llama",
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
        max_position_embeddings=128,
    )
    paths = {}
    for name, seed in (("base", 0), ("ft_a", 1), ("ft_b", 2)):
        path = tmp_path / name
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(cfg).to(torch.float16)
        model.save_pretrained(str(path))
        paths[name] = str(path)
    return paths


def _run_recipe(tmp_path, recipe: dict, out_name: str) -> str:
    cfg = MergeConfiguration.model_validate(recipe)
    out = str(tmp_path / out_name)
    run_merge(cfg, out, options=MergeOptions())
    return out


def test_end_to_end_cabs_and_cabs_plus(tmp_path):
    paths = _make_tiny_models(tmp_path)

    for method, shared in (("cabs", {}), ("cabs_plus", {"consensus": "ties"})):
        recipe = {
            "models": [
                {
                    "model": paths["ft_a"],
                    "parameters": {"weight": 1.15, "n": 16, "m": 64},
                },
                {
                    "model": paths["ft_b"],
                    "parameters": {"weight": 0.95, "n": 16, "m": 64},
                },
            ],
            "merge_method": method,
            "base_model": paths["base"],
            "dtype": "float16",
            "parameters": shared or None,
        }
        out = _run_recipe(tmp_path, recipe, f"merged_{method}")

        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(paths["base"]).state_dict()
        merged = AutoModelForCausalLM.from_pretrained(out).state_dict()

        # merged model must differ from base but share its shapes
        diff = sum(
            not torch.equal(base[k], merged[k])
            for k in base
            if base[k].dtype.is_floating_point
        )
        assert diff > 0, f"{method}: merged model identical to base"

        # single-model degenerate case: weight * nm(task vector)
        single = {
            "models": [
                {"model": paths["ft_a"], "parameters": {"weight": 1.0, "n": 16, "m": 64}},
            ],
            "merge_method": method,
            "base_model": paths["base"],
            "dtype": "float16",
        }
        out1 = _run_recipe(tmp_path, single, f"merged_{method}_single")
        merged1 = AutoModelForCausalLM.from_pretrained(out1).state_dict()
        diff1 = sum(
            not torch.equal(base[k], merged1[k])
            for k in base
            if base[k].dtype.is_floating_point
        )
        assert diff1 > 0, f"{method}: single-model merge identical to base"


def test_recipe_yaml_files_parse_and_register(tmp_path):
    """The shipped recipes parse against mergekit's config model."""
    import mergekit_cabs  # noqa: F401
    from mergekit.config import MergeConfiguration
    from mergekit.merge_methods.registry import get

    for fname in ("cabs_2model.yml", "cabs_plus_2model.yml"):
        src = open(f"recipes/{fname}").read()
        cfg = MergeConfiguration.model_validate(yaml.safe_load(src))
        spec = get(cfg.merge_method).spec
        assert spec.contract.base.value == "required"
    shutil.rmtree(tmp_path, ignore_errors=True)

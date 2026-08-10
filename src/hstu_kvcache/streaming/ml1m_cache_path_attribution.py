from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .ml1m_locked_confirmation import load_locked_config
from .ml1m_opportunity import (
    _atomic_json,
    _evaluate,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
)
from .ml1m_query_objective import _candidate_maps

PROTOCOL = "evokv_ml1m_cache_path_attribution_v0"
VARIANTS = [
    "all_old",
    "current_input",
    "current_input_block0",
    "current_cache_path",
    "all_current",
]
METRICS = [
    "candidate_cross_entropy",
    "hit_rate_at_5",
    "hit_rate_at_10",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
]


def load_attribution_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("source_variants") != VARIANTS
        or document.get("candidate_protocol") != "popular_unseen_50"
        or document.get("candidate_seed") != 917341
    ):
        raise ValueError("ML1m cache-path attribution config differs")
    parent = document.get("parent", {})
    for key in ("confirmation_config", "confirmation_summary"):
        if file_sha256(parent.get(key, "")) != parent.get(f"{key}_sha256"):
            raise ValueError("ML1m cache-path attribution parent binding differs")
    return document


def _load_checkpoint(binding: dict[str, Any]) -> dict[str, torch.Tensor]:
    if file_sha256(binding["path"]) != binding["sha256"]:
        raise ValueError("ML1m cache-path checkpoint binding differs")
    payload = torch.load(binding["path"], map_location="cpu", weights_only=True)
    return payload["state_dict"]


def _source_keys(state: dict[str, torch.Tensor], variant: str) -> set[str]:
    input_prefixes = ("item_emb.", "behavior_emb.", "temporal_enc.", "in_proj.")
    input_keys = {name for name in state if name.startswith(input_prefixes)}
    block0_keys = {name for name in state if name.startswith("blocks.0.")}
    last_kv_keys = {
        name
        for name in state
        if name.startswith("blocks.1.norm.")
        or name.startswith("blocks.1.attn.k_proj.")
        or name.startswith("blocks.1.attn.v_proj.")
    }
    if variant == "all_old":
        return set()
    if variant == "current_input":
        return input_keys
    if variant == "current_input_block0":
        return input_keys | block0_keys
    if variant == "current_cache_path":
        return input_keys | block0_keys | last_kv_keys
    if variant == "all_current":
        return set(state)
    raise ValueError("ML1m cache-path source variant differs")


def _make_source(
    base_document: dict[str, Any],
    old_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    variant: str,
    device: torch.device,
):
    state = {name: value.clone() for name, value in old_state.items()}
    keys = _source_keys(state, variant)
    for name in keys:
        state[name] = current_state[name].clone()
    model = make_model(base_document, "legacy", device)
    model.load_state_dict(state)
    model.eval()
    return model, sorted(keys)


def _tax(metric: str, exact: float, source: float) -> float:
    if metric == "candidate_cross_entropy":
        return source - exact
    return exact - source


def _summarize_recovery(variants: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = variants[0]["summary"]["endpoints"]
    exact = baseline["recompute"]
    output = {}
    for value in variants:
        source = value["summary"]["endpoints"]["reuse"]
        metrics = {}
        for metric in METRICS:
            baseline_tax = _tax(metric, exact[metric], baseline["reuse"][metric])
            residual_tax = _tax(metric, exact[metric], source[metric])
            recovery = None
            if abs(baseline_tax) > 1e-12:
                recovery = 1.0 - residual_tax / baseline_tax
            metrics[metric] = {
                "baseline_tax": baseline_tax,
                "recovery_fraction": recovery,
                "residual_tax": residual_tax,
            }
        output[value["variant"]] = metrics
    return output


def run_cache_path_attribution(config_path: str | Path) -> dict[str, Any]:
    document = load_attribution_config(config_path)
    result_path = Path(document["outputs"]["result"])
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_attribution_result(result, document)
        return result
    confirmation_document = load_locked_config(document["parent"]["confirmation_config"])
    confirmation = json.loads(Path(document["parent"]["confirmation_summary"]).read_text())
    base_document = load_config(confirmation_document["parent"]["base_config"])
    base_document["data"]["user_limit"] = int(confirmation_document["data"]["user_limit"])
    base_document["training"].update(confirmation_document["training"])
    causal = load_causal_records(base_document)
    selected = causal["selected_users"]
    candidates = _candidate_maps(
        causal,
        selected,
        int(base_document["model"]["num_items"]),
        int(document["candidate_seed"]),
    )[document["candidate_protocol"]]
    old_state = _load_checkpoint(confirmation["checkpoints"]["theta0"])
    current_state = _load_checkpoint(confirmation["checkpoints"]["theta1"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m cache-path attribution")
    current = make_model(base_document, "legacy", device)
    current.load_state_dict(current_state)
    current.eval()
    started = time.monotonic()
    variants = []
    for variant in document["source_variants"]:
        source, copied_keys = _make_source(
            base_document,
            old_state,
            current_state,
            variant,
            device,
        )
        evaluation = _evaluate(
            source,
            current,
            causal["splits"]["test"],
            selected,
            base_document,
            True,
            candidate_map=candidates,
            candidate_protocol_override=document["candidate_protocol"],
            prediction_query=True,
        )
        compact = summarize_evaluation(evaluation, base_document)
        variants.append(
            {
                "variant": variant,
                "current_parameter_count": len(copied_keys),
                "current_parameter_names": copied_keys,
                "summary": compact,
            }
        )
        print(
            f"phase=ml1m_cache_path variant={variant} "
            f"cache_k={compact['representation_drift']['cache_k_relative_error']['mean']:.6f} "
            f"cache_v={compact['representation_drift']['cache_v_relative_error']['mean']:.6f}",
            flush=True,
        )
        del source
        torch.cuda.empty_cache()
    recovery = _summarize_recovery(variants)
    exact = variants[-2]["summary"]
    exact_tax = exact["comparisons"]["recompute_over_reuse"]["candidate_cross_entropy"]["absolute"]
    result = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete_development_attribution",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "seed": confirmation["seed"],
        "candidate_protocol": document["candidate_protocol"],
        "variants": variants,
        "recovery": recovery,
        "decision": {
            "cache_path_is_exact_boundary": abs(exact_tax) <= 1e-7,
            "baseline_stale_positive": variants[0]["summary"]["comparisons"][
                "recompute_over_reuse"
            ]["ndcg_at_10"]["positive_direction_with_ci"],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_attribution_result(result, document)
    _atomic_json(result_path, result)
    return result


def validate_attribution_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    variants = result.get("variants", [])
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete_development_attribution"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or [value.get("variant") for value in variants] != document["source_variants"]
        or result.get("config", {}).get("sha256") != file_sha256(result["config"]["path"])
        or not all(value.get("summary", {}).get("sanity", {}).get("passed") for value in variants)
    ):
        raise ValueError("ML1m cache-path attribution result differs")
    expected = _summarize_recovery(variants)
    if result.get("recovery") != expected:
        raise ValueError("ML1m cache-path attribution recovery differs")
    exact = variants[-2]["summary"]["comparisons"]["recompute_over_reuse"]
    if abs(exact["candidate_cross_entropy"]["absolute"]) > 1e-7:
        raise ValueError("ML1m cache-path attribution exact boundary differs")

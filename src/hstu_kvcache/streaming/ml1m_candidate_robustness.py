from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ml1m_opportunity import (
    _atomic_json,
    _evaluate,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
    summarize_records,
)

PROTOCOL_V0 = "evokv_ml1m_candidate_robustness_v0"
PROTOCOL_V1 = "evokv_ml1m_candidate_robustness_seen_filtered_v1"


def load_robustness_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    evaluation = document.get("evaluation")
    if (
        document.get("protocol") not in (PROTOCOL_V0, PROTOCOL_V1)
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(evaluation, dict)
        or evaluation.get("candidate_counts") != [20, 50, 100, 1000, 3883]
        or evaluation.get("primary_candidate_count") != 100
        or document.get("variants")
        != [
            "a0_legacy_raw_full",
            "a1_legacy_normalized_full",
            "a2_dense_normalized_full",
        ]
    ):
        raise ValueError("ML1m candidate robustness config differs")
    if (document.get("protocol") == PROTOCOL_V1) != bool(evaluation.get("filter_seen", False)):
        raise ValueError("ML1m candidate robustness seen-item policy differs")
    for path_key, hash_key in (("config", "config_sha256"), ("summary", "summary_sha256")):
        if file_sha256(parent[path_key]) != parent[hash_key]:
            raise ValueError("ML1m candidate robustness parent binding differs")
    return document


def _load_models(
    parent_variant: dict[str, Any],
    base_document: dict[str, Any],
    device: torch.device,
):
    result_path = Path(parent_variant["result_path"])
    if file_sha256(result_path) != parent_variant["result_sha256"]:
        raise ValueError("ML1m candidate robustness variant result differs")
    result = json.loads(result_path.read_text())
    architecture = result["variant"]["architecture"]
    models = []
    for version in ("theta0", "theta1"):
        binding = result["checkpoints"][version]
        checkpoint_path = Path(binding["path"])
        if file_sha256(checkpoint_path) != binding["sha256"]:
            raise ValueError("ML1m candidate robustness checkpoint differs")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload["architecture"] != architecture:
            raise ValueError("ML1m candidate robustness architecture differs")
        model = make_model(base_document, architecture, device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return result, models[0], models[1]


def _strict_ranking_gate(summary: dict[str, Any]) -> dict[str, Any]:
    comparisons = summary["comparisons"]
    required = {}
    for comparison_name in ("fresh_update_value", "history_value", "recompute_over_reuse"):
        for metric in ("mrr", "ndcg_at_10"):
            required[f"{comparison_name}_{metric}"] = comparisons[comparison_name][metric][
                "positive_direction_with_ci"
            ]
    required["recompute_over_reuse_ce"] = comparisons["recompute_over_reuse"][
        "candidate_cross_entropy"
    ]["positive_direction_with_ci"]
    required["same_model_sanity"] = summary["sanity"]["passed"]
    return {"requirements": required, "passed": all(required.values())}


def _mechanism_summary(records: list[dict[str, Any]], base_document: dict[str, Any]):
    boundaries = [0, 32, 64, 96, 128]
    strata = {}
    for index, lower in enumerate(boundaries[:-1]):
        upper = boundaries[index + 1]
        selected = [value for value in records if lower <= value["prefix_length"] < upper]
        strata[f"{lower}_{upper}"] = summarize_records(selected, base_document)
    prefix = np.asarray([value["prefix_length"] for value in records], dtype=np.float64)
    hidden = np.asarray([value["hidden_relative_error"] for value in records], dtype=np.float64)
    stale_ce = np.asarray(
        [
            value["metrics"]["reuse"]["candidate_cross_entropy"]
            - value["metrics"]["recompute"]["candidate_cross_entropy"]
            for value in records
        ],
        dtype=np.float64,
    )
    stale_ndcg = np.asarray(
        [
            value["metrics"]["recompute"]["ndcg_at_10"]
            - value["metrics"]["reuse"]["ndcg_at_10"]
            for value in records
        ],
        dtype=np.float64,
    )
    return {
        "prefix_length_boundaries": boundaries,
        "strata": strata,
        "pearson": {
            "prefix_length_vs_hidden_relative_error": float(np.corrcoef(prefix, hidden)[0, 1]),
            "hidden_relative_error_vs_stale_ce": float(np.corrcoef(hidden, stale_ce)[0, 1]),
            "hidden_relative_error_vs_stale_ndcg_at_10": float(
                np.corrcoef(hidden, stale_ndcg)[0, 1]
            ),
        },
    }


def run_candidate_robustness(config_path: str | Path) -> dict[str, Any]:
    document = load_robustness_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_robustness_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    base_document = load_config(document["parent"]["config"])
    parent_summary = json.loads(Path(document["parent"]["summary"]).read_text())
    causal = load_causal_records(base_document)
    if causal["selected_users_sha256"] != parent_summary["data"]["selected_users_sha256"]:
        raise ValueError("ML1m candidate robustness selected users differ")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m candidate robustness")
    parent_by_id = {value["id"]: value for value in parent_summary["variants"]}
    variant_summaries = []
    started = time.monotonic()
    for variant_id in document["variants"]:
        print(f"phase=ml1m_candidate_robustness_start variant={variant_id}", flush=True)
        parent_result, previous, current = _load_models(
            parent_by_id[variant_id],
            base_document,
            device,
        )
        candidate_results = []
        mechanism = None
        for count in document["evaluation"]["candidate_counts"]:
            evaluation = _evaluate(
                previous,
                current,
                causal["splits"]["test"],
                causal["selected_users"],
                base_document,
                bool(parent_result["variant"]["normalized_scoring"]),
                candidate_count=int(count),
                candidate_seed=int(document["evaluation"]["candidate_seed"]),
                filter_seen=bool(document["evaluation"].get("filter_seen", False)),
            )
            compact = summarize_evaluation(evaluation, base_document)
            gate = _strict_ranking_gate(compact)
            result_path = output_root / "variants" / variant_id / f"candidate_{count}.json"
            _atomic_json(
                result_path,
                {
                    "variant": variant_id,
                    "candidate_count": count,
                    "summary": compact,
                    "strict_ranking_gate": gate,
                    "records": evaluation["records"],
                },
            )
            candidate_results.append(
                {
                    "candidate_count": count,
                    "result_path": str(result_path),
                    "result_sha256": file_sha256(result_path),
                    "summary": compact,
                    "strict_ranking_gate": gate,
                }
            )
            if count == int(document["evaluation"]["primary_candidate_count"]):
                mechanism = _mechanism_summary(evaluation["records"], base_document)
            stale = compact["comparisons"]["recompute_over_reuse"]
            print(
                f"phase=ml1m_candidate_robustness_result variant={variant_id} candidates={count} "
                f"stale_ce={stale['candidate_cross_entropy']['absolute']:.6f} "
                f"stale_mrr={stale['mrr']['absolute']:.6f} "
                f"stale_ndcg10={stale['ndcg_at_10']['absolute']:.6f} passed={gate['passed']}",
                flush=True,
            )
        variant_summaries.append(
            {
                "id": variant_id,
                "candidate_results": candidate_results,
                "primary_mechanism": mechanism,
            }
        )
        del previous, current
        torch.cuda.empty_cache()
    primary_count = int(document["evaluation"]["primary_candidate_count"])
    passed = []
    for variant in variant_summaries:
        primary = next(
            value for value in variant["candidate_results"] if value["candidate_count"] == primary_count
        )
        full = variant["candidate_results"][-1]
        if primary["strict_ranking_gate"]["passed"] and full["summary"]["comparisons"][
            "recompute_over_reuse"
        ]["candidate_cross_entropy"]["positive_direction_with_ci"]:
            passed.append(variant["id"])
    summary = {
        "protocol": document["protocol"],
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "variants": variant_summaries,
        "decision": {
            "primary_candidate_count": primary_count,
            "passed_variants": passed,
            "positive_candidate_found": bool(passed),
            "next": "all_user_two_seed_replication" if passed else "training_objective_followup",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_robustness_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_robustness_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    variants = summary.get("variants")
    if (
        summary.get("protocol") != document["protocol"]
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or not isinstance(variants, list)
        or [value.get("id") for value in variants] != document["variants"]
    ):
        raise ValueError("ML1m candidate robustness summary differs")
    for variant in variants:
        results = variant.get("candidate_results")
        if not isinstance(results, list) or [value.get("candidate_count") for value in results] != document[
            "evaluation"
        ]["candidate_counts"]:
            raise ValueError("ML1m candidate robustness candidate coverage differs")
        for result in results:
            path = Path(result["result_path"])
            if not path.is_file() or file_sha256(path) != result["result_sha256"]:
                raise ValueError("ML1m candidate robustness result binding differs")
            if not result.get("summary", {}).get("sanity", {}).get("passed"):
                raise ValueError("ML1m candidate robustness sanity failed")

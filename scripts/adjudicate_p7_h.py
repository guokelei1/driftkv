#!/usr/bin/env python3
"""Compute the pre-registered P7.8 metrics only after raw-score sealing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN_PLAN = ROOT / "configs/contracts/p7_8_qualification_run_plan_v1.json"
RAW_SEAL = ROOT / "results/p7/h_qualification/raw_score_seal_v1.json"
OUTPUT_ROOT = ROOT / "results/p7/h_qualification"
MODELS = {
    ("m0_n", "N"): ("quality", "fidelity"),
    ("m0_r", "R"): ("quality_rankable", "fidelity_all_eligible"),
    ("m0_f", "F"): ("quality", "fidelity"),
    ("m1", "N"): ("quality", "fidelity"),
    ("m1", "R"): ("quality_rankable", "fidelity_all_eligible"),
    ("m1", "F"): ("quality", "fidelity"),
}
SEEDS = (17, 37, 71)
PER_SEED_REPLICATES = 2000
HIERARCHICAL_REPLICATES = 10000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_for(*parts: object) -> np.random.Generator:
    token = "p7-8-H-bootstrap-v1|" + "|".join(str(part) for part in parts)
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(value, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return float(output) if output.ndim == 0 else output


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    output = np.exp(shifted)
    return output / output.sum()


def js_divergence(recent: np.ndarray, full: np.ndarray, workload: str) -> float:
    if workload == "F":
        p_like = float(sigmoid(recent[0]))
        q_like = float(sigmoid(full[0]))
        p = np.asarray([p_like, 1.0 - p_like])
        q = np.asarray([q_like, 1.0 - q_like])
    else:
        p, q = softmax(recent), softmax(full)
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


def request_panel(request_id: str) -> int:
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") % 4


def raw_path(model: str, seed: int, workload: str, view: str) -> Path:
    return ROOT / f"results/p7/h_qualification/raw/{model}_seed{seed}/{workload}_{view}.parquet"


def grouped_rows(path: Path):
    frame = pq.read_table(path).to_pandas()
    return frame, frame.groupby("request_id", sort=False)


def fidelity_records(path: Path, workload: str) -> list[dict[str, Any]]:
    _, groups = grouped_rows(path)
    records = []
    full_logits = []
    for request_id, group in groups:
        recent = group["recent32_deployment_logit"].to_numpy(dtype=np.float64)
        full = group["full512_deployment_logit"].to_numpy(dtype=np.float64)
        delta = full - recent
        row: dict[str, Any] = {
            "request_id": request_id,
            "uid": int(group["uid"].iloc[0]),
            "panel": request_panel(str(request_id)),
            "output_js_divergence": js_divergence(recent, full, workload),
        }
        if workload == "F":
            row["logit_delta_square"] = float(delta[0] ** 2)
            row["absolute_probability_difference"] = abs(float(sigmoid(full[0])) - float(sigmoid(recent[0])))
            full_logits.append(float(full[0]))
        else:
            centered_delta = delta - delta.mean()
            centered_full = full - full.mean()
            row["normalized_score_rms"] = float(
                np.sqrt(np.mean(centered_delta**2))
                / (np.sqrt(np.mean(centered_full**2)) + 1e-6)
            )
            tau = kendalltau(recent, full, variant="b").statistic
            row["pairwise_inversion"] = 0.0 if not np.isfinite(tau) else float((1.0 - tau) / 2.0)
            top = min(10, len(full))
            recent_top = set(np.argsort(-recent, kind="stable")[:top].tolist())
            full_top = set(np.argsort(-full, kind="stable")[:top].tolist())
            row["top10_overlap_loss"] = 1.0 - len(recent_top & full_top) / top
        records.append(row)
    if workload == "F":
        denominator = max(float(np.std(full_logits)), 1e-3)
        for row in records:
            row["normalized_score_rms"] = math.sqrt(row.pop("logit_delta_square")) / denominator
    return records


def rank_metrics(scores: np.ndarray, target: int) -> dict[str, float]:
    target_score = float(scores[target])
    others = np.delete(scores.astype(np.float64), target)
    auc = float(np.mean((target_score > others) + 0.5 * (target_score == others)))
    margin = target_score - float(np.max(others))
    tied_before = int(np.sum((scores == target_score) & (np.arange(len(scores)) < target)))
    rank = 1 + int(np.sum(others > target_score)) + tied_before
    return {
        "pairwise_target_AUC": auc,
        "target_hard_negative_margin": margin,
        "NDCG_at_10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "HR_at_10": float(rank <= 10),
        "MRR": 1.0 / rank,
    }


def ranking_quality_records(path: Path) -> list[dict[str, Any]]:
    _, groups = grouped_rows(path)
    records = []
    for request_id, group in groups:
        target = int(group["target_index"].iloc[0])
        recent = group["recent32_deployment_logit"].to_numpy(dtype=np.float64)
        full = group["full512_deployment_logit"].to_numpy(dtype=np.float64)
        recent_metrics = rank_metrics(recent, target)
        full_metrics = rank_metrics(full, target)
        records.append(
            {
                "request_id": request_id,
                "uid": int(group["uid"].iloc[0]),
                **{
                    f"{name}_gain": full_metrics[name] - recent_metrics[name]
                    for name in recent_metrics
                },
                **{f"recent_{name}": value for name, value in recent_metrics.items()},
                **{f"full_{name}": value for name, value in full_metrics.items()},
            }
        )
    return records


def feedback_quality_records(path: Path) -> list[dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    if not (frame.groupby("request_id").size() == 1).all():
        raise RuntimeError("feedback quality must have one candidate per request")
    records = []
    for row in frame.itertuples(index=False):
        label = int(row.label)
        recent_logit = float(row.recent32_deployment_logit)
        full_logit = float(row.full512_deployment_logit)
        recent_probability = float(sigmoid(recent_logit))
        full_probability = float(sigmoid(full_logit))
        recent_loss = float(np.logaddexp(0.0, recent_logit) - label * recent_logit)
        full_loss = float(np.logaddexp(0.0, full_logit) - label * full_logit)
        records.append(
            {
                "request_id": row.request_id,
                "uid": int(row.uid),
                "label": label,
                "recent_probability": recent_probability,
                "full_probability": full_probability,
                "request_weight": float(row.request_weight),
                "log_loss_gain": recent_loss - full_loss,
                "Brier_gain": (recent_probability - label) ** 2 - (full_probability - label) ** 2,
                "prior_30m_same_item": bool(row.prior_30m_same_item),
                "latest_item": bool(row.latest_item),
                "long_gap_at_least_3d": bool(row.long_gap_at_least_3d),
                "is_organic": int(row.is_organic),
                "history_position_cohort": row.history_position_cohort,
            }
        )
    return records


def user_values(records: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in records:
        grouped[int(row["uid"])].append(float(row[key]))
    uids = np.asarray(sorted(grouped), dtype=np.int64)
    values = np.asarray([np.mean(grouped[int(uid)]) for uid in uids], dtype=np.float64)
    return uids, values


def mean_effect_summary(
    records: list[dict[str, Any]], key: str, namespace: tuple[object, ...]
) -> tuple[dict[str, Any], np.ndarray]:
    _, values = user_values(records, key)
    rng = rng_for(*namespace, key)
    draws = values[rng.integers(0, len(values), size=(PER_SEED_REPLICATES, len(values)))].mean(axis=1)
    return {
        "point": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "users": len(values),
    }, draws


def weighted_classification_metrics(
    records: list[dict[str, Any]], user_counts: dict[int, int] | None = None
) -> dict[str, float]:
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    recent = np.asarray([row["recent_probability"] for row in records], dtype=np.float64)
    full = np.asarray([row["full_probability"] for row in records], dtype=np.float64)
    base_weights = np.asarray([row["request_weight"] for row in records], dtype=np.float64)
    if user_counts is not None:
        base_weights = base_weights * np.asarray([user_counts.get(int(row["uid"]), 0) for row in records])
    if base_weights.sum() == 0 or len(np.unique(labels[base_weights > 0])) < 2:
        return {"ROC_AUC_gain": float("nan"), "dislike_PR_AUC_gain": float("nan")}
    roc_recent = roc_auc_score(labels, recent, sample_weight=base_weights)
    roc_full = roc_auc_score(labels, full, sample_weight=base_weights)
    dislike = 1 - labels
    pr_recent = average_precision_score(dislike, 1.0 - recent, sample_weight=base_weights)
    pr_full = average_precision_score(dislike, 1.0 - full, sample_weight=base_weights)
    return {
        "ROC_AUC_gain": float(roc_full - roc_recent),
        "dislike_PR_AUC_gain": float(pr_full - pr_recent),
    }


def classification_effect_summaries(
    records: list[dict[str, Any]], namespace: tuple[object, ...]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    users = sorted({int(row["uid"]) for row in records})
    rng = rng_for(*namespace, "classification")
    point = weighted_classification_metrics(records)
    draws = {key: np.empty(PER_SEED_REPLICATES, dtype=np.float64) for key in point}
    for replicate in range(PER_SEED_REPLICATES):
        sampled = rng.integers(0, len(users), size=len(users))
        counts = np.bincount(sampled, minlength=len(users))
        metrics = weighted_classification_metrics(
            records, {users[index]: int(count) for index, count in enumerate(counts) if count}
        )
        for key, value in metrics.items():
            draws[key][replicate] = value
    summaries = {
        key: {
            "point": value,
            "ci95": [float(np.nanquantile(draws[key], 0.025)), float(np.nanquantile(draws[key], 0.975))],
            "users": len(users),
        }
        for key, value in point.items()
    }
    return summaries, draws


def panel_points(records: list[dict[str, Any]], key: str) -> list[float]:
    output = []
    for panel in range(4):
        selected = [row for row in records if row["panel"] == panel]
        _, values = user_values(selected, key)
        output.append(float(values.mean()))
    return output


def hierarchical_summary(
    seed_summaries: list[dict[str, Any]], seed_draws: list[np.ndarray], namespace: tuple[object, ...]
) -> dict[str, Any]:
    rng = rng_for(*namespace, "hierarchical")
    output = np.empty(HIERARCHICAL_REPLICATES, dtype=np.float64)
    for index in range(HIERARCHICAL_REPLICATES):
        chosen = rng.integers(0, len(seed_draws), size=len(seed_draws))
        output[index] = np.mean(
            [seed_draws[int(seed)][rng.integers(0, len(seed_draws[int(seed)]))] for seed in chosen]
        )
    return {
        "point_equal_seed_mean": float(np.mean([row["point"] for row in seed_summaries])),
        "ci95_hierarchical": [float(np.quantile(output, 0.025)), float(np.quantile(output, 0.975))],
        "seed_points": [row["point"] for row in seed_summaries],
    }


def cohort_report(records: list[dict[str, Any]], namespace: tuple[object, ...]) -> dict[str, Any]:
    filters: dict[str, Callable[[dict[str, Any]], bool]] = {
        "prior_30m_same_item": lambda row: row["prior_30m_same_item"],
        "non_prior_30m_same_item": lambda row: not row["prior_30m_same_item"],
        "latest_item": lambda row: row["latest_item"],
        "non_latest_item": lambda row: not row["latest_item"],
        "long_gap_at_least_3d": lambda row: row["long_gap_at_least_3d"],
        "short_gap_under_3d": lambda row: not row["long_gap_at_least_3d"],
        "like": lambda row: row["label"] == 1,
        "dislike": lambda row: row["label"] == 0,
        "organic": lambda row: row["is_organic"] == 1,
        "recommendation_driven": lambda row: row["is_organic"] == 0,
        "recent_seen": lambda row: row["history_position_cohort"] == "recent_seen",
        "old_seen": lambda row: row["history_position_cohort"] == "old_seen",
        "seen_only_before_512": lambda row: row["history_position_cohort"] == "seen_only_before_512",
        "unseen": lambda row: row["history_position_cohort"] == "unseen",
    }
    output = {}
    for name, predicate in filters.items():
        selected = [row for row in records if predicate(row)]
        if not selected:
            output[name] = {"requests": 0, "users": 0, "log_loss_gain": None}
            continue
        summary, _ = mean_effect_summary(selected, "log_loss_gain", (*namespace, "cohort", name))
        output[name] = {
            "requests": len(selected),
            "users": len({row["uid"] for row in selected}),
            "log_loss_gain": summary,
        }
    return output


def strip_bootstrap(value: tuple[dict[str, Any], np.ndarray]) -> dict[str, Any]:
    return value[0]


def evaluate_condition(model: str, workload: str, floors: dict[str, float]) -> dict[str, Any]:
    quality_view, fidelity_view = MODELS[(model, workload)]
    per_seed = []
    metric_draws: dict[str, list[np.ndarray]] = defaultdict(list)
    metric_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in SEEDS:
        fidelity = fidelity_records(raw_path(model, seed, workload, fidelity_view), workload)
        fidelity_keys = (
            ("output_js_divergence", "normalized_score_rms", "absolute_probability_difference")
            if workload == "F"
            else ("output_js_divergence", "normalized_score_rms", "pairwise_inversion", "top10_overlap_loss")
        )
        fidelity_summary = {}
        panels = {}
        for key in fidelity_keys:
            summary, draws = mean_effect_summary(fidelity, key, (model, workload, seed, "fidelity"))
            fidelity_summary[key] = summary
            metric_summaries[f"fidelity.{key}"].append(summary)
            metric_draws[f"fidelity.{key}"].append(draws)
            panels[key] = panel_points(fidelity, key)

        if workload in {"N", "R"}:
            quality = ranking_quality_records(raw_path(model, seed, workload, quality_view))
            quality_keys = (
                "pairwise_target_AUC_gain",
                "target_hard_negative_margin_gain",
                "NDCG_at_10_gain",
                "HR_at_10_gain",
                "MRR_gain",
            )
            quality_summary = {}
            for key in quality_keys:
                summary, draws = mean_effect_summary(quality, key, (model, workload, seed, "quality"))
                quality_summary[key] = summary
                metric_summaries[f"quality.{key}"].append(summary)
                metric_draws[f"quality.{key}"].append(draws)
            cohorts = None
        else:
            quality = feedback_quality_records(raw_path(model, seed, workload, quality_view))
            quality_summary = {}
            for key in ("log_loss_gain", "Brier_gain"):
                summary, draws = mean_effect_summary(quality, key, (model, workload, seed, "quality"))
                quality_summary[key] = summary
                metric_summaries[f"quality.{key}"].append(summary)
                metric_draws[f"quality.{key}"].append(draws)
            classification_summaries, classification_draws = classification_effect_summaries(
                quality, (model, workload, seed)
            )
            for key, summary in classification_summaries.items():
                quality_summary[key] = summary
                metric_summaries[f"quality.{key}"].append(summary)
                metric_draws[f"quality.{key}"].append(classification_draws[key])
            cohorts = cohort_report(quality, (model, workload, seed))

        primary_quality = (
            quality_summary["pairwise_target_AUC_gain"]
            if workload in {"N", "R"}
            else quality_summary["log_loss_gain"]
        )
        if workload == "R":
            companions_ok = all(
                quality_summary[key]["ci95"][0] >= -0.005
                for key in ("NDCG_at_10_gain", "HR_at_10_gain", "MRR_gain")
            )
        elif workload == "F":
            companions_ok = (
                quality_summary["ROC_AUC_gain"]["ci95"][0] >= -0.005
                and quality_summary["dislike_PR_AUC_gain"]["ci95"][0] >= -0.01
            )
        else:
            companions_ok = True
        h_floor = floors["output_js_divergence"]
        h_ok = fidelity_summary["output_js_divergence"]["ci95"][0] > h_floor
        quality_ok = primary_quality["ci95"][0] > 0.0
        seed_positive = h_ok and quality_ok and companions_ok
        per_seed.append(
            {
                "seed": seed,
                "fidelity": fidelity_summary,
                "fidelity_panel_points": panels,
                "quality": quality_summary,
                "cohorts": cohorts,
                "H_quality_positive": seed_positive,
                "gate_components": {
                    "H_primary_CI_above_floor": h_ok,
                    "quality_primary_CI_positive": quality_ok,
                    "companions_noninferior": companions_ok,
                },
            }
        )

    aggregate = {
        metric: hierarchical_summary(metric_summaries[metric], draws, (model, workload, metric))
        for metric, draws in metric_draws.items()
    }
    h_aggregate = aggregate["fidelity.output_js_divergence"]
    quality_primary_name = (
        "quality.pairwise_target_AUC_gain" if workload in {"N", "R"} else "quality.log_loss_gain"
    )
    quality_aggregate = aggregate[quality_primary_name]
    panel_repeatable = all(
        sum(value > floors["output_js_divergence"] for value in row["fidelity_panel_points"]["output_js_divergence"]) >= 3
        for row in per_seed
    )
    if workload in {"N", "R"}:
        not_scale_only = (
            aggregate["fidelity.pairwise_inversion"]["ci95_hierarchical"][0] > floors["pairwise_inversion"]
            or aggregate["fidelity.top10_overlap_loss"]["ci95_hierarchical"][0] > floors["top10_overlap_loss"]
            or quality_aggregate["ci95_hierarchical"][0] > 0.0
        )
    else:
        not_scale_only = (
            aggregate["fidelity.absolute_probability_difference"]["ci95_hierarchical"][0]
            > floors["absolute_probability_difference"]
        )
    if workload == "R":
        companions_aggregate = all(
            aggregate[f"quality.{key}"]["ci95_hierarchical"][0] >= -0.005
            for key in ("NDCG_at_10_gain", "HR_at_10_gain", "MRR_gain")
        )
    elif workload == "F":
        companions_aggregate = (
            aggregate["quality.ROC_AUC_gain"]["ci95_hierarchical"][0] >= -0.005
            and aggregate["quality.dislike_PR_AUC_gain"]["ci95_hierarchical"][0] >= -0.01
        )
    else:
        companions_aggregate = True
    positive_seed_count = sum(row["H_quality_positive"] for row in per_seed)
    workload_gate = (
        workload != "N"
        and h_aggregate["ci95_hierarchical"][0] > floors["output_js_divergence"]
        and quality_aggregate["ci95_hierarchical"][0] > 0.0
        and sum(point > 0 for point in quality_aggregate["seed_points"]) >= 2
        and companions_aggregate
        and panel_repeatable
        and not_scale_only
    )
    if workload_gate and positive_seed_count == 3:
        classification = "robust_3_of_3"
    elif workload_gate and positive_seed_count == 2:
        classification = "provisional_2_of_3"
    elif positive_seed_count == 1:
        classification = "isolated_mechanism_case_1_of_3"
    else:
        classification = "no_qualified_long_state_object"
    return {
        "model_condition": model,
        "workload": workload,
        "role": "negative_control"
        if workload == "N" and model == "m0_n"
        else "primary"
        if model.startswith("m0_")
        else "multitask_companion",
        "per_seed": per_seed,
        "seed_equal_aggregate": aggregate,
        "positive_seed_count": positive_seed_count,
        "gate_components": {
            "H_primary_aggregate_CI_above_floor": h_aggregate["ci95_hierarchical"][0]
            > floors["output_js_divergence"],
            "quality_primary_aggregate_CI_positive": quality_aggregate["ci95_hierarchical"][0] > 0.0,
            "majority_seed_quality_point_positive": sum(point > 0 for point in quality_aggregate["seed_points"]) >= 2,
            "companions_aggregate_noninferior": companions_aggregate,
            "four_panel_repeatability": panel_repeatable,
            "not_scale_only": not_scale_only,
        },
        "workload_H_gate_passed": workload_gate,
        "classification": classification,
    }


def main() -> None:
    if (OUTPUT_ROOT / "adjudication_report_v1.json").exists():
        raise FileExistsError("refusing to overwrite revealed P7.8 adjudication")
    plan = json.loads(RUN_PLAN.read_text())
    if plan["adjudicator_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("adjudicator changed after run-plan sealing")
    seal = json.loads(RAW_SEAL.read_text())
    if seal["status"] != "sealed_all_raw_scores_before_metrics" or seal["metrics_computed"] is not False:
        raise RuntimeError("raw scores were not sealed before metric computation")
    for artifact in seal["artifacts"]:
        if sha256_file(ROOT / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"raw artifact changed: {artifact['path']}")
    floors = plan["numeric_floors"]
    conditions = [evaluate_condition(model, workload, floors) for model, workload in MODELS]
    primary_r = next(row for row in conditions if row["model_condition"] == "m0_r")
    primary_f = next(row for row in conditions if row["model_condition"] == "m0_f")
    m1_passes = [row["workload"] for row in conditions if row["model_condition"] == "m1" and row["workload_H_gate_passed"]]
    if primary_r["workload_H_gate_passed"] or primary_f["workload_H_gate_passed"]:
        branch = "A_primary_M0_workload_passed"
        version_chain_eligible = [
            row["workload"] for row in (primary_r, primary_f) if row["workload_H_gate_passed"]
        ]
    elif m1_passes:
        branch = "B_only_multitask_companion_passed"
        version_chain_eligible = [f"M1-{workload}" for workload in m1_passes]
    elif any(
        row["gate_components"]["H_primary_aggregate_CI_above_floor"]
        and not row["gate_components"]["quality_primary_aggregate_CI_positive"]
        for row in conditions
        if row["workload"] in {"R", "F"}
    ):
        branch = "C_H_without_quality_no_version_chain"
        version_chain_eligible = []
    else:
        branch = "D_no_R_or_F_long_state_object"
        version_chain_eligible = []
    report = {
        "status": "qualification_revealed_adjudicated_stopped",
        "evidence_level": "one_time_out_of_time_H_qualification",
        "run_plan_sha256": sha256_file(RUN_PLAN),
        "raw_score_seal_sha256": sha256_file(RAW_SEAL),
        "all_seed_results": conditions,
        "release_admitted_subset": {
            "relation": "identical_to_all_seed_results_under_frozen_P7_7_admission_rule",
            "admitted_checkpoints": 12,
        },
        "adjudication_branch": branch,
        "version_chain_eligible_conditions": version_chain_eligible,
        "theta1_theta2_started": False,
        "requires_new_human_authorization": True,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "adjudication_report_v1.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "status": report["status"],
        "adjudication_branch": branch,
        "version_chain_eligible_conditions": version_chain_eligible,
        "conditions": [
            {
                "model": row["model_condition"],
                "workload": row["workload"],
                "classification": row["classification"],
                "gate_passed": row["workload_H_gate_passed"],
                "positive_seed_count": row["positive_seed_count"],
            }
            for row in conditions
        ],
    }
    (OUTPUT_ROOT / "adjudication_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

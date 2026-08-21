#!/usr/bin/env python3
"""Compute frozen P8 H/S metrics from one sealed R1/R2 raw matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import train_p7_theta0 as p7
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score

from hstu_kvcache.data.p7_training import load_p7_requests

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ("r1_edge1", "r1_edge2", "r2")
SEEDS = (17, 37, 71)
P7_NUMERIC_FLOOR = 1e-8
R0_FLOOR = 1e-8
PER_SEED_REPLICATES = 2000
HIERARCHICAL_REPLICATES = 10000
COMPARISONS = {
    "H": "current_recent32_logit",
    "S": "reuse_parent_kv_logit",
}
EVALUATION_SPLIT = {"r1_edge1": "edge1_evaluation", "r1_edge2": "edge2_evaluation", "r2": "edge1_evaluation"}
MANIFEST_ROOT = ROOT / "data/manifests/p8_release_v1"
RAW_LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"


def rng_for(*parts: object) -> np.random.Generator:
    token = "p8-HS-bootstrap-v1|" + "|".join(str(part) for part in parts)
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return float(result) if result.ndim == 0 else result


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    output = np.exp(shifted)
    return output / output.sum()


def js_divergence(reference: np.ndarray, current: np.ndarray, workload: str) -> float:
    if workload == "F":
        p_like, q_like = float(sigmoid(reference[0])), float(sigmoid(current[0]))
        p = np.asarray([p_like, 1.0 - p_like])
        q = np.asarray([q_like, 1.0 - q_like])
    else:
        p, q = softmax(reference), softmax(current)
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


def request_panel(request_id: str) -> int:
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") % 4


def fidelity_records(path: Path, workload: str, alternative: str) -> list[dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    records: list[dict[str, Any]] = []
    full_scale_values = []
    for request_id, group in frame.groupby("request_id", sort=False):
        current = group["current_full512_logit"].to_numpy(dtype=np.float64)
        reference = group[alternative].to_numpy(dtype=np.float64)
        delta = current - reference
        row: dict[str, Any] = {
            "request_id": str(request_id), "uid": int(group["uid"].iloc[0]),
            "panel": request_panel(str(request_id)),
            "output_js_divergence": js_divergence(reference, current, workload),
        }
        if workload == "F":
            row["raw_delta"] = abs(float(delta[0]))
            row["absolute_probability_difference"] = abs(float(sigmoid(current[0])) - float(sigmoid(reference[0])))
            full_scale_values.append(float(current[0]))
        else:
            centered_delta = delta - delta.mean()
            centered_current = current - current.mean()
            row["normalized_score_RMS"] = float(
                np.sqrt(np.mean(centered_delta**2)) / (np.sqrt(np.mean(centered_current**2)) + 1e-6)
            )
            tau = kendalltau(reference, current, variant="b").statistic
            row["pairwise_inversion"] = 0.0 if not np.isfinite(tau) else float((1.0 - tau) / 2.0)
            top = min(10, len(current))
            a = set(np.argsort(-reference, kind="stable")[:top].tolist())
            b = set(np.argsort(-current, kind="stable")[:top].tolist())
            row["top10_overlap_loss"] = 1.0 - len(a & b) / top
        records.append(row)
    if workload == "F":
        denominator = max(float(np.std(full_scale_values)), 1e-3)
        for row in records:
            row["normalized_score_RMS"] = row.pop("raw_delta") / denominator
    return records


def rank_metrics(scores: np.ndarray, target: int) -> dict[str, float]:
    target_score = float(scores[target])
    others = np.delete(scores.astype(np.float64), target)
    auc = float(np.mean((target_score > others) + 0.5 * (target_score == others)))
    tied_before = int(np.sum((scores == target_score) & (np.arange(len(scores)) < target)))
    rank = 1 + int(np.sum(others > target_score)) + tied_before
    return {
        "pairwise_target_AUC": auc,
        "target_hard_negative_margin": target_score - float(np.max(others)),
        "NDCG_at_10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "HR_at_10": float(rank <= 10), "MRR": 1.0 / rank,
    }


def ranking_quality_records(path: Path, alternative: str) -> list[dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    records = []
    for request_id, group in frame.groupby("request_id", sort=False):
        target = int(group["target_index"].iloc[0])
        current = rank_metrics(group["current_full512_logit"].to_numpy(dtype=np.float64), target)
        reference = rank_metrics(group[alternative].to_numpy(dtype=np.float64), target)
        records.append({
            "request_id": str(request_id), "uid": int(group["uid"].iloc[0]),
            **{f"{key}_gain": current[key] - reference[key] for key in current},
        })
    return records


def feedback_gap_map(release: str) -> dict[str, bool]:
    requests = load_p7_requests(MANIFEST_ROOT, RAW_LISTENS, EVALUATION_SPLIT[release], "F", manifest_kind="quality")
    output = {}
    for row in requests:
        latest = int(np.max(row.history_timestamps)) if row.history_timestamps is not None and len(row.history_timestamps) else row.query_timestamp
        output[row.request_id] = row.query_timestamp - latest >= 3 * 86_400
    return output


def feedback_quality_records(path: Path, alternative: str, long_gap: dict[str, bool]) -> list[dict[str, Any]]:
    frame = pq.read_table(path).to_pandas()
    if not (frame.groupby("request_id").size() == 1).all():
        raise RuntimeError("F quality must have exactly one candidate per request")
    records = []
    for row in frame.itertuples(index=False):
        label = int(row.label)
        current_logit = float(row.current_full512_logit)
        reference_logit = float(getattr(row, alternative))
        current_probability = float(sigmoid(current_logit))
        reference_probability = float(sigmoid(reference_logit))
        current_loss = float(np.logaddexp(0.0, current_logit) - label * current_logit)
        reference_loss = float(np.logaddexp(0.0, reference_logit) - label * reference_logit)
        records.append({
            "request_id": str(row.request_id), "uid": int(row.uid), "label": label,
            "current_probability": current_probability, "reference_probability": reference_probability,
            "log_loss_gain": reference_loss - current_loss,
            "Brier_gain": (reference_probability - label) ** 2 - (current_probability - label) ** 2,
            "dislike_only_logloss_gain": reference_loss - current_loss if label == 0 else float("nan"),
            "prior_30m_same_item": bool(row.prior_30m_same_item),
            "latest_item": bool(row.latest_item), "is_organic": int(row.is_organic),
            "history_stratum_v2": str(row.feedback_history_stratum_v2),
            "long_gap_at_least_3d": bool(long_gap[str(row.request_id)]),
        })
    return records


def user_values(records: list[dict[str, Any]], key: str) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in records:
        value = float(row[key])
        if np.isfinite(value):
            grouped[int(row["uid"])].append(value)
    return np.asarray([np.mean(grouped[uid]) for uid in sorted(grouped)], dtype=np.float64)


def mean_summary(records: list[dict[str, Any]], key: str, namespace: tuple[object, ...]) -> tuple[dict, np.ndarray]:
    values = user_values(records, key)
    if not len(values):
        return {"point": None, "ci95": [None, None], "users": 0}, np.full(PER_SEED_REPLICATES, np.nan)
    rng = rng_for(*namespace, key)
    draws = values[rng.integers(0, len(values), size=(PER_SEED_REPLICATES, len(values)))].mean(axis=1)
    return {
        "point": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "users": len(values),
    }, draws


def classification_metrics(records: list[dict[str, Any]], user_multiplicity: dict[int, int] | None = None) -> dict[str, float]:
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    current = np.asarray([row["current_probability"] for row in records], dtype=np.float64)
    reference = np.asarray([row["reference_probability"] for row in records], dtype=np.float64)
    counts = defaultdict(int)
    for row in records:
        counts[int(row["uid"])] += 1
    weights = np.asarray([1.0 / counts[int(row["uid"])] for row in records], dtype=np.float64)
    if user_multiplicity is not None:
        weights *= np.asarray([user_multiplicity.get(int(row["uid"]), 0) for row in records])
    if weights.sum() == 0 or len(np.unique(labels[weights > 0])) < 2:
        return {"ROC_AUC_gain": float("nan"), "dislike_PR_AUC_gain": float("nan")}
    dislike = 1 - labels
    return {
        "ROC_AUC_gain": float(roc_auc_score(labels, current, sample_weight=weights) - roc_auc_score(labels, reference, sample_weight=weights)),
        "dislike_PR_AUC_gain": float(
            average_precision_score(dislike, 1.0 - current, sample_weight=weights)
            - average_precision_score(dislike, 1.0 - reference, sample_weight=weights)
        ),
    }


def classification_summaries(records: list[dict[str, Any]], namespace: tuple[object, ...]) -> tuple[dict, dict[str, np.ndarray]]:
    users = sorted({int(row["uid"]) for row in records})
    point = classification_metrics(records)
    draws = {key: np.empty(PER_SEED_REPLICATES) for key in point}
    rng = rng_for(*namespace, "classification")
    for index in range(PER_SEED_REPLICATES):
        selected = rng.integers(0, len(users), size=len(users))
        counts = np.bincount(selected, minlength=len(users))
        metrics = classification_metrics(records, {users[i]: int(n) for i, n in enumerate(counts) if n})
        for key in draws:
            draws[key][index] = metrics[key]
    return {
        key: {"point": value, "ci95": [float(np.nanquantile(draws[key], 0.025)), float(np.nanquantile(draws[key], 0.975))], "users": len(users)}
        for key, value in point.items()
    }, draws


def hierarchical(summaries: list[dict], draws: list[np.ndarray], namespace: tuple[object, ...]) -> dict:
    finite = [(summary, draw) for summary, draw in zip(summaries, draws, strict=True) if summary["point"] is not None]
    if not finite:
        return {"point_equal_seed_mean": None, "ci95_hierarchical": [None, None], "seed_points": []}
    summaries, draws = map(list, zip(*finite, strict=True))
    rng = rng_for(*namespace, "hierarchical")
    output = np.empty(HIERARCHICAL_REPLICATES)
    for index in range(HIERARCHICAL_REPLICATES):
        chosen = rng.integers(0, len(draws), size=len(draws))
        output[index] = np.mean([draws[i][rng.integers(0, len(draws[i]))] for i in chosen])
    return {
        "point_equal_seed_mean": float(np.mean([row["point"] for row in summaries])),
        "ci95_hierarchical": [float(np.nanquantile(output, 0.025)), float(np.nanquantile(output, 0.975))],
        "seed_points": [row["point"] for row in summaries],
    }


def cohort_report(records: list[dict[str, Any]]) -> dict:
    predicates = {
        "prior_30m_same_item": lambda row: row["prior_30m_same_item"],
        "non_prior_30m": lambda row: not row["prior_30m_same_item"],
        "latest_item": lambda row: row["latest_item"],
        "non_latest_item": lambda row: not row["latest_item"],
        "like": lambda row: row["label"] == 1,
        "dislike": lambda row: row["label"] == 0,
        "organic": lambda row: row["is_organic"] == 1,
        "recommendation_driven": lambda row: row["is_organic"] == 0,
        "long_gap_at_least_3d": lambda row: row["long_gap_at_least_3d"],
        "short_gap_under_3d": lambda row: not row["long_gap_at_least_3d"],
        **{name: (lambda row, name=name: row["history_stratum_v2"] == name) for name in ("recent_seen", "old_seen", "seen_only_before_512", "never_seen")},
    }
    output = {}
    for name, predicate in predicates.items():
        selected = [row for row in records if predicate(row)]
        values = user_values(selected, "log_loss_gain")
        output[name] = {
            "requests": len(selected), "users": len({row["uid"] for row in selected}),
            "log_loss_gain_equal_user": float(values.mean()) if len(values) else None,
        }
    return output


def artifact_path(seal: dict, model: str, seed: int, workload: str, view: str) -> Path:
    matches = [row for row in seal["artifacts"] if (row["model"], row["seed"], row["workload"], row["view"]) == (model, seed, workload, view)]
    if len(matches) != 1:
        raise RuntimeError(f"sealed artifact lookup failed: {(model, seed, workload, view)}")
    return ROOT / matches[0]["path"]


def evaluate_condition(seal: dict, model: str, workload: str, comparison: str, long_gap: dict[str, bool] | None = None) -> dict:
    alternative = COMPARISONS[comparison]
    fidelity_view = "fidelity_all_eligible" if workload == "R" else "fidelity"
    quality_view = "quality_rankable" if workload == "R" else "quality"
    per_seed = []
    summaries: dict[str, list[dict]] = defaultdict(list)
    draws: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed in SEEDS:
        fidelity = fidelity_records(artifact_path(seal, model, seed, workload, fidelity_view), workload, alternative)
        fkeys = ["output_js_divergence", "normalized_score_RMS"]
        fkeys += ["absolute_probability_difference"] if workload == "F" else ["pairwise_inversion", "top10_overlap_loss"]
        fsummary = {}
        for key in fkeys:
            summary, draw = mean_summary(fidelity, key, (seal["release"], model, workload, seed, comparison, "fidelity"))
            summary["P95_request"] = float(np.percentile([row[key] for row in fidelity], 95))
            summary["P99_request"] = float(np.percentile([row[key] for row in fidelity], 99))
            summary["panel_points"] = [
                float(user_values([row for row in fidelity if row["panel"] == panel], key).mean()) for panel in range(4)
            ]
            fsummary[key] = summary
            summaries[f"fidelity.{key}"].append(summary); draws[f"fidelity.{key}"].append(draw)
        quality_path = artifact_path(seal, model, seed, workload, quality_view)
        if workload == "F":
            quality = feedback_quality_records(quality_path, alternative, long_gap or {})
            qsummary = {}
            for key in ("log_loss_gain", "Brier_gain", "dislike_only_logloss_gain"):
                summary, draw = mean_summary(quality, key, (seal["release"], model, workload, seed, comparison, "quality"))
                qsummary[key] = summary; summaries[f"quality.{key}"].append(summary); draws[f"quality.{key}"].append(draw)
            cls, cls_draws = classification_summaries(quality, (seal["release"], model, seed, comparison))
            for key, summary in cls.items():
                qsummary[key] = summary; summaries[f"quality.{key}"].append(summary); draws[f"quality.{key}"].append(cls_draws[key])
            cohorts = cohort_report(quality)
        else:
            quality = ranking_quality_records(quality_path, alternative)
            qsummary, cohorts = {}, None
            for key in ("pairwise_target_AUC_gain", "target_hard_negative_margin_gain", "NDCG_at_10_gain", "HR_at_10_gain", "MRR_gain"):
                summary, draw = mean_summary(quality, key, (seal["release"], model, workload, seed, comparison, "quality"))
                qsummary[key] = summary; summaries[f"quality.{key}"].append(summary); draws[f"quality.{key}"].append(draw)
        admission = next(row["admitted"] for row in seal["runs"] if row["model"] == model and row["seed"] == seed)
        per_seed.append({"seed": seed, "admitted": admission, "fidelity": fsummary, "quality": qsummary, "cohorts": cohorts})
    aggregate = {key: hierarchical(summaries[key], draws[key], (seal["release"], model, workload, comparison, key)) for key in summaries}
    return {"model": model, "workload": workload, "comparison": comparison, "per_seed": per_seed, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=RELEASES, required=True)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seal_path = args.seal or ROOT / f"results/p8/{args.release}/raw_score_seal_v1.json"
    output = args.output or ROOT / f"results/p8/{args.release}/hs_adjudication_v1.json"
    seal = json.loads(seal_path.read_text())
    if seal["release"] != args.release or seal["metrics_computed"] is not False or seal["run_count"] != 6:
        raise RuntimeError("incomplete or mismatched raw seal")
    conditions = []
    long_gap = feedback_gap_map(args.release)
    for model, workloads in (("m0_f", ("F",)), ("m1", ("N", "R", "F"))):
        for workload in workloads:
            for comparison in ("H", "S"):
                conditions.append(evaluate_condition(seal, model, workload, comparison, long_gap))
    candidates = []
    for model in ("m0_f", "m1"):
        h = next(row for row in conditions if (row["model"], row["workload"], row["comparison"]) == (model, "F", "H"))
        s = next(row for row in conditions if (row["model"], row["workload"], row["comparison"]) == (model, "F", "S"))
        h_js = h["aggregate"]["fidelity.output_js_divergence"]
        s_js = s["aggregate"]["fidelity.output_js_divergence"]
        admitted = [row["admitted"] for row in s["per_seed"]]
        seed_s = s_js["seed_points"]
        gate = {
            "current_H_CI_above_floor": h_js["ci95_hierarchical"][0] > P7_NUMERIC_FLOOR,
            "S_CI_above_R0_floor": s_js["ci95_hierarchical"][0] > R0_FLOOR,
            "at_least_two_seed_S_points_above_floor": sum(value > R0_FLOOR for value in seed_s) >= 2,
            "all_seed_admission": all(admitted),
            "admitted_seed_count": sum(admitted),
        }
        candidates.append({"model": model, "gate": gate, "edge_staleness_candidate": all(value for key, value in gate.items() if key != "admitted_seed_count")})
    payload = {
        "status": "P8_H_S_adjudicated_stop_before_tomography_or_controller",
        "release": args.release, "contract_hash": seal["contract_hash"],
        "raw_seal_hash": p7.sha256_file(seal_path), "conditions": conditions,
        "edge_staleness_candidates": candidates,
        "all_seed_results_reported": True, "seed_filtering_by_H_or_S": False,
        "tomography_or_controller_authorized": False,
        "mandatory_caveat": "dislike-only logloss reported without retroactive gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "release": args.release, "candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()

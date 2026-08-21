#!/usr/bin/env python3
"""Fit and freeze P7.6 Base scorers; qualification remains sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.stats import rankdata

from hstu_kvcache.base_fitting import (
    FeatureScaler,
    equal_user_request_weights,
    fit_feature_scaler,
    fit_linear_base,
    linear_scores,
    objective_and_gradient,
    request_row_ids,
)
from hstu_kvcache.data import load_compact_index

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "data/manifests/p7_full_v1"
OUTPUT = ROOT / "results/p7/base_fit"
CONTRACT = ROOT / "configs/contracts/p7_6_base_fit_contract_v1.yaml"
DAY = 86_400
BLOCK = 45 * DAY
LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
FOLDS = (((0,), 1), ((0, 1), 2), ((0, 1, 2), 3))
FEATURE_NAMES = {
    "N": (
        "log1p_item_count",
        "log1p_artist_count",
        "log1p_item_recency_seconds",
        "log1p_artist_recency_seconds",
        "log1p_global_popularity_at_base_fit_cutoff",
        "item_history_missing",
        "artist_history_or_mapping_missing",
    ),
    "R": (
        "log1p_item_count",
        "log1p_artist_count",
        "log1p_item_recency_seconds",
        "log1p_artist_recency_seconds",
        "log1p_global_popularity_at_base_fit_cutoff",
        "log1p_causal_proposal_rank",
        "artist_missing",
    ),
    "F": (
        "log1p_item_count",
        "log1p_artist_count",
        "log1p_item_recency_seconds",
        "log1p_artist_recency_seconds",
        "log1p_global_popularity_at_base_fit_cutoff",
        "item_history_missing",
        "artist_history_or_mapping_missing",
    ),
}
KINDS = {"N": "quality", "R": "quality_rankable", "F": "quality"}
OBJECTIVES = {"N": "listwise", "R": "listwise", "F": "binary"}


def sha256_file(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            output.update(block)
    return output.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def concatenate(paths: list[Path]) -> pa.Table:
    return pa.concat_tables([pq.read_table(path) for path in paths])


@dataclass
class TaskData:
    workload: str
    split: str
    features: np.memmap
    request_ids: np.ndarray
    uids: np.ndarray
    timestamps: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    targets: np.ndarray | None
    labels: np.ndarray | None
    blocks: np.ndarray

    @property
    def row_request(self) -> np.ndarray:
        return request_row_ids(self.lengths)

    @property
    def objective(self) -> str:
        return OBJECTIVES[self.workload]

    @property
    def all_requests(self) -> np.ndarray:
        return np.ones(len(self.uids), dtype=bool)


def load_task_data(
    root: Path,
    split: str,
    workload: str,
    temporary: Path,
) -> TaskData:
    index_path = root / split / "manifest.index.json"
    index = load_compact_index(index_path)
    request_paths = [index_path.parent / row["path"] for row in index["request_shards"]]
    requests = concatenate(request_paths)
    mask = pc.and_(
        pc.equal(requests["workload"], workload),
        pc.equal(requests["manifest_kind"], KINDS[workload]),
    )
    requests = requests.filter(mask).sort_by([("candidate_offset", "ascending")])
    offsets = requests["candidate_offset"].to_numpy(zero_copy_only=False).astype(np.int64)
    lengths = requests["candidate_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    total_rows = int(lengths.sum())
    feature_path = temporary / f"{split}_{workload}_features.f32"
    features = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(total_rows, 7))
    output_cursor = 0
    request_cursor = 0
    shard_start = 0
    for shard in index["candidate_shards"]:
        shard_end = shard_start + int(shard["rows"])
        begin = request_cursor
        while request_cursor < len(offsets) and offsets[request_cursor] < shard_end:
            if offsets[request_cursor] < shard_start:
                raise AssertionError("candidate request offsets are not monotonic")
            if offsets[request_cursor] + lengths[request_cursor] > shard_end:
                raise AssertionError("candidate set crosses a physical shard")
            request_cursor += 1
        if request_cursor > begin:
            table = pq.read_table(index_path.parent / shard["path"], columns=["base_features"])
            for request_index in range(begin, request_cursor):
                local = int(offsets[request_index] - shard_start)
                count = int(lengths[request_index])
                values = np.asarray(table["base_features"].slice(local, count).to_pylist(), dtype=np.float32)
                features[output_cursor : output_cursor + count] = values
                output_cursor += count
        shard_start = shard_end
    if request_cursor != len(offsets) or output_cursor != total_rows:
        raise AssertionError("not every compact candidate range was reconstructed")
    features.flush()
    timestamps = requests["query_timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
    targets = None
    labels = None
    if workload in {"N", "R"}:
        targets = requests["target_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    else:
        labels = requests["label"].to_numpy(zero_copy_only=False).astype(np.int64)
    return TaskData(
        workload=workload,
        split=split,
        features=features,
        request_ids=np.asarray(requests["request_id"].to_pylist()),
        uids=requests["uid"].to_numpy(zero_copy_only=False).astype(np.int64),
        timestamps=timestamps,
        starts=np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64),
        lengths=lengths,
        targets=targets,
        labels=labels,
        blocks=(timestamps // BLOCK).astype(np.int8),
    )


def unregularized_loss(data: TaskData, mask: np.ndarray, scaler: FeatureScaler, parameters: np.ndarray) -> float:
    loss, _ = objective_and_gradient(
        parameters,
        features=data.features,
        starts=data.starts,
        lengths=data.lengths,
        row_request=data.row_request,
        uids=data.uids,
        request_mask=mask,
        scaler=scaler,
        objective=data.objective,
        l2=0.0,
        targets=data.targets,
        labels=data.labels,
    )
    return float(loss)


def fit_cross_validation(data: TaskData) -> tuple[float, list[dict]]:
    traces = []
    row_request = data.row_request
    for fold_index, (train_blocks, validation_block) in enumerate(FOLDS):
        train_mask = np.isin(data.blocks, train_blocks)
        validation_mask = data.blocks == validation_block
        scaler = fit_feature_scaler(data.features, row_request, train_mask)
        initial = None
        fold_results = {}
        for l2 in reversed(LAMBDAS):
            fitted = fit_linear_base(
                features=data.features,
                starts=data.starts,
                lengths=data.lengths,
                row_request=row_request,
                uids=data.uids,
                request_mask=train_mask,
                scaler=scaler,
                objective=data.objective,
                l2=l2,
                targets=data.targets,
                labels=data.labels,
                initial=initial,
            )
            if not fitted.success:
                raise RuntimeError(f"{data.workload} fold {fold_index} L2={l2} failed: {fitted.message}")
            initial = fitted.x
            validation_loss = unregularized_loss(data, validation_mask, scaler, fitted.x)
            fold_results[l2] = validation_loss
            traces.append(
                {
                    "workload": data.workload,
                    "fold": fold_index,
                    "train_blocks": list(train_blocks),
                    "validation_block": validation_block,
                    "l2": l2,
                    "validation_primary_loss": validation_loss,
                    "iterations": int(fitted.nit),
                    "function_evaluations": int(fitted.nfev),
                    "solver_message": str(fitted.message),
                    "train_users": len(set(data.uids[train_mask].tolist())),
                    "validation_users": len(set(data.uids[validation_mask].tolist())),
                    "fold_scaler_hash": digest(scaler.as_dict()),
                }
            )
    means = {
        l2: float(np.mean([row["validation_primary_loss"] for row in traces if row["l2"] == l2]))
        for l2 in LAMBDAS
    }
    selected = min(LAMBDAS, key=lambda value: (means[value], -value))
    for row in traces:
        row["equal_fold_mean_for_l2"] = means[row["l2"]]
        row["selected_l2"] = selected
    return selected, traces


def fit_single_feature_controls(
    data: TaskData,
    scaler: FeatureScaler,
    l2: float,
) -> tuple[dict, dict]:
    candidates = []
    mask = data.all_requests
    row_request = data.row_request
    for feature_index, feature_name in enumerate(FEATURE_NAMES[data.workload]):
        one_scaler = FeatureScaler(
            scaler.clip_low[feature_index : feature_index + 1],
            scaler.clip_high[feature_index : feature_index + 1],
            scaler.mean[feature_index : feature_index + 1],
            scaler.scale[feature_index : feature_index + 1],
        )
        values = data.features[:, feature_index : feature_index + 1]
        fitted = fit_linear_base(
            features=values,
            starts=data.starts,
            lengths=data.lengths,
            row_request=row_request,
            uids=data.uids,
            request_mask=mask,
            scaler=one_scaler,
            objective=data.objective,
            l2=l2,
            targets=data.targets,
            labels=data.labels,
        )
        if not fitted.success:
            raise RuntimeError(f"single-feature {data.workload}/{feature_name} failed")
        loss, _ = objective_and_gradient(
            fitted.x,
            features=values,
            starts=data.starts,
            lengths=data.lengths,
            row_request=row_request,
            uids=data.uids,
            request_mask=mask,
            scaler=one_scaler,
            objective=data.objective,
            l2=0.0,
            targets=data.targets,
            labels=data.labels,
        )
        candidates.append(
            {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "base_fit_primary_loss": float(loss),
                "coefficient": float(fitted.x[0]),
                "intercept": float(fitted.x[-1]) if data.objective == "binary" else 0.0,
                "iterations": int(fitted.nit),
            }
        )
    best = min(candidates, key=lambda row: row["base_fit_primary_loss"])
    return best, {"selection_source": "all_base_fit_only", "candidates": candidates}


def fit_task(data: TaskData) -> tuple[dict, list[dict], dict]:
    selected_l2, cv_trace = fit_cross_validation(data)
    row_request = data.row_request
    scaler = fit_feature_scaler(data.features, row_request, data.all_requests)
    fitted = fit_linear_base(
        features=data.features,
        starts=data.starts,
        lengths=data.lengths,
        row_request=row_request,
        uids=data.uids,
        request_mask=data.all_requests,
        scaler=scaler,
        objective=data.objective,
        l2=selected_l2,
        targets=data.targets,
        labels=data.labels,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"final {data.workload} fit failed: {fitted.message}")
    coefficients = fitted.x[:7]
    intercept = float(fitted.x[-1]) if data.objective == "binary" else 0.0
    single, single_trace = fit_single_feature_controls(data, scaler, selected_l2)
    artifact = {
        "workload": data.workload,
        "objective": data.objective,
        "selected_l2": selected_l2,
        "lambda_at_registered_grid_boundary": selected_l2 in {min(LAMBDAS), max(LAMBDAS)},
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "scaler": scaler.as_dict(),
        "feature_names": list(FEATURE_NAMES[data.workload]),
        "queries": len(data.uids),
        "users": len(set(data.uids.tolist())),
        "candidate_rows": int(data.lengths.sum()),
        "solver": {
            "success": bool(fitted.success),
            "iterations": int(fitted.nit),
            "function_evaluations": int(fitted.nfev),
            "message": str(fitted.message),
        },
        "single_feature_control": single,
    }
    if data.labels is not None:
        artifact["base_fit_label_prevalence"] = float(data.labels.mean())
    artifact["feature_schema_hash"] = digest(artifact["feature_names"])
    artifact["scaler_hash"] = digest(artifact["scaler"])
    artifact["parameter_hash"] = digest(
        {"coefficients": artifact["coefficients"], "intercept": intercept}
    )
    return artifact, cv_trace, single_trace


def subset_data(data: TaskData, request_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    pieces = [
        np.asarray(data.features[start : start + length], dtype=np.float64)
        for start, length in zip(data.starts[request_indices], data.lengths[request_indices], strict=True)
    ]
    features = np.concatenate(pieces)
    lengths = data.lengths[request_indices]
    starts = np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64)
    targets = data.targets[request_indices] if data.targets is not None else None
    labels = data.labels[request_indices] if data.labels is not None else None
    return features, starts, lengths, data.uids[request_indices], targets, labels


def equivalence_canary(base: TaskData, development: TaskData) -> dict:
    scores = np.asarray(
        [int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) for value in base.request_ids],
        dtype=np.uint64,
    )
    indices = np.sort(np.argsort(scores)[: min(128, len(scores))])
    sparse_mask = np.zeros(len(base.uids), dtype=bool)
    sparse_mask[indices] = True
    row_request = base.row_request
    scaler = fit_feature_scaler(base.features, row_request, sparse_mask)
    dimensions = 8 if base.objective == "binary" else 7
    probe = np.linspace(-0.15, 0.15, dimensions)
    streamed_loss, streamed_gradient = objective_and_gradient(
        probe,
        features=base.features,
        starts=base.starts,
        lengths=base.lengths,
        row_request=row_request,
        uids=base.uids,
        request_mask=sparse_mask,
        scaler=scaler,
        objective=base.objective,
        l2=0.01,
        targets=base.targets,
        labels=base.labels,
    )
    features, starts, lengths, uids, targets, labels = subset_data(base, indices)
    expanded_mask = np.ones(len(indices), dtype=bool)
    expanded_row_request = request_row_ids(lengths)
    expanded_loss, expanded_gradient = objective_and_gradient(
        probe,
        features=features,
        starts=starts,
        lengths=lengths,
        row_request=expanded_row_request,
        uids=uids,
        request_mask=expanded_mask,
        scaler=scaler,
        objective=base.objective,
        l2=0.01,
        targets=targets,
        labels=labels,
    )
    stream_fit = fit_linear_base(
        features=base.features,
        starts=base.starts,
        lengths=base.lengths,
        row_request=row_request,
        uids=base.uids,
        request_mask=sparse_mask,
        scaler=scaler,
        objective=base.objective,
        l2=0.01,
        targets=base.targets,
        labels=base.labels,
    )
    expanded_fit = fit_linear_base(
        features=features,
        starts=starts,
        lengths=lengths,
        row_request=expanded_row_request,
        uids=uids,
        request_mask=expanded_mask,
        scaler=scaler,
        objective=base.objective,
        l2=0.01,
        targets=targets,
        labels=labels,
    )
    dev_indices = np.arange(min(32, len(development.uids)))
    dev_features, *_ = subset_data(development, dev_indices)
    dimension = 7
    stream_scores = linear_scores(dev_features, scaler, stream_fit.x[:dimension])
    expanded_scores = linear_scores(dev_features, scaler, expanded_fit.x[:dimension])
    if base.objective == "binary":
        stream_scores += stream_fit.x[-1]
        expanded_scores += expanded_fit.x[-1]
    objective_delta = abs(streamed_loss - expanded_loss)
    gradient_delta = float(np.max(np.abs(streamed_gradient - expanded_gradient)))
    coefficient_delta = float(np.max(np.abs(stream_fit.x - expanded_fit.x)))
    score_delta = float(np.max(np.abs(stream_scores - expanded_scores)))
    passed = (
        objective_delta <= 1e-9
        and gradient_delta <= 1e-9
        and coefficient_delta <= 1e-7
        and score_delta <= 1e-7
    )
    if not passed:
        raise AssertionError(f"{base.workload} streaming equivalence failed")
    streamed_weights = equal_user_request_weights(base.uids, sparse_mask)[indices]
    expanded_weights = equal_user_request_weights(uids, expanded_mask)
    np.testing.assert_allclose(streamed_weights, expanded_weights, rtol=0.0, atol=0.0)
    return {
        "workload": base.workload,
        "requests": len(indices),
        "candidate_rows": len(features),
        "feature_rows_exact": True,
        "request_weights_exact": True,
        "objective_abs_delta": objective_delta,
        "gradient_max_abs_delta": gradient_delta,
        "coefficient_max_abs_delta": coefficient_delta,
        "development_score_max_abs_delta": score_delta,
        "passed": passed,
    }


def user_bootstrap(values: dict[int, list[float]], seed: int, draws: int = 1000) -> dict:
    user_values = np.asarray([np.mean(rows) for rows in values.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        bootstrap[draw] = rng.choice(user_values, size=len(user_values), replace=True).mean()
    return {
        "mean": float(user_values.mean()),
        "user_bootstrap_ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "users": len(user_values),
    }


def ranking_request_metrics(scores: np.ndarray, target: int) -> dict[str, float]:
    positive = float(scores[target])
    negatives = np.delete(scores, target)
    maximum = float(scores.max())
    ce = maximum + math.log(float(np.exp(scores - maximum).sum())) - positive
    greater = int((negatives > positive).sum())
    equal = int((negatives == positive).sum())
    rank = 1.0 + greater + 0.5 * equal
    return {
        "listwise_ce": ce,
        "within_request_pairwise_auc": float(
            ((positive > negatives).sum() + 0.5 * (positive == negatives).sum()) / len(negatives)
        ),
        "target_hard_negative_margin": positive - float(negatives.max()),
        "conditional_ndcg_at_10": 1.0 / math.log2(rank + 1.0) if rank <= 10 else 0.0,
        "hr_at_10": float(rank <= 10),
        "mrr": 1.0 / rank,
    }


def evaluate_ranking(
    data: TaskData,
    parameters: np.ndarray,
    scaler: FeatureScaler,
    seed: int,
) -> dict:
    row_scores = linear_scores(data.features, scaler, parameters[:7])
    per_metric: defaultdict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_scores = []
    for start, length, target, uid in zip(
        data.starts, data.lengths, data.targets, data.uids, strict=True
    ):
        scores = row_scores[start : start + length]
        all_scores.append(scores)
        for name, value in ranking_request_metrics(scores, int(target)).items():
            per_metric[name][int(uid)].append(value)
    return {
        "metrics": {
            name: user_bootstrap(values, seed + position)
            for position, (name, values) in enumerate(sorted(per_metric.items()))
        },
        "candidate_score_variance": float(np.var(np.concatenate(all_scores))),
    }


def random_ranking_control(data: TaskData, seed: int) -> dict:
    per_metric: defaultdict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for length, uid in zip(data.lengths, data.uids, strict=True):
        harmonic = sum(1.0 / rank for rank in range(1, int(length) + 1)) / int(length)
        ndcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, min(10, int(length)) + 1)) / int(length)
        values = {
            "listwise_ce": math.log(int(length)),
            "within_request_pairwise_auc": 0.5,
            "target_hard_negative_margin": 0.0,
            "conditional_ndcg_at_10": ndcg,
            "hr_at_10": min(10, int(length)) / int(length),
            "mrr": harmonic,
        }
        for name, value in values.items():
            per_metric[name][int(uid)].append(value)
    return {
        "metrics": {
            name: user_bootstrap(values, seed + position)
            for position, (name, values) in enumerate(sorted(per_metric.items()))
        },
        "candidate_score_variance": 0.0,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    positives = int(labels.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    probabilities = expit(scores)
    return {
        "log_loss": float(np.mean(np.logaddexp(0.0, scores) - labels * scores)),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc_like": average_precision(labels, scores),
        "pr_auc_dislike": average_precision(1 - labels, -scores),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def bootstrap_binary(
    data: TaskData,
    scores: np.ndarray,
    seed: int,
    mask: np.ndarray | None = None,
) -> dict:
    mask = np.ones(len(data.uids), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    labels = data.labels[mask]
    selected_scores = scores[mask]
    selected_uids = data.uids[mask]
    users = np.unique(selected_uids)
    positions = {uid: np.flatnonzero(selected_uids == uid) for uid in users}
    rng = np.random.default_rng(seed)
    names = tuple(binary_metrics(data.labels, scores))
    draws = {name: [] for name in names}
    for _ in range(1000):
        sampled = rng.choice(users, size=len(users), replace=True)
        indices = np.concatenate([positions[uid] for uid in sampled])
        values = binary_metrics(labels[indices], selected_scores[indices])
        for name in names:
            draws[name].append(values[name])
    point = binary_metrics(labels, selected_scores)
    return {
        name: {
            "value": point[name],
            "user_bootstrap_ci95": [
                float(np.nanpercentile(draws[name], 2.5)),
                float(np.nanpercentile(draws[name], 97.5)),
            ],
            "users": len(users),
        }
        for name in names
    }


def calibration_bins(labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    probabilities = expit(scores)
    assignments = np.minimum((probabilities * 10).astype(np.int64), 9)
    return [
        {
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "queries": int((assignments == index).sum()),
            "mean_probability": float(probabilities[assignments == index].mean())
            if (assignments == index).any()
            else None,
            "empirical_like_rate": float(labels[assignments == index].mean())
            if (assignments == index).any()
            else None,
        }
        for index in range(10)
    ]


def binary_stratum_summary(
    data: TaskData,
    scores: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict:
    probabilities = expit(scores)
    losses = np.logaddexp(0.0, scores) - data.labels * scores
    per_user_loss: defaultdict[int, list[float]] = defaultdict(list)
    per_user_probability: defaultdict[int, list[float]] = defaultdict(list)
    for uid, loss, probability in zip(
        data.uids[mask], losses[mask], probabilities[mask], strict=True
    ):
        per_user_loss[int(uid)].append(float(loss))
        per_user_probability[int(uid)].append(float(probability))
    return {
        "queries": int(mask.sum()),
        "users": len(per_user_loss),
        "log_loss": user_bootstrap(per_user_loss, seed),
        "mean_probability": user_bootstrap(per_user_probability, seed + 1),
    }


def evaluate_binary(
    data: TaskData,
    parameters: np.ndarray,
    scaler: FeatureScaler,
    seed: int,
    base_fit_prevalence: float,
) -> dict:
    scores = linear_scores(data.features, scaler, parameters[:7]) + float(parameters[-1])
    labels = data.labels
    development_prevalence = float(labels.mean())
    prevalence_logit = math.log(base_fit_prevalence / (1.0 - base_fit_prevalence))
    intercept_scores = np.full(len(labels), prevalence_logit)
    cohorts = {
        "like": labels == 1,
        "dislike": labels == 0,
    }
    return {
        "frozen_base": {
            "metrics": bootstrap_binary(data, scores, seed),
            "score_variance": float(np.var(scores)),
            "calibration": calibration_bins(labels, scores),
        },
        "prevalence_intercept": {
            "prevalence_from_base_fit": base_fit_prevalence,
            "development_prevalence": development_prevalence,
            "metrics": bootstrap_binary(data, intercept_scores, seed + 100),
            "score_variance": 0.0,
        },
        "label_strata": {
            name: binary_stratum_summary(data, scores, mask, seed + 200 + position * 10)
            for position, (name, mask) in enumerate(cohorts.items())
        },
        "scores": scores,
    }


def single_feature_parameters(artifact: dict) -> tuple[np.ndarray, FeatureScaler]:
    control = artifact["single_feature_control"]
    index = int(control["feature_index"])
    scaler_values = artifact["scaler"]
    scaler = FeatureScaler(
        np.asarray([scaler_values["clip_low"][index]]),
        np.asarray([scaler_values["clip_high"][index]]),
        np.asarray([scaler_values["mean"][index]]),
        np.asarray([scaler_values["scale"][index]]),
    )
    parameters = np.asarray([control["coefficient"], control["intercept"]])
    return parameters, scaler


def cohort_binary_metrics(
    data: TaskData,
    scores: np.ndarray,
    requests: pa.Table,
    seed: int,
) -> dict:
    definitions = {
        "prior_30m_same_item": np.asarray(requests["prior_30m_same_item"].to_pylist(), dtype=bool),
        "non_prior_30m": ~np.asarray(requests["prior_30m_same_item"].to_pylist(), dtype=bool),
        "latest_item": np.asarray(requests["latest_item"].to_pylist(), dtype=bool),
        "organic": np.asarray(requests["is_organic"].to_pylist()) == 1,
        "recommendation_driven": np.asarray(requests["is_organic"].to_pylist()) == 0,
    }
    return {
        name: {
            "queries": int(mask.sum()),
            "users": len(set(data.uids[mask].tolist())),
            "metrics": bootstrap_binary(data, scores, seed + position, mask),
        }
        for position, (name, mask) in enumerate(definitions.items())
        if mask.any()
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output already exists and is non-empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    qualification_index = args.manifest_root / "qualification/manifest.index.json"
    try:
        load_compact_index(qualification_index)
    except PermissionError:
        qualification_locked_before = True
    else:
        raise RuntimeError("qualification loader unexpectedly opened")

    temporary_path = Path(tempfile.mkdtemp(prefix="p7_base_fit_"))
    artifacts, cv_traces, single_traces, equivalences = {}, [], {}, []
    development_data = {}
    try:
        for workload in ("N", "R", "F"):
            base = load_task_data(args.manifest_root, "base_fit", workload, temporary_path)
            development = load_task_data(
                args.manifest_root, "development", workload, temporary_path
            )
            development_data[workload] = development
            equivalences.append(equivalence_canary(base, development))
            artifact, trace, single = fit_task(base)
            artifacts[workload] = artifact
            cv_traces.extend(trace)
            single_traces[workload] = single

        sanity = {"N": {}, "R": {}, "F": {}}
        for position, workload in enumerate(("N", "R")):
            data = development_data[workload]
            artifact = artifacts[workload]
            scaler = FeatureScaler(
                np.asarray(artifact["scaler"]["clip_low"]),
                np.asarray(artifact["scaler"]["clip_high"]),
                np.asarray(artifact["scaler"]["mean"]),
                np.asarray(artifact["scaler"]["scale"]),
            )
            parameters = np.asarray(artifact["coefficients"])
            frozen = evaluate_ranking(data, parameters, scaler, 7601 + position * 100)
            random = random_ranking_control(data, 7801 + position * 100)
            single_parameters, single_scaler = single_feature_parameters(artifact)
            feature_index = artifact["single_feature_control"]["feature_index"]
            single = evaluate_ranking(
                TaskData(
                    **{**data.__dict__, "features": data.features[:, feature_index : feature_index + 1]}
                ),
                single_parameters[:1],
                single_scaler,
                8001 + position * 100,
            )
            sanity[workload] = {
                "intercept_random": random,
                "best_single_feature": single,
                "frozen_base": frozen,
            }

        f_data = development_data["F"]
        f_artifact = artifacts["F"]
        f_scaler = FeatureScaler(
            np.asarray(f_artifact["scaler"]["clip_low"]),
            np.asarray(f_artifact["scaler"]["clip_high"]),
            np.asarray(f_artifact["scaler"]["mean"]),
            np.asarray(f_artifact["scaler"]["scale"]),
        )
        f_parameters = np.r_[f_artifact["coefficients"], f_artifact["intercept"]]
        f_evaluation = evaluate_binary(
            f_data,
            f_parameters,
            f_scaler,
            8201,
            f_artifact["base_fit_label_prevalence"],
        )
        f_single_parameters, f_single_scaler = single_feature_parameters(f_artifact)
        f_index = f_artifact["single_feature_control"]["feature_index"]
        f_single_scores = linear_scores(
            f_data.features[:, f_index : f_index + 1],
            f_single_scaler,
            f_single_parameters[:1],
        ) + f_single_parameters[-1]
        development_index = load_compact_index(
            args.manifest_root / "development/manifest.index.json"
        )
        request_paths = [
            args.manifest_root / "development" / row["path"]
            for row in development_index["request_shards"]
        ]
        f_requests = concatenate(request_paths)
        f_requests = f_requests.filter(
            pc.and_(
                pc.equal(f_requests["workload"], "F"),
                pc.equal(f_requests["manifest_kind"], "quality"),
            )
        ).sort_by([("candidate_offset", "ascending")])
        f_scores = f_evaluation.pop("scores")
        sanity["F"] = {
            **f_evaluation,
            "best_single_feature": {
                "feature": f_artifact["single_feature_control"]["feature_name"],
                "metrics": bootstrap_binary(f_data, f_single_scores, 8401),
                "score_variance": float(np.var(f_single_scores)),
            },
            "cohorts": cohort_binary_metrics(f_data, f_scores, f_requests, 8601),
        }

        r_index = json.loads(
            (args.manifest_root / "development/manifest.index.json").read_text()
        )
        r_all = r_index["views"]["R:fidelity_all_eligible"]["queries"]
        r_rankable = r_index["views"]["R:quality_rankable"]["queries"]
        sanity["R"]["population"] = {
            "all_eligible_session_starts": r_all,
            "rankable_familiar_returns": r_rankable,
            "conditional_coverage": r_rankable / r_all,
        }

        gates = {
            "streaming_equivalence": all(row["passed"] for row in equivalences),
            "all_parameters_finite": all(
                np.isfinite(row["coefficients"]).all() and math.isfinite(row["intercept"])
                for row in artifacts.values()
            ),
            "candidate_score_variance_positive": all(
                sanity[workload]["frozen_base"]["candidate_score_variance"] > 0
                for workload in ("N", "R")
            )
            and sanity["F"]["frozen_base"]["score_variance"] > 0,
            "base_beats_intercept_primary_dev_loss": all(
                sanity[workload]["frozen_base"]["metrics"]["listwise_ce"]["mean"]
                < sanity[workload]["intercept_random"]["metrics"]["listwise_ce"]["mean"]
                for workload in ("N", "R")
            )
            and sanity["F"]["frozen_base"]["metrics"]["log_loss"]["value"]
            < sanity["F"]["prevalence_intercept"]["metrics"]["log_loss"]["value"],
            "R_complete_universe_preserved": int(development_data["R"].lengths.max()) <= 512
            and int(development_data["R"].lengths.max()) > 500,
            "F_label_direction_valid": sanity["F"]["frozen_base"]["metrics"]["roc_auc"]["value"]
            > 0.5,
            "base_identical_recent32_full512_by_construction": True,
            "qualification_locked_before_and_after": qualification_locked_before,
        }
        try:
            load_compact_index(qualification_index)
        except PermissionError:
            gates["qualification_locked_before_and_after"] &= True
        else:
            gates["qualification_locked_before_and_after"] = False
        if not all(gates.values()):
            raise RuntimeError(f"P7.6 Base sanity gate failed: {gates}")

        contract_hash = sha256_file(CONTRACT)
        base_index_hash = sha256_file(args.manifest_root / "base_fit/manifest.index.json")
        dev_index_hash = sha256_file(args.manifest_root / "development/manifest.index.json")
        common = {
            "contract": "p7_6_base_fit_contract_v1",
            "contract_hash": contract_hash,
            "base_fit_manifest_hash": base_index_hash,
            "development_manifest_hash": dev_index_hash,
            "qualification_index_hash": sha256_file(qualification_index),
            "raw_source_hashes": json.loads(
                (args.manifest_root / "base_fit/manifest.index.json").read_text()
            )["raw_source_hashes"],
            "code_commit": git_commit(),
            "fitter_code_hash": sha256_file(Path(__file__)),
            "seed": 7601,
            "qualification_scored": False,
            "hstu_trained": False,
        }
        coverage = {
            **common,
            "tasks": {
                workload: {
                    "queries": artifact["queries"],
                    "users": artifact["users"],
                    "candidate_rows": artifact["candidate_rows"],
                }
                for workload, artifact in artifacts.items()
            },
            "R_development_population": sanity["R"]["population"],
        }
        equivalence = {**common, "status": "passed", "tasks": equivalences}
        cv = {
            **common,
            "lambda_grid": list(LAMBDAS),
            "folds": cv_traces,
            "single_feature_fit_traces": single_traces,
        }
        schema = {
            **common,
            "feature_schemas": {key: list(value) for key, value in FEATURE_NAMES.items()},
            "transforms": "clip_base_fit_quantiles_then_standardize_base_fit_mean_std",
        }
        parameters = {**common, "tasks": artifacts}
        dev_sanity = {
            **common,
            "status": "passed",
            "bootstrap_unit": "user",
            "bootstrap_draws": 1000,
            "tasks": sanity,
            "gates": gates,
        }
        outputs = {
            "base_fit_coverage_v1.json": coverage,
            "base_streaming_equivalence_v1.json": equivalence,
            "base_cv_trace_v1.json": cv,
            "base_feature_schema_v1.json": schema,
            "base_parameters_v1.json": parameters,
            "base_dev_sanity_v1.json": dev_sanity,
        }
        for workload, artifact in artifacts.items():
            outputs[f"base_{workload.lower()}_v1.json"] = {**common, **artifact}
        for name, value in outputs.items():
            write_json(args.output / name, value)

        bundle = args.output / "frozen_base_bundle_v1"
        bundle.mkdir()
        bundle_files = {}
        for workload in ("N", "R", "F"):
            source = args.output / f"base_{workload.lower()}_v1.json"
            destination = bundle / source.name
            shutil.copyfile(source, destination)
            bundle_files[destination.name] = sha256_file(destination)
        bundle_manifest = {
            **common,
            "status": "frozen_read_only_parameters",
            "files": bundle_files,
            "base_parameters_hash": sha256_file(args.output / "base_parameters_v1.json"),
            "feature_schema_hash": sha256_file(args.output / "base_feature_schema_v1.json"),
            "cv_trace_hash": sha256_file(args.output / "base_cv_trace_v1.json"),
            "development_sanity_hash": sha256_file(args.output / "base_dev_sanity_v1.json"),
            "qualification_scored": False,
        }
        write_json(bundle / "bundle_manifest.json", bundle_manifest)
        summary = {
            **common,
            "status": "p7_6_passed_frozen_base_only",
            "gates": gates,
            "selected_l2": {key: value["selected_l2"] for key, value in artifacts.items()},
            "bundle_manifest": str((bundle / "bundle_manifest.json").relative_to(ROOT)),
            "bundle_manifest_hash": sha256_file(bundle / "bundle_manifest.json"),
            "qualification_scored": False,
            "m0_m1_trained": False,
        }
        write_json(args.output / "p7_6_summary_v1.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        shutil.rmtree(temporary_path)


if __name__ == "__main__":
    main()

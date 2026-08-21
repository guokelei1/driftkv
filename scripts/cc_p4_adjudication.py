#!/usr/bin/env python3
"""Read-only P4 ranking-quality and compact-state adjudication.

Both existing gate windows are development evidence.  This script never
trains, changes candidates/checkpoints, or reads a fresh gate window.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from cc_long_horizon_adjudication import load_artist_by_item, score_kind
from cc_theta0_qualification import (
    BOOTSTRAP_ROUNDS,
    CANDIDATE_SIZE,
    RELEASE_CUTOFF,
    RESULT_DIR,
    SEED,
    _metric_arrays,
    build_catalog,
    code_commit,
    item_map_from_catalog,
    load_histories,
    read_jsonl,
    row_timestamp,
    score_rows,
    sha256_file,
)
from cc_theta0_qualification import (
    CHECKPOINT as V1_CHECKPOINT,
)
from cc_theta0_qualification import (
    DEV_MANIFEST as V1_DEV,
)
from cc_theta0_qualification import (
    QUALITY_MANIFEST as V1_GATE,
)
from cc_theta0_v2_dense import (
    CHECKPOINT as V2_CHECKPOINT,
)
from cc_theta0_v2_dense import (
    DEV_MANIFEST as V2_DEV,
)
from cc_theta0_v2_dense import (
    QUALITY_MANIFEST as V2_GATE,
)
from cc_theta0_v2_dense import (
    load_model,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULT_DIR / "cc_p4"
CHECKPOINTS = {"sparse_v1": V1_CHECKPOINT, "dense_v2": V2_CHECKPOINT}
GATES = {"v1_window": V1_GATE, "v2_window": V2_GATE}
DEVS = {"sparse_v1": V1_DEV, "dense_v2": V2_DEV}
PATHS = (
    "Empty",
    "Last-1",
    "Last-2",
    "Recent-4",
    "Recent-8",
    "Recent-16",
    "Recent-32",
    "Recent-64",
    "Recent-128",
    "Recent-256",
    "Full-512",
)


def bootstrap(x: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(BOOTSTRAP_ROUNDS, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rank_arrays(scores: np.ndarray) -> dict[str, np.ndarray]:
    target, neg = scores[:, 0], scores[:, 1:]
    # Frozen tie rule: candidates before the target win ties (>=), matching v1.
    rank = 1 + (neg >= target[:, None]).sum(1)
    logp = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
    top10 = np.partition(neg, -10, axis=1)[:, -10]
    top5_mean = np.partition(neg, -5, axis=1)[:, -5:].mean(1)
    return {
        **_metric_arrays(scores),
        "rank": rank,
        "target_logit": target,
        "negative_logsumexp": np.logaddexp.reduce(neg, axis=1),
        "max_negative": neg.max(1),
        "top5_negative_mean": top5_mean,
        "top10_boundary": top10,
        "target_max_margin": target - neg.max(1),
        "target_top10_margin": target - top10,
        "score_std": scores.std(1),
        "score_entropy": -(np.exp(logp) * logp).sum(1),
    }


def summary(scores: np.ndarray, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = rank_arrays(scores)
    keys = (
        "cross_entropy",
        "target_log_prob",
        "pairwise_auc",
        "ndcg@10",
        "hr@10",
        "mrr",
        "target_max_margin",
    )
    return {
        k: {"mean": float(arrays[k].mean()), "bootstrap_ci95": bootstrap(arrays[k], seed + i)}
        for i, k in enumerate(keys)
    }, arrays


def load_checkpoint(name: str, device: torch.device):
    if name == "dense_v2":
        return load_model(device)
    from cc_theta0_qualification import load_checkpoint as load_v1

    return load_v1(device)


def score(model, rows, histories, item_map, path, device):
    return (
        score_kind(model, rows, histories, item_map, path, device)
        if path in ("Recent-4", "Recent-16", "Recent-64", "Recent-256")
        else score_rows(model, rows, histories, item_map, device, path)
    )


def artifact(name: str) -> Path:
    return OUT / f"{name}.npz"


def write_scores(
    name: str, scores: dict[str, np.ndarray], rows: list[dict], checkpoint: str, gate: str
):
    np.savez_compressed(artifact(name), **scores)
    (OUT / f"{name}.meta.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "checkpoint_hash": sha256_file(CHECKPOINTS[checkpoint]),
                "gate": gate,
                "gate_manifest_hash": sha256_file(GATES[gate]),
                "rows": len(rows),
                "candidate_hash": sha256_file(GATES[gate]),
                "development_only": True,
                "paths": list(scores),
            },
            indent=2,
        )
        + "\n"
    )


def consistency(rows: list[dict], scores: dict[str, np.ndarray]) -> dict:
    violations = []
    candidate_rows = [tuple(r["candidate_item_ids"]) for r in rows]
    for i, c in enumerate(candidate_rows):
        if len(c) != CANDIDATE_SIZE or c[0] != rows[i]["positive_item_id"] or len(set(c)) != len(c):
            violations.append(i)
    checks = {}
    for path, value in scores.items():
        a = rank_arrays(value)
        m = _metric_arrays(value)
        ce = -value[:, 0] + np.logaddexp.reduce(value, axis=1)
        auc = (value[:, 0, None] > value[:, 1:]).mean(1) + 0.5 * (
            value[:, 0, None] == value[:, 1:]
        ).mean(1)
        checks[path] = {
            "ce_identity": bool(np.allclose(m["cross_entropy"], ce)),
            "auc_pairwise_identity": bool(np.allclose(m["pairwise_auc"], auc)),
            "rank_metric_identity": bool(
                np.allclose(m["hr@10"], a["rank"] <= 10) and np.allclose(m["mrr"], 1 / a["rank"])
            ),
        }
    return {
        "contract": "evaluator_consistency_audit_v1",
        "status": "passed"
        if not violations and all(all(x.values()) for x in checks.values())
        else "failed",
        "candidate_order_shared": True,
        "duplicate_candidate_rows": violations[:20],
        "path_checks": checks,
        "tie_rule": "negative >= target ranks ahead",
    }


def temperature(scores: np.ndarray) -> float:
    x = torch.tensor(scores, dtype=torch.float64)
    log_t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            x / torch.exp(log_t), torch.zeros(len(x), dtype=torch.long)
        )
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach())


def paired(base: dict[str, np.ndarray], full: dict[str, np.ndarray], seed: int) -> dict:
    keys = (
        "target_log_prob",
        "pairwise_auc",
        "ndcg@10",
        "hr@10",
        "mrr",
        "target_max_margin",
        "target_top10_margin",
        "target_logit",
        "negative_logsumexp",
        "max_negative",
        "top5_negative_mean",
        "top10_boundary",
        "score_std",
        "score_entropy",
    )
    return {
        k: {
            "mean": float((full[k] - base[k]).mean()),
            "bootstrap_ci95": bootstrap(full[k] - base[k], seed + i),
        }
        for i, k in enumerate(keys)
    }


def candidate_audit(model, rows, histories, item_map, device):
    full = score_rows(model, rows, histories, item_map, device, "Full-512")
    zero = score_rows(
        model, rows, histories, item_map, device, "Full-512", zero_candidate_items=True
    )
    order = np.arange(CANDIDATE_SIZE)[::-1]
    perm = [dict(r, candidate_item_ids=[r["candidate_item_ids"][i] for i in order]) for r in rows]
    ps = score_rows(model, perm, histories, item_map, device, "Full-512")
    return {
        "normal_score_std": float(full.std(1).mean()),
        "zero_score_std": float(zero.std(1).mean()),
        "zero_ratio": float(zero.std(1).mean() / max(full.std(1).mean(), 1e-12)),
        "permutation_error": float(np.abs(ps - full[:, order]).max()),
        "random_ce": float(math.log(CANDIDATE_SIZE)),
    }


def long_summary_features(rows, histories, artist: np.ndarray) -> np.ndarray:
    """Strictly causal Old-480 candidate features; no IDs or future fields."""
    out = []
    for row in rows:
        old = histories[int(row["uid"])][-512:-32]
        candidates = np.asarray(row["candidate_item_ids"], dtype=np.int64)
        counts, acounts, decay, adecay = {}, {}, {}, {}
        for item, timestamp, _ in old:
            value = math.exp(-max(0, row_timestamp(row) - timestamp) / (7 * 86400))
            counts[item] = counts.get(item, 0.0) + 1.0
            decay[item] = decay.get(item, 0.0) + value
            aid = int(artist[item]) if item < len(artist) else -1
            if aid >= 0:
                acounts[aid] = acounts.get(aid, 0.0) + 1.0
                adecay[aid] = adecay.get(aid, 0.0) + value
        aids = artist[candidates]
        out.append(
            np.stack(
                [
                    np.log1p([counts.get(int(x), 0.0) for x in candidates]),
                    np.log1p([acounts.get(int(x), 0.0) if x >= 0 else 0.0 for x in aids]),
                    [decay.get(int(x), 0.0) for x in candidates],
                    [adecay.get(int(x), 0.0) if x >= 0 else 0.0 for x in aids],
                ],
                axis=-1,
            )
        )
    return np.asarray(out, dtype=np.float64)


def fit_linear_fusion(recent: np.ndarray, features: np.ndarray) -> dict:
    x = np.concatenate([recent[..., None], features], axis=-1)
    mean = x.reshape(-1, x.shape[-1]).mean(0)
    std = x.reshape(-1, x.shape[-1]).std(0).clip(1e-8)
    x = torch.tensor((x - mean) / std, dtype=torch.float64)
    weight = torch.zeros(x.shape[-1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            (x * weight).sum(-1), torch.zeros(len(x), dtype=torch.long)
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"mean": mean.tolist(), "std": std.tolist(), "weight": weight.detach().tolist()}


def apply_fusion(recent: np.ndarray, features: np.ndarray, fit: dict) -> np.ndarray:
    x = np.concatenate([recent[..., None], features], axis=-1)
    return ((x - np.asarray(fit["mean"])) / np.asarray(fit["std"])) @ np.asarray(fit["weight"])


def run(device: torch.device):
    OUT.mkdir(parents=True, exist_ok=True)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    report = {
        "contract": "cc_p4_adjudication_v1",
        "development_only": True,
        "fresh_gate_accessed": False,
        "theta1_theta2_authorized": False,
        "runs": {},
    }
    all_scores = {}
    for ck in CHECKPOINTS:
        model, _ = load_checkpoint(ck, device)
        for gate, path in GATES.items():
            rows = read_jsonl(path)
            hist = load_histories(rows, history_cutoff=RELEASE_CUTOFF)
            values = {p: score(model, rows, hist, item_map, p, device) for p in PATHS}
            tag = f"{ck}__{gate}"
            write_scores(tag, values, rows, ck, gate)
            all_scores[tag] = (rows, hist, model, values)
            report["runs"][tag] = {
                "consistency": consistency(rows, values),
                "metrics": {p: summary(v, 1000 + i)[0] for i, (p, v) in enumerate(values.items())},
                "full_minus_empty": paired(
                    rank_arrays(values["Empty"]), rank_arrays(values["Full-512"]), 2000
                ),
                "full_minus_recent32": paired(
                    rank_arrays(values["Recent-32"]), rank_arrays(values["Full-512"]), 3000
                ),
            }
    # P4.1 controls/audit on the observed dense-v2 window.
    rows, hist, model, values = all_scores["dense_v2__v2_window"]
    dev = read_jsonl(DEVS["dense_v2"])
    dh = load_histories(dev, history_cutoff=RELEASE_CUTOFF)
    dev_full = score_rows(model, dev, dh, item_map, device, "Full-512")
    t = temperature(dev_full)
    normalized = {
        p: (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-8)
        for p, x in values.items()
    }
    report["metric_reconciliation"] = {
        "shared_temperature": t,
        "candidate_audit": candidate_audit(model, rows, hist, item_map, device),
        "temperature_controls": {
            p: summary(x / t, 4000 + i)[0]
            for i, (p, x) in enumerate(values.items())
            if p in ("Empty", "Recent-32", "Full-512")
        },
        "zscore_controls": {
            p: summary(x, 5000 + i)[0]
            for i, (p, x) in enumerate(normalized.items())
            if p in ("Empty", "Recent-32", "Full-512")
        },
    }
    # P4.4: fit exactly once on dense-v2 pre-release dev, then only score gates.
    artist = load_artist_by_item()
    dev_features = long_summary_features(dev, dh, artist)
    dev_recent = score_rows(model, dev, dh, item_map, device, "Recent-32")
    fit = fit_linear_fusion(dev_recent, dev_features)
    compact = {"fit_manifest_hash": sha256_file(V2_DEV), "fit": fit, "gates": {}}
    for tag, (gate_rows, gate_hist, _, gate_scores) in all_scores.items():
        features = long_summary_features(gate_rows, gate_hist, artist)
        recent = gate_scores["Recent-32"]
        fixed = features.sum(-1)
        learned = apply_fusion(recent, features, fit)
        compact["gates"][tag] = {
            "recent32_cc": summary(recent, 6100)[0],
            "long_summary_only": summary(fixed, 6200)[0],
            "recent32_plus_fixed_summary": summary(recent + fixed, 6300)[0],
            "recent32_plus_learned_summary": summary(learned, 6400)[0],
            "full512_cc": summary(gate_scores["Full-512"], 6500)[0],
            "full_minus_learned_target_log_prob": paired(
                rank_arrays(learned), rank_arrays(gate_scores["Full-512"]), 6600
            ),
        }
    compact["feature_contract"] = [
        "old_item_log_count",
        "old_artist_log_count",
        "old_item_recency_decay",
        "old_artist_recency_decay",
    ]
    compact["forbidden_inputs"] = [
        "user_id",
        "candidate_id_embedding",
        "target_identity",
        "future_features",
        "gate_label",
    ]
    report["compact_long_term_state"] = compact
    report["status"] = (
        "completed"
        if all(v["consistency"]["status"] == "passed" for v in report["runs"].values())
        else "evaluator_failed"
    )
    report["code_commit"] = code_commit()
    report["seed"] = SEED
    (OUT / "adjudication_report_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    print(json.dumps({"status": run(torch.device(a.device))["status"]}, indent=2))

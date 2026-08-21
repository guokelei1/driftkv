#!/usr/bin/env python3
"""P6 conditional identifiability adjudication.

No training, no quota/mix/loss change, and no unopened window.  P5 remains
a failed development protocol regardless of this audit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from cc_p5_seenmix_requalification import (
    MANIFEST_DIR,
    QUOTA_PATH,
    WINDOWS,
    assemble_seenmix,
    collect_records,
    ensure_catalog,
    load_cached_artist,
    load_quotas,
    write_json,
    write_jsonl,
)
from cc_theta0_qualification import (
    BOOTSTRAP_ROUNDS,
    DAY,
    QMAIN_POOL_SIZE,
    RESULT_DIR,
    SEED,
    _metric_arrays,
    code_commit,
    read_jsonl,
    row_timestamp,
    sha256_file,
)

from hstu_kvcache.data.identifiability import (
    IDENTIFIABILITY_FEATURES,
    MATCH_CALIPERS,
    MATCH_COMPETITORS,
    MATCH_FEATURES,
    MAX_ABS_SMD,
    MAX_MATCHED_CAUC,
    MIN_HOLDOUT_COMPLETE,
    MIN_STRATUM_COMPLETE,
    MISSING_PROPOSAL_RANK,
    grouped_folds,
    history_item_stats,
    identifiability_vector,
    ks_distance,
    match_feature_indices,
    nearest_within_caliper,
    request_conditional_metrics,
    select_caliper,
    standardized_mean_difference,
)

OUT = RESULT_DIR / "cc_p6"
FEATURE_NAMES = IDENTIFIABILITY_FEATURES
MATCH_IDX = match_feature_indices()
ABLATIONS = {
    "seen_flag": ("seen_flag",),
    "item_count": ("log1p_item_count",),
    "artist_count": ("log1p_artist_count",),
    "recency": ("last_seen_recency_days",),
    "popularity": ("log1p_global_popularity",),
    "proposal_rank": ("log_proposal_rank",),
    "all": FEATURE_NAMES,
}


def freeze_p5() -> dict:
    report = {
        "contract": "cc_p6_identifiability_v1",
        "stage": "P6.0",
        "cc_p5_seenmix_v1": {
            "quota_feasibility": "passed",
            "membership_shortcut_reduced": True,
            "primary_simple_feature_auc": 0.789,
            "primary_identifiability_gate": "failed",
            "matched_diagnostic": {
                "status": "invalid_for_conditional_identifiability",
                "reason": "coarse_bins_leave_continuous_count_recency_imbalance",
            },
            "cc_theta0_v3_authorized": False,
        },
        "cannot_relax_p5_ceilings": True,
        "cannot_change_quotas": True,
        "unopened_window_used": False,
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(OUT / "p5_freeze_v1.json", report)
    return report


def records_by_window():
    first_seen, catalog_order, catalog_order_times, _popularity = ensure_catalog()
    artist = load_cached_artist()
    return collect_records(first_seen, catalog_order, catalog_order_times, artist, _popularity)


def attach_stats(records: list[dict]) -> list[dict]:
    artist = load_cached_artist()
    popularity = ensure_catalog()[3]
    for rec in records:
        rec["item_stats"] = history_item_stats(rec["history"], artist)
        rec["popularity"] = popularity
        rec["artist"] = artist
    return records


def item_vector(rec: dict, item: int) -> np.ndarray:
    return np.asarray(
        identifiability_vector(
            int(item),
            query_ts=rec["timestamp"],
            stats=rec["item_stats"],
            popularity=rec["popularity"],
            qmain_ranks=rec["qmain_ranks"],
            artist_by_item=rec["artist"],
        ),
        dtype=np.float64,
    )


def stack_request_features(records: list[dict], rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_id = {(rec["uid"], rec["timestamp"]): rec for rec in records}
    aligned = []
    uids = []
    for row in rows:
        rec = by_id[(int(row["uid"]), row_timestamp(row))]
        aligned.append(np.stack([item_vector(rec, int(item)) for item in row["candidate_item_ids"]]))
        uids.append(int(row["uid"]))
    return np.asarray(aligned, dtype=np.float64), np.asarray(uids, dtype=np.int64)


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_linear(x: np.ndarray, mean: np.ndarray, std: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return (((x - mean) / std) * weight).sum(-1)


def fit_linear(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = zscore_fit(x)
    z = torch.tensor((x - mean) / std, dtype=torch.float64)
    weight = torch.zeros(z.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([weight], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy((z * weight).sum(-1), torch.zeros(len(z), dtype=torch.long))
        loss.backward()
        return loss

    opt.step(closure)
    return mean, std, weight.detach().numpy()


def global_binary_auc(scores: np.ndarray) -> float:
    target = scores[:, 0].ravel()
    negatives = scores[:, 1:].ravel()
    if len(target) == 0 or len(negatives) == 0:
        return float("nan")
    # Mann-Whitney / Wilcoxon: P(target > neg) + 0.5 P(eq)
    order = np.argsort(np.concatenate([target, negatives]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    target_rank_sum = ranks[: len(target)].sum()
    u = target_rank_sum - len(target) * (len(target) + 1) / 2.0
    return float(u / (len(target) * len(negatives)))


def metric_bundle(scores: np.ndarray) -> dict:
    arrays = _metric_arrays(scores)
    cond = request_conditional_metrics(scores.tolist())
    return {
        "request_cauc": cond["cauc"],
        "global_binary_auc": global_binary_auc(scores),
        "ndcg@10": cond["ndcg@10"],
        "hr@10": cond["hr@10"],
        "mrr": cond["mrr"],
        "target_rank_percentile": cond["target_rank_percentile"],
        "pairwise_auc_identity": float(np.isclose(cond["cauc"], arrays["pairwise_auc"].mean())),
    }


def mask_features(x: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    keep = np.zeros(x.shape[-1], dtype=np.float64)
    for name in names:
        keep[FEATURE_NAMES.index(name)] = 1.0
    return x * keep


def grouped_oof(x: np.ndarray, uids: np.ndarray, names: tuple[str, ...]) -> dict:
    masked = mask_features(x, names)
    oof = np.zeros(masked.shape[:2], dtype=np.float64)
    used = np.zeros(len(masked), dtype=bool)
    for train, test in grouped_folds(uids.tolist(), folds=5, seed=SEED):
        mean, std, weight = fit_linear(masked[train])
        oof[test] = apply_linear(masked[test], mean, std, weight)
        used[test] = True
    scores = oof[used]
    return {**metric_bundle(scores), "requests": int(used.sum()), "unique_uids": int(len(set(uids[used])))}


def time_transfer(train: np.ndarray, test: np.ndarray, names: tuple[str, ...]) -> dict:
    mean, std, weight = fit_linear(mask_features(train, names))
    scores = apply_linear(mask_features(test, names), mean, std, weight)
    return {**metric_bundle(scores), "weight": weight.tolist()}


def p6_1(windows: dict[str, list[dict]]) -> dict:
    attach_stats([rec for rows in windows.values() for rec in rows])
    quality = {
        name: read_jsonl(MANIFEST_DIR / f"cc_p5_{name}_quality_seenmix.jsonl") for name in WINDOWS
    }
    old_rows = {name: read_jsonl(path) for name, path in WINDOWS.items()}
    mix_x, mix_uid = {}, {}
    old_x, old_uid = {}, {}
    for name, records in windows.items():
        mix_x[name], mix_uid[name] = stack_request_features(records, quality[name])
        old_x[name], old_uid[name] = stack_request_features(records, old_rows[name])
    mix_all = np.concatenate([mix_x["v1_window"], mix_x["v2_window"]])
    mix_uids = np.concatenate([mix_uid["v1_window"], mix_uid["v2_window"]])
    old_all = np.concatenate([old_x["v1_window"], old_x["v2_window"]])
    old_uids = np.concatenate([old_uid["v1_window"], old_uid["v2_window"]])
    report = {
        "contract": "cc_p6_identifiability_v1",
        "stage": "P6.1",
        "split": "grouped_5fold_by_uid",
        "proposal_rank_definition": "causal_q_main_pool_rank_not_assembled_panel_slot",
        "p5_pairwise_auc_was_request_conditioned": True,
        "seenmix_grouped_oof": {name: grouped_oof(mix_all, mix_uids, cols) for name, cols in ABLATIONS.items()},
        "old_protocol_grouped_oof": {name: grouped_oof(old_all, old_uids, cols) for name, cols in ABLATIONS.items()},
        "seenmix_v1_to_v2": {name: time_transfer(mix_x["v1_window"], mix_x["v2_window"], cols) for name, cols in ABLATIONS.items()},
        "old_v1_to_v2": {name: time_transfer(old_x["v1_window"], old_x["v2_window"], cols) for name, cols in ABLATIONS.items()},
        "feature_names": list(FEATURE_NAMES),
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(OUT / "simple_feature_audit_v1.json", report)
    return report


def summarize_numeric(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "p50": float(np.median(array)) if len(array) else None,
    }


def request_profile(rec: dict, complete: bool) -> dict:
    last_ts = max((event[1] for event in rec["history"]), default=rec["timestamp"])
    organic = [event[2] == 1 for event in rec["history"]]
    return {
        "complete": complete,
        "prefix_length": len(rec["history"]),
        "recent_unique": rec["recent_n"],
        "old_unique": rec["old_n"],
        "last_activity_age": rec["timestamp"] - last_ts,
        "target_stratum": rec["target_stratum"],
        "organic_ratio": float(np.mean(organic)) if organic else None,
    }


def p6_2(windows: dict[str, list[dict]]) -> dict:
    quotas = load_quotas()
    profiles = []
    for records in windows.values():
        for rec in records:
            complete = (
                rec["recent_n"] >= quotas.m_recent
                and rec["old_n"] >= quotas.m_old
                and rec["discovery_n"] >= quotas.m_discovery
            )
            profiles.append(request_profile(rec, complete))
    retained = [row for row in profiles if row["complete"]]
    dropped = [row for row in profiles if not row["complete"]]

    def by_stratum(rows):
        counts = {}
        for row in rows:
            counts[row["target_stratum"]] = counts.get(row["target_stratum"], 0) + 1
        total = max(len(rows), 1)
        return {key: {"n": n, "share": n / total} for key, n in counts.items()}

    report = {
        "contract": "cc_p6_identifiability_v1",
        "stage": "P6.2",
        "quotas": quotas.__dict__,
        "retained": len(retained),
        "dropped": len(dropped),
        "dropped_rate": len(dropped) / max(len(profiles), 1),
        "retained_summary": {
            "prefix_length": summarize_numeric([row["prefix_length"] for row in retained]),
            "recent_unique": summarize_numeric([row["recent_unique"] for row in retained]),
            "old_unique": summarize_numeric([row["old_unique"] for row in retained]),
            "last_activity_age": summarize_numeric([row["last_activity_age"] for row in retained]),
            "organic_ratio": summarize_numeric(
                [row["organic_ratio"] for row in retained if row["organic_ratio"] is not None]
            ),
            "target_strata": by_stratum(retained),
        },
        "dropped_summary": {
            "prefix_length": summarize_numeric([row["prefix_length"] for row in dropped]),
            "recent_unique": summarize_numeric([row["recent_unique"] for row in dropped]),
            "old_unique": summarize_numeric([row["old_unique"] for row in dropped]),
            "last_activity_age": summarize_numeric([row["last_activity_age"] for row in dropped]),
            "organic_ratio": summarize_numeric(
                [row["organic_ratio"] for row in dropped if row["organic_ratio"] is not None]
            ),
            "target_strata": by_stratum(dropped),
        },
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(OUT / "dropped_cohort_audit_v1.json", report)
    return report


def stratum_universe(rec: dict) -> list[int]:
    if rec["target_stratum"] == "recent_seen":
        return list(rec["pools"].recent_items)
    if rec["target_stratum"] == "old_only":
        return list(rec["pools"].old_items)
    return list(rec["discovery"])


def scale_match_features(vectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (vectors[:, MATCH_IDX] - mean) / std


def fit_match_scaler(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for rec in records:
        rows.append(item_vector(rec, rec["target"])[list(MATCH_IDX)])
        for item in stratum_universe(rec):
            rows.append(item_vector(rec, item)[list(MATCH_IDX)])
    array = np.asarray(rows, dtype=np.float64)
    mean = array.mean(0)
    std = np.where(array.std(0) < 1e-8, 1.0, array.std(0))
    return mean, std


def match_one(rec: dict, mean: np.ndarray, std: np.ndarray, caliper: float):
    universe = stratum_universe(rec)
    if len(universe) < MATCH_COMPETITORS:
        return None
    target = scale_match_features(item_vector(rec, rec["target"])[None, :], mean, std)[0]
    cand = scale_match_features(np.stack([item_vector(rec, item) for item in universe]), mean, std)
    chosen = nearest_within_caliper(target.tolist(), cand.tolist(), k=MATCH_COMPETITORS, caliper=caliper)
    if len(chosen) < MATCH_COMPETITORS:
        return None
    items = [rec["target"]] + [universe[index] for index in chosen]
    features = np.stack([item_vector(rec, item) for item in items])
    return {
        "uid": rec["uid"],
        "timestamp": rec["timestamp"],
        "stratum": rec["target_stratum"],
        "items": items,
        "features": features,
    }


def match_balance(matched: list[dict]) -> dict:
    if not matched:
        return {
            "complete": 0,
            "stratum_complete": {"recent_seen": 0, "old_only": 0, "unseen": 0},
            "max_abs_smd": float("inf"),
            "mean_abs_smd": float("inf"),
            "smd": {},
            "ks": {},
        }
    smd = {}
    ks = {}
    for offset, name in enumerate(MATCH_FEATURES):
        target = [float(row["features"][0, FEATURE_NAMES.index(name)]) for row in matched]
        competitors = [
            float(value)
            for row in matched
            for value in row["features"][1:, FEATURE_NAMES.index(name)]
        ]
        smd[name] = standardized_mean_difference(target, competitors)
        ks[name] = ks_distance(target, competitors)
    abs_smd = [abs(value) for value in smd.values() if math.isfinite(value)]
    strata = {"recent_seen": 0, "old_only": 0, "unseen": 0}
    for row in matched:
        strata[row["stratum"]] += 1
    return {
        "complete": len(matched),
        "stratum_complete": strata,
        "max_abs_smd": max(abs_smd) if abs_smd else float("inf"),
        "mean_abs_smd": float(sum(abs_smd) / len(abs_smd)) if abs_smd else float("inf"),
        "smd": smd,
        "ks": ks,
    }


def p6_3(windows: dict[str, list[dict]]) -> dict:
    attach_stats([rec for rows in windows.values() for rec in rows])
    design = windows["v1_window"]
    holdout = windows["v2_window"]
    mean, std = fit_match_scaler(design)
    design_grid = []
    for caliper in MATCH_CALIPERS:
        matched = [row for rec in design if (row := match_one(rec, mean, std, caliper))]
        summary = match_balance(matched)
        summary["caliper"] = caliper
        design_grid.append(summary)
    chosen = select_caliper(design_grid)
    holdout_matched = [
        row for rec in holdout if (row := match_one(rec, mean, std, float(chosen["caliper"])))
    ]
    holdout_balance = match_balance(holdout_matched)
    design_matched = [
        row for rec in design if (row := match_one(rec, mean, std, float(chosen["caliper"])))
    ]
    if design_matched and holdout_matched:
        train = np.stack([row["features"] for row in design_matched])
        test = np.stack([row["features"] for row in holdout_matched])
        simple = time_transfer(train, test, FEATURE_NAMES)
        ablations = {name: time_transfer(train, test, cols) for name, cols in ABLATIONS.items()}
    else:
        simple, ablations = None, {}
    holdout_ok = (
        chosen["status"] == "balanced"
        and holdout_balance["complete"] >= MIN_HOLDOUT_COMPLETE
        and min(holdout_balance["stratum_complete"].values()) >= MIN_STRATUM_COMPLETE
        and holdout_balance["max_abs_smd"] <= MAX_ABS_SMD
        and simple is not None
        and simple["request_cauc"] <= MAX_MATCHED_CAUC
    )
    path = MANIFEST_DIR / "cc_p6_v2_window_continuous_matched.jsonl"
    write_jsonl(
        path,
        [
            {
                "uid": row["uid"],
                "target_timestamp": row["timestamp"],
                "positive_item_id": row["items"][0],
                "candidate_item_ids": row["items"],
                "target_stratum": row["stratum"],
                "protocol": "Q_continuous_matched_diag_v2",
                "target_injected": True,
                "development_only": True,
            }
            for row in holdout_matched
        ],
    )
    report = {
        "contract": "cc_p6_identifiability_v1",
        "stage": "P6.3",
        "protocol": "Q_continuous_matched_diag_v2",
        "not_a_serving_workload": True,
        "design_window": "v1_window",
        "holdout_window": "v2_window",
        "competitors_per_request": MATCH_COMPETITORS,
        "match_features": list(MATCH_FEATURES),
        "scaler_fit_on": "design_only",
        "caliper_selected_on": "design_balance_only",
        "design_grid": design_grid,
        "selected_caliper": chosen,
        "holdout_balance": holdout_balance,
        "holdout_simple_features": simple,
        "holdout_ablations": ablations,
        "holdout_manifest": str(path),
        "holdout_manifest_hash": sha256_file(path) if path.exists() else None,
        "gates": {
            "max_simple_feature_cauc": MAX_MATCHED_CAUC,
            "max_abs_smd": MAX_ABS_SMD,
            "min_complete": MIN_HOLDOUT_COMPLETE,
            "min_per_stratum": MIN_STRATUM_COMPLETE,
        },
        "status": "passed" if holdout_ok else "failed",
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(OUT / "continuous_matched_v2.json", report)
    return report


def p6_4(windows: dict[str, list[dict]]) -> dict:
    attach_stats([rec for rows in windows.values() for rec in rows])
    quotas = load_quotas()
    rows = []
    for name, records in windows.items():
        for rec in records:
            panel = assemble_seenmix(rec, quotas, False)
            rank = rec["qmain_ranks"].get(rec["target"], MISSING_PROPOSAL_RANK)
            retrieved_100 = rank <= 100
            qmain100 = []
            if retrieved_100:
                ordered = [item for item, _rank in sorted(rec["qmain_ranks"].items(), key=lambda kv: kv[1])]
                qmain100 = [rec["target"]] + [item for item in ordered[:100] if item != rec["target"]][:99]
            rows.append(
                {
                    "window": name,
                    "stratum": rec["target_stratum"],
                    "seenmix_no_inject_contains_target": rec["target"] in panel.item_ids,
                    "seenmix_complete": panel.complete,
                    "qmain_rank": rank,
                    "recall_at_100": retrieved_100,
                    "recall_at_1000": rec["target"] in rec["qmain_ranks"],
                    "qmain100": qmain100,
                    "rec": rec,
                }
            )

    def rate(mask) -> float:
        selected = [row for row, keep in zip(rows, mask) if keep]
        return float(np.mean([row["recall_at_100"] for row in selected])) if selected else float("nan")

    retrieved = [row for row in rows if row["qmain100"]]
    if retrieved:
        features = np.stack([np.stack([item_vector(row["rec"], item) for item in row["qmain100"]]) for row in retrieved])
        uids = np.asarray([row["rec"]["uid"] for row in retrieved], dtype=np.int64)
        retrieved_metrics = grouped_oof(features, uids, FEATURE_NAMES)
        by_stratum = {}
        for stratum in ("recent_seen", "old_only", "unseen"):
            idx = [index for index, row in enumerate(retrieved) if row["stratum"] == stratum]
            if len(idx) >= 20:
                by_stratum[stratum] = grouped_oof(features[idx], uids[idx], FEATURE_NAMES)
    else:
        retrieved_metrics, by_stratum = None, {}
    report = {
        "contract": "cc_p6_identifiability_v1",
        "stage": "P6.4",
        "seenmix_no_inject_contains_target_rate": float(
            np.mean([row["seenmix_no_inject_contains_target"] for row in rows])
        ),
        "seenmix_no_inject_note": "frozen generator excludes the current positive, so panel recall is structurally zero",
        "recall_at_100_qmain": {
            "all": float(np.mean([row["recall_at_100"] for row in rows])),
            **{
                stratum: float(np.mean([row["recall_at_100"] for row in rows if row["stratum"] == stratum]))
                for stratum in ("recent_seen", "old_only", "unseen")
            },
        },
        "recall_at_1000_qmain": {
            "all": float(np.mean([row["recall_at_1000"] for row in rows])),
            **{
                stratum: float(np.mean([row["recall_at_1000"] for row in rows if row["stratum"] == stratum]))
                for stratum in ("recent_seen", "old_only", "unseen")
            },
        },
        "naturally_retrieved_at_100": {
            "n": len(retrieved),
            "share": len(retrieved) / max(len(rows), 1),
            "simple_features": retrieved_metrics,
            "by_stratum": by_stratum,
        },
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(OUT / "no_injection_audit_v1.json", report)
    return report


def adjudicate(feature_audit, dropped, matched, injection) -> dict:
    grouped_ok = "seenmix_grouped_oof" in feature_audit and "all" in feature_audit["seenmix_grouped_oof"]
    matched_pass = matched.get("status") == "passed"
    retrieved = injection.get("naturally_retrieved_at_100", {})
    retrieved_cauc = (retrieved.get("simple_features") or {}).get("request_cauc")
    injection_anomaly = retrieved_cauc is not None and retrieved_cauc >= 0.95
    branch = "A_freeze_protocol_v2" if matched_pass and grouped_ok and not injection_anomaly else "B_yambda_next_item_nogo"
    report = {
        "contract": "cc_p6_identifiability_v1",
        "stage": "P6.5",
        "p5_rejudged": False,
        "cc_p5_seenmix_v1_primary_identifiability_gate": "failed",
        "grouped_and_request_conditioned_metrics_reported": grouped_ok,
        "continuous_matched_status": matched.get("status"),
        "no_injection_severe_anomaly": injection_anomaly,
        "dropped_rate": dropped.get("dropped_rate"),
        "branch": branch,
        "authorizes": {
            "cc_theta0_v3": False,
            "cc_theta1_theta2": False,
            "controller": False,
            "tomography": False,
            "theta3": False,
        },
        "reason": (
            "continuous matched diagnostic passed holdout gates; freeze protocol v2 and wait for explicit v3 launch"
            if branch.startswith("A")
            else "continuous match failed or coverage collapsed; stop Yambda next-item candidate engineering"
        ),
        "yambda_next_item_long_kv_qualification": (
            None
            if branch.startswith("A")
            else {"status": "no_go", "reason": "repeat_count_recency_entanglement"}
        ),
        "code_commit": code_commit(),
        "seed": SEED,
    }
    # Branch A still does not auto-train. The contract requires a new protocol
    # freeze plus an unopened window; that is a separate authorized step.
    write_json(OUT / "adjudication_report_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze", "features", "dropped", "matched", "injection", "all"),
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        print(json.dumps(freeze_p5(), indent=2))
        return
    freeze_p5()
    windows = records_by_window()
    if args.command == "features":
        print(json.dumps(p6_1(windows), indent=2))
        return
    if args.command == "dropped":
        print(json.dumps(p6_2(windows), indent=2))
        return
    if args.command == "matched":
        print(json.dumps(p6_3(windows), indent=2))
        return
    if args.command == "injection":
        print(json.dumps(p6_4(windows), indent=2))
        return
    features = p6_1(windows)
    dropped = p6_2(windows)
    matched = p6_3(windows)
    injection = p6_4(windows)
    decision = adjudicate(features, dropped, matched, injection)
    print(
        json.dumps(
            {
                "p6_1_seenmix_grouped_cauc": features["seenmix_grouped_oof"]["all"]["request_cauc"],
                "p6_1_old_grouped_cauc": features["old_protocol_grouped_oof"]["all"]["request_cauc"],
                "p6_2_dropped_rate": dropped["dropped_rate"],
                "p6_3_status": matched["status"],
                "p6_3_caliper": matched["selected_caliper"].get("caliper"),
                "p6_3_holdout_cauc": (matched.get("holdout_simple_features") or {}).get("request_cauc"),
                "p6_4_recall_at_100": injection["recall_at_100_qmain"]["all"],
                "branch": decision["branch"],
                "theta0_v3_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P5 seen-aware workload requalification.

Protocol stages never consult model scores when choosing quotas.  Old v1/v2
gates stay development diagnosis.  Training and the unopened window are
blocked until the shortcut audit passes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from cc_long_horizon_adjudication import load_artist_by_item
from cc_theta0_qualification import (
    BOOTSTRAP_ROUNDS,
    DAY,
    MAX_HISTORY,
    QMAIN_POOL_SIZE,
    RELEASE_CUTOFF,
    QUERY_ACTION_ID,
    QUERY_TYPE_ID,
    RESULT_DIR,
    SEED,
    _causal_pool,
    _metric_arrays,
    autocast_context,
    build_catalog,
    code_commit,
    item_map_from_catalog,
    read_jsonl,
    row_timestamp,
    sha256_file,
)
from cc_theta0_v2_dense import (
    CHECKPOINT as DENSE_CHECKPOINT,
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
from cc_theta0_qualification import (
    DEV_MANIFEST as V1_DEV,
)
from cc_theta0_qualification import (
    QUALITY_MANIFEST as V1_GATE,
)

from hstu_kvcache.data import (
    SeenMixQuotas,
    build_history_matched_panel,
    build_seenmix_panel,
    match_key,
    seenmix_quota_grid,
    split_seen_pools,
)
from hstu_kvcache.data.cc import SEENMIX_MIN_DISCOVERY, artist_familiarity

ROOT = Path(__file__).resolve().parents[1]
OUT = RESULT_DIR / "cc_p5"
MANIFEST_DIR = ROOT / "data/manifests"
CONTRACT = ROOT / "configs/contracts/cc_p5_seenmix_v1.yaml"
QUOTA_PATH = OUT / "quotas_frozen_v1.json"
FREEZE_PATH = OUT / "p4_freeze_v1.json"
FEASIBILITY_PATH = OUT / "feasibility_audit_v1.json"
SHORTCUT_PATH = OUT / "shortcut_audit_v1.json"
SCREEN_PATH = OUT / "dense_v2_protocol_screen_v1.json"
WINDOWS = {
    "v1_window": V1_GATE,
    "v2_window": V2_GATE,
}
DEV_WINDOWS = {
    "v1_window": V1_DEV,
    "v2_window": V2_DEV,
}
UNOPENED_UPDATE_INDEX = 2
MATCHED_SLOTS = 32
MIN_COVERAGE = 0.80
MIN_COMPLETE = 1500
MAX_SEENMIX_AUC = 0.75
MAX_MATCHED_AUC = 0.60
MIN_OLD_MINUS_SEENMIX = 0.05
WAIVE_OLD_AUC = 0.65
MIN_MATCHED_COMPLETE = 400


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def rng_for(*parts: object) -> random.Random:
    material = ":".join(str(part) for part in (SEED, *parts)).encode()
    return random.Random(int.from_bytes(__import__("hashlib").sha256(material).digest()[:8], "little"))


def freeze_p4() -> dict:
    report = {
        "contract": "cc_p5_seenmix_v1",
        "stage": "P5.0",
        "p4": {
            "evaluator_consistency": "passed",
            "likelihood_vs_topk_conflict": "real",
            "dense_training_regime_effect": "mixed",
            "candidate_protocol": {
                "status": "confounded",
                "confounder": "seen_membership_and_long_term_count_shortcut",
            },
            "compact_summary_result": {
                "status": "protocol_conditioned_diagnostic",
                "cannot_support_state_sufficiency_claim": True,
            },
            "cc_theta1_theta2_authorized": False,
        },
        "candidate_protocol_identifiability": "failed",
        "paper_hypothesis": "unresolved",
        "old_gates_remain_development_diagnosis": True,
        "cannot_restore_qualification_by_rescoring_old_gates": True,
        "authorizes": {
            "cc_theta1_theta2": False,
            "controller": False,
            "tomography": False,
            "theta3": False,
        },
        "authorization_chain": [
            "protocol_validity",
            "H",
            "S",
            "heterogeneity",
            "opportunity",
            "design",
        ],
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(FREEZE_PATH, report)
    return report


def load_window(path: Path, history_cutoff: int | None = None):
    from cc_theta0_qualification import load_histories

    rows = read_jsonl(path)
    histories = load_histories(rows, history_cutoff=history_cutoff)
    return rows, histories


def qmain_ordered_pool(history, timestamp, first_seen, catalog_order, catalog_order_times) -> list[int]:
    """Causal Q_main order without dropping the current target.

    Proposal rank must not be read from an assembled panel.  Quality
    manifests put the injected target at slot 0; using that position as a
    feature would make any linear diagnostic perfectly recover the label.
    """
    return [int(item) for item in _causal_pool(
        history, -1, timestamp, first_seen, catalog_order, catalog_order_times
    )]


def discovery_pool(row, history, first_seen, catalog_order, catalog_order_times, pools) -> list[int]:
    target = int(row.get("positive_item_id", 0) or 0)
    timestamp = row_timestamp(row)
    blocked = set(pools.recent_items) | set(pools.old_items)
    if target > 0:
        blocked.add(target)
    ordered = []
    seen = set()
    for item in qmain_ordered_pool(
        history, timestamp, first_seen, catalog_order, catalog_order_times
    ):
        if item in blocked or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def item_stats(history: list[tuple[int, int, int]], query_ts: int) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = {}
    for position, (item, timestamp, _behavior) in enumerate(history[-MAX_HISTORY:]):
        current = stats.setdefault(int(item), {"count": 0, "last_ts": int(timestamp), "last_pos": position})
        current["count"] += 1
        current["last_ts"] = int(timestamp)
        current["last_pos"] = position
    return stats


def query_record(row, history, first_seen, catalog_order, catalog_order_times, artist, popularity) -> dict:
    target = int(row["positive_item_id"])
    timestamp = row_timestamp(row)
    legal_history = [
        event for event in history if int(event[0]) < len(first_seen) and first_seen[int(event[0])] < RELEASE_CUTOFF
    ]
    pools = split_seen_pools(legal_history, target)
    discovery = discovery_pool(row, legal_history, first_seen, catalog_order, catalog_order_times, pools)
    qmain = qmain_ordered_pool(legal_history, timestamp, first_seen, catalog_order, catalog_order_times)
    qmain_ranks = {item: rank for rank, item in enumerate(qmain, start=1)}
    stats = item_stats(legal_history, timestamp)
    recent_artists = {
        int(artist[item])
        for item in pools.recent_items
        if item < len(artist) and int(artist[item]) >= 0
    }
    old_artists = {
        int(artist[item])
        for item in pools.old_items
        if item < len(artist) and int(artist[item]) >= 0
    }
    target_count = stats.get(target, {}).get("count", 0)
    target_last = stats.get(target, {}).get("last_ts")
    target_artist = int(artist[target]) if target < len(artist) else -1
    return {
        "uid": int(row["uid"]),
        "timestamp": timestamp,
        "target": target,
        "pools": pools,
        "discovery": discovery,
        "qmain_ranks": qmain_ranks,
        "stats": stats,
        "recent_n": len(pools.recent_items),
        "old_n": len(pools.old_items),
        "discovery_n": len(discovery),
        "target_stratum": pools.target_stratum,
        "target_count": target_count,
        "target_recency": None if target_last is None else max(0, timestamp - target_last),
        "target_popularity": int(popularity[target]) if target < len(popularity) else 0,
        "target_artist": target_artist,
        "recent_artists": recent_artists,
        "old_artists": old_artists,
        "source_row": row,
        "history": legal_history,
        "raw_history_len": len(history),
    }


def coverage_for(records: list[dict], quotas: SeenMixQuotas) -> dict:
    complete = [
        rec["recent_n"] >= quotas.m_recent
        and rec["old_n"] >= quotas.m_old
        and rec["discovery_n"] >= quotas.m_discovery
        for rec in records
    ]
    return {
        "quotas": {
            "m_recent": quotas.m_recent,
            "m_old": quotas.m_old,
            "m_discovery": quotas.m_discovery,
        },
        "requests": len(records),
        "complete": int(sum(complete)),
        "full_quota_coverage": float(np.mean(complete)) if records else 0.0,
        "incomplete_rate": float(1.0 - np.mean(complete)) if records else 1.0,
    }


def collect_records(first_seen, catalog_order, catalog_order_times, artist, popularity) -> dict[str, list[dict]]:
    from cc_theta0_qualification import load_histories

    collected = {}
    for name, path in WINDOWS.items():
        rows = read_jsonl(path)
        histories = load_histories(rows)
        collected[name] = [
            query_record(
                row,
                histories[int(row["uid"])],
                first_seen,
                catalog_order,
                catalog_order_times,
                artist,
                popularity,
            )
            for row in rows
        ]
    return collected


def summarize_pools(records: list[dict]) -> dict:
    def quantiles(values: list[int]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        return {
            key: float(np.quantile(array, q))
            for key, q in (("p0", 0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p100", 1))
        }

    strata = Counter(rec["target_stratum"] for rec in records)
    return {
        "requests": len(records),
        "target_strata": {key: {"n": n, "share": n / len(records)} for key, n in strata.items()},
        "recent_pool_unique": quantiles([rec["recent_n"] for rec in records]),
        "old_pool_unique": quantiles([rec["old_n"] for rec in records]),
        "discovery_pool_unique": quantiles([rec["discovery_n"] for rec in records]),
        "target_item_count": quantiles([rec["target_count"] for rec in records]),
        "target_popularity": quantiles([rec["target_popularity"] for rec in records]),
    }


def feasibility() -> dict:
    first_seen, catalog, popularity = build_catalog()
    catalog_order = catalog[np.argsort(first_seen[catalog], kind="stable")]
    catalog_order_times = first_seen[catalog_order]
    artist = load_artist_by_item()
    windows = collect_records(first_seen, catalog_order, catalog_order_times, artist, popularity)
    combined = [rec for rows in windows.values() for rec in rows]
    grid = []
    selected = None
    for quotas in seenmix_quota_grid():
        summary = coverage_for(combined, quotas)
        summary["windows"] = {name: coverage_for(rows, quotas) for name, rows in windows.items()}
        grid.append(summary)
        if (
            selected is None
            and summary["full_quota_coverage"] + 1e-12 >= MIN_COVERAGE
            and summary["complete"] >= MIN_COMPLETE
        ):
            selected = summary
    status = "passed" if selected is not None else "failed"
    report = {
        "contract": "cc_p5_seenmix_v1",
        "stage": "P5.1",
        "status": status,
        "development_only": True,
        "unopened_window_used": False,
        "model_scores_used": False,
        "selection_rule": "first_grid_point_meeting_coverage_descending_old_then_recent",
        "gates": {
            "min_full_quota_coverage": MIN_COVERAGE,
            "min_requests_after_complete_case": MIN_COMPLETE,
            "max_cross_stratum_backfill": 0.0,
            "min_discovery": SEENMIX_MIN_DISCOVERY,
        },
        "pool_summary": {name: summarize_pools(rows) for name, rows in windows.items()},
        "combined_pool_summary": summarize_pools(combined),
        "grid": grid,
        "selected": selected,
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(FEASIBILITY_PATH, report)
    if selected is not None:
        write_json(
            QUOTA_PATH,
            {
                "status": "frozen_from_coverage",
                "m_recent": selected["quotas"]["m_recent"],
                "m_old": selected["quotas"]["m_old"],
                "m_discovery": selected["quotas"]["m_discovery"],
                "full_quota_coverage": selected["full_quota_coverage"],
                "complete": selected["complete"],
                "source": str(FEASIBILITY_PATH),
            },
        )
    return report


def load_quotas() -> SeenMixQuotas:
    if not QUOTA_PATH.exists():
        raise FileNotFoundError("run feasibility before assembling manifests")
    payload = json.loads(QUOTA_PATH.read_text())
    if payload.get("status") != "frozen_from_coverage":
        raise RuntimeError("quotas were not frozen from the coverage rule")
    return SeenMixQuotas(
        m_recent=int(payload["m_recent"]),
        m_old=int(payload["m_old"]),
        m_discovery=int(payload["m_discovery"]),
    )


def panel_row(rec: dict, panel, protocol: str, extra: dict | None = None) -> dict:
    payload = {
        "sample_id": f"{protocol}-{rec['uid']}-{rec['timestamp']}",
        "uid": rec["uid"],
        "target_timestamp": rec["timestamp"],
        "query_timestamp": rec["timestamp"],
        "positive_item_id": rec["target"],
        "candidate_item_ids": list(panel.item_ids),
        "candidate_roles": list(panel.roles),
        "candidate_size": len(panel.item_ids),
        "target_stratum": rec["target_stratum"],
        "target_injected": panel.target_injected,
        "pre_injection_hit": panel.pre_injection_hit,
        "complete": panel.complete,
        "missing": panel.missing,
        "protocol": protocol,
        "query_type_id": QUERY_TYPE_ID,
        "query_action_id": QUERY_ACTION_ID,
        "proposal_cutoff": rec["timestamp"],
        "history_max_timestamp": max((event[1] for event in rec["history"]), default=-1),
    }
    if extra:
        payload.update(extra)
    return payload


def assemble_seenmix(rec: dict, quotas: SeenMixQuotas, inject: bool):
    return build_seenmix_panel(
        rec["pools"],
        rec["discovery"],
        quotas,
        rng=rng_for("seenmix", rec["uid"], rec["timestamp"], inject),
        target_item=rec["target"],
        inject_target=inject,
    )


def competitor_universe(rec: dict) -> tuple[list[int], list[tuple[str, str, str, str, str]]]:
    timestamp = rec["timestamp"]
    items = []
    keys = []
    recent_set = set(rec["pools"].recent_items)
    old_set = set(rec["pools"].old_items)
    for item in list(rec["pools"].recent_items) + list(rec["pools"].old_items) + rec["discovery"]:
        item = int(item)
        if item == rec["target"] or item in {existing for existing in items}:
            continue
        stats = rec["stats"].get(item, {})
        last_ts = stats.get("last_ts")
        if item in recent_set:
            stratum = "recent_seen"
        elif item in old_set:
            stratum = "old_only"
        else:
            stratum = "unseen"
        artist_id = int(load_cached_artist()[item]) if item < len(load_cached_artist()) else -1
        items.append(item)
        keys.append(
            match_key(
                stratum=stratum,
                item_count=int(stats.get("count", 0)),
                recency_seconds=None if last_ts is None else max(0, timestamp - last_ts),
                familiarity=artist_familiarity(artist_id, rec["recent_artists"], rec["old_artists"]),
                global_count=int(load_cached_popularity()[item]) if item < len(load_cached_popularity()) else 0,
            )
        )
    return items, keys


_ARTIST = None
_POPULARITY = None
_FIRST_SEEN = None
_CATALOG_ORDER = None
_CATALOG_ORDER_TIMES = None


def load_cached_artist():
    global _ARTIST
    if _ARTIST is None:
        _ARTIST = load_artist_by_item()
    return _ARTIST


def load_cached_popularity():
    ensure_catalog()
    return _POPULARITY


def ensure_catalog():
    global _FIRST_SEEN, _CATALOG_ORDER, _CATALOG_ORDER_TIMES, _POPULARITY
    if _FIRST_SEEN is None:
        first_seen, catalog, popularity = build_catalog()
        _FIRST_SEEN = first_seen
        _POPULARITY = popularity
        _CATALOG_ORDER = catalog[np.argsort(first_seen[catalog], kind="stable")]
        _CATALOG_ORDER_TIMES = first_seen[_CATALOG_ORDER]
    return _FIRST_SEEN, _CATALOG_ORDER, _CATALOG_ORDER_TIMES, _POPULARITY


def target_key(rec: dict) -> tuple[str, str, str, str, str]:
    return match_key(
        stratum=rec["target_stratum"],
        item_count=rec["target_count"],
        recency_seconds=rec["target_recency"],
        familiarity=artist_familiarity(rec["target_artist"], rec["recent_artists"], rec["old_artists"]),
        global_count=rec["target_popularity"],
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def manifests() -> dict:
    quotas = load_quotas()
    first_seen, catalog_order, catalog_order_times, popularity = ensure_catalog()
    artist = load_cached_artist()
    windows = collect_records(first_seen, catalog_order, catalog_order_times, artist, popularity)
    summary = {"contract": "cc_p5_seenmix_v1", "stage": "P5.2", "quotas": quotas.__dict__, "windows": {}}
    for name, records in windows.items():
        quality, fidelity, matched = [], [], []
        for rec in records:
            quality_panel = assemble_seenmix(rec, quotas, True)
            if quality_panel.complete:
                quality.append(panel_row(rec, quality_panel, "Q_quality_seenmix_v1"))
            fidelity_panel = assemble_seenmix(rec, quotas, False)
            if fidelity_panel.complete:
                fidelity.append(
                    panel_row(
                        rec,
                        fidelity_panel,
                        "Q_fidelity_seenmix_v1",
                        extra={"target_injected": False, "positive_item_id": rec["target"]},
                    )
                )
            universe, keys = competitor_universe(rec)
            matched_panel = build_history_matched_panel(
                rec["target"],
                target_key(rec),
                universe,
                keys,
                competitor_slots=MATCHED_SLOTS,
                rng=rng_for("matched", rec["uid"], rec["timestamp"]),
            )
            if matched_panel.complete:
                matched.append(panel_row(rec, matched_panel, "Q_history_matched_diag_v1"))
        quality_path = MANIFEST_DIR / f"cc_p5_{name}_quality_seenmix.jsonl"
        fidelity_path = MANIFEST_DIR / f"cc_p5_{name}_fidelity_seenmix.jsonl"
        matched_path = MANIFEST_DIR / f"cc_p5_{name}_matched_diag.jsonl"
        write_jsonl(quality_path, quality)
        write_jsonl(fidelity_path, fidelity)
        write_jsonl(matched_path, matched)
        summary["windows"][name] = {
            "source_requests": len(records),
            "quality_complete": len(quality),
            "fidelity_complete": len(fidelity),
            "matched_complete": len(matched),
            "quality_manifest": str(quality_path),
            "fidelity_manifest": str(fidelity_path),
            "matched_manifest": str(matched_path),
            "quality_hash": sha256_file(quality_path),
            "fidelity_hash": sha256_file(fidelity_path),
            "matched_hash": sha256_file(matched_path),
        }
    write_json(OUT / "manifests_v1.json", summary)
    return summary


def candidate_feature_row(item: int, rec: dict, proposal_rank: int | None) -> np.ndarray:
    stats = rec["stats"].get(int(item), {})
    last_ts = stats.get("last_ts")
    recency_days = 365.0 if last_ts is None else max(0, rec["timestamp"] - last_ts) / DAY
    artist = load_cached_artist()
    popularity = load_cached_popularity()
    artist_id = int(artist[item]) if item < len(artist) else -1
    artist_count = 0
    if artist_id >= 0:
        artist_count = sum(
            1
            for hist_item, _ts, _beh in rec["history"][-MAX_HISTORY:]
            if hist_item < len(artist) and int(artist[hist_item]) == artist_id
        )
    recent = int(item) in set(rec["pools"].recent_items) or (
        rec["target_stratum"] == "recent_seen" and int(item) == rec["target"]
    )
    old = int(item) in set(rec["pools"].old_items) or (
        rec["target_stratum"] == "old_only" and int(item) == rec["target"]
    )
    seen = recent or old
    rank = float(proposal_rank) if proposal_rank is not None else float(QMAIN_POOL_SIZE + 1)
    return np.asarray(
        [
            float(seen),
            float(recent),
            float(old),
            math.log1p(float(stats.get("count", 0))),
            math.log1p(float(artist_count)),
            recency_days,
            math.log1p(float(popularity[item]) if item < len(popularity) else 0.0),
            math.log(rank),
        ],
        dtype=np.float64,
    )


def stack_features(records: list[dict], rows: list[dict]) -> np.ndarray:
    by_id = {(rec["uid"], rec["timestamp"]): rec for rec in records}
    matrix = []
    for row in rows:
        rec = by_id[(int(row["uid"]), row_timestamp(row))]
        ranks = rec["qmain_ranks"]
        matrix.append(
            np.stack(
                [candidate_feature_row(int(item), rec, ranks.get(int(item))) for item in row["candidate_item_ids"]]
            )
        )
    return np.asarray(matrix, dtype=np.float64)


def fit_linear_auc(train: np.ndarray, test: np.ndarray) -> dict:
    x = torch.tensor(train, dtype=torch.float64)
    mean = x.reshape(-1, x.shape[-1]).mean(0)
    std = x.reshape(-1, x.shape[-1]).std(0).clamp(min=1e-8)
    xt = (x - mean) / std
    weight = torch.zeros(xt.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([weight], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy((xt * weight).sum(-1), torch.zeros(len(xt), dtype=torch.long))
        loss.backward()
        return loss

    opt.step(closure)
    fitted = weight.detach()
    ztest = (torch.tensor(test, dtype=torch.float64) - mean) / std
    scores = (ztest * fitted).sum(-1).numpy()
    arrays = _metric_arrays(scores)
    return {
        "weight": fitted.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "pairwise_auc": float(arrays["pairwise_auc"].mean()),
        "cross_entropy": float(arrays["cross_entropy"].mean()),
        "ndcg@10": float(arrays["ndcg@10"].mean()),
        "hr@10": float(arrays["hr@10"].mean()),
    }


def seen_rate(rows: list[dict], records: list[dict], role: str) -> float:
    by_id = {(rec["uid"], rec["timestamp"]): rec for rec in records}
    flags = []
    for row in rows:
        rec = by_id[(int(row["uid"]), row_timestamp(row))]
        seen = set(rec["pools"].recent_items) | set(rec["pools"].old_items)
        if rec["target_stratum"] != "unseen":
            seen.add(rec["target"])
        items = [int(item) for item in row["candidate_item_ids"]]
        if role == "target":
            flags.append(items[0] in seen)
        else:
            flags.extend(item in seen for item in items[1:])
    return float(np.mean(flags)) if flags else 0.0


def shortcut() -> dict:
    if not FEASIBILITY_PATH.exists():
        raise FileNotFoundError("run feasibility first")
    quotas = load_quotas()
    first_seen, catalog_order, catalog_order_times, popularity = ensure_catalog()
    artist = load_cached_artist()
    windows = collect_records(first_seen, catalog_order, catalog_order_times, artist, popularity)
    quality = {
        name: read_jsonl(MANIFEST_DIR / f"cc_p5_{name}_quality_seenmix.jsonl") for name in WINDOWS
    }
    matched = {
        name: read_jsonl(MANIFEST_DIR / f"cc_p5_{name}_matched_diag.jsonl") for name in WINDOWS
    }
    old_rows = {name: read_jsonl(path) for name, path in WINDOWS.items()}
    old_fit = fit_linear_auc(
        stack_features(windows["v1_window"], old_rows["v1_window"]),
        stack_features(windows["v2_window"], old_rows["v2_window"]),
    )
    mix_fit = fit_linear_auc(
        stack_features(windows["v1_window"], quality["v1_window"]),
        stack_features(windows["v2_window"], quality["v2_window"]),
    )
    matched_complete = len(matched["v1_window"]) >= MIN_MATCHED_COMPLETE and len(matched["v2_window"]) > 0
    matched_fit = (
        fit_linear_auc(
            stack_features(windows["v1_window"], matched["v1_window"]),
            stack_features(windows["v2_window"], matched["v2_window"]),
        )
        if matched_complete
        else None
    )
    delta = old_fit["pairwise_auc"] - mix_fit["pairwise_auc"]
    waive = old_fit["pairwise_auc"] < WAIVE_OLD_AUC
    hard = {
        "seenmix_auc_le_max": mix_fit["pairwise_auc"] <= MAX_SEENMIX_AUC,
        "matched_auc_le_max": bool(matched_fit and matched_fit["pairwise_auc"] <= MAX_MATCHED_AUC),
        "old_minus_seenmix_ge_min_or_waived": delta + 1e-12 >= MIN_OLD_MINUS_SEENMIX or waive,
        "matched_complete_enough": matched_complete,
        "coverage_already_passed": json.loads(FEASIBILITY_PATH.read_text())["status"] == "passed",
    }
    report = {
        "contract": "cc_p5_seenmix_v1",
        "stage": "P5.3",
        "status": "passed" if all(hard.values()) else "failed",
        "development_only": True,
        "unopened_window_used": False,
        "quotas": quotas.__dict__,
        "proposal_rank_definition": "causal_q_main_pool_rank_not_assembled_panel_slot",
        "simple_feature_auc": {
            "old_qmain_v1_to_v2": old_fit,
            "quality_seenmix_v1_to_v2": mix_fit,
            "matched_diag_v1_to_v2": matched_fit,
            "old_minus_seenmix": delta,
        },
        "composition": {
            name: {
                "old_target_seen_rate": seen_rate(old_rows[name], windows[name], "target"),
                "old_competitor_seen_rate": seen_rate(old_rows[name], windows[name], "competitor"),
                "seenmix_target_seen_rate": seen_rate(quality[name], windows[name], "target"),
                "seenmix_competitor_seen_rate": seen_rate(quality[name], windows[name], "competitor"),
                "quality_complete": len(quality[name]),
                "matched_complete": len(matched[name]),
            }
            for name in WINDOWS
        },
        "hard_conditions": hard,
        "theta1_theta2_authorized": False,
        "v3_training_authorized": bool(all(hard.values())),
        "code_commit": code_commit(),
        "seed": SEED,
    }
    write_json(SHORTCUT_PATH, report)
    return report


def long_summary_scores(rows: list[dict], histories: dict[int, list], artist: np.ndarray) -> np.ndarray:
    out = []
    for row in rows:
        history = histories[int(row.get("history_key", row["uid"]))][-MAX_HISTORY:]
        old = history[:-32]
        counts: dict[int, float] = {}
        acounts: dict[int, float] = {}
        for item, _timestamp, _behavior in old:
            counts[item] = counts.get(item, 0.0) + 1.0
            aid = int(artist[item]) if item < len(artist) else -1
            if aid >= 0:
                acounts[aid] = acounts.get(aid, 0.0) + 1.0
        candidates = np.asarray(row["candidate_item_ids"], dtype=np.int64)
        aids = artist[candidates] if candidates.max(initial=0) < len(artist) else np.full(len(candidates), -1)
        out.append(
            np.log1p([counts.get(int(item), 0.0) for item in candidates])
            + np.log1p([acounts.get(int(aid), 0.0) if aid >= 0 else 0.0 for aid in aids])
        )
    return np.asarray(out, dtype=np.float64)


def collate_variable(rows, histories, item_map, path, device):
    from cc_theta0_qualification import _history_arrays

    arrays = [
        _history_arrays(
            histories[int(row.get("history_key", row["uid"]))],
            item_map,
            row_timestamp(row),
            path,
            int(row["uid"]),
        )
        for row in rows
    ]
    width = max(len(value[0]) for value in arrays)
    cand_w = len(rows[0]["candidate_item_ids"])
    items = np.zeros((len(rows), width), dtype=np.int64)
    behaviors = np.zeros_like(items)
    deltas = np.zeros((len(rows), width), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    query_deltas = np.zeros(len(rows), dtype=np.float32)
    candidates = np.zeros((len(rows), cand_w), dtype=np.int64)
    for index, (row, values) in enumerate(zip(rows, arrays, strict=True)):
        item_ids, behavior_ids, time_deltas, length, query_delta = values
        items[index, :length] = item_ids[:length]
        behaviors[index, :length] = behavior_ids[:length]
        deltas[index, :length] = time_deltas[:length]
        lengths[index] = length
        query_deltas[index] = query_delta
        mapped = []
        for item in row["candidate_item_ids"]:
            item = int(item)
            if item not in item_map:
                raise KeyError(f"catalog-illegal candidate {item}")
            mapped.append(item_map[item])
        candidates[index] = mapped
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (items, behaviors, deltas, lengths, candidates, query_deltas)
    )


def score_variable(model, rows, histories, item_map, path, device, batch_size=16) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        items, behaviors, deltas, lengths, candidates, query_deltas = collate_variable(
            batch, histories, item_map, path, device
        )
        with torch.inference_mode(), autocast_context(device):
            scores = model.score_cc_full(
                items, behaviors, deltas, candidates, query_deltas, lengths=lengths
            )
        outputs.append(scores.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def summarize_scores(scores: np.ndarray, seed: int) -> dict:
    arrays = _metric_arrays(scores)
    rng = np.random.default_rng(seed)

    def ci(values):
        means = values[rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))].mean(1)
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    return {
        key: {"mean": float(values.mean()), "bootstrap_ci95": ci(values)}
        for key, values in arrays.items()
    }


def screen(device: torch.device) -> dict:
    if not SHORTCUT_PATH.exists():
        raise FileNotFoundError("run shortcut audit first")
    from cc_theta0_qualification import load_histories

    model, _saved = load_model(device)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    artist = load_cached_artist()
    report = {
        "contract": "cc_p5_seenmix_v1",
        "stage": "P5.4",
        "status": "completed_development_only",
        "qualification_evidence": False,
        "checkpoint": str(DENSE_CHECKPOINT),
        "checkpoint_hash": sha256_file(DENSE_CHECKPOINT),
        "windows": {},
        "theta1_theta2_authorized": False,
        "code_commit": code_commit(),
        "seed": SEED,
    }
    paths = ("Empty", "Recent-32", "Full-512")
    for name in WINDOWS:
        rows = read_jsonl(MANIFEST_DIR / f"cc_p5_{name}_quality_seenmix.jsonl")
        histories = {
            uid: [event for event in events if event[0] in item_map]
            for uid, events in load_histories(rows).items()
        }
        scores = {path: score_variable(model, rows, histories, item_map, path, device) for path in paths}
        compact = long_summary_scores(rows, histories, artist)
        fused = scores["Recent-32"] + compact
        report["windows"][name] = {
            "rows": len(rows),
            "metrics": {path: summarize_scores(value, 7000 + i) for i, (path, value) in enumerate(scores.items())},
            "compact_summary": summarize_scores(compact, 7100),
            "recent32_plus_compact": summarize_scores(fused, 7200),
            "full_minus_recent32": {
                key: {
                    "mean": float(( _metric_arrays(scores["Full-512"])[key] - _metric_arrays(scores["Recent-32"])[key]).mean())
                }
                for key in ("target_log_prob", "pairwise_auc", "ndcg@10", "hr@10", "mrr")
            },
            "full_minus_compact": {
                "target_log_prob_mean": float(
                    (_metric_arrays(scores["Full-512"])["target_log_prob"] - _metric_arrays(compact)["target_log_prob"]).mean()
                )
            },
        }
    write_json(SCREEN_PATH, report)
    return report


def v3_authorized() -> bool:
    if not SHORTCUT_PATH.exists():
        return False
    payload = json.loads(SHORTCUT_PATH.read_text())
    return bool(payload.get("v3_training_authorized"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze", "feasibility", "manifests", "shortcut", "protocol", "screen", "status"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        print(json.dumps(freeze_p4(), indent=2))
        return
    if args.command == "feasibility":
        freeze_p4()
        print(json.dumps(feasibility(), indent=2))
        return
    if args.command == "manifests":
        print(json.dumps(manifests(), indent=2))
        return
    if args.command == "shortcut":
        print(json.dumps(shortcut(), indent=2))
        return
    if args.command == "protocol":
        freeze_p4()
        feasible = feasibility()
        print(json.dumps({"feasibility": feasible["status"], "selected": feasible.get("selected")}, indent=2))
        if feasible["status"] != "passed":
            raise SystemExit("P5.1 coverage gate failed; quotas not frozen")
        built = manifests()
        print(json.dumps({"manifests": built["windows"]}, indent=2))
        audited = shortcut()
        print(json.dumps({"shortcut": audited["status"], "v3_training_authorized": audited["v3_training_authorized"]}, indent=2))
        return
    if args.command == "screen":
        print(json.dumps(screen(torch.device(args.device)), indent=2))
        return
    if args.command == "status":
        print(
            json.dumps(
                {
                    "freeze": FREEZE_PATH.exists(),
                    "feasibility": FEASIBILITY_PATH.exists(),
                    "quotas": QUOTA_PATH.exists(),
                    "shortcut": SHORTCUT_PATH.exists(),
                    "v3_training_authorized": v3_authorized(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

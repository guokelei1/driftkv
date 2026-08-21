#!/usr/bin/env python3
"""P9.1 label-free user-level H/S distribution from sealed P8 F fidelity rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import train_p7_theta0 as p7
import yaml

import adjudicate_p8_hs as hs

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p9_tomography_contract_v1.yaml"
EVIDENCE = ROOT / "results/p9/p8_evidence_seal_v1.json"
OUTPUT = ROOT / "results/p9/p9_1_hs_distribution_v2.json"
RAW_ROOT = ROOT / "results/p8/staleness_raw"


def rng(*parts: object) -> np.random.Generator:
    token = "p9-distribution-v1|" + "|".join(map(str, parts))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big"))


def segment(value: int, cuts: tuple[int, ...]) -> str:
    for cut in cuts:
        if value <= cut:
            return f"le_{cut}"
    return f"gt_{cuts[-1]}"


def request_records(path: Path, release_cutover: int) -> list[dict]:
    frame = pq.read_table(path).to_pandas()
    records = []
    for request_id, group in frame.groupby("request_id", sort=False):
        if len(group) != 1:
            raise RuntimeError(f"F fidelity must have one candidate: {path}")
        row = group.iloc[0]
        full = np.asarray([row.current_full512_logit], dtype=np.float64)
        recent = np.asarray([row.current_recent32_logit], dtype=np.float64)
        reuse = np.asarray([row.reuse_parent_kv_logit], dtype=np.float64)
        h = hs.js_divergence(recent, full, "F")
        s = hs.js_divergence(reuse, full, "F")
        records.append({
            "uid": int(row.uid), "request_id": str(request_id), "H_js": h, "S_js": s,
            "H_score_rms": abs(float(full[0] - recent[0])) / 1e-3,
            "S_score_rms": abs(float(full[0] - reuse[0])) / 1e-3,
            "H_probability_shift": abs(float(hs.sigmoid(full[0])) - float(hs.sigmoid(recent[0]))),
            "S_probability_shift": abs(float(hs.sigmoid(full[0])) - float(hs.sigmoid(reuse[0]))),
            "prefix_tokens": int(row.prefix_tokens_at_cutover),
            "suffix_tokens": int(row.suffix_tokens_after_cutover),
            "state_age_seconds": int(row.query_timestamp) - release_cutover,
        })
    return records


def user_mean(records: list[dict], key: str) -> np.ndarray:
    values: dict[int, list[float]] = defaultdict(list)
    for row in records:
        values[row["uid"]].append(float(row[key]))
    return np.asarray([np.mean(values[uid]) for uid in sorted(values)], dtype=np.float64)


def summary(values: np.ndarray) -> dict:
    return {
        "users": int(len(values)), "mean_equal_user": float(np.mean(values)),
        "P50_equal_user": float(np.percentile(values, 50)), "P90_equal_user": float(np.percentile(values, 90)),
        "P95_equal_user": float(np.percentile(values, 95)), "P99_equal_user": float(np.percentile(values, 99)),
    }


def cohort_summary(records: list[dict], key: str, cohort_key: str, cuts: tuple[int, ...]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[segment(int(row[cohort_key]), cuts)].append(row)
    return {name: {"requests": len(rows), **summary(user_mean(rows, key))} for name, rows in sorted(groups.items())}


def joint_summary(records: list[dict]) -> dict:
    h = np.asarray([row["H_js"] for row in records], dtype=np.float64)
    s = np.asarray([row["S_js"] for row in records], dtype=np.float64)
    # A non-strict >= own-P90 predicate collapses to 100% when an identity
    # control has a zero-valued P90.  Use the frozen P8 numeric floor and a
    # strict comparison so the tail statistic remains meaningful for R0 too.
    high = max(float(np.percentile(s, 90)), 1e-8)
    return {
        "request_pearson_H_S": float(np.corrcoef(h, s)[0, 1]) if len(h) > 1 else None,
        "S_high_risk_threshold": high,
        "S_strictly_above_max_own_P90_or_numeric_floor_requests": int(np.sum(s > high)),
        "S_strictly_above_max_own_P90_or_numeric_floor_fraction": float(np.mean(s > high)),
        "H_near_zero_S_high_risk_fraction": float(np.mean((h <= 1e-8) & (s > high))),
    }


def build_cell(release: str, model: str, seed: int, cutover: int) -> dict:
    path = RAW_ROOT / release / f"{model}_seed{seed}" / "F_fidelity.parquet"
    records = request_records(path, cutover)
    return {
        "release": release, "model": model, "seed": seed, "source": str(path.relative_to(ROOT)),
        "source_sha256": p7.sha256_file(path), "requests": len(records),
        "H_js": summary(user_mean(records, "H_js")), "S_js": summary(user_mean(records, "S_js")),
        "H_score_rms": summary(user_mean(records, "H_score_rms")),
        "S_score_rms": summary(user_mean(records, "S_score_rms")),
        "H_probability_shift": summary(user_mean(records, "H_probability_shift")),
        "S_probability_shift": summary(user_mean(records, "S_probability_shift")),
        "H_S_joint": joint_summary(records),
        "by_prefix_tokens": {"H_js": cohort_summary(records, "H_js", "prefix_tokens", (32, 128, 256, 512)), "S_js": cohort_summary(records, "S_js", "prefix_tokens", (32, 128, 256, 512))},
        "by_suffix_tokens": {"H_js": cohort_summary(records, "H_js", "suffix_tokens", (1, 8, 32, 128)), "S_js": cohort_summary(records, "S_js", "suffix_tokens", (1, 8, 32, 128))},
        "by_state_age_seconds": {"H_js": cohort_summary(records, "H_js", "state_age_seconds", (86_400, 3 * 86_400, 7 * 86_400)), "S_js": cohort_summary(records, "S_js", "state_age_seconds", (86_400, 3 * 86_400, 7 * 86_400))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8, help="bounded CPU reader/statistic workers")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P9.1 result: {args.output}")
    if not EVIDENCE.exists():
        raise FileNotFoundError("run scripts/seal_p9_evidence.py first")
    evidence = json.loads(EVIDENCE.read_text())
    contract = yaml.safe_load(CONTRACT.read_text())
    if evidence["contract_hash"] != p7.sha256_file(CONTRACT):
        raise RuntimeError("P9 evidence seal and contract differ")
    cutovers = {"r0": 231 * 86_400, "r1_edge1": 231 * 86_400, "r1_edge2": 245 * 86_400, "r2": 231 * 86_400}
    if args.workers < 1 or args.workers > 24:
        raise ValueError("workers must be in [1, 24]")
    jobs = [(release, model, seed, cutovers[release]) for release in contract["scope"]["releases"] for model in contract["scope"]["models"] for seed in contract["scope"]["seeds"]]
    # DataFrame group-by and per-user aggregation are Python-heavy. Processes
    # (rather than threads) make the bounded worker setting real parallelism.
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cells = list(executor.map(build_cell, *(zip(*jobs, strict=True))))
    payload = {
        "status": "P9_1_label_free_H_S_distribution_complete",
        "contract_hash": p7.sha256_file(CONTRACT), "p8_evidence_seal_hash": p7.sha256_file(EVIDENCE),
        "scope": contract["scope"], "cells": cells,
        "notes": [
            "All primary analyses use F fidelity rows, which have no label, target, or feedback-stratum fields.",
            "prefix_tokens is a state-size/activity proxy; it is not a future-label feature.",
            "score RMS uses the frozen F 1e-3 denominator convention from P8, not a quality metric.",
            "v1 used a non-strict own-P90 tail predicate and is retained but invalid for downstream use; v2 uses strict > max(own P90, 1e-8).",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

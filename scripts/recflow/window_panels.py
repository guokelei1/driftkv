#!/usr/bin/env python3
"""Freeze complete fitting pools and balanced three-day development panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hstu_kvcache.recflow.data import PreparedRecFlow, stable_hash  # noqa: E402

WINDOWS = ((19, 21), (22, 24), (25, 27))
FIT_WINDOWS = {"train_1_18": (1, 18), "update_19_21": (19, 21), "update_22_24": (22, 24)}
FREE_PER_DAY = 1024
SAMPLED_PER_DAY = 256
DIAGNOSTIC_REQUEST_SALT = 170256
DIAGNOSTIC_UID_SALT = 256017


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/recflow_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/seed17_complete_epoch_windows")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise SystemExit("Existing panel evidence found; inspect before replacing.")
    started = time.monotonic()
    data = PreparedRecFlow(args.data)
    requests = data.requests
    cohort = data.cohort(min_history=1024, limit=512)
    catalog = data.catalog
    selected = np.lexsort((catalog["raw_item_ids"], -catalog["development_exposure_count"]))[:1_000_000]
    selected.sort()
    known = np.zeros(data.catalog_size + 2, dtype=bool)
    known[selected + 1] = True
    cumulative = np.r_[0, np.cumsum(known[data.positives["item"]])]
    known_counts = cumulative[requests["positive_stop"]] - cumulative[requests["positive_start"]]
    positive_counts = requests["positive_stop"] - requests["positive_start"]

    def counts(indices):
        y, m = positive_counts[indices], known_counts[indices]
        days, per_day = np.unique(requests["day"][indices], return_counts=True)
        return {
            "requests": len(indices), "users_with_requests": len(np.unique(requests["uid"][indices])),
            "positive_requests": int((y > 0).sum()), "zero_positive_requests": int((y == 0).sum()),
            "known_positive_requests": int((m > 0).sum()),
            "all_oov_positive_requests": int(((y > 0) & (m == 0)).sum()),
            "positive_targets": int(y.sum()), "known_positive_targets": int(m.sum()),
            "oov_positive_targets": int((y - m).sum()),
            "target_coverage": float(m.sum() / y.sum()) if y.sum() else None,
            "minimum_prefix_events": int((requests["history_stop"][indices] - requests["history_start"][indices]).min()),
            "requests_per_day": {str(int(day)): int(count) for day, count in zip(days, per_day, strict=True)},
            "request_indices_sha256": digest(indices),
            "request_uids_sha256": digest(requests["uid"][indices]),
            "positive_counts_sha256": digest(y), "known_positive_counts_sha256": digest(m),
        }

    arrays, training, evaluations, daily = {}, {}, {}, []
    for key, (first, last) in FIT_WINDOWS.items():
        indices = data.request_indices(first, last, uids=cohort, limit=None)
        arrays[key] = indices
        all_indices = data.request_indices(first, last, uids=cohort, limit=None, positive_only=False)
        training[key] = {
            "days": [first, last], "all_logged_requests": len(all_indices),
            "zero_positive_requests_not_in_fitting_array": len(all_indices) - len(indices),
            **counts(indices),
            "known_training_indices_sha256": digest(indices[known_counts[indices] > 0]),
        }
    for first, last in WINDOWS:
        free_parts, diagnostic_parts = [], []
        for day in range(first, last + 1):
            all_indices = data.request_indices(day, day, uids=cohort, limit=None, positive_only=False)
            available = data.request_indices(day, day, uids=cohort, limit=None)
            if len(available) < FREE_PER_DAY:
                raise ValueError(f"D{day} has fewer than {FREE_PER_DAY} positive requests; cannot fulfill frozen panel.")
            free = data.request_indices(day, day, uids=cohort, limit=FREE_PER_DAY)
            hashes = stable_hash(requests["request_id"][free], DIAGNOSTIC_REQUEST_SALT)
            hashes ^= stable_hash(requests["uid"][free], DIAGNOSTIC_UID_SALT)
            diagnostic = free[np.argsort(hashes, kind="stable")[:SAMPLED_PER_DAY]]
            assert np.isin(diagnostic, free).all()
            free_parts.append(free)
            diagnostic_parts.append(diagnostic)
            daily.append({"day": day, "all_logged_requests": len(all_indices),
                          "zero_positive_requests": len(all_indices) - len(available),
                          "available_positive_pool": counts(available),
                          "free_panel": counts(free), "sampled_panel": counts(diagnostic)})
        free_key, sampled_key = f"eval_{first}_{last}", f"sampled_{first}_{last}"
        arrays[free_key], arrays[sampled_key] = np.concatenate(free_parts), np.concatenate(diagnostic_parts)
        all_indices = data.request_indices(first, last, uids=cohort, limit=None, positive_only=False)
        all_positive = data.request_indices(first, last, uids=cohort, limit=None)
        evaluations[f"{first}_{last}"] = {
            "days": [first, last], "all_logged_requests": len(all_indices),
            "all_positive_requests": counts(all_positive),
            "free_panel_key": free_key, "free_panel": counts(arrays[free_key]),
            "sampled_panel_key": sampled_key, "sampled_panel": counts(arrays[sampled_key]),
        }
    for key, indices in arrays.items():
        assert np.all(requests["role"][indices] == 0), key
        assert np.isin(requests["uid"][indices], cohort).all(), key
        assert np.all(positive_counts[indices] > 0), key
        req = requests[indices]
        has_history = req["history_stop"] > req["history_start"]
        assert np.all(data.events["ts"][req["history_stop"][has_history] - 1] < req["ts"][has_history]), key
    args.output.mkdir(parents=True, exist_ok=True)
    panel_file = args.output / "window_panels.npz"
    np.savez_compressed(panel_file, **arrays)
    report = {
        "scope": "CPU-only frozen data/panel audit for seed17 complete-epoch DEVELOPMENT experiments. No training, model-quality scoring, formal admission or final-role outcomes.",
        "training_seed": 17, "candidate_seed": 17,
        "cohort_users": len(cohort), "cohort_sha256": digest(cohort),
        "cohort_rule": "Same fixed512 D1-18 development users with >=1024 real exposures, selected by initial stable UID hash. Later activity, OOV coverage and model outcomes never filter users.",
        "catalog_items": len(selected), "catalog_sha256": digest(catalog["raw_item_ids"][selected]),
        "catalog_rule": "Same initial D1-18 catalog top1M by development exposure count descending, raw item ID ascending on ties; mapping sorted by raw ID. No catalog expansion or future-label selection.",
        "epoch_definition": "Training arrays contain EVERY positive request in the window, with no150k cap. The current known-item objective traverses EVERY request containing a known positive exactly once; all-OOV fitting requests are excluded explicitly by train_phase and counted here. One sampled known positive per request is the existing objective; complete epoch does not mean every positive item is separately sampled.",
        "free_panel_rule": "For EACH day select1024 positive requests using PreparedRecFlow.request_indices stable hash: stable_hash(request_id,salt=3718) XOR stable_hash(uid,salt=0). Concatenate days chronologically. Preserve all-OOV requests; do not choose known-target requests preferentially.",
        "sampled_panel_rule": "Within EACH day's fixed1024 free panel, independently sort stable_hash(request_id,salt=170256) XOR stable_hash(uid,salt=256017), keep256, and concatenate days. Uniform1000 and popularity1000 share this768-request subset and candidate seed17. This subset has its own metric denominator/random baseline; do not compare its absolute score directly with the3072-request free panel.",
        "evaluation_windows": [[first, last] for first, last in WINDOWS],
        "primary": "Free beam300 FP32 NDCG@50; matched random baseline on exactly the same3072 requests. A evaluated on D19-21; A/B on D22-24; optional B/C on D25-27.",
        "companions": "NDCG@20/@100, Recall@20/@50/@100, coverage and known-target conditional metrics; uniform/popularity1000 diagnostics on the explicitly separate768-request subset.",
        "zero_label_rule": "No-positive requests are absent from fitting/retrieval-score arrays but are counted and remain in real history. Unknown targets stay in evaluation denominators; all-OOV positive requests score zero.",
        "arrays_file": str(panel_file), "arrays_file_sha256": hashlib.sha256(panel_file.read_bytes()).hexdigest(),
        "array_keys": list(arrays), "training": training, "evaluation": evaluations, "daily": daily,
        "checks": {"development_only": True, "same_fixed_cohort": True, "sampled_is_subset_of_free": True,
                   "complete_timestamp_block_excluded": True, "free_requests_per_day": FREE_PER_DAY,
                   "sampled_requests_per_day": SAMPLED_PER_DAY},
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), ROOT / "src/hstu_kvcache/recflow/data.py")},
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "seconds": report["elapsed_seconds"],
                      "training": training, "evaluation": evaluations}), flush=True)


if __name__ == "__main__":
    main()

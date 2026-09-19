#!/usr/bin/env python3
"""Prepare fixed4096 development users and complete daily RecFlow panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from window_panels import (
    ROOT, DIAGNOSTIC_REQUEST_SALT, DIAGNOSTIC_UID_SALT,
    PreparedRecFlow, digest, stable_hash,
)


COHORT_SHA256 = "5ef4a17d22300436f49c48b8aed17b8a86267b6f9ad1c6382d6c4d15aa16dd2a"
CATALOG_SHA256 = "52dfdccd90c1d66217e08fae51940706e88391b21e15a25cbe634c246af3e4d1"
SAMPLED_PER_DAY = 6144


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/recflow_v1")
    parser.add_argument("--source", type=Path, default=ROOT / "results/recflow/development/daily_window_panels_seed17/window_panels.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/expanded_daily_panels_u4096_seed17")
    args = parser.parse_args()
    args.data, args.source, args.output = args.data.resolve(), args.source.resolve(), args.output.resolve()
    if args.output.exists():
        raise SystemExit("Use a new output directory to preserve existing evidence.")
    begin = time.monotonic()
    source_manifest_path = args.source.with_name("summary.json")
    source_manifest_hash = file_digest(source_manifest_path)
    source = json.loads(source_manifest_path.read_text())
    source_hash = file_digest(args.source)
    assert source_hash == source["arrays_file_sha256"]

    prepared = PreparedRecFlow(args.data)
    requests, positives = prepared.requests, prepared.positives
    cohort = prepared.cohort(min_history=1024, limit=4096)
    old_cohort = prepared.cohort(min_history=1024, limit=512)
    assert len(cohort) == 4096 and digest(cohort) == COHORT_SHA256
    assert np.array_equal(cohort[:512], old_cohort)
    assert len(old_cohort) == source["cohort_users"] == 512
    assert digest(old_cohort) == source["cohort_sha256"]
    assert prepared.manifest["initial_day"] == 18
    population_rows = np.searchsorted(prepared.population["uids"], cohort)
    assert np.array_equal(prepared.population["uids"][population_rows], cohort)
    assert np.all(prepared.population["role"][population_rows] == 0)
    assert np.all(prepared.population["initial_history_count"][population_rows] >= 1024)

    catalog = prepared.catalog
    selected = np.lexsort((catalog["raw_item_ids"], -catalog["development_exposure_count"]))[:1_000_000]
    selected.sort()
    assert len(selected) == 1_000_000
    assert digest(catalog["raw_item_ids"][selected]) == source["catalog_sha256"] == CATALOG_SHA256
    assert file_digest(args.data / "manifest.json") == source["data_manifest_sha256"]
    known = np.zeros(prepared.catalog_size + 2, dtype=bool)
    known[selected + 1] = True
    cumulative = np.r_[0, np.cumsum(known[positives["item"]])]
    known_counts = cumulative[requests["positive_stop"]] - cumulative[requests["positive_start"]]
    positive_counts = requests["positive_stop"] - requests["positive_start"]

    def counts(indices):
        y, m = positive_counts[indices], known_counts[indices]
        days, per_day = np.unique(requests["day"][indices], return_counts=True)
        starts = requests["positive_start"][indices]
        flat = np.repeat(starts - np.r_[0, np.cumsum(y[:-1])], y) + np.arange(y.sum())
        unique_known = np.unique(positives["raw_item_id"][flat][known[positives["item"][flat]]])
        return {
            "requests": len(indices), "users_with_requests": len(np.unique(requests["uid"][indices])),
            "positive_requests": int((y > 0).sum()), "zero_positive_requests": int((y == 0).sum()),
            "known_positive_requests": int((m > 0).sum()),
            "all_oov_positive_requests": int(((y > 0) & (m == 0)).sum()),
            "positive_targets": int(y.sum()), "known_positive_targets": int(m.sum()),
            "oov_positive_targets": int((y - m).sum()), "target_coverage": float(m.sum() / y.sum()),
            "unique_known_positive_videos": len(unique_known),
            "unique_known_positive_raw_ids_sha256": digest(unique_known),
            "minimum_prefix_events": int((requests["history_stop"][indices] - requests["history_start"][indices]).min()),
            "requests_per_day": {str(int(day)): int(count) for day, count in zip(days, per_day, strict=True)},
            "request_indices_sha256": digest(indices), "request_uids_sha256": digest(requests["uid"][indices]),
            "unique_uids_sha256": digest(np.unique(requests["uid"][indices])),
            "positive_counts_sha256": digest(y), "known_positive_counts_sha256": digest(m),
        }

    arrays, training, evaluation = {}, {}, {}
    fit_windows = {"train_1_18": (1, 18), **{f"update_{day}_{day}": (day, day) for day in range(19, 24)}}
    for key, (first, last) in fit_windows.items():
        indices = prepared.request_indices(first, last, uids=cohort, limit=None)
        arrays[key] = indices
        logged = prepared.request_indices(first, last, uids=cohort, limit=None, positive_only=False)
        eligible = indices[known_counts[indices] > 0]
        training[key] = {
            "days": [first, last], "all_logged_requests": len(logged),
            "zero_positive_requests_not_in_fitting_array": len(logged) - len(indices),
            **counts(indices), "known_training_indices_sha256": digest(eligible),
            "global_batch128_steps_per_epoch": (len(eligible) + 127) // 128,
            "global_batch128_final_batch_requests": (len(eligible) - 1) % 128 + 1,
        }
    with np.load(args.source) as frozen:
        initial = arrays["train_1_18"]
        assert np.array_equal(initial[np.isin(requests["uid"][initial], old_cohort)], frozen["train_1_18"])

    for day in range(19, 25):
        key = f"{day}_{day}"
        full = prepared.request_indices(day, day, uids=cohort, limit=None)
        hashes = stable_hash(requests["request_id"][full], DIAGNOSTIC_REQUEST_SALT)
        hashes ^= stable_hash(requests["uid"][full], DIAGNOSTIC_UID_SALT)
        sampled = full[np.argsort(hashes, kind="stable")[:SAMPLED_PER_DAY]]
        assert len(sampled) == SAMPLED_PER_DAY and np.isin(sampled, full).all()
        arrays[f"eval_{key}"], arrays[f"sampled_{key}"] = full, sampled
        logged = prepared.request_indices(day, day, uids=cohort, limit=None, positive_only=False)
        full_counts = counts(full)
        evaluation[key] = {
            "days": [day, day], "all_logged_requests": len(logged),
            "zero_positive_requests_not_in_scoring_array": len(logged) - len(full),
            "all_positive_requests": full_counts, "free_panel_key": f"eval_{key}",
            "free_panel": full_counts, "sampled_panel_key": f"sampled_{key}",
            "sampled_panel": counts(sampled),
        }
        if day < 24:
            assert np.array_equal(arrays[f"update_{key}"], full)

    for key, indices in arrays.items():
        assert len(np.unique(indices)) == len(indices), key
        req = requests[indices]
        assert np.all(req["role"] == 0) and np.isin(req["uid"], cohort).all(), key
        assert np.all(positive_counts[indices] > 0), key
        if key != "train_1_18":
            assert np.all(req["history_stop"] - req["history_start"] >= 1024), key
        assert np.all(req["history_stop"] <= req["event_start"]), key
        # Initial fitting includes short and empty histories. For nonempty
        # histories check the preceding event; always check the target block.
        has_history = req["history_stop"] > req["history_start"]
        last = prepared.events[req["history_stop"][has_history] - 1]
        assert np.all(last["uid"] == req["uid"][has_history]), key
        assert np.all(last["ts"] < req["ts"][has_history]), key
        first = prepared.events[req["history_stop"]]
        assert np.all(first["uid"] == req["uid"]) and np.all(first["ts"] == req["ts"]), key

    data_hashes = {name: file_digest(args.data / name) for name in
                   ("manifest.json", "catalog.npz", "population.npz", "events.npy", "requests.npy", "positives.npy")}
    assert file_digest(args.source) == source_hash
    assert file_digest(source_manifest_path) == source_manifest_hash
    args.output.mkdir(parents=True)
    panels = args.output / "window_panels.npz"
    np.savez_compressed(panels, **arrays)
    report = {
        "scope": "CPU-only fixed4096 DEVELOPMENT data expansion. No training, model outcomes, weights, formal admission or final-role outcomes accessed.",
        "training_seed": 17, "candidate_seed": 17, "context": 1024, "data": str(args.data),
        "cohort_users": len(cohort), "cohort_sha256": digest(cohort),
        "training_cohort_users": len(cohort), "training_cohort_sha256": digest(cohort),
        "evaluation_cohort_users": len(cohort), "evaluation_cohort_sha256": digest(cohort),
        "cohort_rule": "Development-role users with >=1024 D1-18 realshow events, sorted by stable_hash(uid,1818), first4096. Existing512 is the exact prefix. No future activity, labels, target coverage or model outcomes select UIDs.",
        "initial_population_day": 18,
        "minimum_initial_cohort_exposures": int(prepared.population["initial_history_count"][population_rows].min()),
        "catalog_items": len(selected), "catalog_sha256": digest(catalog["raw_item_ids"][selected]),
        "catalog_rule": source["catalog_rule"],
        "epoch_definition": source["epoch_definition"], "zero_label_rule": source["zero_label_rule"],
        "initial_history_rule": "D1-18 training retains short and empty prefixes; initial user eligibility does not require every initial request to have1024 preceding events. All later update/evaluation requests have at least1024 prior events.",
        "request_selection": "Every positive request in each complete D19..24 daily window for the frozen4096 UIDs; request means retain all-OOV misses. No later-activity or coverage selection of UIDs.",
        "evaluation_windows": [[day, day] for day in range(19, 25)],
        "sampled_panel_rule": "For each complete day's positive-request panel, sort stable_hash(request_id,170256) XOR stable_hash(uid,256017) and retain6144. No known-positive filtering; sampled protocols use this subset's own request/OOV denominator and matched random.",
        "sampled_requests_per_day": SAMPLED_PER_DAY,
        "primary": "Not selected by data preparation; model/metric settings must be fixed separately before evaluation.",
        "companions": "Preserve full-catalog and sampled protocol distinctions, all declared cutoffs, request/OOV denominators and matched random baselines.",
        "arrays_file": str(panels), "arrays_file_sha256": file_digest(panels), "array_keys": list(arrays),
        "training": training, "evaluation": evaluation,
        "data_manifest_sha256": data_hashes["manifest.json"], "data_file_sha256": data_hashes,
        "source_panels": str(args.source), "source_panels_sha256": source_hash,
        "source_manifest": str(source_manifest_path), "source_manifest_sha256": source_manifest_hash,
        "checks": dict(development_only=True, initial_only_cohort_eligibility=True,
            fixed512_exact_prefix_of_fixed4096=True, same_frozen1m_catalog=True,
            original512_initial_requests_preserved=True, same_training_evaluation_cohort=True,
            complete_positive_request_windows=True, sampled_is_subset_of_free=True,
            sampled_requests_per_day=SAMPLED_PER_DAY, complete_timestamp_block_excluded=True,
            update_evaluation_minimum_history_at_least1024=True, initial_short_histories_retained=True,
            source_files_unchanged=True, model_results_not_accessed=True),
        "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/recflow/window_panels.py", ROOT / "src/hstu_kvcache/recflow/data.py")},
        "elapsed_seconds": time.monotonic() - begin,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), arrays_file_sha256=report["arrays_file_sha256"],
        seconds=report["elapsed_seconds"],
        training={key: {name: row[name] for name in ("requests", "known_positive_requests", "global_batch128_steps_per_epoch", "global_batch128_final_batch_requests")}
                  for key, row in training.items()},
        evaluation={key: dict(full=row["free_panel"]["requests"], sampled=row["sampled_panel"]["requests"],
                              known=row["free_panel"]["known_positive_requests"])
                    for key, row in evaluation.items()})), flush=True)


if __name__ == "__main__":
    main()

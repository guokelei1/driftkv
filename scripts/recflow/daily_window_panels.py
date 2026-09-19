#!/usr/bin/env python3
"""Freeze one-day RecFlow update panels with the existing 512-user A pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hstu_kvcache.recflow.data import PreparedRecFlow, stable_hash  # noqa: E402
from window_panels import DIAGNOSTIC_REQUEST_SALT, DIAGNOSTIC_UID_SALT, digest  # noqa: E402


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/recflow_v1")
    parser.add_argument("--source", type=Path, default=ROOT / "results/recflow/development/full_window_training_seed17/window_panels.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/daily_window_panels_seed17")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output directory to preserve existing evidence.")
    begin = time.monotonic()
    source_manifest_path = args.source.with_name("summary.json")
    source = json.loads(source_manifest_path.read_text())
    source_hash = file_digest(args.source)
    assert source_hash == source["arrays_file_sha256"]
    prepared = PreparedRecFlow(args.data)
    requests, positives = prepared.requests, prepared.positives
    cohort = prepared.cohort(min_history=1024, limit=512)
    catalog = prepared.catalog
    selected = np.lexsort((catalog["raw_item_ids"], -catalog["development_exposure_count"]))[:1_000_000]
    selected.sort()
    assert digest(cohort) == source["cohort_sha256"]
    assert digest(catalog["raw_item_ids"][selected]) == source["catalog_sha256"]
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
        target_items = positives["item"][flat]
        unique_known = np.unique(positives["raw_item_id"][flat][known[target_items]])
        return {
            "requests": len(indices), "users_with_requests": len(np.unique(requests["uid"][indices])),
            "positive_requests": int((y > 0).sum()), "zero_positive_requests": int((y == 0).sum()),
            "known_positive_requests": int((m > 0).sum()),
            "all_oov_positive_requests": int(((y > 0) & (m == 0)).sum()),
            "positive_targets": int(y.sum()), "known_positive_targets": int(m.sum()),
            "oov_positive_targets": int((y - m).sum()),
            "target_coverage": float(m.sum() / y.sum()),
            "unique_known_positive_videos": len(unique_known),
            "unique_known_positive_raw_ids_sha256": digest(unique_known),
            "minimum_prefix_events": int((requests["history_stop"][indices] - requests["history_start"][indices]).min()),
            "requests_per_day": {str(int(day)): int(count) for day, count in zip(days, per_day, strict=True)},
            "request_indices_sha256": digest(indices), "request_uids_sha256": digest(requests["uid"][indices]),
            "unique_uids_sha256": digest(np.unique(requests["uid"][indices])),
            "positive_counts_sha256": digest(y), "known_positive_counts_sha256": digest(m),
        }

    with np.load(args.source) as frozen:
        arrays = {"train_1_18": frozen["train_1_18"]}
    assert np.array_equal(arrays["train_1_18"], prepared.request_indices(1, 18, uids=cohort, limit=None))
    training = {"train_1_18": dict(source["training"]["train_1_18"])}
    assert digest(arrays["train_1_18"][known_counts[arrays["train_1_18"]] > 0]) == training["train_1_18"]["known_training_indices_sha256"]
    for day in (19, 20):
        key = f"update_{day}_{day}"
        indices = prepared.request_indices(day, day, uids=cohort, limit=None)
        arrays[key] = indices
        logged = prepared.request_indices(day, day, uids=cohort, limit=None, positive_only=False)
        n_known = int((known_counts[indices] > 0).sum())
        training[key] = {
            "days": [day, day], "all_logged_requests": len(logged),
            "zero_positive_requests_not_in_fitting_array": len(logged) - len(indices),
            **counts(indices),
            "known_training_indices_sha256": digest(indices[known_counts[indices] > 0]),
            "global_batch128_steps_per_epoch": (n_known + 127) // 128,
            "global_batch128_final_batch_requests": (n_known - 1) % 128 + 1,
        }
    evaluations = {}
    for day in (20, 21):
        key = f"{day}_{day}"
        full = prepared.request_indices(day, day, uids=cohort, limit=None)
        hashes = stable_hash(requests["request_id"][full], DIAGNOSTIC_REQUEST_SALT)
        hashes ^= stable_hash(requests["uid"][full], DIAGNOSTIC_UID_SALT)
        sampled = full[np.argsort(hashes, kind="stable")[:768]]
        arrays[f"eval_{key}"], arrays[f"sampled_{key}"] = full, sampled
        logged = prepared.request_indices(day, day, uids=cohort, limit=None, positive_only=False)
        evaluations[key] = {
            "days": [day, day], "all_logged_requests": len(logged),
            "all_positive_requests": counts(full), "free_panel_key": f"eval_{key}",
            "free_panel": counts(full), "sampled_panel_key": f"sampled_{key}",
            "sampled_panel": counts(sampled),
        }
        assert np.isin(sampled, full).all()
    for key, indices in arrays.items():
        assert len(np.unique(indices)) == len(indices), key
        req = requests[indices]
        assert np.all(req["role"] == 0) and np.isin(req["uid"], cohort).all(), key
        assert np.all(positive_counts[indices] > 0), key
        with_history = req["history_stop"] > req["history_start"]
        assert np.all(prepared.events["ts"][req["history_stop"][with_history] - 1] < req["ts"][with_history]), key
    args.output.mkdir(parents=True)
    panels = args.output / "window_panels.npz"
    np.savez_compressed(panels, **arrays)
    with zipfile.ZipFile(args.source) as old, zipfile.ZipFile(panels) as new:
        assert old.read("train_1_18.npy") == new.read("train_1_18.npy")
    assert file_digest(args.source) == source_hash
    report = {key: source[key] for key in ("training_seed", "candidate_seed", "cohort_users", "cohort_sha256", "cohort_rule", "catalog_items", "catalog_sha256", "catalog_rule", "epoch_definition", "zero_label_rule", "data_manifest_sha256")}
    report.update(
        scope="One-day-update DEVELOPMENT panels on the existing 512-user cohort. No formal admission or final-role outcomes.",
        training_cohort_users=len(cohort), training_cohort_sha256=digest(cohort),
        evaluation_cohort_users=len(cohort), evaluation_cohort_sha256=digest(cohort),
        request_selection="All positive requests in each complete one-day evaluation window; request mean including all-OOV misses.",
        evaluation_windows=[[20, 20], [21, 21]],
        primary="Free beam300 FP32 NDCG@50; A/B on D20 after B fits D19, then B/C on D21 after C fits D20. Same catalog, positive requests and OOV denominator for each model and matched random.",
        companions="NDCG@20/@100 and Recall@20/@50/@100; per-user means and known-target conditional metrics; uniform/popularity1000 diagnostics on each separate 768-request subset.",
        sampled_panel_rule="Sort every full day's positive requests by stable_hash(request_id,170256) XOR stable_hash(uid,256017), keep first min(768,available). This is a new fixed diagnostic subset, not the previous three-day 256-per-day subset. Candidate seed17 and its own denominator/random baseline apply.",
        arrays_file=str(panels), arrays_file_sha256=file_digest(panels), array_keys=list(arrays),
        training=training, evaluation=evaluations,
        source_panels=str(args.source), source_panels_sha256=source_hash,
        source_manifest=str(source_manifest_path), source_manifest_sha256=file_digest(source_manifest_path),
        checks=dict(development_only=True, same_training_evaluation_cohort=True,
            initial_training_npy_member_byte_identical_to_source=True,
            complete_positive_request_windows=True, sampled_is_subset_of_free=True,
            complete_timestamp_block_excluded=True, source_file_unchanged=True),
        source_sha256={str(path.relative_to(ROOT)): file_digest(path) for path in (Path(__file__).resolve(), ROOT / "scripts/recflow/window_panels.py", ROOT / "src/hstu_kvcache/recflow/data.py")},
        elapsed_seconds=time.monotonic() - begin,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), arrays_file_sha256=report["arrays_file_sha256"], training=training, evaluation=evaluations)), flush=True)


if __name__ == "__main__":
    main()

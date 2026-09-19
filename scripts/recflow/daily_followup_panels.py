#!/usr/bin/env python3
"""Prepare D21 fitting and D22 follow-up development panels; no model access."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from daily_window_panels import (
    ROOT, DIAGNOSTIC_REQUEST_SALT, DIAGNOSTIC_UID_SALT,
    PreparedRecFlow, digest, stable_hash,
)


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/recflow_v1")
    parser.add_argument("--source", type=Path, default=ROOT / "results/recflow/development/daily_window_panels_seed17/window_panels.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/daily_followup_panels_seed17")
    args = parser.parse_args()
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
    cohort = prepared.cohort(min_history=1024, limit=512)
    catalog = prepared.catalog
    selected = np.lexsort((catalog["raw_item_ids"], -catalog["development_exposure_count"]))[:1_000_000]
    selected.sort()
    assert len(cohort) == source["cohort_users"] == 512
    assert len(selected) == source["catalog_items"] == 1_000_000
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
            "oov_positive_targets": int((y - m).sum()), "target_coverage": float(m.sum() / y.sum()),
            "unique_known_positive_videos": len(unique_known),
            "unique_known_positive_raw_ids_sha256": digest(unique_known),
            "minimum_prefix_events": int((requests["history_stop"][indices] - requests["history_start"][indices]).min()),
            "requests_per_day": {str(int(day)): int(count) for day, count in zip(days, per_day, strict=True)},
            "request_indices_sha256": digest(indices), "request_uids_sha256": digest(requests["uid"][indices]),
            "unique_uids_sha256": digest(np.unique(requests["uid"][indices])),
            "positive_counts_sha256": digest(y), "known_positive_counts_sha256": digest(m),
        }

    update = prepared.request_indices(21, 21, uids=cohort, limit=None)
    with np.load(args.source) as frozen:
        assert np.array_equal(update, frozen["eval_21_21"])
    full = prepared.request_indices(22, 22, uids=cohort, limit=None)
    hashes = stable_hash(requests["request_id"][full], DIAGNOSTIC_REQUEST_SALT)
    hashes ^= stable_hash(requests["uid"][full], DIAGNOSTIC_UID_SALT)
    sampled = full[np.argsort(hashes, kind="stable")[:768]]
    arrays = {"update_21_21": update, "eval_22_22": full, "sampled_22_22": sampled}
    update_logged = prepared.request_indices(21, 21, uids=cohort, limit=None, positive_only=False)
    eval_logged = prepared.request_indices(22, 22, uids=cohort, limit=None, positive_only=False)
    known_training = update[known_counts[update] > 0]
    training = {"update_21_21": {
        "days": [21, 21], "all_logged_requests": len(update_logged),
        "zero_positive_requests_not_in_fitting_array": len(update_logged) - len(update),
        **counts(update), "known_training_indices_sha256": digest(known_training),
        "global_batch128_steps_per_epoch": (len(known_training) + 127) // 128,
        "global_batch128_final_batch_requests": (len(known_training) - 1) % 128 + 1,
    }}
    evaluation = {"22_22": {
        "days": [22, 22], "all_logged_requests": len(eval_logged),
        "zero_positive_requests_not_in_scoring_array": len(eval_logged) - len(full),
        "all_positive_requests": counts(full), "free_panel_key": "eval_22_22", "free_panel": counts(full),
        "sampled_panel_key": "sampled_22_22", "sampled_panel": counts(sampled),
    }}
    assert np.isin(sampled, full).all() and len(sampled) == 768
    for key, indices in arrays.items():
        assert len(np.unique(indices)) == len(indices), key
        req = requests[indices]
        assert np.all(req["role"] == 0) and np.isin(req["uid"], cohort).all(), key
        assert np.all(positive_counts[indices] > 0), key
        assert np.all(req["history_stop"] - req["history_start"] >= 1024), key
        # Verify both sides of the block boundary, including other requests at
        # the same timestamp: no part of that target block enters the prefix.
        assert np.all(req["history_stop"] <= req["event_start"]), key
        last = prepared.events[req["history_stop"] - 1]
        block_first = prepared.events[req["history_stop"]]
        assert np.all(last["uid"] == req["uid"]) and np.all(last["ts"] < req["ts"]), key
        assert np.all(block_first["uid"] == req["uid"]) and np.all(block_first["ts"] == req["ts"]), key

    data_hashes = {name: file_digest(args.data / name) for name in
                   ("manifest.json", "catalog.npz", "population.npz", "events.npy", "requests.npy", "positives.npy")}
    args.output.mkdir(parents=True)
    panels = args.output / "window_panels.npz"
    np.savez_compressed(panels, **arrays)
    with zipfile.ZipFile(args.source) as old, zipfile.ZipFile(panels) as new:
        assert old.read("eval_21_21.npy") == new.read("update_21_21.npy")
    assert file_digest(args.source) == source_hash
    assert file_digest(source_manifest_path) == source_manifest_hash
    report = {key: source[key] for key in (
        "training_seed", "candidate_seed", "cohort_users", "cohort_sha256", "cohort_rule",
        "catalog_items", "catalog_sha256", "catalog_rule", "epoch_definition", "zero_label_rule",
        "data_manifest_sha256", "sampled_panel_rule")}
    report.update(
        scope="CPU-only additional chronological DEVELOPMENT confirmation panels: C already fits through D20; a future D fits all eligible D21 requests for one complete epoch and C/D are compared on D22. No training or model results accessed by this script.",
        prior_exposure="D22 was already examined in earlier three-day development experiments. This is not a newly untouched final holdout. Choose and freeze the development configuration before using this extra daily confirmation comparison.",
        configuration_selection="No learning rate, NDCG cutoff, decoder or model configuration selected by data preparation.",
        context=1024, data=str(args.data), data_file_sha256=data_hashes,
        training_cohort_users=len(cohort), training_cohort_sha256=digest(cohort),
        evaluation_cohort_users=len(cohort), evaluation_cohort_sha256=digest(cohort),
        request_selection="Every positive request in complete D22; request mean retains all-OOV misses. No later-activity or coverage selection of UIDs.",
        evaluation_windows=[[22, 22]],
        primary="Not selected by this data-only preparation; freeze the chosen development metric before the C/D comparison.",
        companions="Retain existing full-catalog and sampled diagnostic metric fields and OOV denominators; candidate seed17.",
        arrays_file=str(panels), arrays_file_sha256=file_digest(panels), array_keys=list(arrays),
        training=training, evaluation=evaluation,
        source_panels=str(args.source), source_panels_sha256=source_hash,
        source_manifest=str(source_manifest_path), source_manifest_sha256=source_manifest_hash,
        checks=dict(development_only=True, same_training_evaluation_cohort=True,
                    training_d21_npy_member_byte_identical_to_source_evaluation_d21=True,
                    complete_positive_request_windows=True, sampled_is_subset_of_free=True,
                    complete_timestamp_block_excluded=True, minimum_history_at_least1024=True,
                    source_files_unchanged=True, model_results_not_accessed=True),
        source_sha256={str(path.relative_to(ROOT)): file_digest(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/recflow/daily_window_panels.py",
            ROOT / "scripts/recflow/window_panels.py", ROOT / "src/hstu_kvcache/recflow/data.py")},
        elapsed_seconds=time.monotonic() - begin,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), arrays_file_sha256=report["arrays_file_sha256"],
                         training=training, evaluation=evaluation)), flush=True)


if __name__ == "__main__":
    main()

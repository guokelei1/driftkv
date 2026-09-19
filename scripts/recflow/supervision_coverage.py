#!/usr/bin/env python3
"""Reconstruct one-pass pilot positive draws from saved settings; no fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from development_probe import ROOT, PreparedRecFlow, ProbeData, save_json


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/recflow/development/pilot_6l_h192_c1024_k1m_seed17")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/supervision_coverage")
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_summary = args.run / "summary.json"
    saved = json.loads(source_summary.read_text())
    cfg = saved["configuration"]
    prepared = PreparedRecFlow(cfg["data"])
    data = ProbeData(prepared, cfg["catalog_size"], cfg["cohort_users"], cfg["context"])
    if hashlib.sha256(data.raw_ids.tobytes()).hexdigest() != cfg["catalog_sha256"]:
        raise ValueError("Catalog differs from the recorded training run")
    if hashlib.sha256(np.asarray(data.uids).tobytes()).hexdigest() != cfg["cohort_sha256"]:
        raise ValueError("Cohort differs from the recorded training run")
    # Independently retained CPU count baseline used the same request selectors.
    reference_path = ROOT / "results/recflow/development/popularity_temporal_k1m/summary.json"
    reference = json.loads(reference_path.read_text())
    phases, chosen_sets, possible_sets = {}, {}, {}
    for phase, first, last, reference_key in (
        ("parent", 1, 18, "initial_budget"), ("current", 19, 21, "update_budget"),
    ):
        selected = data.indices(first, last, cfg["train_requests"])
        target_lists = {int(index): data.targets(index)[1] for index in selected}
        known = np.array([index for index in selected if len(target_lists[int(index)])])
        record = saved[phase + "_train"]
        if len(selected) != record["selected_requests"] or len(known) != record["known_target_requests"]:
            raise ValueError(f"{phase} request counts differ from the recorded training run")
        selection_hash = array_sha256(selected)
        if selection_hash != reference[reference_key]["request_indices_sha256"]:
            raise ValueError(f"{phase} request selection differs from the retained independent count baseline")
        processed = record["processed_requests"]
        # These concrete pilot phases both stop before their first reshuffle.
        if processed > len(known) or processed != record["optimizer_steps"] * cfg["batch_size"]:
            raise ValueError("This diagnostic supports the recorded single partial-pass pilots only")
        rng_seed = cfg["seed"] + (1000 if phase == "current" else 0)
        rng = np.random.default_rng(rng_seed)
        order = rng.permutation(known)
        used = order[:processed]
        draws = np.array([int(rng.choice(target_lists[int(index)])) for index in used], dtype=np.int64)
        possible = np.unique(np.concatenate([target_lists[int(index)] for index in known]))
        days, day_counts = np.unique(prepared.requests["day"][used], return_counts=True)
        chosen_sets[phase], possible_sets[phase] = set(draws.tolist()), set(possible.tolist())
        phases[phase] = {
            "days": [first, last], "rng_seed": rng_seed,
            "selected_requests": len(selected), "known_target_requests": len(known),
            "processed_requests": processed, "fraction_known_requests_used": processed / len(known),
            "unique_chosen_positive_videos": len(chosen_sets[phase]),
            "unique_possible_positive_videos": len(possible_sets[phase]),
            "processed_request_day_counts": {str(day): int(count) for day, count in zip(days, day_counts, strict=True)},
            "selected_request_indices_sha256": selection_hash,
            "processed_request_indices_sha256": array_sha256(used),
            "drawn_mapped_positive_ids_sha256": array_sha256(draws),
            "recorded_request_counts_match": True, "independent_selection_hash_matches": True,
            "measured_training_requests_per_second": record["requests_per_second"],
            "recorded_optimizer_steps": record["optimizer_steps"],
        }
    evaluation = {}
    for day in (19, 22):
        indices = data.indices(day, day, cfg["eval_requests"])
        known = np.concatenate([data.targets(index)[1] for index in indices])
        lookup = {
            "parent_chosen_positive": chosen_sets["parent"],
            "parent_possible_positive_pool": possible_sets["parent"],
        }
        # D19 is inside current's fitting window: do not present current's
        # training overlap on that day as held-out evidence.
        if day == 22:
            lookup.update(current_chosen_positive=chosen_sets["current"],
                          current_possible_positive_pool=possible_sets["current"],
                          parent_or_current_chosen_positive=chosen_sets["parent"] | chosen_sets["current"])
        evaluation[str(day)] = {
            "requests": len(indices), "request_indices_sha256": array_sha256(indices),
            "known_positive_occurrences": len(known), "known_distinct_videos": len(np.unique(known)),
            "fraction_known_positive_occurrences_seen": {
                name: float(np.mean([int(item) in values for item in known])) for name, values in lookup.items()
            },
        }
    sources = [Path(__file__).resolve(), ROOT / "scripts/recflow/development_probe.py",
               ROOT / "src/hstu_kvcache/recflow/data.py", source_summary, reference_path,
               prepared.root / "manifest.json"]
    result = {
        "scope": "CPU reconstruction of recorded development training supervision; no fitting or model-quality evaluation.",
        "run": str(args.run), "training_seed": cfg["seed"],
        "settings": {key: cfg[key] for key in ("catalog_size", "cohort_users", "context", "train_requests", "eval_requests", "batch_size")},
        "catalog_sha256": cfg["catalog_sha256"], "cohort_sha256": cfg["cohort_sha256"],
        "reconstruction": "For each phase, retain requests with known positives, reproduce default_rng(seed [+1000 for current]).permutation(known_requests), and call the same rng.choice(known_positives) once per processed request. Both phases stop before the first reshuffle.",
        "verification_limit": "Request counts match saved training summaries and full selected-request hashes match an independently retained count baseline. The original run did not retain a per-draw target hash; positive-draw identity is reconstructed from the recorded RNG algorithm and seed, not independently cross-checked against an original draw log.",
        "interpretation": "Counts measure direct sampled positive supervision, not whether embedding rows received any gradient. Shared item embeddings also receive history-input and full-within-pair softmax gradients. Low direct positive coverage does not prove that more training will succeed.",
        "phases": phases, "future_known_target_coverage": evaluation,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "recorded_training_source_sha256": cfg.get("source_sha256", {}),
        "numpy_version": np.__version__,
    }
    save_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

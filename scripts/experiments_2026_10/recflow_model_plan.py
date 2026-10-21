#!/usr/bin/env python3
"""CPU-only catalog, sample and parameter accounting for the RecFlow design.

No model forward, optimizer, CUDA allocation or quality evaluation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from hstu_kvcache.data.oov import stable_oov_bucket
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.recflow.data import stable_hash


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(requests, mask, exposure_counts, positive_counts, known_counts):
    lengths = requests["history_stop"] - requests["history_start"]
    users = np.unique(requests["uid"][mask])
    targets = int(exposure_counts[mask].sum())
    return {
        "active_users": int(len(users)),
        "eligible_requests": int(mask.sum()),
        "exposure_targets": targets,
        "positive_targets": int(positive_counts[mask].sum()),
        "negative_targets": targets - int(positive_counts[mask].sum()),
        "requests_with_history_ge1024": int((lengths[mask] >= 1024).sum()),
        "history_length_quantiles_capped1024": np.quantile(
            np.minimum(lengths[mask], 1024), [0, 0.25, 0.5, 0.75, 1]
        ).tolist() if mask.any() else [],
        "known_targets": {name: int(counts[mask].sum()) for name, counts in known_counts.items()},
        "known_target_coverage": {
            name: float(counts[mask].sum() / targets) if targets else None
            for name, counts in known_counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/recflow_models_2026_10_21.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/planning/recflow_models_2026_10_21.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["data_root"])
    events = np.load(root / "events.npy", mmap_mode="r")
    req = np.load(root / "requests.npy", mmap_mode="r")
    population = dict(np.load(root / "population.npz"))
    catalog = dict(np.load(root / "catalog.npz"))
    counts = req["event_stop"] - req["event_start"]
    valid = req["history_stop"] > req["history_start"]
    base_mask = (req["role"] == 0) & (req["day"] <= 18) & valid
    eligible_base_events = np.repeat(base_mask, counts)
    support = np.bincount(events["item"][eligible_base_events], minlength=len(catalog["raw_item_ids"]) + 2)[1:-1]
    order = np.lexsort((catalog["raw_item_ids"], -support))
    positives = np.add.reduceat((events["behavior"] & 1).astype(np.int32), req["event_start"])
    known_counts, catalog_results, model_results = {}, {}, {}
    for name, spec in cfg["scales"].items():
        k, buckets = spec["known_videos"], cfg["catalog"]["oov_buckets"]
        selected = order[:k]
        if support[selected].min() < 1:
            raise ValueError("Requested known table contains videos without eligible training targets")
        known_lookup = np.zeros(len(support) + 2, dtype=bool)
        known_lookup[selected + 1] = True
        known_counts[name] = np.add.reduceat(known_lookup[events["item"]].astype(np.int32), req["event_start"])
        tail = np.flatnonzero((support > 0) & ~known_lookup[1:-1])
        tail_buckets = np.fromiter(
            (stable_oov_bucket(int(raw), known_vocab_size=0, buckets=buckets) for raw in catalog["raw_item_ids"][tail]),
            dtype=np.int64, count=len(tail),
        )
        occupancy = np.bincount(tail_buckets, minlength=buckets)
        catalog_results[name] = {
            "known_videos": k,
            "smallest_training_support": int(support[selected].min()),
            "known_videos_seen_once": int((support[selected] == 1).sum()),
            "known_video_ids_sha256": hashlib.sha256(np.sort(catalog["raw_item_ids"][selected]).astype("<i4").tobytes()).hexdigest(),
            "initial_tail_unique_videos": int(len(tail)),
            "oov_buckets": buckets,
            "oov_buckets_with_direct_training_targets": int((occupancy > 0).sum()),
            "initial_tail_unique_ids_per_nonempty_bucket_mean": float(occupancy[occupancy > 0].mean()),
        }
        model_cfg = HSTUConfig(**cfg["shared_model"], num_layers=spec["num_layers"], num_items=k + buckets)
        with torch.device("meta"):
            model = HSTU(model_cfg)
        total = sum(p.numel() for p in model.parameters())
        embedding = model.item_emb.weight.numel()
        reference = torch.load(spec["reference_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
        actual_reference_count = sum(t.numel() for t in reference["model"].values())
        if actual_reference_count != spec["reference_parameters"]:
            raise ValueError("Yambda reference parameter count changed")
        model_results[name] = {
            "model_config": vars(model_cfg),
            "parameters": total,
            "item_embedding_parameters": embedding,
            "other_parameters": total - embedding,
            "reference_yambda_parameters": actual_reference_count,
            "difference_percent": 100 * (total / actual_reference_count - 1),
            "nonembedding_matches_yambda": total - embedding == sum(t.numel() for n, t in reference["model"].items() if n != "item_emb.weight"),
            "fp32_weights_gb": 4 * total / 1e9,
            "fp32_weights_six_endpoints_gb": 6 * 4 * total / 1e9,
            "fp32_model_adam_recovery_gb_each": 12 * total / 1e9,
            "fp32_weights_grad_adam_state_gb_unsharded": 16 * total / 1e9,
            "full_1024_native_kv_bytes_per_user_fp32": 2 * spec["num_layers"] * 1024 * 320 * 4,
            "initial_actual_rows_native_kv_gb_all_residents_fp32": float(np.minimum(population["initial_history_count"], 1024).sum() * 2 * spec["num_layers"] * 320 * 4 / 1e9),
            "initial_actual_rows_native_kv_gb_final_role_fp32": float(np.minimum(population["initial_history_count"][population["role"] == 2], 1024).sum() * 2 * spec["num_layers"] * 320 * 4 / 1e9),
        }
        del model, reference

    roles = {}
    for role_id, role_name in [(0, "development"), (1, "admission"), (2, "final")]:
        role_users = population["role"] == role_id
        periods = {}
        for a, b in [(1, 18), (19, 23), (19, 24), (20, 24), (25, 37)]:
            mask = (req["role"] == role_id) & (req["day"] >= a) & (req["day"] <= b)
            entry = summarize(req, mask & valid, counts, positives, known_counts)
            entry["empty_history_requests"] = int((mask & ~valid).sum())
            entry["empty_history_exposure_targets"] = int(counts[mask & ~valid].sum())
            periods[f"d{a}_d{b}"] = entry
        roles[role_name] = {
            "pool_users": int(role_users.sum()),
            "pool_users_ge1024_at_cutoff": int((population["initial_history_count"][role_users] >= 1024).sum()),
            "periods": periods,
        }

    admission_users = population["uids"][population["role"] == 1]
    admission_users = admission_users[np.lexsort((admission_users, stable_hash(admission_users, salt=20261021)))]
    start, reservations = 0, {}
    for name in ["foundation_validation_users", "correction_fit_pool_users", "correction_validation_users", "unused_reserved_users"]:
        size = cfg["population"]["admission_role"][name]
        users = admission_users[start:start + size]
        mask = np.isin(req["uid"], users) & (req["day"] >= 19) & (req["day"] <= 24) & valid
        reservations[name] = {"users": size, "uids_sha256": hashlib.sha256(users.astype("<i4").tobytes()).hexdigest(), "d19_d24": summarize(req, mask, counts, positives, known_counts)}
        start += size

    daily = []
    for day in range(1, 38):
        daily.append({"day": day, **{name: summarize(req, (req["role"] == role) & (req["day"] == day) & valid, counts, positives, known_counts) for role, name in [(0, "development"), (1, "admission"), (2, "final")]}})

    for name, spec in cfg["scales"].items():
        plan = []
        for ver in cfg["training"]["versions"]:
            a, b = ver["fit_days_inclusive"]
            mask = (req["role"] == 0) & (req["day"] >= a) & (req["day"] <= b) & valid
            nreq, ntarget = int(mask.sum()), int(counts[mask].sum())
            epochs = cfg["training"]["base_epochs"] if ver["name"] == "v0" else cfg["training"]["update_epochs"]
            plan.append({**ver, "requests_per_epoch": nreq, "targets_per_epoch": ntarget, "epochs": epochs, "ideal_request_steps": epochs * math.ceil(nreq / spec["global_request_batch"])})
        steps = sum(v["ideal_request_steps"] for v in plan)
        model_results[name]["training_plan"] = plan
        model_results[name]["permanent_epoch_checkpoint_count"] = sum(v["epochs"] for v in plan)
        model_results[name]["permanent_epoch_model_adam_gb"] = 12 * total * sum(v["epochs"] for v in plan) / 1e9
        model_results[name]["permanent_epochs_plus_two_rolling_model_adam_gb"] = 12 * total * (sum(v["epochs"] for v in plan) + 2) / 1e9
        model_results[name]["training_request_visits"] = sum(v["requests_per_epoch"] * v["epochs"] for v in plan)
        model_results[name]["training_target_visits"] = sum(v["targets_per_epoch"] * v["epochs"] for v in plan)
        model_results[name]["ideal_optimizer_steps"] = steps
        model_results[name]["conditional_4xa40_training_hours"] = [steps * value / 3600 for value in spec["conditional_seconds_per_request_batch_4xa40"]]
        model_results[name]["conditional_4xa40_schedule_hours"] = [hours * cfg["estimate_scope"]["runtime_margin_multiplier"] for hours in model_results[name]["conditional_4xa40_training_hours"]]
        naive_steps = sum(v["epochs"] * math.ceil(v["targets_per_epoch"] / spec["global_request_batch"]) for v in plan)
        model_results[name]["conditional_4xa40_ungrouped_schedule_hours"] = [naive_steps * value / 3600 * cfg["estimate_scope"]["runtime_margin_multiplier"] for value in spec["conditional_seconds_per_request_batch_4xa40"]]

    output = {
        "status": "cpu_planning_statistics_not_training_or_quality_results",
        "inputs": {"configuration": str(args.config), "configuration_sha256": digest(args.config), "script_sha256": digest(Path(__file__)), "prepared_manifest_sha256": digest(root / "manifest.json")},
        "input_shapes": {"events": len(events), "requests": len(req), "initial_catalog_videos": len(support), "initial_videos_with_eligible_development_targets": int((support > 0).sum())},
        "catalogs": catalog_results,
        "models": model_results,
        "roles": roles,
        "admission_reservations": reservations,
        "daily": daily,
        "timing_note": "4xA40 conditional assumptions, not measured RecFlow training time; 201/A800 time remains unmeasured. Grouped prefix/query adapter is not implemented by this script.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "parameters": {name: m["parameters"] for name, m in model_results.items()}, "foundation_data": roles["development"]["periods"]["d1_d18"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Conditional dev NLL with original, shortened, and empty histories; no fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from development_probe import (
    ROOT,
    HSTUConfig,
    PreparedRecFlow,
    ProbeData,
    RecFlowGenerator,
    save_json,
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def truncate_history(batch: dict, keep: int) -> dict:
    """Preserve retained event deltas and the real request gap, including at zero history."""
    result = dict(batch)
    lengths = batch["lengths"].clamp_max(keep)
    width = max(int(lengths.max()), 1)
    columns = torch.arange(width, device=lengths.device)[None]
    positions = (batch["lengths"] - lengths)[:, None] + columns
    positions = positions.clamp_max(batch["item_ids"].shape[1] - 1)
    valid = columns < lengths[:, None]
    for key in ("item_ids", "behaviors", "time_deltas"):
        result[key] = batch[key].gather(1, positions).masked_fill(~valid, 0)
    result["lengths"] = lengths
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/recflow/development/grid_6l_h192_c128_k200k_seed17/development_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/history_grid_c128_k200k_seed17")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--day", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Locally produced trusted checkpoint; no optimizer or fitting is used.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["configuration"]
    torch.manual_seed(saved["seed"])
    prepared = PreparedRecFlow(saved["data"])
    dataset = ProbeData(prepared, saved["catalog_size"], saved["cohort_users"], saved["context"])
    catalog_hash = hashlib.sha256(dataset.raw_ids.tobytes()).hexdigest()
    cohort_hash = hashlib.sha256(np.asarray(dataset.uids).tobytes()).hexdigest()
    if catalog_hash != saved["catalog_sha256"] or cohort_hash != saved["cohort_sha256"]:
        raise ValueError("Checkpoint and reconstructed catalog/cohort differ")
    model = RecFlowGenerator(
        HSTUConfig(**saved["model"]), torch.tensor(dataset.paths),
        history_categories=saved.get("history_categories", False),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    del checkpoint
    selected = dataset.indices(args.day, args.day, args.requests)
    kept, target_counts, case_requests, targets = [], [], [], []
    total_positives, mixed_oov_requests = 0, 0
    for index in selected:
        raw, known = dataset.targets(index)
        total_positives += len(raw)
        if not len(known):
            continue
        mixed_oov_requests += int(len(known) < len(raw))
        kept.append(int(index))
        target_counts.append(len(known))
        case_requests.extend([len(kept) - 1] * len(known))
        targets.extend(known.tolist())
    if not kept:
        raise ValueError("The diagnostic panel has no known positives")
    kept, targets = np.asarray(kept), np.asarray(targets)
    case_requests = np.asarray(case_requests)
    short = min(128, saved["context"] // 4)
    sources = [Path(__file__).resolve(), ROOT / "scripts/recflow/development_probe.py",
               ROOT / "src/hstu_kvcache/recflow/model.py", ROOT / "src/hstu_kvcache/recflow/data.py"]
    config = {
        "scope": "Development-only teacher-forced conditional NLL diagnostic; not free retrieval, model admission, or a training repeat.",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "saved_training_configuration": saved, "day": args.day,
        "catalog_sha256": catalog_hash, "cohort_sha256": cohort_hash,
        "selected_request_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "selected_positive_requests": len(selected), "retained_known_positive_requests": len(kept),
        "excluded_all_oov_requests": len(selected) - len(kept),
        "retained_requests_with_some_oov_targets": mixed_oov_requests,
        "all_positive_targets": total_positives, "known_positive_targets": len(targets),
        "excluded_oov_targets": total_positives - len(targets),
        "original_history_limit": saved["context"], "short_history_limit": short,
        "empty_history_length": 0, "model_max_seq_len_unchanged": model.backbone.cfg.max_seq_len,
        "aggregation": "All known positives are scored. Average token NLL across positives within each request, then average requests; joint NLL sums the three token NLLs.",
        "controls": "Each request uses identical targets and real request-to-last-event time gap in every condition. Shortened histories retain original event deltas. Same model parameters, max_seq_len normalization, and FP32 precision throughout.",
        "input_variant": "Current data.py preserves original elapsed deltas after rolling eviction; historical checkpoints trained before that fix reset the first retained delta. Saved/current source hashes identify the variant. No historical quality score is overwritten.",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "device": args.device, "precision": "FP32; TF32 disabled", "batch_size": args.batch_size,
    }
    save_json(args.output / "configuration.json", config)
    print(json.dumps({k: v for k, v in config.items() if k not in ("saved_training_configuration", "source_sha256")}), flush=True)
    arrays = {name: np.empty((len(targets), 3), dtype=np.float64) for name in ("original", "short", "empty")}
    elapsed = dict.fromkeys(arrays, 0.0)
    original_lengths = np.zeros(len(kept), dtype=np.int64)
    for start in range(0, len(targets), args.batch_size):
        stop = min(start + args.batch_size, len(targets))
        batch = dataset.batch(kept[case_requests[start:stop]], args.device)
        batch["targets"] = torch.as_tensor(targets[start:stop], device=args.device)
        original_lengths[case_requests[start:stop]] = batch["lengths"].cpu().numpy()
        for name, keep in (("original", None), ("short", short), ("empty", 0)):
            inputs = batch if keep is None else truncate_history(batch, keep)
            torch.cuda.synchronize(args.device)
            begin = time.monotonic()
            values = -model.teacher_log_probs(**inputs)
            arrays[name][start:stop] = values.double().cpu().numpy()
            elapsed[name] += time.monotonic() - begin
    names = ["c1", "c2", "leaf", "joint"]
    per_request, results, deltas = {}, {}, {}
    for mode, values in arrays.items():
        grouped = np.zeros((len(kept), 3))
        np.add.at(grouped, case_requests, values)
        grouped /= np.asarray(target_counts)[:, None]
        grouped = np.column_stack([grouped, grouped.sum(-1)])
        per_request[mode] = grouped
        results[mode] = {"mean_nll": dict(zip(names, grouped.mean(0).tolist(), strict=True)),
                         "forward_seconds": elapsed[mode]}
        if mode != "original":
            diff = grouped - per_request["original"]
            deltas[mode] = {
                "mean_nll_increase": dict(zip(names, diff.mean(0).tolist(), strict=True)),
                "median_joint_increase": float(np.median(diff[:, -1])),
                "joint_increase_p10_p90": np.quantile(diff[:, -1], [.1, .9]).tolist(),
                "fraction_requests_joint_worse": float(np.mean(diff[:, -1] > 1e-6)),
            }
    summary = {"configuration": config, "conditions": results, "paired_delta_vs_original": deltas,
               "original_history_lengths": {"min": int(original_lengths.min()), "median": float(np.median(original_lengths)), "max": int(original_lengths.max())}}
    save_json(args.output / "summary.json", summary)
    np.savez_compressed(args.output / "request_nll.npz", selected_indices=selected, retained_indices=kept,
                        uids=prepared.requests["uid"][kept], target_counts=np.asarray(target_counts),
                        original_history_lengths=original_lengths, **per_request)
    print(json.dumps({k: v for k, v in summary.items() if k != "configuration"}), flush=True)


if __name__ == "__main__":
    main()

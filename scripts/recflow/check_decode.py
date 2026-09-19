#!/usr/bin/env python3
"""Compare free beam rankings against all category paths on a fixed dev panel."""

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
    aggregate_metrics,
    request_metrics,
    save_json,
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/recflow/development/grid_6l_h192_c128_k200k_seed17/development_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recflow/development/decode_grid_c128_k200k_seed17")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--exact-bound", action="store_true")
    parser.add_argument("--branch-batch-size", type=int, default=32)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    if args.fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
    # This is a locally produced, trusted research checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["configuration"]
    torch.manual_seed(saved["seed"])
    np.random.seed(saved["seed"])
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
    indices = dataset.indices(19, 19, args.requests)
    full_width = len(model._leaf_spans)
    sources = [Path(__file__).resolve(), ROOT / "scripts/recflow/development_probe.py",
               ROOT / "src/hstu_kvcache/recflow/model.py", ROOT / "src/hstu_kvcache/recflow/data.py",
               ROOT / "src/hstu_kvcache/recflow/metrics.py"]
    config = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "saved_training_configuration": saved, "device": args.device,
        "requests": len(indices), "request_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "day": 19, "catalog_sha256": catalog_hash, "cohort_sha256": cohort_hash,
        "beam_widths": [100, 300, full_width], "full_category_pairs": full_width,
        "include_exact_bound": args.exact_bound, "branch_batch_size": args.branch_batch_size,
        "first_categories": model.num_c1, "topk": 100,
        "primary_criterion": "Label-independent Top-100 set overlap with exhaustive category-path decoding.",
        "scope": "Development decoder accuracy check, not a training run or release admission.",
        "input_variant": "Current data.py preserves the original first retained event delta after rolling eviction; the saved training run reset that delta to zero. All beams here use the same corrected input, so this isolates decoder truncation but is not an exact reproduction of the historical quality panel.",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "precision": "FP32, TF32 disabled" if args.fp32 else "FP32 parameters, BF16 CUDA autocast, FP32 log probabilities",
        "torch_version": torch.__version__,
    }
    save_json(args.output / "configuration.json", config)
    print(json.dumps({k: v for k, v in config.items() if k not in ("saved_training_configuration", "source_sha256")}), flush=True)
    first = dataset.batch(indices[:1], args.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
        model.generate_topk(**first, k=100, beam_width=100)
    torch.cuda.synchronize(args.device)

    rankings, score_panels, results = {}, {}, {}
    uids = prepared.requests["uid"][indices].tolist()
    modes = [("beam100", 100), ("beam300", 300)]
    if args.exact_bound:
        modes.append(("exact_bound", None))
    modes.append(("all_pairs", full_width))
    for name, width in modes:
        torch.cuda.reset_peak_memory_stats(args.device)
        rows, ids_rows, scores_rows = [], [], []
        expanded_pairs = []
        torch.cuda.synchronize(args.device)
        start = time.monotonic()
        for number, index in enumerate(indices):
            batch = dataset.batch([index], args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                if name == "exact_bound":
                    ids, scores, stats = model.generate_exact_topk(
                        **batch, k=100, branch_batch_size=args.branch_batch_size, return_stats=True,
                    )
                    expanded_pairs.extend(stats["expanded_pairs"])
                else:
                    ids, scores = model.generate_topk(**batch, k=100, beam_width=width)
            local_ids = ids[0].cpu().numpy()
            valid = (local_ids > 0) & (local_ids < dataset.oov)
            raw_ids = dataset.raw_ids[local_ids[valid] - 1]
            positives, _ = dataset.targets(index)
            rows.append(request_metrics(raw_ids, positives, dataset.catalog_set))
            ids_rows.append(raw_ids)
            scores_rows.append(scores[0].float().cpu().numpy()[valid])
            if (number + 1) % 16 == 0:
                print(json.dumps({"mode": name, "requests": number + 1, "seconds": time.monotonic() - start}), flush=True)
        torch.cuda.synchronize(args.device)
        elapsed = time.monotonic() - start
        rankings[name], score_panels[name] = ids_rows, scores_rows
        results[name] = {
            "beam_width": width, "seconds": elapsed, "requests_per_second": len(indices) / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
            "metrics": aggregate_metrics(rows, uids),
        }
        if expanded_pairs:
            results[name].update(
                expanded_pairs=expanded_pairs, mean_expanded_pairs=float(np.mean(expanded_pairs)),
                mean_expanded_fraction=float(np.mean(expanded_pairs) / full_width),
            )
    comparisons = {}
    exact = rankings["all_pairs"]
    for name in [mode for mode, _ in modes if mode != "all_pairs"]:
        overlap = [len(set(approx) & set(full)) / len(full)
                   for approx, full in zip(rankings[name], exact, strict=True)]
        comparisons[name] = {
            "mean_top100_overlap": float(np.mean(overlap)), "minimum_top100_overlap": float(min(overlap)),
            "overlap_quantiles": dict(zip(["p10", "p50", "p90"], np.quantile(overlap, [.1, .5, .9]).tolist(), strict=True)),
            "requests_with_missing_exact_items": int(np.sum(np.asarray(overlap) < 1)),
            "per_request_overlap": overlap,
        }
        same_rankings = 0
        max_common_score_error = 0.0
        for row, full in enumerate(exact):
            same_rankings += int(np.array_equal(rankings[name][row], full))
            scores_by_id = dict(zip(rankings[name][row], score_panels[name][row], strict=True))
            for item, score in zip(full, score_panels["all_pairs"][row], strict=True):
                if item in scores_by_id:
                    max_common_score_error = max(max_common_score_error, abs(float(score - scores_by_id[item])))
        comparisons[name].update(
            requests_with_identical_order=same_rankings,
            max_common_item_logprob_error=max_common_score_error,
        )
    summary = {"configuration": config, "decode": results, "against_all_pairs": comparisons}
    save_json(args.output / "summary.json", summary)
    evidence = [{"request_index": int(index), "uid": int(uids[row]),
                 "rankings": {name: values[row].tolist() for name, values in rankings.items()},
                 "scores": {name: values[row].tolist() for name, values in score_panels.items()}}
                for row, index in enumerate(indices)]
    save_json(args.output / "rankings.json", evidence)
    print(json.dumps({"against_all_pairs": comparisons, "decode": results}), flush=True)


if __name__ == "__main__":
    main()

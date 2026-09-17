#!/usr/bin/env python3
"""Motivation 2 (formerly Insight 1) competitors on the current model chains.

Select Medium/6L or Large/10L explicitly. --describe resolves current model and
data provenance without loading weights, user histories, or a CUDA device.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design.competitor_data import (  # noqa: E402
    history_arrays,
    make_candidate_panel,
    read_uid_split,
)
from design.competitor_models import load_model_pair  # noqa: E402
from design.competitor_probe import evaluate_batch, path_records, summarize_scores  # noqa: E402
from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import sha256_file  # noqa: E402

from hstu_kvcache.baselines.kv_translate import fit  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402


def history_tensors(arrays: tuple[np.ndarray, ...], start: int, stop: int, device):
    return tuple(torch.as_tensor(values[start:stop], device=device) for values in arrays[1:4])


@torch.no_grad()
def collect_cache_pair(parent, current, arrays, batch_size: int, device):
    """Generate aligned calibration caches, retained and fitted on CPU."""
    pairs = [[], []]
    for start in range(0, len(arrays[1]), batch_size):
        events = history_tensors(arrays, start, start + batch_size, device)
        for parts, model in zip(pairs, (parent, current), strict=True):
            parts.append(model.compute_kv(*events).to("cpu"))
    return tuple(
        HSTUKVCache(torch.cat([part.k for part in parts], dim=1),
                    torch.cat([part.v for part in parts], dim=1), parts[0].seq_len)
        for parts in pairs
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large"), required=True)
    parser.add_argument("--edge-index", type=int, choices=range(5), required=True,
                        help="0=v0 to v1, ..., 4=v4 to v5 of the current chain")
    parser.add_argument("--describe", action="store_true", help="read model/data metadata only, then exit")
    parser.add_argument("--uids", type=Path, help='JSON {"evaluation":[UIDs],"fit":[UIDs],"selection":[UIDs]}')
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-users", type=int, default=32)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--candidate-chunk", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--history-threads", type=int, default=4)
    parser.add_argument("--layer-intervals", default=None,
                        help="inclusive zero-based pairs; default first, last-two, full; empty disables LR")
    parser.add_argument("--tail-lengths", default=None,
                        help="comma-separated; default 0,context/8,context/4,context; empty disables TR")
    parser.add_argument("--map-ks", default="1,2", help="comma-separated; empty disables KT")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", type=Path, help="new diagnostic directory; existing evidence is never overwritten")
    return parser


def run(args: argparse.Namespace) -> dict:
    pair = load_model_pair(args.scale, args.edge_index, verify_hashes=False)
    if args.describe:
        print(json.dumps(pair, indent=2, allow_nan=False))
        return pair
    if not pair["admission"]["reuse_eligible"]:
        raise RuntimeError("current chain selection does not establish Reuse admission for this edge; see --describe")
    if args.uids is None or args.output is None:
        raise ValueError("execution requires --uids and --output")
    layers, history_length = pair["config"]["num_layers"], pair["config"]["max_seq_len"]
    intervals = ([(0, 0), (layers - 2, layers - 1), (0, layers - 1)]
                 if args.layer_intervals is None else
                 [tuple(int(x) for x in value.split(":")) for value in args.layer_intervals.split(",") if value])
    tails = ([0, history_length // 8, history_length // 4, history_length]
             if args.tail_lengths is None else [int(value) for value in args.tail_lengths.split(",") if value])
    ks = [int(value) for value in args.map_ks.split(",") if value]
    path_records(intervals, tails, {}, history_length, layers)
    if len(set(ks)) != len(ks) or any(not 1 <= k <= layers for k in ks):
        raise ValueError("map k values must be distinct and within the selected model's layer count")
    if not np.isfinite(args.ridge) or args.ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if min(args.batch_size, args.candidate_chunk, args.torch_threads, args.history_threads) < 1:
        raise ValueError("batch, chunk and thread counts must be positive")
    split = read_uid_split(args.uids)
    if ks and not split["fit"]:
        raise ValueError("KT requires fitting users separate from the entire evaluation population")
    if args.eval_offset < 0 or not 1 <= args.max_users <= len(split["evaluation"]) - args.eval_offset:
        raise ValueError("evaluation slice must be nonempty and within the supplied UID population")
    output = args.output.resolve()
    old_results = ROOT / "results/yambda500m_medium_seed17/insight1_locality_v1"
    if output.exists() or output.is_relative_to(old_results):
        raise ValueError("choose a new output directory outside sealed Insight 1 results")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this entry supports explicit CPU or CUDA devices")
    torch.set_num_threads(args.torch_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    pair = load_model_pair(args.scale, args.edge_index, verify_hashes=True)
    if not pair["admission"]["reuse_eligible"]:
        raise RuntimeError("verified edge has not passed Reuse admission")
    needed_uids = split["evaluation"] + (split["fit"] + split["selection"] if ks else [])
    population = set(pq.read_table(pair["dataset"]["users"], columns=["uid"])["uid"].to_pylist())
    if set(needed_uids) - population:
        raise ValueError("UID split contains users outside the selected scale's population")
    cutover = pair["cutover"]
    history = load_histories(
        needed_uids, dataset_path=Path(pair["dataset"]["manifest"]),
        known_vocab_size=pair["dataset"]["known_items"], oov_buckets=pair["dataset"]["oov_buckets"],
        end_timestamp=cutover, threads=args.history_threads,
    )
    # One shared, label-free bank from the ENTIRE supplied evaluation population.
    # Changing batching/max-users never silently changes the candidate panels.
    all_arrays = history_arrays(history, np.asarray(split["evaluation"]), cutover, history_length)
    all_candidates, all_modes = make_candidate_panel(all_arrays[1], pair["dataset"]["known_items"])
    selected = slice(args.eval_offset, args.eval_offset + args.max_users)
    uids = np.asarray(split["evaluation"], dtype=np.int64)[selected]
    evaluation = tuple(values[selected] for values in all_arrays)
    candidates, candidate_modes = all_candidates[selected], all_modes[selected]
    del all_arrays, all_candidates, all_modes, population

    models = []
    for role in ("parent", "current"):
        model, payload = load_model(Path(pair[role]["checkpoint"]), device)
        if payload["config"] != pair["config"]:
            raise ValueError(f"{role} checkpoint config differs from current chain metadata")
        model.requires_grad_(False)
        models.append(model)
        del payload
    parent, current = models
    translators = {}
    if ks:
        print(f"Collect calibration: fit={len(split['fit'])}, selection={len(split['selection'])} users", flush=True)
        arrays = history_arrays(history, np.asarray(split["fit"]), cutover, history_length)
        source, target = collect_cache_pair(parent, current, arrays, args.batch_size, device)
        selection_source = selection_target = None
        if split["selection"]:
            arrays = history_arrays(history, np.asarray(split["selection"]), cutover, history_length)
            selection_source, selection_target = collect_cache_pair(parent, current, arrays, args.batch_size, device)
        for k in ks:
            print(f"Fit shared kt_k{k} on CPU", flush=True)
            translators[f"kt_k{k}"] = fit(
                source, target, num_heads=current.cfg.num_heads, k=k, ridge=args.ridge,
                selection_source=selection_source, selection_target=selection_target,
            )
        del arrays, source, target, selection_source, selection_target

    records = path_records(intervals, tails, translators, history_length, layers)
    pieces = {record["config_id"]: [] for record in records}
    for start in range(0, len(uids), args.batch_size):
        stop = min(start + args.batch_size, len(uids))
        scores = evaluate_batch(
            parent, current, *history_tensors(evaluation, start, stop, device),
            torch.as_tensor(candidates[start:stop], device=device),
            torch.as_tensor(evaluation[4][start:stop], device=device),
            layer_intervals=intervals, tail_lengths=tails, translators=translators,
            candidate_chunk=args.candidate_chunk,
        )
        for name, values in scores.items():
            pieces[name].append(values.float().cpu().numpy())
        print(f"{args.scale}/{pair['edge']}: scored {stop}/{len(uids)} users", flush=True)
    scores_np = {name: np.concatenate(values) for name, values in pieces.items()}
    rows = summarize_scores(scores_np, pair["edge"], records)
    summary = {
        "status": "motivation2_single_edge_functional_diagnostic", "scale": args.scale, "edge": pair["edge"],
        "cutover": cutover, "history_length": history_length,
        "settings": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "model_pair": pair, "uid_split_sha256": sha256_file(args.uids), "torch_version": torch.__version__,
        "model_dtype": str(next(current.parameters()).dtype),
        "candidate_panel": {"rule": "insight1_recent16_old16_novel_bank", "population_users": len(split["evaluation"]),
                            "scope": "new explicit per-scale UID panel; old frozen panels/results unchanged"},
        "calibration": {
            "fit": split["fit"] if ks else [], "selection": split["selection"] if ks else [],
            "fit_device": "cpu", "ridge": args.ridge,
            "teacher_users": len(split["fit"]) + len(split["selection"]) if ks else 0,
            "teacher_tokens": (len(split["fit"]) + len(split["selection"])) * history_length if ks else 0,
            "excludes_entire_evaluation_population": True,
            "selection_scores": {name: mapper.selection_scores.tolist() for name, mapper in translators.items()},
        },
        "paths": records, "metrics": rows,
        "interpretation": "Probability-gap recovery; not AUC, a FLOPs benchmark, or continuous adaptation. No winner selected.",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "scores.npz", uids=uids, path_ids=np.asarray(list(scores_np)),
        scores=np.stack(list(scores_np.values()), axis=1), candidates=candidates,
        candidate_modes=candidate_modes, query_deltas=evaluation[4],
        panel_population_uids=np.asarray(split["evaluation"], dtype=np.int64),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "parameters": json.dumps(row["parameters"], sort_keys=True)})
    return summary


if __name__ == "__main__":
    run(build_parser().parse_args())

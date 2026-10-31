"""FP32 complete-candidate native quality and adjacent-version snapshot reuse."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .batches import RequestBatches
from .data_common import json_save, sha256
from .model import RequestRanker
from .protocol import paired_uid_interval, request_metrics
from .runtime import distributed_init, load_checkpoint, source_binding, to_device


def baseline_metrics(ids, labels, popularity):
    scores = np.zeros(len(ids))
    known = ids < len(popularity)
    scores[known] = popularity[ids[known]]
    result = request_metrics(ids, scores, labels)
    positives = int((labels == 1).sum())
    random = {}
    for k in [5, 10, 20]:
        discounts = 1/np.log2(np.arange(2, min(k, len(ids))+2))
        random[f"ndcg@{k}"] = float(positives/len(ids)*discounts.sum()/discounts[:min(k, positives)].sum()) if positives else None
        random[f"recall@{k}"] = min(k, len(ids))/len(ids) if positives else None
    return result, random


def summarize(rows, methods):
    valid = [row for row in rows if row["positives"] > 0]
    result = {"requests": len(rows), "positive_requests": len(valid),
              "zero_positive_requests": len(rows)-len(valid), "users": len(set(row["uid"] for row in rows)),
              "positive_users": len(set(row["uid"] for row in valid)), "metrics": {}}
    for method in methods:
        result["metrics"][method] = {f"{metric}@{k}": float(np.mean([r[f"{method}_{metric}@{k}"] for r in valid])) if valid else None
                                      for metric in ["ndcg", "recall"] for k in [5, 10, 20]}
        loss_rows = [r[f"{method}_observed_bce"] for r in rows if r.get(f"{method}_observed_bce") is not None]
        if loss_rows:
            result["metrics"][method]["observed_bce"] = float(np.mean(loss_rows))
    if valid:
        result["paired_current_minus"] = {method: paired_uid_interval(
            [r["uid"] for r in valid], [r["current_ndcg@10"] for r in valid],
            [r[f"{method}_ndcg@10"] for r in valid]) for method in methods if method != "current"}
    return result


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--days", type=int, nargs="+", required=True)
    p.add_argument("--role", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--first-only", action="store_true")
    p.add_argument("--allow-confirmation", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--history-limit", type=int, choices=[128, 1024], default=1024)
    args = p.parse_args()
    if args.role == 3 and not (args.data / "formal_recipe_freeze.json").exists():
        raise ValueError("Confirmation requires an actual formal recipe freeze artifact")
    rank, world, device = distributed_init(17)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / f"rank_{rank}.csv").exists():
        raise ValueError("Existing scores must not be overwritten")
    payload, sidecar = load_checkpoint(args.checkpoint)
    if payload.get("quality_use_permitted") is False:
        raise ValueError("Diagnostic profiling state cannot produce quality evidence")
    if args.role != 1 and min(args.days) <= max(payload["days"]):
        raise ValueError("Development/confirmation metrics must be strictly after training cutoff")
    vocabulary = "vocabulary_max" if payload["scale"] == "max" else "vocabulary"
    if payload["vocabulary_sha256"] != sha256(args.data / vocabulary / "manifest.json"):
        raise ValueError("Vocabulary changed")
    current_binding = source_binding(args.data, vocabulary)
    if payload["source_binding"] != current_binding:
        raise ValueError("Checkpoint numerical sources or data binding changed")
    model = RequestRanker(args.data / vocabulary, payload["scale"], payload["proposal"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    expected_parent = payload.get("parent_checkpoint_sha256")
    del payload
    parent, parent_sidecar = None, None
    if args.parent:
        parent_payload, parent_sidecar = load_checkpoint(args.parent)
        if parent_sidecar["checkpoint_sha256"] != expected_parent:
            raise ValueError("Supplied parent is not the current checkpoint's actual training parent")
        if parent_payload["vocabulary_sha256"] != sha256(args.data / vocabulary / "manifest.json"):
            raise ValueError("Parent vocabulary differs")
        parent = RequestRanker(args.data / vocabulary, parent_payload["scale"], parent_payload["proposal"]).to(device)
        parent.load_state_dict(parent_payload["model"], strict=True)
        parent.eval()
        del parent_payload
    dataset = RequestBatches(args.data, args.days, role=args.role, first_only=args.first_only,
                             limit=args.limit, allow_confirmation=args.allow_confirmation, history_limit=args.history_limit)
    local_indices = np.arange(rank, len(dataset), world)
    batches = [local_indices[i:i+args.batch_size].tolist() for i in range(0, len(local_indices), args.batch_size)]
    options = {"multiprocessing_context": "spawn", "persistent_workers": True, "prefetch_factor": 2} if args.workers else {}
    loader = DataLoader(dataset, batch_size=None, sampler=batches, num_workers=args.workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(17+rank), **options)
    popularity = np.load(args.data / vocabulary / "foundation_effective_popularity.npy", mmap_mode="r")
    methods = ["current", "popularity", "random"] + (["parent", "reuse"] if parent is not None else [])
    rows = []
    score_dir = args.output / f"rank_{rank}_scores"
    score_dir.mkdir(exist_ok=False)
    with torch.inference_mode():
        for batch_number, cpu in enumerate(loader):
            batch = to_device(cpu, device)
            current_cache = model.encode_history(batch)
            scores = {"current": model.read(batch, current_cache).cpu().numpy()}
            if parent is not None:
                parent_cache = parent.encode_history(batch)
                scores["parent"] = parent.read(batch, parent_cache).cpu().numpy()
                scores["reuse"] = model.read(batch, parent_cache).cpu().numpy()
            np.savez(score_dir / f"batch_{batch_number:06d}.npz", uid=cpu["uid"].numpy(),
                     rid=cpu["rid"].numpy(), day=cpu["day"].numpy(), counts=cpu["counts"].numpy(),
                     raw_video_ids=cpu["candidate_features"][..., 0].numpy(), labels=cpu["labels"].numpy(),
                     **scores)
            for j in range(len(cpu["uid"])):
                count = int(cpu["counts"][j])
                ids = cpu["candidate_features"][j, :count, 0].numpy()
                labels = cpu["labels"][j, :count].numpy()
                record = {"uid": int(cpu["uid"][j]), "rid": int(cpu["rid"][j]), "day": int(cpu["day"][j]),
                          "candidates": count, "exposed": int((labels >= 0).sum()), "positives": int((labels == 1).sum()),
                          "history_length": int(cpu["lengths"][j])}
                evaluated = {name: request_metrics(ids, value[j, :count], labels) for name, value in scores.items()}
                evaluated["popularity"], evaluated["random"] = baseline_metrics(ids, labels, popularity)
                for name, value in evaluated.items():
                    record.update({f"{name}_{metric}@{k}": value[f"{metric}@{k}"] for metric in ["ndcg", "recall"] for k in [5, 10, 20]})
                observed = labels >= 0
                for name, values in scores.items():
                    logits = values[j, :count][observed].astype(np.float64)
                    record[f"{name}_observed_bce"] = float(np.mean(np.logaddexp(0, logits)-labels[observed]*logits)) if len(logits) else None
                rows.append(record)
    columns = ["uid", "rid", "day", "candidates", "exposed", "positives", "history_length"]
    columns += [f"{method}_{metric}@{k}" for method in methods for metric in ["ndcg", "recall"] for k in [5, 10, 20]]
    columns += [f"{method}_observed_bce" for method in methods if method not in ["popularity", "random"]]
    with (args.output / f"rank_{rank}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if world > 1:
        dist.barrier()
    if rank == 0:
        all_rows = []
        for index in range(world):
            with (args.output / f"rank_{index}.csv").open() as handle:
                for row in csv.DictReader(handle):
                    all_rows.append({key: (int(value) if key in columns[:7] else float(value)) if value else None
                                     for key, value in row.items()})
        keys = {(r["uid"], r["day"], r["rid"]) for r in all_rows}
        if len(all_rows) != len(dataset) or len(keys) != len(dataset):
            raise ValueError("Distributed evaluation omitted or duplicated requests")
        summary = summarize(all_rows, methods)
        summary.update(status="passed", checkpoint_sha256=sidecar["checkpoint_sha256"],
                       parent_sha256=parent_sidecar["checkpoint_sha256"] if parent_sidecar else None,
                       days=args.days, role=args.role, first_only=args.first_only, history_limit=args.history_limit,
                       precision="float32_parameters_and_KV_tf32_disabled", world_size=world,
                       source_binding=current_binding, evaluator_sha256=sha256(__file__),
                       wall_seconds=time.monotonic()-started,
                       arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        json_save(args.output / "summary.json", summary)
        print(json.dumps(summary["metrics"]), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Separate existing-user temporal holdout using the original FP32 score paths.

Statically derived from evaluate.py; its original sources and results are frozen.
The new population selector retains full requests including no-feedback requests.
"""
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
from .evaluate import baseline_metrics, summarize
from .model import RequestRanker
from .protocol import request_metrics
from .runtime import distributed_init, load_checkpoint, source_binding, to_device
from .temporal_cohort import PROTOCOL, TemporalCohortBatches, read_plan, validate_checkpoint_pair
from .version_chain import selection_digest


def score_three(model, parent, batch):
    current_cache = model.encode_history(batch)
    scores = {"current": model.read(batch, current_cache).cpu().numpy()}
    parent_cache = parent.encode_history(batch)
    scores["parent"] = parent.read(batch, parent_cache).cpu().numpy()
    scores["reuse"] = model.read(batch, parent_cache).cpu().numpy()
    return scores


def compare_parity(output, reference, world):
    maximum, hashes = {}, {}
    for rank in range(world):
        fresh = output / f"rank_{rank}_scores/batch_000000.npz"
        old = reference / f"rank_{rank}_scores/batch_000000.npz"
        with np.load(fresh) as new, np.load(old) as prior:
            for name in ["uid", "rid", "day", "counts", "raw_video_ids", "labels"]:
                if not np.array_equal(new[name], prior[name]):
                    raise ValueError(f"Actual parity input mismatch: {rank}/{name}")
            keep = np.arange(new["labels"].shape[1])[None, :] < new["counts"][:, None]
            for name in ["current", "parent", "reuse"]:
                errors = np.abs(new[name][keep].astype(np.float64)-prior[name][keep])
                error = float(errors.max())
                if not np.isfinite(errors).all() or error > 1e-6:
                    raise ValueError(f"Actual parity score mismatch: {rank}/{name}/{error}")
                maximum[name] = max(maximum.get(name, 0.), error)
        hashes[str(fresh)], hashes[str(old)] = sha256(fresh), sha256(old)
        with (output / f"rank_{rank}.csv").open() as handle:
            current_rows = list(csv.DictReader(handle))
        with (reference / f"rank_{rank}.csv").open() as handle:
            old_rows = list(csv.DictReader(handle))[:len(current_rows)]
        if current_rows != old_rows:
            raise ValueError("Actual first-batch metrics differ from the original evaluator")
    return {"status": "passed", "requests": 16*world, "score_max_abs": maximum,
            "quality_use_permitted": False, "source_artifact_sha256": hashes}


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--parity-reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--days", type=int, nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(role=0, first_only=False, allow_confirmation=False, limit=None, history_limit=1024)
    args = p.parse_args()
    plan = read_plan(args.plan)
    cases = [case for case in plan["schedule"] if args.days == [case["day"]]]
    if len(cases) != 1 or args.batch_size != 16 or Path(plan["data"]).resolve() != args.data.resolve():
        raise ValueError("Different window, data or batch shape from the frozen plan")
    case = cases[0]
    if (Path(case["checkpoint"]).resolve() != args.checkpoint.resolve()
            or Path(case["parent"]).resolve() != args.parent.resolve()):
        raise ValueError("Different checkpoint paths from the frozen pair")
    if args.parity_reference:
        if args.days != [28] or args.parity_reference.resolve() != Path(plan["parity_reference"]).resolve():
            raise ValueError("Parity must reuse the original first 64 development requests")
    else:
        readiness = json.loads((args.plan.parent / "readiness.json").read_text())
        if (readiness["status"] != "passed" or readiness["plan_sha256"] != sha256(args.plan)
                or sha256(readiness["parity_report"]) != readiness["parity_sha256"]):
            raise ValueError("Actual parity and CPU readiness are required before cohort scores")
    rank, world, device = distributed_init(17)
    if world != 4:
        raise ValueError("The frozen protocol uses all four GPUs")
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
    current_metadata = {key: payload[key] for key in ["parent_checkpoint_sha256", "days", "scale", "quality_use_permitted"]}
    del payload
    parent, parent_sidecar = None, None
    if args.parent:
        parent_payload, parent_sidecar = load_checkpoint(args.parent)
        if parent_sidecar["checkpoint_sha256"] != expected_parent:
            raise ValueError("Supplied parent is not the current checkpoint's actual training parent")
        if parent_payload["vocabulary_sha256"] != sha256(args.data / vocabulary / "manifest.json"):
            raise ValueError("Parent vocabulary differs")
        validate_checkpoint_pair(current_metadata, parent_payload, sidecar["checkpoint_sha256"],
                                 parent_sidecar["checkpoint_sha256"], case)
        if parent_payload["source_binding"] != current_binding:
            raise ValueError("Parent numerical sources or data changed")
        parent = RequestRanker(args.data / vocabulary, parent_payload["scale"], parent_payload["proposal"]).to(device)
        parent.load_state_dict(parent_payload["model"], strict=True)
        parent.eval()
        del parent_payload
    if args.parity_reference:
        dataset = RequestBatches(args.data, args.days, role=2, history_limit=1024)
    else:
        dataset = TemporalCohortBatches(args.data, args.days, plan["population"])
        if len(dataset) != case["requests"] or selection_digest(dataset.records) != case["selection_sha256"]:
            raise ValueError("Actual future request population differs from the frozen plan")
    local_indices = np.arange(rank, len(dataset), world)
    batches = [local_indices[i:i+args.batch_size].tolist() for i in range(0, len(local_indices), args.batch_size)]
    if args.parity_reference:
        batches = batches[:1]
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
            scores = score_three(model, parent, batch)
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
    if args.parity_reference:
        if rank == 0:
            result = compare_parity(args.output, args.parity_reference, world)
            result.update(plan_sha256=sha256(args.plan), evaluator_sha256=sha256(__file__),
                          checkpoint_sha256=sidecar["checkpoint_sha256"],
                          parent_sha256=parent_sidecar["checkpoint_sha256"],
                          wall_seconds=time.monotonic()-started)
            json_save(args.output / "parity.json", result)
        if world > 1:
            dist.destroy_process_group()
        return
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
        summary.update(protocol=PROTOCOL, plan_sha256=sha256(args.plan),
                       population_sha256=sha256(plan["population"]),
                       actual_current_training_days=current_metadata["days"],
                       population_selection="fixed UID hash at D26; no future activity or outcome filtering",
                       holdout="future requests of existing training users; not unseen-user confirmation")
        json_save(args.output / "summary.json", summary)
        print(json.dumps(summary["metrics"]), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""One sampled release-interval experiment, using existing model checkpoints."""
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

from experiments.recflow_full_population.code.batches import RequestBatches
from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_full_population.code.evaluate import summarize
from experiments.recflow_full_population.code.model import RequestRanker
from experiments.recflow_full_population.code.protocol import day_bounds_ms, request_metrics
from experiments.recflow_full_population.code.runtime import distributed_init, to_device
from .interval_protocol import SampledRequestBatches, ancestry


def read(path):
    return json.loads(Path(path).read_text())


def load_model(checkpoint, data, device):
    # mmap avoids reading AdamW moments that are unnecessary for inference.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("quality_use_permitted") is False:
        raise ValueError("Expected a trained recommendation checkpoint")
    model = RequestRanker(data / "vocabulary_max", payload["scale"], payload["proposal"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    metadata = {key: payload[key] for key in ("days", "proposal", "arguments")}
    del payload
    return model, metadata


def native_reference(model, data, day, reference, output, rank, world, device):
    """Original 64 requests and original per-rank batching, before sampled scores."""
    dataset = RequestBatches(data, [day], role=2)
    cpu = dataset[list(range(rank, min(64, len(dataset)), world))]
    with torch.inference_mode():
        batch = to_device(cpu, device)
        actual = model.read(batch, model.encode_history(batch)).cpu().numpy()
    with np.load(reference / f"rank_{rank}_scores/batch_000000.npz") as old:
        for key in ("uid", "rid", "day", "counts", "labels"):
            if not np.array_equal(cpu[key].numpy(), old[key]):
                raise ValueError(f"Original diagnostic request field differs: {key}")
        if not np.array_equal(cpu["candidate_features"][..., 0].numpy(), old["raw_video_ids"]):
            raise ValueError("Original diagnostic candidate IDs differ")
        valid = np.arange(actual.shape[1])[None, :] < cpu["counts"].numpy()[:, None]
        error = float(np.max(np.abs(actual[valid] - old["current"][valid])))
        if not np.allclose(actual[valid], old["current"][valid], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Rebuild numerical diagnostic differs: {error}")
    json_save(output / f"native_reference_rank_{rank}.json",
              dict(status="passed", requests=len(actual), max_abs_score_error=error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    plan = read(output / "plan.json")
    data, checkpoint, source = map(Path, (plan["data"], plan["checkpoint"], plan["cache_source"]))
    day = plan["eval_day"]
    lineage = ancestry(checkpoint, source)
    if day != lineage[-1]["train_end_day"] + 1:
        raise ValueError("Use the complete next day after the target training cutoff")
    rank, world, device = distributed_init(17)
    started = time.monotonic()
    if (output / f"rank_{rank}.csv").exists():
        raise ValueError("Use a new experiment directory; retain existing results")
    current, current_meta = load_model(checkpoint, data, device)
    native_reference(current, data, day, Path(plan["native_reference"]), output, rank, world, device)
    ancestor, source_meta = load_model(source, data, device)
    if current_meta["proposal"] != source_meta["proposal"]:
        raise ValueError("Cache and target architectures differ")
    recorded_parent = Path(current_meta["arguments"]["parent"]).resolve()
    if recorded_parent != Path(lineage[-2]["checkpoint"]).resolve():
        raise ValueError("Loaded target's actual optimizer parent disagrees with ancestry")
    dataset = SampledRequestBatches(data, day, output / "selection.npy")
    lo, hi = day_bounds_ms(day)
    if np.any((dataset.records["ts"] < lo) | (dataset.records["ts"] >= hi)):
        raise ValueError("Request timestamp outside the complete evaluation day")
    local = np.arange(rank, len(dataset), world)
    batch_size = plan.get("batch_size", 16)
    batches = [local[i:i+batch_size].tolist() for i in range(0, len(local), batch_size)]
    loader = DataLoader(dataset, batch_size=None, sampler=batches, num_workers=2,
                        multiprocessing_context="spawn", persistent_workers=True,
                        prefetch_factor=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(17+rank))
    score_dir = output / f"rank_{rank}_scores"
    score_dir.mkdir(exist_ok=False)
    rows = []
    with torch.inference_mode():
        for number, cpu in enumerate(loader):
            batch = to_device(cpu, device)
            current_cache = current.encode_history(batch)
            rebuilt = current.read(batch, current_cache).cpu().numpy()
            del current_cache
            old_cache = ancestor.encode_history(batch)
            reused = current.read(batch, old_cache).cpu().numpy()
            del old_cache
            scores = {"current": rebuilt, "reuse": reused}
            np.savez(score_dir / f"batch_{number:06d}.npz", uid=cpu["uid"].numpy(),
                     rid=cpu["rid"].numpy(), day=cpu["day"].numpy(), counts=cpu["counts"].numpy(),
                     raw_video_ids=cpu["candidate_features"][..., 0].numpy(), labels=cpu["labels"].numpy(), **scores)
            for j in range(len(cpu["uid"])):
                count = int(cpu["counts"][j])
                ids = cpu["candidate_features"][j, :count, 0].numpy()
                labels = cpu["labels"][j, :count].numpy()
                row = {"uid": int(cpu["uid"][j]), "rid": int(cpu["rid"][j]), "day": day,
                       "candidates": count, "exposed": int((labels >= 0).sum()),
                       "positives": int((labels == 1).sum()), "history_length": int(cpu["lengths"][j])}
                for method, values in scores.items():
                    values = request_metrics(ids, values[j, :count], labels)
                    row.update({f"{method}_{metric}@{k}": values[f"{metric}@{k}"]
                                for metric in ("ndcg", "recall") for k in (5, 10, 20)})
                rows.append(row)
            if rank == 0 and number % 25 == 0:
                json_save(output / "progress.json", dict(status="running", rank0_requests=len(rows),
                          total_requests=len(dataset), wall_seconds=time.monotonic()-started))
    columns = list(rows[0]) if rows else []
    with (output / f"rank_{rank}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    dist.barrier()
    if rank == 0:
        rows = []
        integer = {"uid", "rid", "day", "candidates", "exposed", "positives", "history_length"}
        for index in range(world):
            with (output / f"rank_{index}.csv").open() as handle:
                rows.extend({k: (int(v) if k in integer else float(v)) if v else None for k, v in row.items()}
                            for row in csv.DictReader(handle))
        keys = {(r["uid"], r["day"], r["rid"]) for r in rows}
        expected = {(int(r["uid"]), int(r["day"]), int(r["rid"])) for r in dataset.records}
        if keys != expected or len(rows) != len(dataset):
            raise ValueError("Distributed results omitted or duplicated a selected request")
        summary = summarize(rows, ["current", "reuse"])
        rebuilt, reused = (summary["metrics"][name]["ndcg@10"] for name in ("current", "reuse"))
        summary.update(status="completed", relative_loss_pct=100*(rebuilt-reused)/rebuilt,
                       plan=plan, actual_training_path=lineage, world_size=world,
                       precision="float32_tf32_disabled", history_limit=1024,
                       wall_seconds=time.monotonic()-started, rank0_maximum_allocated_bytes=torch.cuda.max_memory_allocated(),
                       protocol="matched-prefix snapshot; existing daily updates, altered publication interval")
        json_save(output / "summary.json", summary)
        json_save(output / "progress.json", dict(status="scoring_completed", requests=len(rows), wall_seconds=summary["wall_seconds"]))
        print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

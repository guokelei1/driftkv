#!/usr/bin/env python3
"""One preregistered dense-supervision CC-theta0-v2 retraining run.

This entry point has one model, one fixed budget and one fresh, previously
unscored gate manifest.  It contains no theta1/theta2 or hyperparameter-search
path.  It is enabled only by P3 adjudication evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from cc_theta0_qualification import (
    BOOTSTRAP_ROUNDS,
    CANDIDATE_SIZE,
    MAX_HISTORY,
    QMAIN_DECAY,
    QMAIN_POOL_SIZE,
    QUERY_ACTION_ID,
    QUERY_TYPE_ID,
    RAW,
    RELEASE_CUTOFF,
    RESULT_DIR,
    SEED,
    TRAIN_END,
    _causal_pool,
    _sample_qmain_negatives,
    autocast_context,
    build_catalog,
    code_commit,
    digest_object,
    item_map_from_catalog,
    load_histories,
    make_model,
    read_jsonl,
    score_rows,
    sha256_file,
    summarize_metrics,
)
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = ROOT / "data/manifests/cc_theta0_v2_dense_train.jsonl"
DEV_MANIFEST = ROOT / "data/manifests/cc_theta0_v2_dev_candidates.jsonl"
QUALITY_MANIFEST = ROOT / "data/manifests/cc_theta0_v2_qualification_candidates.jsonl"
CHECKPOINT = ROOT / "checkpoints/cc_theta0_v2_dense.pt"
MAX_ANCHORS_PER_USER = 16
TIME_BINS = MAX_ANCHORS_PER_USER
MIN_HISTORY = 6


class CCScorer(torch.nn.Module):
    """Expose CC scoring as a DDP-forward-compatible module."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, items, behaviors, deltas, candidates, query_deltas, lengths):
        return self.model.score_cc_full(
            items, behaviors, deltas, candidates, query_deltas, lengths=lengths
        )


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))].mean(
        axis=1
    )
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def anchors_by_user() -> dict[int, dict[int, tuple[int, int, int]]]:
    """Take the final eligible event in each fixed foundation-time bin."""
    anchors: dict[int, dict[int, tuple[int, int, int]]] = {}
    counts: Counter[int] = Counter()
    for batch in pq.ParquetFile(RAW).iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        uid = batch.column("uid").to_numpy(zero_copy_only=False)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False)
        item = batch.column("item_id").to_numpy(zero_copy_only=False)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False)
        for user, timestamp, raw_item, org in zip(uid, ts, item, organic, strict=True):
            user, timestamp = int(user), int(timestamp)
            if timestamp >= TRAIN_END:
                continue
            counts[user] += 1
            if counts[user] < MIN_HISTORY + 1:
                continue
            bucket = min(TIME_BINS - 1, timestamp * TIME_BINS // TRAIN_END)
            value = (timestamp, int(raw_item), int(org))
            previous = anchors.setdefault(user, {}).get(bucket)
            if previous is None or timestamp > previous[0]:
                anchors[user][bucket] = value
    return anchors


def build_manifest(force: bool = False) -> dict:
    if TRAIN_MANIFEST.exists() and not force:
        return {
            "status": "reused",
            "rows": sum(1 for _ in TRAIN_MANIFEST.open()),
            "training_manifest_hash": sha256_file(TRAIN_MANIFEST),
        }
    anchors = anchors_by_user()
    selected = {
        (uid, ts, item): bucket
        for uid, choices in anchors.items()
        for bucket, (ts, item, _) in choices.items()
    }
    first_seen, catalog, _ = build_catalog()
    catalog_order = catalog[np.argsort(first_seen[catalog], kind="stable")]
    catalog_order_times = first_seen[catalog_order]
    buffers: dict[int, deque[tuple[int, int, int]]] = {}
    rows: list[dict] = []
    per_user: Counter[int] = Counter()
    for batch in pq.ParquetFile(RAW).iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        for uid, timestamp, item, organic in zip(
            *(
                batch.column(name).to_numpy(zero_copy_only=False)
                for name in ("uid", "timestamp", "item_id", "is_organic")
            ),
            strict=True,
        ):
            uid, timestamp, item, organic = int(uid), int(timestamp), int(item), int(organic)
            if timestamp >= TRAIN_END or uid not in anchors:
                continue
            history = buffers.setdefault(uid, deque(maxlen=MAX_HISTORY))
            anchor_key = (uid, timestamp, item)
            if anchor_key in selected and len(history) >= MIN_HISTORY:
                pool = _causal_pool(
                    list(history), item, timestamp, first_seen, catalog_order, catalog_order_times
                )
                try:
                    negatives = _sample_qmain_negatives(pool, uid, timestamp)
                except ValueError:
                    negatives = []
                if negatives:
                    proposal = {
                        "version": "Q_main_rank_decay_v1_cc_causal",
                        "pool_size": len(pool),
                        "decay": QMAIN_DECAY,
                        "cutoff": timestamp,
                        "negative_records": negatives,
                    }
                    rows.append(
                        {
                            "sample_id": f"cc-theta0-v2-dense-{len(rows):08d}",
                            "uid": uid,
                            "target_timestamp": timestamp,
                            "positive_item_id": item,
                            "history_max_timestamp": max(event[1] for event in history),
                            "proposal_cutoff": timestamp,
                            "query_timestamp": timestamp,
                            "query_type_id": QUERY_TYPE_ID,
                            "query_action_id": QUERY_ACTION_ID,
                            "candidate_item_ids": [item]
                            + [record["item_id"] for record in negatives],
                            "negative_records": negatives,
                            "proposal_version": proposal["version"],
                            "proposal_hash": digest_object(proposal),
                            "anchor_time_bin": selected.pop(anchor_key),
                            "seen_item_policy": {
                                "exclude_current_positive_from_negatives": True,
                                "exclude_all_historically_seen_items": False,
                            },
                        }
                    )
                    per_user[uid] += 1
            history.append((item, timestamp, 1 + (1 - organic)))
    # Equal total loss weight per user; anchor selection may leave a user with
    # fewer than 16 bins, but no user is duplicated to fill a bin.
    for row in rows:
        row["training_weight"] = 1.0 / per_user[int(row["uid"])]
    TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_MANIFEST.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "status": "created",
        "rows": len(rows),
        "users": len(per_user),
        "anchors_per_user": {
            "min": min(per_user.values()),
            "max": max(per_user.values()),
            "mean": float(np.mean(list(per_user.values()))),
        },
        "training_manifest_hash": sha256_file(TRAIN_MANIFEST),
    }


def sample_histories(rows: list[dict]) -> dict[str, list[tuple[int, int, int]]]:
    """Reconstruct only requested dense samples, keyed by sample ID."""
    wanted = {
        (int(row["uid"]), int(row["target_timestamp"]), int(row["positive_item_id"])): row[
            "sample_id"
        ]
        for row in rows
    }
    result: dict[str, list[tuple[int, int, int]]] = {}
    buffers: dict[int, deque[tuple[int, int, int]]] = {}
    for batch in pq.ParquetFile(RAW).iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        for uid, timestamp, item, organic in zip(
            *(
                batch.column(name).to_numpy(zero_copy_only=False)
                for name in ("uid", "timestamp", "item_id", "is_organic")
            ),
            strict=True,
        ):
            uid, timestamp, item, organic = int(uid), int(timestamp), int(item), int(organic)
            if timestamp >= TRAIN_END:
                continue
            history = buffers.setdefault(uid, deque(maxlen=MAX_HISTORY))
            key = (uid, timestamp, item)
            if key in wanted:
                result[wanted.pop(key)] = list(history)
            history.append((item, timestamp, 1 + (1 - organic)))
    if wanted:
        raise RuntimeError(f"failed to reconstruct {len(wanted)} selected dense histories")
    return result


def audit(manifest: dict) -> dict:
    rows = read_jsonl(TRAIN_MANIFEST)[:1000]
    first_seen, catalog, _ = build_catalog()
    catalog = set(map(int, catalog))
    histories = sample_histories(rows)
    failed = []
    for row in rows:
        history, negatives, ts = (
            histories[row["sample_id"]],
            row["negative_records"],
            int(row["target_timestamp"]),
        )
        checks = [
            max(event[1] for event in history) < ts,
            int(row["proposal_cutoff"]) <= ts,
            all(
                int(x["item_id"]) in catalog and first_seen[int(x["item_id"])] < ts
                for x in negatives
            ),
            int(row["positive_item_id"]) not in {int(x["item_id"]) for x in negatives},
            len(row["candidate_item_ids"]) == CANDIDATE_SIZE,
            row["query_timestamp"] == ts,
            row["query_action_id"] == QUERY_ACTION_ID,
            row["training_weight"] > 0,
        ]
        if not all(checks):
            failed.append(row["sample_id"])
    result = {
        "contract": "cc_theta0_v2_dense_training_contract_audit_v1",
        "status": "passed" if not failed else "failed",
        "audited_samples": len(rows),
        "violations": failed[:20],
        "training_manifest_hash": manifest["training_manifest_hash"],
        "validation_manifest_hash": sha256_file(DEV_MANIFEST),
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "proposal": {
            "version": "Q_main_rank_decay_v1_cc_causal",
            "pool_size": QMAIN_POOL_SIZE,
            "decay": QMAIN_DECAY,
        },
        "selection": {
            "max_causal_anchors_per_user": MAX_ANCHORS_PER_USER,
            "sampling": "last eligible target in each of 16 fixed foundation-time bins",
            "user_weighting": "equal total loss weight per user",
            "min_history": MIN_HISTORY,
        },
        "code_commit": code_commit(),
        "seed": SEED,
    }
    (RESULT_DIR / "cc_theta0_v2_dense_training_contract_audit_v1.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def stream_dense_batches(
    rows: list[dict], item_map: dict[int, int], device: torch.device, batch_size: int = 16
):
    """Rebuild causal prefixes once and train with a bounded fixed-seed shuffle."""
    from cc_theta0_qualification import collate_paths

    wanted = {
        (int(row["uid"]), int(row["target_timestamp"]), int(row["positive_item_id"])): row
        for row in rows
    }
    buffers: dict[int, deque[tuple[int, int, int]]] = {}
    pool: list[tuple[dict, list[tuple[int, int, int]]]] = []
    rng = np.random.default_rng(SEED)

    def collate(values):
        batch_rows = [dict(row, history_key=index) for index, (row, _) in enumerate(values)]
        history_map = {index: history for index, (_, history) in enumerate(values)}
        return batch_rows, collate_paths(batch_rows, history_map, item_map, "Full-512", device)

    for batch in pq.ParquetFile(RAW).iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        for uid, timestamp, item, organic in zip(
            *(
                batch.column(name).to_numpy(zero_copy_only=False)
                for name in ("uid", "timestamp", "item_id", "is_organic")
            ),
            strict=True,
        ):
            uid, timestamp, item, organic = int(uid), int(timestamp), int(item), int(organic)
            if timestamp >= TRAIN_END:
                continue
            history = buffers.setdefault(uid, deque(maxlen=MAX_HISTORY))
            key = (uid, timestamp, item)
            if key in wanted:
                pool.append((wanted.pop(key), list(history)))
                if len(pool) >= 4096:
                    selected = rng.choice(len(pool), size=batch_size, replace=False)
                    values = [pool[index] for index in selected]
                    for index in sorted(selected, reverse=True):
                        pool.pop(int(index))
                    yield collate(values)
            history.append((item, timestamp, 1 + (1 - organic)))
    if wanted:
        raise RuntimeError(f"failed to stream {len(wanted)} dense samples")
    rng.shuffle(pool)
    for start in range(0, len(pool), batch_size):
        yield collate(pool[start : start + batch_size])


def train(manifest: dict, device: torch.device, rank: int = 0, world_size: int = 1) -> dict | None:
    if CHECKPOINT.exists():
        raise FileExistsError(f"refusing to overwrite {CHECKPOINT}")
    rows = read_jsonl(TRAIN_MANIFEST)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base_model = make_model(device, len(item_map)).train()
    scorer = CCScorer(base_model)
    model = (
        DDP(scorer, device_ids=[device.index], output_device=device.index)
        if world_size > 1
        else scorer
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    losses = []
    batch_size = 16
    for batch_rows, tensors in stream_dense_batches(rows, item_map, device, batch_size):
        start, end = rank * 4, (rank + 1) * 4
        local_rows = batch_rows[start:end]
        items, behaviors, deltas, lengths, candidates, query_deltas = (
            value[start:end] for value in tensors
        )
        opt.zero_grad(set_to_none=True)
        with autocast_context(device):
            scores = model(items, behaviors, deltas, candidates, query_deltas, lengths)
            ce = F.cross_entropy(
                scores,
                torch.zeros(len(local_rows), device=device, dtype=torch.long),
                reduction="none",
            )
            weights = torch.tensor(
                [row["training_weight"] for row in local_rows], device=device, dtype=ce.dtype
            )
            local_numerator = (ce * weights).sum()
            total_weight = weights.sum().detach().clone()
            if world_size > 1:
                dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
            loss = local_numerator * world_size / total_weight
        loss.backward()
        opt.step()
        losses.append(float((local_numerator / weights.sum()).detach()))
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        return None
    base_model.eval()
    dev_rows = read_jsonl(DEV_MANIFEST)
    dev_hist = load_histories(dev_rows, history_cutoff=RELEASE_CUTOFF)
    dev_scores = score_rows(base_model, dev_rows, dev_hist, item_map, device, "Full-512")
    dev_summary, _ = summarize_metrics(dev_scores)
    payload = {
        "config": base_model.cfg.__dict__.copy(),
        "model": base_model.state_dict(),
        "seed": SEED,
        "contract": "cc_theta0_v2_dense",
        "training_manifest_hash": manifest["training_manifest_hash"],
        "quality_manifest_read": False,
        "checkpoint_rule": "final_after_one_preregistered_epoch",
    }
    torch.save(payload, CHECKPOINT)
    result = {
        "contract": "cc_theta0_v2_dense_train_v1",
        "status": "completed",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": manifest["training_manifest_hash"],
        "validation_manifest_hash": sha256_file(DEV_MANIFEST),
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "training": {
            "rows": len(rows),
            "optimizer_steps": len(losses),
            "epochs": 1,
            "batch_size": batch_size,
            "per_rank_batch_size": 4,
            "world_size": world_size,
            "distributed": world_size > 1,
            "shuffle": "fixed-seed bounded causal streaming buffer (4096)",
            "mean_loss": float(np.mean(losses)),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "quality_manifest_used_for_selection": False,
        },
        "validation": {"rows": len(dev_rows), "full_512": dev_summary},
        "checkpoint_rule": "final checkpoint after exactly one registered epoch; no gate manifest read for training/selection",
        "code_commit": code_commit(),
        "seed": SEED,
    }
    (RESULT_DIR / "cc_theta0_v2_dense_train_v1.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def load_model(device):
    from hstu_kvcache.models import HSTU, HSTUConfig

    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = HSTU(HSTUConfig(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    return model, saved


def gates(device: torch.device) -> dict:
    model, saved = load_model(device)
    rows = read_jsonl(QUALITY_MANIFEST)
    histories = load_histories(rows, history_cutoff=RELEASE_CUTOFF)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    paths = [
        "Empty",
        "Last-1",
        "Last-2",
        "Recent-8",
        "Recent-16",
        "Recent-32",
        "Recent-64",
        "Recent-128",
        "Full-512",
    ]
    arrays = {}
    summaries = {}
    for path in paths:
        score = score_rows(model, rows, histories, item_map, device, path)
        summaries[path], arrays[path] = summarize_metrics(score)
    full, empty, recent = (
        arrays["Full-512"]["target_log_prob"],
        arrays["Empty"]["target_log_prob"],
        arrays["Recent-32"]["target_log_prob"],
    )
    g1_delta = full - empty
    g2_delta = full - recent
    gate1 = {
        "contract": "cc_theta0_v2_gate1_v1",
        "status": "passed"
        if bootstrap_ci(g1_delta, SEED + 11)[0] > 0
        and summaries["Full-512"]["cross_entropy"] < np.log(CANDIDATE_SIZE)
        and summaries["Full-512"]["pairwise_auc"] > 0.5
        else "failed",
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": saved["training_manifest_hash"],
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "rows": len(rows),
        "paths": {"empty": summaries["Empty"], "full_512": summaries["Full-512"]},
        "full_minus_empty_target_log_prob": {
            "mean": float(g1_delta.mean()),
            "bootstrap_ci95": bootstrap_ci(g1_delta, SEED + 11),
        },
        "fresh_gate_manifest": True,
        "code_commit": code_commit(),
        "seed": SEED,
    }
    rlong = float(g2_delta.mean() / max(g1_delta.mean(), 1e-8))
    g2ci = bootstrap_ci(g2_delta, SEED + 12)
    gate2 = {
        "contract": "cc_theta0_v2_gate2_v1",
        "status": "passed" if g2ci[0] > 0 and rlong >= 0.05 else "failed",
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": saved["training_manifest_hash"],
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "rows": len(rows),
        "path_metrics": summaries,
        "full_minus_empty_target_log_prob": {
            "mean": float(g1_delta.mean()),
            "bootstrap_ci95": bootstrap_ci(g1_delta, SEED + 13),
        },
        "full_minus_recent32_target_log_prob": {
            "mean": float(g2_delta.mean()),
            "bootstrap_ci95": g2ci,
        },
        "r_long": rlong,
        "hard_conditions": {
            "full_minus_recent32_ci_lower_gt_zero": g2ci[0] > 0,
            "r_long_ge_0_05": rlong >= 0.05,
        },
        "fresh_gate_manifest": True,
        "cc_theta1_theta2_authorized": False,
        "code_commit": code_commit(),
        "seed": SEED,
    }
    (RESULT_DIR / "cc_theta0_v2_gate1_v1.json").write_text(json.dumps(gate1, indent=2) + "\n")
    (RESULT_DIR / "cc_theta0_v2_gate2_v1.json").write_text(json.dumps(gate2, indent=2) + "\n")
    return {"gate1": gate1, "gate2": gate2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("build", "audit", "train", "gates", "all"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-manifest", action="store_true")
    a = p.parse_args()
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device(a.device)
    if rank == 0:
        manifest = build_manifest(force=a.force_manifest)
        print(json.dumps(manifest, indent=2))
    if distributed:
        dist.barrier()
    if rank != 0:
        manifest = build_manifest()
    if a.command == "build":
        if distributed:
            dist.destroy_process_group()
        return
    checked = audit(manifest) if rank == 0 else {"status": "passed"}
    if rank == 0:
        print(json.dumps(checked, indent=2))
    if distributed:
        dist.barrier()
    if checked["status"] != "passed":
        raise RuntimeError("dense contract audit failed")
    if a.command == "audit":
        if distributed:
            dist.destroy_process_group()
        return
    if a.command in ("train", "all"):
        result = train(manifest, device, rank, world_size)
        if rank == 0:
            print(json.dumps(result, indent=2))
    if a.command in ("gates", "all"):
        if rank == 0:
            print(json.dumps(gates(device), indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P2.0--P2.4 CC-theta0 qualification on the real Yambda trace.

This is deliberately a single-edge qualification entry point.  It creates a
causal training manifest, runs a real-data overfit canary, trains exactly one
CC-theta0 checkpoint, and evaluates Gate 1/2.  It has no theta1/theta2,
controller, tomography, or theta3 path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from hstu_kvcache.data import event_time_deltas
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
MANIFEST_DIR = ROOT / "data/manifests"
RESULT_DIR = ROOT / "results/data_audit/yambda50m_v2"
CHECKPOINT = ROOT / "checkpoints/cc_theta0_v1.pt"
TRAIN_MANIFEST = MANIFEST_DIR / "cc_theta0_train_v1.jsonl"
QUALITY_MANIFEST = MANIFEST_DIR / "yambda50m_v2_theta0_qualification_candidates.jsonl"
DEV_MANIFEST = MANIFEST_DIR / "yambda50m_v2_theta0_dev_candidates.jsonl"

DAY = 86_400
TRAIN_END = 203 * DAY
RELEASE_CUTOFF = 210 * DAY
MAX_HISTORY = 512
CANDIDATE_SIZE = 100
QMAIN_POOL_SIZE = 1_000
QMAIN_DECAY = 0.5
SEED = 1
QUERY_TYPE_ID = 0
QUERY_ACTION_ID = 1
NUM_QUERY_ACTIONS = 2
MAX_RAW_ITEM_ID = 9_390_624
BOOTSTRAP_ROUNDS = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_object(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_timestamp(row: dict) -> int:
    value = row.get("target_timestamp", row.get("request_timestamp"))
    if value is None:
        raise KeyError("quality/training row has neither target_timestamp nor request_timestamp")
    return int(value)


def model_config() -> HSTUConfig:
    return HSTUConfig(
        num_items=1,
        num_behaviors=3,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=MAX_HISTORY,
        input_dropout=0.0,
        num_query_types=1,
        num_query_actions=NUM_QUERY_ACTIONS,
        query_type_id=QUERY_TYPE_ID,
        query_action_id=QUERY_ACTION_ID,
    )


def build_catalog() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return first-seen times, raw catalog IDs, and base popularity counts."""
    first_seen = np.full(MAX_RAW_ITEM_ID, np.iinfo(np.int64).max, dtype=np.int64)
    popularity = np.zeros(MAX_RAW_ITEM_ID, dtype=np.int64)
    parquet = pq.ParquetFile(RAW)
    columns = ["timestamp", "item_id", "played_ratio_pct"]
    for batch in parquet.iter_batches(batch_size=262_144, columns=columns):
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = timestamp < RELEASE_CUTOFF
        np.minimum.at(first_seen, item[mask], timestamp[mask])
        positive = mask & (played > 50)
        popularity += np.bincount(item[positive], minlength=MAX_RAW_ITEM_ID)
    catalog = np.flatnonzero(first_seen < np.iinfo(np.int64).max)
    return first_seen, catalog, popularity


def item_map_from_catalog(catalog: np.ndarray) -> dict[int, int]:
    return {int(raw): index + 1 for index, raw in enumerate(catalog)}


def _causal_pool(
    history: list[tuple[int, int, int]],
    target_item: int,
    target_timestamp: int,
    first_seen: np.ndarray,
    catalog_order: np.ndarray,
    catalog_order_times: np.ndarray,
) -> list[int]:
    visible_count = int(np.searchsorted(catalog_order_times, target_timestamp, side="left"))
    visible_catalog = catalog_order[:visible_count]
    counts = Counter(item for item, timestamp, _ in history if timestamp < target_timestamp)
    last_position: dict[int, int] = {}
    for position, (item, timestamp, _) in enumerate(history):
        if timestamp < target_timestamp:
            last_position[item] = position
    seen_order = sorted(
        counts,
        key=lambda item: (-counts[item], -last_position[item]),
    )
    pool: list[int] = []
    included: set[int] = set()
    for item in seen_order:
        item = int(item)
        if item != target_item and first_seen[item] < target_timestamp:
            pool.append(item)
            included.add(item)
    for raw_item in visible_catalog:
        item = int(raw_item)
        if item == target_item or item in included:
            continue
        pool.append(item)
        included.add(item)
        if len(pool) >= QMAIN_POOL_SIZE:
            break
    return pool[:QMAIN_POOL_SIZE]


def _sample_qmain_negatives(pool: list[int], uid: int, target_timestamp: int) -> list[dict]:
    if len(pool) < CANDIDATE_SIZE - 1:
        raise ValueError("causal visible proposal pool is smaller than the negative panel")
    ranks = np.arange(1, len(pool) + 1, dtype=np.float64)
    weights = ranks ** (-QMAIN_DECAY)
    weights /= weights.sum()
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED}:cc-qmain:{uid}:{target_timestamp}".encode()).digest()[:8],
        "little",
    )
    keys = np.random.default_rng(seed).exponential(size=len(pool)) / weights
    selected = np.argpartition(keys, CANDIDATE_SIZE - 2)[: CANDIDATE_SIZE - 1]
    selected = np.sort(selected)
    records = []
    for index in selected.tolist():
        rank = int(index + 1)
        weight = float(weights[index])
        records.append({
            "item_id": int(pool[index]),
            "proposal_rank": rank,
            "weight": weight,
            "log_q_main": float(math.log(weight)),
        })
    return records


def build_training_manifest(force: bool = False) -> dict:
    if TRAIN_MANIFEST.exists() and not force:
        rows = read_jsonl(TRAIN_MANIFEST)
        return {
            "status": "reused_existing_manifest",
            "training_manifest": str(TRAIN_MANIFEST),
            "training_manifest_hash": sha256_file(TRAIN_MANIFEST),
            "sample_count": len(rows),
        }

    first_seen, catalog, _ = build_catalog()
    catalog_order = catalog[np.argsort(first_seen[catalog], kind="stable")]
    catalog_order_times = first_seen[catalog_order]
    train_buffers: dict[int, deque[tuple[int, int, int]]] = {}
    parquet = pq.ParquetFile(RAW)
    columns = ["uid", "timestamp", "item_id", "is_organic"]
    for batch in parquet.iter_batches(batch_size=262_144, columns=columns):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = timestamp < TRAIN_END
        for user, ts, raw_item, org in zip(
            uid[mask], timestamp[mask], item[mask], organic[mask], strict=True
        ):
            user = int(user)
            if user not in train_buffers:
                train_buffers[user] = deque(maxlen=MAX_HISTORY + 1)
            train_buffers[user].append((int(raw_item), int(ts), int(1 + (1 - int(org)))))

    rows = []
    for user in sorted(train_buffers):
        events = list(train_buffers[user])
        if len(events) < 6:
            continue
        history = events[:-1]
        target_item, target_timestamp, _ = events[-1]
        pool = _causal_pool(
            history,
            target_item,
            target_timestamp,
            first_seen,
            catalog_order,
            catalog_order_times,
        )
        try:
            negatives = _sample_qmain_negatives(pool, user, target_timestamp)
        except ValueError:
            continue
        proposal = {
            "version": "Q_main_rank_decay_v1_cc_causal",
            "pool_size": len(pool),
            "decay": QMAIN_DECAY,
            "cutoff": target_timestamp,
            "negative_records": negatives,
        }
        rows.append({
            "sample_id": f"cc-theta0-train-{len(rows):07d}",
            "uid": user,
            "target_timestamp": target_timestamp,
            "positive_item_id": target_item,
            "history_max_timestamp": max(event[1] for event in history),
            "proposal_cutoff": target_timestamp,
            "query_timestamp": target_timestamp,
            "query_type_id": QUERY_TYPE_ID,
            "query_action_id": QUERY_ACTION_ID,
            "candidate_item_ids": [target_item] + [record["item_id"] for record in negatives],
            "negative_records": negatives,
            "proposal_version": proposal["version"],
            "proposal_hash": digest_object(proposal),
            "seen_item_policy": {
                "exclude_current_positive_from_negatives": True,
                "exclude_all_historically_seen_items": False,
            },
        })

    if not rows:
        raise RuntimeError("no causal CC training rows were produced")
    TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_MANIFEST.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "status": "created",
        "training_manifest": str(TRAIN_MANIFEST),
        "training_manifest_hash": sha256_file(TRAIN_MANIFEST),
        "sample_count": len(rows),
        "catalog_count": int(len(catalog)),
    }


def load_histories(
    rows: list[dict],
    *,
    history_cutoff: int | None = None,
) -> dict[int, list[tuple[int, int, int]]]:
    target_by_uid = {int(row["uid"]): row_timestamp(row) for row in rows}
    if len(target_by_uid) != len(rows):
        raise ValueError("qualification/training rows must contain one target per uid")
    histories: dict[int, deque[tuple[int, int, int]]] = {
        uid: deque(maxlen=MAX_HISTORY) for uid in target_by_uid
    }
    user_ids = np.asarray(sorted(target_by_uid), dtype=np.int64)
    parquet = pq.ParquetFile(RAW)
    columns = ["uid", "timestamp", "item_id", "is_organic"]
    for batch in parquet.iter_batches(batch_size=262_144, columns=columns):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        mask = np.isin(uid, user_ids)
        for user, ts, raw_item, org in zip(
            uid[mask], timestamp[mask], item[mask], organic[mask], strict=True
        ):
            user = int(user)
            cutoff = target_by_uid[user]
            if history_cutoff is not None:
                cutoff = min(cutoff, history_cutoff)
            if int(ts) < cutoff:
                histories[user].append((int(raw_item), int(ts), int(1 + (1 - int(org)))))
    return {uid: list(events) for uid, events in histories.items()}


def audit_training_manifest(manifest_info: dict, audit_count: int = 1_000) -> dict:
    rows = read_jsonl(TRAIN_MANIFEST)
    if len(rows) < audit_count:
        raise RuntimeError(f"only {len(rows)} training rows; need at least {audit_count}")
    rows = rows[:audit_count]
    first_seen, catalog, _ = build_catalog()
    catalog_set = set(int(item) for item in catalog)
    histories = load_histories(rows)
    violations: list[dict] = []
    proposal_hashes = set()
    for row in rows:
        uid = int(row["uid"])
        target_ts = int(row["target_timestamp"])
        history = histories[uid]
        negatives = row["negative_records"]
        checks = {
            "history_max_timestamp_lt_target": max(event[1] for event in history) < target_ts,
            "proposal_cutoff_le_target": int(row["proposal_cutoff"]) <= target_ts,
            "all_negative_ids_in_catalog": all(record["item_id"] in catalog_set for record in negatives),
            "all_negative_ids_visible_at_cutoff": all(
                first_seen[int(record["item_id"])] < int(row["proposal_cutoff"])
                for record in negatives
            ),
            "positive_not_in_negatives": int(row["positive_item_id"]) not in {
                int(record["item_id"]) for record in negatives
            },
            "all_candidates_same_query_timestamp": row["query_timestamp"] == target_ts,
            "query_action_separate_from_pad_mask": (
                int(row["query_action_id"]) == QUERY_ACTION_ID
                and QUERY_ACTION_ID != 0
                and NUM_QUERY_ACTIONS > QUERY_ACTION_ID
            ),
            "candidate_count": len(row["candidate_item_ids"]) == CANDIDATE_SIZE,
            "negative_record_count": len(negatives) == CANDIDATE_SIZE - 1,
            "proposal_hash_present": bool(row.get("proposal_hash")),
        }
        proposal_hashes.add(row["proposal_hash"])
        failed = [name for name, value in checks.items() if not value]
        if failed:
            violations.append({"sample_id": row["sample_id"], "failed_checks": failed})
    result = {
        "status": "passed" if not violations else "failed",
        "contract": "cc_training_contract_audit_v1",
        "audited_samples": len(rows),
        "training_sample_count": len(read_jsonl(TRAIN_MANIFEST)),
        "violations": violations[:20],
        "violation_count": len(violations),
        "unique_proposal_hashes": len(proposal_hashes),
        "training_manifest_hash": manifest_info["training_manifest_hash"],
        "raw_source_sha256": sha256_file(RAW),
        "code_commit": code_commit(),
        "seed": SEED,
        "q_main": {
            "version": "Q_main_rank_decay_v1_cc_causal",
            "pool_size": QMAIN_POOL_SIZE,
            "decay": QMAIN_DECAY,
        },
        "seen_item_policy": {
            "exclude_current_positive_from_negatives": True,
            "exclude_all_historically_seen_items": False,
        },
        "query_contract": {
            "query_type_id": QUERY_TYPE_ID,
            "query_action_id": QUERY_ACTION_ID,
            "num_query_actions": NUM_QUERY_ACTIONS,
            "behavior_pad_id": 0,
            "behavior_mask_id": None,
            "query_action_table_has_padding_idx": False,
            "query_is_transient": True,
        },
        "training_reads_quality_manifest": False,
        "manifest_usage": "training_only_causal_proposal",
    }
    output = RESULT_DIR / "cc_training_contract_audit_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _history_arrays(
    history: list[tuple[int, int, int]],
    item_map: dict[int, int],
    target_timestamp: int,
    path: str,
    uid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    if path == "Empty":
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.float32),
            0,
            0.0,
        )
    if path == "Last-1":
        selected = history[-1:]
    elif path == "Last-2":
        selected = history[-2:]
    elif path.startswith("Recent-"):
        selected = history[-int(path.split("-")[1]) :]
    elif path in ("Full-512", "Shuffled Full"):
        selected = history[-MAX_HISTORY:]
    else:
        raise ValueError(f"unknown history path: {path}")
    if path == "Shuffled Full" and selected:
        rng = np.random.default_rng(SEED + int(uid))
        permutation = rng.permutation(len(selected))
        # Keep the original timestamp slots and their deltas; only item/action
        # content moves between those slots.
        content = [selected[index] for index in permutation]
        shuffled = [
            (content[index][0], selected[index][1], content[index][2])
            for index in range(len(selected))
        ]
        selected = shuffled
        timestamps = np.asarray([event[1] for event in history[-MAX_HISTORY:]], dtype=np.int64)
        deltas = np.zeros(len(selected), dtype=np.float32)
        if len(deltas) > 1:
            deltas[1:] = np.diff(timestamps).clip(0, DAY * 7)
    else:
        deltas = event_time_deltas(selected).astype(np.float32)
    items = np.asarray([item_map[event[0]] for event in selected], dtype=np.int64)
    behaviors = np.asarray([event[2] for event in selected], dtype=np.int64)
    query_delta = float(np.clip(target_timestamp - selected[-1][1], 0, DAY * 7))
    return items, behaviors, deltas, len(selected), query_delta


def collate_paths(
    rows: list[dict],
    histories: dict[int, list[tuple[int, int, int]]],
    item_map: dict[int, int],
    path: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    arrays = [
        _history_arrays(
            histories[int(row.get("history_key", row["uid"]))],
            item_map,
            row_timestamp(row),
            path,
            int(row["uid"]),
        )
        for row in rows
    ]
    width = max(len(value[0]) for value in arrays)
    items = np.zeros((len(rows), width), dtype=np.int64)
    behaviors = np.zeros_like(items)
    deltas = np.zeros((len(rows), width), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    query_deltas = np.zeros(len(rows), dtype=np.float32)
    candidates = np.zeros((len(rows), CANDIDATE_SIZE), dtype=np.int64)
    for index, (row, values) in enumerate(zip(rows, arrays, strict=True)):
        item_ids, behavior_ids, time_deltas, length, query_delta = values
        items[index, :length] = item_ids[:length]
        behaviors[index, :length] = behavior_ids[:length]
        deltas[index, :length] = time_deltas[:length]
        lengths[index] = length
        query_deltas[index] = query_delta
        candidates[index] = [item_map[int(item)] for item in row["candidate_item_ids"]]
    return tuple(
        tensor.to(device)
        for tensor in (
            torch.from_numpy(items),
            torch.from_numpy(behaviors),
            torch.from_numpy(deltas),
            torch.from_numpy(lengths),
            torch.from_numpy(candidates),
            torch.from_numpy(query_deltas),
        )
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def make_model(device: torch.device, num_items: int) -> HSTU:
    cfg = model_config()
    cfg.num_items = num_items
    return HSTU(cfg).to(device)


def _metric_arrays(scores: np.ndarray) -> dict[str, np.ndarray]:
    target = scores[:, 0]
    negatives = scores[:, 1:]
    rank = 1 + (negatives >= target[:, None]).sum(axis=1)
    log_probs = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
    auc = ((target[:, None] > negatives).astype(np.float64) + 0.5 * (target[:, None] == negatives)).mean(axis=1)
    ndcg = np.where(rank <= 10, 1.0 / np.log2(rank + 1.0), 0.0)
    return {
        "cross_entropy": -log_probs[:, 0],
        "target_log_prob": log_probs[:, 0],
        "pairwise_auc": auc,
        "ndcg@10": ndcg,
        "hr@10": (rank <= 10).astype(np.float64),
        "mrr": 1.0 / rank,
    }


def summarize_metrics(scores: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = _metric_arrays(scores)
    return {key: float(values.mean()) for key, values in arrays.items()}, arrays


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def score_rows(
    model: HSTU,
    rows: list[dict],
    histories: dict[int, list[tuple[int, int, int]]],
    item_map: dict[int, int],
    device: torch.device,
    path: str,
    *,
    zero_candidate_items: bool = False,
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        items, behaviors, deltas, lengths, candidates, query_deltas = collate_paths(
            batch_rows, histories, item_map, path, device
        )
        item_vectors = None
        if zero_candidate_items:
            item_vectors = torch.zeros(
                (*candidates.shape, model.cfg.hidden_size),
                device=device,
                dtype=model.item_emb.weight.dtype,
            )
        with torch.inference_mode(), autocast_context(device):
            scores = model.score_cc_full(
                items,
                behaviors,
                deltas,
                candidates,
                query_deltas,
                lengths=lengths,
                candidate_item_vectors=item_vectors,
            )
        outputs.append(scores.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def load_checkpoint(device: torch.device) -> tuple[HSTU, dict]:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"missing CC theta0 checkpoint: {CHECKPOINT}")
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = HSTU(HSTUConfig(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    return model, saved


def run_overfit_canary(manifest_info: dict, device: torch.device) -> dict:
    rows = read_jsonl(TRAIN_MANIFEST)[:128]
    histories = load_histories(rows)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = make_model(device, len(item_map))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def evaluate() -> tuple[float, float, float, float]:
        model.eval()
        scores = score_rows(model, rows, histories, item_map, device, "Full-512", batch_size=8)
        target = scores[:, 0]
        margin = target - scores[:, 1:].max(axis=1)
        empty = score_rows(model, rows, histories, item_map, device, "Empty", batch_size=8)
        return (
            float(-_metric_arrays(scores)["target_log_prob"].mean()),
            float(margin.mean()),
            float(np.sqrt(np.mean((scores - empty) ** 2))),
            float(scores.std()),
        )

    before = evaluate()
    model.train()
    for _ in range(30):
        indices = rows[:32]
        batch = collate_paths(indices, histories, item_map, "Full-512", device)
        items, behaviors, deltas, lengths, candidates, query_deltas = batch
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            scores = model.score_cc_full(
                items, behaviors, deltas, candidates, query_deltas, lengths=lengths
            )
            loss = F.cross_entropy(scores, torch.zeros(len(indices), dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
    after = evaluate()
    model.eval()
    normal = score_rows(model, rows[:32], histories, item_map, device, "Full-512", batch_size=8)
    zeroed = score_rows(
        model,
        rows[:32],
        histories,
        item_map,
        device,
        "Full-512",
        zero_candidate_items=True,
        batch_size=8,
    )
    permutation = np.arange(CANDIDATE_SIZE)[::-1]
    permuted_rows = [dict(row, candidate_item_ids=[row["candidate_item_ids"][i] for i in permutation]) for row in rows[:32]]
    permuted = score_rows(model, permuted_rows, histories, item_map, device, "Full-512", batch_size=8)
    gradients = {
        "history_encoder_q_proj": float(model.blocks[0].attn.q_proj.weight.grad.abs().sum()) if model.blocks[0].attn.q_proj.weight.grad is not None else 0.0,
        "candidate_item_embedding": float(model.item_emb.weight.grad.abs().sum()) if model.item_emb.weight.grad is not None else 0.0,
        "query_type_embedding": float(model.query_encoder.type_embedding.weight.grad.abs().sum()) if model.query_encoder.type_embedding.weight.grad is not None else 0.0,
        "query_action_embedding": float(model.query_encoder.action_embedding.weight.grad.abs().sum()) if model.query_encoder.action_embedding.weight.grad is not None else 0.0,
    }
    normal_candidate_std = float(normal.std(axis=1).mean())
    zero_candidate_std = float(zeroed.std(axis=1).mean())
    canary_hard = {
        "candidate_ce_decreased": after[0] < before[0],
        "positive_margin_increased": after[1] > before[1],
        "candidate_item_ablation_reduces_variance": zero_candidate_std < normal_candidate_std * 0.05,
        "candidate_permutation_preserves_reordered_scores": float(np.max(np.abs(permuted - normal[:, permutation]))) < 1e-5,
        "all_required_gradients_present": all(value > 0 for value in gradients.values()),
    }
    result = {
        "status": "passed" if all(canary_hard.values()) else "failed",
        "contract": "cc_theta0_overfit_canary_v1",
        "samples": len(rows),
        "steps": 30,
        "before": {"candidate_ce": before[0], "positive_margin": before[1], "empty_full_rms": before[2], "score_std": before[3]},
        "after": {"candidate_ce": after[0], "positive_margin": after[1], "empty_full_rms": after[2], "score_std": after[3]},
        "candidate_content": {
            "normal_candidate_score_std_mean": normal_candidate_std,
            "zero_candidate_item_score_std_mean": zero_candidate_std,
            "zero_to_normal_std_ratio": float(zero_candidate_std / max(normal_candidate_std, 1e-12)),
            "permutation_max_reordered_error": float(np.max(np.abs(permuted - normal[:, permutation]))),
        },
        "gradients": gradients,
        "hard_conditions": canary_hard,
        "query_action_id": QUERY_ACTION_ID,
        "query_action_padding_idx": None,
        "training_manifest_hash": manifest_info["training_manifest_hash"],
        "code_commit": code_commit(),
        "seed": SEED,
    }
    output = RESULT_DIR / "cc_theta0_overfit_canary_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def train_theta0(manifest_info: dict, device: torch.device, force: bool = False) -> dict:
    if CHECKPOINT.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {CHECKPOINT}; pass --force")
    rows = read_jsonl(TRAIN_MANIFEST)
    histories = load_histories(rows)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = make_model(device, len(item_map)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    order = np.arange(len(rows))
    np.random.default_rng(SEED).shuffle(order)
    batch_size = 16
    losses = []
    for start in range(0, len(order), batch_size):
        batch_rows = [rows[index] for index in order[start : start + batch_size]]
        items, behaviors, deltas, lengths, candidates, query_deltas = collate_paths(
            batch_rows, histories, item_map, "Full-512", device
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            scores = model.score_cc_full(
                items,
                behaviors,
                deltas,
                candidates,
                query_deltas,
                lengths=lengths,
            )
            loss = F.cross_entropy(scores, torch.zeros(len(batch_rows), dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    dev_rows = read_jsonl(DEV_MANIFEST)
    dev_histories = load_histories(dev_rows, history_cutoff=RELEASE_CUTOFF)
    dev_scores = score_rows(model, dev_rows, dev_histories, item_map, device, "Full-512")
    dev_summary, _ = summarize_metrics(dev_scores)
    config = model.cfg.__dict__.copy()
    checkpoint_payload = {
        "config": config,
        "model": model.state_dict(),
        "seed": SEED,
        "contract": "cc_theta0_v1",
        "training_manifest_hash": manifest_info["training_manifest_hash"],
        "quality_manifest_read": False,
        "checkpoint_rule": "final_after_one_epoch_fixed_budget",
        "code_commit": code_commit(),
    }
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, CHECKPOINT)
    result = {
        "status": "completed",
        "contract": "cc_theta0_train_v1",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": manifest_info["training_manifest_hash"],
        "validation_manifest_hash": sha256_file(DEV_MANIFEST),
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "raw_source_sha256": sha256_file(RAW),
        "code_commit": code_commit(),
        "random_seed": SEED,
        "device": str(device),
        "config": config,
        "training": {
            "sample_count": len(rows),
            "epochs": 1,
            "optimizer_steps": len(losses),
            "batch_size": batch_size,
            "mean_training_loss": float(np.mean(losses)),
            "first_training_loss": losses[0],
            "last_training_loss": losses[-1],
            "objective": "candidate_set_CE",
            "proposal": "causal_Q_main_rank_decay",
            "quality_manifest_used_for_selection": False,
        },
        "validation": {"manifest": str(DEV_MANIFEST), "rows": len(dev_rows), "full_512": dev_summary},
        "checkpoint_rule": "one registered epoch; final checkpoint only; no qualification-manifest early stopping",
    }
    output = RESULT_DIR / "cc_theta0_train_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def baseline_scores(rows: list[dict], popularity: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray([row["candidate_item_ids"] for row in rows], dtype=np.int64)
    proposal = np.log1p(popularity[candidates]).astype(np.float64)
    random = np.random.default_rng(seed).standard_normal(candidates.shape)
    return proposal, random


def run_gate1(device: torch.device) -> dict:
    model, checkpoint = load_checkpoint(device)
    rows = read_jsonl(QUALITY_MANIFEST)
    histories = load_histories(rows, history_cutoff=RELEASE_CUTOFF)
    first_seen, catalog, popularity = build_catalog()
    item_map = item_map_from_catalog(catalog)
    full = score_rows(model, rows, histories, item_map, device, "Full-512")
    empty = score_rows(model, rows, histories, item_map, device, "Empty")
    zeroed = score_rows(model, rows, histories, item_map, device, "Full-512", zero_candidate_items=True)
    proposal, random = baseline_scores(rows, popularity, SEED + 100)
    full_summary, full_arrays = summarize_metrics(full)
    empty_summary, _ = summarize_metrics(empty)
    proposal_summary, _ = summarize_metrics(proposal)
    random_summary, _ = summarize_metrics(random)
    delta = full_arrays["target_log_prob"] - _metric_arrays(empty)["target_log_prob"]
    permutation = np.arange(CANDIDATE_SIZE)[::-1]
    permuted_rows = [dict(row, candidate_item_ids=[row["candidate_item_ids"][i] for i in permutation]) for row in rows]
    permuted = score_rows(model, permuted_rows, histories, item_map, device, "Full-512")
    normal_std = full.std(axis=1)
    zero_std = zeroed.std(axis=1)
    candidate_content = {
        "normal_candidate_score_std_mean": float(normal_std.mean()),
        "zero_candidate_item_score_std_mean": float(zero_std.mean()),
        "zero_to_normal_std_ratio": float(zero_std.mean() / max(normal_std.mean(), 1e-12)),
        "zeroing_reduces_score_variance": bool(zero_std.mean() < normal_std.mean()),
        "candidate_identity_permutation_max_reordered_error": float(np.max(np.abs(permuted - full[:, permutation]))),
        "candidate_identity_changes_score": bool(float(normal_std.mean()) > 1e-8),
    }
    hard = {
        "user_bootstrap_delta_ci95_lower_gt_zero": bootstrap_ci(delta, SEED + 101)[0] > 0,
        "full_ce_below_random_log_candidate_count": full_summary["cross_entropy"] < math.log(CANDIDATE_SIZE),
        "full_auc_gt_0_5": full_summary["pairwise_auc"] > 0.5,
        "full_ndcg_above_random": full_summary["ndcg@10"] > random_summary["ndcg@10"],
        "full_hr_above_random": full_summary["hr@10"] > random_summary["hr@10"],
        "full_mrr_above_random": full_summary["mrr"] > random_summary["mrr"],
        "candidate_content_influences_score": candidate_content["zeroing_reduces_score_variance"],
    }
    result = {
        "status": "passed" if all(hard.values()) else "failed",
        "contract": "cc_theta0_gate1_v1",
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": checkpoint["training_manifest_hash"],
        "validation_manifest_hash": sha256_file(DEV_MANIFEST),
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "q_main_proposal_hash": digest_object({"version": "Q_main_rank_decay_v1_cc_causal", "decay": QMAIN_DECAY}),
        "code_commit": code_commit(),
        "random_seed": SEED,
        "rows": len(rows),
        "random_level": math.log(CANDIDATE_SIZE),
        "baselines": {"random": random_summary, "q_main_proposal_order": proposal_summary, "empty": empty_summary},
        "full_512": full_summary,
        "primary_full_minus_empty_target_log_prob": {
            "mean": float(delta.mean()),
            "bootstrap_ci95": bootstrap_ci(delta, SEED + 101),
        },
        "candidate_content_audit": candidate_content,
        "hard_conditions": hard,
        "qualification_manifest_used_for_training_or_selection": False,
        "source": {"raw_sha256": sha256_file(RAW), "first_seen_catalog_cutoff": RELEASE_CUTOFF},
    }
    output = RESULT_DIR / "cc_theta0_gate1_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_gate2(device: torch.device) -> dict:
    model, checkpoint = load_checkpoint(device)
    rows = read_jsonl(QUALITY_MANIFEST)
    histories = load_histories(rows, history_cutoff=RELEASE_CUTOFF)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    paths = ["Empty", "Last-1", "Last-2", "Recent-8", "Recent-32", "Recent-128", "Full-512", "Shuffled Full"]
    summaries = {}
    arrays = {}
    for path in paths:
        scores = score_rows(model, rows, histories, item_map, device, path)
        summaries[path], arrays[path] = summarize_metrics(scores)
    full_logp = arrays["Full-512"]["target_log_prob"]
    empty_logp = arrays["Empty"]["target_log_prob"]
    recent32_logp = arrays["Recent-32"]["target_log_prob"]
    last2_logp = arrays["Last-2"]["target_log_prob"]
    long_delta = full_logp - recent32_logp
    empty_delta = full_logp - empty_logp
    long_ratio = float(long_delta.mean() / max(float(empty_delta.mean()), 1e-8))
    comparisons = {
        "full_minus_empty": {"mean": float(empty_delta.mean()), "bootstrap_ci95": bootstrap_ci(empty_delta, SEED + 201)},
        "full_minus_last1": {"mean": float((full_logp - arrays["Last-1"]["target_log_prob"]).mean()), "bootstrap_ci95": bootstrap_ci(full_logp - arrays["Last-1"]["target_log_prob"], SEED + 202)},
        "full_minus_last2": {"mean": float((full_logp - last2_logp).mean()), "bootstrap_ci95": bootstrap_ci(full_logp - last2_logp, SEED + 203)},
        "full_minus_recent8": {"mean": float((full_logp - arrays["Recent-8"]["target_log_prob"]).mean()), "bootstrap_ci95": bootstrap_ci(full_logp - arrays["Recent-8"]["target_log_prob"], SEED + 204)},
        "full_minus_recent32": {"mean": float(long_delta.mean()), "bootstrap_ci95": bootstrap_ci(long_delta, SEED + 205)},
        "full_minus_recent128": {"mean": float((full_logp - arrays["Recent-128"]["target_log_prob"]).mean()), "bootstrap_ci95": bootstrap_ci(full_logp - arrays["Recent-128"]["target_log_prob"], SEED + 206)},
    }
    curve_path = RESULT_DIR / "cc_theta0_history_length_curve_v1.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "target_log_prob", "cross_entropy", "pairwise_auc", "ndcg_at_10", "hr_at_10", "mrr"])
        for path in paths:
            value = summaries[path]
            writer.writerow([path, value["target_log_prob"], value["cross_entropy"], value["pairwise_auc"], value["ndcg@10"], value["hr@10"], value["mrr"]])
    bootstrap_path = RESULT_DIR / "cc_theta0_history_utility_bootstrap_v1.csv"
    with bootstrap_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "mean_target_log_prob_delta", "ci95_lower", "ci95_upper"])
        for name, value in comparisons.items():
            writer.writerow([name, value["mean"], value["bootstrap_ci95"][0], value["bootstrap_ci95"][1]])
    hard = {
        "full_minus_recent32_bootstrap_ci95_lower_gt_zero": comparisons["full_minus_recent32"]["bootstrap_ci95"][0] > 0,
        "full_minus_last2_bootstrap_ci95_lower_gt_zero": comparisons["full_minus_last2"]["bootstrap_ci95"][0] > 0,
        "r_long_not_near_noise": long_ratio >= 0.05,
    }
    result = {
        "status": "passed" if all(hard.values()) else "failed",
        "contract": "cc_theta0_gate2_v1",
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": checkpoint["training_manifest_hash"],
        "validation_manifest_hash": sha256_file(DEV_MANIFEST),
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "q_main_proposal_hash": digest_object({"version": "Q_main_rank_decay_v1_cc_causal", "decay": QMAIN_DECAY}),
        "code_commit": code_commit(),
        "random_seed": SEED,
        "rows": len(rows),
        "paths": paths,
        "path_metrics": summaries,
        "comparisons": comparisons,
        "delta_long_full_minus_recent32": float(long_delta.mean()),
        "r_long": long_ratio,
        "r_long_definition": "mean(Full-512 minus Recent-32 target log-prob) / mean(Full-512 minus Empty target log-prob)",
        "shuffled_full_role": "mechanism_companion_only; timestamp slots fixed and item/action content permuted",
        "hard_conditions": hard,
        "history_length_curve_csv": str(curve_path),
        "history_utility_bootstrap_csv": str(bootstrap_path),
        "theta1_theta2_authorized": False,
    }
    output = RESULT_DIR / "cc_theta0_gate2_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "canary", "train", "gate1", "gate2", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_info = build_training_manifest(force=args.force)
    audit = audit_training_manifest(manifest_info)
    if audit["status"] != "passed":
        raise RuntimeError("cc_training_contract_audit_v1 failed; refusing to train")
    print(json.dumps(audit, indent=2))
    if args.command == "audit":
        return
    if args.command == "canary":
        canary = run_overfit_canary(manifest_info, device)
        print(json.dumps(canary, indent=2))
        if canary["status"] != "passed":
            raise RuntimeError("cc_theta0_overfit_canary_v1 failed; refusing to train")
        return
    if args.command == "train":
        canary = run_overfit_canary(manifest_info, device)
        print(json.dumps(canary, indent=2))
        if canary["status"] != "passed":
            raise RuntimeError("cc_theta0_overfit_canary_v1 failed; refusing to train")
        train = train_theta0(manifest_info, device, force=args.force)
        print(json.dumps(train, indent=2))
        return
    if args.command == "gate1":
        gate1 = run_gate1(device)
        print(json.dumps(gate1, indent=2))
        return
    if args.command == "gate2":
        gate2 = run_gate2(device)
        print(json.dumps(gate2, indent=2))
        return
    canary = run_overfit_canary(manifest_info, device)
    print(json.dumps(canary, indent=2))
    if canary["status"] != "passed":
        raise RuntimeError("cc_theta0_overfit_canary_v1 failed; refusing to train")
    train = train_theta0(manifest_info, device, force=args.force)
    print(json.dumps(train, indent=2))
    gate1 = run_gate1(device)
    print(json.dumps(gate1, indent=2))
    gate2 = run_gate2(device)
    print(json.dumps(gate2, indent=2))


if __name__ == "__main__":
    main()

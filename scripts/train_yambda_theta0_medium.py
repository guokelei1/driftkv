#!/usr/bin/env python3
"""Train/evaluate the first medium theta0 under the frozen Yambda v2 contract."""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.data import event_time_deltas


DAY = 86_400
TRAIN_END = 203 * DAY
FOUNDATION_END = 210 * DAY
MAX_HISTORY = 512
CANDIDATE_SIZE = 100
SEED = 1


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_candidates(target: int, popular_items: np.ndarray) -> list[int]:
    candidates = [target]
    for item in popular_items:
        item = int(item)
        if item != target:
            candidates.append(item)
        if len(candidates) == CANDIDATE_SIZE:
            return candidates
    raise RuntimeError("popularity catalog is too small")


def build_foundation_data(path: Path, eval_uids: set[int]):
    popularity = np.zeros(9_390_624, dtype=np.int64)
    catalog_counts = np.zeros(9_390_624, dtype=np.int64)
    train_histories: dict[int, deque] = {}
    eval_histories: dict[int, deque] = {uid: deque(maxlen=MAX_HISTORY) for uid in eval_uids}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"]
    ):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        played = batch.column("played_ratio_pct").to_numpy(zero_copy_only=False).astype(np.int64)
        base_mask = ts < FOUNDATION_END
        popular_mask = (ts < FOUNDATION_END) & (played > 50)
        catalog_counts += np.bincount(items[base_mask], minlength=len(catalog_counts))
        popularity += np.bincount(items[popular_mask], minlength=len(popularity))
        for u, t, item, org in zip(uid[base_mask], ts[base_mask], items[base_mask], organic[base_mask]):
            u, t, item = int(u), int(t), int(item)
            event = (item, t, int(1 + (1 - int(org))))
            if t < TRAIN_END:
                if u not in train_histories:
                    train_histories[u] = deque(maxlen=MAX_HISTORY + 1)
                train_histories[u].append(event)
            if u in eval_histories:
                eval_histories[u].append(event)
    popular_items = np.flatnonzero(popularity)
    popular_items = popular_items[np.argsort(-popularity[popular_items], kind="stable")]
    item_ids = np.flatnonzero(catalog_counts)
    item_map = {int(raw): index + 1 for index, raw in enumerate(item_ids)}
    return train_histories, eval_histories, item_map, popular_items


def history_to_arrays(
    history: list[tuple[int, int, int]],
    item_map: dict[int, int],
    *,
    previous_timestamp: int | None = None,
):
    history = history[-MAX_HISTORY:]
    deltas = event_time_deltas(history, previous_timestamp=previous_timestamp)
    return (
        # Item id 0 is reserved for the explicit, unscored readout token used
        # only by target-independent compatibility profiling.
        np.asarray([item_map.get(x[0], 0) for x in history], dtype=np.int64),
        np.asarray([x[2] for x in history], dtype=np.int64),
        deltas,
    )


def collate_histories(histories, item_map):
    L = MAX_HISTORY
    items = np.zeros((len(histories), L), dtype=np.int64)
    behaviors = np.zeros_like(items)
    deltas = np.zeros((len(histories), L), dtype=np.float32)
    lengths = np.zeros(len(histories), dtype=np.int64)
    for i, history in enumerate(histories):
        a, b, d = history_to_arrays(list(history), item_map)
        n = len(a)
        # HSTU's ``lengths`` mask and ``last_hidden`` index valid tokens in
        # the leading prefix. Keep this convention aligned with
        # ``data.kuairand.collate_batch``.
        items[i, :n] = a
        behaviors[i, :n] = b
        deltas[i, :n] = d
        lengths[i] = n
    return (
        torch.from_numpy(items),
        torch.from_numpy(behaviors),
        torch.from_numpy(deltas),
        torch.from_numpy(lengths),
    )


def metrics(scores: torch.Tensor) -> dict[str, float]:
    target = scores[:, 0]
    negative = scores[:, 1:]
    # The injected target is stored at candidate index 0. Count equal scores
    # as worse than the target so a constant no-history score cannot inherit
    # rank 1 from the manifest ordering.
    rank = 1 + (negative >= target.unsqueeze(1)).sum(dim=1)
    target_log_prob = F.log_softmax(scores, dim=1)[:, 0]
    auc = ((target.unsqueeze(1) > negative).float() + 0.5 * (target.unsqueeze(1) == negative).float()).mean(dim=1)
    ndcg = torch.where(rank <= 10, 1.0 / torch.log2(rank.float() + 1.0), torch.zeros_like(rank, dtype=torch.float32))
    return {
        "cross_entropy": float((-target_log_prob).mean()),
        "target_log_prob": float(target_log_prob.mean()),
        "pairwise_auc": float(auc.mean()),
        "ndcg@10": float(ndcg.mean()),
        "hr@10": float((rank <= 10).float().mean()),
        "mrr": float((1.0 / rank.float()).mean()),
    }


def evaluate(model, histories, candidates, item_map, popular_items, device, batch_size=32):
    model.eval()
    all_full, all_no, all_pop, all_random = [], [], [], []
    popularity_rank = {
        int(item_map[int(raw)]): 1.0 - (i / max(1, len(popular_items) - 1))
        for i, raw in enumerate(popular_items)
    }
    random_generator = np.random.default_rng(37)
    for start in range(0, len(histories), batch_size):
        h = histories[start : start + batch_size]
        c = candidates[start : start + batch_size]
        items, behaviors, deltas, lengths = collate_histories(h, item_map)
        items, behaviors, deltas, lengths = [x.to(device) for x in (items, behaviors, deltas, lengths)]
        candidate_tensor = torch.from_numpy(np.asarray(c, dtype=np.int64)).to(device)
        popularity_scores = torch.from_numpy(
            np.asarray([[popularity_rank.get(int(item), 0) for item in row] for row in c], dtype=np.float32)
        )
        random_scores = torch.from_numpy(random_generator.standard_normal((len(c), len(c[0]))).astype(np.float32))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _ = model(items, behaviors, deltas, lengths=lengths)
            full = model.score_candidates(hidden, candidate_tensor, lengths)
            no_items = torch.zeros((len(h), 1), dtype=torch.long, device=device)
            no_behaviors = torch.zeros_like(no_items)
            no_deltas = torch.zeros((len(h), 1), dtype=torch.float32, device=device)
            no_lengths = torch.zeros(len(h), dtype=torch.long, device=device)
            no_hidden, _ = model(no_items, no_behaviors, no_deltas, lengths=no_lengths)
            no = model.score_candidates(no_hidden, candidate_tensor, no_lengths)
        all_full.append(full.float().cpu())
        all_no.append(no.float().cpu())
        all_pop.append(popularity_scores)
        all_random.append(random_scores)
    full_scores = torch.cat(all_full)
    no_scores = torch.cat(all_no)
    pop_scores = torch.cat(all_pop)
    random_scores = torch.cat(all_random)
    full_metrics = metrics(full_scores)
    no_metrics = metrics(no_scores)
    full_logprob = F.log_softmax(full_scores, dim=1)[:, 0]
    no_logprob = F.log_softmax(no_scores, dim=1)[:, 0]
    delta = full_logprob - no_logprob
    rng = np.random.default_rng(37)
    boot = np.asarray([delta.numpy()[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)])
    return {
        "full": full_metrics,
        "no_history": no_metrics,
        "popularity": metrics(pop_scores),
        "random": metrics(random_scores),
        "history_log_prob_delta_mean": float(delta.mean()),
        "history_log_prob_delta_median": float(delta.median()),
        "history_log_prob_delta_positive_fraction": float((delta > 0).float().mean()),
        "history_log_prob_delta_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_path = Path("data/raw/yambda/flat/50m/listens.parquet")
    dev_rows = read_manifest(Path("data/manifests/yambda50m_v2_theta0_dev_candidates.jsonl"))
    qual_rows = read_manifest(Path("data/manifests/yambda50m_v2_theta0_qualification_candidates.jsonl"))
    eval_uids = {int(row["uid"]) for row in dev_rows + qual_rows}
    train_histories, eval_histories, item_map, popular_items = build_foundation_data(raw_path, eval_uids)

    train_users = sorted(train_histories)
    train_histories_list = []
    train_targets = []
    for uid in train_users:
        history = list(train_histories[uid])
        if len(history) < 6:
            continue
        target = history.pop()[0]
        if target not in item_map:
            continue
        train_histories_list.append(history)
        train_targets.append(target)
    train_candidates = [
        [item_map[item] for item in make_candidates(target, popular_items)]
        for target in train_targets
    ]

    cfg = HSTUConfig(
        num_items=len(item_map), num_behaviors=3, hidden_size=128, num_layers=4, num_heads=4,
        max_seq_len=MAX_HISTORY, input_dropout=0.0,
    )
    model = HSTU(cfg).to(device)
    checkpoint = Path("checkpoints/yambda50m_v2_theta0_medium_batchfix_v3.pt")
    steps = 0
    if args.eval_only:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        batch_size = 16
        model.train()
        order = np.arange(len(train_histories_list))
        for epoch in range(1):
            np.random.default_rng(SEED).shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                h = [train_histories_list[i] for i in indices]
                items, behaviors, deltas, lengths = collate_histories(h, item_map)
                candidates = torch.from_numpy(np.asarray([train_candidates[i] for i in indices], dtype=np.int64)).to(device)
                items, behaviors, deltas, lengths = [x.to(device) for x in (items, behaviors, deltas, lengths)]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden, _ = model(items, behaviors, deltas, lengths=lengths)
                    scores = model.score_candidates(hidden, candidates, lengths)
                    loss = F.cross_entropy(scores, torch.zeros(len(indices), dtype=torch.long, device=device))
                loss.backward()
                optimizer.step()
                steps += 1
    dev_histories = [list(eval_histories[int(row["uid"])]) for row in dev_rows]
    qual_histories = [list(eval_histories[int(row["uid"])]) for row in qual_rows]
    dev_candidates = [[item_map[int(x)] for x in row["candidate_item_ids"]] for row in dev_rows]
    qual_candidates = [[item_map[int(x)] for x in row["candidate_item_ids"]] for row in qual_rows]
    dev_result = evaluate(model, dev_histories, dev_candidates, item_map, popular_items, device)
    qual_result = evaluate(model, qual_histories, qual_candidates, item_map, popular_items, device)
    result = {
        "status": "medium_theta0_batchfix_v3_quality_screen_not_three_version_result",
        "seed": SEED,
        "device": str(device),
        "config": asdict(cfg),
        "foundation": {"train_end_days": 203, "release_cutoff_days": 210, "train_users": len(train_histories_list), "optimizer_steps": steps},
        "dev_users": len(dev_rows),
        "qualification_users": len(qual_rows),
        "dev": dev_result,
        "qualification": qual_result,
    }
    output = Path("results/data_audit/yambda50m_v2/theta0_medium_batchfix_v3_screen.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        torch.save({"config": asdict(cfg), "model": model.state_dict(), "seed": SEED, "contract": "yambda50m_v2"}, checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

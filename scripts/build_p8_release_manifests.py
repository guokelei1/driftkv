#!/usr/bin/env python3
"""Build prospective, label-disciplined P8 release/update manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import build_p7_compact_manifests as p7
import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.data.stateful_workloads import build_return_to_familiar_request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda"
OUTPUT = ROOT / "data/manifests/p8_release_v1"
DAY = 86_400
CONTEXT = 512
RETURN_GAP = 3 * DAY
WINDOWS = {
    "update1_train": (217 * DAY, 229 * DAY),
    "update1_admission_dev": (229 * DAY, 231 * DAY),
    "edge1_evaluation": (231 * DAY, 238 * DAY),
    "update2_train": (231 * DAY, 243 * DAY),
    "update2_admission_dev": (243 * DAY, 245 * DAY),
    "edge2_evaluation": (245 * DAY, 252 * DAY),
}
TRAIN_K = {"update1_train": 2176, "update2_train": 2304}
EVALUATION_SPLITS = {"edge1_evaluation", "edge2_evaluation"}
CONTRACT = ROOT / "configs/contracts/f_release_chain_contract_v1.yaml"


def window_names(timestamp: int) -> tuple[str, ...]:
    return tuple(
        name for name, (start, end) in WINDOWS.items() if start <= timestamp < end
    )


def catalog_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapping = pq.read_table(RAW / "artist_item_mapping.parquet", columns=["artist_id", "item_id"])
    items = mapping["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    artists = mapping["artist_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    size = int(items.max()) + 1
    artist = np.full(size, -1, dtype=np.int64)
    artist[items] = artists
    first_seen = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    popularity = np.zeros(size, dtype=np.int64)
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    for batch in parquet.iter_batches(batch_size=524_288, columns=["timestamp", "item_id"]):
        timestamp = batch["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        visible = timestamp < WINDOWS["edge2_evaluation"][1]
        np.minimum.at(first_seen, item[visible], timestamp[visible])
        base = item[timestamp < 180 * DAY]
        popularity += np.bincount(base, minlength=size)
    catalog = np.flatnonzero(first_seen < np.iinfo(np.int64).max)
    order = catalog[np.argsort(first_seen[catalog], kind="stable")]
    return artist, popularity, first_seen, order


def load_feedback() -> dict[int, list[tuple[int, int, int, int]]]:
    output: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for label, name in ((1, "likes"), (0, "dislikes")):
        table = pq.read_table(
            RAW / f"flat/50m/{name}.parquet",
            columns=["uid", "timestamp", "item_id", "is_organic"],
        )
        for uid, timestamp, item, organic in zip(
            *(table[column].to_numpy(zero_copy_only=False) for column in table.column_names),
            strict=True,
        ):
            if window_names(int(timestamp)):
                output[int(uid)].append((int(timestamp), int(item), label, int(organic)))
    for values in output.values():
        values.sort()
    return output


def choose(current: p7.Spec | None, candidate: p7.Spec, namespace: str) -> p7.Spec:
    return p7.choose(current, candidate, namespace)


def select_specs(feedback: dict[int, list[tuple[int, int, int, int]]]) -> tuple[list[p7.Spec], dict]:
    n_by_user: dict[tuple[str, int], p7.Spec] = {}
    r_by_split: dict[str, list[p7.Spec]] = defaultdict(list)
    f_by_user: dict[tuple[str, int], p7.Spec] = {}
    f_eval: list[p7.Spec] = []
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    ts_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    user_row_start = 0
    global_end = 0

    def consume() -> None:
        if current_uid is None:
            return
        timestamps = np.concatenate(ts_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        last_position: dict[int, int] = {}
        window: deque[int] = deque()
        counts: Counter[int] = Counter()
        boundaries = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1, len(timestamps)]
        for group in range(len(boundaries) - 1):
            start, end = int(boundaries[group]), int(boundaries[group + 1])
            timestamp = int(timestamps[start])
            splits = window_names(timestamp)
            for split in splits:
                if start:
                    target = int(items[start])
                    spec = p7.Spec(
                        "N", split, current_uid, timestamp, start, user_row_start,
                        user_row_start + start, target_item=target,
                        target_stratum=p7.stratum(last_position.get(target), start),
                    )
                    key = (split, current_uid)
                    n_by_user[key] = choose(n_by_user.get(key), spec, f"p8-{split}-N-v1")
            gap = timestamp - int(timestamps[start - 1]) if start else 0
            if gap >= RETURN_GAP and len(counts) >= 2:
                for split in splits:
                    target = int(items[start])
                    rankable = target in counts
                    spec = p7.Spec(
                        "R", split, current_uid, timestamp, start, user_row_start,
                        user_row_start + start, target_item=target, rankable=rankable,
                        target_stratum=p7.stratum(last_position.get(target), start),
                    )
                    if rankable or split in EVALUATION_SPLITS:
                        r_by_split[split].append(spec)
            for position in range(start, end):
                item = int(items[position])
                window.append(item)
                counts[item] += 1
                last_position[item] = position
                if len(window) > CONTEXT:
                    removed = window.popleft()
                    counts[removed] -= 1
                    if not counts[removed]:
                        del counts[removed]
        for feedback_index, (timestamp, item, label, organic) in enumerate(feedback.get(current_uid, ())):
            splits = window_names(timestamp)
            if not splits:
                continue
            prefix = int(np.searchsorted(timestamps, timestamp, side="left"))
            if not prefix:
                continue
            for split in splits:
                spec = p7.Spec(
                    "F", split, current_uid, timestamp, prefix, user_row_start,
                    -(current_uid * 1_000_000 + feedback_index + 1), target_item=item,
                    label=label, is_organic=organic,
                    target_stratum=p7.stratum(last_before(items, item, prefix), prefix),
                )
                if split in EVALUATION_SPLITS or "admission_dev" in split:
                    f_eval.append(spec)
                else:
                    key = (split, current_uid)
                    f_by_user[key] = choose(f_by_user.get(key), spec, f"p8-{split}-F-v1")

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id"]):
        uid = batch["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(uid)], strict=True):
            user = int(uid[start])
            if current_uid is not None and user != current_uid:
                consume()
                ts_parts.clear()
                item_parts.clear()
                user_row_start = global_end
            current_uid = user
            ts_parts.append(timestamp[start:end])
            item_parts.append(item[start:end])
            global_end += end - start
    consume()

    selected: list[p7.Spec] = []
    for split in WINDOWS:
        n = [value for (name, _), value in n_by_user.items() if name == split]
        f = [value for (name, _), value in f_by_user.items() if name == split]
        r = r_by_split[split]
        if split in TRAIN_K:
            k = TRAIN_K[split]
            rankable = [value for value in r if value.rankable]
            if len(rankable) != k:
                raise RuntimeError(f"{split} R coverage changed: {len(rankable)} != {k}")
            n.sort(key=lambda value: p7.hash_score(f"p8-{split}-N-users-v1", value.uid))
            f.sort(key=lambda value: p7.hash_score(f"p8-{split}-F-users-v1", value.uid))
            if len(n) < k or len(f) < k:
                raise RuntimeError(f"{split} cannot meet equal task query budget")
            selected += n[:k] + rankable + f[:k]
        else:
            selected += n + r
            selected += [value for value in f_eval if value.split == split]
    selected.sort(key=lambda value: (value.uid, value.query_timestamp, value.workload, value.source_row))
    counts = Counter((value.split, value.workload) for value in selected)
    return selected, {
        "total_specs": len(selected),
        "by_split_workload": {f"{split}:{workload}": count for (split, workload), count in sorted(counts.items())},
        "selection_digest": p7.digest([value.key for value in selected]),
        "selection_uses_H_or_S": False,
    }


def last_before(items: np.ndarray, target: int, end: int) -> int | None:
    matches = np.flatnonzero(items[:end] == target)
    return None if not len(matches) else int(matches[-1])


def qmain(spec: p7.Spec, history: list[tuple[int, int, int]], first_seen: np.ndarray, order: np.ndarray):
    assert spec.target_item is not None
    ordered_first_seen = first_seen[order]
    quality_pool = p7.ranked_pool(history, spec.query_timestamp, first_seen, order, ordered_first_seen, exclude=spec.target_item)
    quality = [spec.target_item] + p7.sample_panel(
        quality_pool, 99, f"p8-{spec.split}-N-quality-v1", spec.uid, spec.query_timestamp
    )
    fidelity_pool = p7.ranked_pool(history, spec.query_timestamp, first_seen, order, ordered_first_seen, exclude=None)
    fidelity = p7.sample_panel(
        fidelity_pool, 100, f"p8-{spec.split}-N-fidelity-v1", spec.uid, spec.query_timestamp
    )
    return (quality, 0), (fidelity, None)


def materialize(specs: list[p7.Spec], artist: np.ndarray, popularity: np.ndarray, first_seen: np.ndarray, order: np.ndarray, output: Path) -> dict:
    writers = {split: p7.SplitWriter(output, split) for split in WINDOWS}
    by_uid: dict[int, list[p7.Spec]] = defaultdict(list)
    for spec in specs:
        by_uid[spec.uid].append(spec)
    view_counts: Counter = Counter()
    for spec in specs:
        if spec.split in EVALUATION_SPLITS:
            kinds = ("quality", "fidelity") if spec.workload in {"N", "F"} else (
                ("fidelity_all_eligible", "quality_rankable", "fidelity_rankable_companion")
                if spec.rankable else ("fidelity_all_eligible",)
            )
        else:
            kinds = ("quality_rankable",) if spec.workload == "R" else ("quality",)
        for kind in kinds:
            view_counts[(spec.workload, spec.split, kind, spec.uid)] += 1

    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    ts_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    organic_parts: list[np.ndarray] = []

    def add(spec: p7.Spec, kind: str, candidate_set_id: str, offset: int, count: int, history: list[tuple[int, int, int]], *, label: int | None, target_index: int | None, prior: bool | None = None, latest: bool | None = None) -> None:
        writers[spec.split].add_request({
            "workload": spec.workload, "split": spec.split, "manifest_kind": kind,
            "request_id": p7.digest([candidate_set_id, kind])[:24], "candidate_set_id": candidate_set_id,
            "uid": spec.uid, "query_timestamp": spec.query_timestamp,
            "raw_user_row_start": spec.raw_user_row_start,
            "raw_prefix_end_exclusive": spec.raw_user_row_start + spec.prefix_end_local,
            "effective_prefix_length": len(history), "candidate_offset": offset,
            "candidate_count": count,
            "base_feature_schema_hash": p7.digest(p7.FEATURE_SCHEMAS[spec.workload]),
            "request_weight": 1.0 / view_counts[(spec.workload, spec.split, kind, spec.uid)],
            "label": label, "target_index": target_index,
            "rankable": spec.rankable if kind == "quality_rankable" else None,
            "target_stratum": spec.target_stratum if "quality" in kind else None,
            "is_organic": spec.is_organic if kind == "quality" else None,
            "prior_30m_same_item": prior if kind == "quality" else None,
            "latest_item": latest if kind == "quality" else None,
            "source_row": spec.source_row,
        })

    def consume() -> None:
        if current_uid is None or current_uid not in by_uid:
            return
        timestamps = np.concatenate(ts_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        organic = np.concatenate(organic_parts).astype(np.int64, copy=False)
        for spec in by_uid[current_uid]:
            start = max(0, spec.prefix_end_local - CONTEXT)
            history = [
                (int(items[index]), int(timestamps[index]), 1 + (1 - int(organic[index])))
                for index in range(start, spec.prefix_end_local)
                if int(timestamps[index]) < spec.query_timestamp
            ]
            writer = writers[spec.split]
            evaluation = spec.split in EVALUATION_SPLITS
            if spec.workload == "N":
                quality, fidelity = qmain(spec, history, first_seen, order)
                views = [("quality", *quality)] + ([ ("fidelity", *fidelity) ] if evaluation else [])
                for kind, candidates, target in views:
                    features = p7.generic_features(history, candidates, spec.query_timestamp, artist, popularity)
                    cid = p7.digest([spec.key, kind, candidates])[:24]
                    offset, count = writer.add_candidate_set("N", cid, spec.query_timestamp, candidates, features)
                    add(spec, kind, cid, offset, count, history, label=None, target_index=target)
            elif spec.workload == "R":
                request = build_return_to_familiar_request(history, spec.query_timestamp, p7.ArrayLookup(artist), p7.ArrayLookup(popularity))
                candidates = list(request.item_ids)
                features = [list(candidate.base_features()) for candidate in request.candidates]
                cid = p7.digest([spec.key, candidates])[:24]
                offset, count = writer.add_candidate_set("R", cid, spec.query_timestamp, candidates, features)
                if evaluation:
                    add(spec, "fidelity_all_eligible", cid, offset, count, history, label=None, target_index=None)
                if spec.rankable:
                    target = request.quality_target_index(int(spec.target_item))
                    if target is None:
                        raise RuntimeError("rankable R target missing")
                    add(spec, "quality_rankable", cid, offset, count, history, label=None, target_index=target)
                    if evaluation:
                        add(spec, "fidelity_rankable_companion", cid, offset, count, history, label=None, target_index=None)
            else:
                candidates = [int(spec.target_item)]
                features = p7.generic_features(history, candidates, spec.query_timestamp, artist, popularity)
                cid = p7.digest([spec.key, candidates])[:24]
                offset, count = writer.add_candidate_set("F", cid, spec.query_timestamp, candidates, features)
                previous = [index for index, (item, _, _) in enumerate(history) if item == spec.target_item]
                prior = bool(previous and spec.query_timestamp - history[previous[-1]][1] <= 1800)
                latest = history[-1][0] == spec.target_item
                add(spec, "quality", cid, offset, count, history, label=spec.label, target_index=None, prior=prior, latest=latest)
                if evaluation:
                    add(spec, "fidelity", cid, offset, count, history, label=None, target_index=None)

    for batch in parquet.iter_batches(batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]):
        uid = batch["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch["item_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch["is_organic"].to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(uid)], strict=True):
            user = int(uid[start])
            if current_uid is not None and user != current_uid:
                consume(); ts_parts.clear(); item_parts.clear(); organic_parts.clear()
            current_uid = user
            ts_parts.append(timestamp[start:end]); item_parts.append(item[start:end]); organic_parts.append(organic[start:end])
    consume()
    common = {
        "contract": "f_release_chain_contract_v1", "status": "sealed_prospective_development",
        "contract_hash": p7.sha256_file(CONTRACT), "materializer_code_hash": p7.sha256_file(Path(__file__)),
        "raw_source_hashes": {name: p7.sha256_file(path) for name, path in {
            "listens": RAW / "flat/50m/listens.parquet", "likes": RAW / "flat/50m/likes.parquet",
            "dislikes": RAW / "flat/50m/dislikes.parquet", "artist_mapping": RAW / "artist_item_mapping.parquet",
        }.items()},
        "P7_8_rewritten": False, "H_or_S_used_for_selection": False,
    }
    indices = {split: writer.finish(common) for split, writer in writers.items()}
    return {split: {"path": str(path.relative_to(ROOT)), "sha256": p7.sha256_file(path)} for split, path in indices.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output already exists and is non-empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    artist, popularity, first_seen, order = catalog_arrays()
    specs, selection = select_specs(load_feedback())
    indices = materialize(specs, artist, popularity, first_seen, order, args.output)
    payload = {"status": "materialized_and_sealed", "selection": selection, "indices": indices, "P7_8_rewritten": False}
    path = args.output / "materialization_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

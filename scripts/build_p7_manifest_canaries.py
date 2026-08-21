#!/usr/bin/env python3
"""Build and audit small P7 N/R/F quality and fidelity canaries only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.data import (
    build_explicit_feedback_query,
    build_return_to_familiar_request,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yambda"
OUTPUT = ROOT / "data/manifests/p7_canary"
DAY = 86_400
BASE_END = 180 * DAY
DEV_START, DEV_END = 203 * DAY, 210 * DAY
RELEASE_CUTOFF = 18_234_000
SIZE = 128
CONTEXT = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_artist_map() -> np.ndarray:
    table = pq.read_table(RAW / "artist_item_mapping.parquet", columns=["artist_id", "item_id"])
    items = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artists = table.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    output = np.full(int(items.max()) + 1, -1, dtype=np.int64)
    output[items] = artists
    return output


def load_base_popularity(size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.int64)
    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    for batch in parquet.iter_batches(batch_size=524_288, columns=["timestamp", "item_id"]):
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        items = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        output += np.bincount(items[timestamp < BASE_END], minlength=size)
    return output


class ArrayLookup:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def get(self, key: int, default: int = 0) -> int:
        return int(self.values[key]) if 0 <= key < len(self.values) else default


def base_features(
    history: list[tuple[int, int, int]],
    candidates: list[int],
    query_timestamp: int,
    artist_by_item: np.ndarray,
    popularity: np.ndarray,
) -> list[list[float]]:
    item_counts = Counter(item for item, _, _ in history)
    item_last = {item: timestamp for item, timestamp, _ in history}
    candidate_artists = {
        item: int(artist_by_item[item]) if item < len(artist_by_item) else -1
        for item in candidates
    }
    artist_counts: Counter[int] = Counter()
    artist_last: dict[int, int] = {}
    for item, timestamp, _ in history:
        artist = int(artist_by_item[item]) if item < len(artist_by_item) else -1
        if artist >= 0:
            artist_counts[artist] += 1
            artist_last[artist] = timestamp
    rows = []
    for item in candidates:
        artist = candidate_artists[item]
        item_missing = item not in item_last
        artist_missing = artist < 0 or artist not in artist_last
        rows.append(
            [
                math.log1p(item_counts[item]),
                math.log1p(artist_counts[artist]) if not artist_missing else 0.0,
                math.log1p(query_timestamp - item_last[item]) if not item_missing else 0.0,
                math.log1p(query_timestamp - artist_last[artist]) if not artist_missing else 0.0,
                math.log1p(popularity[item]) if item < len(popularity) else 0.0,
                float(item_missing),
                float(artist_missing),
            ]
        )
    return rows


def read_jsonl(path: Path, *, min_timestamp: int | None = None) -> list[dict]:
    result = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                timestamp = row.get("request_timestamp", row.get("target_timestamp"))
                if min_timestamp is not None and int(timestamp) < min_timestamp:
                    continue
                result.append(row)
                if len(result) == SIZE:
                    break
    return result


def n_sources() -> tuple[dict[int, dict], dict[int, dict]]:
    quality_rows = read_jsonl(
        ROOT / "data/manifests/cc_theta0_train_v1.jsonl", min_timestamp=180 * DAY
    )
    quality = {int(row["uid"]): row for row in quality_rows}
    panels = pq.read_table(
        ROOT / "data/manifests/yambda50m_v2_qmain32_v2_theta0_theta1.parquet",
        filters=[("panel_id", "=", 0)],
    ).to_pylist()
    snapshots = {
        int(row["uid"]): row
        for row in pq.read_table(
            ROOT / "data/manifests/yambda50m_v2_release_snapshot_theta0_theta1.parquet"
        ).to_pylist()
    }
    fidelity = {}
    for row in panels:
        uid = int(row["uid"])
        if uid not in snapshots or uid in fidelity:
            continue
        fidelity[uid] = {**row, "query_timestamp": int(snapshots[uid]["release_cutoff"])}
        if len(fidelity) == SIZE:
            break
    if len(quality) != SIZE or len(fidelity) != SIZE:
        raise RuntimeError("N source manifests do not contain enough canary users")
    return quality, fidelity


def load_feedback() -> dict[int, list[tuple[int, int, int, int]]]:
    result: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for label, name in ((1, "likes"), (0, "dislikes")):
        table = pq.read_table(
            RAW / f"flat/50m/{name}.parquet",
            columns=["uid", "timestamp", "item_id", "is_organic"],
        )
        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
        for uid, timestamp, item, organic in zip(*columns, strict=True):
            if DEV_START <= int(timestamp) < DEV_END:
                result[int(uid)].append((int(timestamp), int(item), label, int(organic)))
    for values in result.values():
        values.sort()
    return result


def event_rows(items: np.ndarray, timestamps: np.ndarray, organic: np.ndarray, end: int) -> list[tuple[int, int, int]]:
    start = max(0, end - CONTEXT)
    return [
        (int(items[index]), int(timestamps[index]), 1 + (1 - int(organic[index])))
        for index in range(start, end)
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def audit_rows(name: str, rows: list[dict], *, fidelity: bool) -> dict:
    forbidden = {"label", "positive_item_id", "target_item_id", "target_index"}
    if fidelity and any(forbidden & set(row) for row in rows):
        raise ValueError(f"{name} fidelity rows contain label/target fields")
    for row in rows:
        history = row["history"]
        if not history or any(int(event[1]) >= int(row["query_timestamp"]) for event in history):
            raise ValueError(f"{name} contains a non-causal or empty history")
        candidates = row["candidate_item_ids"]
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"{name} candidate universe is empty or duplicated")
        if len(row["base_features"]) != len(candidates):
            raise ValueError(f"{name} base features do not align with candidates")
        if any(len(features) != 7 for features in row["base_features"]):
            raise ValueError(f"{name} base feature width differs from the frozen schema")
        if not fidelity:
            if name.startswith("R") and candidates.count(row["target_item_id"]) != 1:
                raise ValueError("R target must occur exactly once without injection")
            if name.startswith("N") and candidates.count(row["positive_item_id"]) != 1:
                raise ValueError("N injected target must occur exactly once")
    return {
        "rows": len(rows),
        "unique_users": len({int(row["uid"]) for row in rows}),
        "candidate_count": {
            "min": min(len(row["candidate_item_ids"]) for row in rows),
            "median": float(np.median([len(row["candidate_item_ids"]) for row in rows])),
            "max": max(len(row["candidate_item_ids"]) for row in rows),
        },
        "label_or_target_fields_absent": fidelity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    artist_by_item = load_artist_map()
    popularity = load_base_popularity(len(artist_by_item))
    popularity_lookup = ArrayLookup(popularity)
    n_quality_source, n_fidelity_source = n_sources()
    feedback = load_feedback()
    rows = {key: [] for key in ("N_quality", "N_fidelity", "R_quality", "R_fidelity", "F_quality", "F_fidelity")}

    parquet = pq.ParquetFile(RAW / "flat/50m/listens.parquet")
    current_uid: int | None = None
    timestamp_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    organic_parts: list[np.ndarray] = []

    def consume() -> None:
        if current_uid is None:
            return
        timestamps = np.concatenate(timestamp_parts).astype(np.int64, copy=False)
        items = np.concatenate(item_parts).astype(np.int64, copy=False)
        organic = np.concatenate(organic_parts).astype(np.int64, copy=False)

        if current_uid in n_quality_source:
            source = n_quality_source[current_uid]
            query = int(source.get("request_timestamp", source["target_timestamp"]))
            end = int(np.searchsorted(timestamps, query, side="left"))
            history = event_rows(items, timestamps, organic, end)
            candidates = [int(value) for value in source["candidate_item_ids"]]
            rows["N_quality"].append(
                {
                    "workload": "N",
                    "manifest_kind": "quality",
                    "uid": current_uid,
                    "query_timestamp": query,
                    "history": history,
                    "candidate_item_ids": candidates,
                    "base_features": base_features(history, candidates, query, artist_by_item, popularity),
                    "positive_item_id": int(source["positive_item_id"]),
                    "target_index": candidates.index(int(source["positive_item_id"])),
                    "target_injected": True,
                    "source_request_id": source.get("request_id", source["sample_id"]),
                }
            )
        if current_uid in n_fidelity_source:
            source = n_fidelity_source[current_uid]
            query = int(source["query_timestamp"])
            end = int(np.searchsorted(timestamps, query, side="left"))
            history = event_rows(items, timestamps, organic, end)
            candidates = [int(value) for value in source["candidate_item_ids"]]
            rows["N_fidelity"].append(
                {
                    "workload": "N",
                    "manifest_kind": "fidelity",
                    "uid": current_uid,
                    "query_timestamp": query,
                    "history": history,
                    "candidate_item_ids": candidates,
                    "base_features": base_features(history, candidates, query, artist_by_item, popularity),
                    "source_state_hash": source["snapshot_state_hash"],
                    "source_panel_id": int(source["panel_id"]),
                }
            )

        # One first eligible R request and one first rankable R request per user.
        boundaries = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1]
        for start_value in boundaries[1:]:
            start = int(start_value)
            query = int(timestamps[start])
            if query >= DEV_END:
                break
            if query < DEV_START or query - int(timestamps[start - 1]) < 3 * DAY:
                continue
            history = event_rows(items, timestamps, organic, start)
            if len({event[0] for event in history}) < 2:
                continue
            request = build_return_to_familiar_request(
                history, query, ArrayLookup(artist_by_item), popularity_lookup
            )
            candidate_ids = list(request.item_ids)
            common = {
                "workload": "R",
                "uid": current_uid,
                "query_timestamp": query,
                "history": history,
                "candidate_item_ids": candidate_ids,
                "base_features": [list(candidate.base_features()) for candidate in request.candidates],
                "inactivity_gap_seconds": request.inactivity_gap_seconds,
            }
            if len(rows["R_fidelity"]) < SIZE and not any(
                row["uid"] == current_uid for row in rows["R_fidelity"]
            ):
                rows["R_fidelity"].append({**common, "manifest_kind": "fidelity"})
            target = int(items[start])
            target_index = request.quality_target_index(target)
            if (
                target_index is not None
                and len(rows["R_quality"]) < SIZE
                and not any(row["uid"] == current_uid for row in rows["R_quality"])
            ):
                rows["R_quality"].append(
                    {
                        **common,
                        "manifest_kind": "quality",
                        "target_item_id": target,
                        "target_index": target_index,
                        "target_injected": False,
                    }
                )
            if any(row["uid"] == current_uid for row in rows["R_fidelity"]) and any(
                row["uid"] == current_uid for row in rows["R_quality"]
            ):
                break

        if current_uid in feedback and len(rows["F_quality"]) < SIZE:
            selected_feedback = next(
                (
                    event
                    for event in feedback[current_uid]
                    if int(np.searchsorted(timestamps, event[0], side="left")) > 0
                ),
                None,
            )
            if selected_feedback is None:
                return
            timestamp, candidate, label, is_organic = selected_feedback
            listens = [
                (int(item), int(ts), 1 + (1 - int(org)))
                for item, ts, org in zip(items, timestamps, organic, strict=True)
                if int(ts) <= timestamp
            ]
            quality = build_explicit_feedback_query(
                listens, candidate, timestamp, label=label
            )
            fidelity = build_explicit_feedback_query(
                listens, candidate, timestamp, label=None
            )
            features = base_features(
                list(quality.causal_prefix), [candidate], timestamp, artist_by_item, popularity
            )
            common = {
                "workload": "F",
                "uid": current_uid,
                "query_timestamp": timestamp,
                "history": list(quality.causal_prefix),
                "candidate_item_ids": [candidate],
                "base_features": features,
                "candidate_history_position": quality.candidate_history_position,
                "coincident_target_listens_excluded": quality.coincident_target_listens_excluded,
            }
            rows["F_quality"].append(
                {
                    **common,
                    "manifest_kind": "quality",
                    "label": label,
                    "is_organic": is_organic,
                }
            )
            rows["F_fidelity"].append(
                {
                    **common,
                    "manifest_kind": "fidelity",
                    "history": list(fidelity.causal_prefix),
                }
            )

    for batch in parquet.iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        uid = batch.column("uid").to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = batch.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        item = batch.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        organic = batch.column("is_organic").to_numpy(zero_copy_only=False).astype(np.int64)
        boundaries = np.flatnonzero(uid[1:] != uid[:-1]) + 1
        starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(uid)]
        for start, end in zip(starts, ends, strict=True):
            user = int(uid[start])
            if current_uid is not None and user != current_uid:
                consume()
                timestamp_parts.clear()
                item_parts.clear()
                organic_parts.clear()
            current_uid = user
            timestamp_parts.append(timestamp[start:end])
            item_parts.append(item[start:end])
            organic_parts.append(organic[start:end])
    consume()

    if any(len(value) != SIZE for value in rows.values()):
        raise RuntimeError(f"canary size mismatch: { {key: len(value) for key, value in rows.items()} }")
    audit = {}
    for name, values in rows.items():
        fidelity = name.endswith("fidelity")
        audit[name] = audit_rows(name, values, fidelity=fidelity)
        path = args.output / f"{name.lower()}_v1.jsonl"
        write_jsonl(path, values)
        audit[name]["artifact"] = str(path.relative_to(ROOT))
        audit[name]["sha256"] = sha256(path)
        audit[name]["content_digest"] = digest(values)

    exclusions = {
        "contract": "p7_5_materialization_contract_v1",
        "status": "frozen_before_full_qualification_materialization",
        "labeled_canaries_from_qualification": 0,
        "labeled_canary_sources": {
            "N_quality": "residual_train",
            "R_quality": "development",
            "F_quality": "development",
        },
        "conservative_exact_request_exclusions": [
            {
                "workload": "N",
                "manifest_kind": "fidelity",
                "uid": int(row["uid"]),
                "query_timestamp": int(row["query_timestamp"]),
                "candidate_digest": digest(row["candidate_item_ids"]),
                "reason": "existing qualification-time target-free panel inspected by P7.5 canary",
            }
            for row in rows["N_fidelity"]
        ],
        "exclusion_scope": "exact_request_signature_not_entire_user",
    }
    exclusion_path = args.output / "qualification_exclusions_v1.json"
    exclusion_path.write_text(json.dumps(exclusions, indent=2) + "\n")

    metadata = {
        "contract": "p7_4_training_contract_v1",
        "stage": "P7.5_manifest_canary",
        "status": "passed",
        "canary_size_per_manifest": SIZE,
        "manifests": audit,
        "invariants": {
            "all_histories_strictly_causal": True,
            "quality_and_fidelity_separate": True,
            "fidelity_has_no_label_or_target_fields": True,
            "R_complete_familiar_universe_no_sampling": True,
            "R_target_not_injected_and_occurs_once": True,
            "F_coincident_listen_excluded": True,
            "base_features_materialized_once_per_request_path": True,
            "query_token_persistent_state_regression": "tests/test_cc_scoring_contract.py",
            "R_chunked_exactness_regression": "tests/test_p7_stateful_workload_contract.py",
        },
        "qualification_exclusions": {
            "artifact": str(exclusion_path.relative_to(ROOT)),
            "sha256": sha256(exclusion_path),
            "count": len(exclusions["conservative_exact_request_exclusions"]),
            "labeled_qualification_canaries": 0,
        },
        "sources": {
            "N_quality": "data/manifests/cc_theta0_train_v1.jsonl residual_train slice",
            "N_fidelity": "data/manifests/yambda50m_v2_qmain32_v2_theta0_theta1.parquet panel_id=0",
            "R_F": "raw Yambda-50M causal events",
        },
        "limitations": [
            "Canaries validate contracts only and are not quality or H evidence.",
            "No P7 qualification quality manifest was generated, opened, or scored.",
            "N remains the retained P6 No-Go negative control; this canary does not requalify it.",
        ],
    }
    meta_path = args.output / "p7_manifest_canary_v1.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "passed", "metadata": str(meta_path), "sizes": {key: len(value) for key, value in rows.items()}}, indent=2))


if __name__ == "__main__":
    main()

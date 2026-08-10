from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from hstu_kvcache.data.qb_large_multifield import (
    behavior_values as qb_behavior_values,
)
from hstu_kvcache.data.qb_large_multifield import (
    load_catalog as load_qb_catalog,
)
from hstu_kvcache.data.qb_large_multifield import (
    load_qb_frame,
)
from hstu_kvcache.data.qb_large_multifield import (
    positive_values as qb_positive_values,
)
from hstu_kvcache.data.qk_xp_edge_inputs import (
    EdgeInputConfig as QKReadConfig,
)
from hstu_kvcache.data.qk_xp_edge_inputs import (
    action_masks,
    behavior_values,
    consume_user_positions,
    read_qk_chunks,
    source_fingerprint,
)
from hstu_kvcache.data.qk_xp_edge_inputs import (
    load_catalog as load_qk_catalog,
)
from hstu_kvcache.migration.variable_inference import (
    ROLE_CODES,
    array_sha256,
    file_sha256,
    load_corpus,
    prefix_schedule,
    stable_order,
    write_corpus,
)
from hstu_kvcache.streaming.qb_multifield_training import (
    load_qb_large_corpus,
)

DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
QK_MEMBER = "Tenrec/QK-video.csv"
QB_MEMBER = "Tenrec/QB-video.csv"
QK_HET = Path("data/processed/evokv_foundation/x_qk_het_foundation.npz")
QK_CATALOG = Path(
    "data/processed/evokv_d3_m1_qk_entity_cache/entity_catalog_base64_top250000.npz"
)
QB_CORPUS = Path(
    "data/processed/evokv_qb_large_multifield/mf9_e4096_corpus.npz"
)
QB_CATALOG = Path(
    "data/processed/evokv_qb_large_multifield/mf9_e4096_catalog.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("qk", "qb"), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--qk-het", type=Path, default=QK_HET)
    parser.add_argument("--qk-catalog", type=Path, default=QK_CATALOG)
    parser.add_argument("--qk-records", type=int, default=4096)
    parser.add_argument("--qk-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--qb-corpus", type=Path, default=QB_CORPUS)
    parser.add_argument("--qb-catalog", type=Path, default=QB_CATALOG)
    parser.add_argument("--qb-fit-records", type=int, default=128)
    parser.add_argument("--qb-probe-records", type=int, default=32)
    parser.add_argument("--qb-qualification-records", type=int, default=1000)
    parser.add_argument("--minimum-initial-tokens", type=int, default=64)
    parser.add_argument(
        "--selection-salt",
        default="evokv-large-variable-inference-20260805-v0",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def existing_corpus(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.output.exists():
        if args.summary.exists():
            raise FileNotFoundError("variable inference summary exists without corpus")
        return None
    corpus = load_corpus(args.output)
    expected_roles = (
        {"fit": 0, "probe": 0, "qualification": args.qk_records}
        if args.dataset == "qk"
        else {
            "fit": args.qb_fit_records,
            "probe": args.qb_probe_records,
            "qualification": args.qb_qualification_records,
        }
    )
    expected_edges = 3 if args.dataset == "qk" else 2
    record_roles = (
        ("qualification",)
        if args.dataset == "qk"
        else ("fit", "probe", "qualification")
    )
    expected_record_bindings = {
        f"{role}_{kind}_ids_sha256": array_sha256(
            corpus.arrays[name][corpus.role_records(role)]
        )
        for role in record_roles
        for kind, name in (
            ("source", "record_source_ids"),
            ("user", "record_user_ids"),
        )
    }
    if (
        corpus.dataset != args.dataset
        or corpus.edge_count != expected_edges
        or corpus.metadata.get("roles") != expected_roles
        or int(corpus.metadata.get("minimum_initial_tokens", -1))
        != args.minimum_initial_tokens
        or corpus.metadata.get("selection_salt") != args.selection_salt
        or corpus.metadata.get("record_bindings")
        != expected_record_bindings
        or corpus.metadata.get("source", {}).get("sha256")
        != file_sha256(args.source)
        or args.dataset == "qk"
        and int(corpus.metadata.get("item_alignment_events_verified", -1))
        != int(corpus.arrays["record_offsets"][-1])
        or args.dataset == "qb"
        and int(corpus.metadata.get("frozen_overlap_events_verified", 0)) < 1
    ):
        raise ValueError("existing variable inference corpus differs")
    bindings = corpus.metadata.get("bindings", {})
    if args.dataset == "qk":
        expected_bindings = {
            "het": file_sha256(args.qk_het),
            "catalog": file_sha256(args.qk_catalog),
        }
    else:
        expected_bindings = {
            "fixed_role_corpus": file_sha256(args.qb_corpus),
            "catalog": file_sha256(args.qb_catalog),
        }
    if any(
        bindings.get(name, {}).get("sha256") != digest
        for name, digest in expected_bindings.items()
    ):
        raise ValueError("existing variable inference bindings differ")
    descriptor = {
        "path": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": corpus.file_sha256,
        "content_sha256": corpus.content_sha256,
    }
    if args.summary.exists():
        summary = json.loads(args.summary.read_text())
        if (
            summary.get("schema")
            != "evokv_large_variable_inference_build_v0"
            or summary.get("status") != "pass"
            or summary.get("dataset") != args.dataset
            or summary.get("corpus") != descriptor
            or summary.get("metadata") != corpus.metadata
        ):
            raise ValueError("existing variable inference summary differs")
        return summary
    summary = {
        "schema": "evokv_large_variable_inference_build_v0",
        "status": "pass",
        "scientific_result": False,
        "formal_result": False,
        "dataset": args.dataset,
        "corpus": descriptor,
        "metadata": corpus.metadata,
        "elapsed_seconds": 0.0,
        "recovered_existing_corpus": True,
        "source_code": {
            "path": str(Path(__file__)),
            "sha256": file_sha256(Path(__file__)),
        },
    }
    atomic_json(args.summary, summary)
    return summary


def quantiles(value: np.ndarray) -> dict[str, float]:
    return {
        str(point): float(np.quantile(value, point))
        for point in (0, 0.5, 0.95, 0.99, 1)
    }


def positive_audit(
    labels: np.ndarray,
    offsets: np.ndarray,
    schedule: np.ndarray,
) -> dict[str, object]:
    edges = []
    for edge in range(schedule.shape[1] - 1):
        records_with_positive = 0
        positives = 0
        for record in range(len(schedule)):
            start = int(offsets[record])
            left = start + int(schedule[record, edge]) + 1
            right = start + int(schedule[record, edge + 1]) + 1
            count = int(labels[left:right].sum())
            positives += count
            records_with_positive += int(count > 0)
        edges.append(
            {
                "edge_ordinal": edge,
                "positive_targets": positives,
                "records_with_positive": records_with_positive,
                "record_coverage": records_with_positive / len(schedule),
            }
        )
    return {"edges": edges, "all_edges_have_positive_targets": all(x["positive_targets"] > 0 for x in edges)}


def qk_selected_records(path: Path, count: int, salt: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        records = len(source["record_user_ids"])
    if count < 1 or count > records:
        raise ValueError("QK variable inference record count differs")
    ordered = stable_order(np.arange(records, dtype=np.int64), f"{salt}:qk")
    return np.sort(ordered[:count])


def build_qk(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    selected = qk_selected_records(args.qk_het, args.qk_records, args.selection_salt)
    with np.load(args.qk_het, allow_pickle=False) as source:
        users = source["record_user_ids"][selected].astype(np.int64, copy=True)
        history_offsets = source["history_offsets"].astype(np.int64, copy=True)
        history_items = source["history_item_idx"].astype(np.uint32, copy=True)
        target_start = source["target_start"][selected].astype(np.int64, copy=True)
        target_end = source["target_end"][selected].astype(np.int64, copy=True)
        het_metadata = json.loads(str(source["metadata_json"].item()))
    lengths = target_end - target_start
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    items = np.empty(int(offsets[-1]), dtype=np.uint32)
    for output_record, source_record in enumerate(selected):
        source_left = int(history_offsets[source_record] + target_start[output_record])
        source_right = int(history_offsets[source_record] + target_end[output_record])
        output_left = int(offsets[output_record])
        output_right = int(offsets[output_record + 1])
        items[output_left:output_right] = history_items[source_left:source_right]
    behaviors = np.zeros(len(items), dtype=np.uint8)
    raw_labels = np.zeros(len(items), dtype=np.uint8)
    filled = np.zeros(len(items), dtype=np.bool_)
    user_to_record = np.full(int(users.max()) + 1, -1, dtype=np.int32)
    user_to_record[users] = np.arange(len(users), dtype=np.int32)
    seen = np.zeros(0, dtype=np.int32)
    read_config = QKReadConfig(
        source=args.source,
        member=QK_MEMBER,
        catalog_cache=args.qk_catalog,
        roles=Path("configs/evokv_foundation/qk_post_base_roles.json"),
        output=args.output,
        summary=args.summary,
        chunk_size=args.qk_chunk_size,
    )
    qk_catalog = load_qk_catalog(read_config, source_fingerprint(read_config))
    scanned = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(read_qk_chunks(read_config), start=1):
        chunk_users = chunk["user_id"].to_numpy(dtype=np.int64, copy=False)
        positions, seen = consume_user_positions(chunk_users, seen)
        in_range = (chunk_users >= 0) & (chunk_users < len(user_to_record))
        records = np.full(len(chunk_users), -1, dtype=np.int32)
        records[in_range] = user_to_record[chunk_users[in_range]]
        selected_rows = records >= 0
        if selected_rows.any():
            indices = np.flatnonzero(selected_rows)
            row_records = records[selected_rows].astype(np.int64, copy=False)
            row_positions = positions[selected_rows]
            within = (row_positions >= target_start[row_records]) & (
                row_positions < target_end[row_records]
            )
            if within.any():
                indices = indices[within]
                row_records = row_records[within]
                row_positions = row_positions[within]
                destinations = offsets[row_records] + row_positions - target_start[row_records]
                mapped_items, _, _, _ = qk_catalog.map(
                    chunk["item_id"].to_numpy(dtype=np.int64, copy=False)[indices]
                )
                if not np.array_equal(mapped_items, items[destinations]):
                    raise ValueError("QK variable inference item alignment differs")
                click = chunk["click"].to_numpy(dtype=np.uint8, copy=False)[indices]
                follow = chunk["follow"].to_numpy(dtype=np.uint8, copy=False)[indices]
                like = chunk["like"].to_numpy(dtype=np.uint8, copy=False)[indices]
                share = chunk["share"].to_numpy(dtype=np.uint8, copy=False)[indices]
                if filled[destinations].any():
                    raise ValueError("QK variable inference rows overlap")
                behaviors[destinations] = behavior_values(click, follow, like, share)
                raw_labels[destinations] = (
                    action_masks(click, follow, like, share) > 0
                ).astype(np.uint8)
                filled[destinations] = True
        scanned += len(chunk)
        print(
            json.dumps(
                {
                    "phase": "qk_variable_inference",
                    "chunk": chunk_index,
                    "source_rows": scanned,
                    "materialized_rows": int(filled.sum()),
                    "required_rows": len(filled),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if filled.all():
            break
    if not filled.all():
        raise ValueError("QK variable inference histories are incomplete")
    predicted = items <= 250_000
    labels = raw_labels & predicted.astype(np.uint8)
    deltas = np.ones(len(items), dtype=np.float32)
    for record in range(len(users)):
        if target_start[record] == 0:
            deltas[int(offsets[record])] = 0.0
    schedule = prefix_schedule(lengths, 3, args.minimum_initial_tokens)
    arrays = {
        "record_source_ids": selected.astype(np.int64, copy=False),
        "record_user_ids": users,
        "record_role": np.full(len(users), ROLE_CODES["qualification"], dtype=np.uint8),
        "record_offsets": offsets,
        "record_valid_lengths": lengths.astype(np.int64, copy=False),
        "edge_prefix_lengths": schedule,
        "feature_ids": items[:, None],
        "target_item_ids": items.copy(),
        "behaviors": behaviors,
        "time_deltas": deltas,
        "labels": labels.astype(np.uint8, copy=False),
        "is_prediction_item": predicted.astype(np.uint8),
    }
    metadata = {
        "dataset": "qk",
        "edge_count": 3,
        "feature_fields": 1,
        "minimum_initial_tokens": args.minimum_initial_tokens,
        "selection_salt": args.selection_salt,
        "roles": {"fit": 0, "probe": 0, "qualification": len(users)},
        "source": {
            "path": str(args.source.resolve()),
            "sha256": file_sha256(args.source),
            "member": QK_MEMBER,
        },
        "bindings": {
            "het": {"path": str(args.qk_het), "sha256": file_sha256(args.qk_het), "content_sha256": het_metadata.get("content_sha256")},
            "catalog": {"path": str(args.qk_catalog), "sha256": file_sha256(args.qk_catalog)},
        },
        "record_selection": "stable hash sample from the frozen 65536-record QK HET universe",
        "record_bindings": {
            "qualification_source_ids_sha256": array_sha256(selected),
            "qualification_user_ids_sha256": array_sha256(users),
        },
        "schedule": "c0=min(T-1-E,max(64,floor((T-1)/2))); cE=T-1; interior boundaries evenly partition valid ordinal positions",
        "quality_action_independence": True,
        "source_rows_scanned": scanned,
        "item_alignment_events_verified": int(filled.sum()),
        "valid_length_quantiles": quantiles(lengths),
        "prefix_length_quantiles_by_boundary": [quantiles(schedule[:, index]) for index in range(schedule.shape[1])],
        "positive_audit": positive_audit(labels, offsets, schedule),
        "scientific_result": False,
        "formal_result": False,
    }
    return arrays, metadata


def qb_role_selection(corpus, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tuning_records = corpus.role_records("tuning")
    qualification_records = corpus.role_records("qualification")
    required_tuning = args.qb_fit_records + args.qb_probe_records
    if (
        min(args.qb_fit_records, args.qb_probe_records, args.qb_qualification_records) < 1
        or required_tuning > len(tuning_records)
        or args.qb_qualification_records > len(qualification_records)
    ):
        raise ValueError("QB variable inference role counts differ")
    tuning_users = corpus.arrays["role_record_user_ids"][tuning_records]
    qualification_users = corpus.arrays["role_record_user_ids"][qualification_records]
    ordered_tuning_users = stable_order(tuning_users, f"{args.selection_salt}:qb:tuning")
    ordered_qualification_users = stable_order(
        qualification_users, f"{args.selection_salt}:qb:qualification"
    )
    user_to_record = {
        int(user): int(record)
        for user, record in zip(
            corpus.arrays["role_record_user_ids"],
            np.arange(len(corpus.arrays["role_record_user_ids"])),
            strict=True,
        )
    }
    fit_users = ordered_tuning_users[: args.qb_fit_records]
    probe_users = ordered_tuning_users[
        args.qb_fit_records : required_tuning
    ]
    qualification_users = ordered_qualification_users[: args.qb_qualification_records]
    users = np.concatenate((fit_users, probe_users, qualification_users))
    source_records = np.asarray([user_to_record[int(user)] for user in users], dtype=np.int64)
    roles = np.concatenate(
        (
            np.full(len(fit_users), ROLE_CODES["fit"], dtype=np.uint8),
            np.full(len(probe_users), ROLE_CODES["probe"], dtype=np.uint8),
            np.full(len(qualification_users), ROLE_CODES["qualification"], dtype=np.uint8),
        )
    )
    return users, source_records, roles


def build_qb(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    corpus = load_qb_large_corpus(args.qb_corpus, args.qb_catalog)
    catalog = load_qb_catalog(args.qb_catalog)
    users, source_records, roles = qb_role_selection(corpus, args)
    frame = load_qb_frame(args.source, QB_MEMBER)
    selected = frame[frame["user_id"].isin(users)].copy()
    user_order = {int(user): index for index, user in enumerate(users)}
    selected["record_order"] = selected["user_id"].map(user_order)
    lengths_by_user = selected.groupby("record_order", sort=True).size().to_numpy(dtype=np.int64)
    boundary = np.minimum(lengths_by_user, 544)
    starts = np.maximum(0, boundary - 512)
    selected = selected[
        (selected["raw_ordinal"] >= selected["user_id"].map({int(user): int(starts[index]) for index, user in enumerate(users)}))
        & (selected["raw_ordinal"] < selected["user_id"].map({int(user): int(boundary[index]) for index, user in enumerate(users)}))
    ].copy()
    selected.sort_values(["record_order", "raw_ordinal"], inplace=True, kind="stable")
    counts = selected.groupby("record_order", sort=True).size().to_numpy(dtype=np.int64)
    if len(counts) != len(users) or not np.array_equal(counts, boundary - starts):
        raise ValueError("QB variable inference histories are incomplete")
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    features, targets, direct = catalog.map_frame(selected)
    behaviors = qb_behavior_values(selected)
    raw_labels = qb_positive_values(selected).astype(np.uint8)
    labels = raw_labels & direct.astype(np.uint8)
    raw_ordinals = selected["raw_ordinal"].to_numpy(dtype=np.int64, copy=False)
    frozen_overlap_events = 0
    frozen_offsets = corpus.arrays["role_record_offsets"]
    for record, source_record in enumerate(source_records):
        output_left = int(offsets[record])
        output_right = int(offsets[record + 1])
        ordinal = raw_ordinals[output_left:output_right]
        overlap = ordinal < 104
        if not overlap.any():
            raise ValueError("QB variable inference lacks frozen overlap")
        frozen_left = int(frozen_offsets[source_record])
        expected = frozen_left + ordinal[overlap]
        if (
            not np.array_equal(features[output_left:output_right][overlap], corpus.arrays["role_feature_ids"][expected])
            or not np.array_equal(targets[output_left:output_right][overlap], corpus.arrays["role_target_item_ids"][expected])
            or not np.array_equal(behaviors[output_left:output_right][overlap], corpus.arrays["role_behavior"][expected])
            or not np.array_equal(labels[output_left:output_right][overlap], corpus.arrays["role_label"][expected])
            or not np.array_equal(direct[output_left:output_right][overlap], corpus.arrays["role_is_prediction_item"][expected])
        ):
            raise ValueError("QB variable inference frozen overlap differs")
        frozen_overlap_events += int(overlap.sum())
    deltas = np.ones(len(selected), dtype=np.float32)
    for record in range(len(users)):
        if starts[record] == 0:
            deltas[int(offsets[record])] = 0.0
    schedule = prefix_schedule(counts, 2, args.minimum_initial_tokens)
    arrays = {
        "record_source_ids": source_records,
        "record_user_ids": users.astype(np.int64, copy=False),
        "record_role": roles,
        "record_offsets": offsets,
        "record_valid_lengths": counts.astype(np.int64, copy=False),
        "edge_prefix_lengths": schedule,
        "feature_ids": features.astype(np.uint32, copy=False),
        "target_item_ids": targets.astype(np.uint32, copy=False),
        "behaviors": behaviors.astype(np.uint8, copy=False),
        "time_deltas": deltas,
        "labels": labels.astype(np.uint8, copy=False),
        "is_prediction_item": direct.astype(np.uint8),
    }
    metadata = {
        "dataset": "qb",
        "edge_count": 2,
        "feature_fields": catalog.profile.feature_count,
        "minimum_initial_tokens": args.minimum_initial_tokens,
        "selection_salt": args.selection_salt,
        "roles": {
            "fit": args.qb_fit_records,
            "probe": args.qb_probe_records,
            "qualification": args.qb_qualification_records,
        },
        "source": {
            "path": str(args.source.resolve()),
            "sha256": file_sha256(args.source),
            "member": QB_MEMBER,
        },
        "bindings": {
            "fixed_role_corpus": {"path": str(args.qb_corpus), "sha256": corpus.file_sha256, "content_sha256": corpus.content_sha256},
            "catalog": {"path": str(args.qb_catalog), "sha256": file_sha256(args.qb_catalog), "content_sha256": catalog.metadata.get("content_sha256")},
        },
        "record_selection": "stable fit/probe sample from frozen tuning users and stable qualification sample from frozen report-only users",
        "record_bindings": {
            f"{role}_{kind}_ids_sha256": array_sha256(
                arrays[name][np.flatnonzero(roles == ROLE_CODES[role])]
            )
            for role in ("fit", "probe", "qualification")
            for kind, name in (
                ("source", "record_source_ids"),
                ("user", "record_user_ids"),
            )
        },
        "schedule": "c0=min(T-1-E,max(64,floor((T-1)/2))); cE=T-1; interior boundaries evenly partition valid ordinal positions",
        "quality_action_independence": True,
        "frozen_overlap_events_verified": frozen_overlap_events,
        "valid_length_quantiles": quantiles(counts),
        "prefix_length_quantiles_by_boundary": [quantiles(schedule[:, index]) for index in range(schedule.shape[1])],
        "positive_audit": positive_audit(labels, offsets, schedule),
        "scientific_result": False,
        "formal_result": False,
    }
    return arrays, metadata


def main() -> None:
    args = parse_args()
    existing = existing_corpus(args)
    if existing is not None:
        print(
            json.dumps(
                {
                    "status": "pass_reused",
                    "output": str(args.output),
                    "summary": str(args.summary),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    started = time.perf_counter()
    arrays, metadata = build_qk(args) if args.dataset == "qk" else build_qb(args)
    descriptor = write_corpus(args.output, arrays, metadata)
    materialized = load_corpus(args.output)
    summary = {
        "schema": "evokv_large_variable_inference_build_v0",
        "status": "pass",
        "scientific_result": False,
        "formal_result": False,
        "dataset": args.dataset,
        "corpus": descriptor,
        "metadata": materialized.metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "source_code": {"path": str(Path(__file__)), "sha256": file_sha256(Path(__file__))},
    }
    atomic_json(args.summary, summary)
    print(json.dumps({"status": "pass", "output": str(args.output), "summary": str(args.summary)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

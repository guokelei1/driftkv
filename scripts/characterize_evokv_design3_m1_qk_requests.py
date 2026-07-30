from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hstu_kvcache.data import load_prepared_exposure_plan
from hstu_kvcache.migration.design2_plan import canonical_sha256, file_sha256
from hstu_kvcache.utils import save_json

PROTOCOL = "evokv_design3_m1_qk_request_characterization_dev_v0"
DEFAULT_PREPARED_DATA = (
    "data/processed/evokv_d3_m1_qk_ctx512144_8704.npz"
)
DEFAULT_ACTION_SNAPSHOT = (
    "configs/evokv_d3/m1/"
    "qk_ctx512144_adjacent_action_snapshot.json"
)
DEFAULT_OUTPUT = (
    "configs/evokv_d3/m1/"
    "qk_ctx512144_request_characterization.json"
)
WORLD_SIZE = 2
HIDDEN_SIZE = 512
NUM_LAYERS = 16
KV_COMPONENTS = 2
KV_DTYPE_BYTES = 2
EMBEDDING_DTYPE_BYTES = 4
ITEM_ID_DTYPE_BYTES = 8
HISTORY_FIELDS = (
    "item_ids",
    "behaviors",
    "time_deltas",
    "timestamps",
)
ROLE_NAMES = ("prediction", "context")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", default=DEFAULT_PREPARED_DATA)
    parser.add_argument(
        "--action-snapshot",
        default=DEFAULT_ACTION_SNAPSHOT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-records", type=int, default=8192)
    parser.add_argument(
        "--expected-history-tokens",
        type=int,
        default=512,
    )
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def typed_array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def history_identity_sha256(
    history: dict[str, np.ndarray],
    start: int,
    stop: int,
) -> str:
    if not 0 <= start <= stop <= len(history["item_ids"]):
        raise ValueError("target history identity extent is invalid")
    return canonical_sha256(
        {
            "tokens": stop - start,
            "item_ids_sha256": typed_array_sha256(
                history["item_ids"][start:stop],
                "<i8",
            ),
            "behaviors_sha256": typed_array_sha256(
                history["behaviors"][start:stop],
                "<i8",
            ),
            "time_deltas_sha256": typed_array_sha256(
                history["time_deltas"][start:stop],
                "<f4",
            ),
            "timestamps_sha256": typed_array_sha256(
                history["timestamps"][start:stop],
                "<i8",
            ),
        }
    )


def validate_snapshot_integrity(
    snapshot: dict[str, object],
    prepared_data: str | Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    expected_prepared_sha256 = file_sha256(prepared_data)
    if snapshot.get("prepared_data_sha256") != expected_prepared_sha256:
        raise ValueError("action snapshot prepared-data hash differs")
    recorded_plan_sha256 = snapshot.get(
        "owner_independent_plan_sha256"
    )
    unhashed = {
        key: value
        for key, value in snapshot.items()
        if key not in {"owner_independent_plan_sha256", "bindings"}
    }
    if (
        not isinstance(recorded_plan_sha256, str)
        or canonical_sha256(unhashed) != recorded_plan_sha256
    ):
        raise ValueError("action snapshot content hash differs")
    raw_records = snapshot.get("records")
    raw_layout = snapshot.get("layout")
    if not isinstance(raw_records, list) or not isinstance(raw_layout, dict):
        raise ValueError("action snapshot records or layout are invalid")
    records = []
    for value in raw_records:
        if not isinstance(value, dict):
            raise ValueError("action snapshot record is invalid")
        records.append(value)
    record_ids = [int(value.get("record_id", -1)) for value in records]
    if record_ids != list(range(len(records))):
        raise ValueError("action snapshot record IDs must be contiguous")
    actions = [str(value.get("requested_action", "")) for value in records]
    if any(value not in {"exact", "compiled"} for value in actions):
        raise ValueError("action snapshot contains an unsupported action")
    counts = snapshot.get("counts")
    if (
        not isinstance(counts, dict)
        or int(counts.get("records", -1)) != len(records)
        or int(counts.get("exact", -1)) != actions.count("exact")
        or int(counts.get("compiled", -1)) != actions.count("compiled")
    ):
        raise ValueError("action snapshot counts differ from records")
    layout_names = (
        "history_tokens",
        "target_filtered_start",
        "target_filtered_stop",
        "delta_start",
        "delta_tokens",
        "target_prefix_tokens",
        "latest_tokens",
        "final_tokens",
    )
    try:
        layout = {
            name: int(raw_layout[name])
            for name in layout_names
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("action snapshot layout is incomplete") from error
    if (
        layout["history_tokens"] < 1
        or layout["target_filtered_start"] < 0
        or layout["target_filtered_stop"]
        - layout["target_filtered_start"]
        != layout["history_tokens"]
        or layout["delta_start"] < 0
        or layout["delta_tokens"] < 0
        or layout["target_prefix_tokens"]
        != layout["delta_start"] + layout["delta_tokens"]
        or layout["final_tokens"]
        != layout["target_prefix_tokens"] + layout["latest_tokens"]
        or layout["final_tokens"] != layout["history_tokens"]
    ):
        raise ValueError("action snapshot adjacent layout is inconsistent")
    return records, layout


def strict_cow_lpt_owners(
    records: list[dict[str, object]],
    world_size: int,
) -> dict[int, int]:
    if world_size < 1:
        raise ValueError("world size must be positive")
    loads = [0] * world_size
    weighted = []
    for value in records:
        record_id = int(value["record_id"])
        weight = int(value["old_tokens"]) + int(value["final_tokens"])
        if weight < 1:
            raise ValueError("record has no strict-COW payload")
        weighted.append((record_id, weight))
    owners: dict[int, int] = {}
    for record_id, weight in sorted(
        weighted,
        key=lambda value: (-value[1], value[0]),
    ):
        rank = min(
            range(world_size),
            key=lambda value: (loads[value], value),
        )
        owners[record_id] = rank
        loads[rank] += weight
    return dict(sorted(owners.items()))


def owner_map_sha256(owners: dict[int, int]) -> str:
    return canonical_sha256(
        {
            "record_owner_map": [
                {
                    "record_id": record_id,
                    "owner_rank": owner,
                }
                for record_id, owner in sorted(owners.items())
            ]
        }
    )


def addressable_item_rows(
    num_items: int,
    rank: int,
    world_size: int,
) -> int:
    first = world_size if rank == 0 else rank
    if first > num_items:
        return 0
    return (num_items - first) // world_size + 1


def physical_embedding_rows(
    num_items: int,
    rank: int,
    world_size: int,
) -> int:
    if rank > num_items:
        return 0
    return (num_items - rank) // world_size + 1


class RequestAccumulator:
    def __init__(
        self,
        num_items: int,
        num_prediction_items: int,
        world_size: int,
    ) -> None:
        self.num_items = num_items
        self.num_prediction_items = num_prediction_items
        self.world_size = world_size
        rows = num_items + 1
        self.request_tokens = 0
        self.role_tokens = np.zeros(2, dtype=np.int64)
        self.rank_tokens = np.zeros(world_size, dtype=np.int64)
        self.rank_local_tokens = np.zeros(world_size, dtype=np.int64)
        self.rank_offrank_tokens = np.zeros(world_size, dtype=np.int64)
        self.rank_role_tokens = np.zeros(
            (world_size, 2),
            dtype=np.int64,
        )
        self.rank_role_local_tokens = np.zeros(
            (world_size, 2),
            dtype=np.int64,
        )
        self.rank_role_offrank_tokens = np.zeros(
            (world_size, 2),
            dtype=np.int64,
        )
        self.seen = np.zeros(rows, dtype=np.bool_)
        self.role_seen = np.zeros((2, rows), dtype=np.bool_)
        self.rank_seen = np.zeros(
            (world_size, rows),
            dtype=np.bool_,
        )
        self.rank_local_seen = np.zeros(
            (world_size, rows),
            dtype=np.bool_,
        )
        self.rank_offrank_seen = np.zeros(
            (world_size, rows),
            dtype=np.bool_,
        )
        self.rank_role_seen = np.zeros(
            (world_size, 2, rows),
            dtype=np.bool_,
        )
        self.rank_role_local_seen = np.zeros(
            (world_size, 2, rows),
            dtype=np.bool_,
        )
        self.rank_role_offrank_seen = np.zeros(
            (world_size, 2, rows),
            dtype=np.bool_,
        )

    def add(self, requester_rank: int, item_ids: np.ndarray) -> None:
        ids = np.asarray(item_ids, dtype=np.int64)
        if (
            ids.ndim != 1
            or np.any(ids < 1)
            or np.any(ids > self.num_items)
            or not 0 <= requester_rank < self.world_size
        ):
            raise ValueError("embedding request IDs or requester are invalid")
        roles = (ids > self.num_prediction_items).astype(
            np.int64,
            copy=False,
        )
        embedding_owners = np.remainder(ids, self.world_size)
        local = embedding_owners == requester_rank
        offrank = ~local
        count = len(ids)
        self.request_tokens += count
        self.rank_tokens[requester_rank] += count
        self.rank_local_tokens[requester_rank] += int(
            np.count_nonzero(local)
        )
        self.rank_offrank_tokens[requester_rank] += int(
            np.count_nonzero(offrank)
        )
        self.seen[ids] = True
        self.rank_seen[requester_rank, ids] = True
        self.rank_local_seen[requester_rank, ids[local]] = True
        self.rank_offrank_seen[requester_rank, ids[offrank]] = True
        for role in range(2):
            role_mask = roles == role
            role_ids = ids[role_mask]
            local_role_ids = ids[role_mask & local]
            offrank_role_ids = ids[role_mask & offrank]
            self.role_tokens[role] += len(role_ids)
            self.rank_role_tokens[requester_rank, role] += len(
                role_ids
            )
            self.rank_role_local_tokens[requester_rank, role] += len(
                local_role_ids
            )
            self.rank_role_offrank_tokens[
                requester_rank,
                role,
            ] += len(offrank_role_ids)
            self.role_seen[role, role_ids] = True
            self.rank_role_seen[
                requester_rank,
                role,
                role_ids,
            ] = True
            self.rank_role_local_seen[
                requester_rank,
                role,
                local_role_ids,
            ] = True
            self.rank_role_offrank_seen[
                requester_rank,
                role,
                offrank_role_ids,
            ] = True

    @staticmethod
    def _fraction(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _unique(values: np.ndarray) -> int:
        return int(np.count_nonzero(values))

    def _role_summary(
        self,
        role: int,
        requester_rank: int | None = None,
    ) -> dict[str, int | float]:
        if requester_rank is None:
            tokens = int(self.role_tokens[role])
            unique = self._unique(self.role_seen[role])
            offrank_tokens = int(
                self.rank_role_offrank_tokens[:, role].sum()
            )
            offrank_unique = sum(
                self._unique(
                    self.rank_role_offrank_seen[rank, role]
                )
                for rank in range(self.world_size)
            )
        else:
            tokens = int(
                self.rank_role_tokens[requester_rank, role]
            )
            unique = self._unique(
                self.rank_role_seen[requester_rank, role]
            )
            offrank_tokens = int(
                self.rank_role_offrank_tokens[
                    requester_rank,
                    role,
                ]
            )
            offrank_unique = self._unique(
                self.rank_role_offrank_seen[
                    requester_rank,
                    role,
                ]
            )
        return {
            "request_tokens": tokens,
            "unique_item_rows": unique,
            "offrank_request_tokens": offrank_tokens,
            "offrank_request_fraction": self._fraction(
                offrank_tokens,
                tokens,
            ),
            "per_requester_unique_offrank_item_rows": (
                offrank_unique
            ),
        }

    def summary(self) -> dict[str, object]:
        vector_bytes = HIDDEN_SIZE * EMBEDDING_DTYPE_BYTES
        offrank_tokens = int(self.rank_offrank_tokens.sum())
        unique = self._unique(self.seen)
        unique_offrank_per_requester = sum(
            self._unique(self.rank_offrank_seen[rank])
            for rank in range(self.world_size)
        )
        ranks = []
        for rank in range(self.world_size):
            rank_tokens = int(self.rank_tokens[rank])
            local_tokens = int(self.rank_local_tokens[rank])
            remote_tokens = int(self.rank_offrank_tokens[rank])
            ranks.append(
                {
                    "requester_rank": rank,
                    "request_tokens": rank_tokens,
                    "unique_item_rows": self._unique(
                        self.rank_seen[rank]
                    ),
                    "local_request_tokens": local_tokens,
                    "local_unique_item_rows": self._unique(
                        self.rank_local_seen[rank]
                    ),
                    "offrank_request_tokens": remote_tokens,
                    "offrank_unique_item_rows": self._unique(
                        self.rank_offrank_seen[rank]
                    ),
                    "offrank_request_fraction": self._fraction(
                        remote_tokens,
                        rank_tokens,
                    ),
                    "by_item_role": {
                        ROLE_NAMES[role]: self._role_summary(
                            role,
                            rank,
                        )
                        for role in range(2)
                    },
                }
            )
        shards = []
        unique_ids = np.flatnonzero(self.seen)
        for rank in range(self.world_size):
            addressable = addressable_item_rows(
                self.num_items,
                rank,
                self.world_size,
            )
            touched = int(
                np.count_nonzero(
                    np.remainder(unique_ids, self.world_size)
                    == rank
                )
            )
            shards.append(
                {
                    "embedding_owner_rank": rank,
                    "addressable_item_rows": addressable,
                    "physical_rows_including_padding": (
                        physical_embedding_rows(
                            self.num_items,
                            rank,
                            self.world_size,
                        )
                    ),
                    "unique_item_rows_touched": touched,
                    "coverage_of_addressable_item_rows": (
                        self._fraction(touched, addressable)
                    ),
                }
            )
        return {
            "request_tokens": self.request_tokens,
            "unique_item_rows": unique,
            "coverage_of_addressable_item_rows": self._fraction(
                unique,
                self.num_items,
            ),
            "by_item_role": {
                ROLE_NAMES[role]: self._role_summary(role)
                for role in range(2)
            },
            "requester_ranks": ranks,
            "embedding_shards": shards,
            "traffic_bytes": {
                "logical_id_payload_if_every_lookup_were_routed": (
                    self.request_tokens * ITEM_ID_DTYPE_BYTES
                ),
                "logical_return_vector_payload_fp32_if_every_lookup_were_routed": (
                    self.request_tokens * vector_bytes
                ),
                "offrank_id_request_payload": (
                    offrank_tokens * ITEM_ID_DTYPE_BYTES
                ),
                "offrank_return_vector_payload_fp32": (
                    offrank_tokens * vector_bytes
                ),
                "offrank_total_payload_without_collective_metadata": (
                    offrank_tokens
                    * (ITEM_ID_DTYPE_BYTES + vector_bytes)
                ),
                "perfect_wave_dedup_offrank_id_lower_bound": (
                    unique_offrank_per_requester
                    * ITEM_ID_DTYPE_BYTES
                ),
                "perfect_wave_dedup_offrank_return_vector_fp32_lower_bound": (
                    unique_offrank_per_requester * vector_bytes
                ),
            },
        }


def reconstruct_and_characterize(
    prepared_data: str | Path,
    records: list[dict[str, object]],
    layout: dict[str, int],
    owners: dict[int, int],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    plan, metadata = load_prepared_exposure_plan(
        prepared_data,
        max_seq_len=layout["history_tokens"],
    )
    num_items = int(plan.num_items)
    num_prediction_items = int(plan.num_prediction_items)
    context_hash_buckets = int(
        metadata.get("context_hash_buckets", 0)
    )
    if (
        num_prediction_items + context_hash_buckets != num_items
        or int(metadata.get("history_length", 0))
        != layout["history_tokens"]
        or int(metadata.get("slide", -1))
        != layout["target_filtered_start"]
    ):
        raise ValueError("prepared QK item roles or adjacent layout differ")
    with np.load(prepared_data, allow_pickle=False) as source:
        original_user_ids = source["original_user_ids"].astype(
            np.int64,
            copy=True,
        )
    plan.init_base()
    plan.ingest_day("window_0")
    methods = {
        "all_exact": RequestAccumulator(
            num_items,
            num_prediction_items,
            WORLD_SIZE,
        ),
        "mixed": RequestAccumulator(
            num_items,
            num_prediction_items,
            WORLD_SIZE,
        ),
    }
    verified = []
    exact_records = 0
    compiled_records = 0
    for record in records:
        record_id = int(record["record_id"])
        prepared_user_id = int(record["prepared_user_id"])
        original_user_id = int(record["original_user_id"])
        if (
            not 1 <= prepared_user_id <= len(original_user_ids)
            or int(original_user_ids[prepared_user_id - 1])
            != original_user_id
        ):
            raise ValueError(
                f"record {record_id} prepared/original user binding differs"
            )
        history = plan.user_histories.get(prepared_user_id)
        if history is None:
            raise ValueError(
                f"record {record_id} target history is absent"
            )
        start = layout["target_filtered_start"]
        stop = layout["target_filtered_stop"]
        actual_sha256 = history_identity_sha256(
            history,
            start,
            stop,
        )
        if actual_sha256 != record.get("target_history_sha256"):
            raise ValueError(
                f"record {record_id} target history hash differs"
            )
        target_ids = np.asarray(
            history["item_ids"][start:stop],
            dtype=np.int64,
        )
        if len(target_ids) != layout["history_tokens"]:
            raise ValueError(
                f"record {record_id} target extent length differs"
            )
        owner = owners[record_id]
        methods["all_exact"].add(owner, target_ids)
        action = str(record["requested_action"])
        if action == "exact":
            exact_records += 1
            methods["mixed"].add(owner, target_ids)
        else:
            compiled_records += 1
            methods["mixed"].add(
                owner,
                target_ids[
                    layout["delta_start"] : layout["final_tokens"]
                ],
            )
        verified.append(
            {
                "record_id": record_id,
                "target_history_sha256": actual_sha256,
            }
        )
    method_summaries = {
        name: accumulator.summary()
        for name, accumulator in methods.items()
    }
    return (
        method_summaries,
        {
            "records_verified": len(verified),
            "aggregate_target_history_sha256": (
                canonical_sha256(verified)
            ),
        },
        {
            "num_items": num_items,
            "num_prediction_items": num_prediction_items,
            "context_hash_buckets": context_hash_buckets,
            "num_embedding_rows_including_padding": num_items + 1,
            "hidden_size": HIDDEN_SIZE,
            "embedding_dtype": "float32",
            "embedding_dtype_bytes": EMBEDDING_DTYPE_BYTES,
            "item_id_dtype": "int64",
            "item_id_dtype_bytes": ITEM_ID_DTYPE_BYTES,
            "return_vector_bytes_per_request": (
                HIDDEN_SIZE * EMBEDDING_DTYPE_BYTES
            ),
            "full_embedding_table_bytes_fp32_including_padding": (
                (num_items + 1)
                * HIDDEN_SIZE
                * EMBEDDING_DTYPE_BYTES
            ),
            "exact_records": exact_records,
            "compiled_records": compiled_records,
        },
    )


def build_result(
    prepared_data: str | Path,
    action_snapshot: str | Path,
    expected_records: int,
    expected_history_tokens: int,
) -> dict[str, object]:
    if expected_records < 1 or expected_history_tokens < 1:
        raise ValueError("expected M1 dimensions must be positive")
    snapshot = load_json(action_snapshot)
    records, layout = validate_snapshot_integrity(
        snapshot,
        prepared_data,
    )
    if (
        len(records) != expected_records
        or layout["history_tokens"] != expected_history_tokens
    ):
        raise ValueError("M1 action snapshot dimensions differ")
    owners = strict_cow_lpt_owners(records, WORLD_SIZE)
    modulo_owners = {
        record_id: record_id % WORLD_SIZE
        for record_id in range(len(records))
    }
    if owners != modulo_owners:
        raise ValueError(
            "strict-COW LPT is not record_id modulo two for this snapshot"
        )
    method_summaries, history_binding, table = (
        reconstruct_and_characterize(
            prepared_data,
            records,
            layout,
            owners,
        )
    )
    all_exact = method_summaries["all_exact"]
    mixed = method_summaries["mixed"]
    old_bytes_per_record = (
        int(records[0]["old_tokens"])
        * NUM_LAYERS
        * KV_COMPONENTS
        * HIDDEN_SIZE
        * KV_DTYPE_BYTES
    )
    per_rank_records = [
        sum(owner == rank for owner in owners.values())
        for rank in range(WORLD_SIZE)
    ]
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "formal_design3": False,
        "artifact_role": (
            "read_only_m1_embedding_request_characterization"
        ),
        "bindings": {
            "prepared_data": {
                "path": str(prepared_data),
                "sha256": file_sha256(prepared_data),
                "snapshot_hash_verified": True,
            },
            "action_snapshot": {
                "path": str(action_snapshot),
                "sha256": file_sha256(action_snapshot),
                "owner_independent_plan_sha256": snapshot[
                    "owner_independent_plan_sha256"
                ],
                "content_hash_verified": True,
            },
            "target_histories": {
                **history_binding,
                "all_record_hashes_verified": True,
            },
        },
        "configuration": {
            "world_size": WORLD_SIZE,
            "record_owner_strategy": "strict_cow_lpt",
            "record_owner_equivalent_to": "record_id_modulo_2",
            "record_owner_map_sha256": owner_map_sha256(owners),
            "records": len(records),
            "history_tokens": layout["history_tokens"],
            "compiled_append_tokens": (
                layout["final_tokens"] - layout["delta_start"]
            ),
            "mixed_lookup_definition": (
                "exact records request full target histories; compiled "
                "records request only delta-plus-latest append tokens"
            ),
            "compiled_retained_prefix_embedding_requests": 0,
            "embedding": table,
        },
        "methods": method_summaries,
        "comparison": {
            "mixed_over_all_exact_request_tokens": (
                int(mixed["request_tokens"])
                / int(all_exact["request_tokens"])
            ),
            "mixed_over_all_exact_unique_item_rows": (
                int(mixed["unique_item_rows"])
                / int(all_exact["unique_item_rows"])
            ),
            "mixed_over_all_exact_offrank_return_vector_bytes_fp32": (
                int(
                    mixed["traffic_bytes"][
                        "offrank_return_vector_payload_fp32"
                    ]
                )
                / int(
                    all_exact["traffic_bytes"][
                        "offrank_return_vector_payload_fp32"
                    ]
                )
            ),
        },
        "timing_scope": {
            "this_artifact": (
                "static request characterization; no runtime is measured"
            ),
            "primary_runtime_timer": (
                "begins only after old source K/V is materialized"
            ),
            "old_source_materialization": {
                "classification": (
                    "setup_only_excluded_from_primary_timer"
                ),
                "cache_geometry": {
                    "layers": NUM_LAYERS,
                    "kv_components": KV_COMPONENTS,
                    "kv_width": HIDDEN_SIZE,
                    "dtype": "float16",
                    "dtype_bytes": KV_DTYPE_BYTES,
                },
                "records": len(records),
                "logical_kv_bytes_fp16": (
                    old_bytes_per_record * len(records)
                ),
                "per_rank": [
                    {
                        "rank": rank,
                        "records": per_rank_records[rank],
                        "logical_kv_bytes_fp16": (
                            old_bytes_per_record
                            * per_rank_records[rank]
                        ),
                    }
                    for rank in range(WORLD_SIZE)
                ],
            },
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_result(
        args.prepared_data,
        args.action_snapshot,
        args.expected_records,
        args.expected_history_tokens,
    )
    save_json(result, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": args.output,
                "prepared_data_sha256": result["bindings"][
                    "prepared_data"
                ]["sha256"],
                "action_snapshot_sha256": result["bindings"][
                    "action_snapshot"
                ]["sha256"],
                "records": result["configuration"]["records"],
                "all_exact_request_tokens": result["methods"][
                    "all_exact"
                ]["request_tokens"],
                "mixed_request_tokens": result["methods"]["mixed"][
                    "request_tokens"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "evokv_qb_large_multifield_foundation_audit_development_v0"
DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QB-video.csv"
DEFAULT_OUTPUT = Path("configs/evokv_foundation/qb_large_multifield_audit_development_v0.json")
FIELDS = {
    "item": ("item_id",),
    "user": ("user_id",),
    "user_item": ("user_id", "item_id"),
    "user_category": ("user_id", "video_category"),
    "item_demographic": ("item_id", "gender", "age"),
    "item_behavior": ("item_id", "behavior_signature"),
    "item_watch": ("item_id", "watch_bucket"),
    "user_watch": ("user_id", "watch_bucket"),
    "user_item_behavior": ("user_id", "item_id", "behavior_signature"),
}
PROFILES = {
    "mf5_e8192": {
        "fields": (
            "item",
            "user",
            "user_item",
            "user_category",
            "item_demographic",
        ),
        "embedding_width": 8192,
    },
    "mf8_e6656": {
        "fields": tuple(FIELDS)[:8],
        "embedding_width": 6656,
    },
    "mf9_e4096": {
        "fields": tuple(FIELDS),
        "embedding_width": 4096,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--window-width", type=int, default=8)
    parser.add_argument("--update-windows", type=int, default=4)
    parser.add_argument("--evaluation-windows", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--single-card-allocatable-bytes", type=int, default=47_699_722_240)
    parser.add_argument("--watch-bucket-maximum", type=int, default=15)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = (
        args.base_prefix,
        args.window_width,
        args.update_windows,
        args.evaluation_windows,
        args.hidden_size,
        args.layers,
        args.heads,
        args.head_dim,
        args.max_context,
        args.single_card_allocatable_bytes,
        args.watch_bucket_maximum,
    )
    if min(values) < 1 or args.heads * args.head_dim != args.hidden_size:
        raise ValueError("QB large foundation audit arguments are invalid")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path, member: str) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    return {
        "path": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "member": member,
        "member_bytes": info.file_size,
        "member_compressed_bytes": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
    }


def load_frame(path: Path, member: str) -> pd.DataFrame:
    columns = (
        "user_id",
        "item_id",
        "click",
        "follow",
        "like",
        "share",
        "video_category",
        "watching_times",
        "gender",
        "age",
    )
    dtypes = {
        "user_id": "int64",
        "item_id": "int64",
        "click": "int8",
        "follow": "int8",
        "like": "int8",
        "share": "int8",
        "video_category": "string",
        "watching_times": "int64",
        "gender": "int8",
        "age": "int8",
    }
    with zipfile.ZipFile(path) as archive, archive.open(member) as source:
        return pd.read_csv(source, usecols=columns, dtype=dtypes)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
    return {
        "count": len(values),
        "minimum": int(values.min()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
    }


def dense_core_parameters(
    layers: int,
    hidden_size: int,
    embedding_width: int,
    num_behaviors: int = 5,
) -> dict[str, int]:
    qkv = 3 * hidden_size * hidden_size
    out = hidden_size * hidden_size
    gate = hidden_size * hidden_size
    norms = hidden_size
    block = qkv + out + gate + norms
    behavior = (num_behaviors + 1) * hidden_size
    temporal = hidden_size * 32
    input_projection = hidden_size * hidden_size
    final_norm = hidden_size
    core = layers * block + behavior + temporal + input_projection + final_norm
    projection = hidden_size * embedding_width
    return {
        "core_excluding_owner_projection": core,
        "owner_projection": projection,
        "total": core + projection,
    }


def field_counts(
    base: pd.DataFrame,
    gradient_eligible: pd.DataFrame,
) -> dict[str, int]:
    positive = (
        base["click"].to_numpy(dtype=np.int8, copy=False)
        | base["follow"].to_numpy(dtype=np.int8, copy=False)
        | base["like"].to_numpy(dtype=np.int8, copy=False)
        | base["share"].to_numpy(dtype=np.int8, copy=False)
    ).astype(bool)
    item_sources = pd.concat(
        (
            gradient_eligible[["item_id"]],
            base.loc[positive, ["item_id"]],
        ),
        ignore_index=True,
    )
    result = {"item": int(len(item_sources.drop_duplicates()))}
    for name, columns in FIELDS.items():
        if name == "item":
            continue
        result[name] = int(len(gradient_eligible[list(columns)].drop_duplicates()))
    return result


def profile_record(
    name: str,
    profile: dict[str, object],
    counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    fields = tuple(str(value) for value in profile["fields"])
    embedding_width = int(profile["embedding_width"])
    semantic_rows = sum(counts[field] for field in fields)
    physical_rows = semantic_rows + 1
    embedding_bytes = physical_rows * embedding_width * 4
    active_embedding_bytes = semantic_rows * embedding_width * 4
    parameters = dense_core_parameters(
        args.layers,
        args.hidden_size,
        embedding_width,
    )
    dense_bytes = parameters["total"] * 4
    fixed_bytes = embedding_bytes + dense_bytes
    active_fixed_bytes = active_embedding_bytes + dense_bytes
    return {
        "name": name,
        "fields": list(fields),
        "feature_rows_per_token": len(fields),
        "semantic_rows": semantic_rows,
        "padding_rows": 1,
        "physical_rows": physical_rows,
        "embedding_width": embedding_width,
        "embedding_bytes_fp32": embedding_bytes,
        "embedding_gib_fp32": embedding_bytes / 2**30,
        "active_embedding_bytes_fp32": active_embedding_bytes,
        "active_embedding_gib_fp32": active_embedding_bytes / 2**30,
        "dense_parameters": parameters,
        "dense_bytes_fp32": dense_bytes,
        "global_fixed_bytes_fp32": fixed_bytes,
        "global_fixed_gib_fp32": fixed_bytes / 2**30,
        "global_active_fixed_bytes_fp32": active_fixed_bytes,
        "global_active_fixed_gib_fp32": active_fixed_bytes / 2**30,
        "single_card_allocatable_bytes": args.single_card_allocatable_bytes,
        "forced_sharding_gate_definition": (
            "optimizer-eligible semantic rows plus dense/projection bytes; padding excluded"
        ),
        "forced_sharding_gate_passed": (
            active_fixed_bytes > args.single_card_allocatable_bytes
        ),
        "forced_sharding_margin_bytes": (
            active_fixed_bytes - args.single_card_allocatable_bytes
        ),
        "physical_allocation_margin_bytes": (
            fixed_bytes - args.single_card_allocatable_bytes
        ),
        "two_rank_embedding_bytes_per_rank_upper_bound": (
            (physical_rows + 1) // 2 * embedding_width * 4
        ),
        "four_rank_embedding_bytes_per_rank_upper_bound": (
            (physical_rows + 3) // 4 * embedding_width * 4
        ),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    frame = load_frame(args.source, args.member)
    frame["raw_ordinal"] = frame.groupby("user_id", sort=False).cumcount()
    user_lengths = frame.groupby("user_id", sort=False).size()
    frame["user_length"] = frame["user_id"].map(user_lengths)
    frame["watch_bucket"] = np.minimum(
        frame["watching_times"].to_numpy(dtype=np.int64, copy=False),
        args.watch_bucket_maximum,
    )
    frame["behavior_signature"] = (
        frame["click"].to_numpy(dtype=np.int64, copy=False)
        | (frame["follow"].to_numpy(dtype=np.int64, copy=False) << 1)
        | (frame["like"].to_numpy(dtype=np.int64, copy=False) << 2)
        | (frame["share"].to_numpy(dtype=np.int64, copy=False) << 3)
    )
    base = frame[frame["raw_ordinal"] < args.base_prefix].copy()
    gradient_eligible = base[
        base["raw_ordinal"] + 1 < np.minimum(base["user_length"], args.base_prefix)
    ].copy()
    counts = field_counts(base, gradient_eligible)
    required_horizon = args.base_prefix + args.window_width * (
        args.update_windows + args.evaluation_windows
    )
    lengths = user_lengths.to_numpy(dtype=np.int64, copy=False)
    kv_bytes_per_token = 2 * args.layers * args.hidden_size * 2
    capped_tokens = int(np.minimum(lengths, args.max_context).sum())
    profiles = {
        name: profile_record(name, profile, counts, args) for name, profile in PROFILES.items()
    }
    result = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "status": "pass"
        if all(value["forced_sharding_gate_passed"] for value in profiles.values())
        else "fail",
        "source": source_identity(args.source, args.member),
        "data": {
            "rows": len(frame),
            "users": len(user_lengths),
            "raw_items": int(frame["item_id"].nunique()),
            "base_prefix": args.base_prefix,
            "base_rows": len(base),
            "gradient_eligible_base_rows": len(gradient_eligible),
            "user_length": distribution(lengths),
            "required_horizon": required_horizon,
            "users_reaching_required_horizon": int((lengths >= required_horizon).sum()),
            "users_reaching_104": int((lengths >= 104).sum()),
            "users_reaching_112": int((lengths >= 112).sum()),
        },
        "feature_catalog": {
            "fit_boundary": "base period only",
            "direct_row_rule": (
                "compact rows for feature values appearing in a base input position with a "
                "following base target; item rows also include engaged base targets"
            ),
            "unseen_rule": "field-local deterministic hashing into existing direct rows",
            "fields": {
                name: {
                    "columns": list(FIELDS[name]),
                    "optimizer_eligible_semantic_rows": count,
                }
                for name, count in counts.items()
            },
        },
        "model": {
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "maximum_context": args.max_context,
            "feature_composition": "owner-projected vectors summed with inverse-sqrt-field scaling",
        },
        "profiles": profiles,
        "kv_capacity": {
            "fp16_bytes_per_valid_token": kv_bytes_per_token,
            "all_users_capped_valid_tokens": capped_tokens,
            "all_users_single_version_valid_bytes": capped_tokens * kv_bytes_per_token,
            "all_users_single_version_valid_gib": capped_tokens * kv_bytes_per_token / 2**30,
            "larger_points": (
                "use multiple real prefix records per sufficiently long user before any "
                "systems-only record replication"
            ),
        },
        "screen": {
            "sequential_candidates": [
                {"devices": [0, 1], "profile": "mf9_e4096", "update": "lr15_epoch1"},
                {"devices": [0, 1], "profile": "mf9_e4096", "update": "lr30_epoch1"},
                {"devices": [0, 1], "profile": "mf9_e4096", "update": "lr15_epoch3"},
                {"devices": [0, 1], "profile": "mf9_e4096", "update": "lr30_epoch3"},
            ],
            "common_theta0_built_once": True,
            "shared_core_and_stream": True,
            "comparison": (
                "one fixed active-row model and stream with a 2x2 learning-rate/epoch screen"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    atomic_json(args.output, result)
    result["output"] = str(args.output)
    result["output_sha256"] = file_sha256(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

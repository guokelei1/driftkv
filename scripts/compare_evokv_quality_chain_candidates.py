from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="label=result-root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"candidate comparison differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def cluster_interval(
    record_ids: np.ndarray,
    gains: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    unique, inverse = np.unique(record_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, gains)
    rng = np.random.default_rng(seed)
    values = []
    remaining = replicates
    while remaining:
        current = min(remaining, 256)
        selected = rng.integers(0, len(unique), size=(current, len(unique)))
        values.append(
            sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
        )
        remaining -= current
    samples = np.concatenate(values)
    return {
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def load_candidate(
    label: str,
    root: Path,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    summary_path = root / "summary.json"
    training_path = root / "training.json"
    if not summary_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(f"candidate is incomplete: {root}")
    summary = json.loads(summary_path.read_text())
    training = json.loads(training_path.read_text())
    if (
        summary.get("status") != "complete"
        or len(summary.get("edges", [])) != 3
        or training.get("status") != "complete"
        or len(training.get("updates", [])) != 4
        or training.get("execution", {}).get("world_size") != 2
    ):
        raise ValueError(f"candidate contract differs: {root}")
    rows = []
    pair_keys = {}
    semantic_pair_bindings = {}
    for edge_index, edge in enumerate(summary["edges"]):
        edge_name = str(edge["edge"])
        cell_path = root / "cells" / f"{edge_name}.json"
        cell = json.loads(cell_path.read_text())
        quality = cell["quality_by_negative_count"]["999"]
        paired = quality["paired_target_contributions"]
        record_ids = np.asarray(paired["record_ids"], dtype=np.int64)
        reuse = np.asarray(
            paired["all_reuse"]["sampled_cross_entropy"],
            dtype=np.float64,
        )
        exact = np.asarray(
            paired["all_exact"]["sampled_cross_entropy"],
            dtype=np.float64,
        )
        if (
            len(record_ids) == 0
            or len(record_ids) != len(reuse)
            or len(record_ids) != len(exact)
            or not np.all(np.isfinite(reuse))
            or not np.all(np.isfinite(exact))
        ):
            raise ValueError(f"paired contributions differ: {cell_path}")
        gain = reuse - exact
        interval = cluster_interval(
            record_ids,
            gain,
            replicates,
            seed ^ (edge_index + 1) ^ int(hashlib.sha256(label.encode()).hexdigest()[:8], 16),
        )
        observed = float(np.mean(gain))
        declared = float(
            edge["metrics"]["sampled_cross_entropy"][
                "exact_over_reuse_gain"
            ]
        )
        if not np.isclose(observed, declared, rtol=1e-8, atol=1e-7):
            raise ValueError(f"candidate CE aggregate differs: {cell_path}")
        pair_keys[edge_name] = paired["pair_key_sha256"]
        candidate_hashes = cell["role"][
            "candidate_sha256_per_rank_by_negative_count"
        ]["999"]
        role_audit = cell["role"]["audit"]
        suffix_offsets = np.asarray(
            paired["suffix_offsets"], dtype="<i8"
        )
        semantic_pair_bindings[edge_name] = {
            "candidate_sha256_per_rank": candidate_hashes,
            "history_end": int(role_audit["history_end"]),
            "positive_targets": len(record_ids),
            "suffix_offsets_sha256": hashlib.sha256(
                suffix_offsets.tobytes()
            ).hexdigest(),
            "update_end": int(role_audit["update_end"]),
            "update_width": int(role_audit["update_width"]),
        }
        rows.append(
            {
                "edge": edge_name,
                "positive_targets": len(record_ids),
                "streaming_update_ce_utility": float(
                    edge["metrics"]["sampled_cross_entropy"][
                        "streaming_update_utility"
                    ]
                ),
                "exact_over_reuse_ce_gain": observed,
                "exact_over_reuse_ce_gain_record_cluster_95": interval,
                "exact_over_reuse_ndcg_gain": float(
                    edge["metrics"]["ndcg_at_10"][
                        "exact_over_reuse_gain"
                    ]
                ),
                "cache_relative_error": float(
                    edge["cache_relative_error"]
                ),
                "reuse_top10_overlap_with_exact": float(
                    edge["reuse_top10_overlap_with_exact"]
                ),
            }
        )
    ce_gains = np.asarray(
        [row["exact_over_reuse_ce_gain"] for row in rows],
        dtype=np.float64,
    )
    stream_utilities = np.asarray(
        [row["streaming_update_ce_utility"] for row in rows],
        dtype=np.float64,
    )
    return {
        "label": label,
        "root": str(root),
        "summary_sha256": file_sha256(summary_path),
        "training_sha256": file_sha256(training_path),
        "stack_identity": training["stack_identity"],
        "model": training["model"],
        "training_users": training["corpus_audit"]["split_users"]["train"][
            "users"
        ],
        "qualification_users": training["corpus_audit"]["split_users"][
            "quality"
        ]["users"],
        "qualification_user_ids_sha256": training["corpus_audit"][
            "split_users"
        ]["quality"]["user_ids_sha256"],
        "training_wall_seconds": float(
            training["execution"]["total_wall_seconds"]
        ),
        "pair_keys": pair_keys,
        "semantic_pair_bindings": semantic_pair_bindings,
        "edges": rows,
        "stability": {
            "all_edges_exact_ce_better_than_reuse": bool(
                np.all(ce_gains > 0)
            ),
            "positive_streaming_update_edges": int(
                np.count_nonzero(stream_utilities > 0)
            ),
            "minimum_exact_over_reuse_ce_gain": float(np.min(ce_gains)),
            "median_exact_over_reuse_ce_gain": float(np.median(ce_gains)),
            "maximum_exact_over_reuse_ce_gain": float(np.max(ce_gains)),
            "median_streaming_update_ce_utility": float(
                np.median(stream_utilities)
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap replicate count differs")
    specifications = []
    for value in args.candidate:
        label, separator, root = value.partition("=")
        if not separator or not label or not root:
            raise ValueError("candidate must be label=result-root")
        specifications.append((label, Path(root)))
    if len(specifications) < 2 or len({label for label, _ in specifications}) != len(
        specifications
    ):
        raise ValueError("candidate set differs")
    candidates = [
        load_candidate(
            label,
            root,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        for label, root in specifications
    ]
    model_bindings = {
        json.dumps(candidate["model"], sort_keys=True)
        for candidate in candidates
    }
    qualification_bindings = {
        json.dumps(
            {
                "qualification_user_ids_sha256": candidate[
                    "qualification_user_ids_sha256"
                ],
                "semantic_pair_bindings": candidate[
                    "semantic_pair_bindings"
                ],
            },
            sort_keys=True,
        )
        for candidate in candidates
    }
    if len(model_bindings) != 1 or len(qualification_bindings) != 1:
        raise ValueError("quality candidates are not directly comparable")
    admissible = [
        candidate
        for candidate in candidates
        if candidate["stability"]["all_edges_exact_ce_better_than_reuse"]
        and candidate["stability"]["positive_streaming_update_edges"] >= 2
        and candidate["stability"]["median_streaming_update_ce_utility"] > 0
    ]
    selected = None
    if admissible:
        selected = max(
            admissible,
            key=lambda candidate: (
                candidate["stability"][
                    "minimum_exact_over_reuse_ce_gain"
                ],
                candidate["stability"][
                    "median_exact_over_reuse_ce_gain"
                ],
            ),
        )["label"]
    result = {
        "schema": "evokv_quality_chain_candidate_comparison_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "selection_role": "development qualification",
        "formal_promotion_requires_new_seed_or_frozen_repeat": True,
        "candidates": candidates,
        "development_recommendation": selected,
        "recommendation_policy": (
            "require exact CE to beat reuse on all three edges and useful "
            "streaming updates on at least two edges; then maximize the "
            "worst-edge CE opportunity before the median opportunity"
        ),
    }
    atomic_text(
        args.output,
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "candidate\ttraining_users\tedge\tpositive_targets\tstreaming_update_ce_utility\texact_over_reuse_ce_gain\tce_gain_lower_95\tce_gain_upper_95\tcache_relative_error\treuse_top10_overlap_with_exact"
    ]
    for candidate in candidates:
        for row in candidate["edges"]:
            interval = row[
                "exact_over_reuse_ce_gain_record_cluster_95"
            ]
            lines.append(
                "\t".join(
                    str(value)
                    for value in (
                        candidate["label"],
                        candidate["training_users"],
                        row["edge"],
                        row["positive_targets"],
                        row["streaming_update_ce_utility"],
                        row["exact_over_reuse_ce_gain"],
                        interval["lower_95"],
                        interval["upper_95"],
                        row["cache_relative_error"],
                        row["reuse_top10_overlap_with_exact"],
                    )
                )
            )
    atomic_text(args.tsv, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "candidates": [candidate["label"] for candidate in candidates],
                "development_recommendation": selected,
                "output": str(args.output),
                "status": "complete",
                "tsv": str(args.tsv),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

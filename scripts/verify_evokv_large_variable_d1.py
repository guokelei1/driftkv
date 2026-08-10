from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from hstu_kvcache.migration.variable_inference import (
    array_sha256,
    file_sha256,
    load_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/evokv_d1/development/large_variable_score_sweep_two_gpu_v0.json"
        ),
    )
    parser.add_argument("--require-corpora", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(path)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def score_frontier(schedule: np.ndarray, thresholds: list[int]) -> dict[str, object]:
    records, boundaries = schedule.shape
    edge_count = boundaries - 1
    all_exact_tokens = int(schedule[:, :edge_count].sum())
    minimum_prefix_tokens = int(schedule[:, :edge_count].min())
    result = {}
    for threshold in thresholds:
        scores = np.zeros(records, dtype=np.int64)
        exact_counts = np.zeros(records, dtype=np.int64)
        exact_tokens = 0
        edges = []
        for edge in range(edge_count):
            prefix = schedule[:, edge].astype(np.int64)
            candidates = scores + prefix
            exact = candidates >= threshold
            edge_exact_tokens = int(prefix[exact].sum())
            exact_tokens += edge_exact_tokens
            exact_counts += exact.astype(np.int64)
            scores = np.where(exact, 0, candidates)
            edges.append(
                {
                    "edge_ordinal": edge,
                    "exact_records": int(exact.sum()),
                    "exact_record_fraction": float(exact.mean()),
                    "exact_valid_tokens": edge_exact_tokens,
                    "exact_valid_token_fraction": edge_exact_tokens
                    / int(prefix.sum()),
                }
            )
        result[str(threshold)] = {
            "cumulative_exact_valid_tokens": exact_tokens,
            "cumulative_exact_valid_token_fraction": exact_tokens
            / all_exact_tokens,
            "cumulative_exact_action_fraction": float(exact_counts.mean())
            / edge_count,
            "never_exact_records": int(np.count_nonzero(exact_counts == 0)),
            "never_exact_fraction": float(np.mean(exact_counts == 0)),
            "maximum_exact_count": int(exact_counts.max()),
            "ending_score_maximum": int(scores.max()),
            "debt_strictly_below_threshold": bool(scores.max() < threshold),
            "forced_exact_within_updates_at_minimum_prefix": int(
                np.ceil(threshold / minimum_prefix_tokens)
            ),
            "minimum_prefix_tokens": minimum_prefix_tokens,
            "edges": edges,
        }
    return result


def validate_corpus(
    dataset: str,
    selected: dict[str, object],
    thresholds: list[int],
    minimum_initial_tokens: int,
    selection_salt: str,
) -> dict[str, object]:
    path = Path(selected["corpus"])
    summary_path = Path(selected["corpus_summary"])
    corpus = load_corpus(path)
    summary = json.loads(summary_path.read_text())
    expected_roles = {
        "fit": int(selected.get("fit_records", 0)),
        "probe": int(selected.get("probe_records", 0)),
        "qualification": int(selected["qualification_records"]),
    }
    descriptor = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": corpus.file_sha256,
        "content_sha256": corpus.content_sha256,
    }
    qualification = corpus.role_records("qualification")
    lengths = corpus.arrays["record_valid_lengths"][qualification]
    roles = (
        ("qualification",)
        if dataset == "qk"
        else ("fit", "probe", "qualification")
    )
    observed_record_bindings = {
        f"{role}_{kind}_ids_sha256": array_sha256(
            corpus.arrays[name][corpus.role_records(role)]
        )
        for role in roles
        for kind, name in (
            ("source", "record_source_ids"),
            ("user", "record_user_ids"),
        )
    }
    if (
        corpus.dataset != dataset
        or corpus.edge_count != int(selected["edge_count"])
        or corpus.feature_fields != int(selected["feature_fields"])
        or corpus.metadata.get("roles") != expected_roles
        or corpus.metadata.get("minimum_initial_tokens")
        != minimum_initial_tokens
        or corpus.metadata.get("selection_salt") != selection_salt
        or observed_record_bindings != selected["record_bindings"]
        or corpus.metadata.get("record_bindings")
        != observed_record_bindings
        or corpus.metadata.get("quality_action_independence") is not True
        or corpus.metadata.get("positive_audit", {}).get(
            "all_edges_have_positive_targets"
        )
        is not True
        or dataset == "qk"
        and int(corpus.metadata.get("item_alignment_events_verified", -1))
        != int(corpus.arrays["record_offsets"][-1])
        or dataset == "qb"
        and int(corpus.metadata.get("frozen_overlap_events_verified", 0)) < 1
        or len(qualification) != int(selected["qualification_records"])
        or len(set(lengths.tolist())) < 2
        or summary.get("schema")
        != "evokv_large_variable_inference_build_v0"
        or summary.get("status") != "pass"
        or summary.get("dataset") != dataset
        or summary.get("corpus") != descriptor
        or summary.get("metadata") != corpus.metadata
    ):
        raise ValueError(f"large variable D1 {dataset} corpus differs")
    schedule = corpus.arrays["edge_prefix_lengths"][qualification]
    return {
        "descriptor": descriptor,
        "records": len(qualification),
        "valid_length": {
            "minimum": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "maximum": int(lengths.max()),
            "distinct_lengths": len(set(lengths.tolist())),
        },
        "score_frontier_before_quality": score_frontier(schedule, thresholds),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if (
        config.get("schema")
        != "evokv_large_variable_d1_score_sweep_two_gpu_v0"
        or config.get("status") != "ready_for_user_execution"
        or config.get("scientific_result") is not False
        or config.get("formal_result") is not False
        or int(config.get("world_size", -1)) != 2
        or config.get("large_model_only") is not True
        or config.get("serving_model_invariant", {}).get(
            "concurrent_recommendation_models"
        )
        != 1
        or config.get("serving_model_invariant", {}).get(
            "multi_version_serving"
        )
        is not False
        or config.get("numeric_precision")
        != {
            "evaluation_float32_matmul_precision": "high",
            "ridge_float32_matmul_precision": "highest",
            "nvidia_tf32_override": "unset",
            "ridge_gram_accumulation": "ieee_fp32",
        }
    ):
        raise ValueError("large variable D1 config differs")
    thresholds = [int(value) for value in config["score_thresholds"]]
    if thresholds != sorted(set(thresholds)) or min(thresholds) < 1:
        raise ValueError("large variable D1 thresholds differ")
    registry_binding = config["bindings"]["checkpoint_registry"]
    registry_path = Path(registry_binding["path"])
    if (
        not registry_path.is_file()
        or file_sha256(registry_path) != registry_binding["sha256"]
    ):
        raise ValueError("large variable D1 registry binding differs")
    registry = json.loads(registry_path.read_text())
    datasets = {}
    checkpoint_count = 0
    for dataset in ("qk", "qb"):
        selected = config["datasets"][dataset]
        versions = [int(value) for value in selected["versions"]]
        chain_entries = {
            int(value["version"]): value
            for value in registry["selected_chains"][dataset][
                "checkpoint_manifests"
            ]
        }
        if (
            len(versions) != int(selected["edge_count"]) + 1
            or dataset == "qk"
            and versions != [1, 2, 3, 4]
            or dataset == "qb"
            and versions != [1, 2, 3]
        ):
            raise ValueError(f"large variable D1 {dataset} versions differ")
        checkpoint_bindings = []
        for version in versions:
            manifest = Path(selected["checkpoint_root"]) / f"theta_{version}" / "manifest.json"
            entry = chain_entries[version]
            if (
                str(manifest) != entry["manifest_path"]
                or not manifest.is_file()
                or file_sha256(manifest) != entry["manifest_sha256"]
            ):
                raise ValueError(f"large variable D1 checkpoint differs: {manifest}")
            checkpoint_bindings.append(
                {
                    "version": version,
                    "manifest_path": str(manifest),
                    "manifest_sha256": entry["manifest_sha256"],
                }
            )
        checkpoint_count += len(checkpoint_bindings)
        report = {"checkpoint_bindings": checkpoint_bindings}
        if dataset == "qk":
            source_summary = selected["program_source_summary"]
            source_summary_path = Path(source_summary["path"])
            if (
                not source_summary_path.is_file()
                or source_summary_path.stat().st_size
                != int(source_summary["bytes"])
                or file_sha256(source_summary_path) != source_summary["sha256"]
            ):
                raise ValueError(
                    f"large variable D1 QK program source differs: {source_summary_path}"
                )
            programs = {}
            for edge, descriptor in selected["programs"].items():
                path = Path(descriptor["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(descriptor["bytes"])
                    or file_sha256(path) != descriptor["sha256"]
                ):
                    raise ValueError(f"large variable D1 QK program differs: {path}")
                programs[edge] = descriptor
            report["programs"] = programs
            report["program_source_summary"] = source_summary
        if args.require_corpora:
            report["corpus"] = validate_corpus(
                dataset,
                selected,
                thresholds,
                int(config["minimum_initial_tokens"]),
                str(config["record_selection_salt"]),
            )
        datasets[dataset] = report
    if checkpoint_count != 7:
        raise ValueError("large variable D1 checkpoint count differs")
    result = {
        "schema": "evokv_large_variable_d1_input_verification_v0",
        "status": "pass",
        "scientific_result": False,
        "formal_result": False,
        "large_model_checkpoints": checkpoint_count,
        "single_current_serving_model": True,
        "corpora_required": args.require_corpora,
        "score_thresholds": thresholds,
        "datasets": datasets,
        "bindings": {
            "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
            "checkpoint_registry": registry_binding,
            "source_code": {
                "path": str(Path(__file__)),
                "sha256": file_sha256(Path(__file__)),
            },
        },
    }
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

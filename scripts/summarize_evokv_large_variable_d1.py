from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path

from hstu_kvcache.migration.variable_inference import file_sha256, load_corpus

CONFIG_SCHEMA = "evokv_large_variable_d1_score_sweep_two_gpu_v0"
RESULT_SCHEMA = "evokv_large_variable_d1_score_sweep_result_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qk-result", type=Path, required=True)
    parser.add_argument("--qb-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--return-manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(path)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def expected_methods(config: dict[str, object]) -> list[str]:
    return [
        "reuse",
        "compiled",
        *[f"score_{int(value)}" for value in config["score_thresholds"]],
        "exact",
    ]


def validate_result(
    path: Path,
    dataset: str,
    config_path: Path,
    config: dict[str, object],
) -> dict[str, object]:
    result = load_json(path)
    selected = config["datasets"][dataset]
    methods = expected_methods(config)
    versions = [int(value) for value in selected["versions"]]
    thresholds = [int(value) for value in config["score_thresholds"]]
    roles = {
        "fit_records": int(selected.get("fit_records", 0)),
        "probe_records": int(selected.get("probe_records", 0)),
        "qualification_records": int(selected["qualification_records"]),
        "pairwise_disjoint": True,
    }
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "pass"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result.get("dataset") != dataset
        or result.get("large_model_only") is not True
        or result.get("full_kv_payloads_persisted") != 0
        or result.get("serving_model_invariant")
        != config["serving_model_invariant"]
        or result.get("versions") != versions
        or result.get("score_thresholds") != thresholds
        or result.get("selection_reads_quality") is not False
        or result.get("roles") != roles
        or result.get("execution", {}).get("numeric_precision")
        != config["numeric_precision"]
        or result.get("execution", {}).get(
            "observed_float32_matmul_precision"
        )
        != config["numeric_precision"][
            "evaluation_float32_matmul_precision"
        ]
        or result.get("execution", {}).get("observed_nvidia_tf32_override")
        != config["numeric_precision"]["nvidia_tf32_override"]
        or result.get("bindings", {}).get("config", {}).get("sha256")
        != file_sha256(config_path)
    ):
        raise ValueError(f"large variable D1 result header differs: {path}")
    spec = result["spec"]
    if (
        int(spec["embedding_width"]) != 4096
        or int(spec["hidden_size"]) != 1536
        or int(spec["num_layers"]) != 24
        or int(spec["num_heads"]) != 24
        or int(spec["head_dim"]) != 64
        or int(spec["max_seq_len"]) != 512
    ):
        raise ValueError(f"large variable D1 model geometry differs: {path}")
    corpus_path = Path(result["corpus"]["path"])
    corpus = load_corpus(corpus_path)
    if (
        corpus.dataset != dataset
        or corpus.file_sha256 != result["corpus"]["sha256"]
        or corpus.content_sha256 != result["corpus"]["content_sha256"]
        or corpus.feature_fields != int(selected["feature_fields"])
        or len(corpus.role_records("qualification"))
        != int(selected["qualification_records"])
        or len(corpus.role_records("fit")) != int(selected.get("fit_records", 0))
        or len(corpus.role_records("probe"))
        != int(selected.get("probe_records", 0))
        or len(set(corpus.arrays["record_valid_lengths"].tolist())) < 2
    ):
        raise ValueError(f"large variable D1 corpus binding differs: {path}")
    bindings = result["checkpoint_bindings"]
    if [int(value["version"]) for value in bindings] != versions:
        raise ValueError(f"large variable D1 checkpoint sequence differs: {path}")
    for binding in bindings:
        manifest = Path(binding["manifest_path"])
        if not manifest.is_file() or file_sha256(manifest) != binding["manifest_sha256"]:
            raise ValueError(f"large variable D1 checkpoint binding differs: {manifest}")
    edges = result["edges"]
    if len(edges) != len(versions) - 1:
        raise ValueError(f"large variable D1 edge count differs: {path}")
    previous_exact_fractions = {method: [] for method in methods}
    for edge_index, edge in enumerate(edges):
        if (
            int(edge["edge_ordinal"]) != edge_index
            or int(edge["source_version"]) != versions[edge_index]
            or int(edge["target_version"]) != versions[edge_index + 1]
            or int(edge["records"]) != int(selected["qualification_records"])
            or int(edge["prefix_length"]["minimum"])
            >= int(edge["prefix_length"]["maximum"])
            or int(edge["append_length"]["minimum"])
            >= int(edge["append_length"]["maximum"])
        ):
            raise ValueError(f"large variable D1 variable edge differs: {path}")
        recommendation = edge["metrics"]["recommendation"]
        fidelity = edge["metrics"]["cache_fidelity"]
        selection = edge["selection"]
        if set(recommendation) != set(methods) or set(fidelity) != set(methods):
            raise ValueError(f"large variable D1 method matrix differs: {path}")
        positive_counts = {
            int(recommendation[method]["positive_targets"]) for method in methods
        }
        if len(positive_counts) != 1 or next(iter(positive_counts)) < 1:
            raise ValueError(f"large variable D1 target counts differ: {path}")
        total_tokens = int(edge["valid_prefix_tokens"])
        for method in methods:
            fraction = float(selection[method]["exact_valid_token_fraction"])
            previous_exact_fractions[method].append(fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"large variable D1 edge fraction differs: {path}")
            if method in {"reuse", "compiled"} and fraction != 0.0:
                raise ValueError(f"large variable D1 non-exact route differs: {path}")
            if method == "exact" and fraction != 1.0:
                raise ValueError(f"large variable D1 exact route differs: {path}")
            if method.startswith("score_"):
                scheduled = int(selection[method]["scheduled_exact_valid_tokens"])
                if not close(fraction, scheduled / total_tokens):
                    raise ValueError(f"large variable D1 score accounting differs: {path}")
        program = edge["program"]
        program_path = Path(program["path"])
        if (
            not program_path.is_file()
            or program_path.stat().st_size != int(program["bytes"])
            or file_sha256(program_path) != program["sha256"]
        ):
            raise ValueError(f"large variable D1 program binding differs: {program_path}")
    cumulative = result["cumulative"]
    all_exact_tokens = sum(int(edge["valid_prefix_tokens"]) for edge in edges)
    for method in methods:
        selection = cumulative["selection"][method]
        tokens = int(selection["exact_valid_tokens"])
        if (
            int(selection["all_exact_valid_tokens"]) != all_exact_tokens
            or not close(float(selection["exact_valid_token_fraction"]), tokens / all_exact_tokens)
            or selection["edge_exact_valid_token_fractions"]
            != previous_exact_fractions[method]
        ):
            raise ValueError(f"large variable D1 cumulative accounting differs: {path}")
    fractions = [
        float(cumulative["selection"][f"score_{threshold}"]["exact_valid_token_fraction"])
        for threshold in thresholds
    ]
    if any(left < right for left, right in zip(fractions, fractions[1:], strict=False)):
        raise ValueError(f"large variable D1 threshold frontier is not monotone: {path}")
    for threshold in thresholds:
        fairness = cumulative["fairness"][f"score_{threshold}"]
        if (
            int(fairness["records"]) != int(selected["qualification_records"])
            or int(fairness["ending_score"]["maximum"]) >= threshold
            or fairness.get("debt_strictly_below_threshold") is not True
            or int(
                fairness["forced_exact_within_updates_at_minimum_prefix"]
            )
            != math.ceil(
                threshold / int(fairness["minimum_prefix_tokens"])
            )
        ):
            raise ValueError(f"large variable D1 debt bound differs: {path}")
    return result


def threshold_value(method: str) -> int | None:
    return int(method.removeprefix("score_")) if method.startswith("score_") else None


def cumulative_cache_metrics(
    result: dict[str, object],
    methods: list[str],
) -> dict[str, dict[str, float | None]]:
    total_records = sum(int(edge["records"]) for edge in result["edges"])
    means = {
        method: sum(
            int(edge["records"])
            * float(edge["metrics"]["cache_fidelity"][method]["mean"])
            for edge in result["edges"]
        )
        / total_records
        for method in methods
    }
    reuse = means["reuse"]
    return {
        method: {
            "mean": means[method],
            "mean_error_recovery": None if reuse <= 0 else 1.0 - means[method] / reuse,
        }
        for method in methods
    }


def result_rows(
    dataset: str,
    result: dict[str, object],
    methods: list[str],
) -> list[dict[str, object]]:
    rows = []
    cache_cumulative = cumulative_cache_metrics(result, methods)
    fairness = result["cumulative"]["fairness"]
    edge_count = len(result["edges"])
    qualification_records = int(result["roles"]["qualification_records"])
    for method in methods:
        selection = result["cumulative"]["selection"][method]
        recommendation = result["cumulative"]["recommendation"][method]
        exact_actions = (
            qualification_records * edge_count
            if method == "exact"
            else 0
            if method in {"reuse", "compiled"}
            else sum(
                int(edge["selection"][method]["scheduled_exact_records"])
                for edge in result["edges"]
            )
        )
        method_fairness = fairness.get(method, {})
        rows.append(
            {
                "dataset": dataset,
                "scope": "cumulative",
                "edge": "all",
                "method": method,
                "score_threshold": threshold_value(method),
                "exact_action_fraction": exact_actions
                / (qualification_records * edge_count),
                "exact_valid_token_fraction": float(selection["exact_valid_token_fraction"]),
                "sampled_cross_entropy": float(recommendation["sampled_cross_entropy"]),
                "ce_gap_recovery": recommendation["ce_gap_recovery"],
                "mean_cache_relative_error": cache_cumulative[method]["mean"],
                "cache_error_recovery": cache_cumulative[method]["mean_error_recovery"],
                "never_exact_fraction": method_fairness.get("never_exact_fraction"),
                "maximum_exact_count": method_fairness.get("maximum_exact_count"),
            }
        )
    for edge in result["edges"]:
        for method in methods:
            selection = edge["selection"][method]
            recommendation = edge["metrics"]["recommendation"][method]
            fidelity = edge["metrics"]["cache_fidelity"][method]
            exact_records = (
                int(edge["records"])
                if method == "exact"
                else 0
                if method in {"reuse", "compiled"}
                else int(selection["scheduled_exact_records"])
            )
            rows.append(
                {
                    "dataset": dataset,
                    "scope": "edge",
                    "edge": f"theta{edge['source_version']}->theta{edge['target_version']}",
                    "method": method,
                    "score_threshold": threshold_value(method),
                    "exact_action_fraction": exact_records / int(edge["records"]),
                    "exact_valid_token_fraction": float(selection["exact_valid_token_fraction"]),
                    "sampled_cross_entropy": float(recommendation["sampled_cross_entropy"]),
                    "ce_gap_recovery": recommendation["ce_gap_recovery"],
                    "mean_cache_relative_error": float(fidelity["mean"]),
                    "cache_error_recovery": fidelity["mean_error_recovery"],
                    "never_exact_fraction": None,
                    "maximum_exact_count": None,
                }
            )
    return rows


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("status") != "ready_for_user_execution"
        or config.get("scientific_result") is not False
        or config.get("formal_result") is not False
        or int(config.get("world_size", -1)) != 2
    ):
        raise ValueError("large variable D1 config differs")
    registry = config["bindings"]["checkpoint_registry"]
    registry_path = Path(registry["path"])
    if not registry_path.is_file() or file_sha256(registry_path) != registry["sha256"]:
        raise ValueError("large variable D1 registry binding differs")
    results = {
        "qk": validate_result(args.qk_result, "qk", args.config, config),
        "qb": validate_result(args.qb_result, "qb", args.config, config),
    }
    methods = expected_methods(config)
    rows = [
        row
        for dataset, result in results.items()
        for row in result_rows(dataset, result, methods)
    ]
    fields = list(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    summary = {
        "schema": "evokv_large_variable_d1_score_sweep_summary_v0",
        "status": "complete_development_measurement",
        "scientific_result": False,
        "formal_result": False,
        "large_model_checkpoints": 7,
        "datasets": ["qk", "qb"],
        "single_current_serving_model": True,
        "variable_inference_baseline_fixed": True,
        "full_kv_payloads_persisted": 0,
        "score_thresholds": config["score_thresholds"],
        "methods": methods,
        "cumulative_rows": [row for row in rows if row["scope"] == "cumulative"],
        "interpretation_boundary": {
            "exact_work_axis": "valid prefix tokens selected for exact recomputation relative to all-exact",
            "gpu_speedup_claimed": False,
            "quality_axis": "paired sampled cross-entropy recovery from recursive reuse toward recursive all-exact on identical variable histories",
            "next_decision_requires_result_interpretation": True,
        },
        "inputs": {
            "config": artifact(args.config),
            "checkpoint_registry": artifact(registry_path),
            "qk_result": artifact(args.qk_result),
            "qb_result": artifact(args.qb_result),
        },
    }
    atomic_text(args.tsv, stream.getvalue())
    atomic_text(args.output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return_manifest = {
        "schema": "evokv_large_variable_d1_return_manifest_v0",
        "status": "complete",
        "return_first": [
            artifact(args.output),
            artifact(args.tsv),
            artifact(args.qk_result),
            artifact(args.qb_result),
        ],
        "do_not_return": [
            "full checkpoint payloads",
            "generated QB program payloads unless later debugging requires them",
            "any in-memory K/V state",
        ],
        "full_kv_payloads_persisted": 0,
    }
    atomic_text(
        args.return_manifest,
        json.dumps(return_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate, audit, budget, and seal P7.5-Full manifests without scoring."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data/manifests/p7_full_v1"
RAW_LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
SPLIT_AUDIT = ROOT / "results/data_audit/yambda50m_p7/split_coverage_audit_v1.json"
CONTRACT = ROOT / "configs/contracts/p7_5_materialization_contract_v1.yaml"
EXCLUSIONS = ROOT / "data/manifests/p7_canary/qualification_exclusions_v1.json"
LAYERS = 4
SEEDS = 3
FIDELITY_FORBIDDEN = (
    "label",
    "target_index",
    "rankable",
    "target_stratum",
    "is_organic",
    "prior_30m_same_item",
    "latest_item",
)


def sha256_file(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            output.update(block)
    return output.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def concatenate(paths: list[Path], *, use_threads: bool = True) -> pa.Table:
    return pa.concat_tables([pq.read_table(path, use_threads=use_threads) for path in paths])


def distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        f"p{quantile}": float(np.percentile(values, quantile))
        for quantile in (50, 90, 95, 99)
    } | {"max": int(values.max())}


def request_summary(table: pa.Table) -> dict:
    uids = table["uid"].to_numpy(zero_copy_only=False).astype(np.int64)
    candidates = table["candidate_count"].to_numpy(zero_copy_only=False).astype(np.int64)
    histories = table["effective_prefix_length"].to_numpy(zero_copy_only=False).astype(np.int64)
    per_user = np.asarray(list(Counter(uids.tolist()).values()), dtype=np.int64)
    weights = table["request_weight"].to_numpy(zero_copy_only=False).astype(np.float64)
    weight_sums: defaultdict[int, float] = defaultdict(float)
    for uid, weight in zip(uids, weights, strict=True):
        weight_sums[int(uid)] += float(weight)
    return {
        "materialized_queries": len(table),
        "unique_users": len(set(uids.tolist())),
        "queries_per_user": distribution(per_user),
        "candidate_count": distribution(candidates),
        "history_length": distribution(histories),
        "total_candidate_rows_logical": int(candidates.sum()),
        "total_history_tokens": int(histories.sum()),
        "user_weight_sum_max_abs_error_from_one": max(
            (abs(value - 1.0) for value in weight_sums.values()), default=0.0
        ),
    }


class RawRowReader:
    def __init__(self, path: Path) -> None:
        self.parquet = pq.ParquetFile(path)
        sizes = [self.parquet.metadata.row_group(index).num_rows for index in range(self.parquet.num_row_groups)]
        self.ends = np.cumsum(sizes).tolist()
        self.cache: dict[int, pa.Table] = {}

    def rows(self, start: int, end: int) -> pa.Table:
        pieces = []
        cursor = start
        while cursor < end:
            group = bisect.bisect_right(self.ends, cursor)
            group_start = 0 if group == 0 else self.ends[group - 1]
            group_end = self.ends[group]
            table = self.cache.get(group)
            if table is None:
                table = self.parquet.read_row_group(
                    group, columns=["uid", "timestamp", "item_id", "is_organic"]
                )
                self.cache[group] = table
            local_start = cursor - group_start
            take = min(end, group_end) - cursor
            pieces.append(table.slice(local_start, take))
            cursor += take
        return pa.concat_tables(pieces)


def shard_rows(paths: list[Path], row_start: int, count: int) -> pa.Table:
    pieces = []
    cursor = 0
    wanted_end = row_start + count
    for path in paths:
        rows = pq.ParquetFile(path).metadata.num_rows
        shard_end = cursor + rows
        if shard_end > row_start and cursor < wanted_end:
            table = pq.read_table(path)
            local_start = max(row_start, cursor) - cursor
            local_end = min(wanted_end, shard_end) - cursor
            pieces.append(table.slice(local_start, local_end - local_start))
        cursor = shard_end
        if cursor >= wanted_end:
            break
    if not pieces:
        raise AssertionError(f"candidate range not found: {row_start}:{wanted_end}")
    return pa.concat_tables(pieces)


def validate_roundtrip(root: Path, indices: dict[str, dict]) -> dict:
    references = json.loads((root / "expanded_reference_samples.json").read_text())
    requests = {
        split: concatenate([root / split / row["path"] for row in index["request_shards"]])
        for split, index in indices.items()
    }
    request_rows = {
        row["request_id"]: row
        for table in requests.values()
        for row in table.to_pylist()
    }
    raw = RawRowReader(RAW_LISTENS)
    checked = 0
    for reference in references.values():
        row = request_rows[reference["request_id"]]
        split = row["split"]
        history_table = raw.rows(
            int(row["raw_user_row_start"]), int(row["raw_prefix_end_exclusive"])
        )
        raw_rows = history_table.to_pylist()[-int(row["effective_prefix_length"]):]
        reconstructed = [
            [
                int(value["item_id"]),
                int(value["timestamp"]),
                1 + (1 - int(value["is_organic"])),
            ]
            for value in raw_rows
            if int(value["timestamp"]) < int(row["query_timestamp"])
        ]
        assert reconstructed == reference["history"]
        assert all(value[1] < int(row["query_timestamp"]) for value in reconstructed)
        candidate_paths = [
            root / split / shard["path"] for shard in indices[split]["candidate_shards"]
        ]
        candidate_table = shard_rows(
            candidate_paths, int(row["candidate_offset"]), int(row["candidate_count"])
        )
        assert candidate_table["candidate_set_id"].to_pylist() == [
            row["candidate_set_id"]
        ] * int(row["candidate_count"])
        assert candidate_table["candidate_position"].to_pylist() == list(
            range(int(row["candidate_count"]))
        )
        assert candidate_table["candidate_item_id"].to_pylist() == reference["candidates"]
        stored_features = np.asarray(candidate_table["base_features"].to_pylist(), dtype=np.float32)
        expected_features = np.asarray(reference["base_features"], dtype=np.float32)
        np.testing.assert_allclose(stored_features, expected_features, rtol=0.0, atol=1e-6)
        assert row["label"] == reference["label"]
        assert row["target_index"] == reference["target_index"]
        assert abs(float(row["request_weight"]) - float(reference["weight"])) < 1e-12
        checked += 1

    deterministic_reads = {}
    for split, index in indices.items():
        paths = [root / split / row["path"] for row in index["request_shards"]]
        serial = concatenate(paths, use_threads=False)
        threaded = concatenate(paths, use_threads=True)
        assert serial.equals(threaded)
        ordered_ids = serial["request_id"].to_pylist()
        reversed_ids = concatenate(list(reversed(paths)))["request_id"].to_pylist()
        assert sorted(ordered_ids) == sorted(reversed_ids)
        deterministic_reads[split] = {
            "serial_equals_threaded": True,
            "shard_order_preserves_request_set": True,
            "request_set_digest": digest(sorted(ordered_ids)),
        }
    return {
        "status": "passed",
        "expanded_reference_samples_checked": checked,
        "float_tolerance": 1e-6,
        "deterministic_reads": deterministic_reads,
    }


def validate_and_audit(root: Path, indices: dict[str, dict]) -> tuple[dict, dict]:
    audit_source = json.loads(SPLIT_AUDIT.read_text())["workloads"]
    exclusions = json.loads(EXCLUSIONS.read_text())["conservative_exact_request_exclusions"]
    excluded_signatures = {
        (str(row["workload"]), int(row["uid"]), int(row["query_timestamp"]))
        for row in exclusions
    }
    coverage: dict[str, dict] = {}
    invariant_counts = Counter()
    feature_coverage: dict[str, dict] = {}
    residual_views: dict[str, dict] = {}

    for split, index in indices.items():
        request_paths = [root / split / row["path"] for row in index["request_shards"]]
        candidate_paths = [root / split / row["path"] for row in index["candidate_shards"]]
        for metadata, path in zip(index["request_shards"], request_paths, strict=True):
            assert sha256_file(path) == metadata["sha256"]
            assert pq.ParquetFile(path).metadata.num_rows == metadata["rows"]
        for metadata, path in zip(index["candidate_shards"], candidate_paths, strict=True):
            assert sha256_file(path) == metadata["sha256"]
            assert pq.ParquetFile(path).metadata.num_rows == metadata["rows"]
        requests = concatenate(request_paths)
        candidates = concatenate(candidate_paths)
        assert len(requests) == index["total_request_rows"]
        assert len(candidates) == index["total_candidate_rows"]
        assert candidates["candidate_global_index"].to_pylist() == list(range(len(candidates)))
        assert pc.all(pc.is_finite(pc.list_flatten(candidates["base_features"]))).as_py()

        for workload in ("N", "R", "F"):
            workload_candidates = candidates.filter(pc.equal(candidates["workload"], workload))
            if len(workload_candidates):
                feature_values = np.asarray(
                    workload_candidates["base_features"].to_pylist(), dtype=np.float32
                )
                feature_coverage[f"{workload}:{split}"] = {
                    "physical_candidate_rows": len(feature_values),
                    "nonfinite_values": int((~np.isfinite(feature_values)).sum()),
                    "missing_indicator_1_rows": int((feature_values[:, -1] == 1.0).sum()),
                    "zero_popularity_rows": int((feature_values[:, 4] == 0.0).sum()),
                }

        for workload, kind in sorted(
            set(
                zip(
                    requests["workload"].to_pylist(),
                    requests["manifest_kind"].to_pylist(),
                    strict=True,
                )
            )
        ):
            mask = pc.and_(
                pc.equal(requests["workload"], workload),
                pc.equal(requests["manifest_kind"], kind),
            )
            view = requests.filter(mask)
            key = f"{workload}:{split}:{kind}"
            summary = request_summary(view)
            raw = audit_source[workload][split]
            summary["raw_eligible_queries"] = int(raw["query_count"])
            summary["excluded_or_not_selected_queries"] = int(raw["query_count"] - len(view))
            summary["selection_reason"] = (
                "frozen deterministic anchor/query budget"
                if workload != "R" or split == "base_fit"
                else "complete eligible population or its frozen rankable subset"
            )
            coverage[key] = summary
            assert summary["user_weight_sum_max_abs_error_from_one"] < 1e-10
            if "fidelity" in kind:
                for column in FIDELITY_FORBIDDEN:
                    assert view[column].null_count == len(view)
                invariant_counts["fidelity_rows_without_target_values"] += len(view)
            if workload == "R" and kind == "quality_rankable":
                targets = view["target_index"].to_numpy(zero_copy_only=False)
                counts = view["candidate_count"].to_numpy(zero_copy_only=False)
                assert np.all((targets >= 0) & (targets < counts))
                invariant_counts["R_rankable_targets_in_range"] += len(view)
            if split == "qualification":
                for row in view.select(["workload", "uid", "query_timestamp"]).to_pylist():
                    assert (row["workload"], row["uid"], row["query_timestamp"]) not in excluded_signatures
            if split == "residual_train" and (
                (workload == "N" and kind == "quality")
                or (workload == "R" and kind == "quality_rankable")
                or (workload == "F" and kind == "quality")
            ):
                residual_views[workload] = summary

        if split in {"development", "qualification", "residual_train"}:
            r_all = requests.filter(
                pc.and_(
                    pc.equal(requests["workload"], "R"),
                    pc.equal(requests["manifest_kind"], "fidelity_all_eligible"),
                )
            )
            r_quality = requests.filter(
                pc.and_(
                    pc.equal(requests["workload"], "R"),
                    pc.equal(requests["manifest_kind"], "quality_rankable"),
                )
            )
            raw_r = audit_source["R"][split]
            assert len(r_all) == int(raw_r["query_count"])
            expected_rankable = round(
                int(raw_r["query_count"]) * float(raw_r["rankable_coverage"])
            )
            assert len(r_quality) == expected_rankable
            coverage[f"R:{split}:population_denominators"] = {
                "eligible_session_starts": len(r_all),
                "nonempty_familiar_universe": len(r_all),
                "rankable_familiar_returns": len(r_quality),
                "quality_coverage": len(r_quality) / len(r_all),
                "fidelity_coverage": 1.0,
            }

        f_quality = requests.filter(
            pc.and_(
                pc.equal(requests["workload"], "F"),
                pc.equal(requests["manifest_kind"], "quality"),
            )
        )
        if len(f_quality):
            labels = f_quality["label"].to_pylist()
            organic = f_quality["is_organic"].to_pylist()
            prior = f_quality["prior_30m_same_item"].to_pylist()
            latest = f_quality["latest_item"].to_pylist()
            coverage[f"F:{split}:cohorts"] = {
                "likes": sum(value == 1 for value in labels),
                "dislikes": sum(value == 0 for value in labels),
                "prior_30m_same_item": sum(bool(value) for value in prior),
                "non_prior_30m": sum(not bool(value) for value in prior),
                "latest_item": sum(bool(value) for value in latest),
                "organic": sum(value == 1 for value in organic),
                "recommendation_driven": sum(value == 0 for value in organic),
            }

    budgets = {}
    for workload, summary in sorted(residual_views.items()):
        queries = summary["materialized_queries"]
        candidates = summary["total_candidate_rows_logical"]
        histories = summary["total_history_tokens"]
        budgets[f"M0-{workload}"] = {
            "per_seed_unique_queries": queries,
            "per_seed_query_presentations": queries,
            "all_3_seeds_query_presentations": queries * SEEDS,
            "per_seed_candidate_rows": candidates,
            "per_seed_history_tokens": histories,
            "per_seed_token_layer_work": LAYERS * (candidates + histories),
        }
    budgets["M1"] = {
        key: sum(value[key] for value in budgets.values())
        for key in next(iter(budgets.values()))
    }
    budgets["interpretation"] = {
        "strict_compute_matched": False,
        "query_presentation_definition": "one frozen manifest pass per seed",
        "token_layer_work_proxy": "4 * (history_tokens + candidate_query_rows)",
        "M1_extra_cross_task_supervision": "intended treatment",
    }
    return {
        "status": "passed",
        "coverage": coverage,
        "feature_coverage": feature_coverage,
        "invariants": dict(invariant_counts),
    }, budgets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    summary_path = root / "materialization_summary.json"
    summary = json.loads(summary_path.read_text())
    indices = {
        split: json.loads((root / split / "manifest.index.json").read_text())
        for split in ("base_fit", "residual_train", "development", "qualification")
    }
    roundtrip = validate_roundtrip(root, indices)
    audit, budgets = validate_and_audit(root, indices)

    roundtrip_path = root / "roundtrip_equivalence_v1.json"
    roundtrip_path.write_text(json.dumps(roundtrip, indent=2) + "\n")
    coverage_path = root / "coverage_conservation_v1.json"
    coverage_path.write_text(json.dumps(audit, indent=2) + "\n")
    budget_path = root / "m0_m1_budget_v1.json"
    budget_path.write_text(json.dumps(budgets, indent=2) + "\n")

    qualification_index = root / "qualification/manifest.index.json"
    qualification = indices["qualification"]
    seal = {
        "contract": "p7_5_materialization_contract_v1",
        "status": "sealed_unread_for_scoring",
        "qualification_index_hash": sha256_file(qualification_index),
        "qualification_manifest_hash": digest(
            {
                "request_shards": qualification["request_shards"],
                "candidate_shards": qualification["candidate_shards"],
            }
        ),
        "raw_source_hashes": qualification["raw_source_hashes"],
        "materialization_contract_hash": sha256_file(CONTRACT),
        "materializer_code_hash": qualification["materializer_code_hash"],
        "verification_code_hash": sha256_file(Path(__file__)),
        "code_commit": git_commit(),
        "canary_exclusion_hash": sha256_file(EXCLUSIONS),
        "roundtrip_report_hash": sha256_file(roundtrip_path),
        "coverage_report_hash": sha256_file(coverage_path),
        "budget_report_hash": sha256_file(budget_path),
        "qualification_scored": False,
        "base_fitted": False,
        "hstu_trained": False,
    }
    seal_path = root / "qualification_seal_v1.json"
    seal_path.write_text(json.dumps(seal, indent=2) + "\n")
    summary.update(
        {
            "status": "verified_and_sealed_unscored",
            "roundtrip_report": str(roundtrip_path.relative_to(ROOT)),
            "coverage_report": str(coverage_path.relative_to(ROOT)),
            "budget_report": str(budget_path.relative_to(ROOT)),
            "qualification_seal": str(seal_path.relative_to(ROOT)),
            "qualification_seal_hash": sha256_file(seal_path),
            "qualification_scored": False,
            "base_fitted": False,
            "hstu_trained": False,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "seal": seal}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

EDGES = (
    "theta1_to_theta2",
    "theta2_to_theta3",
    "theta3_to_theta4",
)
METHODS = (
    "all_reuse",
    "compiled_direct_oldkv",
    "mixed_fixed20",
    "all_exact",
)
PROTOCOL = "evokv_xp_d1_quality_development_v1"
BASELINE_PROTOCOL = "evokv_xp_reuse_exact_suffix_diagnostic_development_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"summary differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def close_fraction(reuse: float, candidate: float, exact: float) -> float:
    denominator = reuse - exact
    if denominator <= 0:
        raise ValueError("selected baseline no longer has a positive reuse gap")
    return (reuse - candidate) / denominator


def main() -> None:
    args = parse_args()
    rows = []
    fit_kinds = set()
    for edge in EDGES:
        result_path = args.result_root / f"{edge}.json"
        baseline_path = args.baseline_root / "cells" / f"{edge}.json"
        result = json.loads(result_path.read_text())
        baseline = json.loads(baseline_path.read_text())
        if (
            result.get("protocol") != PROTOCOL
            or result.get("status") != "complete"
            or result.get("world_size") != 2
            or baseline.get("protocol") != BASELINE_PROTOCOL
            or baseline.get("status") != "complete"
            or set(result["quality"]["qualification_test"]["methods"])
            != set(METHODS)
        ):
            raise ValueError(f"D1 bridge result contract differs: {edge}")
        baseline_quality = baseline["quality_by_negative_count"]["999"]
        result_quality = result["quality"]["qualification_test"]
        if (
            result["roles"]["qualification_test"][
                "candidate_sha256_per_rank"
            ]
            != baseline["role"][
                "candidate_sha256_per_rank_by_negative_count"
            ]["999"]
            or result_quality["record_ids_sha256"]
            != baseline_quality["record_ids_sha256"]
            or result["recommendation_contract"]["negative_candidates"]
            != 999
            or result["recommendation_contract"]["common_cache_endpoint"][
                "storage_dtype"
            ]
            != "torch.float16"
        ):
            raise ValueError(f"D1 bridge comparison binding differs: {edge}")
        methods = result_quality["methods"]
        fit_kinds.add(
            result.get("compile_metrics", {}).get(
                "fit_kind",
                "analytic_model_projection_no_quality_label_fit",
            )
        )
        baseline_methods = baseline_quality["methods"]
        for endpoint in ("all_reuse", "all_exact"):
            result_ce = float(
                methods[endpoint]["recommendation"]["sampled_cross_entropy"]
            )
            baseline_ce = float(
                baseline_methods[endpoint]["recommendation"][
                    "sampled_cross_entropy"
                ]
            )
            if not math.isclose(result_ce, baseline_ce, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"D1 bridge endpoint differs from baseline: {edge}")
        ce = {
            method: float(
                methods[method]["recommendation"]["sampled_cross_entropy"]
            )
            for method in METHODS
        }
        ndcg = {
            method: float(methods[method]["recommendation"]["ndcg_at_10"])
            for method in METHODS
        }
        cache_error = {
            method: float(
                methods[method]["cache_fidelity"]["relative_error_mean"]
            )
            for method in METHODS
        }
        maintenance = {
            method: float(
                methods[method]["gpu_cost"][
                    "max_rank_maintenance_milliseconds"
                ]
            )
            for method in METHODS
        }
        selection = result["roles"]["qualification_test"][
            "mixed_fixed20_selection"
        ]
        row = {
            "edge": edge,
            "positive_targets": int(
                methods["all_exact"]["recommendation"]["positive_targets"]
            ),
            "reuse_exact_ce_gap": ce["all_reuse"] - ce["all_exact"],
            "compiled_ce_gap_closed": close_fraction(
                ce["all_reuse"], ce["compiled_direct_oldkv"], ce["all_exact"]
            ),
            "mixed_ce_gap_closed": close_fraction(
                ce["all_reuse"], ce["mixed_fixed20"], ce["all_exact"]
            ),
            "mixed_record_exact_fraction": float(
                selection["actual_record_fraction"]
            ),
            "mixed_retained_token_exact_fraction": float(
                selection["actual_retained_token_fraction"]
            ),
            "ce": ce,
            "ndcg_at_10": ndcg,
            "cache_relative_error": cache_error,
            "max_rank_maintenance_milliseconds": maintenance,
            "compiled_maintenance_over_exact": (
                maintenance["compiled_direct_oldkv"] / maintenance["all_exact"]
            ),
            "mixed_component_bound_over_exact": (
                maintenance["mixed_fixed20"] / maintenance["all_exact"]
            ),
            "candidate_sha256_per_rank": result["roles"][
                "qualification_test"
            ]["candidate_sha256_per_rank"],
            "result_path": str(result_path),
        }
        rows.append(row)
    summary = {
        "schema": "evokv_selected_d1_bridge_summary_development_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "result_root": str(args.result_root),
        "baseline_root": str(args.baseline_root),
        "protocol": PROTOCOL,
        "fit_kinds": sorted(fit_kinds),
        "endpoint_parity_with_selected_baseline": True,
        "mixed_cost_is_end_to_end": False,
        "interpretation": (
            "development-only large-XP D1 bridge diagnostic with endpoint parity; "
            "the fit kind is recorded per result and never uses qualification labels"
        ),
        "edges": rows,
    }
    atomic_text(
        args.output,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    header = (
        "edge\tpositive_targets\treuse_exact_ce_gap\tcompiled_ce_gap_closed\t"
        "mixed_ce_gap_closed\tmixed_record_exact_fraction\t"
        "mixed_retained_token_exact_fraction\tcompiled_maintenance_over_exact\t"
        "mixed_component_bound_over_exact"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[field])
                for field in (
                    "edge",
                    "positive_targets",
                    "reuse_exact_ce_gap",
                    "compiled_ce_gap_closed",
                    "mixed_ce_gap_closed",
                    "mixed_record_exact_fraction",
                    "mixed_retained_token_exact_fraction",
                    "compiled_maintenance_over_exact",
                    "mixed_component_bound_over_exact",
                )
            )
        )
    atomic_text(args.tsv, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": "complete",
                "tsv": str(args.tsv),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

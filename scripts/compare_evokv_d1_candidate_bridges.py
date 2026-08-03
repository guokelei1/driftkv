from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"comparison differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def load_summary(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if (
        value.get("status") != "complete"
        or value.get("endpoint_parity_with_selected_baseline") is not True
        or len(value.get("edges", [])) != 3
    ):
        raise ValueError(f"D1 bridge summary differs: {path}")
    return value


def summarize(name: str, path: Path, control: dict[str, object]) -> dict[str, object]:
    value = load_summary(path)
    control_edges = {edge["edge"]: edge for edge in control["edges"]}
    rows = []
    for edge in value["edges"]:
        control_edge = control_edges[edge["edge"]]
        if not math.isclose(
            float(edge["reuse_exact_ce_gap"]),
            float(control_edge["reuse_exact_ce_gap"]),
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(f"candidate endpoint gap differs: {name} {edge['edge']}")
        rows.append(
            {
                "edge": edge["edge"],
                "reuse_exact_ce_gap": float(edge["reuse_exact_ce_gap"]),
                "compiled_ce_gap_closed": float(edge["compiled_ce_gap_closed"]),
                "mixed_ce_gap_closed": float(edge["mixed_ce_gap_closed"]),
                "compiled_maintenance_over_exact": float(
                    edge["compiled_maintenance_over_exact"]
                ),
                "compiled_recovery_delta_from_control": float(
                    edge["compiled_ce_gap_closed"]
                )
                - float(control_edge["compiled_ce_gap_closed"]),
                "mixed_recovery_delta_from_control": float(
                    edge["mixed_ce_gap_closed"]
                )
                - float(control_edge["mixed_ce_gap_closed"]),
            }
        )
    compiled = [row["compiled_ce_gap_closed"] for row in rows]
    compiled_delta = [row["compiled_recovery_delta_from_control"] for row in rows]
    mixed = [row["mixed_ce_gap_closed"] for row in rows]
    return {
        "name": name,
        "summary_path": str(path),
        "summary_sha256": sha256(path),
        "fit_kinds": value.get("fit_kinds", []),
        "minimum_compiled_recovery": min(compiled),
        "mean_compiled_recovery": sum(compiled) / len(compiled),
        "minimum_compiled_recovery_delta_from_control": min(compiled_delta),
        "mean_compiled_recovery_delta_from_control": sum(compiled_delta)
        / len(compiled_delta),
        "mean_mixed_recovery": sum(mixed) / len(mixed),
        "uniformly_improves_compiled_recovery": min(compiled_delta) > 0.0,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    control = load_summary(args.control)
    parsed = []
    for item in args.candidate:
        if "=" not in item:
            raise ValueError("candidate must use name=summary-path")
        name, raw_path = item.split("=", 1)
        parsed.append(summarize(name, Path(raw_path), control))
    admissible = [
        value
        for value in parsed
        if value["uniformly_improves_compiled_recovery"]
    ]
    preferred = max(
        admissible,
        key=lambda value: (
            value["minimum_compiled_recovery_delta_from_control"],
            value["mean_compiled_recovery"],
        ),
        default=None,
    )
    output = {
        "schema": "evokv_d1_residual_candidate_comparison_development_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "control_summary": str(args.control),
        "control_summary_sha256": sha256(args.control),
        "selection_rule": (
            "require positive compiled-recovery improvement on all three edges, "
            "then maximize the worst-edge improvement and mean recovery"
        ),
        "preferred_candidate": None if preferred is None else preferred["name"],
        "candidates": parsed,
    }
    atomic_text(args.output, json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = [
        "candidate\tedge\treuse_exact_ce_gap\tcompiled_ce_gap_closed\t"
        "compiled_delta_from_control\tmixed_ce_gap_closed\t"
        "mixed_delta_from_control\tcompiled_maintenance_over_exact"
    ]
    for candidate in parsed:
        for row in candidate["rows"]:
            lines.append(
                "\t".join(
                    str(value)
                    for value in (
                        candidate["name"],
                        row["edge"],
                        row["reuse_exact_ce_gap"],
                        row["compiled_ce_gap_closed"],
                        row["compiled_recovery_delta_from_control"],
                        row["mixed_ce_gap_closed"],
                        row["mixed_recovery_delta_from_control"],
                        row["compiled_maintenance_over_exact"],
                    )
                )
            )
    atomic_text(args.tsv, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "preferred_candidate": output["preferred_candidate"],
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

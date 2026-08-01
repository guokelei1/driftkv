from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = (
    "fixed_deep_suffix",
    "plain_progressive_prefix",
    "recent_token_rectangles",
    "arbitrary_contiguous_intervals",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "results/baseline_foundation/"
            "d1_same_sla_development_v0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/baseline_foundation/"
            "d1_same_sla_development_v0_summary.json"
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_cell(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("protocol")
        != "d1_same_sla_baseline_development_v0"
        or value.get("scientific_result") is not False
        or value.get("formal_result") is not False
    ):
        raise ValueError(f"D1 baseline bundle differs: {path}")
    return value


def family_summary(value: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    selection = value["selection"]
    if not isinstance(selection, dict):
        raise ValueError("D1 baseline selection is invalid")
    raw_families = selection["families"]
    if not isinstance(raw_families, dict):
        raise ValueError("D1 baseline families are invalid")
    for name in FAMILIES:
        raw = raw_families.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"D1 baseline family is missing: {name}")
        probe = raw.get("probe")
        test = raw.get("test")
        if not isinstance(probe, dict) or not isinstance(test, dict):
            raise ValueError(f"D1 baseline family split differs: {name}")
        output[name] = {
            "selected": raw["selected"],
            "exact_fallback": bool(raw["fallback_used"]),
            "probe_cache_fidelity_recovery": float(
                probe["cache_fidelity_recovery"]
            ),
            "probe_cost_ratio_to_exact": float(
                probe["migration_ratio_to_recompute"]
            ),
            "test_cache_fidelity_recovery": float(
                test["cache_fidelity_recovery"]
            ),
            "test_cost_ratio_to_exact": float(
                test["migration_ratio_to_recompute"]
            ),
        }
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = sorted(args.input_dir.glob("*.json"))
    if not paths:
        raise ValueError("no D1 baseline bundles found")
    cells = []
    selected_non_exact = 0
    total_family_cells = 0
    for path in paths:
        value = load_cell(path)
        split = value["common_split"]
        if not isinstance(split, dict):
            raise ValueError("D1 baseline split is invalid")
        families = family_summary(value)
        selected_non_exact += sum(
            not bool(item["exact_fallback"])
            for item in families.values()
        )
        total_family_cells += len(families)
        cells.append(
            {
                "cell": value["cell"],
                "seed": int(value["seed"]),
                "probe_users": int(split["probe_users"]),
                "test_users": int(split["test_users"]),
                "probe_test_disjoint": bool(split["disjoint"]),
                "candidate_manifest_sha256": value[
                    "candidate_manifest_sha256"
                ],
                "artifact": {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                },
                "families": families,
            }
        )
    result = {
        "protocol": "d1_same_sla_baseline_development_v0_summary",
        "source_protocol": "d1_same_sla_baseline_development_v0",
        "protocol_status": "development",
        "scientific_result": False,
        "formal_result": False,
        "selection_contract": {
            "probe_cache_fidelity_recovery_target": 0.5,
            "selection": (
                "minimum measured GPU cost among candidates meeting "
                "the probe recovery target; exact fallback otherwise"
            ),
            "test_role": "held-out reporting only",
        },
        "cells": cells,
        "aggregate": {
            "cells": len(cells),
            "family_cells": total_family_cells,
            "non_exact_selections": selected_non_exact,
            "exact_fallbacks": total_family_cells - selected_non_exact,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(
        args.output.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(args.output)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize label-free daily slices and streaming version-chain capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hstu_kvcache.data.release_windows import (  # noqa: E402
    DAY_SECONDS,
    ReleaseWindowRecipe,
    daily_slices,
    max_equal_train_days,
    plan_release_slots,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_streaming_windows_v1.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = yaml.safe_load(CONTRACT.read_text())
    output = args.output or ROOT / contract["output"]
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing window plan: {output}")
    coverage = contract["coverage_seconds"]
    base_cutoff = int(coverage["theta0_cutoff"])
    complete_end = int(coverage["complete_day_end_exclusive"])
    slices = daily_slices(base_cutoff, complete_end)
    lock_from = int(contract["blind_boundary"]["lock_from_update_index"])

    recipes = {}
    for name, values in contract["recipes"].items():
        recipe = ReleaseWindowRecipe(name=name, **values)
        capacities = {}
        for view, view_values in contract["capacity_views"].items():
            if not isinstance(view_values, dict) or "usable_end_day" not in view_values:
                continue
            usable_end = int(view_values["usable_end_day"]) * DAY_SECONDS
            slots = plan_release_slots(
                base_cutoff=base_cutoff,
                usable_end=usable_end,
                recipe=recipe,
                blind_lock_from_update=lock_from,
            )
            capacities[view] = {
                "usable_end": usable_end,
                "max_fully_evaluable_updates": len(slots),
                "max_versions_including_theta0": len(slots) + 1,
                "slots": [slot.as_dict() for slot in slots],
            }
        recipes[name] = {"recipe": asdict(recipe), "capacity": capacities}

    allocation_contract = contract["capacity_views"]["equal_allocation_examples"]
    equal_allocation = {}
    for view, view_values in contract["capacity_views"].items():
        if not isinstance(view_values, dict) or "usable_end_day" not in view_values:
            continue
        usable_end = int(view_values["usable_end_day"]) * DAY_SECONDS
        equal_allocation[view] = {
            str(total_versions): max_equal_train_days(
                base_cutoff=base_cutoff,
                usable_end=usable_end,
                updates=int(total_versions) - 1,
                admission_days=int(allocation_contract["admission_days_per_update"]),
                evaluation_days=int(allocation_contract["final_evaluation_days"]),
            )
            for total_versions in allocation_contract["total_versions_including_theta0"]
        }

    payload = {
        "plan": "yambda500m_streaming_windows_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256_file(CONTRACT),
        "planner": str(Path(__file__).resolve().relative_to(ROOT)),
        "planner_sha256": sha256_file(Path(__file__).resolve()),
        "population_manifest": "data/processed/yambda500m_unified_v1/manifest.json",
        "physical_daily_slices": [asdict(window) for window in slices],
        "partial_tail": coverage["partial_tail"],
        "recipes": recipes,
        "max_equal_train_days_by_total_versions": equal_allocation,
        "unlocked_F_coverage_reference": contract["unlocked_F_coverage_reference"],
        "locked_slot_F_counts_materialized": False,
        "authorization": {
            "training": False,
            "theta3_data_or_labels": False,
            "calendar_capacity_only": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "daily_slices": len(slices),
        "capacity": {
            name: {
                view: values["max_fully_evaluable_updates"]
                for view, values in recipe["capacity"].items()
            }
            for name, recipe in recipes.items()
        },
        "theta3_and_later_locked": True,
        "max_equal_train_days_by_total_versions": equal_allocation,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

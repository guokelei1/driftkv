from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from hstu_kvcache.migration.stage5_accounting import (
    build_stage5_source_state_accounting,
    validate_stage5_source_state_accounting,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "configs/cohortkv_single_config_v1/stage2_compiler_summary.json"
STAGE4 = ROOT / "configs/cohortkv_single_config_v1/stage4_system_summary.json"
STAGE4_5 = (
    ROOT / "configs/cohortkv_single_config_v1/stage4_5_source_plan_summary.json"
)
OUTPUT = (
    ROOT
    / "results/system/cohortkv_single_config_full_chain_v1"
    / "stage5_source_state_accounting_seed0.json"
)
RESULT_SCHEMA = (
    ROOT / "configs/cohortkv_single_config_v1/result.schema.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, default=STAGE2)
    parser.add_argument("--stage4", type=Path, default=STAGE4)
    parser.add_argument("--stage4-5", type=Path, default=STAGE4_5)
    parser.add_argument("--result-schema", type=Path, default=RESULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def validate_schema(value: dict[str, object], schema_path: Path) -> None:
    integrated_schema = json.loads(schema_path.read_text())
    schema = integrated_schema["properties"]["source_state_accounting"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)


def main() -> None:
    args = parse_args()
    payload = build_stage5_source_state_accounting(
        args.stage2,
        args.stage4,
        args.stage4_5,
    )
    validate_stage5_source_state_accounting(
        payload,
        args.stage2,
        args.stage4,
        args.stage4_5,
    )
    validate_schema(payload, args.result_schema)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if args.stdout:
        print(encoded)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(f"{encoded}\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

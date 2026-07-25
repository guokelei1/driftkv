"""Prepare a supported KuaiRand 16-day long-context artifact."""

from __future__ import annotations

import argparse
import json

from motivation_validity import STANDARD_LOGS

from hstu_kvcache.data import (
    StreamingDataPlan,
    load_prepared_kuairand_plan,
    save_prepared_kuairand_plan,
)
from hstu_kvcache.streaming import (
    SUPPORTED_LONG_CONTEXT_BASE_DAYS,
    long_context_split_name,
    prepared_protocol_for_base_days,
    validate_long_context_plan,
)


def default_output(base_days: int) -> str:
    split = long_context_split_name(base_days)
    if base_days == 8:
        return "data/processed/kuairand_long_context_8plus8_v2.npz"
    return f"data/processed/kuairand_long_context_{split}_exploration_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-days",
        type=int,
        choices=SUPPORTED_LONG_CONTEXT_BASE_DAYS,
        default=8,
    )
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_output(args.base_days)
    if args.validate_existing:
        plan, metadata = load_prepared_kuairand_plan(output)
        validate_long_context_plan(plan, metadata, args.base_days)
        print(json.dumps(metadata, indent=2), flush=True)
        return
    plan = StreamingDataPlan.from_csvs(
        STANDARD_LOGS,
        base_num_days=args.base_days,
        total_num_days=16,
        max_seq_len=2048,
        max_items=50000,
        max_users=1000,
        min_interactions_per_user=5,
        fit_vocabulary_on_base=True,
        context_hash_buckets=262144,
        history_window_days=8,
    )
    metadata = save_prepared_kuairand_plan(
        plan,
        output,
        source_paths=STANDARD_LOGS,
        overwrite=args.overwrite,
        protocol=prepared_protocol_for_base_days(args.base_days),
    )
    validate_long_context_plan(plan, metadata, args.base_days)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

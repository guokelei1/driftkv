from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import (
    D2ActionPlan,
    build_d2_phase_ledger,
    characterize_d2_requests,
    characterize_d2_scoped_dedup,
)
from hstu_kvcache.migration.design2_plan import file_sha256
from hstu_kvcache.streaming import (
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_OUTPUT = (
    "configs/cohortkv_d2/stage_a_request_characterization.json"
)
_VERSION = re.compile(r"^theta([0-9]+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--prepared-data")
    parser.add_argument(
        "--world-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64, 128, 682],
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(args: argparse.Namespace) -> dict[str, object]:
    action_plan_path = _path(args.action_plan)
    action_plan = D2ActionPlan.load(action_plan_path)
    prepared_path = _path(
        args.prepared_data
        or action_plan.provenance.prepared_data
    )
    output_path = _path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "D2 Stage A request output exists; pass --force"
        )
    if (
        file_sha256(prepared_path)
        != action_plan.provenance.prepared_data_sha256
    ):
        raise ValueError("D2 prepared data hash differs")
    plan, metadata = load_prepared_kuairand_plan(prepared_path)
    validate_long_context_plan(plan, metadata, 4)
    user_ids = tuple(
        value.prepared_user_id for value in action_plan.records
    )
    windows = reconstruct_organic_windows(plan, user_ids)
    match = _VERSION.fullmatch(action_plan.target_version)
    if match is None:
        raise ValueError("D2 target version differs")
    target_index = int(match.group(1))
    target_window = windows[target_index]
    record_by_id = {
        value.record_id: value for value in action_plan.records
    }
    target_items = {}
    history_checks = []
    for record_id, record in record_by_id.items():
        window_record = target_window.records[
            record.prepared_user_id
        ]
        history = window_record.history
        history_checks.append(
            history is not None
            and window_record.history_sha256
            == record.target_history_sha256
            and len(history) == record.final_tokens
        )
        if history is None:
            raise ValueError("D2 target history is missing")
        target_items[record_id] = tuple(
            int(value) for value in history.item_ids
        )
    characterization = characterize_d2_requests(
        action_plan,
        target_items,
        embedding_dim=512,
    )
    scoped_dedup = characterize_d2_scoped_dedup(
        action_plan,
        target_items,
        num_embedding_rows=plan.trace.num_items + 1,
        world_sizes=tuple(args.world_sizes),
        batch_sizes=tuple(args.batch_sizes),
    )
    ledger = build_d2_phase_ledger(
        action_plan,
        embedding_dim=512,
    )
    mixed_requested = int(
        characterization["mixed"]["full_wave"]["requested_ids"]
    )
    exact_requested = int(
        characterization["all_exact"]["full_wave"][
            "requested_ids"
        ]
    )
    exact_ceiling = characterization["coalescing_ceilings"][
        "exact_retained_or_natural"
    ]
    checks = {
        "prepared_hash": True,
        "target_window_hash": (
            target_window.content_sha256
            == action_plan.provenance.target_window_content_sha256
        ),
        "history_identity_and_length": all(history_checks),
        "mixed_tokens_match_ledger": (
            mixed_requested
            == ledger.boundaries["integrated_post_append"][
                "mixed_lookup_tokens"
            ]
        ),
        "all_exact_tokens_match_ledger": (
            exact_requested
            == ledger.boundaries["integrated_post_append"][
                "all_exact_lookup_tokens"
            ]
        ),
        "exact_prefix_combined_ceiling": (
            exact_ceiling["requested_ids"] == 132711
            and exact_ceiling["unique_ids"] == 96844
        ),
        "scoped_world_size_coverage": (
            {
                value["world_size"]
                for value in scoped_dedup["points"]
            }
            == set(args.world_sizes)
        ),
        "scoped_remote_and_fanout_fields": all(
            "remote_requested_fraction"
            in value["exact_prefix_coalesced"]
            and "requester_rank_fanout_distribution"
            in value["exact_prefix_coalesced"]
            for value in scoped_dedup["points"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"D2 Stage A request checks failed: {checks}"
        )
    result = {
        **characterization,
        "status": "complete",
        "action_plan": {
            "path": str(action_plan_path.relative_to(ROOT)),
            "content_sha256": action_plan.content_sha256,
            "file_sha256": file_sha256(action_plan_path),
        },
        "prepared_data": {
            "path": str(prepared_path.relative_to(ROOT)),
            "sha256": file_sha256(prepared_path),
        },
        "target_window": {
            "version": target_index,
            "target_date": target_window.target_date,
            "content_sha256": target_window.content_sha256,
        },
        "scoped_dedup": scoped_dedup,
        "phase_ledger_boundaries": ledger.boundaries,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

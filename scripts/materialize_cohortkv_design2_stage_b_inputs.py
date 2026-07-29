from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hstu_kvcache.data import load_prepared_kuairand_plan
from hstu_kvcache.migration import D2ActionPlan, build_d2_record_owner_map
from hstu_kvcache.migration.design2_plan import canonical_sha256, file_sha256
from hstu_kvcache.streaming import (
    reconstruct_organic_windows,
    validate_long_context_plan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_OUTPUT = "configs/cohortkv_d2/stage_b_sample_inputs.json"
DEFAULT_STAGE_A_SUMMARY = "configs/cohortkv_d2/stage_a_summary.json"
PROTOCOL = "cohortkv_d2_stage_b_sample_inputs_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--stage-a-summary", default=DEFAULT_STAGE_A_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _history(value) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "item_ids": value.item_ids.tolist(),
        "behaviors": value.behaviors.tolist(),
        "time_deltas": value.time_deltas.tolist(),
        "available_length_before_token_cap": (
            value.available_length_before_token_cap
        ),
        "token_truncated": value.token_truncated,
    }


def _select(action_plan: D2ActionPlan) -> tuple[dict[str, object], set[int]]:
    selections = {}
    stage_a_parity = tuple(
        next(
            value.record_id
            for value in action_plan.records
            if value.requested_reason == reason
        )
        for reason in (
            "migrate",
            "scheduled_exact",
            "natural_exact",
        )
    )
    selected_ids: set[int] = set(stage_a_parity)
    selections["stage_a_parity"] = {
        "record_ids": list(stage_a_parity),
        "requested_reasons": [
            "migrate",
            "scheduled_exact",
            "natural_exact",
        ],
    }
    for world_size in (1, 2, 4):
        owner_map = build_d2_record_owner_map(
            action_plan,
            world_size,
            "strict_cow_lpt",
        )
        ranks = {}
        for rank in range(world_size):
            owned = tuple(
                value
                for value in action_plan.records
                if owner_map[value.record_id] == rank
            )
            compiled = min(
                (
                    value
                    for value in owned
                    if value.requested_action == "compiled"
                ),
                key=lambda value: (
                    value.old_tokens,
                    value.retained_tokens,
                    value.record_id,
                ),
            )
            natural = min(
                (
                    value
                    for value in owned
                    if value.requested_reason == "natural_exact"
                ),
                key=lambda value: (
                    value.target_prefix_tokens,
                    value.record_id,
                ),
            )
            scheduled = min(
                (
                    value
                    for value in owned
                    if value.requested_reason == "scheduled_exact"
                ),
                key=lambda value: (
                    value.retained_tokens,
                    value.record_id,
                ),
            )
            ranks[str(rank)] = {
                "compiled": compiled.record_id,
                "natural_exact": natural.record_id,
                "scheduled_exact": scheduled.record_id,
            }
            selected_ids.update(
                (
                    compiled.record_id,
                    natural.record_id,
                    scheduled.record_id,
                )
            )
        selections[str(world_size)] = {
            "owner_strategy": "strict_cow_lpt",
            "owner_map_sha256": canonical_sha256(
                {
                    "record_owner_map": [
                        {
                            "record_id": record_id,
                            "owner_rank": owner,
                        }
                        for record_id, owner in owner_map.items()
                    ]
                }
            ),
            "ranks": ranks,
        }
    return selections, selected_ids


def run(args: argparse.Namespace) -> dict[str, object]:
    output_path = _path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "D2 Stage B sample inputs exist; pass --force"
        )
    action_path = _path(args.action_plan)
    stage_a_path = _path(args.stage_a_summary)
    action_plan = D2ActionPlan.load(action_path)
    stage_a = json.loads(stage_a_path.read_text())
    if (
        stage_a["status"] != "complete"
        or stage_a["stage_b_entry"] != "go"
        or stage_a["action_plan"]["content_sha256"]
        != action_plan.content_sha256
        or stage_a["action_plan"]["file_sha256"]
        != file_sha256(action_path)
    ):
        raise ValueError("D2 Stage A summary differs from the action plan")
    prepared_path = _path(action_plan.provenance.prepared_data)
    data_plan, metadata = load_prepared_kuairand_plan(prepared_path)
    validate_long_context_plan(data_plan, metadata, 4)
    selections, selected_ids = _select(action_plan)
    selected = tuple(
        value
        for value in action_plan.records
        if value.record_id in selected_ids
    )
    windows = reconstruct_organic_windows(
        data_plan,
        (value.prepared_user_id for value in selected),
    )
    source_index = int(
        action_plan.source_version.removeprefix("theta")
    )
    target_index = int(
        action_plan.target_version.removeprefix("theta")
    )
    records = []
    for action in selected:
        source_record = windows[source_index].records[
            action.prepared_user_id
        ]
        target_record = windows[target_index].records[
            action.prepared_user_id
        ]
        if (
            source_record.history_sha256
            != action.old_history_sha256
            or target_record.history_sha256
            != action.target_history_sha256
            or (
                source_record.history is not None
                and len(source_record.history) != action.old_tokens
            )
            or target_record.history is None
            or len(target_record.history) != action.final_tokens
        ):
            raise ValueError("D2 Stage B sampled history differs")
        records.append(
            {
                "action": action.to_dict(),
                "source_history": _history(source_record.history),
                "source_history_sha256": source_record.history_sha256,
                "target_history": _history(target_record.history),
                "target_history_sha256": target_record.history_sha256,
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_result": False,
        "action_plan": {
            "path": str(action_path.relative_to(ROOT)),
            "content_sha256": action_plan.content_sha256,
            "file_sha256": file_sha256(action_path),
        },
        "stage_a_summary": {
            "path": str(stage_a_path.relative_to(ROOT)),
            "sha256": file_sha256(stage_a_path),
        },
        "prepared_data": {
            "path": str(prepared_path.relative_to(ROOT)),
            "sha256": file_sha256(prepared_path),
        },
        "source_version": action_plan.source_version,
        "target_version": action_plan.target_version,
        "selections": selections,
        "records": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_path, output_path)
    return payload


def main() -> None:
    output = run(parse_args())
    print(
        json.dumps(
            {
                "status": output["status"],
                "records": len(output["records"]),
                "content_sha256": output["content_sha256"],
                "scientific_result": output["scientific_result"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

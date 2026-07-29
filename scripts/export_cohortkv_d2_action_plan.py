from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.migration import (
    D2WavePlan,
    D2WaveReport,
    build_d2_phase_ledger,
    export_stage49_h12_action_plan,
)
from hstu_kvcache.migration.design2_plan import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    "results/system/cohortkv_single_config_full_chain_v1/"
    "stage4_9_staggered_renewal_h12_seed0.json"
)
DEFAULT_OUTPUT = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_SUMMARY = "configs/cohortkv_d2/stage_a_plan_adapter_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--step-index", type=int, default=1)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    artifact_path = _path(args.artifact)
    output_path = _path(args.output)
    summary_path = _path(args.summary)
    if not args.force and (
        output_path.exists() or summary_path.exists()
    ):
        raise FileExistsError("D2 Stage A output exists; pass --force")
    plan = export_stage49_h12_action_plan(
        artifact_path.relative_to(ROOT),
        step_index=args.step_index,
    )
    plan.write(output_path)
    reloaded = type(plan).load(output_path)
    ledger = build_d2_phase_ledger(reloaded, embedding_dim=512)
    wave_plan = D2WavePlan.single_rank(
        reloaded,
        job_id="d2-stage-a-theta1-to-theta2-h12",
    )
    wave_plan.validate_against_action_plan(reloaded)
    stage5_requests = wave_plan.to_stage5_requests()
    wave_report = D2WaveReport.from_single_rank_adapter(
        wave_plan,
        ledger.to_dict(),
    )
    checks = {
        "canonical_reload": reloaded == plan,
        "content_hash_stable": (
            reloaded.content_sha256 == plan.content_sha256
        ),
        "record_coverage": (
            tuple(value.record_id for value in stage5_requests)
            == tuple(value.record_id for value in plan.records)
        ),
        "stage5_action_mapping": all(
            request.requested_action
            == (
                "migrate"
                if record.requested_action == "compiled"
                else "exact"
            )
            for request, record in zip(
                stage5_requests,
                plan.records,
                strict=True,
            )
        ),
        "single_owner": all(
            value.old_owner_rank == 0 for value in wave_plan.records
        ),
        "wave_plan_bound_to_action_plan": True,
        "phase_ledger": all(ledger.checks.values()),
        "report_coverage": (
            len(wave_report.coverage_record_ids)
            == plan.counts.records
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D2 Stage A plan checks failed: {checks}")
    summary = {
        "protocol": "cohortkv_d2_stage_a_plan_adapter_v1",
        "status": "complete",
        "scientific_result": False,
        "action_plan": {
            "path": str(output_path.relative_to(ROOT)),
            "content_sha256": plan.content_sha256,
            "file_sha256": file_sha256(output_path),
            "counts": plan.counts.to_dict(),
            "source_version": plan.source_version,
            "target_version": plan.target_version,
        },
        "upstream": plan.provenance.to_dict(),
        "wave_plan": {
            "protocol": wave_plan.protocol,
            "job_id": wave_plan.job_id,
            "world_size": wave_plan.world_size,
            "serving_layout": wave_plan.serving_layout,
            "publication_mode": wave_plan.publication_mode,
            "records": len(wave_plan.records),
        },
        "wave_report": wave_report.to_dict(),
        "scope": {
            "schema_adapter_only": True,
            "stage5_execution_performed": False,
            "transaction_validation_artifact": (
                "configs/cohortkv_d2/"
                "stage_a_stage5_adapter_validation.json"
            ),
        },
        "checks": checks,
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

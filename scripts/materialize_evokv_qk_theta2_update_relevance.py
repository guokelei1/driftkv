from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256

PROTOCOL = "evokv_qk_theta2_update_relevance_round_plan_v0"
FALLBACKS = {
    "theta2_relevance_e1_lr100_n8": (1, 1.5e-5, 1.5e-5, 1.5e-4, 8),
    "theta2_relevance_e2_lr075_n8": (2, 1.125e-5, 1.125e-5, 1.125e-4, 8),
    "theta2_relevance_e2_lr075_n32": (2, 1.125e-5, 1.125e-5, 1.125e-4, 32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--canary-records-per-rank", type=int, default=0)
    return parser.parse_args()


def _write_frozen(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != payload:
            raise FileExistsError(f"frozen QK update relevance config differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _validate_plan(plan_path: Path, plan: dict[str, object]) -> None:
    edge = plan.get("edge")
    source = plan.get("source_checkpoint")
    fallback = plan.get("fallback_training", {}).get("candidates", [])
    observed = {
        value["candidate_name"]: (
            value["epochs"],
            value["dense_learning_rate"],
            value["projection_learning_rate"],
            value["embedding_learning_rate"],
            value["negative_count"],
        )
        for value in fallback
    }
    if (
        plan.get("protocol") != PROTOCOL
        or plan.get("status") != "ready_for_user_execution"
        or plan.get("scientific_result") is not False
        or plan.get("formal_result") is not False
        or edge
        != {
            "source_version": 1,
            "target_version": 2,
            "edge": 2,
            "training_window": 2,
            "evaluation_window": 3,
        }
        or not isinstance(source, dict)
        or source.get("version") != 1
        or observed != FALLBACKS
    ):
        raise ValueError("QK update relevance round plan differs")
    source_directory = Path(source["root"]) / "theta_1"
    if (
        file_sha256(source_directory / "manifest.json") != source["manifest_sha256"]
        or file_sha256(source_directory / "training_state.json")
        != source["training_state_sha256"]
        or file_sha256(source_directory / "optimizer_resume.pt")
        != source["optimizer_resume_sha256"]
        or file_sha256(Path(plan["data"]["config"])) != plan["data"]["config_sha256"]
        or file_sha256(Path(plan["data"]["corpus"])) != plan["data"]["corpus_sha256"]
    ):
        raise ValueError(f"QK update relevance frozen input differs: {plan_path}")
    for candidate in plan["existing_candidates"]:
        manifest = Path(candidate["checkpoint_root"]) / "theta_2" / "manifest.json"
        if (
            file_sha256(manifest) != candidate["manifest_sha256"]
            or file_sha256(Path(candidate["training_result"]))
            != candidate["training_result_sha256"]
            or file_sha256(Path(candidate["previous_alignment_result"]))
            != candidate["previous_alignment_result_sha256"]
        ):
            raise ValueError(
                f"QK retained theta2 input differs: {candidate['candidate_name']}"
            )


def _common(plan: dict[str, object], candidate: str) -> dict[str, object]:
    return {
        "status": "ready_for_user_execution",
        "scientific_result": False,
        "formal_result": False,
        "candidate": candidate,
        "edge": {
            "source_version": 1,
            "target_version": 2,
            "edge": 2,
            "training_window": 2,
            "evaluation_window": 3,
        },
        "source_checkpoint": {
            "root": plan["source_checkpoint"]["root"],
            "version": 1,
            "manifest_sha256": plan["source_checkpoint"]["manifest_sha256"],
        },
        "data": {
            key: plan["data"][key]
            for key in ("config", "config_sha256", "roles", "corpus", "corpus_sha256", "summary")
        },
    }


def _relevance_config(
    plan: dict[str, object],
    candidate: str,
    checkpoint_root: Path,
    manifest_sha256: str,
    origin: dict[str, object],
) -> dict[str, object]:
    round_root = Path(plan["outputs"]["round_root"]) / "candidates" / candidate
    return {
        "protocol": "evokv_qk_update_relevance_evaluation_v0",
        "round_id": f"{plan['round_id']}_{candidate}_relevance",
        **_common(plan, candidate),
        "current_checkpoint": {
            "root": str(checkpoint_root),
            "version": 2,
            "manifest_sha256": manifest_sha256,
        },
        "origin": origin,
        "quality": deepcopy(plan["relevance_quality"]),
        "execution": {
            "world_size": 2,
            "cuda_visible_devices": "0,1",
            "snapshot_batch_size_per_rank": plan["execution"]["snapshot_batch_size_per_rank"],
            "progress_every_records": plan["execution"]["progress_every_records"],
            "record_limit_per_rank": plan["execution"]["record_limit_per_rank"],
            "minimum_free_hbm_bytes_per_rank": plan["execution"]["minimum_free_hbm_bytes_per_rank"],
            "minimum_free_disk_bytes": plan["execution"]["minimum_free_disk_bytes"],
        },
        "outputs": {
            "round_root": str(round_root / "relevance"),
            "result": str(round_root / "relevance" / "result.json"),
        },
    }


def _write_relevance(
    path: Path,
    config: dict[str, object],
    canary_records_per_rank: int,
) -> list[Path]:
    _write_frozen(path, config)
    written = [path]
    if canary_records_per_rank:
        canary = deepcopy(config)
        canary["round_id"] = f"{config['round_id']}_canary{canary_records_per_rank}"
        canary["quality"]["bootstrap_samples"] = 20
        canary["quality"]["minimum_cohort_targets"] = 1
        canary["execution"]["record_limit_per_rank"] = canary_records_per_rank
        canary_root = path.parent / f"canary_{canary_records_per_rank}"
        canary["outputs"] = {
            "round_root": str(canary_root),
            "result": str(canary_root / "result.json"),
        }
        canary_path = canary_root / "frozen_config.json"
        _write_frozen(canary_path, canary)
        written.append(canary_path)
    return written


def materialize(
    plan_path: Path,
    only_candidate: str | None,
    canary_records_per_rank: int,
) -> dict[str, object]:
    plan = json.loads(plan_path.read_text())
    _validate_plan(plan_path, plan)
    if canary_records_per_rank < 0:
        raise ValueError("QK update relevance canary size differs")
    round_root = Path(plan["outputs"]["round_root"])
    checkpoint_parent = Path(plan["outputs"]["checkpoint_parent"])
    written = []
    known = {
        value["candidate_name"] for value in plan["existing_candidates"]
    } | set(FALLBACKS)
    if only_candidate is not None and only_candidate not in known:
        raise ValueError(f"unknown QK update relevance candidate: {only_candidate}")
    for candidate in plan["existing_candidates"]:
        name = candidate["candidate_name"]
        if only_candidate is not None and name != only_candidate:
            continue
        config = _relevance_config(
            plan,
            name,
            Path(candidate["checkpoint_root"]),
            candidate["manifest_sha256"],
            {
                "kind": "retained_route_a_candidate",
                "training_result": candidate["training_result"],
                "training_result_sha256": candidate["training_result_sha256"],
                "previous_alignment_result": candidate["previous_alignment_result"],
                "previous_alignment_result_sha256": candidate[
                    "previous_alignment_result_sha256"
                ],
            },
        )
        path = round_root / "candidates" / name / "relevance" / "frozen_config.json"
        written.extend(_write_relevance(path, config, canary_records_per_rank))
    for candidate in plan["fallback_training"]["candidates"]:
        name = candidate["candidate_name"]
        if only_candidate is not None and name != only_candidate:
            continue
        candidate_root = round_root / "candidates" / name
        work_root = checkpoint_parent / f".{plan['round_id']}_{name}_work"
        training = {
            "objective": plan["fallback_training"]["objective"],
            "negative_count": candidate["negative_count"],
            "weight_decay": plan["fallback_training"]["weight_decay"],
            "seed": plan["fallback_training"]["seed"],
            "negative_seed": plan["fallback_training"]["negative_seed"],
            "optimizer_continuity": plan["fallback_training"]["optimizer_continuity"],
            "dense_learning_rate": candidate["dense_learning_rate"],
            "projection_learning_rate": candidate["projection_learning_rate"],
            "embedding_learning_rate": candidate["embedding_learning_rate"],
            "epochs": candidate["epochs"],
        }
        training_config = {
            "protocol": "evokv_qk_stream_full_catalog_tuning_v1",
            "round_id": f"{plan['round_id']}_{name}_training",
            **_common(plan, name),
            "edge": {
                "source_version": 1,
                "target_version": 2,
                "edge": 2,
                "candidate_name": name,
            },
            "exploration": {
                "plan": str(plan_path),
                "plan_sha256": file_sha256(plan_path),
                "axis": (
                    "ranking_negative_scale"
                    if candidate["negative_count"] != 8
                    else "lower_update_strength"
                ),
            },
            "training": training,
            "quality": {
                **deepcopy(plan["fit_quality"]),
                "qualification_consumed": False,
                "final_consumed": False,
            },
            "execution": {
                "world_size": 2,
                "cuda_visible_devices": "0,1",
                "batch_size_per_rank": plan["execution"]["batch_size_per_rank"],
                "length_bucket_records": plan["execution"]["length_bucket_records"],
                "snapshot_batch_size_per_rank": plan["execution"]["snapshot_batch_size_per_rank"],
                "progress_every_steps": plan["execution"]["progress_every_steps"],
                "quality_progress_every_records": plan["execution"]["progress_every_records"],
                "minimum_free_hbm_bytes_per_rank": plan["execution"]["minimum_free_hbm_bytes_per_rank"],
                "minimum_free_disk_bytes": plan["execution"]["minimum_free_disk_bytes"],
                "retain_provisional_checkpoint": True,
            },
            "outputs": {
                "checkpoint_root": str(checkpoint_parent),
                "work_checkpoint_root": str(work_root),
                "round_root": str(candidate_root / "training"),
                "result": str(candidate_root / "training" / "result.json"),
                "metric_table_json": str(candidate_root / "training" / "reuse_recompute_metrics.json"),
                "metric_table_csv": str(candidate_root / "training" / "reuse_recompute_metrics.csv"),
                "metric_table_markdown": str(candidate_root / "training" / "reuse_recompute_metrics.md"),
            },
        }
        training_path = candidate_root / "training" / "frozen_config.json"
        _write_frozen(training_path, training_config)
        written.append(training_path)
        manifest = work_root / "theta_2" / "manifest.json"
        training_result = candidate_root / "training" / "result.json"
        if manifest.is_file() and training_result.is_file():
            relevance = _relevance_config(
                plan,
                name,
                work_root,
                file_sha256(manifest),
                {
                    "kind": "fallback_training_candidate",
                    "training_config": str(training_path),
                    "training_config_sha256": file_sha256(training_path),
                    "training_result": str(training_result),
                    "training_result_sha256": file_sha256(training_result),
                },
            )
            relevance_path = candidate_root / "relevance" / "frozen_config.json"
            written.extend(
                _write_relevance(
                    relevance_path,
                    relevance,
                    canary_records_per_rank,
                )
            )
    return {
        "status": "pass",
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "configs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in written
        ],
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            materialize(
                args.plan,
                args.candidate,
                args.canary_records_per_rank,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

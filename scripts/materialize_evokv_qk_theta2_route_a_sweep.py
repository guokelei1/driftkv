from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256

PROTOCOL = "evokv_qk_theta2_route_a_sweep_plan_v0"
EXPECTED_CANDIDATES = {
    "theta2_route_a_e3_lr100": (3, 1.5e-5, 1.5e-5, 1.5e-4),
    "theta2_route_a_e4_lr100": (4, 1.5e-5, 1.5e-5, 1.5e-4),
    "theta2_route_a_e3_lr150": (3, 2.25e-5, 2.25e-5, 2.25e-4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def _write_frozen(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != payload:
            raise FileExistsError(f"frozen theta2 config differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _validate_plan(path: Path, plan: dict[str, object]) -> None:
    edge = plan.get("edge")
    source = plan.get("source_checkpoint")
    candidates = plan.get("candidates")
    if (
        plan.get("protocol") != PROTOCOL
        or plan.get("status") != "ready_for_user_execution"
        or plan.get("scientific_result") is not False
        or plan.get("formal_result") is not False
        or not isinstance(edge, dict)
        or edge.get("source_version") != 1
        or edge.get("target_version") != 2
        or edge.get("edge") != 2
        or edge.get("training_window") != 2
        or edge.get("evaluation_window") != 3
        or not isinstance(source, dict)
        or source.get("version") != 1
        or not isinstance(candidates, list)
        or len(candidates) != len(EXPECTED_CANDIDATES)
    ):
        raise ValueError("QK theta2 sweep plan differs")
    observed = {}
    for candidate in candidates:
        name = candidate.get("candidate_name")
        observed[name] = (
            candidate.get("epochs"),
            candidate.get("dense_learning_rate"),
            candidate.get("projection_learning_rate"),
            candidate.get("embedding_learning_rate"),
        )
        if not Path(str(candidate.get("entry"))).is_file():
            raise FileNotFoundError(f"theta2 entry is absent: {name}")
    if observed != EXPECTED_CANDIDATES:
        raise ValueError("QK theta2 candidate matrix differs")
    freeze = Path(plan["route"]["source_freeze"])
    source_root = Path(source["root"])
    directory = source_root / "theta_1"
    if (
        file_sha256(freeze) != plan["route"]["source_freeze_sha256"]
        or file_sha256(directory / "manifest.json")
        != source["manifest_sha256"]
        or file_sha256(directory / "training_state.json")
        != source["training_state_sha256"]
        or file_sha256(directory / "optimizer_resume.pt")
        != source["optimizer_resume_sha256"]
        or file_sha256(Path(plan["data"]["config"]))
        != plan["data"]["config_sha256"]
        or file_sha256(Path(plan["data"]["corpus"]))
        != plan["data"]["corpus_sha256"]
    ):
        raise ValueError(f"QK theta2 frozen input differs: {path}")


def materialize(plan_path: Path) -> list[Path]:
    plan = json.loads(plan_path.read_text())
    _validate_plan(plan_path, plan)
    result_parent = Path(plan["outputs"]["result_parent"])
    checkpoint_parent = Path(plan["outputs"]["checkpoint_parent"])
    execution = plan["execution"]
    paths = []
    for candidate in plan["candidates"]:
        name = candidate["candidate_name"]
        round_id = candidate["round_id"]
        round_root = result_parent / round_id
        work_root = checkpoint_parent / f".{round_id}_work"
        training = {
            **deepcopy(plan["training"]),
            "dense_learning_rate": candidate["dense_learning_rate"],
            "projection_learning_rate": candidate[
                "projection_learning_rate"
            ],
            "embedding_learning_rate": candidate[
                "embedding_learning_rate"
            ],
            "epochs": candidate["epochs"],
        }
        quality = {
            **deepcopy(plan["quality"]),
            "qualification_consumed": False,
            "final_consumed": False,
        }
        config = {
            "protocol": "evokv_qk_stream_full_catalog_tuning_v1",
            "status": "ready_for_user_execution",
            "round_id": round_id,
            "scientific_result": False,
            "formal_result": False,
            "exploration": {
                "route": "A",
                "sweep_plan": str(plan_path),
                "sweep_plan_sha256": file_sha256(plan_path),
                "candidate": name,
                "training_window": 2,
                "evaluation_window": 3,
            },
            "edge": {
                "source_version": 1,
                "target_version": 2,
                "edge": 2,
                "candidate_name": name,
            },
            "source_checkpoint": {
                "root": plan["source_checkpoint"]["root"],
                "version": 1,
                "manifest_sha256": plan["source_checkpoint"][
                    "manifest_sha256"
                ],
            },
            "data": {
                key: plan["data"][key]
                for key in (
                    "config",
                    "config_sha256",
                    "roles",
                    "corpus",
                    "corpus_sha256",
                    "summary",
                )
            },
            "training": training,
            "quality": quality,
            "post_training_evaluation": deepcopy(
                plan["post_training_evaluation"]
            ),
            "execution": {
                key: execution[key]
                for key in (
                    "world_size",
                    "cuda_visible_devices",
                    "batch_size_per_rank",
                    "length_bucket_records",
                    "snapshot_batch_size_per_rank",
                    "progress_every_steps",
                    "quality_progress_every_records",
                    "minimum_free_hbm_bytes_per_rank",
                    "minimum_free_disk_bytes",
                )
            }
            | {
                "estimated_wall_minutes": execution[
                    "estimated_wall_minutes_per_candidate"
                ],
                "retain_provisional_checkpoint": True,
            },
            "outputs": {
                "checkpoint_root": str(checkpoint_parent),
                "work_checkpoint_root": str(work_root),
                "round_root": str(round_root),
                "result": str(round_root / "training" / "result.json"),
                "metric_table_json": str(
                    round_root / "training" / "reuse_recompute_metrics.json"
                ),
                "metric_table_csv": str(
                    round_root / "training" / "reuse_recompute_metrics.csv"
                ),
                "metric_table_markdown": str(
                    round_root / "training" / "reuse_recompute_metrics.md"
                ),
                "alignment_round_root": str(round_root / "alignment"),
                "protocol_sweep_round_root": str(
                    round_root / "protocol_sweep"
                ),
                "summary_json": str(round_root / "summary.json"),
                "summary_markdown": str(round_root / "summary.md"),
            },
        }
        config_path = round_root / "frozen_training_config.json"
        _write_frozen(config_path, config)
        paths.append(config_path)
    return paths


def main() -> None:
    paths = materialize(parse_args().plan)
    print(
        json.dumps(
            {
                "status": "pass",
                "configs": [
                    {"path": str(path), "sha256": file_sha256(path)}
                    for path in paths
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

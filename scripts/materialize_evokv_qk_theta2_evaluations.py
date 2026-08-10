from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256

EXPECTED_CANDIDATES = {
    "theta2_route_a_e3_lr100",
    "theta2_route_a_e4_lr100",
    "theta2_route_a_e3_lr150",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    return parser.parse_args()


def _write_frozen(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != payload:
            raise FileExistsError(f"frozen theta2 evaluation differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def materialize(
    training_config: Path,
) -> tuple[Path, Path]:
    document = json.loads(training_config.read_text())
    edge = document.get("edge")
    candidate = edge.get("candidate_name") if isinstance(edge, dict) else None
    if (
        document.get("protocol")
        != "evokv_qk_stream_full_catalog_tuning_v1"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(edge, dict)
        or edge.get("source_version") != 1
        or edge.get("target_version") != 2
        or edge.get("edge") != 2
        or candidate not in EXPECTED_CANDIDATES
    ):
        raise ValueError("QK theta2 training config differs")
    outputs = document["outputs"]
    training_result = Path(outputs["result"])
    result = json.loads(training_result.read_text())
    checkpoint_root = Path(outputs["work_checkpoint_root"])
    manifest = checkpoint_root / "theta_2" / "manifest.json"
    manifest_sha256 = file_sha256(manifest)
    if (
        result.get("status") != "complete_tuning_measurement"
        or result.get("checkpoint", {}).get("path")
        != str(checkpoint_root / "theta_2")
        or result.get("checkpoint", {}).get("manifest_sha256")
        != manifest_sha256
    ):
        raise ValueError("QK theta2 checkpoint binding differs")
    common = {
        "status": "ready_for_user_execution",
        "scientific_result": False,
        "formal_result": False,
        "edge": {
            "source_version": 1,
            "target_version": 2,
            "edge": 2,
            "training_window": 2,
            "evaluation_window": 3,
        },
        "source_checkpoint": deepcopy(document["source_checkpoint"]),
        "current_checkpoint": {
            "root": str(checkpoint_root),
            "version": 2,
            "manifest_sha256": manifest_sha256,
        },
        "data": {
            key: document["data"][key]
            for key in ("corpus", "corpus_sha256", "summary", "roles")
        },
        "exploration": {
            "route": "A",
            "candidate": candidate,
            "training_config": str(training_config),
            "training_config_sha256": file_sha256(training_config),
            "training_result": str(training_result),
            "training_result_sha256": file_sha256(training_result),
        },
    }
    execution = {
        "world_size": 2,
        "cuda_visible_devices": "0,1",
        "snapshot_batch_size_per_rank": int(
            document["execution"]["snapshot_batch_size_per_rank"]
        ),
        "progress_every_records": int(
            document["execution"]["quality_progress_every_records"]
        ),
        "minimum_free_hbm_bytes_per_rank": int(
            document["execution"]["minimum_free_hbm_bytes_per_rank"]
        ),
        "minimum_free_disk_bytes": int(
            document["execution"]["minimum_free_disk_bytes"]
        ),
    }
    alignment_quality = deepcopy(
        document["post_training_evaluation"][
            "update_local_full_catalog_alignment"
        ]
    )
    alignment_protocol = alignment_quality.pop("protocol")
    alignment_root = Path(outputs["alignment_round_root"])
    alignment = {
        "protocol": alignment_protocol,
        "round_id": f"{document['round_id']}_alignment",
        **common,
        "quality": alignment_quality,
        "execution": {**execution, "estimated_wall_minutes": [10, 25]},
        "outputs": {
            "round_root": str(alignment_root),
            "result": str(alignment_root / "result.json"),
        },
    }
    protocol_quality = deepcopy(
        document["post_training_evaluation"]["candidate_protocol_sweep"]
    )
    protocol_name = protocol_quality.pop("protocol")
    protocol_root = Path(outputs["protocol_sweep_round_root"])
    protocol = {
        "protocol": protocol_name,
        "round_id": f"{document['round_id']}_protocol_sweep",
        **common,
        "quality": protocol_quality,
        "execution": {**execution, "estimated_wall_minutes": [8, 20]},
        "outputs": {
            "round_root": str(protocol_root),
            "result": str(protocol_root / "result.json"),
        },
    }
    alignment_path = alignment_root / "frozen_config.json"
    protocol_path = protocol_root / "frozen_config.json"
    _write_frozen(alignment_path, alignment)
    _write_frozen(protocol_path, protocol)
    return alignment_path, protocol_path


def main() -> None:
    alignment, protocol = materialize(parse_args().training_config)
    print(
        json.dumps(
            {
                "status": "pass",
                "alignment_config": {
                    "path": str(alignment),
                    "sha256": file_sha256(alignment),
                },
                "protocol_config": {
                    "path": str(protocol),
                    "sha256": file_sha256(protocol),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

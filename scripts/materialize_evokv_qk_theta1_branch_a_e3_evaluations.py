from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--alignment-config", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    return parser.parse_args()


def _write_frozen(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text() != payload:
            raise FileExistsError(f"frozen evaluation config differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def materialize(
    training_config: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    document = json.loads(training_config.read_text())
    if (
        document.get("round_id")
        != "qk_theta1_branch_a_e3_lr100_20260806_round1"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("edge", {}).get("candidate_name")
        != "theta1_branch_a_e3_lr100"
    ):
        raise ValueError("QK branch-A training config differs")
    outputs = document["outputs"]
    result = json.loads(Path(outputs["result"]).read_text())
    checkpoint_root = Path(outputs["work_checkpoint_root"])
    manifest = checkpoint_root / "theta_1" / "manifest.json"
    manifest_sha256 = file_sha256(manifest)
    if (
        result.get("status") != "complete_tuning_measurement"
        or result.get("checkpoint", {}).get("path")
        != str(checkpoint_root / "theta_1")
        or result.get("checkpoint", {}).get("manifest_sha256")
        != manifest_sha256
    ):
        raise ValueError("QK branch-A checkpoint binding differs")
    source = deepcopy(document["source_checkpoint"])
    data = document["data"]
    current = {
        "root": str(checkpoint_root),
        "version": 1,
        "manifest_sha256": manifest_sha256,
    }
    common = {
        "status": "ready_for_user_execution",
        "scientific_result": False,
        "formal_result": False,
        "edge": {
            "source_version": 0,
            "target_version": 1,
            "edge": 1,
            "training_window": 1,
            "evaluation_window": 2,
        },
        "source_checkpoint": source,
        "current_checkpoint": current,
        "data": {
            "corpus": data["corpus"],
            "corpus_sha256": data["corpus_sha256"],
            "summary": data["summary"],
            "roles": data["roles"],
        },
        "exploration": {
            "branch": "A",
            "candidate": "theta1_branch_a_e3_lr100",
            "training_config": str(training_config),
            "training_config_sha256": file_sha256(training_config),
            "training_result": outputs["result"],
            "training_result_sha256": file_sha256(outputs["result"]),
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
        "execution": {
            **execution,
            "estimated_wall_minutes": [10, 25],
        },
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
        "execution": {
            **execution,
            "estimated_wall_minutes": [8, 20],
        },
        "outputs": {
            "round_root": str(protocol_root),
            "result": str(protocol_root / "result.json"),
        },
    }
    return alignment, protocol


def main() -> None:
    args = parse_args()
    alignment, protocol = materialize(args.training_config)
    _write_frozen(args.alignment_config, alignment)
    _write_frozen(args.protocol_config, protocol)
    print(
        json.dumps(
            {
                "status": "pass",
                "alignment_config": str(args.alignment_config),
                "alignment_config_sha256": file_sha256(
                    args.alignment_config
                ),
                "protocol_config": str(args.protocol_config),
                "protocol_config_sha256": file_sha256(args.protocol_config),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from hstu_kvcache.migration import (
    D2ActionPlan,
    characterize_d2_capacity,
)
from hstu_kvcache.migration.design2_plan import file_sha256
from hstu_kvcache.migration.stage45_oldkv import (
    load_direct_oldkv_program,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_PLAN = (
    "configs/cohortkv_d2/"
    "action_plan_theta1_theta2_staggered_renewal_h12.json"
)
DEFAULT_TRAINING = (
    "results/motivation_scale/"
    "long_context_4plus12_training_exploration_seed0.json"
)
DEFAULT_CHECKPOINT = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/theta_2.pt"
)
DEFAULT_PROGRAM = (
    "checkpoints/kuairand_long_context_4plus12_exploration/"
    "seed0/single_config_v1/stage4_7_organic_runtime/"
    "theta1_to_theta2_direct_oldkv_fp16.pt"
)
DEFAULT_OUTPUT = (
    "configs/cohortkv_d2/stage_a_capacity_characterization.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-plan", default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--training-result", default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _gpu_inventory() -> list[dict[str, object]]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,"
            "memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    output = []
    for line in query.strip().splitlines():
        index, uuid, bus_id, name, total, used, free = (
            value.strip() for value in line.split(",")
        )
        output.append(
            {
                "index": int(index),
                "uuid": uuid,
                "pci_bus_id": bus_id,
                "name": name,
                "nvidia_smi_total_mib": int(total),
                "nvidia_smi_used_mib": int(used),
                "nvidia_smi_free_mib": int(free),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    action_plan_path = _path(args.action_plan)
    training_path = _path(args.training_result)
    checkpoint_path = _path(args.checkpoint)
    program_path = _path(args.program)
    output_path = _path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            "D2 Stage A capacity output exists; pass --force"
        )
    plan = D2ActionPlan.load(action_plan_path)
    artifact = json.loads(
        _path(plan.provenance.artifact).read_text()
    )
    step = artifact["steps"][plan.provenance.step_index]
    checkpoint_descriptor = next(
        value
        for value in artifact["input_provenance"]["checkpoints"]
        if value["version"] == plan.target_version
    )
    if file_sha256(checkpoint_path) != checkpoint_descriptor["sha256"]:
        raise ValueError("D2 capacity checkpoint hash differs")
    expected_program_sha256 = str(step["program"]["sha256"])
    program, program_metadata = load_direct_oldkv_program(
        program_path,
        expected_sha256=expected_program_sha256,
        expected_source_version=plan.source_version,
        expected_target_version=plan.target_version,
        expected_num_layers=16,
        expected_kv_width=512,
    )
    training = json.loads(training_path.read_text())
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model_bytes = sum(
        value.numel() * value.element_size()
        for value in state.values()
    )
    item_embedding = state["item_emb.weight"]
    item_embedding_bytes = (
        item_embedding.numel() * item_embedding.element_size()
    )
    inventory = _gpu_inventory()
    if not inventory or not torch.cuda.is_available():
        raise RuntimeError("D2 Stage A capacity requires CUDA inventory")
    capacity_bytes = min(
        torch.cuda.get_device_properties(value["index"]).total_memory
        for value in inventory
    )
    result = characterize_d2_capacity(
        plan,
        model_bytes=model_bytes,
        item_embedding_bytes=item_embedding_bytes,
        program_bytes=program.nbytes,
        capacity_bytes=capacity_bytes,
    )
    expected_old = int(step["memory"]["previous_actual_kv_bytes"])
    expected_new = int(step["memory"]["next_actual_kv_bytes"])
    checks = {
        "action_plan_artifact_hash": (
            file_sha256(_path(plan.provenance.artifact))
            == plan.provenance.artifact_sha256
        ),
        "checkpoint_target": (
            plan.target_version == "theta2"
            and checkpoint_descriptor["path"]
            == str(checkpoint_path.relative_to(ROOT))
        ),
        "checkpoint_hash": (
            file_sha256(checkpoint_path)
            == checkpoint_descriptor["sha256"]
        ),
        "model_tensor_bytes": model_bytes == 724328448,
        "item_embedding_tensor_bytes": (
            item_embedding_bytes == 639272960
        ),
        "program_hash": (
            program_metadata["sha256"]
            == expected_program_sha256
        ),
        "program_tensor_bytes": program.nbytes == 33587200,
        "old_kv_bytes": (
            result["cohort"]["old_kv_bytes"] == expected_old
        ),
        "new_kv_bytes": (
            result["cohort"]["complete_new_kv_bytes"]
            == expected_new
        ),
        "one_rank_strict_cow_rejected": not next(
            value
            for value in result["layouts"]
            if value["world_size"] == 1
            and value["owner_strategy"] == "modulo"
        )["all_full_model_total_capacity_admitted"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"D2 Stage A capacity checks failed: {checks}"
        )
    result.update(
        {
            "status": "complete",
            "action_plan": {
                "path": str(action_plan_path.relative_to(ROOT)),
                "content_sha256": plan.content_sha256,
                "file_sha256": file_sha256(action_plan_path),
            },
            "training_result": {
                "path": str(training_path.relative_to(ROOT)),
                "sha256": file_sha256(training_path),
                "model": training["model"],
            },
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": checkpoint_descriptor["sha256"],
            },
            "program": {
                **program_metadata,
                "path": str(program_path.relative_to(ROOT)),
                "tensor_bytes": program.nbytes,
            },
            "gpu_inventory_at_characterization": inventory,
            "checks": checks,
        }
    )
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

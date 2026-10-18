#!/usr/bin/env python3
"""CPU inspection of checkpoint payload, retained runtime copies and cost sums.

No models, GPU allocation or inference are needed. Runtime payload follows
StructuredEncoder's actual effective_weights/stack layout and the frozen
static4096 IDs; allocator overhead and per-call scratch are separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import torch

from code_structure_2026_10.cost import LAYOUTS, actual_calibration, price
from code_structure_2026_10.structure import StructuredKVAdapter
from code_structure_2026_10.session import ITEM_TABLES


def read(path):
    return json.loads(Path(path).read_text())


def tensor_bytes(tensors):
    """Count unique underlying storages, including any retained view bases."""
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def inspect(directory, layout):
    directory = directory.resolve()
    calibration = directory / layout / "calibration"
    recorded = read(directory / layout / "cost.json")[layout]
    recalculated = price(layout, actual_calibration(calibration))
    compared = ("shared_calibration_flops", "maintenance_flops", "extra_query_flops",
        "complete_added_flops", "complete_percent_of_once_rebuild",
        "complete_with_source_reconstruction_percent")
    for key in compared:
        if recorded[key] != recalculated[key]:
            raise RuntimeError(f"completed cost differs: {layout}/{key}")
    by_release = {}
    for version in ("v3", "v4", "v5"):
        path = calibration / version / "calibration.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = read(path.with_suffix(".json"))
        state = payload["state_dict"]
        adapter = StructuredKVAdapter(**payload["config"])
        adapter.load_state_dict(state)
        parameters = tensor_bytes(adapter.parameters())
        buffers = tensor_bytes(adapter.buffers())
        categories = defaultdict(int)
        for name, tensor in adapter.named_buffers():
            category = "access_indices" if name.endswith("access_indices") else "calibration_coordinate_buffers"
            categories[category] += tensor.numel() * tensor.element_size()
        checkpoint = tensor_bytes(state.values())
        if parameters + buffers != checkpoint:
            raise RuntimeError("unexpected checkpoint storage aliases")
        weights = adapter.effective_weights()
        # This reproduces the persistent tensor transforms in StructuredEncoder.
        kv = weights["kv_weights"]
        item = weights["item_weight"].permute(1, 0, 2).reshape(192, -1).contiguous()
        bias = weights["bias"].reshape(-1)
        encoder_copies = tensor_bytes([*kv, item, bias])
        with np.load(ITEM_TABLES) as tables:
            static_ids = np.unique(tables[f"{version}_4096"])
        slots = {int(item_id): index for index, item_id in enumerate(static_ids)}
        slot_bytes = sys.getsizeof(slots) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in slots.items())
        table = len(static_ids) * item.shape[1] * item.element_size()
        runtime = checkpoint + encoder_copies + table
        stage = metadata["cost"]["stage_components"]
        local = sum(stage["inherited_local_calibration"].values())
        joint = sum(stage["replacement_joint"].values())
        if local + joint != metadata["cost"]["calibration_flops"]:
            raise RuntimeError("calibration stage sums differ")
        by_release[version] = dict(
            checkpoint=str(path.relative_to(ROOT)),
            checkpoint_tensor_payload_bytes=checkpoint,
            learned_parameter_payload_bytes=parameters,
            buffer_payload_bytes=buffers,
            buffer_categories_bytes=dict(categories),
            checkpoint_file_bytes=path.stat().st_size,
            structured_encoder_additional_tensor_copies_bytes=encoder_copies,
            encoder_copy_components_bytes=dict(kv_branch_stacks=tensor_bytes(kv),
                item_projection_copy=tensor_bytes([item]), bias_stack=tensor_bytes([bias])),
            static_table_payload_bytes=table,
            static_id_numpy_payload_bytes=static_ids.nbytes,
            python_slot_dictionary_bytes_estimate=slot_bytes,
            current_runtime_shared_tensor_payload_bytes_per_worker=runtime,
            current_runtime_shared_tensor_payload_mib_per_worker=runtime / 2**20,
            current_runtime_shared_tensor_payload_bytes_four_workers=4 * runtime,
            local_calibration_flops=local, replacement_joint_flops=joint,
            complete_calibration_flops=local + joint,
            selected_joint_epoch=metadata["diagnostics"]["selected_epoch"],
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    offline = sum(row["current_runtime_shared_tensor_payload_bytes_per_worker"] for row in by_release.values())
    return dict(layout=layout, by_release=by_release,
        verified_cost_fields={key: recalculated[key] for key in compared},
        retained_offline_three_release_shared_tensor_payload_bytes_per_worker=offline,
        retained_offline_three_release_shared_tensor_payload_mib_per_worker=offline / 2**20,
        retained_offline_three_release_shared_tensor_payload_bytes_four_workers=4 * offline,
        per_user_retained_row_payload_bytes=dict(native_fp32_kv=9216, code=1152, support_flags=6),
        codes_and_flags_percent_of_native_kv=100 * 1158 / 9216)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/code_structure_2026_10")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = dict(status="complete", device="cpu", inspected_runtime="StructuredEncoder with full StructuredKVAdapter retained",
        layouts={layout: inspect(args.results, layout) for layout in LAYOUTS if layout != "headlocal8"},
        notes=[
            "Checkpoint tensor payload includes weights and all persisted calibration-coordinate/access buffers; its file size additionally includes serialization metadata.",
            "The current evaluator retains full adapters, their buffers, stacked/copied effective encoder parameters and static item tables. Report this runtime sum, not learned weights alone.",
            "All quantities above are shared release/worker payload. Calibration coordinate buffers and access tables are not per-user cache state.",
            "Offline user-major replay retains three releases. A single Current serving release uses its own corresponding shared payload; global release retirement is not implemented here.",
            "The inspection loads CPU checkpoint copies solely for analysis. Evaluation's last CPU payload variable and foundation model memory are separate from the GPU method payload reported here.",
            "Transient per-call code gathers/item projections, constructor scratch, CPU Branch dictionaries, allocator overhead and loaded-serving peaks are not included in persistent tensor payload."])
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({layout: dict(
        checkpoint_payload_bytes=data["by_release"]["v3"]["checkpoint_tensor_payload_bytes"],
        runtime_per_release_worker_mib=data["by_release"]["v3"]["current_runtime_shared_tensor_payload_mib_per_worker"],
        offline_three_release_worker_mib=data["retained_offline_three_release_shared_tensor_payload_mib_per_worker"],
        complete_cost_percent=data["verified_cost_fields"]["complete_percent_of_once_rebuild"])
        for layout, data in report["layouts"].items()}, indent=2))


if __name__ == "__main__":
    main()

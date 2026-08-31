#!/usr/bin/env python3
"""Label-free correctness canary for the no-materialized-prefix PRO reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_pro_lazy_reader_v1.yaml"
DEFAULT_OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/correctness_cost"

import sys

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from broadcast_residual import generate_av_broadcast_residual  # noqa: E402
from evaluate_yambda500m_foundation_raw import DAY, load_histories, load_model  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402
from one_release_refinement import (  # noqa: E402
    build_broadcast_probe_source_cache,
    cast_prefix,
    parameter_cast_maps,
)
from pro_lazy_reader import (  # noqa: E402
    build_parent_conditioned_carriers,
    generate_lazy_pro_sidecar,
)


EDGES = (
    ("v0_to_v1", 231, 0, 1),
    ("v1_to_v2", 245, 1, 2),
    ("v2_to_v3", 259, 2, 3),
    ("v3_to_v4", 273, 3, 4),
    ("v4_to_v5", 287, 4, 5),
)
KNOWN_ITEMS = 781_678


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(version: int) -> Path:
    if version == 0:
        return ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"
    return (
        ROOT
        / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
        / f"train_14d/checkpoints/v{version}/checkpoint_100.pt"
    )


def verify_contract(contract_path: Path) -> tuple[dict, str]:
    contract = yaml.safe_load(contract_path.read_text())
    if contract["scope"]["request_labels"] != "prohibited":
        raise RuntimeError("PRO correctness canary must prohibit labels")
    for record in contract["frozen_inputs"].values():
        if "path" in record:
            path = ROOT / record["path"]
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"frozen input differs: {path}")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version) or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs")
    return contract, sha256(contract_path)


def prefix_batch(history, uids: list[int], cutover: int, device: torch.device):
    rows = []
    for uid in uids:
        timestamps, items, behaviors = history.rows[uid]
        stop = int(np.searchsorted(timestamps, cutover, side="left"))
        if stop < 512:
            raise RuntimeError(f"uid {uid} lacks a full pre-cutover history")
        rows.append(
            (
                timestamps[stop - 512 : stop],
                items[stop - 512 : stop],
                behaviors[stop - 512 : stop],
            )
        )
    times = torch.tensor(
        np.stack([row[0] for row in rows]), dtype=torch.long, device=device
    )
    items = torch.tensor(
        np.stack([row[1] for row in rows]), dtype=torch.long, device=device
    )
    behaviors = torch.tensor(
        np.stack([row[2] for row in rows]), dtype=torch.long, device=device
    )
    deltas = torch.zeros_like(times, dtype=torch.float32)
    deltas[:, 1:] = times[:, 1:] - times[:, :-1]
    return items, behaviors, deltas


def padded_reference(
    mapped_old: HSTUKVCache,
    carriers: HSTUKVCache,
    nominal: int,
) -> HSTUKVCache:
    padding = nominal - mapped_old.seq_len - carriers.seq_len
    if padding < 0:
        raise ValueError("reference components exceed nominal cache length")
    zero_k = mapped_old.k.new_zeros(
        mapped_old.k.shape[0], mapped_old.k.shape[1], padding, mapped_old.k.shape[-1]
    )
    zero_v = mapped_old.v.new_zeros(
        mapped_old.v.shape[0], mapped_old.v.shape[1], padding, mapped_old.v.shape[-1]
    )
    return HSTUKVCache(
        k=torch.cat([zero_k, mapped_old.k, carriers.k], dim=2),
        v=torch.cat([zero_v, mapped_old.v, carriers.v], dim=2),
        seq_len=nominal,
    )


def flatten(corrections: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([value.float().flatten(1) for value in corrections], dim=1)


def comparison_rows(
    *,
    uids: list[int],
    edge: str,
    carriers: int,
    lazy,
    reference,
    legacy,
) -> list[dict]:
    lazy_flat = flatten(lazy.corrections)
    reference_flat = flatten(reference.corrections)
    legacy_flat = flatten(legacy.corrections)
    exact_abs = (lazy_flat - reference_flat).abs().amax(dim=1)
    exact_rel = (lazy_flat - reference_flat).norm(dim=1) / reference_flat.norm(dim=1).clamp_min(1e-12)
    legacy_cosine = F.cosine_similarity(lazy_flat, legacy_flat, dim=1)
    legacy_norm_ratio = lazy_flat.norm(dim=1) / legacy_flat.norm(dim=1).clamp_min(1e-12)
    legacy_rel = (lazy_flat - legacy_flat).norm(dim=1) / legacy_flat.norm(dim=1).clamp_min(1e-12)
    layer_rel = []
    for actual, expected in zip(lazy.corrections, legacy.corrections, strict=True):
        layer_rel.append(
            (actual.float() - expected.float()).flatten(1).norm(dim=1)
            / expected.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        )
    layer_rel = torch.stack(layer_rel, dim=1)
    rows = []
    for index, uid in enumerate(uids):
        rows.append(
            {
                "uid": int(uid),
                "edge": edge,
                "carriers": carriers,
                "represented_mass": 128 // carriers,
                "labels_read": False,
                "materialized_version_translated_prefix_positions_in_action": 0,
                "fused_reference_max_abs_error": float(exact_abs[index]),
                "fused_reference_relative_l2": float(exact_rel[index]),
                "fused_replay_max_abs_error": lazy.replay_max_abs_error,
                "legacy_sidecar_direction_cosine": float(legacy_cosine[index]),
                "legacy_sidecar_norm_ratio": float(legacy_norm_ratio[index]),
                "legacy_sidecar_relative_l2": float(legacy_rel[index]),
                "legacy_layer_relative_l2": [float(value) for value in layer_rel[index]],
            }
        )
    return rows


@torch.inference_mode()
def evaluate_edge(
    *,
    edge: str,
    cutover_day: int,
    parent,
    current,
    history,
    uids: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], float]:
    if parent.cfg != current.cfg:
        raise RuntimeError("Parent and Current configurations differ")
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    started = time.perf_counter()
    maps = parameter_cast_maps(parent, current)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    map_seconds = time.perf_counter() - started
    output = []
    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start : start + batch_size]
        items, behaviors, deltas = prefix_batch(
            history, batch_uids, cutover_day * DAY, device
        )
        parent_cache = parent.compute_kv(items, behaviors, deltas)
        legacy_source, legacy_layout = build_broadcast_probe_source_cache(
            parent_cache=parent_cache,
            current=current,
            item_ids=items,
            behaviors=behaviors,
            time_deltas=deltas,
            cast_maps=maps,
        )
        if legacy_layout.carriers != 32:
            raise RuntimeError("legacy sidecar reference no longer has 32 carriers")
        probe_items = items[:, -1]
        legacy = generate_av_broadcast_residual(
            current, legacy_source, parent_cache, probe_items
        )
        mapped_old = cast_prefix(parent_cache, maps, length=384)
        for carrier_count in (16, 32):
            carrier_cache, layout = build_parent_conditioned_carriers(
                parent_cache=parent_cache,
                current=current,
                item_ids=items,
                behaviors=behaviors,
                time_deltas=deltas,
                repair_width=128,
                carrier_count=carrier_count,
            )
            if layout.old_positions != 384:
                raise RuntimeError("lightweight PRO old boundary differs")
            lazy = generate_lazy_pro_sidecar(
                current,
                parent_cache,
                carrier_cache,
                maps,
                probe_items,
                old_positions=layout.old_positions,
            )
            reference_cache = padded_reference(mapped_old, carrier_cache, nominal=512)
            reference = generate_av_broadcast_residual(
                current, reference_cache, parent_cache, probe_items
            )
            output.extend(
                comparison_rows(
                    uids=batch_uids,
                    edge=edge,
                    carriers=carrier_count,
                    lazy=lazy,
                    reference=reference,
                    legacy=legacy,
                )
            )
    return output, map_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("the prospective correctness canary freezes batch size 8")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    contract, contract_hash = verify_contract(args.contract)
    population_path = ROOT / contract["frozen_inputs"]["population"]["path"]
    population = pd.read_parquet(population_path).sort_values(["selector_rank", "uid"])
    population_offset = int(contract["scope"].get("population_offset", 0))
    uids = [
        int(value)
        for value in population.iloc[population_offset : population_offset + 32].uid
    ]
    if len(uids) != 32:
        raise RuntimeError("PRO correctness canary requires 32 fixed users")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    first_model, first_payload = load_model(checkpoint(0), device)
    oov_buckets = int(first_payload["config"]["num_items"]) - KNOWN_ITEMS
    history = load_histories(uids, oov_buckets=oov_buckets)
    del first_model, first_payload
    torch.cuda.empty_cache() if device.type == "cuda" else None

    all_rows = []
    map_times = {}
    for edge, cutover, parent_version, current_version in EDGES:
        parent, parent_payload = load_model(checkpoint(parent_version), device)
        current, current_payload = load_model(checkpoint(current_version), device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError(f"{edge} Parent/Current configuration differs")
        rows, map_seconds = evaluate_edge(
            edge=edge,
            cutover_day=cutover,
            parent=parent,
            current=current,
            history=history,
            uids=uids,
            device=device,
            batch_size=args.batch_size,
        )
        all_rows.extend(rows)
        map_times[edge] = map_seconds
        del parent, current, parent_payload, current_payload
        torch.cuda.empty_cache() if device.type == "cuda" else None

    args.output.mkdir(parents=True)
    raw = args.output / "raw.jsonl"
    with raw.open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    seal = {
        "status": "pro_lazy_reader_correctness_raw_sealed",
        "contract_sha256": contract_hash,
        "raw_sha256": sha256(raw),
        "rows": len(all_rows),
        "users_per_edge": len(uids),
        "population_offset": population_offset,
        "edges": len(EDGES),
        "carrier_axis": [16, 32],
        "labels_read": False,
        "parameter_map_build_seconds_by_edge": map_times,
        "materialized_version_translated_prefix_positions_in_action": 0,
        "reference_only_materialized_prefix_used_for_equivalence": True,
    }
    (args.output / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    main()

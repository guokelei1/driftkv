#!/usr/bin/env python3
"""Shared prospective configuration for the frozen 8L EvoKV reproduction."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import P7Request, load_p7_requests
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_v1.yaml"
P7_MANIFEST = ROOT / "data/manifests/p7_full_v1"
P8_MANIFEST = ROOT / "data/manifests/p8_release_v1"
RAW = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
BASE_ROOT = ROOT / "results/p7/base_fit/frozen_base_bundle_v1"
OUTPUT = ROOT / "results/scale_8l_v1"

MODELS = {"m0_f": ("F",), "m1": ("N", "R", "F")}
SEEDS = (17, 37, 71)
RELEASES = ("r0", "r1_edge1", "r1_edge2", "r2")
CONTEXT = 1024
LAYERS = 8
HIDDEN = 256
HEADS = 8
EPOCHS = 3
LOGICAL_BATCH = 8


def sha256_file(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            output.update(block)
    return output.hexdigest()


def contract() -> dict:
    value = yaml.safe_load(CONTRACT.read_text())
    checks = {
        "p10_full_stack_freeze_sha256": ROOT / "configs/contracts/p10_6_full_stack_freeze_v1.yaml",
        "p11_recursive_quality_contract_sha256": ROOT / "configs/contracts/p11_4_recursive_policy_quality_v1.yaml",
        "p7_residual_train_manifest_sha256": P7_MANIFEST / "residual_train/manifest.index.json",
        "p7_development_manifest_sha256": P7_MANIFEST / "development/manifest.index.json",
        "p8_materialization_summary_sha256": P8_MANIFEST / "materialization_summary.json",
        "frozen_base_bundle_sha256": BASE_ROOT / "bundle_manifest.json",
    }
    for key, path in checks.items():
        if sha256_file(path) != value["inputs"][key]:
            raise RuntimeError(f"scale contract input hash mismatch: {key}")
    return value


def make_model(seed: int, device: torch.device) -> HSTU:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = HSTU(
        HSTUConfig(
            num_items=p7.max_item_id(),
            num_behaviors=2,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            num_heads=HEADS,
            max_seq_len=CONTEXT,
            temporal_num_freqs=16,
            temporal_max_period=86_400.0,
            gating="silu_gate",
            activation="elu_plus1",
            input_dropout=0.1,
            attn_dropout=0.0,
            block_variant="legacy",
            relative_position_bias=False,
            causal_diagonal="inclusive",
            num_query_types=3,
            num_query_actions=1,
            query_type_id=0,
            query_action_id=0,
        )
    )
    return model.to(device)


def load_requests(root: Path, split: str, task: str) -> list[P7Request]:
    return load_p7_requests(root, RAW, split, task, history_limit=CONTEXT)


def load_theta0_data(tasks: tuple[str, ...]) -> tuple[dict[str, list[P7Request]], dict[str, list[P7Request]]]:
    train = {task: load_requests(P7_MANIFEST, "residual_train", task) for task in tasks}
    development = {task: load_requests(P7_MANIFEST, "development", task) for task in tasks}
    if any(len(rows) != 3939 for rows in train.values()):
        raise RuntimeError("scale query budget differs from frozen 3939-per-task contract")
    return train, development


def model_metadata(model: HSTU) -> dict:
    parameters = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    return {
        "config": asdict(model.cfg),
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_bytes_fp32": parameters * 4,
    }


def deterministic_subset(rows: list[P7Request], count: int, namespace: str) -> list[P7Request]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{namespace}:{row.request_id}".encode()).digest(),
    )[:count]


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

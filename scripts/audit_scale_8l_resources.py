#!/usr/bin/env python3
"""Static S1 audit for the prospective 8L/H256/context1024 scale point."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

import scale_8l_common as scale
from hstu_kvcache.models import HSTU, HSTUConfig


def request_coverage(split: str, task: str) -> dict:
    index_path = scale.P7_MANIFEST / split / "manifest.index.json"
    index = json.loads(index_path.read_text())
    tables = [pq.read_table(index_path.parent / row["path"]) for row in index["request_shards"]]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    kind = {"N": "quality", "R": "quality_rankable", "F": "quality"}[task]
    table = table.filter(pc.and_(pc.equal(table["workload"], task), pc.equal(table["manifest_kind"], kind)))
    full_available = pc.subtract(table["raw_prefix_end_exclusive"], table["raw_user_row_start"]).to_numpy()
    materialized = table["effective_prefix_length"].to_numpy()
    histogram = Counter(
        "ge1024" if value >= 1024 else "513_to_1023" if value > 512 else "le512"
        for value in full_available
    )
    return {
        "split": split,
        "task": task,
        "queries": len(full_available),
        "sealed_manifest_max": int(materialized.max()),
        "causal_available_max": int(full_available.max()),
        "causal_history_buckets": dict(histogram),
        "fraction_with_more_than_512": float((full_available > 512).mean()),
        "fraction_with_at_least_1024": float((full_available >= 1024).mean()),
        "scale_history_tokens": int(sum(min(1024, int(value)) for value in full_available)),
    }


def model_size() -> dict:
    cfg = HSTUConfig(
        num_items=__import__("train_p7_theta0").max_item_id(), num_behaviors=2,
        hidden_size=256, num_layers=8, num_heads=8, max_seq_len=1024,
        temporal_num_freqs=16, temporal_max_period=86_400.0,
        gating="silu_gate", activation="elu_plus1", input_dropout=0.1,
        attn_dropout=0.0, block_variant="legacy", relative_position_bias=False,
        causal_diagonal="inclusive", num_query_types=3, num_query_actions=1,
        query_type_id=0, query_action_id=0,
    )
    with torch.device("meta"):
        model = HSTU(cfg)
    parts = {
        "item_embedding": sum(p.numel() for p in model.item_emb.parameters()),
        "encoder_blocks": sum(p.numel() for p in model.blocks.parameters()),
        "query_and_head": sum(
            p.numel() for name, p in model.named_parameters()
            if name.startswith(("query_encoder.", "cc_score_head."))
        ),
    }
    total = sum(p.numel() for p in model.parameters())
    return {
        "config": {"layers": 8, "hidden": 256, "heads": 8, "context": 1024},
        "parameters": total,
        "parameter_parts": parts,
        "checkpoint_fp32_bytes": total * 4,
        "adam_training_state_rough_bytes": total * 16,
        "single_state_KV_bytes_fp32": 2 * 8 * 1024 * 256 * 4,
        "single_state_KV_bytes_bf16": 2 * 8 * 1024 * 256 * 2,
        "exact_token_layer_work_per_full_state": 8 * 1024,
        "exact_attention_pairs_per_full_state": 8 * 1024 * 1025 // 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=scale.OUTPUT / "s1_resource_audit.json")
    args = parser.parse_args()
    contract = scale.contract()
    coverage = [request_coverage(split, task) for split in ("residual_train", "development") for task in ("N", "R", "F")]
    payload = {
        "status": "scale_8l_static_resource_audit_complete",
        "contract_sha256": scale.sha256_file(scale.CONTRACT),
        "GPU_allowlist": contract["execution"]["GPU_allowlist"],
        "visible_CPU_threads": os.cpu_count(),
        "model": model_size(),
        "coverage": coverage,
        "gate": {
            "f_has_history_beyond_512": any(row["task"] == "F" and row["fraction_with_more_than_512"] > 0 for row in coverage),
            "frozen_queries_candidates_and_base_features_reused": True,
            "long_training_launched": False,
        },
    }
    scale.json_dump(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


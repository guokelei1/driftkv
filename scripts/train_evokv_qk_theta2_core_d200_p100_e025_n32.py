from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_full_catalog_runner import (
    run_full_catalog_tuning_edge,
)
from hstu_kvcache.streaming.qk_stream_version import QKStreamFullCatalogBinding

BINDING = QKStreamFullCatalogBinding(
    source_version=1,
    target_version=2,
    edge=2,
    candidate_name="theta2_core_d200_p100_e025_n32",
    dense_learning_rate=2.25e-5,
    projection_learning_rate=1.125e-5,
    embedding_learning_rate=2.8125e-5,
    epochs=2,
    train_negative_count=32,
    bootstrap_samples=10_000,
    training_seed=2026080611,
    negative_seed=2026080623,
    bootstrap_seed=2026080651,
    full_catalog_item_chunk=32_768,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = parse_args().config
    result = run_full_catalog_tuning_edge(config, BINDING)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "candidate": BINDING.candidate_name,
                    "result": json.loads(config.read_text())["outputs"]["result"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

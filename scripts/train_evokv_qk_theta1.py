from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_runner import run_edge
from hstu_kvcache.streaming.qk_stream_version import QKStreamEdgeBinding

BINDING = QKStreamEdgeBinding(
    source_version=0,
    target_version=1,
    edge=1,
    candidate_name="theta1_candidate_a_lr015",
    dense_learning_rate=1.5e-5,
    projection_learning_rate=1.5e-5,
    embedding_learning_rate=1.5e-4,
    epochs=1,
    train_negative_count=8,
    quality_negative_count=999,
    quality_epsilon_ce=0.005,
    bootstrap_samples=10_000,
    training_seed=2026080511,
    negative_seed=2026080523,
    quality_seed=2026080537,
    bootstrap_seed=2026080551,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_edge(parse_args().config, BINDING)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

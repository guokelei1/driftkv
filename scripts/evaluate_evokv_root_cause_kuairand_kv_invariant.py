from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_kv_invariant import run_invariant_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_invariant_evaluation(parse_args().config)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "classification": result["decision"]["classification"],
                    "pooled_cross_entropy_retention_percent": result["decision"][
                        "pooled_cross_entropy_update_advantage"
                    ]["retention_percent"],
                    "runtime_seconds": result["execution"]["runtime_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

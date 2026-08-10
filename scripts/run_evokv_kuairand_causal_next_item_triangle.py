from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_next_item_triangle import (
    run_triangle_evaluation,
    run_triangle_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    args = parser.parse_args()
    if args.phase == "train":
        result = run_triangle_training(args.config)
    else:
        result = run_triangle_evaluation(args.config)
    if result is not None:
        print(
            json.dumps(
                {
                    "protocol": result["protocol"],
                    "status": result["status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

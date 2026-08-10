from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_history_path_screen import (
    run_history_path_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_history_path_training(args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidates": [
                    value["candidate"]["id"] for value in result["candidates"]
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


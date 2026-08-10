from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_next_item_hard_update import (
    run_next_item_hard_update_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_next_item_hard_update_training(args.config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidates": [
                    value["candidate"]["candidate_id"] for value in result["candidates"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

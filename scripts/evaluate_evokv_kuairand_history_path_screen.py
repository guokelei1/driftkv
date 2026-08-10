from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_history_path_screen import (
    run_history_path_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_history_path_evaluation(args.config)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "same_user_set": result["same_user_set_across_candidates"],
                    "implementation_passed": result[
                        "all_implementation_checks_passed"
                    ],
                    "candidates": {
                        value["candidate"]["id"]: value["summary"]
                        for value in result["candidates"]
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()


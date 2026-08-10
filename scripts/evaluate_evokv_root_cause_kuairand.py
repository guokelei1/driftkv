from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_root_cause import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_evaluation(parse_args().config)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "edges": [value["edge"] for value in result["edges"]],
                    "implementation_passed": result["sanity"]["implementation_passed"],
                    "runtime_seconds": result["execution"]["runtime_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

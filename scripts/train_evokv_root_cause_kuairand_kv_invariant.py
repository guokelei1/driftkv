from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_kv_invariant import run_invariant_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_invariant_training(parse_args().config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "versions": [value["version"] for value in result["versions"]],
                "runtime_seconds": result["execution"]["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

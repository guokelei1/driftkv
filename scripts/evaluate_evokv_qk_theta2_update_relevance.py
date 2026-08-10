from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_update_relevance_runner import (
    run_qk_update_relevance_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = parse_args().config
    result = run_qk_update_relevance_evaluation(config)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "candidate": result["candidate"],
                    "result": json.loads(config.read_text())["outputs"]["result"],
                    "records": result["quality"]["evaluation"]["records"],
                    "primary_admission_status": result["quality"]["evaluation"][
                        "primary_admission_gate"
                    ]["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

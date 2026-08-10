from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_protocol_sweep_runner import (
    run_qk_candidate_protocol_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_qk_candidate_protocol_sweep(parse_args().config)
    if result is not None:
        gate = result["quality"]["primary_update_local"]["stable_gap_gate"]
        print(
            json.dumps(
                {
                    "protocol": result["protocol"],
                    "status": result["status"],
                    "stable_gap_gate": gate,
                    "result": result["config"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

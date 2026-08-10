from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_alignment_runner import (
    run_qk_stream_alignment_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_qk_stream_alignment_diagnostic(parse_args().config)
    if result is not None:
        evaluation = result["quality"]["evaluation"]
        print(
            json.dumps(
                {
                    "protocol": result["protocol"],
                    "status": result["status"],
                    "alignment_gate": evaluation["alignment_gate"],
                    "result": result["config"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

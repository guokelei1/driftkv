from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_root_cause_sanity import (
    run_qk_root_cause_sanity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run_qk_root_cause_sanity(parse_args().config)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "records": result["aggregate"]["records"],
                    "targets": result["aggregate"]["positive_targets"],
                    "implementation_passed": result["aggregate"]["sanity"]["implementation_passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

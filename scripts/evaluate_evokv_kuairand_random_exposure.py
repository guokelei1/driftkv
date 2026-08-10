from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_random_exposure_screen import (
    run_random_exposure_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_random_exposure_screen(args.config)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

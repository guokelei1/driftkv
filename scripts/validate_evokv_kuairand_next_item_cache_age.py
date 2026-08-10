from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_next_item_cache_age_screen import (
    validate_cache_age_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    validate_cache_age_result(json.loads(args.result.read_text()))
    print(f"validated={args.result}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_next_item_cache_age_screen import (
    summarize_cache_age_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_cache_age_shards(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

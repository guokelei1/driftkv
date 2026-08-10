from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_next_item_holdout import (
    load_next_item_holdout_config,
    validate_next_item_holdout_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    load_next_item_holdout_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_next_item_holdout_result(result)
    print(json.dumps({"status": "valid", "version": result["completed_version"]}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_next_item_holdout import (
    run_next_item_holdout_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_next_item_holdout_training(args.config)
    print(json.dumps({"status": result["status"], "version": result["completed_version"]}, indent=2))


if __name__ == "__main__":
    main()

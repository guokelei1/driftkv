from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_next_item_rollout import (
    run_next_item_rollout_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_next_item_rollout_evaluation(args.config)
    if result is not None:
        print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()

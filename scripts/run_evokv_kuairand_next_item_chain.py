from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_next_item_chain import (
    run_next_item_chain_evaluation,
    run_next_item_chain_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    args = parser.parse_args()
    if args.mode == "train":
        result = run_next_item_chain_training(args.config)
        print(json.dumps({"status": result["status"], "version": result["completed_version"]}, indent=2))
    else:
        result = run_next_item_chain_evaluation(args.config)
        if result is not None:
            print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

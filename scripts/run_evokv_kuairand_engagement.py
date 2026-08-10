from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_engagement import run_evaluation, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    training = run_training(args.config)
    if args.training_only:
        print(json.dumps({"training_status": training["status"]}, indent=2, sort_keys=True))
        return
    evaluation = run_evaluation(args.config)
    print(
        json.dumps(
            {
                "training_status": training["status"],
                "decision": evaluation["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

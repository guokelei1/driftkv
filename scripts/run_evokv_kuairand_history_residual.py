from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_history_residual import run_evaluation, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    training = run_training(args.config)
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

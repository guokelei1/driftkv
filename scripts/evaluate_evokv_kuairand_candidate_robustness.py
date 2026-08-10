from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_candidate_robustness import (
    run_candidate_robustness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_candidate_robustness(args.config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "targets": result["targets"],
                "candidate_counts": result["candidate_counts"],
                "rows": len(result["rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

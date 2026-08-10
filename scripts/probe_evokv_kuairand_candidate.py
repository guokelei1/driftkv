from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    run_candidate_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_candidate_probe(
        args.config,
        args.version,
        args.candidate,
        args.output,
        args.candidate_config,
    )
    comparison = result["summary"]["comparisons"]["recompute_over_reuse"]
    print(
        json.dumps(
            {
                "candidate": result["candidate"]["name"],
                "mrr_relative_percent": comparison["mrr"]["relative_percent"],
                "ndcg_at_5_relative_percent": comparison["ndcg_at_5"][
                    "relative_percent"
                ],
                "hit_rate_at_5_relative_percent": comparison["hit_rate_at_5"][
                    "relative_percent"
                ],
                "would_admit": result["would_admit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

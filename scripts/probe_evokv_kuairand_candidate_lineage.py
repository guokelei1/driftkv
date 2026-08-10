from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    run_candidate_lineage_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-config")
    parser.add_argument("--minimum-source-version", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_candidate_lineage_probe(
        args.config,
        args.version,
        args.candidate,
        args.output,
        args.candidate_config,
        args.minimum_source_version,
    )
    print(
        json.dumps(
            {
                "candidate": result["candidate"]["name"],
                "row_gate": result["row_gate"],
                "ndcg_at_5_relative_percent": [
                    {
                        "source_version": row["source_version"],
                        "value": row["summary"]["comparisons"][
                            "recompute_over_reuse"
                        ]["ndcg_at_5"]["relative_percent"],
                    }
                    for row in result["lineage"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

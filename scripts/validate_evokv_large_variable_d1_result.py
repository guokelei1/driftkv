from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_evokv_large_variable_d1 import load_json, validate_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("qk", "qb"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    result = validate_result(
        args.result,
        args.dataset,
        args.config,
        config,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "dataset": args.dataset,
                "result": str(args.result),
                "versions": result["versions"],
                "full_kv_payloads_persisted": result[
                    "full_kv_payloads_persisted"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

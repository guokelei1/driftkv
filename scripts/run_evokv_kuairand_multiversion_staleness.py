from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_multiversion_staleness import run_multiversion_staleness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_multiversion_staleness(args.config)
    print(
        json.dumps(
            {
                "current_version": result["current_version"],
                "source_versions": [lag["source_version"] for lag in result["lags"]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    run_persistent_lineage_prefix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--final-version", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_persistent_lineage_prefix(
        args.config,
        args.final_version,
        args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

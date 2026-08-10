from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_capacity_lift import (
    preflight_capacity_lift,
    run_capacity_lift,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after-version", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-source-hashes", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight_capacity_lift(
            args.config,
            args.stop_after_version,
            args.verify_source_hashes,
        )
    else:
        result = run_capacity_lift(
            args.config,
            args.stop_after_version,
            args.verify_source_hashes,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

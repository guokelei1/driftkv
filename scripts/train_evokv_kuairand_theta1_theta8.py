from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    preflight_persistent_chain,
    run_persistent_chain,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-version", type=int)
    parser.add_argument("--candidate-priority")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight_persistent_chain(args.config)
    else:
        result = run_persistent_chain(
            args.config,
            stop_after_version=args.stop_after_version,
            candidate_priority=args.candidate_priority,
        )
    print(json.dumps(result.get("decision", result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

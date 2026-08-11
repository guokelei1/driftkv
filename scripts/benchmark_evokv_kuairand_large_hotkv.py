from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_hotkv_timing import (
    preflight_timing,
    run_timing,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("canary", "full"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-selection-preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight_timing(
            args.config,
            args.profile,
            build_selections=not args.skip_selection_preflight,
        )
    else:
        result = run_timing(args.config, args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

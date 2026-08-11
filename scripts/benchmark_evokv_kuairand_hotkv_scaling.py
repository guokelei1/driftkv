from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_hotkv_scaling import (
    preflight_scaling,
    run_scaling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("canary", "full"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = (
        preflight_scaling(args.config, args.profile)
        if args.preflight_only
        else run_scaling(args.config, args.profile)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

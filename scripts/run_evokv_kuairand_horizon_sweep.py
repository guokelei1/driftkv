from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_horizon_sweep import run_horizon_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_horizon_sweep(args.config)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

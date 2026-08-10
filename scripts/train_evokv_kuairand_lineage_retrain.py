from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_lineage_retrain import run_lineage_retrain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after-version", type=int)
    args = parser.parse_args()
    result = run_lineage_retrain(args.config, args.stop_after_version)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

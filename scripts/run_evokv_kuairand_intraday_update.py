from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_intraday_update import run_intraday_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_intraday_update(args.config)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_scale import run_projected_chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_projected_chain(args.config)
    if int(__import__("os").environ.get("RANK", "0")) == 0:
        print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_context_length_sweep import run_context_length_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_context_length_sweep(args.config)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

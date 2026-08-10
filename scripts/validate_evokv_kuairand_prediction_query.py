from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_query_transition import load_config, validate_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    document = load_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_summary(result, document)
    print(json.dumps({"status": "valid", "result": args.result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

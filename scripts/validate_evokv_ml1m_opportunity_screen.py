from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.ml1m_opportunity import load_config, validate_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    document = load_config(args.config)
    summary = json.loads(open(args.result).read())
    validate_summary(summary, document)
    print(json.dumps({"status": "valid", "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()

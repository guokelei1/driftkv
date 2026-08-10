from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.ml1m_query_objective import (
    load_query_config,
    validate_query_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    document = load_query_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_query_summary(result, document)
    print(json.dumps({"status": "valid", "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

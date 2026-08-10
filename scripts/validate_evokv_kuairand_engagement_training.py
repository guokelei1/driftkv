from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_engagement import (
    load_engagement_config,
    validate_training_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    document = load_engagement_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_training_result(result, document)
    print(json.dumps({"status": "valid", "training_status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()

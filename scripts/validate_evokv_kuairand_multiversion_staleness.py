from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_multiversion_staleness import (
    load_multiversion_config,
    validate_multiversion_staleness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    load_multiversion_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_multiversion_staleness(result)
    print(json.dumps({"status": "valid", "lags": len(result["lags"])}, indent=2))


if __name__ == "__main__":
    main()

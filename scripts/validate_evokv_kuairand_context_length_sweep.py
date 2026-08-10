from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_context_length_sweep import (
    load_context_sweep_config,
    validate_context_length_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    load_context_sweep_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_context_length_sweep(result)
    print(json.dumps({"status": "valid", "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

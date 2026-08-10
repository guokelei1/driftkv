from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_parameter_group_sweep import (
    load_sweep_config,
    validate_parameter_group_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    load_sweep_config(args.config)
    result = json.loads(Path(args.result).read_text())
    validate_parameter_group_sweep(result)
    print(json.dumps({"status": "valid", "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

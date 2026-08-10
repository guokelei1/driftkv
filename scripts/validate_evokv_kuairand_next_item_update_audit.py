from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_next_item_update_audit import (
    validate_next_item_update_audit_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.result).read_text())
    validate_next_item_update_audit_result(result)
    print(json.dumps({"status": "valid"}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_kv_only_candidate_triangle import (
    run_candidate_triangle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_candidate_triangle(parser.parse_args().config)
    if result is not None:
        print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

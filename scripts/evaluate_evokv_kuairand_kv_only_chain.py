from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_kv_only_chain import (
    run_kv_only_chain_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_kv_only_chain_evaluation(args.config)
    if result is not None:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "cells": len(result["cells"]),
                    "decision": result["decision"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()


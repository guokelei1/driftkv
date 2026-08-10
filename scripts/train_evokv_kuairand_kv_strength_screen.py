from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_kv_strength_screen import (
    run_kv_strength_training,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_kv_strength_training(parser.parse_args().config)
    print(
        json.dumps(
            {"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

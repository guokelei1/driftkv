from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_kv_strength_screen import (
    run_kv_strength_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_kv_strength_evaluation(parser.parse_args().config)
    if result is not None:
        print(
            json.dumps(
                {"status": result["status"], "selection": result["selection"]},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

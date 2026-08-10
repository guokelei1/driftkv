from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_kv_only_logged_window import (
    run_logged_window_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_logged_window_screen(parser.parse_args().config)
    if result is not None:
        print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

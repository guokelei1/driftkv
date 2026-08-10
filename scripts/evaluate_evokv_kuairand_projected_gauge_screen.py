from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_gauge_screen import (
    run_projected_gauge_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_projected_gauge_screen(parser.parse_args().config)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

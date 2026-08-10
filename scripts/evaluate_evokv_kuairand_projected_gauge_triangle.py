from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_gauge_triangle import (
    run_projected_gauge_triangle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    result = run_projected_gauge_triangle(parser.parse_args().config)
    if result is not None:
        print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

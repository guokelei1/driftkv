from __future__ import annotations

import argparse
import json

from hstu_kvcache.streaming.kuairand_projected_persistent import (
    render_persistent_reuse_loss_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render_persistent_reuse_loss_table(args.result),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_theta0 import (
    QKTheta0CorpusConfig,
    build_canary_from_fixed_edges,
    build_qk_theta0_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--canary-fixed-edge", type=Path)
    parser.add_argument("--canary-records", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.config.read_text())
    data = document["data"]
    if args.canary_fixed_edge is not None:
        result = build_canary_from_fixed_edges(
            args.canary_fixed_edge,
            Path(data["corpus"]),
            Path(data["corpus_summary"]),
            maximum_records=args.canary_records,
        )
    else:
        result = build_qk_theta0_corpus(
            QKTheta0CorpusConfig(
                source=Path(data["source"]),
                member=str(data["member"]),
                catalog=Path(data["catalog"]),
                user_lengths=Path(data["user_lengths"]),
                cache_dir=Path(data["builder_cache"]),
                output=Path(data["corpus"]),
                summary=Path(data["corpus_summary"]),
                base_prefix=int(data["base_prefix"]),
                prediction_rows=int(document["model"]["num_prediction_items"]),
                representative_users=int(data["representative_users"]),
                minimum_eligible_rows=int(data["minimum_optimizer_active_rows"]),
                eligible_row_margin=int(data.get("eligible_row_margin", 0)),
                selection_seed=int(data["selection_seed"]),
                chunk_size=int(data["chunk_size"]),
                checkpoint_every_chunks=int(data["checkpoint_every_chunks"]),
                derive_user_block=int(data["derive_user_block"]),
                refresh=args.refresh,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

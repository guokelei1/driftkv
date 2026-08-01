from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_base_row_coverage import (
    CoverageConfig,
    run,
)

DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QK-video.csv"
DEFAULT_CATALOG = Path(
    "data/processed/evokv_d3_m1_qk_entity_cache/"
    "entity_catalog_base64_top250000.npz"
)
DEFAULT_USER_LENGTHS = Path(
    "data/processed/evokv_foundation/qk_full_user_lengths.npz"
)
DEFAULT_CACHE_DIR = Path(
    "data/processed/evokv_foundation/qk_base_row_coverage_cache"
)
DEFAULT_OUTPUT = Path(
    "data/processed/evokv_foundation/"
    "qk_xp_base_row_cooccurrence.npz"
)
DEFAULT_SUMMARY = Path(
    "configs/evokv_foundation/"
    "qk_xp_base_row_cooccurrence_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--user-length-cache",
        type=Path,
        default=DEFAULT_USER_LENGTHS,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument(
        "--checkpoint-every-chunks",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--derive-user-block",
        type=int,
        default=100_000,
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CoverageConfig:
    return CoverageConfig(
        source=args.source,
        member=args.member,
        catalog_cache=args.catalog_cache,
        user_length_cache=args.user_length_cache,
        cache_dir=args.cache_dir,
        output=args.output,
        summary=args.summary,
        base_prefix=args.base_prefix,
        chunk_size=args.chunk_size,
        checkpoint_every_chunks=args.checkpoint_every_chunks,
        derive_user_block=args.derive_user_block,
        refresh=args.refresh,
    )


def main() -> None:
    args = parse_args()
    result = run(config_from_args(args), audit_only=args.audit_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

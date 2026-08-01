from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_xp_edge_inputs import EdgeInputConfig, run

DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QK-video.csv"
DEFAULT_CATALOG = Path(
    "data/processed/evokv_d3_m1_qk_entity_cache/"
    "entity_catalog_base64_top250000.npz"
)
DEFAULT_ROLES = Path(
    "configs/evokv_foundation/qk_post_base_roles.json"
)
DEFAULT_OUTPUT = Path(
    "data/processed/evokv_foundation/qk_xp_fixed_edge_inputs.npz"
)
DEFAULT_SUMMARY = Path(
    "configs/evokv_foundation/qk_xp_fixed_edge_inputs_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--hash-salt",
        default="evokv-qk-successor-foundation-v1",
    )
    parser.add_argument("--prediction-catalog-size", type=int, default=250_000)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--theta01-history-end", type=int, default=64)
    parser.add_argument("--theta01-update-end", type=int, default=96)
    parser.add_argument("--theta12-history-end", type=int, default=544)
    parser.add_argument("--theta12-update-end", type=int, default=576)
    parser.add_argument(
        "--qualification-history-end",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--qualification-update-end",
        type=int,
        default=96,
    )
    parser.add_argument("--theta01-users", type=int, default=2_560)
    parser.add_argument("--theta12-users", type=int, default=2_048)
    parser.add_argument("--qualification-users", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> EdgeInputConfig:
    return EdgeInputConfig(
        source=args.source,
        member=args.member,
        catalog_cache=args.catalog_cache,
        roles=args.roles,
        output=args.output,
        summary=args.summary,
        hash_salt=args.hash_salt,
        prediction_catalog_size=args.prediction_catalog_size,
        base_prefix=args.base_prefix,
        theta01_history_end=args.theta01_history_end,
        theta01_update_end=args.theta01_update_end,
        theta12_history_end=args.theta12_history_end,
        theta12_update_end=args.theta12_update_end,
        qualification_history_end=args.qualification_history_end,
        qualification_update_end=args.qualification_update_end,
        theta01_users=args.theta01_users,
        theta12_users=args.theta12_users,
        qualification_users=args.qualification_users,
        chunk_size=args.chunk_size,
    )


def main() -> None:
    summary = run(config_from_args(parse_args()))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

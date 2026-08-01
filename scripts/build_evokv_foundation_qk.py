from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.migration.foundation_workload import (
    FoundationConfig,
    run,
)

DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QK-video.csv"
DEFAULT_CATALOG = Path(
    "data/processed/evokv_d3_m1_qk_entity_cache/"
    "entity_catalog_base64_top250000.npz"
)
DEFAULT_LENGTH_CACHE = Path(
    "data/processed/evokv_foundation/qk_full_user_lengths.npz"
)
DEFAULT_UPSTREAM_PREPARED = Path(
    "data/processed/evokv_d3_m1_qk_entity_2560.npz"
)
DEFAULT_OUTPUT = Path(
    "data/processed/evokv_foundation/x_qk_het_foundation.npz"
)
DEFAULT_SUMMARY = Path(
    "configs/evokv_foundation/qk_foundation_summary.json"
)
DEFAULT_ROLES = Path(
    "configs/evokv_foundation/qk_post_base_roles.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--length-cache",
        type=Path,
        default=DEFAULT_LENGTH_CACHE,
    )
    parser.add_argument(
        "--upstream-prepared",
        type=Path,
        default=DEFAULT_UPSTREAM_PREPARED,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument(
        "--hash-salt",
        default="evokv-qk-successor-foundation-v1",
    )
    parser.add_argument("--theta12-users", type=int, default=2_048)
    parser.add_argument("--theta01-users", type=int, default=2_560)
    parser.add_argument("--fit-users", type=int, default=512)
    parser.add_argument("--profile-users", type=int, default=512)
    parser.add_argument("--qualification-users", type=int, default=512)
    parser.add_argument("--final-users", type=int, default=65_536)
    parser.add_argument(
        "--capacity-gib",
        type=float,
        nargs="+",
        default=[36, 72, 144, 288, 576, 720],
    )
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--refresh-lengths", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> FoundationConfig:
    return FoundationConfig(
        source=args.source,
        member=args.member,
        catalog_cache=args.catalog_cache,
        length_cache=args.length_cache,
        upstream_prepared=args.upstream_prepared,
        output=args.output,
        summary=args.summary,
        roles=args.roles,
        hash_salt=args.hash_salt,
        theta12_users=args.theta12_users,
        theta01_users=args.theta01_users,
        fit_users=args.fit_users,
        profile_users=args.profile_users,
        qualification_users=args.qualification_users,
        final_users=args.final_users,
        capacity_gib=tuple(args.capacity_gib),
        chunk_size=args.chunk_size,
        refresh_lengths=args.refresh_lengths,
    )


def main() -> None:
    args = parse_args()
    result = run(config_from_args(args), audit_only=args.audit_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

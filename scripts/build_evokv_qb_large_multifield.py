from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hstu_kvcache.data.qb_large_multifield import (
    PROFILE_DEFINITIONS,
    PROTOCOL,
    atomic_json,
    build_catalog,
    file_sha256,
    load_qb_frame,
    materialize_corpus,
    profile_from_name,
    save_catalog,
    save_corpus,
    source_identity,
)

DEFAULT_SOURCE = Path("data/tenrec/Tenrec.zip")
DEFAULT_MEMBER = "Tenrec/QB-video.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFINITIONS), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--required-horizon", type=int, default=104)
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--train-users", type=int, default=3500)
    parser.add_argument("--tuning-users", type=int, default=500)
    parser.add_argument("--qualification-users", type=int, default=1000)
    parser.add_argument("--role-salt", default="evokv-qb-large-multifield-v0")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path("data/processed/evokv_qb_large_multifield")
    catalog = args.catalog or root / f"{args.profile}_catalog.npz"
    corpus = args.corpus or root / f"{args.profile}_corpus.npz"
    summary = args.summary or Path("configs/evokv_foundation") / (
        f"qb_large_{args.profile}_summary_development_v0.json"
    )
    return catalog, corpus, summary


def main() -> None:
    args = parse_args()
    if (
        min(
            args.base_prefix,
            args.required_horizon,
            args.users,
            args.train_users,
            args.tuning_users,
            args.qualification_users,
        )
        < 1
        or args.required_horizon <= args.base_prefix
    ):
        raise ValueError("QB large builder arguments are invalid")
    catalog_path, corpus_path, summary_path = resolve_outputs(args)
    existing = [str(path) for path in (catalog_path, corpus_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(f"QB large outputs already exist: {existing}")
    started = time.perf_counter()
    frame = load_qb_frame(args.source, args.member)
    profile = profile_from_name(args.profile)
    catalog = build_catalog(frame, profile, args.base_prefix)
    catalog_descriptor = save_catalog(catalog_path, catalog)
    catalog = type(catalog)(
        profile=catalog.profile,
        keys=catalog.keys,
        offsets=catalog.offsets,
        metadata={**catalog.metadata, "content_sha256": catalog_descriptor["content_sha256"]},
    )
    arrays, metadata = materialize_corpus(
        frame,
        catalog,
        base_prefix=args.base_prefix,
        required_horizon=args.required_horizon,
        users=args.users,
        train_users=args.train_users,
        tuning_users=args.tuning_users,
        qualification_users=args.qualification_users,
        role_salt=args.role_salt,
    )
    corpus_descriptor = save_corpus(corpus_path, arrays, metadata)
    summary = {
        "protocol": PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "status": "pass",
        "source": source_identity(args.source, args.member),
        "profile": {
            "name": profile.name,
            "fields": list(profile.fields),
            "feature_count": profile.feature_count,
            "embedding_width": profile.embedding_width,
            "semantic_rows": catalog.semantic_rows,
            "num_embeddings": catalog.num_embeddings,
            "num_prediction_items": catalog.num_prediction_items,
        },
        "catalog": catalog_descriptor,
        "corpus": corpus_descriptor,
        "corpus_metadata": metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "source_code": {
            "builder": {
                "path": str(Path(__file__)),
                "sha256": file_sha256(Path(__file__)),
            },
            "module": {
                "path": "src/hstu_kvcache/data/qb_large_multifield.py",
                "sha256": file_sha256(Path("src/hstu_kvcache/data/qb_large_multifield.py")),
            },
        },
    }
    atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "pass",
                "profile": args.profile,
                "catalog": str(catalog_path),
                "corpus": str(corpus_path),
                "summary": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

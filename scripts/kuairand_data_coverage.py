from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hstu_kvcache.utils import save_json

STANDARD_LOGS = [
    "data/kuairand/log_standard_4_08_to_4_21_1k.csv",
    "data/kuairand/log_standard_4_22_to_5_08_1k.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-days", type=int, default=14)
    parser.add_argument("--output", default="results/scaling/kuairand_data_coverage.json")
    return parser.parse_args()


def load_frame() -> pd.DataFrame:
    frames = [
        pd.read_csv(path, usecols=["user_id", "video_id", "date"])
        for path in STANDARD_LOGS
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = frame["date"].astype(str)
    return frame


def catalog_summary(
    frame: pd.DataFrame,
    base: pd.DataFrame,
    max_items: int | None,
    sequence_lengths: list[int],
) -> dict:
    item_counts = base["video_id"].value_counts()
    keep_items = item_counts.index if max_items is None else item_counts.head(max_items).index
    kept_base = base[base["video_id"].isin(keep_items)]
    active_counts = kept_base["user_id"].value_counts()
    active_users = active_counts[active_counts >= 5].index
    retained = frame[
        frame["video_id"].isin(keep_items) & frame["user_id"].isin(active_users)
    ]
    base_active = kept_base[kept_base["user_id"].isin(active_users)]
    per_user = base_active["user_id"].value_counts()
    return {
        "max_items": max_items,
        "fitted_items": int(base_active["video_id"].nunique()),
        "active_users": int(len(active_users)),
        "retained_standard_rows": int(len(retained)),
        "retained_standard_row_fraction": float(len(retained) / len(frame)),
        "retained_base_rows": int(len(base_active)),
        "base_tokens_per_epoch": {
            str(length): int(per_user.clip(upper=length).sum())
            for length in sequence_lengths
        },
    }


def main() -> None:
    args = parse_args()
    frame = load_frame()
    dates = sorted(frame["date"].unique())
    base_dates = dates[: args.base_days]
    base = frame[frame["date"].isin(base_dates)]
    catalogs = {
        str(max_items): catalog_summary(frame, base, max_items, [128, 256, 512])
        for max_items in [5000, 10000, 20000, 50000, 100000]
    }
    catalogs["all_base_items"] = catalog_summary(frame, base, None, [128, 256, 512])
    result = {
        "protocol": "kuairand_1k_standard_log_coverage_v1",
        "files": [str(Path(path)) for path in STANDARD_LOGS],
        "base_days": args.base_days,
        "base_dates": base_dates,
        "all_dates": dates,
        "raw": {
            "standard_rows": int(len(frame)),
            "base_rows": int(len(base)),
            "users": int(frame["user_id"].nunique()),
            "items": int(frame["video_id"].nunique()),
            "base_items": int(base["video_id"].nunique()),
        },
        "catalogs": catalogs,
        "scope": {
            "dataset": "KuaiRand-1K standard logs only",
            "random_log_included": False,
            "kuairand_27k_included": False,
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

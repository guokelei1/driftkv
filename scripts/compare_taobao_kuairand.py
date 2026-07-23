from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hstu_kvcache.utils import save_json

STANDARD_LOGS = [
    "data/kuairand/log_standard_4_08_to_4_21_1k.csv",
    "data/kuairand/log_standard_4_22_to_5_08_1k.csv",
]
BEHAVIOR_COLUMNS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taobao-audit", default="results/taobao/data_audit.json")
    parser.add_argument(
        "--output", default="results/taobao/kuairand_matched_comparison.json"
    )
    parser.add_argument("--max-items", type=int, default=50000)
    parser.add_argument("--base-days", type=int, default=14)
    parser.add_argument("--min-base-events", type=int, default=5)
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "min": int(np.min(values)),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": int(np.max(values)),
    }


def load_kuairand() -> pd.DataFrame:
    dtypes = {
        "user_id": "int32",
        "video_id": "int32",
        "date": "int32",
        **{column: "int8" for column in BEHAVIOR_COLUMNS},
    }
    frames = [
        pd.read_csv(path, usecols=list(dtypes), dtype=dtypes) for path in STANDARD_LOGS
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = frame["date"].astype(str)
    return frame


def daily_summary(frame: pd.DataFrame, dates: list[str]) -> list[dict]:
    grouped = frame.groupby("date", sort=True)
    return [
        {
            "date": date,
            "rows": int(len(grouped.get_group(date))),
            "users": int(grouped.get_group(date)["user_id"].nunique()),
        }
        for date in dates
        if date in grouped.groups
    ]


def per_user_summary(frame: pd.DataFrame) -> dict:
    counts = frame.groupby("user_id", sort=False).size().to_numpy()
    active_days = frame.groupby("user_id", sort=False)["date"].nunique().to_numpy()
    return {
        "events": distribution(counts),
        "active_days": distribution(active_days),
        "sequence_length_coverage": {
            str(length): int(np.count_nonzero(counts >= length))
            for length in (128, 256, 512)
        },
    }


def main() -> None:
    args = parse_args()
    with open(args.taobao_audit) as source:
        taobao = json.load(source)
    frame = load_kuairand()
    dates = sorted(frame["date"].unique())
    base_dates = dates[: args.base_days]
    stream_dates = dates[args.base_days :]
    base = frame[frame["date"].isin(base_dates)]
    item_counts = base["video_id"].value_counts()
    keep_items = item_counts.head(args.max_items).index
    catalog_base = base[base["video_id"].isin(keep_items)]
    base_user_counts = catalog_base["user_id"].value_counts()
    keep_users = base_user_counts[base_user_counts >= args.min_base_events].index
    retained = frame[
        frame["video_id"].isin(keep_items) & frame["user_id"].isin(keep_users)
    ]
    retained_base = retained[retained["date"].isin(base_dates)]
    retained_counts = retained.groupby("user_id", sort=False).size()
    retained_base_counts = retained_base.groupby("user_id", sort=False).size()
    aligned = retained_base_counts.index.intersection(retained_counts.index)
    retained_total_values = retained_counts.reindex(aligned).to_numpy()
    retained_base_values = retained_base_counts.reindex(aligned).to_numpy()
    retained_stream_values = retained_total_values - retained_base_values
    window_plan = []
    for model_t in range(1, 6):
        start = (model_t - 1) * 3
        train_dates = stream_dates[start : start + 3]
        eval_index = start + 3
        if len(train_dates) < 3 or eval_index >= len(stream_dates):
            break
        eval_date = stream_dates[eval_index]
        eval_frame = retained[retained["date"] == eval_date]
        window_plan.append(
            {
                "model_t": model_t,
                "train_dates": train_dates,
                "eval_date": eval_date,
                "train_rows": int(retained[retained["date"].isin(train_dates)].shape[0]),
                "eval_rows": int(len(eval_frame)),
                "eval_users": int(eval_frame["user_id"].nunique()),
            }
        )
    positive_columns = [column for column in BEHAVIOR_COLUMNS if column != "is_hate"]
    positive = frame[positive_columns].any(axis=1)
    top50k_taobao = taobao["catalog_filtered_cohorts"]["catalogs"][
        str(args.max_items)
    ]["cohorts"]
    taobao_threshold_128 = top50k_taobao["minimum_base_event_cohorts"]["128"]
    taobao_top1000 = top50k_taobao["top_base_activity_cohorts"]["1000"]
    result = {
        "protocol": "taobao_kuairand_matched_distribution_v1",
        "comparison_rule": {
            "catalog": f"top-{args.max_items} items fitted on base only",
            "user_selection": "base-window interactions only; no future activity used",
            "sequence_order": "chronological per user",
            "evaluation": "next-window evaluation before ingesting that window",
        },
        "kuairand": {
            "source_files": [str(Path(path)) for path in STANDARD_LOGS],
            "raw": {
                "rows": int(len(frame)),
                "users": int(frame["user_id"].nunique()),
                "items": int(frame["video_id"].nunique()),
                "dates": dates,
                "per_user": per_user_summary(frame),
                "daily": daily_summary(frame, dates),
                "behaviors": {
                    column: int(frame[column].sum()) for column in BEHAVIOR_COLUMNS
                },
                "positive_rows": int(positive.sum()),
                "positive_row_fraction": float(positive.mean()),
            },
            "matched_top50k_protocol": {
                "base_dates": base_dates,
                "base_rows": int(len(retained_base)),
                "stream_rows": int(len(retained) - len(retained_base)),
                "users": int(len(aligned)),
                "base_events_per_user": distribution(retained_base_values),
                "stream_events_per_user": distribution(retained_stream_values),
                "all_events_per_user": distribution(retained_total_values),
                "sequence_length_coverage": {
                    str(length): {
                        "base_users": int(
                            np.count_nonzero(retained_base_values >= length)
                        ),
                        "all_days_users": int(
                            np.count_nonzero(retained_total_values >= length)
                        ),
                    }
                    for length in (128, 256, 512)
                },
                "daily": daily_summary(retained, dates),
                "five_update_plan": window_plan,
            },
        },
        "taobao": {
            "source": taobao["source"],
            "raw": {
                "rows": taobao["scan"]["usable_rows_in_official_range"],
                "users": taobao["cardinality"]["users"],
                "items": taobao["cardinality"]["items"],
                "categories": taobao["cardinality"]["categories"],
                "dates": taobao["official_range"]["dates"],
                "per_user": taobao["per_user"],
                "daily": taobao["daily_asia_shanghai"],
                "behaviors": taobao["behaviors"],
            },
            "matched_top50k_protocol": {
                "base_dates": taobao["catalog_filtered_cohorts"][
                    "base_dates_asia_shanghai"
                ],
                "minimum_128_base_event_cohort": taobao_threshold_128,
                "top_1000_by_base_activity": taobao_top1000,
                "all_base_active_users": top50k_taobao["all_base_active_users"],
                "five_update_plan": [
                    {
                        "model_t": model_t,
                        "train_date": taobao["official_range"]["dates"][
                            2 + model_t
                        ],
                        "eval_date": taobao["official_range"]["dates"][
                            3 + model_t
                        ],
                    }
                    for model_t in range(1, 6)
                ],
            },
        },
        "interpretation": {
            "recommended_taobao_cohort": (
                "top-50k base-only catalog and every user with at least 128 retained "
                "base events; keep all 1,785 users"
            ),
            "recommended_primary_sequence_length": 128,
            "not_supported_as_primary": [256, 512],
            "semantic_limit": (
                "KuaiRand standard logs contain impressions including non-engagement; "
                "Taobao contains user actions only and has no unclicked impressions"
            ),
        },
    }
    save_json(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

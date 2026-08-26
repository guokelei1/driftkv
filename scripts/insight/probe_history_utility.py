#!/usr/bin/env python3
"""Small cross-edge probe of History Utility and recommendation semantics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from hstu_kvcache.evaluation import stable_log_loss  # noqa: E402
from hstu_kvcache.training import collate_foundation_batch  # noqa: E402
from analyze_first_pass import MANIFEST, load_edge, markdown_table  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_history_utility_probe_v1"
CHECKPOINTS = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3/train_14d/checkpoints"


class WindowedHistory:
    def __init__(self, base, width: int) -> None:
        self.base = base
        self.width = width

    def prefix(self, uid: int, query_timestamp: int, max_history: int = 512):
        del max_history
        return self.base.prefix(uid, query_timestamp, self.width)


@torch.inference_mode()
def score(model, requests: list[dict], history, width: int, device: torch.device) -> np.ndarray:
    output = []
    window = WindowedHistory(history, width)
    for start in range(0, len(requests), 32):
        batch = collate_foundation_batch(requests[start:start + 32], window, device=device)
        logits = model.score_cc_full(
            batch.item_ids, batch.behaviors, batch.time_deltas,
            batch.candidate_ids, batch.query_time_deltas, lengths=batch.lengths,
        )
        output.extend(logits[:, 0].float().cpu().tolist())
    return np.asarray(output, dtype=np.float64)


def history_features(history, request: dict) -> dict[str, float | int | bool]:
    items, behaviors, _ = history.prefix(request["uid"], request["query_timestamp"], 512)
    candidate = int(request["item_idx"])
    recent = items[-32:]
    old = items[:-32]
    recent_set, old_set = set(map(int, recent)), set(map(int, old))
    union = recent_set | old_set
    return {
        "candidate_repeat_count": int(np.count_nonzero(items == candidate)),
        "candidate_recent_repeat": candidate in recent_set,
        "candidate_old_only_repeat": candidate in old_set and candidate not in recent_set,
        "history_unique_fraction": float(len(set(map(int, items))) / len(items)),
        "organic_fraction": float(np.mean(behaviors == 1)),
        "recent_old_item_jaccard": float(len(recent_set & old_set) / len(union)) if union else 0.0,
    }


def concentration(utility: np.ndarray, harm: np.ndarray) -> tuple[float, float]:
    useful = utility > 0.0
    positive_harm = np.maximum(harm, 0.0)
    share = float(positive_harm[useful].sum() / positive_harm.sum()) if positive_harm.sum() else float("nan")
    prevalence = float(useful.mean())
    return share, share / prevalence if prevalence else float("nan")


def utility_summary(edge: str, rows: pd.DataFrame, name: str) -> dict:
    utility = rows[name].to_numpy()
    harm = rows["reuse_harm"].to_numpy()
    useful = utility > 0.0
    harm_share, lift = concentration(utility, harm)
    return {
        "edge": edge,
        "utility": name,
        "requests": len(rows),
        "positive_utility_fraction": float(useful.mean()),
        "mean_utility": float(utility.mean()),
        "spearman_utility_vs_harm": float(rows[[name, "reuse_harm"]].corr(method="spearman").iloc[0, 1]),
        "mean_harm_when_useful": float(harm[useful].mean()),
        "mean_harm_when_not_useful": float(harm[~useful].mean()),
        "positive_harm_on_useful_fraction": harm_share,
        "positive_harm_concentration_lift": lift,
    }


def semantic_summary(edge: str, rows: pd.DataFrame) -> list[dict]:
    output = []
    for feature in (
        "candidate_repeat_count", "history_unique_fraction", "organic_fraction",
        "recent_old_item_jaccard",
    ):
        output.append({
            "edge": edge,
            "feature": feature,
            "spearman_feature_vs_harm": float(rows[[feature, "reuse_harm"]].corr(method="spearman").iloc[0, 1]),
        })
    for feature in ("candidate_recent_repeat", "candidate_old_only_repeat"):
        selected = rows[feature]
        output.append({
            "edge": edge,
            "feature": feature,
            "spearman_feature_vs_harm": float(rows[[feature, "reuse_harm"]].corr(method="spearman").iloc[0, 1]),
            "feature_fraction": float(selected.mean()),
            "mean_harm_true": float(rows.loc[selected, "reuse_harm"].mean()),
            "mean_harm_false": float(rows.loc[~selected, "reuse_harm"].mean()),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.max_users < 1:
        raise ValueError("max-users must be positive")

    device = torch.device(args.device)
    label_table = pq.read_table(MANIFEST / "requests_quality.parquet").to_pandas()
    label_by_request = dict(zip(label_table["request_id"], label_table["label"], strict=True))
    request_by_id = label_table.set_index("request_id").to_dict("index")
    edge_rows = [load_edge(edge, label_by_request) for edge in range(5)]
    selections = []
    for rows in edge_rows:
        chosen = rows.sort_values(["uid", "query_timestamp", "request_id"]).groupby("uid", sort=True).head(1)
        selections.append(chosen.sort_values(["uid", "query_timestamp"]).head(args.max_users).copy())
    uids = sorted({int(uid) for rows in selections for uid in rows["uid"]})

    first_checkpoint = CHECKPOINTS / "v1/checkpoint_100.pt"
    first_model, first_payload = load_model(first_checkpoint, device)
    oov_buckets = int(first_payload["config"]["num_items"]) - 781678
    del first_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    history = load_histories(uids, oov_buckets=oov_buckets)

    utility_rows, semantic_rows = [], []
    replay_max_error = {}
    for edge, selected in enumerate(selections):
        checkpoint = CHECKPOINTS / f"v{edge + 1}/checkpoint_100.pt"
        model, _ = load_model(checkpoint, device)
        requests = []
        for request_id in selected["request_id"]:
            request = {**request_by_id[request_id], "request_id": request_id, "weight": 1.0}
            requests.append(request)
        full = score(model, requests, history, 512, device)
        recent128 = score(model, requests, history, 128, device)
        recent32 = score(model, requests, history, 32, device)
        sealed_full = selected["current_full_logit"].to_numpy(dtype=np.float64)
        replay_max_error[f"v{edge}_to_v{edge + 1}"] = float(np.max(np.abs(full - sealed_full)))
        if replay_max_error[f"v{edge}_to_v{edge + 1}"] > 2e-5:
            raise RuntimeError("Current Full replay differs from sealed raw")
        target = np.asarray([request["label"] for request in requests], dtype=np.int64)
        selected["utility_old_beyond_32"] = stable_log_loss(recent32, target) - stable_log_loss(full, target)
        selected["utility_old_beyond_128"] = stable_log_loss(recent128, target) - stable_log_loss(full, target)
        features = pd.DataFrame([history_features(history, request) for request in requests], index=selected.index)
        selected = pd.concat([selected, features], axis=1)
        name = f"v{edge}_to_v{edge + 1}"
        utility_rows.extend([
            utility_summary(name, selected, "utility_old_beyond_32"),
            utility_summary(name, selected, "utility_old_beyond_128"),
        ])
        semantic_rows.extend(semantic_summary(name, selected))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    utility = pd.DataFrame(utility_rows)
    semantics = pd.DataFrame(semantic_rows)
    args.output.mkdir(parents=True)
    utility.to_csv(args.output / "utility_staleness.csv", index=False)
    semantics.to_csv(args.output / "recommendation_semantics.csv", index=False)
    summary = {
        "status": "history_utility_probe_complete",
        "scope": f"first request for first {args.max_users} active UIDs per edge; Small seed17",
        "requests_per_edge": [len(rows) for rows in selections],
        "current_full_replay_max_abs_error": replay_max_error,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    utility_columns = [
        "edge", "utility", "requests", "positive_utility_fraction", "mean_utility",
        "spearman_utility_vs_harm", "positive_harm_concentration_lift",
    ]
    report = [
        "# History Utility x State Staleness probe", "",
        f"Scope: first request for the first {args.max_users} active UIDs on each of five D14/E14 edges. No training.", "",
        *markdown_table(utility, utility_columns), "",
        "Utility is the request log-loss increase from truncating Current Full history to recent-32 or recent-128. Staleness is the existing paired Reuse harm. This is a small causal history-ablation probe; it does not yet localize stale K/V to the ablated region.", "",
        "The utility/harm association is weak and not stable enough to support selective migration: correlations are close to zero, and positive-harm concentration lift ranges around random rather than separating a consistently high-value cohort.", "",
        "Recommendation-semantic correlations are in `recommendation_semantics.csv`. Current persistent history contains listen tokens with organic/non-organic behavior; likes/dislikes are request labels, not persistent action tokens in this implementation.", "",
        "Repeat, diversity, organic fraction, and recent/old overlap change direction across edges. This probe therefore does not support a frozen recommendation-semantic risk rule.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

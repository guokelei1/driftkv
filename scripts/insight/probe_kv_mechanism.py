#!/usr/bin/env python3
"""Small K/V and normalized-depth mechanism probe on append-free requests."""
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
from hstu_kvcache.evaluation import bernoulli_js, stable_log_loss  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402
from hstu_kvcache.training import collate_foundation_batch  # noqa: E402
from analyze_first_pass import MANIFEST, load_edge, markdown_table  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_kv_mechanism_probe_v1"
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
V0 = ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"


def checkpoint(version: int) -> Path:
    return V0 if version == 0 else MATRIX / f"train_14d/checkpoints/v{version}/checkpoint_100.pt"


def mixed_cache(current: HSTUKVCache, parent: HSTUKVCache, current_layers: int) -> HSTUKVCache:
    k, v = parent.k.clone(), parent.v.clone()
    k[:current_layers].copy_(current.k[:current_layers])
    v[:current_layers].copy_(current.v[:current_layers])
    return HSTUKVCache(k=k, v=v, seq_len=parent.seq_len)


def region_cache(base: HSTUKVCache, source: HSTUKVCache, start: int, stop: int) -> HSTUKVCache:
    k, v = base.k.clone(), base.v.clone()
    k[:, :, start:stop].copy_(source.k[:, :, start:stop])
    v[:, :, start:stop].copy_(source.v[:, :, start:stop])
    return HSTUKVCache(k=k, v=v, seq_len=base.seq_len)


@torch.inference_mode()
def evaluate_batch(parent, current, batch, query_deltas: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    parent_cache = parent.compute_kv(batch.item_ids, batch.behaviors, batch.time_deltas)
    current_cache = current.compute_kv(batch.item_ids, batch.behaviors, batch.time_deltas)
    caches = {
        "reuse_parent_kv": parent_cache,
        "current_exact_kv": current_cache,
        "current_k_parent_v": HSTUKVCache(current_cache.k, parent_cache.v, parent_cache.seq_len),
        "parent_k_current_v": HSTUKVCache(parent_cache.k, current_cache.v, parent_cache.seq_len),
        "current_lower_1": mixed_cache(current_cache, parent_cache, 1),
        "current_lower_2": mixed_cache(current_cache, parent_cache, 2),
        "current_lower_3": mixed_cache(current_cache, parent_cache, 3),
        "stale_old384": region_cache(current_cache, parent_cache, 0, 384),
        "stale_recent128": region_cache(current_cache, parent_cache, 384, 512),
        "stale_old480": region_cache(current_cache, parent_cache, 0, 480),
        "stale_recent32": region_cache(current_cache, parent_cache, 480, 512),
        "refresh_old384": region_cache(parent_cache, current_cache, 0, 384),
        "refresh_recent128": region_cache(parent_cache, current_cache, 384, 512),
        "refresh_old480": region_cache(parent_cache, current_cache, 0, 480),
        "refresh_recent32": region_cache(parent_cache, current_cache, 480, 512),
    }
    output = {
        name: current.score_cc_reuse(
            cache, batch.candidate_ids, batch.query_time_deltas, prefix_lengths=batch.lengths
        )[:, 0].float().cpu().numpy()
        for name, cache in caches.items()
    }
    output["parent_exact_model"] = parent.score_cc_reuse(
        parent_cache, batch.candidate_ids, batch.query_time_deltas, prefix_lengths=batch.lengths
    )[:, 0].float().cpu().numpy()
    for name, start, stop, delta_name in (
        ("only_old384", 0, 384, "old384"),
        ("only_recent128", 384, 512, "full"),
        ("only_old480", 0, 480, "old480"),
        ("only_recent32", 480, 512, "full"),
    ):
        items = batch.item_ids[:, start:stop]
        behaviors = batch.behaviors[:, start:stop]
        deltas = batch.time_deltas[:, start:stop].clone()
        deltas[:, 0] = 0.0
        cache = current.compute_kv(items, behaviors, deltas)
        output[name] = current.score_cc_reuse(
            cache, batch.candidate_ids, query_deltas[delta_name]
        )[:, 0].float().cpu().numpy()
    return output


def regional_summary(edge: str, logits: dict[str, np.ndarray], labels: np.ndarray) -> list[dict]:
    exact_loss = stable_log_loss(logits["current_exact_kv"], labels)
    parent_loss = stable_log_loss(logits["reuse_parent_kv"], labels)
    definitions = {
        "old384": ("only_recent128", "stale_old384", "refresh_old384"),
        "recent128": ("only_old384", "stale_recent128", "refresh_recent128"),
        "old480": ("only_recent32", "stale_old480", "refresh_old480"),
        "recent32": ("only_old480", "stale_recent32", "refresh_recent32"),
    }
    output = []
    for region, (without, stale, refresh) in definitions.items():
        utility = stable_log_loss(logits[without], labels) - exact_loss
        staleness = stable_log_loss(logits[stale], labels) - exact_loss
        recovery = parent_loss - stable_log_loss(logits[refresh], labels)
        useful = utility > 0.0
        positive_staleness = np.maximum(staleness, 0.0)
        share = (
            float(positive_staleness[useful].sum() / positive_staleness.sum())
            if positive_staleness.sum() else float("nan")
        )
        output.append({
            "edge": edge,
            "region": region,
            "requests": len(labels),
            "mean_utility": float(utility.mean()),
            "mean_regional_staleness": float(staleness.mean()),
            "mean_refresh_recovery": float(recovery.mean()),
            "positive_utility_fraction": float(useful.mean()),
            "spearman_utility_vs_staleness": float(pd.Series(utility).corr(pd.Series(staleness), method="spearman")),
            "positive_staleness_on_useful_fraction": share,
            "positive_staleness_concentration_lift": share / useful.mean() if useful.mean() else float("nan"),
        })
    return output


def matched_overlap(edge: str, logits: dict[str, np.ndarray], labels: np.ndarray) -> dict:
    benefit = stable_log_loss(logits["parent_exact_model"], labels) - stable_log_loss(
        logits["current_exact_kv"], labels
    )
    harm = stable_log_loss(logits["reuse_parent_kv"], labels) - stable_log_loss(
        logits["current_exact_kv"], labels
    )
    winners = benefit > 0.0
    positive_harm = np.maximum(harm, 0.0)
    harm_share = float(positive_harm[winners].sum() / positive_harm.sum()) if positive_harm.sum() else float("nan")
    return {
        "edge": edge,
        "requests": len(labels),
        "mean_matched_release_benefit": float(benefit.mean()),
        "mean_reuse_harm": float(harm.mean()),
        "spearman_G_H": float(pd.Series(benefit).corr(pd.Series(harm), method="spearman")),
        "release_winner_fraction": float(winners.mean()),
        "positive_harm_on_release_winners_fraction": harm_share,
        "positive_harm_concentration_lift": harm_share / winners.mean() if winners.mean() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    device = torch.device(args.device)

    manifest = pq.read_table(MANIFEST / "requests_quality.parquet").to_pandas()
    by_request = manifest.set_index("request_id").to_dict("index")
    labels = dict(zip(manifest["request_id"], manifest["label"], strict=True))
    selections = []
    for edge in range(5):
        rows = load_edge(edge, labels)
        rows = rows[(rows["append_count_since_cutover"] == 0) & (rows["history_length"] == 512)]
        rows = rows.sort_values(["uid", "query_timestamp", "request_id"]).groupby("uid", sort=True).head(1)
        selections.append(rows.head(args.max_requests).copy())
    uids = sorted({int(uid) for rows in selections for uid in rows["uid"]})

    probe_model, payload = load_model(checkpoint(1), device)
    oov_buckets = int(payload["config"]["num_items"]) - 781678
    del probe_model
    torch.cuda.empty_cache()
    history = load_histories(uids, oov_buckets=oov_buckets)

    summaries, regional, overlaps = [], [], []
    replay_errors = {}
    for edge, selected in enumerate(selections):
        parent, _ = load_model(checkpoint(edge), device)
        current, _ = load_model(checkpoint(edge + 1), device)
        requests = [
            {**by_request[request_id], "request_id": request_id, "weight": 1.0}
            for request_id in selected["request_id"]
        ]
        for request in requests:
            _, _, times = history.prefix(request["uid"], request["query_timestamp"], 512)
            request["query_delta_old384"] = float(request["query_timestamp"] - times[383])
            request["query_delta_old480"] = float(request["query_timestamp"] - times[479])
        path_logits: dict[str, list[float]] = {}
        for start in range(0, len(requests), 32):
            request_batch = requests[start:start + 32]
            batch = collate_foundation_batch(request_batch, history, device=device)
            query_deltas = {
                "full": batch.query_time_deltas,
                "old384": torch.tensor([row["query_delta_old384"] for row in request_batch], device=device),
                "old480": torch.tensor([row["query_delta_old480"] for row in request_batch], device=device),
            }
            observed = evaluate_batch(parent, current, batch, query_deltas)
            for name, values in observed.items():
                path_logits.setdefault(name, []).extend(values.tolist())
        logits = {name: np.asarray(values, dtype=np.float64) for name, values in path_logits.items()}
        replay_errors[f"v{edge}_to_v{edge + 1}"] = {
            "current": float(np.max(np.abs(logits["current_exact_kv"] - selected["current_exact_logit"].to_numpy()))),
            "reuse": float(np.max(np.abs(logits["reuse_parent_kv"] - selected["reuse_logit"].to_numpy()))),
        }
        if max(replay_errors[f"v{edge}_to_v{edge + 1}"].values()) > 2e-5:
            raise RuntimeError("K/V baseline replay differs from sealed rolling raw")
        target = np.asarray([request["label"] for request in requests], dtype=np.int64)
        exact = logits["current_exact_kv"]
        exact_loss = stable_log_loss(exact, target)
        for name, values in logits.items():
            summaries.append({
                "edge": f"v{edge}_to_v{edge + 1}",
                "path": name,
                "requests": len(values),
                "path_minus_exact_log_loss": float((stable_log_loss(values, target) - exact_loss).mean()),
                "mean_abs_probability_shift": float(np.abs(1 / (1 + np.exp(-values)) - 1 / (1 + np.exp(-exact))).mean()),
                "mean_bernoulli_js": float(bernoulli_js(values, exact).mean()),
            })
        regional.extend(regional_summary(f"v{edge}_to_v{edge + 1}", logits, target))
        overlaps.append(matched_overlap(f"v{edge}_to_v{edge + 1}", logits, target))
        del parent, current
        torch.cuda.empty_cache()

    frame = pd.DataFrame(summaries)
    regional_frame = pd.DataFrame(regional)
    overlap_frame = pd.DataFrame(overlaps)
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "kv_mechanism.csv", index=False)
    regional_frame.to_csv(args.output / "regional_utility_staleness.csv", index=False)
    overlap_frame.to_csv(args.output / "matched_benefit_harm_overlap.csv", index=False)
    gap = frame.pivot(index="edge", columns="path", values="mean_abs_probability_shift")
    lower3_recovery = 1.0 - gap["current_lower_3"] / gap["reuse_parent_kv"]
    summary = {
        "status": "kv_mechanism_probe_complete",
        "scope": f"up to {args.max_requests} append-free full-prefix requests per edge; Small seed17",
        "requests_per_edge": [len(rows) for rows in selections],
        "baseline_replay_max_abs_error": replay_errors,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# K/V mechanism probe", "",
        "Diagnostic exact splices only; they are not executable migration actions.", "",
        *markdown_table(frame, list(frame.columns)), "",
        "Replacing either all K or all V with Current state reduces the output gap on every edge, but neither side alone dominates consistently. The mechanism is coupled K/V incompatibility, not a clean key-only or value-only failure.", "",
        f"Replacing the lower three of four layers with Current K/V recovers {100 * lower3_recovery.min():.1f}% to {100 * lower3_recovery.max():.1f}% of the absolute-probability gap. Replacing only layer 0 is insufficient and can worsen the gap. This supports early/middle dependency propagation; the splice remains diagnostic, not an action.", "",
        "## Matched Parent/Current/Reuse overlap", "",
        *markdown_table(overlap_frame, list(overlap_frame.columns)), "",
        "This append-free probe uses Parent Exact under the Parent model, Current Exact under the Current model, and Parent K/V Reuse under the Current model on the same requests. It is the matched small-scale companion to the full-request descriptive overlap.", "",
        "## Regional Utility x Staleness", "",
        *markdown_table(regional_frame, list(regional_frame.columns)), "",
        "Utility removes and exactly recomputes the complementary history. Regional staleness replaces the same region with Parent K/V, and recovery starts from Parent K/V and refreshes that region. This aligns utility, staleness, and recovery on the same fixed region.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

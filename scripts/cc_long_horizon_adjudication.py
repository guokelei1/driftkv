#!/usr/bin/env python3
"""P3.0--P3.3 long-horizon opportunity adjudication for CC-theta0.

This script is deliberately evaluation-only.  It loads the already frozen
CC-theta0 checkpoint and never changes model weights, manifests, candidates,
or training targets.  The observed Gate-2 manifest is explicitly treated as
development evidence; no result written here authorizes theta1/theta2 or v2
training automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch
from cc_theta0_qualification import (
    BOOTSTRAP_ROUNDS,
    CANDIDATE_SIZE,
    CHECKPOINT,
    DAY,
    MAX_HISTORY,
    QUALITY_MANIFEST,
    RAW,
    RELEASE_CUTOFF,
    RESULT_DIR,
    SEED,
    TRAIN_END,
    TRAIN_MANIFEST,
    _metric_arrays,
    autocast_context,
    build_catalog,
    code_commit,
    event_time_deltas,
    item_map_from_catalog,
    load_checkpoint,
    load_histories,
    read_jsonl,
    row_timestamp,
    sha256_file,
)

ARTIST_MAPPING = RAW.parents[2] / "artist_item_mapping.parquet"
HORIZONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TAU_SECONDS = 7 * DAY


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    """User-level bootstrap; every evaluation row is one user."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metric_summary(scores: np.ndarray, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = _metric_arrays(scores)
    summary = {
        key: {
            "mean": float(value.mean()),
            "bootstrap_ci95": bootstrap_ci(value, seed + index),
        }
        for index, (key, value) in enumerate(arrays.items())
    }
    return summary, arrays


def selected_history(
    history: list[tuple[int, int, int]], kind: str, uid: int
) -> list[tuple[int, int, int]]:
    full = history[-MAX_HISTORY:]
    if kind == "Empty":
        return []
    if kind.startswith("Recent-"):
        return full[-int(kind.split("-", 1)[1]) :]
    if kind == "Full-512":
        return full
    if kind == "Old-480 Only":
        return full[:-32]
    if kind == "Full with Old-480 Masked":
        # The model only supports contiguous valid prefixes.  A causal mask
        # removing old keys/values is exactly the Recent-32 token sequence;
        # keeping a distinct name makes this equivalence auditable.
        return full[-32:]
    if kind == "Full with Old-480 Shuffled":
        old, recent = full[:-32], full[-32:]
        if not old:
            return full
        content = [old[index] for index in np.random.default_rng(SEED + uid).permutation(len(old))]
        old = [(content[index][0], old[index][1], content[index][2]) for index in range(len(old))]
        return old + recent
    raise ValueError(f"unknown selection {kind}")


def replacement_histories(
    histories: dict[int, list[tuple[int, int, int]]], rows: list[dict]
) -> dict[int, list[tuple[int, int, int]]]:
    """Replace Old-480 content by another user's old content on fixed slots."""
    eligible = [
        int(row["uid"]) for row in rows if len(histories[int(row["uid"])][-MAX_HISTORY:-32])
    ]
    result: dict[int, list[tuple[int, int, int]]] = {}
    for index, uid in enumerate(sorted(int(row["uid"]) for row in rows)):
        full = histories[uid][-MAX_HISTORY:]
        old, recent = full[:-32], full[-32:]
        if not old:
            result[uid] = full
            continue
        donor = eligible[(index + 1) % len(eligible)]
        donor_old = histories[donor][-MAX_HISTORY:-32]
        # Eligible donors have at least one old token. Repeat cyclically only
        # when necessary; timestamp slots always remain the receiver's slots.
        content = [donor_old[position % len(donor_old)] for position in range(len(old))]
        replaced = [
            (value[0], old[position][1], value[2]) for position, value in enumerate(content)
        ]
        result[uid] = replaced + recent
    return result


def collate_selected(
    rows: list[dict],
    histories: dict[int, list[tuple[int, int, int]]],
    item_map: dict[int, int],
    kind: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    selected = [selected_history(histories[int(row["uid"])], kind, int(row["uid"])) for row in rows]
    width = max(1, max((len(value) for value in selected), default=0))
    items = np.zeros((len(rows), width), dtype=np.int64)
    behaviors = np.zeros_like(items)
    deltas = np.zeros((len(rows), width), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    query_deltas = np.zeros(len(rows), dtype=np.float32)
    candidates = np.zeros((len(rows), CANDIDATE_SIZE), dtype=np.int64)
    for index, (row, history) in enumerate(zip(rows, selected, strict=True)):
        length = len(history)
        if length:
            items[index, :length] = [item_map[event[0]] for event in history]
            behaviors[index, :length] = [event[2] for event in history]
            deltas[index, :length] = event_time_deltas(history).astype(np.float32)
            query_deltas[index] = float(np.clip(row_timestamp(row) - history[-1][1], 0, 7 * DAY))
        lengths[index] = length
        candidates[index] = [item_map[int(value)] for value in row["candidate_item_ids"]]
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (items, behaviors, deltas, lengths, candidates, query_deltas)
    )


def score_kind(
    model,
    rows: list[dict],
    histories: dict[int, list[tuple[int, int, int]]],
    item_map: dict[int, int],
    kind: str,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    output = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = collate_selected(
            rows[start : start + batch_size], histories, item_map, kind, device
        )
        items, behaviors, deltas, lengths, candidates, query_deltas = batch
        with torch.inference_mode(), autocast_context(device):
            scores = model.score_cc_full(
                items, behaviors, deltas, candidates, query_deltas, lengths=lengths
            )
        output.append(scores.float().cpu().numpy())
    return np.concatenate(output, axis=0)


def span_summary(
    rows: list[dict], histories: dict[int, list[tuple[int, int, int]]], k: int
) -> dict:
    spans, actual = [], []
    if k == 0:
        return {
            "actual_token_count": {"p50": 0, "p90": 0, "p99": 0},
            "calendar_span_seconds": {"p50": 0, "p90": 0, "p99": 0},
        }
    for row in rows:
        selected = histories[int(row["uid"])][-MAX_HISTORY:][-k:]
        if selected:
            actual.append(len(selected))
            spans.append(max(0, row_timestamp(row) - selected[0][1]))
    return {
        "actual_token_count": {f"p{q}": float(np.percentile(actual, q)) for q in (50, 90, 99)},
        "calendar_span_seconds": {f"p{q}": float(np.percentile(spans, q)) for q in (50, 90, 99)},
    }


def run_effective_horizon(device: torch.device) -> dict:
    model, checkpoint = load_checkpoint(device)
    rows = read_jsonl(QUALITY_MANIFEST)
    # This is the deployed release snapshot prefix used by Gate 2, not an
    # updated post-target prefix.
    histories = load_histories(rows, history_cutoff=RELEASE_CUTOFF)
    _, catalog, _ = build_catalog()
    item_map = item_map_from_catalog(catalog)
    scores, summaries, arrays = {}, {}, {}
    for k in HORIZONS:
        name = "Empty" if k == 0 else f"Recent-{k}"
        scores[name] = score_kind(model, rows, histories, item_map, name, device)
        summaries[name], arrays[name] = metric_summary(scores[name], SEED + 1_000 + k * 10)
    full_name = "Recent-512"
    full = arrays[full_name]["target_log_prob"]
    empty = arrays["Empty"]["target_log_prob"]
    total = float((full - empty).mean())
    incremental = {}
    for k in HORIZONS[2:]:
        now, previous = (
            arrays[f"Recent-{k}"]["target_log_prob"],
            arrays[f"Recent-{k // 2}"]["target_log_prob"],
        )
        delta = now - previous
        incremental[f"recent_{k}_minus_recent_{k // 2}"] = {
            "mean": float(delta.mean()),
            "bootstrap_ci95": bootstrap_ci(delta, SEED + 2_000 + k),
        }
    gain_ratio = {
        str(k): float(
            (arrays[("Empty" if k == 0 else f"Recent-{k}")]["target_log_prob"] - empty).mean()
            / total
        )
        if total > 0
        else None
        for k in HORIZONS
    }
    l_values = {}
    for fraction in (0.90, 0.95):
        qualifying = [
            k
            for k in HORIZONS
            if k and gain_ratio[str(k)] is not None and gain_ratio[str(k)] >= fraction
        ]
        l_values[f"L{int(fraction * 100)}"] = min(qualifying) if qualifying else None
    curve = RESULT_DIR / "cc_theta0_effective_history_horizon_curve_v1.csv"
    with curve.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "tokens",
                "target_log_prob",
                "target_log_prob_ci95_lower",
                "target_log_prob_ci95_upper",
                "cross_entropy",
                "pairwise_auc",
                "ndcg_at_10",
                "hr_at_10",
                "mrr",
                "gain_fraction_of_full_empty",
                "span_p50_seconds",
                "span_p90_seconds",
                "span_p99_seconds",
            ]
        )
        for k in HORIZONS:
            name = "Empty" if k == 0 else f"Recent-{k}"
            values = summaries[name]
            spans = span_summary(rows, histories, k)["calendar_span_seconds"]
            writer.writerow(
                [
                    k,
                    values["target_log_prob"]["mean"],
                    *values["target_log_prob"]["bootstrap_ci95"],
                    values["cross_entropy"]["mean"],
                    values["pairwise_auc"]["mean"],
                    values["ndcg@10"]["mean"],
                    values["hr@10"]["mean"],
                    values["mrr"]["mean"],
                    gain_ratio[str(k)],
                    spans["p50"],
                    spans["p90"],
                    spans["p99"],
                ]
            )
    old_kinds = (
        "Recent-32",
        "Old-480 Only",
        "Full-512",
        "Full with Old-480 Masked",
        "Full with Old-480 Shuffled",
    )
    old_scores = {
        kind: score_kind(model, rows, histories, item_map, kind, device) for kind in old_kinds
    }
    replaced = replacement_histories(histories, rows)
    old_scores["Full with Old-480 Cross-User Replaced"] = score_kind(
        model, rows, replaced, item_map, "Full-512", device
    )
    old_arrays = {kind: _metric_arrays(value) for kind, value in old_scores.items()}
    full_old = old_arrays["Full-512"]["target_log_prob"]
    ablations = {"paths": {}, "full_minus_path_target_log_prob": {}}
    for index, (kind, value) in enumerate(old_scores.items()):
        summary, _ = metric_summary(value, SEED + 3_000 + index * 20)
        ablations["paths"][kind] = summary
        if kind != "Full-512":
            delta = full_old - old_arrays[kind]["target_log_prob"]
            ablations["full_minus_path_target_log_prob"][kind] = {
                "mean": float(delta.mean()),
                "bootstrap_ci95": bootstrap_ci(delta, SEED + 4_000 + index),
            }
    mask_error = float(
        np.max(np.abs(old_scores["Recent-32"] - old_scores["Full with Old-480 Masked"]))
    )
    ablations["mask_semantics"] = {
        "implementation": "remove Old-480 keys/values; contiguous-token model makes this exactly Recent-32",
        "max_score_error_vs_recent32": mask_error,
    }
    ablations["applicability"] = {
        "rows_with_old_480_tokens": int(
            sum(len(histories[int(row["uid"])][-MAX_HISTORY:-32]) > 0 for row in rows)
        ),
        "rows": len(rows),
    }
    trace = trace_fields(checkpoint)
    horizon_result = {
        "contract": "effective_history_horizon_v1",
        "status": "completed_development_only",
        **trace,
        "gate_manifest_status": "downgraded_to_development_after_gate2_observation",
        "prefix_cutoff": RELEASE_CUTOFF,
        "rows": len(rows),
        "token_horizons": list(HORIZONS),
        "path_metrics": summaries,
        "incremental_target_log_prob": incremental,
        "full_minus_empty_target_log_prob": total,
        "gain_fraction_of_full_minus_empty": gain_ratio,
        "saturation": {
            "definition": "smallest observed k whose mean gain reaches fraction of Full-512 minus Empty; descriptive, not a gate",
            **l_values,
        },
        "calendar_horizon": {str(k): span_summary(rows, histories, k) for k in HORIZONS},
        "curve_csv": str(curve),
    }
    (RESULT_DIR / "cc_theta0_effective_history_horizon_v1.json").write_text(
        json.dumps(horizon_result, indent=2) + "\n"
    )
    ablation_result = {
        "contract": "cc_theta0_old480_causal_ablations_v1",
        "status": "completed_development_only",
        **trace,
        "prefix_cutoff": RELEASE_CUTOFF,
        **ablations,
    }
    (RESULT_DIR / "cc_theta0_old480_causal_ablations_v1.json").write_text(
        json.dumps(ablation_result, indent=2) + "\n"
    )
    return {"horizon": horizon_result, "ablations": ablation_result}


def load_artist_by_item() -> np.ndarray:
    table = pq.read_table(ARTIST_MAPPING, columns=["artist_id", "item_id"])
    size = int(table.column("item_id").to_numpy(zero_copy_only=False).max()) + 1
    values = np.full(size, -1, dtype=np.int64)
    items = table.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
    artists = table.column("artist_id").to_numpy(zero_copy_only=False).astype(np.int64)
    values[items] = artists
    return values


def target_metadata(rows: list[dict]) -> dict[int, int]:
    wanted = {(int(row["uid"]), row_timestamp(row), int(row["positive_item_id"])) for row in rows}
    found: dict[int, int] = {}
    for batch in pq.ParquetFile(RAW).iter_batches(
        batch_size=262_144, columns=["uid", "timestamp", "item_id", "is_organic"]
    ):
        for uid, timestamp, item, organic in zip(
            *(
                batch.column(name).to_numpy(zero_copy_only=False)
                for name in ("uid", "timestamp", "item_id", "is_organic")
            ),
            strict=True,
        ):
            key = (int(uid), int(timestamp), int(item))
            if key in wanted:
                found[int(uid)] = int(organic)
    return found


def candidate_baseline(
    history: list[tuple[int, int, int]], row: dict, artist: np.ndarray, *, mode: str, window: int
) -> np.ndarray:
    selected = history[-MAX_HISTORY:][-window:]
    candidates = np.asarray(row["candidate_item_ids"], dtype=np.int64)
    if mode.startswith("item"):
        keys = np.asarray([event[0] for event in selected], dtype=np.int64)
        target_keys = candidates
    else:
        keys = (
            artist[np.asarray([event[0] for event in selected], dtype=np.int64)]
            if selected
            else np.empty(0, dtype=np.int64)
        )
        target_keys = artist[candidates]
    weights = np.ones(len(keys), dtype=np.float64)
    if "recency" in mode and len(keys):
        ages = np.asarray(
            [max(0, row_timestamp(row) - event[1]) for event in selected], dtype=np.float64
        )
        weights = np.exp(-ages / TAU_SECONDS)
    counts: dict[int, float] = {}
    for key, weight in zip(keys, weights, strict=True):
        if key >= 0:
            counts[int(key)] = counts.get(int(key), 0.0) + float(weight)
    return np.asarray(
        [counts.get(int(key), 0.0) if key >= 0 else 0.0 for key in target_keys], dtype=np.float64
    )


def cohort_summary(indices: np.ndarray, delta_arrays: dict[str, np.ndarray]) -> dict:
    result = {"n": int(indices.sum()), "share": float(indices.mean())}
    if indices.sum() >= 10:
        result["full_minus_recent32_target_log_prob"] = {
            name: {
                "mean": float(value[indices].mean()),
                "bootstrap_ci95": bootstrap_ci(value[indices], SEED + 6_000 + position),
            }
            for position, (name, value) in enumerate(delta_arrays.items())
        }
    return result


def run_data_opportunity() -> dict:
    rows = read_jsonl(QUALITY_MANIFEST)
    # Unlike model scoring, this is a strict causal data audit at each actual
    # target time, so it intentionally includes pre-target events after release.
    histories = load_histories(rows)
    artist = load_artist_by_item()
    target_org = target_metadata(rows)
    modes = ("item_frequency", "artist_frequency", "item_recency_decay", "artist_recency_decay")
    score_arrays = {}
    for mode in modes:
        recent = np.stack(
            [
                candidate_baseline(histories[int(row["uid"])], row, artist, mode=mode, window=32)
                for row in rows
            ]
        )
        full = np.stack(
            [
                candidate_baseline(histories[int(row["uid"])], row, artist, mode=mode, window=512)
                for row in rows
            ]
        )
        score_arrays[f"recent32_{mode}"] = _metric_arrays(recent)
        score_arrays[f"full512_{mode}"] = _metric_arrays(full)
    baseline_metrics = {}
    deltas = {}
    for index, mode in enumerate(modes):
        recent, full = score_arrays[f"recent32_{mode}"], score_arrays[f"full512_{mode}"]
        delta = full["target_log_prob"] - recent["target_log_prob"]
        deltas[mode] = delta
        baseline_metrics[mode] = {
            "recent32": {key: float(value.mean()) for key, value in recent.items()},
            "full512": {key: float(value.mean()) for key, value in full.items()},
            "full_minus_recent32_target_log_prob": {
                "mean": float(delta.mean()),
                "bootstrap_ci95": bootstrap_ci(delta, SEED + 5_000 + index),
            },
        }
    item_recent, item_old, artist_recent, artist_old = [], [], [], []
    gaps, repeats, diversity, activity = [], [], [], []
    for row in rows:
        history = histories[int(row["uid"])][-MAX_HISTORY:]
        recent, old = history[-32:], history[:-32]
        target = int(row["positive_item_id"])
        target_artist = int(artist[target]) if target < len(artist) else -1
        recent_items, old_items = {event[0] for event in recent}, {event[0] for event in old}
        recent_artists = {
            int(artist[event[0]])
            for event in recent
            if event[0] < len(artist) and artist[event[0]] >= 0
        }
        old_artists = {
            int(artist[event[0]])
            for event in old
            if event[0] < len(artist) and artist[event[0]] >= 0
        }
        item_recent.append(target in recent_items)
        item_old.append(target in old_items)
        artist_recent.append(target_artist >= 0 and target_artist in recent_artists)
        artist_old.append(target_artist >= 0 and target_artist in old_artists)
        gaps.append(row_timestamp(row) - history[-1][1] if history else np.inf)
        repeats.append(1.0 - len({event[0] for event in history}) / max(1, len(history)))
        diversity.append(len({event[0] for event in history}) / max(1, len(history)))
        activity.append(sum(event[1] >= row_timestamp(row) - 7 * DAY for event in history))
    item_recent, item_old = np.asarray(item_recent), np.asarray(item_old)
    artist_recent, artist_old = np.asarray(artist_recent), np.asarray(artist_old)
    gaps, repeats, diversity, activity = map(np.asarray, (gaps, repeats, diversity, activity))
    strata = {
        "target_item_recent": item_recent,
        "target_item_old_only": ~item_recent & item_old,
        "target_item_unseen_in_full": ~item_recent & ~item_old,
        "target_artist_recent": artist_recent,
        "target_artist_old_only": ~artist_recent & artist_old,
        "target_artist_unseen_in_full": ~artist_recent & ~artist_old,
        "short_gap_lt_30m": gaps < 30 * 60,
        "session_start_30m_to_1d": (gaps >= 30 * 60) & (gaps < DAY),
        "long_gap_ge_1d": gaps >= DAY,
        "repeat_item": item_recent | item_old,
        "new_item_known_artist": ~(item_recent | item_old) & (artist_recent | artist_old),
        "new_artist": ~(artist_recent | artist_old),
        "organic_target": np.asarray([target_org.get(int(row["uid"])) == 1 for row in rows]),
        "recommendation_driven_target": np.asarray(
            [target_org.get(int(row["uid"])) == 0 for row in rows]
        ),
        "high_repeat_user": repeats >= np.median(repeats),
        "high_diversity_user": diversity >= np.median(diversity),
        "high_activity_user": activity >= np.median(activity),
        "low_activity_user": activity < np.median(activity),
    }
    result = {
        "contract": "yambda_long_history_data_opportunity_v1",
        "status": "completed_development_only",
        "raw_source_sha256": sha256_file(RAW),
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "rows": len(rows),
        "history_definition": "strictly pre-target causal, capped at 512; this data audit is separate from the release-capped model prefix",
        "artist_mapping_sha256": sha256_file(ARTIST_MAPPING),
        "target_metadata_found": len(target_org),
        "baseline_definition": {
            "windows": [32, 512],
            "recency_decay_tau_seconds": TAU_SECONDS,
            "scores": list(modes),
        },
        "baseline_metrics": baseline_metrics,
        "cohorts": {name: cohort_summary(mask, deltas) for name, mask in strata.items()},
        "thresholds": {
            "session_start_seconds": 30 * 60,
            "long_gap_seconds": DAY,
            "user_medians": {
                "repeat_ratio": float(np.median(repeats)),
                "diversity_ratio": float(np.median(diversity)),
                "events_last_7d": float(np.median(activity)),
            },
        },
        "target_dependent_diagnostic_only": [
            "target_item_*",
            "target_artist_*",
            "repeat_item",
            "new_item_known_artist",
            "new_artist",
            "organic_target",
            "recommendation_driven_target",
        ],
        "query_time_defined_cohorts": [
            "short_gap_lt_30m",
            "session_start_30m_to_1d",
            "long_gap_ge_1d",
            "high_repeat_user",
            "high_diversity_user",
            "high_activity_user",
            "low_activity_user",
        ],
        "code_commit": code_commit(),
        "seed": SEED,
    }
    output = RESULT_DIR / "yambda_long_history_data_opportunity_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    with (RESULT_DIR / "yambda_long_history_data_opportunity_v1.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cohort",
                "n",
                "share",
                "item_frequency_full_minus_recent32",
                "artist_frequency_full_minus_recent32",
                "item_recency_full_minus_recent32",
                "artist_recency_full_minus_recent32",
            ]
        )
        for name, value in result["cohorts"].items():
            values = value.get("full_minus_recent32_target_log_prob", {})
            writer.writerow(
                [
                    name,
                    value["n"],
                    value["share"],
                    *[values.get(mode, {}).get("mean") for mode in modes],
                ]
            )
    return result


def quantiles(values: list[int | float]) -> dict:
    return {f"p{q}": float(np.percentile(values, q)) for q in (0, 25, 50, 75, 90, 99, 100)}


def run_training_coverage() -> dict:
    rows = read_jsonl(TRAIN_MANIFEST)
    histories = load_histories(rows)
    train = json.loads((RESULT_DIR / "cc_theta0_train_v1.json").read_text())
    model, _ = load_checkpoint(torch.device("cpu"))
    counts = Counter(int(row["uid"]) for row in rows)
    lengths, gaps, target_times = [], [], []
    for row in rows:
        history = histories[int(row["uid"])][-MAX_HISTORY:]
        lengths.append(len(history))
        gaps.append(row_timestamp(row) - history[-1][1])
        target_times.append(row_timestamp(row))
    edges = np.linspace(0, TRAIN_END, 15, dtype=np.int64)
    bins = np.histogram(target_times, bins=edges)[0]
    result = {
        "contract": "cc_training_coverage_audit_v1",
        "status": "completed_development_only",
        "training_manifest_hash": sha256_file(TRAIN_MANIFEST),
        "validation_manifest_hash": train["validation_manifest_hash"],
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "raw_source_sha256": sha256_file(RAW),
        "rows": len(rows),
        "eligible_users": len(counts),
        "targets_per_user": {
            "quantiles": quantiles(list(counts.values())),
            "exactly_one_target_fraction": float(np.mean(np.asarray(list(counts.values())) == 1)),
        },
        "context_tokens": {"total": int(sum(lengths)), "per_query": quantiles(lengths)},
        "target_time_coverage": {
            "foundation_end_seconds": TRAIN_END,
            "14_equal_time_bins": bins.tolist(),
            "nonempty_bins": int((bins > 0).sum()),
            "target_timestamp_quantiles": quantiles(target_times),
        },
        "session_coverage": {
            "target_gap_seconds": quantiles(gaps),
            "session_start_ge_30m_fraction": float(np.mean(np.asarray(gaps) >= 30 * 60)),
            "long_gap_ge_1d_fraction": float(np.mean(np.asarray(gaps) >= DAY)),
        },
        "optimization": {
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "optimizer_steps": train["training"]["optimizer_steps"],
            "batch_size": train["training"]["batch_size"],
            "epochs": train["training"]["epochs"],
            "first_training_loss": train["training"]["first_training_loss"],
            "last_training_loss": train["training"]["last_training_loss"],
            "mean_training_loss": train["training"]["mean_training_loss"],
        },
        "validation_saturation": {
            "curve_available": False,
            "observed_validation_points": 1,
            "final_validation": train["validation"],
            "interpretation": "fixed one-epoch protocol saved/evaluated only its final checkpoint; whether validation was still improving cannot be inferred without a new preregistered training run",
        },
        "code_commit": code_commit(),
        "seed": SEED,
    }
    (RESULT_DIR / "cc_training_coverage_audit_v1.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def trace_fields(checkpoint: dict) -> dict:
    return {
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "training_manifest_hash": checkpoint["training_manifest_hash"],
        "qualification_manifest_hash": sha256_file(QUALITY_MANIFEST),
        "raw_source_sha256": sha256_file(RAW),
        "code_commit": code_commit(),
        "seed": SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("horizon", "opportunity", "coverage", "all"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but it is unavailable")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    if args.command in ("horizon", "all"):
        output["effective_history_horizon"] = run_effective_horizon(device)
    if args.command in ("opportunity", "all"):
        output["data_opportunity"] = run_data_opportunity()
    if args.command in ("coverage", "all"):
        output["training_coverage"] = run_training_coverage()
    if args.command == "all":
        summary = {
            "contract": "long_horizon_opportunity_adjudication_v1",
            "status": "completed_development_only",
            "gate1": "passed",
            "gate2": {
                "status": "failed",
                "failure_mode": "history_utility_saturates_by_recent_32",
                "cc_theta1_theta2_authorized": False,
            },
            "next_action": "human evidence review required; this script never trains CC-theta0-v2",
            "outputs": [str(path) for path in sorted(RESULT_DIR.glob("*history*"))]
            + [
                str(RESULT_DIR / "yambda_long_history_data_opportunity_v1.json"),
                str(RESULT_DIR / "cc_training_coverage_audit_v1.json"),
            ],
            "code_commit": code_commit(),
        }
        (RESULT_DIR / "long_horizon_opportunity_adjudication_v1.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        output["summary"] = summary
    print(
        json.dumps(
            {
                key: value.get("status", "completed") if isinstance(value, dict) else "completed"
                for key, value in output.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

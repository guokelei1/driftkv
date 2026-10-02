#!/usr/bin/env python3
"""Read-only descriptive cohort audit; no user reselection or model execution."""
from pathlib import Path
import itertools
import json
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from read_correction_v4.common import PANEL_ROOT, RESERVATIONS, sha256, write_json
from hstu_kvcache.evaluation.binary_metrics import sigmoid

OUTPUT = Path(__file__).resolve().parent
SCALES = ("medium", "large", "max")
QTAG = "query_pure_cross_phi_joint8_c512"


def describe(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": len(x), "mean": float(x.mean()), **dict(zip(
        ("min", "p10", "p25", "p50", "p75", "p90", "max"),
        (float(v) for v in np.quantile(x, [0, .1, .25, .5, .75, .9, 1]))))}


def auc(labels, logits, weights=None):
    y, score = np.asarray(labels), sigmoid(np.asarray(logits))
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    order = np.argsort(score, kind="stable")
    score, y, w = score[order], y[order], w[order]
    if not len(y) or (w*y).sum() == 0 or (w*(1-y)).sum() == 0:
        return None
    starts = np.r_[0, np.flatnonzero(score[1:] != score[:-1])+1]
    pos, neg = np.add.reduceat(w*y, starts), np.add.reduceat(w*(1-y), starts)
    return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(pos.sum()*neg.sum()))


def quality(frame, weights=None):
    result = {"users": int(frame.uid.nunique()), "requests": len(frame),
              "negative_fraction": float((frame.label == 0).mean()) if len(frame) else None}
    if weights is not None:
        result["weighted_negative_fraction"] = float(np.sum(np.asarray(weights)*(frame.label == 0))/np.sum(weights)) if np.sum(weights) else None
    for field in ("full_logit", "reuse_logit", "query_logit"):
        if field in frame:
            result[field.replace("_logit", "_auc")] = auc(frame.label, frame[field], weights)
    full, reuse, query = (result.get(f"{name}_auc") for name in ("full", "reuse", "query"))
    if full is not None and reuse is not None:
        result["full_minus_reuse_auc_pp"] = 100*(full-reuse)
        if query is not None:
            result["query_minus_reuse_auc_pp"] = 100*(query-reuse)
            result["recovery_percent"] = 100*(query-reuse)/(full-reuse) if full != reuse else None
    return result


def features(frame):
    result = {"users": int(frame.uid.nunique()), "requests": len(frame),
        "negative_fraction": float((frame.label == 0).mean()) if len(frame) else None,
        "requests_per_user": describe(frame.groupby("uid").size()),
        "unique_candidate_items": int(frame.raw_item_id.nunique()),
        "candidate_unique_fraction": float(frame.raw_item_id.nunique()/len(frame)) if len(frame) else None}
    for column in ("activity", "history_length", "append_count_since_cutover", "old_fraction",
                   "source_candidate_frequency", "source_candidate_probability"):
        if column in frame:
            result[column] = describe(frame[column])
    if len(frame) and "history_length" in frame:
        result["full_history_fraction"] = float((frame.history_length == 1024).mean())
    return result


def add_age(frame):
    frame["old_fraction"] = np.maximum(0, frame.history_length-frame.append_count_since_cutover)/frame.history_length
    activity = frame.groupby("uid").size()
    frame["activity"] = frame.uid.map(activity)
    age = np.digitize(frame.old_fraction, [.25, .75])
    count = np.digitize(frame.activity, [2, 5, 10, 20])
    frame["stratum"] = list(zip(count.tolist(), age.tolist(), (frame.history_length == 1024).astype(int).tolist()))
    return frame


def standardized(frames):
    """Same class-conditional feature support and mass, with no score-based choices."""
    counts = {s: f.groupby(["label", "stratum"]).size() for s, f in frames.items()}
    support = set.intersection(*(set(c.index) for c in counts.values()))
    common = {key: min(int(c[key]) for c in counts.values()) for key in support}
    totals = {label: sum(value for (y, _), value in common.items() if y == label) for label in (0, 1)}
    results = {}
    for scale, frame in frames.items():
        w = np.asarray([common.get((int(y), stratum), 0)/max(1, totals[int(y)])/counts[scale].get((int(y), stratum), 1)
                        for y, stratum in zip(frame.label, frame.stratum)])
        chosen = w > 0
        results[scale] = {**quality(frame.loc[chosen], w[chosen]),
            "retained_request_fraction": float(chosen.mean()), "effective_n": float(w.sum()**2/(w*w).sum()) if w.sum() else 0}
    return {"rule": "common min-count mass within each label over fixed activity[1,2-4,5-9,10-19,20+] x old-fraction[<.25,.25-.75,>=.75] x history[<1024,1024]; descriptor only, not causal adjustment",
        "common_label_strata": len(common), "results": results}


def main():
    pa.set_cpu_count(4)
    panels, source_overlaps, selected_overlaps, paired, standards, user_records = [], [], [], [], [], []
    for edge in range(1, 6):
        name = f"v{edge-1}_to_v{edge}"
        frames, source_uids = {}, {}
        for scale in SCALES:
            folder = PANEL_ROOT / scale / name
            binding = json.loads((folder / "binding.json").read_text())
            ds_path = ROOT / binding["sources"]["dataset"]["path"]
            dataset = json.loads(ds_path.read_text())
            reservation = json.loads((RESERVATIONS / scale / name / "calibration_users.json").read_text())
            lo, hi = binding["days"]
            source = pq.read_table(ROOT / binding["sources"]["labels"]["path"], filters=[
                ("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
                ("query_timestamp", ">=", lo*86400), ("query_timestamp", "<", hi*86400)],
                columns=["request_id", "uid", "query_timestamp", "raw_item_id", "item_idx", "label"]).to_pandas()
            if len(source) != binding["population"]["requests"] or source.uid.nunique() != binding["population"]["users"]:
                raise RuntimeError("source labels differ from frozen source population")
            source_uids[scale] = set(int(u) for u in source.uid.unique())
            frequency = source.raw_item_id.value_counts()
            source["source_candidate_frequency"] = source.raw_item_id.map(frequency)
            source["source_candidate_probability"] = source.source_candidate_frequency/len(source)
            source["activity"] = source.uid.map(source.groupby("uid").size())
            panel = pq.read_table(folder / "evaluation_requests.parquet").to_pandas()
            raw = source.set_index("request_id")["raw_item_id"]
            panel["raw_item_id"] = panel.request_id.map(raw)
            if panel.raw_item_id.isna().any():
                raise RuntimeError("panel request missing from source")
            panel["source_candidate_frequency"] = panel.raw_item_id.map(frequency)
            panel["source_candidate_probability"] = panel.source_candidate_frequency/len(source)
            panel = add_age(panel)
            summary_path = ROOT / "results/read_correction_2026_09/v5/population_run/evaluation" / scale / name / "summary.json"
            summary = json.loads(summary_path.read_text())
            scores_path = Path(summary["scores"][QTAG]["path"])
            if sha256(scores_path) != summary["scores"][QTAG]["sha256"]:
                raise RuntimeError("Q512 scores changed")
            scores = pq.read_table(scores_path).to_pandas().set_index("request_id")
            selected = panel.set_index("request_id")
            if set(scores.index) != set(selected.index):
                raise RuntimeError("Q evaluation panel differs")
            for column in ("uid", "label", "full_logit", "reuse_logit"):
                if not np.array_equal(scores.loc[selected.index, column], selected[column]):
                    raise RuntimeError(f"Q control {column} differs")
            panel["query_logit"] = panel.request_id.map(scores.hstu_logit)
            cal_path = Path(summary["inputs"]["policies"][QTAG]["calibration"]["path"])
            calibration = json.loads(cal_path.read_text())
            cal_uids, val_uids = set(calibration["uids"]), set(calibration["validation_uids"])
            if (cal_uids | val_uids) & set(panel.uid):
                raise RuntimeError("calibration/evaluation overlap")
            shard_summary = json.loads((ROOT / binding["sources"]["reuse_summary"]["path"]).read_text())
            needed = cal_uids | val_uids
            history_frames = []
            for shard in shard_summary["reuse_shards"]:
                history = pq.read_table(ROOT / shard["path"], columns=["request_id", "uid", "history_length", "append_count_since_cutover"]).to_pandas()
                history_frames.append(history.loc[history.uid.isin(needed)].drop(columns="uid"))
            later = source.loc[source.uid.isin(needed)].merge(pd.concat(history_frames), on="request_id", validate="one_to_one")
            later = add_age(later)
            valid_gaps, categories = [], {"positive_only": 0, "negative_only": 0, "both_classes": 0}
            for uid, group in panel.groupby("uid", sort=True):
                count = int(group.label.sum())
                category = "negative_only" if count == 0 else "positive_only" if count == len(group) else "both_classes"
                categories[category] += 1
                values = quality(group)
                gap = values.get("full_minus_reuse_auc_pp")
                if gap is not None:
                    valid_gaps.append(gap)
                user_records.append({"scale": scale, "edge": name, "uid": int(uid), "label_scope": category, **values})
            stats = quality(panel)
            if abs(stats["full_auc"]-binding["panels"]["evaluation"]["current_full"]["ROC_AUC"]) > 1e-12:
                raise RuntimeError("recomputed panel AUC differs")
            per_user = {"label_categories": categories, "defined_auc_users": len(valid_gaps),
                "gap_auc_pp": describe(valid_gaps), "full_better": sum(g > 0 for g in valid_gaps),
                "equal": sum(g == 0 for g in valid_gaps), "reuse_better": sum(g < 0 for g in valid_gaps),
                "note": "single-class users have null AUC and remain in all global metrics; selection score was cross-user concordance, not per-user AUC"}
            record = {"scale": scale, "edge": name, "selection": binding["selection"],
                "days": binding["days"], "cutover": binding["cutover"], "actual_model_config": calibration["model_config"],
                "dataset": {k: dataset.get(k) for k in ("dataset", "users", "foundation_items", "history_tie_order")},
                "source_uid_sha256": hashlib.sha256(np.asarray(sorted(source_uids[scale]), dtype="<i8").tobytes()).hexdigest(),
                "source_metrics": binding["population"], "source_features": features(source),
                "selected_fraction_of_source_users": 3000/len(source_uids[scale]), "global": stats,
                "selected_features": features(panel), "per_user": per_user,
                "selected_by_label": {str(y): features(panel.loc[panel.label == y]) for y in (0, 1)},
                "source_by_label": {str(y): features(source.loc[source.label == y]) for y in (0, 1)},
                "calibration_reserved": {"users": len(cal_uids), "validation_users": len(val_uids),
                    "exclusion_rule": reservation["excluded"], "selection_rule": reservation["rule"],
                    "candidate_rule": calibration["candidate_rule"], "pre_release_capture_users": calibration.get("shared_snapshot_users"),
                    "pre_release_capture_mean_history_length": calibration["cache_bytes"]/(16*calibration["model_config"]["num_layers"]*calibration["model_config"]["hidden_size"]*calibration["shared_snapshot_users"]),
                    "followup_requests_features": features(later),
                    "followup_by_label": {str(y): features(later.loc[later.label == y]) for y in (0, 1)},
                    "followup_note": "descriptive later real requests of reserved users only; these labels/requests were not used for fitting"},
                "sources": {"binding": str(folder / "binding.json"), "binding_sha256": sha256(folder / "binding.json"),
                    "Q_summary": str(summary_path), "Q_summary_sha256": sha256(summary_path), "calibration": str(cal_path)}}
            panels.append(record)
            panel["paired_key"] = list(zip(panel.uid, panel.query_timestamp, panel.raw_item_id.astype(int)))
            if panel.paired_key.duplicated().any():
                raise RuntimeError("shared UID/time/raw-item key is ambiguous")
            frames[scale] = panel
            print(json.dumps({"scale": scale, "edge": name, "gap_pp": stats["full_minus_reuse_auc_pp"],
                "Q512_recovery": stats["recovery_percent"], "negative_fraction": stats["negative_fraction"],
                "per_user_auc_defined": len(valid_gaps)}), flush=True)
        for left, right in itertools.combinations(SCALES, 2):
            a, b = source_uids[left], source_uids[right]
            source_overlaps.append({"edge": name, "left": left, "right": right, "left_users": len(a), "right_users": len(b),
                "intersection": len(a & b), "union": len(a | b), "jaccard": len(a & b)/len(a | b)})
            a, b = set(frames[left].uid), set(frames[right].uid)
            selected_overlaps.append({"edge": name, "left": left, "right": right, "intersection": len(a & b),
                "union": len(a | b), "jaccard": len(a & b)/len(a | b)})
            shared = set(frames[left].paired_key) & set(frames[right].paired_key)
            subset = {s: frames[s].set_index("paired_key").loc[sorted(shared)] for s in (left, right)}
            if not np.array_equal(subset[left].label, subset[right].label):
                raise RuntimeError("shared raw requests have different labels")
            paired.append({"edge": name, "left": left, "right": right, "shared_uids": len(a & b),
                "exact_shared_requests": len(shared), "metrics": {s: quality(f) for s, f in subset.items()},
                "history_length_mismatches": int((subset[left].history_length != subset[right].history_length).sum()),
                "append_count_mismatches": int((subset[left].append_count_since_cutover != subset[right].append_count_since_cutover).sum())})
        common = set.intersection(*(set(f.paired_key) for f in frames.values()))
        paired.append({"edge": name, "scales": list(SCALES), "exact_shared_requests": len(common),
            "metrics": {s: quality(f.loc[f.paired_key.isin(common)]) for s, f in frames.items()}})
        standards.append({"edge": name, **standardized(frames)})
    result = {"status": "complete", "panels": panels, "source_uid_overlaps": source_overlaps,
        "selected_uid_overlaps": selected_overlaps, "exact_request_intersections": paired,
        "class_conditional_standardization": standards,
        "limits": ["class prevalence alone does not change AUC for fixed class-conditional score distributions",
            "cohorts are outcome-conditioned and not population estimates; no cohort or frozen outcome changed",
            "source_candidate_frequency is descriptive frequency in the same evaluation window, not pre-release popularity or a fitting feature",
            "all requests are target_known; true OOV/cold catalog behavior cannot be inferred",
            "shared-request and coarse weighting comparisons reduce measured cohort differences but do not isolate architecture from checkpoint/training differences"]}
    write_json(OUTPUT / "audit.json", result)
    pd.DataFrame(user_records).to_parquet(OUTPUT / "per_user_auc.parquet", index=False)
    pd.DataFrame([{"scale": p["scale"], "edge": p["edge"], "source_users": p["source_metrics"]["users"],
        "source_gap_pp": p["source_metrics"]["full_minus_reuse_auc_pp"],
        "selected_fraction": p["selected_fraction_of_source_users"], **p["global"],
        **p["per_user"]["label_categories"], "defined_user_gap_mean_pp": p["per_user"]["gap_auc_pp"].get("mean")}
        for p in panels]).to_csv(OUTPUT / "panel_summary.csv", index=False)


if __name__ == "__main__":
    main()

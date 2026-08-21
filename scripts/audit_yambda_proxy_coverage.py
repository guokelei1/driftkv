#!/usr/bin/env python3
"""Audit strict observed-event proxy coverage and pre-release cohort bias."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EDGES = ("theta0_theta1", "theta1_theta2")
FEATURES = (
    "effective_prefix_length", "raw_prefix_length", "last_activity_age_seconds",
    "events_last_1d", "events_last_7d", "events_last_30d",
    "unique_items_last_7d", "unique_artists_last_7d",
    "repeat_ratio_last_7d", "organic_ratio_last_7d", "exact_token_layer_work",
)
STRATA = {
    "last_activity_age_quartile": "last_activity_age_seconds",
    "activity_7d_quartile": "events_last_7d",
    "effective_prefix_length_quartile": "effective_prefix_length",
}


def top_overlap(cut: np.ndarray, proxy: np.ndarray, fraction: float = 0.1) -> dict:
    count = max(1, int(np.ceil(len(cut) * fraction)))
    # Stable ordering freezes the tie rule by original release-snapshot order.
    selected = set(np.argsort(-cut, kind="stable")[:count].tolist())
    truth = set(np.argsort(-proxy, kind="stable")[:count].tolist())
    overlap = len(selected & truth)
    return {"fraction": fraction, "count": count, "recall": overlap / len(truth), "precision": overlap / len(selected), "enrichment_over_random": overlap / len(truth) / fraction}


def bootstrap_validity(cut: np.ndarray, proxy: np.ndarray, seed: int = 37, draws: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    values = {"spearman": [], "top10_recall": []}
    for _ in range(draws):
        sample = rng.integers(0, len(cut), size=len(cut))
        values["spearman"].append(float(spearmanr(cut[sample], proxy[sample]).statistic))
        values["top10_recall"].append(top_overlap(cut[sample], proxy[sample])["recall"])
    return {name: {"p2_5": float(np.quantile(series, .025)), "p97_5": float(np.quantile(series, .975))} for name, series in values.items()}


def smd(covered: np.ndarray, uncovered: np.ndarray) -> float:
    return float((covered.mean() - uncovered.mean()) / np.sqrt((covered.var() + uncovered.var()) / 2 + 1e-12))


def qlabel(values: np.ndarray) -> np.ndarray:
    # Rank avoids duplicated quantile cut points when count features are tied.
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return np.minimum(3, ranks * 4 // len(values)) + 1


def edge_audit(edge: str, validity: dict) -> tuple[dict, list[dict]]:
    snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet").to_pydict()
    n = len(snapshot["uid"])
    covered_rows = validity["edges"][edge]["records"]
    covered_uids = {int(row["uid"]) for row in covered_rows}
    uid = np.asarray(snapshot["uid"], dtype=np.int64)
    covered = np.asarray([item in covered_uids for item in uid])
    columns = {name: np.asarray(snapshot[name], dtype=float) for name in FEATURES}
    no_observed = validity["edges"][edge]["skipped_reasons"]["no_observed_event"]
    after_next = int((~covered).sum()) - no_observed
    cohort = {
        "covered_states": int(covered.sum()),
        "uncovered_states": int((~covered).sum()),
        "proxy_coverage_rate": float(covered.mean()),
        "coverage_conservation": {
            "release_snapshot": n,
            "strict_proxy_before_next_release": int(covered.sum()),
            "first_observed_event_at_or_after_next_release": after_next,
            "no_observed_event_after_release": no_observed,
            "oov_or_listen_filter_exclusion": "not_applicable: canonical proxy consumes only event time, not event item",
        },
        "feature_balance": {},
    }
    conservation = cohort["coverage_conservation"]
    if (
        conservation["strict_proxy_before_next_release"]
        + conservation["first_observed_event_at_or_after_next_release"]
        + conservation["no_observed_event_after_release"]
        != n
    ):
        raise AssertionError("coverage categories do not conserve the snapshot")
    for name, values in columns.items():
        yes, no = values[covered], values[~covered]
        cohort["feature_balance"][name] = {
            "covered": {"mean": float(yes.mean()), "p50": float(np.quantile(yes, .5)), "p90": float(np.quantile(yes, .9))},
            "uncovered": {"mean": float(no.mean()), "p50": float(np.quantile(no, .5)), "p90": float(np.quantile(no, .9))},
            "standardized_mean_difference": smd(yes, no),
            "ks_distance": float(ks_2samp(yes, no).statistic),
        }
    delays = np.asarray([row["first_observed_event_delay_seconds"] for row in covered_rows], dtype=float)
    cohort["first_proxy_delay_seconds"] = {key: float(np.quantile(delays, q)) for key, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))}

    X = np.column_stack([columns[name] for name in FEATURES])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=37))
    oof = np.zeros(n)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=37)
    for train, test in folds.split(X, covered):
        classifier.fit(X[train], covered[train])
        oof[test] = classifier.predict_proba(X[test])[:, 1]
    cohort["proxy_availability_diagnostic_auc"] = float(roc_auc_score(covered, oof))

    by_uid = {int(row["uid"]): row for row in covered_rows}
    stratified, csv_rows = {}, []
    for label, feature in STRATA.items():
        groups = qlabel(columns[feature])
        stratified[label] = []
        for group in range(1, 5):
            in_group = groups == group
            rows = [by_uid[int(item)] for item in uid[in_group] if int(item) in by_uid]
            if len(rows) < 20:
                continue
            cut = np.asarray([row["cutover_top10_regret"] for row in rows])
            proxy = np.asarray([row["request_top10_regret"] for row in rows])
            item = {
                "quartile": group,
                "snapshot_states": int(in_group.sum()),
                "covered_states": len(rows),
                "coverage": len(rows) / int(in_group.sum()),
                "spearman": float(spearmanr(cut, proxy).statistic),
                "top10": top_overlap(cut, proxy),
                "bootstrap_95ci": bootstrap_validity(cut, proxy, seed=37 + group),
            }
            stratified[label].append(item)
            csv_rows.append({"edge_id": edge, "stratifier": label, **item, "top10_recall": item["top10"]["recall"], "top10_precision": item["top10"]["precision"]})
    cohort["stratified_proxy_validity"] = stratified
    return cohort, csv_rows


def main() -> None:
    root = Path("results/data_audit/yambda50m_v2")
    validity = json.loads((root / "cutover_probe_validity_v2.json").read_text())
    panel_results = {
        "a_to_a": validity,
        "a_to_b": json.loads((root / "cutover_probe_validity_panel_a_to_b_v1.json").read_text()),
        "b_to_a": json.loads((root / "cutover_probe_validity_panel_b_to_a_v1.json").read_text()),
    }
    panel_matrix = {}
    for edge in EDGES:
        panel_matrix[edge] = {
            key: {
                "spearman": value["edges"][edge]["primary_validity"]["spearman_rho"],
                "top10_recall": value["edges"][edge]["primary_validity"]["top_10pct_high_risk_recall"],
            }
            for key, value in panel_results.items()
        }
    result, rows = {
        "status": "strict_proxy_coverage_cohort_bias_development",
        "proxy_semantics": validity["evaluation_semantics"],
        "panel_robustness": {
            "matrix": panel_matrix,
            "gate": "failed_cross_panel_generalization",
            "interpretation": "single-panel cutover risk is not sufficiently panel-robust for ranker external-validity claims",
        },
        "edges": {},
    }, []
    for edge in EDGES:
        result["edges"][edge], csv_rows = edge_audit(edge, validity)
        rows.extend(csv_rows)
    (root / "proxy_coverage_cohort_bias_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "proxy_coverage_cohort_bias_v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["edge_id", "stratifier", "quartile", "snapshot_states", "covered_states", "coverage", "spearman", "top10", "bootstrap_95ci", "top10_recall", "top10_precision"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({edge: {key: value[key] for key in ("covered_states", "uncovered_states", "proxy_coverage_rate", "proxy_availability_diagnostic_auc")} for edge, value in result["edges"].items()}, indent=2))


if __name__ == "__main__":
    main()

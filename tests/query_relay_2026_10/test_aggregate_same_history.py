from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from query_relay_2026_10.aggregate_same_history import centered_quality, merge_gates
from query_relay_2026_10.analyze_same_history import run as old_analysis


def test_centered_bootstrap_matches_existing_analysis(tmp_path):
    rng, rows = np.random.default_rng(19), []
    users, versions = list(range(1, 9)), ["v3", "v4", "v5"]
    # Include an idle UID, unequal active set counts and exact ties.
    for uid in users[:-1]:
        for version in versions:
            for round_index in range(1 + uid % 3):
                wanted = rng.normal(size=16)
                dynamic = wanted + .03 * rng.normal(size=16) + .7
                constant = wanted + .05 * rng.normal(size=16) + .2
                if uid == 2 and round_index == 0:
                    dynamic[:] = constant[:] = .5
                for j in range(16):
                    rows.append(dict(uid=uid, version=version, round_id=f"{uid}:{version}:{round_index}",
                        label=int(j == 0), partial_logit=float(dynamic[j]),
                        same_set_mean_logit=float(constant[j]),
                        first_request_rebuild_logit=float(wanted[j])))
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path / "scores.parquet")
    (tmp_path / "summary.json").write_text(json.dumps(dict(selected_uids=users,
        configuration=dict(versions=versions))))
    old_analysis(tmp_path, repetitions=1000, seed=17)
    original = json.loads((tmp_path / "within_set_analysis.json").read_text())
    actual = centered_quality(table, users, versions, repetitions=1000, seed=17)
    for version in versions:
        for metric, value in original["by_release"][version].items():
            np.testing.assert_allclose(actual["by_release"][version][metric], value, atol=1e-18)
    for metric, record in original["paired"].items():
        for field in ("estimate", "percentile_95_interval", "finite_draws", "undefined_draws"):
            np.testing.assert_allclose(actual["paired"][metric][field], record[field], rtol=1e-13, atol=1e-15)


def gate_row(data, inactive=0):
    means, variance = data.mean(axis=2), data.var(axis=2)
    return dict(version="v3", layer=0, side="key", active_sets=len(data), inactive_sets=inactive,
        mean_gate=means.mean(axis=0).tolist(),
        within_set_coordinate_rms_std=np.sqrt(variance.mean(axis=0)).tolist(),
        mean_within_set_coordinate_std=float(np.sqrt(variance).mean()),
        rms_within_set_coordinate_std=float(np.sqrt(variance.mean())),
        rms_between_set_mean_std=float(np.sqrt(means.var(axis=0).mean())),
        nonconstant_coordinate_set_fraction=float((np.sqrt(variance) > 1e-5).mean()),
        saturated_coordinate_fraction=float(((data < .05) | (data > 1.95)).mean()))


def test_gate_moment_merge_recovers_between_set_variance():
    rng = np.random.default_rng(11)
    data = rng.uniform(.1, 1.9, size=(7, 2, 16, 3))
    data[:2, 0, :, 0] = .001
    # Deliberate separation of shard means makes a naive std average wrong.
    data[3:] *= .4
    summaries = [dict(gate_statistics=dict(rows=[gate_row(data[:3], inactive=2)])),
                 dict(gate_statistics=dict(rows=[gate_row(data[3:], inactive=1)]))]
    actual = merge_gates(summaries)["rows"][0]
    expected = gate_row(data, inactive=3)
    for key in ("active_sets", "inactive_sets", "mean_gate", "within_set_coordinate_rms_std",
                "mean_within_set_coordinate_std", "rms_within_set_coordinate_std",
                "rms_between_set_mean_std", "nonconstant_coordinate_set_fraction", "saturated_coordinate_fraction"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-14)

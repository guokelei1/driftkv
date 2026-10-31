#!/usr/bin/env python3
"""CPU E6 workload census from frozen UID roles and observed request keys.

No model outcomes select users. The feedback-union panel uses exactly the
qualification runner's predeclared UID hash and feedback-presence rule.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.common.artifacts import sha256


def count_population(rows, uids, *, name, edge, histories=None, release_timestamp=None, end_timestamp=None):
    selected = set(uids)
    per_user = defaultdict(list)
    for row in rows:
        if int(row["uid"]) in selected:
            per_user[int(row["uid"])].append(row)
    census = []
    for uid in uids:
        user = per_user[uid]
        times = {int(row["query_timestamp"]) for row in user}
        targets = {int(row["item_idx"]) for row in user}
        ordered = {}
        at_time = defaultdict(list)
        for row in sorted(user, key=lambda row: (row["query_timestamp"], row["item_idx"])):
            ordered.setdefault(int(row["item_idx"]), len(ordered))
            at_time[row["query_timestamp"]].append(int(row["item_idx"]))
        panel_reads = sum(len({ordered[item]//230 for item in items}) for items in at_time.values())
        record = dict(uid=uid, feedback_rows=len(user), timestamps=len(times),
            unique_targets=len(targets), fixed_panels=(len(targets)+229)//230,
            fixed_panel_snapshot_reads=panel_reads)
        if histories is not None:
            events = histories.rows[uid][0]
            prefix = int(np.searchsorted(events, release_timestamp, side="left"))
            initial = min(prefix, 1024)
            columns, cleared_reads = 0, 0
            for timestamp, items in at_time.items():
                appended = int(np.searchsorted(events, timestamp, side="left"))-prefix
                inherited = max(0, min(initial, 1024-appended))
                panels = len({ordered[item]//230 for item in items})
                columns += inherited*panels
                cleared_reads += panels*int(inherited == 0)
            record.update(initial_history_rows=initial,
                all_e14_append_events=int(np.searchsorted(events, end_timestamp, side="left"))-prefix,
                append_events_before_last_feedback=int(np.searchsorted(events, max(times), side="left"))-prefix,
                inherited_column_sum_over_panel_reads=columns, zero_inherited_panel_reads=cleared_reads)
        census.append(record)
    def distribution(key):
        values = np.asarray([row[key] for row in census])
        return dict(total=int(values.sum()), minimum=int(values.min()),
            q25=float(np.quantile(values, .25)), median=float(np.median(values)),
            q75=float(np.quantile(values, .75)), maximum=int(values.max()))
    timestamps = sum(row["timestamps"] for row in census)
    panel_reads = sum(row["fixed_panel_snapshot_reads"] for row in census)
    return dict(edge=edge, panel=name, users=len(uids),
        users_with_feedback=sum(row["feedback_rows"] > 0 for row in census),
        feedback_rows=distribution("feedback_rows"), timestamps=distribution("timestamps"),
        unique_targets=distribution("unique_targets"),
        fixed_panels=distribution("fixed_panels"),
        fixed_panel_snapshot_reads=distribution("fixed_panel_snapshot_reads"),
        causal_append_events=distribution("all_e14_append_events") if histories is not None else None,
        append_events_before_last_feedback=distribution("append_events_before_last_feedback") if histories is not None else None,
        inherited_columns_1024_row_equivalent_snapshots=sum(
            row.get("inherited_column_sum_over_panel_reads", 0) for row in census)/1024,
        zero_inherited_panel_reads=sum(row.get("zero_inherited_panel_reads", 0) for row in census),
        users_with_more_than_230_unique_targets=sum(row["unique_targets"] > 230 for row in census),
        fixed_230_panel_contains_all_real_targets=all(row["unique_targets"] <= 230 for row in census),
        matrix_decompositions_per_variant=dict(first_snapshot=100*len(uids), every_timestamp=100*timestamps,
            bounded_fixed_panels_every_timestamp=100*panel_reads),
        snapshot_multiplier_over_one_per_user=timestamps/len(uids),
        gpu_hours_at_snapshot_seconds={str(seconds): panel_reads*seconds/3600 for seconds in (1, 5, 10, 30)},
        per_user=census)


def run(input_root, output):
    population_path = input_root / "population.json"
    population = json.loads(population_path.read_text())
    union = set(population["main"] + population["confirmation"] +
        population["population_sensitivity"]["20261023"] + population["population_sensitivity"]["20261024"])
    reports, sources = [], {}
    for edge in range(1, 6):
        directory = input_root / f"edge_{edge}"
        binding_path = directory / "inputs.json"
        binding = json.loads(binding_path.read_text())
        request_path = directory / binding["requests"]["path"]
        history_path = directory / binding["history"]["path"]
        if binding["population_sha256"] != sha256(population_path) or sha256(request_path) != binding["requests"]["sha256"]:
            raise ValueError("workload census uses a changed request/role binding")
        if sha256(history_path) != binding["history"]["sha256"]:
            raise ValueError("workload census uses a changed causal raw-history pack")
        histories = restore_pack(history_path)
        rows = pq.read_table(request_path).to_pylist()
        active = {int(row["uid"]) for row in rows}
        feedback = sorted(union & active, key=lambda uid: hashlib.sha256(
            f"scenario-feedback-2000/{edge}/{uid}".encode()).digest())[:2000]
        for name, users in (("original_e6_users", binding["e6_users"]),
                            ("scenario_feedback_union_2000", feedback)):
            reports.append(count_population(rows, users, name=name, edge=edge, histories=histories,
                release_timestamp=binding["settings"]["release_day"]*86400,
                end_timestamp=binding["settings"]["end_day"]*86400))
        sources[str(edge)] = dict(input_binding_sha256=sha256(binding_path),
            requests_sha256=sha256(request_path))
    result = dict(role="CPU request-count/performance-planning census; no prediction or quality result",
        source_population_sha256=sha256(population_path), edge_inputs=sources,
        candidate_protocol="one fixed per-user/release panel including all E14 actual targets, uniformly filled to230; no labels or per-user model gains used",
        semantics="fixed panels remove repeated resampling but do not remove h dependence on actual query/cache snapshot; all-timestamp decompositions remain necessary for that protocol",
        estimates="1/5/10/30 second scenarios are conditional arithmetic, not measured GPU runtime",
        reports=reports)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    print(json.dumps([{key: value for key, value in report.items() if key != "per_user"} for report in reports]), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_root, args.output)

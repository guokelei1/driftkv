#!/usr/bin/env python3
"""Explore all retained NDCG pools/cutoffs without revising the frozen primary gate."""

import argparse
import json
from pathlib import Path

from daily_comparison import ROOT, digest, read, require

DEFAULT = ROOT / "results/recflow/development/window_6l_daily_lr_seed17"
BASELINE = ROOT / "results/recflow/development/window_6l_daily_seed17"
SCOPES = ("full_catalog", "uniform_1000", "popularity_1000")
KS = (20, 50, 100)


def summarize(root):
    cells, inputs, signatures = [], {}, {}
    for branch, rate, directory in (("lr1e3", 1e-3, BASELINE),
                                  ("lr1e4", 1e-4, root / "lr1e4"),
                                  ("lr3e5", 3e-5, root / "lr3e5")):
        for edge, day in (("AB", 20), ("BC", 21)):
            pair_path = directory / (f"{edge}_1epoch" if branch == "lr1e3" else f"{edge}_comparison") / "summary.json"
            pair = read(pair_path)
            random_path = Path(pair["random_comparison_file"])
            random = read(random_path)
            inputs.update({str(path): digest(path) for path in (pair_path, random_path)})
            matched = {row["run"]: row for row in random["results"]}
            require(pair["training_seed"] == 17 and pair["current"]["configuration"]["learning_rate"] == rate,
                    f"{branch}/{edge}: seed or learning rate differs")
            require(set(pair["primary"]["per_day"]) == {str(day)}, f"{branch}/{edge}: future day differs")
            for scope in SCOPES:
                panel_key = "full_catalog" if scope == "full_catalog" else "sampled"
                records, panels = {}, {}
                for role in ("parent", "current"):
                    source, cfg = pair[role], pair[role]["configuration"]
                    require(cfg["evaluation_window"] == f"{day}_{day}"
                            and cfg["evaluation_canary_limit"] is None, f"{branch}/{edge}: limited or wrong panel")
                    evidence = source["panel_evidence"][panel_key]
                    row = matched[source["run"]]["comparisons"][scope]
                    require(row["requests"] == evidence["requests"]
                            and row["request_indices_sha256"] == evidence["indices_sha256"]
                            and row["input_panel_sha256"] == evidence["sha256"],
                            f"{branch}/{edge}/{scope}: random denominator or panel differs")
                    expected = (10171 if day == 20 else 8866) if scope == "full_catalog" else 768
                    require(row["requests"] == expected, f"{branch}/{edge}/{scope}: request count differs")
                    signature = (row["requests"], row["request_indices_sha256"],
                                 row["candidate_count_min"], row["candidate_count_max"])
                    key = (edge, scope)
                    require(key not in signatures or signatures[key] == signature,
                            f"{branch}/{edge}/{scope}: cross-model pool or panel differs")
                    signatures[key] = signature
                    panels[role] = dict(requests=row["requests"], indices_sha256=row["request_indices_sha256"],
                        raw_panel_sha256=row["input_panel_sha256"], candidate_count_min=row["candidate_count_min"],
                        candidate_count_max=row["candidate_count_max"])
                    evaluation = source["evaluation"]
                    values = evaluation["full_catalog"] if scope == "full_catalog" else evaluation["sampled_candidate_diagnostics"][scope]
                    records[role] = (row, values, source["hit_counts"][scope])
                for k in KS:
                    models = {}
                    for role, (row, values, hits) in records.items():
                        metric = row["metrics"][f"ndcg@{k}"]
                        score, expectation, p99 = metric["trained"], metric["analytic_expectation"], metric["null_p99"]
                        require(score == values[f"ndcg@{k}"], f"{branch}/{edge}/{scope}: metric differs from evaluation")
                        passed = score >= 2 * expectation and score > p99
                        require(passed == metric["both_conditions"], f"{branch}/{edge}/{scope}: random flag differs")
                        models[role] = dict(ndcg=score, random_expectation=expectation, random_null_p99=p99,
                            random_checks_passed=passed, hit_requests=hits[str(k)]["requests"],
                            hit_users=hits[str(k)]["users"], panel=panels[role])
                    old, new = models["parent"]["ndcg"], models["current"]["ndcg"]
                    cells.append(dict(branch=branch, learning_rate=rate, edge=edge, future_day=day,
                        scope=scope, cutoff=k, **models, absolute_gain=new - old,
                        relative_gain_percent=100 * (new - old) / old if old else None,
                        descriptive_edge_work=new > old and all(v["random_checks_passed"] for v in models.values()),
                        comparison_file=str(pair_path), matched_random_file=str(random_path),
                        random_draws=random["draws"], random_seed=random["seed"]))
    settings = []
    for branch in ("lr1e3", "lr1e4", "lr3e5"):
        for scope in SCOPES:
            for k in KS:
                rows = [row for row in cells if (row["branch"], row["scope"], row["cutoff"]) == (branch, scope, k)]
                require([row["edge"] for row in rows] == ["AB", "BC"], "Missing or duplicated edge")
                gains = [row["relative_gain_percent"] for row in rows]
                defined = all(value is not None for value in gains)
                settings.append(dict(branch=branch, learning_rate=rows[0]["learning_rate"], scope=scope, cutoff=k,
                    relative_gains_percent=dict(zip(("AB", "BC"), gains)),
                    gain_range_percent=[min(gains), max(gains)] if defined else None,
                    gain_absolute_difference_percentage_points=abs(gains[1] - gains[0]) if defined else None,
                    both_edges_work=all(row["descriptive_edge_work"] for row in rows)))
    return dict(scope="Exploratory protocol grid: every recorded learning rate, candidate pool and NDCG cutoff.",
        training_seed=17, repeat_units=1, cells=cells, settings=settings, admissible=False,
        frozen_primary="The running learning-rate experiment retains full_catalog NDCG@50 and its original pass/fail. This grid does not revise those results.",
        task_boundary="Sampled uniform/popularity ranking uses each day's separate 768-request panel and actual candidate count. It is a different ranking task from full-catalog free generation; its denominator is never the full10171/8866 requests.",
        working_rule="Both edges must improve and every parent/current must reach twice matched random expectation and exceed random null99. No target gain percentage, combined score or automatic winner.",
        stability_note="Gain ranges and differences describe only two development edges; they are not statistical stability or model-seed uncertainty. Random draws describe random policies. Any chosen protocol needs subsequent confirmation.",
        input_sha256=inputs, source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "metric_grid"
    if output.exists():
        parser.error("Use a fresh output directory to preserve previous evidence.")
    report = summarize(args.root.resolve())
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("lr       scope             K      D20 gain      D21 gain   both_work")
    for row in report["settings"]:
        gains = [f"{v:+.2f}%" if v is not None else "undefined" for v in row["relative_gains_percent"].values()]
        print(f"{row['learning_rate']:<8g} {row['scope']:17} {row['cutoff']:3}  {gains[0]:>12}  {gains[1]:>12}   {row['both_edges_work']}")
    print(json.dumps(dict(output=str(output), cells=len(report["cells"]), settings=len(report["settings"]))))


if __name__ == "__main__":
    main()

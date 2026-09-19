#!/usr/bin/env python3
"""Report the two selected daily edges and their subsequent development probe."""

import argparse
import json
from pathlib import Path

from daily_comparison import ROOT, digest, read, require

DEFAULT = ROOT / "results/recflow/development/window_6l_daily_confirmation_seed17"
SETTINGS = ROOT / "configs/recflow/window_6l_daily_confirmation_seed17.json"
GRID = ROOT / "results/recflow/development/window_6l_daily_lr_seed17/metric_grid/summary.json"


def summarize(root, settings_path=SETTINGS, grid_path=GRID, comparison_path=None):
    inputs = {}

    def load(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return read(path)

    settings, grid = load(settings_path), load(grid_path)
    args = settings["arguments"]
    scope, metric_name, rate = settings["primary_scope"], settings["primary_metric"], args["learning_rate"]
    require((scope, metric_name, rate) in (("full_catalog", "ndcg@100", 1e-4),
                                          ("uniform_1000", "ndcg@50", 1e-3))
            and args["epochs"] == 1 and args["seed"] == 17,
            "Selected scope, metric, learning rate, epoch count or seed differs")
    cutoff = int(metric_name.split("@")[1])
    boundary = ("further_exploratory_development_after_day22_reuse" if scope == "uniform_1000"
                else "prospective_development_confirmation")
    require(settings["windows"]["D"]["fit"] == [21, 21]
            and settings["windows"]["D"]["compare_parent_and_current"] == [22, 22]
            and settings["windows"]["D"]["eligible_requests_per_epoch"] == 4312,
            "Sealed D dates or eligible request count differs")
    record = load(root / "D/summary.json")
    cfg = record["configuration"]
    require(record["complete"] and not cfg["canary"], "D is incomplete or a canary")
    require(cfg == load(root / "D/configuration.json"), "D configuration records differ")
    require(all(cfg[key] == value for key, value in args.items()), "D arguments differ from sealed settings")
    require(cfg["settings_sha256"] == digest(settings_path)
            and cfg["phase"] == "D" and cfg["start_epoch"] == 0
            and cfg["requested_complete_epochs"] == 1, "D settings hash, phase or epoch differs")
    require(cfg["training_window_days"] == [21, 21] and cfg["evaluation_window_days"] == [22, 22]
            and cfg["global_batch_size"] == 128 and cfg["optimizer_state_steps_before"] == [4166]
            and cfg["optimizer_learning_rates"] == [rate], "D dates, batch or actual AdamW start differs")
    require(record["eligible_requests"] == record["audited_eligible_requests"] == 4312
            and len(record["epochs"]) == 1, "D complete-window coverage differs")
    entry = record["epochs"][0]
    train = entry["training"]
    require(train["phase"] == "D" and train["epoch"] == 1 and train["complete"]
            and train["processed_requests"] == train["eligible_requests"] == 4312
            and train["optimizer_steps"] == 34, "D final epoch is not the complete 4312-request pass")
    require(load(root / "D/D_epoch1/training.json") ==
            {key: entry[key] for key in ("training", "fixed_training_nll")}, "Final saved D training record differs")
    source = (ROOT / settings["parent_checkpoint"]).resolve()
    source_sha = settings["parent_checkpoint_sha256"]
    require(Path(cfg["source_checkpoint"]).resolve() == source
            and cfg["source_checkpoint_sha256"] == source_sha
            and cfg["panels_sha256"] == settings["panels_sha256"], "D source C or panels differs")

    comparisons, pairs = {}, {}
    selected = [row for row in grid["cells"] if (row["learning_rate"], row["scope"], row["cutoff"])
                == (rate, scope, cutoff)]
    require([row["edge"] for row in selected] == ["AB", "BC"], "Selected exploratory edges missing or duplicated")
    paths = [(row["edge"], row["future_day"], Path(row["comparison_file"]), row) for row in selected]
    paths.append(("CD", 22, comparison_path or root / "CD_comparison/summary.json", None))
    for edge, day, path, exploratory in paths:
        pair = load(path)
        random_path = Path(pair["random_comparison_file"])
        random = load(random_path)
        require(pair["training_seed"] == 17 and random["draws"] == 5000 and random["seed"] == 20260918,
                f"{edge}: seed or matched random protocol differs")
        if exploratory:
            require(all(grid["input_sha256"][str(p)] == inputs[str(p.resolve())] for p in (path, random_path)),
                    f"{edge}: exploratory grid inputs changed")
        else:
            require(pair["primary_metric"] == f"{scope} {metric_name}", "CD comparison uses another primary")
        matched = {row["run"]: row["comparisons"] for row in random["results"]}
        models = {}
        for role in ("parent", "current"):
            item, ecfg = pair[role], pair[role]["configuration"]
            require(ecfg["evaluation_window"] == f"{day}_{day}" and ecfg["evaluation_canary_limit"] is None
                    and set(item["evaluation"]["per_day"]) == {str(day)}, f"{edge}: wrong or limited future panel")
            require(all(ecfg[key] == args[key] for key in ("candidate_seed", "catalog_size", "cohort_users",
                        "context", "decoder", "beam_width", "eval_precision", "sampled_distractors")),
                    f"{edge}: evaluation protocol differs")
            panels = item["panel_evidence"]
            for pool, row in matched[item["run"]].items():
                panel = panels["full_catalog" if pool == "full_catalog" else "sampled"]
                require(row["requests"] == panel["requests"]
                        and row["request_indices_sha256"] == panel["indices_sha256"]
                        and row["input_panel_sha256"] == panel["sha256"], f"{edge}/{pool}: random panel differs")
            require(panels["full_catalog"]["requests"] == {20: 10171, 21: 8866, 22: 9039}[day]
                    and panels["sampled"]["requests"] == 768, f"{edge}: full/sample denominators differ")
            full_null = matched[item["run"]]["full_catalog"]
            require(full_null["candidate_count_min"] == full_null["candidate_count_max"] == 1000000,
                    f"{edge}: random catalog differs")
            null = matched[item["run"]][scope]
            values = (item["evaluation"]["full_catalog"] if scope == "full_catalog"
                      else item["evaluation"]["sampled_candidate_diagnostics"][scope])
            metric = null["metrics"][metric_name]
            require(metric["trained"] == values[metric_name], f"{edge}: NDCG differs")
            passed = metric["trained"] >= 2 * metric["analytic_expectation"] and metric["trained"] > metric["null_p99"]
            require(passed == metric["both_conditions"], f"{edge}: random flag differs")
            models[role] = dict(ndcg=metric["trained"], random_expectation=metric["analytic_expectation"],
                random_null_p99=metric["null_p99"], random_checks_passed=passed,
                hits=item["hit_counts"][scope][str(cutoff)], panels=panels)
            if exploratory:
                require(exploratory[role]["ndcg"] == metric["trained"], f"{edge}: selected grid value differs")
        current = pair["current"]["configuration"]
        require(all(pair["parent"]["panel_evidence"][key][field] == pair["current"]["panel_evidence"][key][field]
                    for key in ("full_catalog", "sampled") for field in ("requests", "indices_sha256")),
                f"{edge}: parent/current panels differ")
        phase = {"AB": "B", "BC": "C", "CD": "D"}[edge]
        # The retained LR1e-3 B1 endpoint was saved during a declared three-epoch
        # run, before actual optimizer LR was added to evaluation metadata.
        actual_lrs = current.get("optimizer_learning_rates")
        require(actual_lrs == [rate] or (exploratory and rate == 1e-3 and actual_lrs is None),
                f"{edge}: actual optimizer learning rate differs")
        require(current["learning_rate"] == rate and current["evaluation_checkpoint_epoch"] == 1
                and (current["requested_complete_epochs"] == 1
                     or (exploratory and rate == 1e-3 and phase == "B" and current["requested_complete_epochs"] == 3))
                and current["evaluation_checkpoint_phase"] == phase
                and current["training_window_days"] == [day - 1, day - 1], f"{edge}: selected endpoint differs")
        old, new = (models[role]["ndcg"] for role in ("parent", "current"))
        comparisons[edge] = dict(boundary="exploratory_selection" if exploratory else boundary,
            future_day=day, **models, absolute_gain=new - old, current_over_parent=new / old if old else None,
            relative_gain_percent=100 * (new - old) / old if old else None,
            descriptive_edge_work=new > old and all(v["random_checks_passed"] for v in models.values()),
            comparison_file=str(path), matched_random_file=str(random_path))
        pairs[edge] = pair

    parent, current = (pairs["CD"][role]["configuration"] for role in ("parent", "current"))
    previous = pairs["BC"]["current"]["configuration"]
    require(previous["evaluation_checkpoint_sha256"] == parent["evaluation_checkpoint_sha256"] == source_sha
            and all(Path(c["evaluation_checkpoint"]).resolve() == source for c in (previous, parent))
            and parent["evaluation_checkpoint_phase"] == "C" and parent["evaluation_checkpoint_epoch"] == 1,
            "D evaluated parent is not the selected exact C1")
    require(current["evaluation_checkpoint_phase"] == "D" and current["evaluation_checkpoint_epoch"] == 1
            and Path(current["evaluation_checkpoint"]).resolve() == root / "D/D_epoch1/checkpoint.pt"
            and current["source_checkpoint_sha256"] == source_sha
            and current["optimizer_state_steps_before"] == [4166], "Evaluated current is not final saved D1")
    require(all(current[key] == cfg[key] for key in ("model", "catalog_sha256", "cohort_sha256",
                "settings_sha256", "source_checkpoint", "panels_sha256")), "Saved D evaluation configuration differs")
    require(all(c["evaluation_panels_sha256"] == settings["panels_sha256"]
                and Path(c["evaluation_panels"]).resolve() == Path(args["panels"]).resolve()
                for c in (parent, current)), "CD actual evaluation panels differ from the new sealed panels")
    require(entry["evaluation"] == pairs["CD"]["current"]["evaluation"]
            and record["parent_evaluation"] == pairs["CD"]["parent"]["evaluation"], "D recorded evaluations differ from pair")
    optimizer_check = load(root / "optimizer_check.json")
    require(optimizer_check["passed"] and optimizer_check["saved_optimizer_steps"] == [4200]
            and optimizer_check["saved_learning_rates"] == [rate]
            and optimizer_check["checkpoint_sha256"] == current["evaluation_checkpoint_sha256"]
            and optimizer_check["source_checkpoint_sha256"] == source_sha,
            "Serialized D AdamW/checkpoint check differs from the evaluated final model")
    return dict(scope=settings["scope"], training_seed=17, repeat_units=1, primary_metric=f"{scope} {metric_name}",
        learning_rate=rate, complete_epochs_per_update=1, comparisons=comparisons, evaluation_boundary=boundary,
        next_edge_working_flags_met=comparisons["CD"]["descriptive_edge_work"],
        confirmation_working_flags_met=comparisons["CD"]["descriptive_edge_work"],
        all_three_descriptive_edges_work=all(row["descriptive_edge_work"] for row in comparisons.values()),
        admissible=False, training=dict(configuration=cfg, epoch=entry["training"],
            pre_training_nll=record["pre_training_nll"], fixed_training_nll=entry["fixed_training_nll"],
            expected_optimizer_steps_after=4200, serialized_optimizer_check=optimizer_check,
            final_checkpoint_sha256=current["evaluation_checkpoint_sha256"]),
        lineage_verification=f"Recorded training/evaluation paths and SHA checked. A separate CPU mmap check reads actual saved AdamW step4200/LR{rate:g} and hashes the final checkpoint; this report verifies that evidence against the evaluated model. Historical exploratory endpoints may predate recorded actual LR; new D requires it. Pair reports validate raw request metrics.",
        selection=settings["selection"], data_boundary=settings["data_boundary"],
        interpretation=("AB/BC selected LR, pool and cutoff using exploratory results. "
            + ("CD is further exploratory development after the prior D22 failure, not independent confirmation; sampled ranking is a different task and cannot rescue the failed free-generation result. "
               if scope == "uniform_1000" else "CD tests the subsequent frozen choice; D22 was previously seen by other branches. ")
            + "Preserve negative, failed or large gains; no desired percentage gate. One training seed and three development edges do not establish stable gains; random trials are not uncertainty for the update."),
        all_exploratory_settings_file=str(grid_path), input_sha256=inputs,
        source_sha256={str(path): digest(path) for path in (Path(__file__).resolve(), Path(__file__).with_name("daily_comparison.py"))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "daily_confirmation"
    if output.exists():
        parser.error("Use a fresh output directory to preserve earlier evidence.")
    report = summarize(root, args.settings.resolve(), args.grid.resolve(), args.comparison)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("edge  day  parent        current       delta         gain       hits(requests/users)  work")
    for edge, row in report["comparisons"].items():
        old, new = row["parent"], row["current"]
        gain = f"{row['relative_gain_percent']:+.2f}%" if row["relative_gain_percent"] is not None else "undefined"
        print(f"{edge:4}  {row['future_day']}   {old['ndcg']:.7g}  {new['ndcg']:.7g}  {row['absolute_gain']:+.7g}  "
              f"{gain:>9}  {old['hits']['requests']}/{old['hits']['users']} -> "
              f"{new['hits']['requests']}/{new['hits']['users']}  {row['descriptive_edge_work']}")
    print(json.dumps(dict(output=str(output), primary_metric=report["primary_metric"],
        evaluation_boundary=report["evaluation_boundary"], next_edge_working_flags_met=report["next_edge_working_flags_met"],
        all_three_descriptive_edges_work=report["all_three_descriptive_edges_work"], admissible=False)))


if __name__ == "__main__":
    main()

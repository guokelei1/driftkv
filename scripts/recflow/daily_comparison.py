#!/usr/bin/env python3
"""Describe all four predeclared daily endpoints; never select or admit one."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/recflow/development/window_6l_daily_seed17"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(root):
    inputs, runs = {}, {}
    for name, phase, epochs, day, count, steps, before in (
        ("B_day19", "B", 3, 19, 6929, 55, 4065),
        ("C_day20_1epoch", "C", 1, 20, 5805, 46, 4120),
        ("C_day20_3epoch", "C", 3, 20, 5805, 46, 4230),
    ):
        path = root / name / "summary.json"
        record = read(path)
        cfg = record["configuration"]
        require(record["complete"] and not cfg["canary"], f"{name}: incomplete or canary")
        require(cfg["phase"] == phase and cfg["seed"] == 17 and cfg["start_epoch"] == 0
                and cfg["requested_complete_epochs"] == epochs, f"{name}: phase/epoch/seed differs")
        require(cfg["training_window_days"] == [day, day]
                and cfg["evaluation_window_days"] == [day + 1, day + 1], f"{name}: day boundary differs")
        require(cfg["global_batch_size"] == 128 and cfg["optimizer_state_steps_before"] == [before],
                f"{name}: global batch or inherited AdamW steps differ")
        require(record["eligible_requests"] == record["audited_eligible_requests"] == count,
                f"{name}: complete-window count differs")
        training = [entry["training"] for entry in record["epochs"]]
        require([t["epoch"] for t in training] == list(range(1, epochs + 1)), f"{name}: missing epoch")
        require(all(t["complete"] and t["processed_requests"] == t["eligible_requests"] == count
                    and t["optimizer_steps"] == steps for t in training), f"{name}: incomplete epoch")
        inputs[str(path)] = digest(path)
        runs[name] = dict(configuration=cfg,
            epochs=[{key: entry[key] for key in ("training", "fixed_training_nll")}
                    for entry in record["epochs"]],
            optimizer_steps_before=before, expected_optimizer_steps_after=before + epochs * steps,
            pre_training_nll=record["pre_training_nll"])
    one, three = (runs[f"C_day20_{epoch}epoch"] for epoch in (1, 3))
    for key in ("order_sha256", "sampled_targets_sha256"):
        require(one["epochs"][0]["training"][key] == three["epochs"][0]["training"][key],
                f"C branches: first-epoch {key} differs")
    for key in ("indices_sha256", "sampled_targets_sha256"):
        require(one["pre_training_nll"][key] == three["pre_training_nll"][key],
                f"C branches: training-only probe {key} differs")
    comparisons, endpoints, panels = {}, {}, {}
    initial = runs["B_day19"]["configuration"]
    for edge in ("AB", "BC"):
        for epoch in (1, 3):
            name = f"{edge}_{epoch}epoch"
            path = root / name / "summary.json"
            pair = read(path)
            parent, current = (pair[role]["configuration"] for role in ("parent", "current"))
            run_name = "B_day19" if edge == "AB" else f"C_day20_{epoch}epoch"
            cfg = runs[run_name]["configuration"]
            phase, day = ("B", 20) if edge == "AB" else ("C", 21)
            checkpoint = root / run_name / f"{phase}_epoch{epoch}" / "checkpoint.pt"
            require(pair["training_seed"] == 17 and pair["primary_metric"] == "full_catalog ndcg@50",
                    f"{name}: seed or primary metric differs")
            require(Path(current["evaluation_checkpoint"]).resolve() == checkpoint
                    and current["evaluation_checkpoint_epoch"] == epoch
                    and current["evaluation_checkpoint_phase"] == phase, f"{name}: wrong current checkpoint")
            require(current["source_checkpoint_sha256"] == cfg["source_checkpoint_sha256"]
                    and Path(parent["evaluation_checkpoint"]).resolve() == Path(cfg["source_checkpoint"]).resolve()
                    and parent["evaluation_checkpoint_sha256"] == cfg["source_checkpoint_sha256"],
                    f"{name}: parent checkpoint lineage differs")
            require(current["training_window_days"] == [day - 1, day - 1]
                    and current["optimizer_state_steps_before"] == cfg["optimizer_state_steps_before"],
                    f"{name}: current fit or optimizer metadata differs")
            require(all(c["evaluation_window"] == f"{day}_{day}" and c["evaluation_canary_limit"] is None
                        for c in (parent, current)), f"{name}: wrong future window or limited panel")
            require(set(pair["primary"]["per_day"]) == {str(day)}, f"{name}: actual evaluated days differ")
            evidence = pair["current"]["panel_evidence"]
            signature = {key: value["indices_sha256"] for key, value in evidence.items()}
            require(evidence["full_catalog"]["requests"] == (10171 if day == 20 else 8866),
                    f"{name}: incomplete future panel")
            require(edge not in panels or panels[edge] == signature, f"{name}: cross-branch panels differ")
            panels[edge] = signature
            if edge == "AB":
                fit = parent.get("training_window_days") or read(Path(parent["settings_path"]))["windows"]["A"]["fit"]
                require(parent["evaluation_checkpoint_phase"] == "A"
                        and parent["evaluation_checkpoint_epoch"] == 3 and fit == [1, 18], f"{name}: wrong common A3")
                endpoints[epoch] = (checkpoint, current["evaluation_checkpoint_sha256"])
            else:
                source, sha = endpoints[epoch]
                require(Path(cfg["source_checkpoint"]).resolve() == source
                        and cfg["source_checkpoint_sha256"] == sha, f"{name}: B1/B3 branch or SHA swapped")
                require(parent["training_window_days"] == [19, 19], f"{name}: parent fit overlaps evaluation")
            primary = pair["primary"]
            old, new = primary["parent"]["trained"], primary["current"]["trained"]
            multiple = new / old if old else None
            work = new > old and all(primary[role]["both_conditions"] for role in ("parent", "current"))
            inputs[str(path)] = digest(path)
            comparisons[name] = dict(primary=primary, absolute_gain=new - old, current_over_parent=multiple,
                relative_gain_percent=100 * (multiple - 1) if multiple is not None else None,
                descriptive_edge_work=work,
                hits={role: pair[role]["hit_counts"]["full_catalog"] for role in ("parent", "current")},
                companions_file=str(path), matched_random_file=pair["random_comparison_file"])
    branches = {f"{epoch}epoch": all(comparisons[f"{edge}_{epoch}epoch"]["descriptive_edge_work"]
                for edge in ("AB", "BC")) for epoch in (1, 3)}
    return dict(scope="All predeclared seed17 daily development endpoints; no winner selection or admission.",
        training_seed=17, repeat_units=1, primary_metric="full_catalog ndcg@50",
        common_A3=dict(checkpoint=initial["source_checkpoint"], sha256=initial["source_checkpoint_sha256"]),
        training=runs, comparisons=comparisons, branch_work=branches,
        all_tested_settings_work=all(branches.values()),
        any_tested_setting_work=any(branches.values()), admissible=False,
        lineage_verification="Training/evaluation recorded checkpoint paths and SHA agree; AdamW start steps plus complete epoch steps checked. No weight file loaded; final optimizer steps are inferred, not re-read.",
        interpretation="D20 compares one shared D19 training trajectory at epochs1/3; D21 compares cumulative dose policies. Different inherited states prevent isolating only the second update's epoch count. All companion metrics remain linked; random trials are not seed repeats or uncertainty for the gain. This is not a same-window daily-versus-three-day schedule comparison.",
        input_sha256=inputs, source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "daily_comparison"
    if output.exists():
        parser.error("Use a fresh output directory to preserve earlier evidence.")
    report = summarize(root)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("endpoint       parent       current      delta        multiple    work    hits(requests/users)")
    for name, row in report["comparisons"].items():
        p = row["primary"]
        multiple = f"{row['current_over_parent']:.3f}x" if row["current_over_parent"] is not None else "undefined"
        hits = [row["hits"][role]["50"] for role in ("parent", "current")]
        print(f"{name:14} {p['parent']['trained']:.6g}  {p['current']['trained']:.6g}  "
              f"{row['absolute_gain']:+.6g}  {multiple:>9}  {str(row['descriptive_edge_work']):5}  "
              f"{hits[0]['requests']}/{hits[0]['users']} -> {hits[1]['requests']}/{hits[1]['users']}")
    print(json.dumps(dict(output=str(output), branch_work=report["branch_work"],
                         all_tested_settings_work=report["all_tested_settings_work"],
                         any_tested_setting_work=report["any_tested_setting_work"], admissible=False)))


if __name__ == "__main__":
    main()

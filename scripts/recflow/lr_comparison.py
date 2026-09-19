#!/usr/bin/env python3
"""Report every daily learning-rate branch against the retained one-epoch baseline."""

import argparse
import json
from pathlib import Path

from daily_comparison import ROOT, digest, read, require

DEFAULT = ROOT / "results/recflow/development/window_6l_daily_lr_seed17"
BASELINE = ROOT / "results/recflow/development/window_6l_daily_seed17"
SHARED = ("model", "catalog_sha256", "cohort_sha256", "panels_sha256", "seed",
          "candidate_seed", "context", "batch_size", "weight_decay", "clip_norm",
          "training_order", "decoder", "beam_width", "eval_precision", "sampled_distractors")


def summarize(root):
    inputs, reference, training, comparisons, panel_signatures = {}, {}, {}, {}, {}

    def load(path):
        inputs[str(path)] = digest(path)
        return read(path)

    for phase, name in (("B", "B_day19"), ("C", "C_day20_1epoch")):
        reference[phase] = load(BASELINE / name / "summary.json")
    common = reference["B"]["configuration"]
    initial = dict(checkpoint=common["source_checkpoint"], sha256=common["source_checkpoint_sha256"])
    branches = (("lr1e3", 1e-3, BASELINE), ("lr1e4", 1e-4, root / "lr1e4"),
                ("lr3e5", 3e-5, root / "lr3e5"))
    for branch, rate, directory in branches:
        old_branch = branch == "lr1e3"
        training[branch], comparisons[branch] = {}, {}
        endpoint = None
        for phase, edge, day, count, steps, before in (
                ("B", "AB", 19, 6929, 55, 4065), ("C", "BC", 20, 5805, 46, 4120)):
            name = ("B_day19" if phase == "B" else "C_day20_1epoch") if old_branch else phase
            run = reference[phase] if old_branch else load(directory / name / "summary.json")
            cfg, ref = run["configuration"], reference[phase]
            label = f"{branch}/{phase}"
            require(run["complete"] and not cfg["canary"], f"{label}: incomplete or canary")
            require(cfg["phase"] == phase and cfg["start_epoch"] == 0
                    and cfg["learning_rate"] == rate, f"{label}: phase, resume or learning rate differs")
            require(all(cfg[key] == common[key] for key in SHARED), f"{label}: frozen settings differ")
            require(cfg["seed"] == 17 and cfg["global_batch_size"] == 128
                    and cfg["optimizer_state_steps_before"] == [before], f"{label}: seed/batch/AdamW differs")
            require(cfg["training_window_days"] == [day, day]
                    and cfg["evaluation_window_days"] == [day + 1, day + 1], f"{label}: day boundary differs")
            require(run["eligible_requests"] == run["audited_eligible_requests"] == count,
                    f"{label}: fitting population differs")
            if not old_branch:
                require(cfg["requested_complete_epochs"] == 1 and len(run["epochs"]) == 1,
                        f"{label}: expected one complete epoch")
                require(cfg["optimizer_learning_rates"] == [rate], f"{label}: restored optimizer lr differs")
            entry = run["epochs"][0]
            fit = entry["training"]
            require(fit["epoch"] == 1 and fit["complete"]
                    and fit["eligible_requests"] == fit["processed_requests"] == count
                    and fit["optimizer_steps"] == steps, f"{label}: incomplete first epoch")
            for key in ("order_sha256", "sampled_targets_sha256"):
                require(fit[key] == ref["epochs"][0]["training"][key], f"{label}: {key} differs from baseline")
            for probe in (run["pre_training_nll"], entry["fixed_training_nll"]):
                require(all(probe[key] == ref["pre_training_nll"][key]
                            for key in ("indices_sha256", "sampled_targets_sha256", "requests")),
                        f"{label}: training-only probe changed")
            training[branch][phase] = dict(summary_file=str(directory / name / "summary.json"),
                training=fit, pre_training_nll=run["pre_training_nll"],
                fixed_training_nll=entry["fixed_training_nll"],
                optimizer_learning_rates=cfg.get("optimizer_learning_rates"),
                optimizer_steps_before=before, inferred_optimizer_steps_after=before + steps)

            pair_path = directory / (f"{edge}_1epoch" if old_branch else f"{edge}_comparison") / "summary.json"
            pair = load(pair_path)
            parent, current = (pair[role]["configuration"] for role in ("parent", "current"))
            checkpoint = (directory / name / f"{phase}_epoch1/checkpoint.pt").resolve()
            require(pair["training_seed"] == 17 and pair["primary_metric"] == "full_catalog ndcg@50",
                    f"{label}: evaluation seed or metric differs")
            require(Path(current["evaluation_checkpoint"]).resolve() == checkpoint
                    and current["evaluation_checkpoint_epoch"] == 1
                    and current["evaluation_checkpoint_phase"] == phase, f"{label}: wrong evaluated endpoint")
            require(all(current[key] == cfg[key] for key in SHARED +
                        ("learning_rate", "training_window_days", "optimizer_state_steps_before")),
                    f"{label}: endpoint configuration differs from training")
            if not old_branch:
                require(current["optimizer_learning_rates"] == [rate], f"{label}: evaluated optimizer lr differs")
            require(current["source_checkpoint_sha256"] == cfg["source_checkpoint_sha256"]
                    and Path(parent["evaluation_checkpoint"]).resolve() == Path(cfg["source_checkpoint"]).resolve()
                    and parent["evaluation_checkpoint_sha256"] == cfg["source_checkpoint_sha256"],
                    f"{label}: parent/training/evaluation lineage differs")
            if phase == "B":
                require(cfg["source_checkpoint_sha256"] == initial["sha256"]
                        and Path(cfg["source_checkpoint"]).resolve() == Path(initial["checkpoint"]).resolve()
                        and parent["evaluation_checkpoint_phase"] == "A"
                        and parent["evaluation_checkpoint_epoch"] == 3, f"{label}: common A3 differs")
            else:
                require((Path(cfg["source_checkpoint"]).resolve(), cfg["source_checkpoint_sha256"]) == endpoint,
                        f"{label}: C did not inherit this branch's B1")
                require(parent["training_window_days"] == [19, 19], f"{label}: parent fit differs")
            endpoint = (checkpoint, current["evaluation_checkpoint_sha256"])
            future = day + 1
            require(all(c["evaluation_window"] == f"{future}_{future}"
                        and c["evaluation_canary_limit"] is None for c in (parent, current))
                    and set(pair["primary"]["per_day"]) == {str(future)}, f"{label}: wrong future panel")
            for role in ("parent", "current"):
                evidence = pair[role]["panel_evidence"]
                signature = {key: (value["requests"], value["indices_sha256"]) for key, value in evidence.items()}
                require(evidence["full_catalog"]["requests"] == (10171 if future == 20 else 8866),
                        f"{label}: incomplete future panel")
                require(edge not in panel_signatures or panel_signatures[edge] == signature,
                        f"{label}: parent/current or cross-branch panels differ")
                panel_signatures[edge] = signature
            primary = pair["primary"]
            old, new = primary["parent"]["trained"], primary["current"]["trained"]
            random_checks = {}
            for role in ("parent", "current"):
                row = primary[role]
                passed = row["trained"] >= 2 * row["analytic_expectation"] and row["trained"] > row["null_p99"]
                require(passed == row["both_conditions"], f"{label}: random check differs")
                random_checks[role] = dict(expectation=row["analytic_expectation"], null_p99=row["null_p99"],
                                           passed=passed)
            comparisons[branch][edge] = dict(learning_rate=rate, future_day=future,
                parent_ndcg50=old, current_ndcg50=new, absolute_gain=new - old,
                relative_gain_percent=100 * (new - old) / old if old else None,
                descriptive_edge_work=new > old and all(v["passed"] for v in random_checks.values()),
                random_checks=random_checks,
                top50_hits={role: pair[role]["hit_counts"]["full_catalog"]["50"] for role in ("parent", "current")},
                comparison_file=str(pair_path), matched_random_file=pair["random_comparison_file"])

    working = {branch: all(row["descriptive_edge_work"] for row in edges.values())
               for branch, edges in comparisons.items()}
    return dict(scope="All predeclared daily learning-rate settings, including the retained one-epoch baseline.",
        training_seed=17, repeat_units=1, primary_metric="full_catalog ndcg@50", common_A3=initial,
        training=training, comparisons=comparisons, branch_work=working, admissible=False,
        relative_gain_definition="100 * (current - parent) / parent, on the same future request panel",
        user_preference="The suggested 20–50 percent gain is descriptive preference, never a gate or winner-selection rule.",
        lineage_verification="Recorded checkpoint paths/SHA and AdamW start steps checked; final optimizer steps inferred from complete epochs. New branches record actual restored optimizer learning rates. No weights loaded.",
        interpretation="D20 isolates the first update learning rate from shared A3; D21 compares cumulative daily policies with different parents. All outcomes are retained. One seed and two development days do not establish long-term stability; random-policy draws are not uncertainty for the update.",
        input_sha256=inputs, source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "lr_comparison"
    if output.exists():
        parser.error("Use a fresh output directory to preserve previous evidence.")
    report = summarize(root)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("lr       day    parent       current      relative_gain     work    hits(requests/users)")
    for edges in report["comparisons"].values():
        for row in edges.values():
            gain = row["relative_gain_percent"]
            text = f"{gain:+.2f}%" if gain is not None else "undefined"
            hits = [row["top50_hits"][role] for role in ("parent", "current")]
            print(f"{row['learning_rate']:<8g} {row['future_day']:2}     {row['parent_ndcg50']:.6g}  "
                  f"{row['current_ndcg50']:.6g}  {text:>12}     {str(row['descriptive_edge_work']):5}   "
                  f"{hits[0]['requests']}/{hits[0]['users']} -> {hits[1]['requests']}/{hits[1]['users']}")
    print(json.dumps(dict(output=str(output), branch_work=report["branch_work"], admissible=False)))


if __name__ == "__main__":
    main()

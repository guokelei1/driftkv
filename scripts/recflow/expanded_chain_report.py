#!/usr/bin/env python3
"""Read the completed prefix of the fixed4096-user daily development chain."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from daily_comparison import digest, read, require

PHASES = "ABCDEF"
SCOPE, METRIC = "uniform_1000", "ndcg@50"


def summarize(root, settings_path):
    root, settings_path = Path(root).resolve(), Path(settings_path).resolve()
    inputs, phases, edges, endpoint, base, waiting = {}, {}, {}, None, None, None

    def load(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return read(path)

    settings = load(settings_path)
    args = settings["arguments"]
    require(args["seed"] == 17 and args["cohort_users"] == 4096 and args["batch_size"] == 128
            and args["learning_rate"] == 1e-3 and args["epochs"] == 3,
            "Expanded chain seed, population, batch, learning rate or base epoch setting differs")
    require(settings["primary_scope"] == SCOPE and settings["primary_metric"] == METRIC,
            "Expanded chain primary changed")
    panel_path = Path(args["panels"]).resolve()
    manifest_path = panel_path.with_name("summary.json")
    manifest = load(manifest_path)
    panel_sha = digest(panel_path)
    inputs[str(panel_path)] = panel_sha
    require(panel_sha == settings["panels_sha256"] == manifest["arrays_file_sha256"] and manifest["cohort_users"] == 4096,
            "Expanded panels or population differ")
    signatures = {}
    with np.load(panel_path) as panels:
        for day in range(19, 25):
            key = f"{day}_{day}"
            full, selected = panels[f"eval_{key}"], panels[f"sampled_{key}"]
            sample = full[np.isin(full, selected)]
            audited = manifest["evaluation"][key]
            require(len(sample) == len(selected) == audited["sampled_panel"]["requests"] == 6144,
                    f"D{day}: sampled panel differs")
            require(len(full) == audited["free_panel"]["requests"], f"D{day}: full panel differs")
            signatures[day] = {name: dict(requests=len(indices), indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
                               for name, indices in (("full_catalog", full), ("sampled", sample))}

    def check_evaluation(configuration, evaluation, day):
        require(configuration["evaluation_window"] == f"{day}_{day}"
                and configuration["evaluation_canary_limit"] is None
                and set(evaluation["per_day"]) == {str(day)}, f"D{day}: wrong or limited evaluation")
        require(configuration["evaluation_panels_sha256"] == panel_sha
                and Path(configuration["evaluation_panels"]).resolve() == panel_path,
                f"D{day}: evaluated another panel file")
        for key in ("catalog_sha256", "cohort_sha256"):
            require(configuration[key] == manifest[key], f"D{day}: evaluated another {key}")
        for key in ("seed", "candidate_seed", "catalog_size", "cohort_users", "context",
                    "decoder", "beam_width", "eval_precision", "sampled_distractors"):
            require(configuration[key] == args[key], f"D{day}: evaluation {key} differs")
        for name, field in (("full_catalog", "full_request_panel"), ("sampled", "sampled_request_panel")):
            require(all(evaluation[field][key] == value for key, value in signatures[day][name].items()),
                    f"D{day}: {name} counts/order differ")

    def random_score(metric, score):
        require(metric["trained"] == score, "Matched random report refers to another model score")
        passed = metric["analytic_expectation"] > 0 and score >= 2 * metric["analytic_expectation"] and score > metric["null_p99"]
        require(passed == metric["both_conditions"], "Matched random flag differs")
        return dict(ndcg=score, random_expectation=metric["analytic_expectation"],
                    random_null_p99=metric["null_p99"], random_checks_passed=passed)

    expected_step = 0
    for phase in PHASES:
        directory = root / phase
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            waiting = str(summary_path)
            break
        record = load(summary_path)
        if not record["complete"]:
            waiting = f"{phase}: training/evaluation incomplete"
            break
        epoch_count, day = (3, 19) if phase == "A" else (1, 19 + PHASES.index(phase))
        fit_days = [1, 18] if phase == "A" else [day - 1, day - 1]
        fit_key = "train_1_18" if phase == "A" else f"update_{day-1}_{day-1}"
        window = settings["windows"][phase]
        require(window["fit"] == fit_days and window.get("evaluate", window.get("compare_parent_and_current")) == [day, day],
                f"{phase}: sealed dates differ")
        cfg = record["configuration"]
        require(cfg == load(directory / "configuration.json") and not cfg["canary"], f"{phase}: configuration/canary differs")
        require(all(cfg[key] == value for key, value in args.items()), f"{phase}: declared arguments differ")
        require(cfg["settings_sha256"] == inputs[str(settings_path)]
                and cfg["window_manifest_sha256"] == inputs[str(manifest_path)]
                and cfg["panels_sha256"] == panel_sha, f"{phase}: settings or panel hashes differ")
        require(all(cfg[key] == manifest[key] for key in ("catalog_sha256", "cohort_sha256")), f"{phase}: catalog/cohort differs")
        require(cfg["phase"] == phase and cfg["start_epoch"] == 0 and cfg["requested_complete_epochs"] == epoch_count
                and cfg["training_window_days"] == fit_days and cfg["evaluation_window_days"] == [day, day]
                and cfg["global_batch_size"] == 128 and cfg["optimizer_learning_rates"] == [args["learning_rate"]],
                f"{phase}: phase, epoch, dates, batch or actual learning rate differs")
        require(cfg["optimizer_state_steps_before"] == ([] if phase == "A" else [expected_step]), f"{phase}: AdamW start differs")
        if endpoint is None:
            require(cfg["source_checkpoint"] is None and cfg["source_checkpoint_sha256"] is None, "A must start fresh")
        else:
            require((Path(cfg["source_checkpoint"]).resolve(), cfg["source_checkpoint_sha256"]) == endpoint,
                    f"{phase}: predecessor checkpoint lineage differs")
            require(cfg["model"] == phases["A"]["model"], f"{phase}: model architecture differs")
        audit = manifest["training"][fit_key]
        count, steps = audit["known_positive_requests"], audit["global_batch128_steps_per_epoch"]
        require(window["eligible_requests_per_epoch"] == record["eligible_requests"] == record["audited_eligible_requests"] == count
                and steps == (count + 127) // 128, f"{phase}: complete fitting window differs")
        require([e["training"]["epoch"] for e in record["epochs"]] == list(range(1, epoch_count + 1)), f"{phase}: epochs missing")
        for entry in record["epochs"]:
            train = entry["training"]
            require(train["phase"] == phase and train["complete"] and train["processed_requests"] == train["eligible_requests"] == count
                    and train["optimizer_steps"] == steps, f"{phase}: incomplete epoch")
            require(all(entry["fixed_training_nll"][key] == record["pre_training_nll"][key]
                        for key in ("indices_sha256", "sampled_targets_sha256")), f"{phase}: NLL probe changed")
        evaluation_dir = directory / f"{phase}_epoch{epoch_count}" / "evaluation"
        evaluation_record = load(evaluation_dir / "summary.json")
        ecfg = evaluation_record["configuration"]
        labels = [key for key in evaluation_record if key != "configuration"]
        require(len(labels) == 1, f"{phase}: ambiguous endpoint evaluation")
        evaluation = evaluation_record[labels[0]]
        require(evaluation == record["epochs"][-1]["evaluation"], f"{phase}: saved endpoint evaluation differs")
        check_evaluation(ecfg, evaluation, day)
        checkpoint = directory / f"{phase}_epoch{epoch_count}" / "checkpoint.pt"
        require(checkpoint.exists() and Path(ecfg["evaluation_checkpoint"]).resolve() == checkpoint
                and ecfg["evaluation_checkpoint_phase"] == phase and ecfg["evaluation_checkpoint_epoch"] == epoch_count
                and all(ecfg[key] == cfg[key] for key in ("model", "settings_sha256", "source_checkpoint_sha256",
                                                         "optimizer_state_steps_before", "optimizer_learning_rates")),
                f"{phase}: final checkpoint metadata differs")
        previous_endpoint = endpoint
        endpoint = checkpoint, ecfg["evaluation_checkpoint_sha256"]
        before = expected_step
        expected_step += epoch_count * steps
        phases[phase] = dict(summary_file=str(summary_path), model=cfg["model"], fitting_days=fit_days,
            complete_epochs=epoch_count, requests_per_epoch=count, steps_per_epoch=steps,
            optimizer_steps_before=before, inferred_optimizer_steps_after=expected_step,
            checkpoint=str(checkpoint), recorded_checkpoint_sha256=endpoint[1],
            evaluation_file=str(evaluation_dir / "summary.json"), evaluation_panels=signatures[day],
            training_seconds=sum(e["training"]["seconds"] for e in record["epochs"]),
            fixed_training_nll=[record["pre_training_nll"]["mean_nll"]] + [e["fixed_training_nll"]["mean_nll"] for e in record["epochs"]])
        optimizer_path = directory / "optimizer_check.json"
        if optimizer_path.exists():
            check = load(optimizer_path)
            require(check["passed"] and check["phase"] == phase and check["epoch"] == epoch_count
                    and check["saved_optimizer_steps"] == [expected_step] and check["saved_learning_rates"] == [args["learning_rate"]]
                    and check["optimizer_state_steps_before"] == cfg["optimizer_state_steps_before"]
                    and Path(check["checkpoint"]).resolve() == checkpoint and check["checkpoint_sha256"] == endpoint[1]
                    and check["source_checkpoint_sha256"] == cfg["source_checkpoint_sha256"],
                    f"{phase}: serialized optimizer/checkpoint verification differs")
            phases[phase]["serialized_optimizer_check"] = check
        if phase == "A":
            path = root / "A_random/summary.json"
            if not path.exists():
                waiting = str(path)
                break
            random = load(path)
            require(random["draws"] == 5000 and random["seed"] == 20260918, "A random protocol differs")
            rows = [r for r in random["results"] if Path(r["run"]).resolve() == evaluation_dir]
            require(len(rows) == 1, "A random evaluation is missing or duplicated")
            null = rows[0]["comparisons"][SCOPE]
            require(null["requests"] == 6144 and null["request_indices_sha256"] == signatures[19]["sampled"]["indices_sha256"],
                    "A random uses another sampled denominator")
            base = random_score(null["metrics"][METRIC], evaluation["sampled_candidate_diagnostics"][SCOPE][METRIC])
            raw = evaluation_dir / f"{labels[0]}_sampled_request_metrics.npz"
            require(digest(raw) == null["input_panel_sha256"], "A sampled evidence differs from matched random")
            with np.load(raw) as arrays:
                hit = arrays[f"{SCOPE}__recall@50"] > 0
                base.update(hit_requests=int(hit.sum()), hit_users=int(len(np.unique(arrays["uids"][hit]))))
            base.update(future_day=19, sampled_panel=signatures[19]["sampled"], random_file=str(path))
            continue
        edge = PHASES[PHASES.index(phase) - 1] + phase
        path = root / f"{edge}_comparison/summary.json"
        if not path.exists():
            waiting = str(path)
            break
        pair = load(path)
        require(pair["primary_metric"] == f"{SCOPE} {METRIC}" and pair["training_seed"] == 17,
                f"{edge}: primary or seed differs")
        random = load(pair["random_comparison_file"])
        require(random["draws"] == 5000 and random["seed"] == 20260918, f"{edge}: random protocol differs")
        matched = {row["run"]: row["comparisons"][SCOPE] for row in random["results"]}
        models = {}
        for role, expected in (("parent", previous_endpoint), ("current", endpoint)):
            item, pcfg = pair[role], pair[role]["configuration"]
            require((Path(pcfg["evaluation_checkpoint"]).resolve(), pcfg["evaluation_checkpoint_sha256"]) == expected,
                    f"{edge}: evaluated checkpoint lineage differs")
            check_evaluation(pcfg, item["evaluation"], day)
            for name in ("full_catalog", "sampled"):
                require(all(item["panel_evidence"][name][key] == value for key, value in signatures[day][name].items()),
                        f"{edge}: panel evidence differs")
            null = matched[item["run"]]
            require(null["metrics"][METRIC] == pair["primary"][role]
                    and null["requests"] == 6144 and null["request_indices_sha256"] == signatures[day]["sampled"]["indices_sha256"]
                    and null["input_panel_sha256"] == item["panel_evidence"]["sampled"]["sha256"],
                    f"{edge}: matched random score/panel differs")
            models[role] = random_score(null["metrics"][METRIC], item["evaluation"]["sampled_candidate_diagnostics"][SCOPE][METRIC])
            models[role].update(hits=item["hit_counts"][SCOPE]["50"])
        require(pair["current"]["evaluation"] == evaluation and pair["parent"]["evaluation"] == record["parent_evaluation"],
                f"{edge}: pair differs from completed training record")
        old, new = models["parent"]["ndcg"], models["current"]["ndcg"]
        edges[edge] = dict(future_day=day, **models, absolute_gain=new - old,
            relative_gain_percent=100 * (new - old) / old if old else None,
            working_flags_met=new > old and all(v["random_checks_passed"] for v in models.values()),
            evaluation_panels=signatures[day], comparison_file=str(path), matched_random_file=pair["random_comparison_file"])

    completed = len(phases) == 6 and len(edges) == 5 and base is not None
    return dict(scope="Expanded4096-user descriptive development chain; no formal admission or cache qualification.",
        completed=completed, waiting_for=waiting, training_seed=17, repeat_units=1,
        primary_metric=f"{SCOPE} {METRIC}", phases=phases, base=base, edges=edges,
        all_completed_quality_checks_pass=(base is not None and base["random_checks_passed"]
                                         and all(row["working_flags_met"] for row in edges.values())),
        admissible=False, lineage_verification="Checkpoint paths and recorded SHA checked across training/evaluation. Expected AdamW endpoints are inferred from start steps and complete epochs. Where serialized_optimizer_check is present, retained CPU mmap verification also substantiates the actual saved state and weight-file hash. This reporter does not load weights.",
        continuation_rule="Retain every quality failure and continue the fixed descriptive chain; numerical, causal or lineage failures stop execution. Fixed endpoints are not admitted releases.",
        interpretation="Primary sampled ranking uses6144 requests per day, injected known positives plus1000 uniform distractors, and its own OOV/random denominator. Full-catalog companions use their separate complete panels. All companion metrics and failures remain in linked pair reports. One seed and five development edges do not establish independent replications or untouched final qualification.",
        input_sha256=inputs, source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root, args.config)
    output = args.root / "chain_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), completed=report["completed"], waiting_for=report["waiting_for"],
                         completed_edges=len(report["edges"]), all_completed_quality_checks_pass=report["all_completed_quality_checks_pass"])))


if __name__ == "__main__":
    main()

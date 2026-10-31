#!/usr/bin/env python3
"""Fresh three anchors for the predeclared long-history active 2,000-user cohort."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code import common, qualify_scenario
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population, load_scenario_data
from hstu_kvcache.evaluation import fast_baselines as fast
from hstu_kvcache.common.artifacts import sha256


def run(args):
    source_manifest = json.loads(args.population_file.read_text())
    binding, users, population, provenance = load_scenario_population(args.population_file, args.edge,
        minimum_users=2000, require_qualification=False, require_positive=False)
    if (len(users) != 2000 or provenance["selection"]["panel_name"] != "long_feedback_active" or
            provenance["selection"]["definition"] != dict(retained_history_rows_at_least=512,
                pre14_behavior_events_at_least=32, pre14_feedback_events_at_least=4)):
        raise ValueError("use the existing fixed active-cohort definition and all2,000 users")
    strata = Path(provenance["selection"]["strata_manifest_path"])
    if sha256(strata) != provenance["selection"]["strata_manifest_sha256"]:
        raise ValueError("publishing-time scenario selection changed")
    selected = json.loads(strata.read_text())["subgroups_by_edge"][str(args.edge)]["long_feedback_active"]["first_2000_uid_hashes"]
    if users != selected:
        raise ValueError("natural cohort differs from its original feature/hash selection")
    common.quality_role(users)
    if args.check_inputs:
        return dict(status="inputs_checked", edge=args.edge, users=len(users), provenance=provenance,
                    quality_auc_computed=False, gpu_used=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    code = qualify_scenario.sources()
    for name in ("experiments/ch3_motivation/code/scenario_inputs.py", "experiments/ch3_motivation/code/qualify_natural_scenario.py"):
        code[name] = sha256(ROOT / name)
    cfg = binding["settings"]
    histories, by_user = load_scenario_data(provenance)
    parent, current, models = common.load_models(args.edge, device)
    identity = dict(experiment="fixed_natural_three_anchors", edge=args.edge, models=models,
        input_binding_sha256=provenance["input_binding_sha256"], source_population_sha256=provenance["source_population_sha256"],
        scenario=provenance, settings=cfg, sources=code,
        old_full="parent checkpoint for initial state, causal behavior writes and scoring",
        current_rebuild="once before first history consumer; Current native writes afterward",
        current_reuse="same old initial state; Current native writes afterward")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("natural anchor attempt identity changed")
    common.write_json(path, identity)
    started = time.perf_counter()
    all_scores, execution = [], {}
    for branch, model, policies in (
        ("old_full", parent, [dict(tag="old_full", method="reuse", calibration_flops=0)]),
        ("current", current, [dict(tag=name, method=name, calibration_flops=0) for name in ("reuse", "rebuild")])):
        folder = output / branch
        folder.mkdir(exist_ok=True)
        def progress(value):
            row = dict(edge=args.edge, branch=branch, **value)
            common.write_json(output / "progress.json", row)
            print(json.dumps(row), flush=True)
        scores, resources = fast.evaluate_checkpointed(parent, model, histories, by_user, users,
            policies, cfg, device, folder, identity=dict(identity, branch=branch), progress=progress)
        all_scores.extend(scores)
        execution[branch] = resources
    summary = qualify_scenario.bootstrap_anchors(all_scores, users, replicates=1000, seed=28121 + args.edge)
    summary.update(status="complete", edge=args.edge, panel="long_feedback_active", identity=identity,
        evaluation_window=cfg.get("evaluation_window", dict(name="E14", complete_source_horizon=True)), execution=execution,
        seconds_excluding_input_model_loading=time.perf_counter() - started,
        scope="predeclared feature/hash cohort; all five edges retained regardless of anchor signs; no method outcomes used for selection")
    for name, digest in code.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("natural anchor numerical source changed")
    if sha256(args.population_file) != provenance["manifest_sha256"]:
        raise ValueError("natural anchor population manifest changed")
    common.write_json(output / "summary.json", summary)
    manifest = copy.deepcopy(source_manifest)
    manifest["qualification"] = dict(summary_path=str((output / "summary.json").resolve()),
        summary_sha256=sha256(output / "summary.json"), scope=summary["scope"],
        **{key:summary[key] for key in ("new_rebuild_minus_old_full", "new_rebuild_minus_reuse", "point_qualifies", "uid95_qualifies")})
    common.write_json(output / "scenario_manifest.json", manifest)
    # Audit the resulting measured manifest including negative/unqualified edges.
    load_scenario_population(output / "scenario_manifest.json", args.edge, minimum_users=2000, require_positive=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-file", type=Path, required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key:result[key] for key in ("status", "edge")}), flush=True)

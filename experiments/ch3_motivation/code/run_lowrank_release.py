#!/usr/bin/env python3
"""E6 joint request-instance matrices, one publishing row space per release.

This is a separate protocol from per-timestamp independent SVD. Every actual
E14 feedback instance is retained, including repeated targets and cleared
reads. Only the trained FG/full-H references write state; ranks intervene on
copied full-H snapshots. Diagnostic mode never computes AUC.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import pyarrow.parquet as pq
import torch

from experiments.ch3_motivation.code.common import load_models, write_json
from experiments.ch3_motivation.code.fg_trajectory_control import (
    TRAINED_FG_METHOD, H_DISABLED_DIAGNOSTIC, load_fg_control,
)
from experiments.ch3_motivation.code.probe_projected_fast import ProjectedGraphPool
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.release_lowrank import VARIANTS, evaluate_release_user
from experiments.ch3_motivation.code.sampling_identity import sampling_source_paths
from experiments.ch3_motivation.code.run_lowrank import (
    scenario_inputs, scenario_qualification_record, validate_fit_roles, bootstrap_retention, _write_block,
)
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.common.artifacts import sha256


def sources():
    names = ("run_lowrank_release.py", "release_lowrank.py", "run_lowrank.py", "analysis.py",
        "lowrank_batch.py", "lowrank_rank_batch.py", "energy_certificate.py", "fg_trajectory_control.py", "scenario_inputs.py",
        "probes.py", "probe_variants.py", "probe_replay.py", "probe_fast.py", "probe_projected_fast.py",
        "probe_cost.py", "common.py")
    paths = [Path(__file__).with_name(name) for name in names]
    paths.extend(sampling_source_paths())
    paths.extend((ROOT/"src/hstu_kvcache/models").glob("*.py"))
    paths.extend([ROOT/"src/hstu_kvcache/evaluation/trajectory.py", ROOT/"src/hstu_kvcache/adaptation/reader.py",
        ROOT/"src/hstu_kvcache/evaluation/statistics.py", ROOT/"src/hstu_kvcache/evaluation/history.py",
        ROOT/"src/hstu_kvcache/evaluation/yambda.py", ROOT/"src/hstu_kvcache/common/artifacts.py"])
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(args.threads)
    if min(args.checkpoint_users, args.candidates, args.batch_heads) < 1:
        raise ValueError("positive score block, candidate panel, and head batch sizes required")
    diagnostic = args.diagnostic_users is not None
    binding, users, population, provenance = scenario_inputs(args.input_root, args.edge,
        population_file=args.population_file, diagnostic=diagnostic)
    if diagnostic:
        if not 1 <= args.diagnostic_users <= 16:
            raise ValueError("bounded release readiness uses1..16 diagnostic users")
        users = users[:args.diagnostic_users]
    elif args.sensitivity_users is not None:
        if not 512 <= args.sensitivity_users <= len(users):
            raise ValueError("additional protocol sensitivity needs at least512 frozen users")
        users = users[:args.sensitivity_users]
    elif len(users) < 2000:
        raise ValueError("main release E6 requires at least2000 feedback users")
    if diagnostic and args.sensitivity_users is not None:
        raise ValueError("diagnostic and quality-sensitivity roles must be separate")
    if args.compare_rank_execution and not diagnostic:
        raise ValueError("serial/batched duplicate scoring is a bounded numerical diagnostic")
    qualification = scenario_qualification_record(provenance)
    histories = restore_pack(provenance["history_path"])
    by_user = defaultdict(lambda: defaultdict(list))
    at, end = binding["settings"]["release_day"]*86400, binding["settings"]["end_day"]*86400
    for row in pq.read_table(provenance["requests_path"], filters=[("uid", "in", users)]).to_pylist():
        if not at <= int(row["query_timestamp"]) < end or row["label"] not in (0, 1):
            raise ValueError("real feedback timestamp or binary label outside the frozen protocol")
        by_user[int(row["uid"])][int(row["query_timestamp"])].append(row)
    if any(uid not in histories.rows or not by_user.get(uid) for uid in users):
        raise ValueError("every selected feedback user must have causal history and actual feedback")
    code = sources()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    parent, current, models = load_models(args.edge, device)
    probe, fit = load_variant(args.probe, device)
    probe.eval().requires_grad_(False)
    if probe.stage != "fgh":
        raise ValueError("joint E6 needs the frozen full-H probe")
    fitting = validate_fit_roles(fit, population, users, args.edge, models, provenance["calibration_input_binding_sha256"])
    fg, _, fg_identity = load_fg_control(args.fg_probe, probe, fit, device=device,
        source_population=population, evaluation_uids=users, edge=args.edge, models=models,
        calibration_binding_sha256=provenance["calibration_input_binding_sha256"],
        expected_method_label=args.fg_method_label, allow_diagnostic=diagnostic)
    if provenance.get("anchor_models", models) != models:
        raise ValueError("scenario anchor backbones differ from the release diagnostic")
    probe.cfg.query_chunk, probe.cfg.row_chunk = args.candidate_chunk, args.row_chunk
    variants = tuple(args.variants.split(","))
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError("unknown joint E6 K/V variant")
    role = "numerical_performance_diagnostic" if diagnostic else (
        "additional_fixed_subset_protocol_sensitivity" if args.sensitivity_users is not None else "main_2000_feedback")
    identity = dict(protocol="joint_e14_request_instance_release_v1", edge=args.edge, users=users, role=role,
        population=provenance, scenario_qualification=qualification, models=models, full_h_fitting=fitting,
        full_h_checkpoint_sha256=sha256(args.probe), trained_fg_control=fg_identity, sources=code,
        candidates=args.candidates, candidate_seed=20261022, variants=variants, ranks=[0, 1, 2, 4, 6, 8, 16],
        decomposition=dict(method=args.method, oversample=args.oversample, power_iterations=args.power_iterations,
            batch_heads=args.batch_heads, verification_heads=args.verify_exact_heads,
            batch_scope="flattened layers*heads, layer-major; identical per-layer random sketches across batch sizes",
            energy_guard=dict(enabled=True, ranks=[2, 6, 16], threshold=.98,
                relative_energy_tolerance=1e-5,
                failure="exact SVD on the same centered normalized matrix for only the failed head",
                scope="numerical screen with measured orthogonality defect and allowances; not a floating-point proof"),
            verification_scope="first non-cleared panel per user" if args.verify_every_user else "first non-cleared panel of the attempt",
            verification_selection="evenly spaced flattened layer/head indices, across each K/V variant"),
        execution=dict(mode=args.execution, candidate_chunk=args.candidate_chunk, row_chunk=args.row_chunk,
            batch_rank_scores=args.batch_rank_scores, compare_rank_execution=args.compare_rank_execution,
            checkpoint_users=args.checkpoint_users),
        candidate_instances="all actual event instances; duplicate ordinal distinguishes same-time repeated targets; chronological panels padded to230 using deterministic samples at their own real contexts",
        matrix_columns="all publishing-time immutable old history rows; FIFO only crops actual read columns, no mask injected into SVD",
        state="full-H and trainedFG own causal E14 writes; ranks only read copied full-H CPU snapshots; hindsight, no rank-specific appends",
        main_retention_baseline=TRAINED_FG_METHOD, structural_baseline=H_DISABLED_DIAGNOSTIC)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = output/"configuration.json"
    if config.exists() and json.loads(config.read_text()) != json.loads(json.dumps(identity)):
        raise ValueError("joint release attempt identity changed; use a distinct attempt")
    write_json(config, identity)
    blocks = output/"blocks"
    blocks.mkdir(exist_ok=True)
    full_pool = ProjectedGraphPool(current, probe, window=binding["settings"]["history_length"]) if args.execution == "projected" else None
    fg_pool = ProjectedGraphPool(current, fg, window=binding["settings"]["history_length"]) if args.execution == "projected" else None
    rows, reports, verified = [], [], set()
    started = time.perf_counter()
    for offset in range(0, len(users), args.checkpoint_users):
        selected = users[offset:offset+args.checkpoint_users]
        marker_path = blocks/f"part_{offset:06d}.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker["identity"] != digest or marker["report"]["uids"] != selected or sha256(blocks/marker["parquet"]) != marker["sha256"]:
                raise ValueError("joint release committed score block changed")
            rows.extend(pq.read_table(blocks/marker["parquet"]).to_pylist())
            reports.extend(marker["report"]["users"])
            for report in marker["report"]["users"]:
                verified.update(check["variant"] for panel in report["panels"] for check in panel["svd_checks"])
            continue
        block_rows, block_reports = [], []
        for uid in selected:
            values, report = evaluate_release_user(parent, current, probe, fg, histories.rows[uid], by_user[uid], uid,
                binding["settings"], known_items=current.cfg.num_items-256, candidate_size=args.candidates,
                execution=args.execution, fg_pool=fg_pool, full_h_pool=full_pool,
                verification_state=None if args.verify_every_user else verified,
                panel_options=dict(variants=variants, method=args.method, oversample=args.oversample,
                    power_iterations=args.power_iterations, batch_heads=args.batch_heads,
                    batch_rank_scores=args.batch_rank_scores, verify_exact_heads=args.verify_exact_heads,
                    compare_rank_execution=args.compare_rank_execution))
            block_rows.extend(values)
            block_reports.append(report)
            print(json.dumps(dict(status="release_user_complete", uid=uid, users=offset+len(block_reports),
                total=len(users), seconds=report["seconds"], cpu_snapshot_bytes=report["cpu_snapshot_bytes"])), flush=True)
        _write_block(blocks, offset, block_rows, dict(uids=selected, users=block_reports), digest)
        rows.extend(block_rows)
        reports.extend(block_reports)
        write_json(output/"progress.json", dict(users=len(reports), total=len(users), seconds=time.perf_counter()-started))
    if sources() != code:
        raise ValueError("joint release numerical sources changed while executing")
    rank_checks = [check for report in reports for panel in report["panels"]
        for frame in panel["frames"] for check in frame["rank_execution_checks"]]
    exact_check_seconds = sum(panel["exact_check_seconds"] for report in reports for panel in report["panels"])
    alternate_execution_seconds = sum(check["alternate_execution_seconds"] for check in rank_checks)
    panels = [panel for report in reports for panel in report["panels"]]
    result = dict(status="complete", role=role, users=len(users), identity_hash=digest,
        seconds=time.perf_counter()-started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        numerical_diagnostic_overhead=dict(exact_check_seconds=exact_check_seconds,
            alternate_rank_execution_seconds=alternate_execution_seconds,
            excluded_from_formal_runtime_projection="exact oracle checks and alternate rank execution are diagnostic costs; graph setup and actual causal writes remain part of the measured path"),
        rank_execution_comparison=dict(checks=len(rank_checks),
            logit_max_error=max((check["logit_max_error"] for check in rank_checks), default=None),
            serial_seconds=sum(check["serial_seconds"] for check in rank_checks),
            batched_seconds=sum(check["batched_seconds"] for check in rank_checks)),
        energy_guard=dict(matrices=sum(panel["energy_guard"]["matrices"] for panel in panels),
            passed_without_exact=sum(panel["energy_guard"]["passed_without_exact"] for panel in panels),
            exact_fallback_matrices=sum(panel["energy_guard"]["exact_fallback_matrices"] for panel in panels),
            worst_ratio_lower_bound=min((panel["energy_guard"]["worst_ratio_lower_bound"] for panel in panels
                if panel["energy_guard"]["worst_ratio_lower_bound"] is not None), default=None),
            guard_seconds=sum(panel["energy_guard_seconds"] for panel in panels),
            exact_fallback_seconds=sum(panel["exact_fallback_seconds"] for panel in panels),
            scope="all matrices screened; failed heads alone use same-matrix exact SVD; full sketch spectra saved in blocks"),
        cpu_snapshot_max_bytes=max(report["cpu_snapshot_bytes"] for report in reports),
        actual_feedback_instances=sum(report["actual_feedback_instances"] for report in reports),
        cleared_feedback_instances=sum(report["cleared_feedback_instances"] for report in reports),
        panels=sum(len(report["panels"]) for report in reports),
        full_rank_max_error=max(panel["full_rank_max_error"] for report in reports for panel in report["panels"]),
        scenario_qualification=qualification, main_retention_baseline=TRAINED_FG_METHOD,
        structural_baseline=H_DISABLED_DIAGNOSTIC, quality_auc_computed=not diagnostic,
        e6_main_result_eligible=not diagnostic and args.sensitivity_users is None and len(users) >= 2000,
        full_h_graph_pool=full_pool.report() if full_pool else None, trained_fg_graph_pool=fg_pool.report() if fg_pool else None,
        scope=identity["state"])
    if not diagnostic:
        groups = defaultdict(list)
        for row in rows:
            groups[row["method"]].append(row)
        points, confidence = bootstrap_retention(groups, users)
        structural = confidence.pop("same_state_h_disabled_diagnostic")
        result.update(points=points, confidence=confidence, same_state_h_disabled_diagnostic=structural)
        fields = ("method", "baseline_method", "auc", "auc_gain_over_baseline",
                  "interaction_gain_retention", "interaction_gain_retention_ci95")
        for name, selected_points in (("points.csv", points), ("same_state_structure_points.csv", structural["points"])):
            with (output/name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(selected_points)
    write_json(output/("diagnostic.json" if diagnostic else "summary.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--population-file", type=Path)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--fg-probe", type=Path, required=True)
    parser.add_argument("--fg-method-label", choices=("fg", "fg_extended"), default="fg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--diagnostic-users", type=int)
    parser.add_argument("--sensitivity-users", type=int)
    parser.add_argument("--candidates", type=int, default=230)
    parser.add_argument("--candidate-chunk", type=int, default=230)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--checkpoint-users", type=int, default=4)
    parser.add_argument("--variants", default="both_norm")
    parser.add_argument("--method", choices=("exact", "randomized"), default="randomized")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--batch-heads", type=int, default=10)
    parser.add_argument("--verify-exact-heads", type=int, default=2)
    parser.add_argument("--verify-every-user", action="store_true")
    parser.add_argument("--batch-rank-scores", action="store_true")
    parser.add_argument("--compare-rank-execution", action="store_true")
    parser.add_argument("--execution", choices=("dense", "projected"), default="projected")
    run(parser.parse_args())

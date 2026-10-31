#!/usr/bin/env python3
"""E6 hindsight read-compression on the frozen full-h E14 trajectory.

At every actual feedback timestamp the full-h trajectory supplies one cache
snapshot. All compressed ranks and the h-disabled structural control read that
same snapshot. The main E5 trained-FG baseline owns a separate E14 trajectory,
with its own fitted weights and behavior writes. Rank interventions still
measure hindsight read compression on fixed full-H state, not independent
rank-specific trajectories or a release-only prediction task.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.ch3_motivation.code.analysis import decompose_interaction
from experiments.ch3_motivation.code.common import load_models, quality_role, write_json
from experiments.ch3_motivation.code.fg_trajectory_control import (
    TRAINED_FG_METHOD, H_DISABLED_DIAGNOSTIC, load_fg_control, make_paired_trajectories,
    score_fg_timestamp,
)
from experiments.ch3_motivation.code.lowrank_batch import BatchedReconstruction, decompose_interaction_batch
from experiments.ch3_motivation.code.lowrank_rank_batch import reconstruct_rank_batch
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population
from experiments.ch3_motivation.code.sampling_identity import sampling_source_paths
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, ProjectedGraphPool, project_row_features,
    projected_interaction_correction,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

VARIANTS = dict(both_norm=dict(only="both", normalize_kv=True),
                both_raw=dict(only="both", normalize_kv=False),
                key_norm=dict(only="key", normalize_kv=True),
                value_norm=dict(only="value", normalize_kv=True))


def scenario_inputs(input_root, edge, *, population_file=None, diagnostic=False):
    """Use the shared checked scenario loader; original roles are diagnostic only."""
    if population_file is not None:
        return load_scenario_population(population_file, edge, role="e6",
            minimum_users=1 if diagnostic else 2000, require_qualification=not diagnostic,
            require_positive=False)
    if not diagnostic:
        raise ValueError("formal E6 requires --population-file for a qualified feedback population")
    directory = Path(input_root) / f"edge_{edge}"
    binding_path = directory / "inputs.json"
    binding = json.loads(binding_path.read_text())
    population_path = Path(input_root) / "population.json"
    if sha256(population_path) != binding["population_sha256"]:
        raise ValueError("original diagnostic population changed")
    population = json.loads(population_path.read_text())
    paths = {name: directory / binding[name]["path"] for name in ("history", "requests")}
    for name, path in paths.items():
        if sha256(path) != binding[name]["sha256"]:
            raise ValueError(f"original diagnostic {name} changed")
    users = binding["e6_users"]
    if set(population["fit"]) & set(population["validation"]) or set(users) & set(
            population["fit"] + population["validation"]):
        raise ValueError("diagnostic users overlap fitting/validation roles")
    provenance = dict(mode="original_unqualified_numerical_diagnostic",
        input_binding_path=str(binding_path.resolve()), input_binding_sha256=sha256(binding_path),
        calibration_input_binding_path=str(binding_path.resolve()),
        calibration_input_binding_sha256=sha256(binding_path),
        history_path=str(paths["history"].resolve()), history_sha256=sha256(paths["history"]),
        requests_path=str(paths["requests"].resolve()), requests_sha256=sha256(paths["requests"]),
        source_population_path=str(population_path.resolve()), source_population_sha256=sha256(population_path))
    return binding, users, population, provenance


def validate_fit_roles(metadata, source_population, users, edge, models, input_binding_sha256):
    """Check the two actual supported fitting formats, not inferred architectures."""
    identity = metadata["identity"]
    if identity["edge"] != edge or identity["models"] != models:
        raise ValueError("probe was fitted for different edge/backbone checkpoints")
    if identity["input_binding_sha256"] != input_binding_sha256:
        raise ValueError("probe fitting used another frozen raw-data binding")
    if "fit_uids" in identity:
        fit = identity["fit_uids"]
    else:
        # The actual old head checkpoint format records a count. Its runner
        # used the first C frozen fitting UIDs, which this binding reconstructs.
        fit = source_population["fit"][:identity["fit_users"]]
    validation = identity["validation_users"]
    if not fit or not validation or not set(fit) <= set(source_population["fit"]) or not set(
            validation) <= set(source_population["validation"]):
        raise ValueError("probe fitting roles are not the frozen reserved users")
    if set(fit) & set(validation) or set(users) & set(fit + validation):
        raise ValueError("probe fit/validation/evaluation users overlap")
    if metadata.get("fitting_uses_future_feedback") is True:
        raise ValueError("E6 does not accept a future-label-fitted probe")
    return dict(fit_uids=fit, validation_uids=validation, training_seed=identity["seed"],
        variant=identity.get("variant", "head"), fit_metadata_sha256=hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()).hexdigest())


def scenario_qualification_record(provenance):
    """Keep per-edge adverse anchors/UID uncertainty visible in E6 outputs."""
    if "qualification_summary_path" not in provenance:
        return dict(scope="unqualified numerical diagnostic")
    summary = json.loads(Path(provenance["qualification_summary_path"]).read_text())
    fields = ("new_rebuild_minus_old_full", "new_rebuild_minus_old_full_ci95",
              "new_rebuild_minus_reuse", "new_rebuild_minus_reuse_ci95",
              "point_qualifies", "uid95_qualifies")
    result = {key: summary[key] for key in fields}
    result["evaluation_window"] = summary.get("evaluation_window")
    result["limitation"] = ("This edge does not show a positive Rebuild-minus-OldFull point estimate; compression remains a conditional cache-mismatch diagnostic, not recovered model-update benefit."
        if summary["new_rebuild_minus_old_full"] <= 0 else
        "Positive update/cache-gap point estimates are retained with their UID intervals; no per-user or per-edge pruning.")
    return result


def candidate_panel(requests, *, uid, timestamp, known_items, size=230, seed=20261022):
    """Include every real candidate now; unique uniform samples fill the panel."""
    if known_items < 1 or size < 1 or not requests:
        raise ValueError("a candidate panel requires real requests and a nonempty catalog")
    actual = [int(row["item_idx"]) for row in requests]
    panel = list(dict.fromkeys(actual))
    if any(item <= 0 or item > known_items for item in panel):
        raise ValueError("feedback target outside the fixed known-item catalog")
    required = max(size, len(panel))
    if required > known_items:
        raise ValueError("candidate panel exceeds the fixed known-item catalog")
    key = hashlib.sha256(f"ch3-e6/{seed}/{uid}/{timestamp}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(key[:8], "little"))
    occupied = set(panel)
    while len(panel) < required:
        for item in generator.integers(1, known_items + 1, size=2 * (required - len(panel))):
            item = int(item)
            if item not in occupied:
                panel.append(item)
                occupied.add(item)
                if len(panel) == required:
                    break
    positions = {item: index for index, item in enumerate(panel)}
    return panel, [positions[item] for item in actual]


def release_candidate_panels(requests_by_timestamp, *, uid, release_timestamp, known_items,
                             size=230, seed=20261022):
    """Fixed hindsight panels include every actual target without giant panels.

    Targets are ordered by first timestamp then item ID. Users with more than
    `size` distinct targets receive deterministic target blocks; each is filled
    to `size`. An actual target has one designated panel even if a distractor
    sample elsewhere happens to include it. No user or feedback row is dropped.
    """
    if size < 1:
        raise ValueError("fixed candidate panel size must be positive")
    ordered = {}
    for timestamp, rows in sorted(requests_by_timestamp.items()):
        for row in sorted(rows, key=lambda row: int(row["item_idx"])):
            ordered.setdefault(int(row["item_idx"]), None)
    targets = list(ordered)
    panels, coordinates = [], {}
    for start in range(0, len(targets), size):
        block = targets[start:start+size]
        index = len(panels)
        panel, positions = candidate_panel([dict(item_idx=item) for item in block], uid=uid,
            timestamp=f"release{release_timestamp}/panel{index}", known_items=known_items, size=size, seed=seed)
        panels.append(panel)
        coordinates.update({item: (index, position) for item, position in zip(block, positions, strict=True)})
    return panels, coordinates


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compact_factors(decomposition):
    """Release full-SVD U/V storage hidden behind top-r tensor views."""
    decomposition.left = decomposition.left.clone()
    decomposition.singular_values = decomposition.singular_values.clone()
    decomposition.right = decomposition.right.clone()
    return decomposition


@torch.no_grad()
def compress_snapshot(model, probe, trajectory, panel, actual_positions, query_delta, *,
                      ranks=(0, 1, 2, 4, 6, 8, 16), variants=("both_norm",),
                      method="randomized", oversample=8, power_iterations=1,
                      seed=17, verify_svd_heads=0, decomposition_batch_heads=10,
                      diagnostic_h_callback=None, enforce_svd_accuracy=True,
                      interaction_execution="projected", batch_rank_scores=False):
    """Trace one real snapshot; replay fixed h_r through the complete model.

    The full-rank check uses exactly the full candidate panel to avoid changing
    numerical GEMM shapes. Integer-rank runs score only actual candidates but
    select their coordinates from the same full-panel decomposition.

    The returned legacy key ``fg`` denotes h deletion on this full-H state.
    The formal driver writes its explicit structural name and separately
    evaluates the independently fitted E5 FG checkpoint and trajectory.
    """
    if probe.stage != "fgh":
        raise ValueError("E6 requires a frozen, unrestricted fgh probe")
    _synchronize(trajectory.device)
    prepare_started = time.perf_counter()
    trajectory._prepare()
    _synchronize(trajectory.device)
    prepare_seconds = time.perf_counter()-prepare_started
    device = trajectory.device
    ids = torch.as_tensor([panel], device=device, dtype=torch.long)
    positions = torch.as_tensor(actual_positions, device=device, dtype=torch.long)
    real_ids = ids[:, positions]
    h_tensors = {}
    inherited = trajectory.source.seq_len
    if interaction_execution not in ("dense", "projected"):
        raise ValueError("unknown h execution")
    projected = None
    project_started = time.perf_counter()
    if interaction_execution == "projected" and inherited:
        if isinstance(trajectory, ProjectedFastProbeTrajectory):
            trajectory._ensure_padded()
            projected = tuple(rows[:, :, :inherited] for rows in trajectory.padded)
        else:
            projected = project_row_features(probe, trajectory.features)
    _synchronize(device)
    project_seconds = time.perf_counter()-project_started

    def capture(layer, qc, rows, query_start, row_start):
        value = (probe.interaction_correction(layer, qc, rows) if projected is None else
            projected_interaction_correction(probe, layer, qc, projected[layer][:, :, row_start:row_start+rows.shape[2]]))
        if layer not in h_tensors:
            h_tensors[layer] = value.new_empty((probe.cfg.num_heads, len(panel), inherited,
                                                2 * probe.cfg.head_dim))
        h_tensors[layer][:, query_start:query_start + qc.shape[2], row_start:row_start + rows.shape[2]] = value[0]
        return value

    _synchronize(device)
    started = time.perf_counter()
    full, full_trace = probe.score(model, trajectory.source, trajectory.source_ids, ids, query_delta,
        prepared_cache=trajectory.cache, read_features=trajectory.features, trace=True,
        interaction_override=capture)
    _synchronize(device)
    timing = dict(trace_seconds=time.perf_counter() - started, prepare_seconds=prepare_seconds,
                  projected_row_prepare_or_view_seconds=project_seconds)
    diagnostic_started = time.perf_counter()
    head_diagnostic = diagnostic_h_callback(h_tensors[min(h_tensors)]) if diagnostic_h_callback and h_tensors else None
    _synchronize(device)
    timing["head_diagnostic_seconds"] = time.perf_counter()-diagnostic_started if head_diagnostic else 0.

    def fixed_full(layer, qc, rows, query_start, row_start):
        return h_tensors[layer][:, query_start:query_start + qc.shape[2],
                                 row_start:row_start + rows.shape[2]][None]

    control_started = time.perf_counter()
    check, check_trace = probe.score(model, trajectory.source, trajectory.source_ids, ids, query_delta,
        prepared_cache=trajectory.cache, read_features=trajectory.features, trace=True,
        interaction_override=fixed_full)
    logit_error = float((check - full).abs().max())
    read_error = max(float((a - b).abs().max()) for a, b in zip(
        check_trace.history_heads, full_trace.history_heads, strict=True))
    torch.testing.assert_close(check, full, atol=1e-5, rtol=1e-6)
    for actual, wanted in zip(check_trace.history_heads, full_trace.history_heads, strict=True):
        torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-6)

    def fixed_real_full(layer, qc, rows, query_start, row_start):
        indices = positions[query_start:query_start+qc.shape[2]]
        return h_tensors[layer][:, indices, row_start:row_start+rows.shape[2]][None]

    real_full, _ = probe.score(model, trajectory.source, trajectory.source_ids, real_ids, query_delta,
        prepared_cache=trajectory.cache, read_features=trajectory.features, interaction_override=fixed_real_full)
    real_shape_error = float((real_full[0]-full[0, positions]).abs().max())
    torch.testing.assert_close(real_full[0], full[0, positions], atol=1e-5, rtol=1e-6)

    def zero_h(layer, qc, rows, query_start, row_start):
        del layer, query_start, row_start
        return qc.new_zeros(*qc.shape[:3], rows.shape[2], 2 * probe.cfg.head_dim)

    fg, _ = probe.score(model, trajectory.source, trajectory.source_ids, real_ids, query_delta,
        prepared_cache=trajectory.cache, read_features=trajectory.features, interaction_override=zero_h)
    _synchronize(device)
    timing["full_and_control_replay_seconds"] = time.perf_counter()-control_started
    results = dict(fg=fg[0].detach(), full=real_full[0].detach())
    matrix_shapes, retained_energies, svd_checks = Counter(), defaultdict(list), []
    verified_variants = Counter()
    _synchronize(device)
    replay_seconds = decomposition_seconds = 0.
    for variant in variants:
        options = VARIANTS[variant]
        decompositions = {}
        t0 = time.perf_counter()
        for layer, h in h_tensors.items():
            # Compact one head immediately: exact SVD's [:,:r] is a view of
            # the full U, otherwise 100 tiny-rank factors retain ~6 GB of U.
            if decomposition_batch_heads < 1:
                raise ValueError("decomposition head batch must be positive")
            heads = []
            for start in range(0, len(h), decomposition_batch_heads):
                if decomposition_batch_heads == 1:
                    heads.append(compact_factors(decompose_interaction(h[start], **options, method=method,
                        max_rank=max(ranks), oversample=oversample, power_iterations=power_iterations, seed=seed+layer)))
                else:
                    heads.extend(decompose_interaction_batch(h[start:start+decomposition_batch_heads],
                        **options, method=method, max_rank=max(ranks), oversample=oversample,
                        power_iterations=power_iterations, seed=seed+layer))
            decompositions[layer] = BatchedReconstruction(h, heads)
            for head, obj in enumerate(heads):
                matrix_shapes[variant, *obj.metadata["matrix_shape"]] += 1
                retained_energies[variant].append(obj.metadata["normalized_retained_energy_at_max_rank"])
                if method == "randomized" and verified_variants[variant] < verify_svd_heads:
                    exact = decompose_interaction(h[head], **options, method="exact", max_rank=max(ranks))
                    energies = []
                    for rank in (2, 6, max(ranks)):
                        count = min(rank, exact.max_rank)
                        wanted = float(exact.singular_values[:count].square().sum())
                        got = float(obj.singular_values[:count].square().sum())
                        ratio = got / wanted if wanted > 1e-20 else 1.
                        energies.append(dict(rank=rank, captured_energy_ratio_to_exact=ratio,
                                             passes_98_percent=ratio >= .98))
                        if enforce_svd_accuracy and wanted > 1e-20 and ratio < .98:
                            raise RuntimeError("randomized SVD captures less than 98% of reference top-r energy; increase numerical accuracy before measurement")
                    svd_checks.append(dict(layer=layer, head=head, variant=variant, energies=energies))
                    verified_variants[variant] += 1
                    del exact
        _synchronize(device)
        decomposition_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        if batch_rank_scores:
            def replacement(layer, qc, rows, query_start, row_start):
                candidate_indices = positions[query_start:query_start+qc.shape[2]]
                chunk = slice(row_start, row_start+rows.shape[2])
                return reconstruct_rank_batch(decompositions[layer], ranks,
                    candidates=candidate_indices, rows=chunk)
            value, _ = probe.score(model, trajectory.source, trajectory.source_ids,
                real_ids.expand(len(ranks), -1), query_delta.expand(len(ranks), *query_delta.shape[1:]),
                prepared_cache=trajectory.cache, read_features=trajectory.features, interaction_override=replacement)
            results.update({f"{variant}_r{rank}": value[index].detach() for index, rank in enumerate(ranks)})
        else:
            for rank in ranks:
                def replacement(layer, qc, rows, query_start, row_start):
                    candidate_indices = positions[query_start:query_start + qc.shape[2]]
                    chunk = slice(row_start, row_start + rows.shape[2])
                    obj = decompositions[layer]
                    return obj.reconstruct(min(rank, obj.max_rank), candidates=candidate_indices, rows=chunk)[None]
                value, _ = probe.score(model, trajectory.source, trajectory.source_ids, real_ids, query_delta,
                    prepared_cache=trajectory.cache, read_features=trajectory.features, interaction_override=replacement)
                results[f"{variant}_r{rank}"] = value[0].detach()
        _synchronize(device)
        replay_seconds += time.perf_counter() - t0
        del decompositions
    timing.update(decomposition_seconds=decomposition_seconds, rank_replay_seconds=replay_seconds)
    return results, dict(full_rank_logit_max_error=logit_error, full_rank_read_max_error=read_error,
                        full_rank_real_candidate_shape_logit_max_error=real_shape_error,
                        inherited_rows=inherited, current_history_rows=trajectory.cache.seq_len,
                        native_appended_rows=trajectory.cache.seq_len-inherited,
                        interaction_columns="inherited source rows only; appended suffix has exactly zero h",
                        interaction_execution=interaction_execution,
                        rank_score_execution="batched across ranks" if batch_rank_scores else "serial ranks",
                        panel_size=len(panel), real_candidates=len(actual_positions),
                        matrix_shapes=[dict(variant=v, rows=m, columns=n, count=count)
                            for (v, m, n), count in sorted(matrix_shapes.items())],
                        maximum_rank_energy=[dict(variant=v, count=len(values),
                            minimum=min(values), median=float(np.median(values)), maximum=max(values))
                            for v, values in retained_energies.items()],
                        svd_checks=svd_checks, timing=timing, head_diagnostic=head_diagnostic)


def bootstrap_retention(groups, uids, *, replicates=1000, seed=17021,
                        baseline_method=TRAINED_FG_METHOD,
                        structural_baseline_method=H_DISABLED_DIAGNOSTIC):
    """Paired UID intervals with both scientifically distinct FG controls.

    AUC/bootstrap draws are computed once. The primary denominator is the
    independently trained E5 FG trajectory. The optional structural ratio
    instead deletes h on the full-H state, retaining its final f/g weights.
    Row order may differ after deterministic candidate-panel partitioning;
    exact feedback keys, multiplicity and labels must still coincide.
    """
    def ordered(rows):
        return sorted(rows, key=lambda row: (row["uid"], row["query_timestamp"], row["raw_item_id"],
            row.get("request_instance_ordinal", 0), row["label"]))
    def feedback_key(row):
        return (row["uid"], row["query_timestamp"], row["raw_item_id"],
                row.get("request_instance_ordinal", 0), row["label"])
    base = ordered(groups[baseline_method])
    keys = [feedback_key(row) for row in base]
    if len(set(keys)) != len(keys):
        raise ValueError("E6 scores contain duplicate feedback rows")
    population = np.asarray(sorted(uids), dtype=np.int64)
    uid_values = np.asarray([r["uid"] for r in base], dtype=np.int64)
    uid_index = np.searchsorted(population, uid_values)
    if np.any(uid_index >= len(population)) or not np.array_equal(population[uid_index], uid_values):
        raise ValueError("score UID outside the frozen E6 population")
    labels = np.asarray([r["label"] for r in base], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.asarray([np.bincount(generator.integers(len(population), size=len(population)),
        minlength=len(population)) for _ in range(replicates)], dtype=np.uint16)
    ones = np.ones((1, len(population)), dtype=np.uint16)
    auc, intervals = {}, {}
    for method, unordered_rows in groups.items():
        rows = ordered(unordered_rows)
        if keys != [feedback_key(row) for row in rows]:
            raise ValueError("E6 methods do not share exactly the same feedback rows")
        scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
        auc[method] = float(weighted_auc(scores, labels, uid_index, ones)[0])
        intervals[method] = weighted_auc(scores, labels, uid_index, draws)

    def ci(value):
        finite = value[np.isfinite(value)]
        return np.quantile(finite, [.025, .975]).tolist() if finite.size else None

    def relative_to(baseline):
        denominator = auc["full"]-auc[baseline]
        draw_denominator = intervals["full"]-intervals[baseline]
        valid = np.isfinite(draw_denominator) & (draw_denominator > 0)
        points = []
        for method in groups:
            gain = auc[method]-auc[baseline]
            sampled_gain = intervals[method]-intervals[baseline]
            points.append(dict(method=method, baseline_method=baseline,
                auc=auc[method], auc_ci95=ci(intervals[method]),
                auc_gain_over_baseline=gain, auc_gain_over_baseline_ci95=ci(sampled_gain),
                interaction_gain_retention=gain/denominator if denominator != 0 else None,
                interaction_gain_retention_ci95=ci(sampled_gain/draw_denominator) if valid.all() else None))
        confidence = dict(replicates=replicates, seed=seed, users=len(population),
            baseline_method=baseline, full_minus_baseline_auc=denominator,
            full_minus_baseline_auc_ci95=ci(draw_denominator),
            nonpositive_or_invalid_interaction_gain_draws=int((~valid).sum()),
            scope="paired UID clusters; fixed fitted weights; ranks share full-H state, trained FG owns its own state; not training-seed variability",
            retention=f"(AUC(h_r)-AUC({baseline}))/(AUC(full_h)-AUC({baseline})); common recovery normalization cancels")
        return points, confidence

    points, confidence = relative_to(baseline_method)
    if structural_baseline_method is not None:
        structural_points, structural_confidence = relative_to(structural_baseline_method)
        confidence["same_state_h_disabled_diagnostic"] = dict(points=structural_points, confidence=structural_confidence)
    return points, confidence


def _write_block(output, offset, rows, report, identity):
    stem = f"part_{offset:06d}"
    temporary = output / f"{stem}.parquet.writing"
    final = output / f"{stem}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(final)
    write_json(output / f"{stem}.json", dict(identity=identity, parquet=final.name,
        sha256=sha256(final), report=report))


@torch.no_grad()
def evaluate_e6_user(parent, current, probe, fg_probe, events, requests_by_timestamp, uid, cfg, *,
                     known_items, candidate_size=230, execution="projected", fg_pool=None,
                     full_h_pool=None, verified=None, verification_heads=2, compression_options=None):
    """Actual formal per-user path, exposed for bounded CPU numeric oracles."""
    result, snapshots = [], []
    verified = Counter() if verified is None else verified
    compression_options = compression_options or dict(ranks=(0, 1, 2, 4, 6, 8, 16),
        variants=("both_norm",), method="randomized")
    paired = make_paired_trajectories(parent, current, fg_probe, probe, events,
        cfg, fg_pool=fg_pool, full_h_pool=full_h_pool, execution=execution)
    trajectory, fg_trajectory = paired.full_h, paired.fg
    device = trajectory.device
    panels, coordinates = release_candidate_panels(requests_by_timestamp, uid=uid,
        release_timestamp=int(cfg["release_day"])*86400, known_items=known_items, size=candidate_size)
    for timestamp, requests in sorted(requests_by_timestamp.items()):
        result.extend(score_fg_timestamp(fg_trajectory, requests, uid=uid, timestamp=timestamp,
            query_batch=int(cfg.get("query_batch", 16))))
        trajectory.advance(timestamp)
        gap = 0 if trajectory.native.last_timestamp is None else timestamp-trajectory.native.last_timestamp
        delta = torch.tensor([gap], device=device, dtype=torch.float32)
        groups = defaultdict(list)
        for row in requests:
            groups[coordinates[int(row["item_idx"])][0]].append(row)
        for panel_index, panel_rows in sorted(groups.items()):
            panel = panels[panel_index]
            positions = [coordinates[int(row["item_idx"])][1] for row in panel_rows]
            values, report = compress_snapshot(current, probe, trajectory, panel, positions, delta,
                **compression_options, interaction_execution=execution,
                verify_svd_heads=verification_heads if any(
                    verified[v] < verification_heads for v in compression_options["variants"]) else 0)
            for method, scores in values.items():
                formal_method = H_DISABLED_DIAGNOSTIC if method == "fg" else method
                for row, value in zip(panel_rows, scores.cpu().tolist(), strict=True):
                    result.append(dict(row, method=formal_method, score=float(value)))
            report.update(uid=uid, timestamp=int(timestamp), panel_index=panel_index,
                panels_per_release=len(panels), panel_sha256=hashlib.sha256(json.dumps(panel).encode()).hexdigest())
            snapshots.append(report)
            verified.update(check["variant"] for check in report["svd_checks"])
    # The frozen quality configuration already excludes history after
    # the final scored consumer. Omitting that suffix changes no query
    # snapshot and is not a full-E14 deployment-cost measurement.
    stop_after_last = bool(cfg.get("stop_after_last_query", False))
    if not stop_after_last:
        trajectory.advance(int(cfg["end_day"]) * 86400)
        fg_trajectory.advance(int(cfg["end_day"]) * 86400)
    trajectory_report = trajectory.report()
    fg_report = fg_trajectory.report()
    if stop_after_last:
        trajectory_report["scope"] = "all E14 scored consumers and every preceding causal write; no writes after the final scored consumer"
        fg_report["scope"] = trajectory_report["scope"]
    trajectory_report = dict(uid=uid, **trajectory_report, unscored_suffix_advanced=not stop_after_last,
        trained_fg_own_trajectory=fg_report,
        shared_old_prefill=dict(history_length=paired.initial.cache.seq_len,
            calls=1, state="read-only initial prefill; two independent f preparations and write paths"))
    return result, snapshots, trajectory_report


def source_paths():
    """Freeze score mathematics and the concrete fitting-identity validators."""
    paths = [Path(__file__), Path(__file__).with_name("analysis.py"),
        Path(__file__).with_name("probes.py"), Path(__file__).with_name("probe_replay.py"),
        Path(__file__).with_name("probe_variants.py"),
        Path(__file__).with_name("probe_projected_fast.py"), Path(__file__).with_name("probe_fast.py"),
        Path(__file__).with_name("probe_cost.py"),
        Path(__file__).with_name("energy_certificate.py"),
        Path(__file__).with_name("lowrank_batch.py"), Path(__file__).with_name("scenario_inputs.py"),
        Path(__file__).with_name("lowrank_rank_batch.py"),
        Path(__file__).with_name("fg_trajectory_control.py"),
        Path(__file__).with_name("common.py"), ROOT / "src/hstu_kvcache/adaptation/reader.py",
        ROOT / "src/hstu_kvcache/evaluation/trajectory.py",
        ROOT / "src/hstu_kvcache/evaluation/statistics.py"]
    paths.extend((ROOT / "src/hstu_kvcache/models").glob("*.py"))
    paths.extend(sampling_source_paths())
    return paths


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    diagnostic = args.diagnostic_users is not None
    binding, uids, role_source, population_provenance = scenario_inputs(args.input_root,
        args.edge, population_file=args.population_file, diagnostic=diagnostic)
    if diagnostic:
        if not 1 <= args.diagnostic_users <= len(uids):
            raise ValueError("diagnostic users must be a positive subset of the frozen E6 population")
        uids = uids[:args.diagnostic_users]
    elif args.sensitivity_users is not None:
        if not 512 <= args.sensitivity_users <= len(uids):
            raise ValueError("protocol sensitivity needs at least512 fixed feedback users")
        uids = uids[:args.sensitivity_users]
    elif len(uids) < 2000:
        raise ValueError("E6 requires at least 2,000 feedback users per edge")
    if diagnostic and args.sensitivity_users is not None:
        raise ValueError("diagnostic and quality-sensitivity roles must be separate")
    role = quality_role(uids, diagnostic=diagnostic)
    if args.sensitivity_users is not None:
        role = "additional_fixed_subset_protocol_sensitivity"
    scenario_qualification = scenario_qualification_record(population_provenance)
    if args.checkpoint_users < 1 or not uids:
        raise ValueError("nonempty user population and positive checkpoint size required")
    histories = restore_pack(population_provenance["history_path"])
    if not set(uids) <= set(histories.rows):
        raise ValueError("frozen scenario users are missing from the raw history pack")
    by_user = defaultdict(lambda: defaultdict(list))
    requests = pq.read_table(population_provenance["requests_path"], filters=[("uid", "in", uids)])
    seen = set()
    at, end = int(binding["settings"]["release_day"]) * 86400, int(binding["settings"]["end_day"]) * 86400
    for row in requests.to_pylist():
        key = row["uid"], row["query_timestamp"], row["raw_item_id"]
        if key in seen or not at <= int(row["query_timestamp"]) < end or row["label"] not in (0, 1):
            raise ValueError("E6 feedback keys, labels, or evaluation window are invalid")
        seen.add(key)
        by_user[int(row["uid"])][int(row["query_timestamp"])].append(row)
    if any(not by_user.get(uid) for uid in uids):
        raise ValueError("E6 requires every frozen user to have real feedback; no users are silently dropped")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    parent, current, models = load_models(args.edge, device)
    probe, fit = load_variant(args.probe, device)
    probe.eval().requires_grad_(False)
    if probe.stage != "fgh":
        raise ValueError("E6 checkpoint must be fgh")
    fit_roles = validate_fit_roles(fit, role_source, uids, args.edge, models,
        population_provenance["calibration_input_binding_sha256"])
    if not diagnostic and (fit.get("diagnostic_only") or fit.get("quality_use_permitted") is False):
        raise ValueError("a diagnostic checkpoint cannot be used for formal E6 quality")
    if population_provenance.get("anchor_models", models) != models:
        raise ValueError("scenario qualification used other backbone checkpoints")
    fg_probe, fg_fit, fg_identity = load_fg_control(args.fg_probe, probe, fit, device=device,
        source_population=role_source, evaluation_uids=uids, edge=args.edge, models=models,
        calibration_binding_sha256=population_provenance["calibration_input_binding_sha256"],
        expected_method_label=args.fg_method_label, allow_diagnostic=diagnostic)
    # Chunking changes execution only; all weights/statistics stay frozen.
    probe.cfg.query_chunk = args.candidate_chunk
    probe.cfg.row_chunk = args.row_chunk
    variants = args.variants.split(",")
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError("unknown E6 block/normalization variant")
    ranks = (0, 1, 2, 4, 6, 8, 16)
    sources = source_paths()
    identity = dict(protocol_version="e6_trained_fg_own_state_v2", edge=args.edge, role=role, users=uids,
        input_binding_sha256=population_provenance["input_binding_sha256"],
        population=population_provenance, fitting=fit_roles,
        scenario_qualification=scenario_qualification,
        checkpoint_sha256=sha256(args.probe), models=models,
        trained_fg_control=fg_identity,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        candidate_size=args.candidates, candidate_seed=20261022, ranks=list(ranks), variants=variants,
        candidate_protocol=f"fixed per-user/per-release hindsight panels; all true E14 targets in deterministic blocks of at most{args.candidates}, then uniform unique fill; no omitted users/events",
        decomposition=dict(method=args.method, oversample=args.oversample, power_iterations=args.power_iterations,
                           verification_heads_per_variant=args.verify_svd_heads,
                           batch_heads=args.decomposition_batch_heads),
        execution=dict(candidate_chunk=args.candidate_chunk, row_chunk=args.row_chunk, checkpoint_users=args.checkpoint_users,
            stop_after_last_query=bool(binding["settings"].get("stop_after_last_query", False)),
            interaction_and_writer=args.execution),
        batch_rank_scores=args.batch_rank_scores,
        snapshot_protocol="real E14 timestamps on full-h persistent trajectory; all rank interventions share its snapshot; no rank-specific appends",
        main_retention_baseline=TRAINED_FG_METHOD,
        additional_structural_baseline=H_DISABLED_DIAGNOSTIC,
        fg_control="separately fitted E5 FG checkpoint, its own f preparation/causal E14 writes/read scores; shared old prefill only",
        h_disabled_control="h deleted with final fgh f/g weights on full-H state; additional same-state structural diagnostic")
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "configuration.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("E6 attempt inputs, code, weights, or settings changed")
    write_json(path, identity)
    blocks = output / "blocks"
    blocks.mkdir(exist_ok=True)
    all_rows, all_reports = [], []
    verified = Counter()
    pool = ProjectedGraphPool(current, probe, window=binding["settings"]["history_length"]) if args.execution == "projected" else None
    fg_pool = ProjectedGraphPool(current, fg_probe, window=binding["settings"]["history_length"]) if args.execution == "projected" else None
    started = time.perf_counter()
    for offset in range(0, len(uids), args.checkpoint_users):
        block_uids = uids[offset:offset + args.checkpoint_users]
        marker_path = blocks / f"part_{offset:06d}.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker["identity"] != identity_hash or marker["report"]["uids"] != block_uids:
                raise ValueError("E6 committed block inputs changed")
            if sha256(blocks / marker["parquet"]) != marker["sha256"]:
                raise ValueError("E6 committed block scores changed")
            all_rows.extend(pq.read_table(blocks / marker["parquet"]).to_pylist())
            all_reports.append(marker["report"])
            for snapshot in marker["report"]["snapshots"]:
                verified.update(check["variant"] for check in snapshot["svd_checks"])
            continue
        result, snapshots, trajectories = [], [], []
        for uid in block_uids:
            user_rows, user_snapshots, user_report = evaluate_e6_user(parent, current, probe, fg_probe,
                histories.rows[uid], by_user[uid], uid, binding["settings"],
                known_items=current.cfg.num_items-256, candidate_size=args.candidates,
                execution=args.execution, fg_pool=fg_pool, full_h_pool=pool, verified=verified,
                verification_heads=args.verify_svd_heads,
                compression_options=dict(ranks=ranks, variants=variants, method=args.method,
                    oversample=args.oversample, power_iterations=args.power_iterations, seed=17,
                    decomposition_batch_heads=args.decomposition_batch_heads,
                    batch_rank_scores=args.batch_rank_scores))
            result.extend(user_rows)
            snapshots.extend(user_snapshots)
            trajectories.append(user_report)
            print(json.dumps(dict(status="e6_user_complete", uid=uid, users=offset + len(trajectories),
                                  total=len(uids), snapshots=len(snapshots), seconds=time.perf_counter() - started)), flush=True)
        report = dict(uids=block_uids, snapshots=snapshots, trajectories=trajectories)
        _write_block(blocks, offset, result, report, identity_hash)
        all_rows.extend(result)
        all_reports.append(report)
        write_json(output / "progress.json", dict(status="e6_block_committed", users=offset + len(block_uids),
            total=len(uids), seconds=time.perf_counter() - started))
    result = dict(status="complete", role=role, users=len(uids), identity_hash=identity_hash,
        execution_seconds=time.perf_counter() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        snapshots=sum(len(report["snapshots"]) for report in all_reports),
        paper_result_eligible=not diagnostic and len(uids) >= 2000,
        additional_protocol_result_eligible=not diagnostic and len(uids) >= 512,
        e6_main_result_eligible=not diagnostic and args.sensitivity_users is None and len(uids) >= 2000,
        graph_pool=pool.report() if pool is not None else None,
        trained_fg_graph_pool=fg_pool.report() if fg_pool is not None else None,
        scenario_qualification=scenario_qualification,
        main_retention_baseline=TRAINED_FG_METHOD,
        additional_structural_baseline=H_DISABLED_DIAGNOSTIC,
        scope="hindsight read-compression on fixed full-h E14 snapshots; not a deployment encoder or independent rank trajectory")
    if diagnostic:
        write_json(output / "diagnostic.json", result)
    else:
        groups = defaultdict(list)
        for row in all_rows:
            groups[row["method"]].append(row)
        points, confidence = bootstrap_retention(groups, uids)
        structural = confidence.pop("same_state_h_disabled_diagnostic")
        result.update(points=points, confidence=confidence,
            same_state_h_disabled_diagnostic=structural,
            r0_retention_over_trained_fg_above_20_percent=[point["method"] for point in points
                if point["method"].endswith("_r0") and
                point["interaction_gain_retention"] is not None and point["interaction_gain_retention"] > .2],
            r0_retention_over_same_state_h_disabled_above_20_percent=[point["method"] for point in structural["points"]
                if point["method"].endswith("_r0") and point["interaction_gain_retention"] is not None
                and point["interaction_gain_retention"] > .2])
        fields = ["edge", "method", "baseline_method", "auc", "auc_ci95_low", "auc_ci95_high", "auc_gain_over_baseline",
                  "retention", "retention_ci95_low", "retention_ci95_high", "users"]
        for filename, selected_points in (("points.csv", points), ("same_state_structure_points.csv", structural["points"])):
            with (output / filename).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                for point in selected_points:
                    writer.writerow([args.edge, point["method"], point["baseline_method"], point["auc"], *(point["auc_ci95"] or [None, None]),
                        point["auc_gain_over_baseline"], point["interaction_gain_retention"],
                        *(point["interaction_gain_retention_ci95"] or [None, None]), len(uids)])
        write_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", users=len(uids), role=role, output=str(output))), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--population-file", type=Path,
                        help="frozen feedback-specific scenario_manifest.json; required for formal E6")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--fg-probe", type=Path, required=True,
                        help="matching frozen E5 trained FG checkpoint; owns an independent E14 trajectory")
    parser.add_argument("--fg-method-label", choices=("fg", "fg_extended"), default="fg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--diagnostic-users", type=int)
    parser.add_argument("--sensitivity-users", type=int)
    parser.add_argument("--candidates", type=int, default=230)
    parser.add_argument("--candidate-chunk", type=int, default=230)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--checkpoint-users", type=int, default=4)
    parser.add_argument("--variants", default="both_norm,both_raw,key_norm,value_norm")
    parser.add_argument("--method", choices=("exact", "randomized"), default="randomized")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--verify-svd-heads", type=int, default=2)
    parser.add_argument("--decomposition-batch-heads", type=int, default=10)
    parser.add_argument("--batch-rank-scores", action="store_true",
                        help="equivalent rank-as-batch complete read; requires bounded GPU validation before formal use")
    parser.add_argument("--execution", choices=("dense", "projected"), default="projected",
                        help="equivalent h execution; projected caches first-layer row map and uses a shared append graph")
    run(parser.parse_args())

#!/usr/bin/env python3
"""Numerical/memory readiness for E6; never computes paper AUC or distribution results.

CPU mode compares exact and randomized rank16 on the SAME synthetic matrix,
and evaluates a small real-model scoring path. GPU mode uses eight real
1,024-row prefixes and 230 candidates, but can initialize a clearly tagged
synthetic h when no fitted h exists. Such runs establish shape/correctness/
performance readiness only, not interaction-quality or low-rank evidence.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch

from experiments.ch3_motivation.code.analysis import decompose_interaction
from experiments.ch3_motivation.code.common import load_models, write_json
from experiments.ch3_motivation.code.lowrank_batch import decompose_interaction_batch
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, ProjectedGraphPool,
)
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant
from experiments.ch3_motivation.code.run_lowrank import compact_factors, compress_snapshot
from hstu_kvcache.evaluation.history import restore_pack
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256


def numerical_sources():
    paths = [Path(__file__), Path(__file__).with_name("run_lowrank.py"),
        Path(__file__).with_name("analysis.py"), Path(__file__).with_name("lowrank_batch.py"),
        Path(__file__).with_name("common.py"), Path(__file__).with_name("probe_replay.py"),
        Path(__file__).with_name("probes.py"), Path(__file__).with_name("probe_variants.py"),
        Path(__file__).with_name("probe_projected_fast.py"), Path(__file__).with_name("probe_fast.py"),
        Path(__file__).with_name("probe_cost.py"),
        Path(__file__).with_name("scenario_inputs.py"), ROOT/"src/hstu_kvcache/adaptation/reader.py"]
    paths.extend((ROOT/"src/hstu_kvcache/models").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def matrix_comparison(h, *, oversample=8, power_iterations=1):
    """Reference/approximation errors use one h; no data/seed mixing."""
    started = time.perf_counter()
    exact = compact_factors(decompose_interaction(h, normalize_kv=True, method="exact", max_rank=16))
    exact_seconds = time.perf_counter() - started
    started = time.perf_counter()
    approximate = compact_factors(decompose_interaction(h, normalize_kv=True, method="randomized",
        max_rank=16, oversample=oversample, power_iterations=power_iterations, seed=17))
    randomized_seconds = time.perf_counter() - started
    main = exact.reconstruct(0)
    centered_energy = float((h - main).double().square().sum())
    points = []
    for rank in (2, 6, 16):
        count = min(rank, exact.max_rank)
        a, b = exact.reconstruct(count), approximate.reconstruct(count)
        exact_error = float((a - h).double().square().sum())
        approximate_error = float((b - h).double().square().sum())
        true_top_energy = float(exact.singular_values[:count].square().sum())
        got_top_energy = float(approximate.singular_values[:count].square().sum())
        points.append(dict(rank=rank,
            exact_squared_error_over_interaction=exact_error / centered_energy if centered_energy else 0.,
            randomized_squared_error_over_interaction=approximate_error / centered_energy if centered_energy else 0.,
            randomized_vs_exact_reconstruction_squared_error_over_interaction=float(
                (a-b).double().square().sum()) / centered_energy if centered_energy else 0.,
            captured_normalized_energy_ratio_to_exact=got_top_energy / true_top_energy if true_top_energy else 1.))
    return dict(shape=list(h.shape), h_sha256=hashlib.sha256(h.numpy().tobytes()).hexdigest(),
        exact_seconds=exact_seconds, randomized_seconds=randomized_seconds,
        oversample=oversample, power_iterations=power_iterations, points=points,
        original_h_bytes=h.numel()*h.element_size(),
        exact_compact_factor_bytes=sum(x.numel()*x.element_size() for x in (
            exact.left, exact.singular_values, exact.right)),
        same_matrix_for_both=True)


def batch_comparison(h, *, threads):
    """Time serial/batched calls on the same independent full-size heads."""
    reports, reference = [], {}
    def sync():
        if h.device.type == "cuda":
            torch.cuda.synchronize(h.device)
    for method in ("exact", "randomized"):
        sync()
        started = time.perf_counter()
        serial = [compact_factors(decompose_interaction(head, normalize_kv=True, method=method,
            max_rank=16, oversample=8, power_iterations=1, seed=17)) for head in h]
        sync()
        serial_seconds = time.perf_counter()-started
        started = time.perf_counter()
        batch = decompose_interaction_batch(h, normalize_kv=True, method=method,
            max_rank=16, oversample=8, power_iterations=1, seed=17)
        sync()
        batch_seconds = time.perf_counter()-started
        points = []
        for rank in (0, 2, 6, 16):
            error, denominator = 0., 0.
            for scalar, grouped in zip(serial, batch, strict=True):
                wanted, got = scalar.reconstruct(rank), grouped.reconstruct(rank)
                error += float((wanted-got).double().square().sum())
                denominator += float(wanted.double().square().sum())
            points.append(dict(rank=rank, serial_vs_batch_squared_relative_error=error/denominator))
        reports.append(dict(method=method, shape=list(h.shape), serial_seconds=serial_seconds,
            batched_seconds=batch_seconds, serial_over_batch_speedup=serial_seconds/batch_seconds, points=points))
        reference[method] = [[float(obj.singular_values[:rank].square().sum()) for rank in (2, 6, 16)] for obj in batch]
        del serial, batch
    ratios = np.asarray(reference["randomized"])/np.asarray(reference["exact"])
    return dict(role="same independent matrix/head; batching is numerical execution only", threads=threads,
        device=str(h.device),
        reports=reports, randomized_energy_ratio_to_exact_min_by_rank=ratios.min(0).tolist())


@torch.no_grad()
def cpu_score_comparison():
    """Synthetic score-delta retention is a numerical check, never AUC."""
    os.environ["EVOKV_ATTENTION_BACKEND"] = "torch"
    torch.manual_seed(17)
    cfg = HSTUConfig(num_items=512, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=64, input_dropout=0., attn_dropout=0.)
    parent = HSTU(cfg).eval().requires_grad_(False)
    current = copy.deepcopy(parent)
    current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.01)
    probe = build_variant("head", current, stage="fgh", hidden_width=16, depth=1,
        query_chunk=230, row_chunk=64).eval().requires_grad_(False)
    for parameter in probe.parameters():
        parameter.normal_(std=.08)
    times = np.r_[np.arange(64)*10+100, 86401]
    events = (times, np.arange(1, 66), np.zeros(65, dtype=np.int64))
    trajectory = ProbeTrajectory(parent, current, probe, events,
        dict(edge=1, release_day=1, end_day=2, history_length=64, append_band_size=16))
    trajectory.advance(86402)
    panel, positions, delta = list(range(100, 330)), list(range(16)), torch.tensor([1.])
    exact, exact_report = compress_snapshot(current, probe, trajectory, panel, positions, delta,
        variants=("both_norm",), method="exact")
    approximate, approximate_report = compress_snapshot(current, probe, trajectory, panel, positions, delta,
        variants=("both_norm",), method="randomized", oversample=8, power_iterations=1)
    denominator = float((exact["full"]-exact["fg"]).double().square().sum())
    comparisons = []
    for rank in (0, 1, 2, 4, 6, 8, 16):
        name = f"both_norm_r{rank}"
        exact_error = float((exact[name]-exact["full"]).double().square().sum())
        approximate_error = float((approximate[name]-approximate["full"]).double().square().sum())
        comparisons.append(dict(rank=rank,
            exact_score_delta_retention=1-exact_error/denominator if denominator else None,
            randomized_score_delta_retention=1-approximate_error/denominator if denominator else None,
            randomized_vs_exact_logit_max_error=float((approximate[name]-exact[name]).abs().max())))
    return dict(role="synthetic_numerical_diagnostic", users=1, candidates=230,
        inherited_rows=trajectory.source.seq_len, points=comparisons,
        score_delta_definition="1-||score_r-score_full||²/||score_h_disabled-score_full||²; numerical response, not AUC gain retention",
        exact_full_replay=exact_report["full_rank_logit_max_error"],
        randomized_full_replay=approximate_report["full_rank_logit_max_error"])


def cpu_check(args):
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(17)
    c, n, d = (230, 1024, 64) if args.full_size else (40, 96, 16)
    low = torch.randn(c, d, 24, generator=generator) @ torch.randn(24, n, generator=generator)
    structured = low.transpose(1, 2) + .01*torch.randn(c, n, d, generator=generator)
    unstructured = torch.randn(c, n, d, generator=generator)
    results = dict(structured=matrix_comparison(structured),
                   unstructured=matrix_comparison(unstructured), scoring=cpu_score_comparison())
    if args.batch_heads:
        hs = []
        for _ in range(args.batch_heads):
            value = torch.randn(c, d, 24, generator=generator) @ torch.randn(24, n, generator=generator)
            hs.append(value.transpose(1, 2)+.01*torch.randn(c, n, d, generator=generator))
        results["batched_heads"] = batch_comparison(torch.stack(hs), threads=args.threads)
    return dict(status="complete", role="CPU numerical/performance diagnostic; no paper quality evidence",
        full_size=args.full_size, torch_threads=args.threads, results=results,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation="The full-rank random matrix intentionally tests a difficult spectrum; an approximate SVD is not assumed exact.")


def memory_estimate(probe, *, candidates=230, rows=1024):
    cfg = probe.cfg
    element = probe.row_mean.element_size()
    pairs = cfg.num_heads*candidates*rows
    return dict(candidates=candidates, history_rows=rows, head_kv_dimensions=2*cfg.head_dim,
        h_per_head_bytes=candidates*rows*2*cfg.head_dim*element,
        all_layers_full_h_bytes=cfg.num_layers*pairs*2*cfg.head_dim*element,
        pair_input_per_layer_bytes=pairs*(cfg.head_dim+cfg.row_dim)*element,
        one_hidden_activation_per_layer_bytes=pairs*cfg.hidden_width*element,
        accounting="explicit major tensors only; actual peak including allocator/workspace measured below")


@torch.no_grad()
def gpu_check(args):
    if not args.probe or not 1 <= args.users <= 16:
        raise ValueError("GPU readiness needs a checkpoint and 1..16 diagnostic users")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    directory = args.input_root / f"edge_{args.edge}"
    binding = json.loads((directory / "inputs.json").read_text())
    population_path = args.input_root / "population.json"
    if sha256(population_path) != binding["population_sha256"]:
        raise ValueError("frozen diagnostic population changed")
    histories_path = directory / binding["history"]["path"]
    if sha256(histories_path) != binding["history"]["sha256"]:
        raise ValueError("frozen diagnostic raw inputs changed")
    population = json.loads(population_path.read_text())
    histories = restore_pack(histories_path)
    at = int(binding["settings"]["release_day"]) * 86400
    users = [uid for uid in population["development"] if int(np.searchsorted(
        histories.rows[uid][0], at, side="left")) >= 1024][:args.users]
    if len(users) != args.users:
        raise ValueError("not enough pre-release 1024-row diagnostic histories")
    parent, current, models = load_models(args.edge, device)
    probe, metadata = load_variant(args.probe, device)
    if metadata["identity"]["models"] != models or metadata["identity"]["edge"] != args.edge:
        raise ValueError("readiness probe/model edge identities differ")
    if probe.stage != "fgh":
        if not args.synthetic_interaction:
            raise ValueError("no fitted h: use an explicit --synthetic-interaction for readiness only")
        if probe.stage == "f":
            probe = probe.expanded("fg")
        probe = probe.expanded("fgh")
    if args.synthetic_interaction:
        torch.manual_seed(17)
        for networks in (probe.g, probe.h):
            for network in networks:
                network.weights[-1].normal_(std=.001)
    probe.eval().requires_grad_(False)
    probe.cfg.query_chunk, probe.cfg.row_chunk = args.candidate_chunk, args.row_chunk
    variants = tuple(args.variants.split(","))
    checkpoints = dict(models=models, probe=str(args.probe.resolve()), probe_sha256=sha256(args.probe),
        probe_variant=metadata.get("variant", metadata["identity"].get("variant", "head")),
        diagnostic_checkpoint_only=metadata.get("diagnostic_only", False),
        fit_metadata_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        input_binding_sha256=sha256(directory/"inputs.json"), history_sha256=sha256(histories_path),
        population_sha256=sha256(population_path), candidates=list(range(1, 231)),
        snapshot_scope="real publishing-time1024 prefixes; 230 known-item candidates; readiness only, not E14 quality")
    results = []
    pool = ProjectedGraphPool(current, probe, window=binding["settings"]["history_length"]) if args.execution == "projected" else None
    for index, uid in enumerate(users):
        trajectory = (ProjectedFastProbeTrajectory(parent, current, probe, histories.rows[uid], binding["settings"], pool=pool)
            if pool is not None else ProbeTrajectory(parent, current, probe, histories.rows[uid], binding["settings"]))
        panel = list(range(1, 231))
        gap = at-trajectory.native.last_timestamp
        delta = torch.tensor([gap], device=device, dtype=torch.float32)
        torch.cuda.synchronize(device)
        before = time.perf_counter()
        values, report = compress_snapshot(current, probe, trajectory, panel, list(range(16)), delta,
            variants=variants, method="randomized", oversample=args.oversample,
            power_iterations=args.power_iterations,
            verify_svd_heads=args.verify_svd_heads if index == 0 else 0,
            decomposition_batch_heads=args.decomposition_batch_heads,
            enforce_svd_accuracy=False,
            interaction_execution=args.execution,
            diagnostic_h_callback=(lambda h: batch_comparison(h, threads=torch.get_num_threads()))
                if args.compare_decomposition and index == 0 else None)
        torch.cuda.synchronize(device)
        report.update(uid=uid, seconds=time.perf_counter()-before,
            allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if args.compare_full_score and index == 0:
            exact_started = time.perf_counter()
            exact, exact_report = compress_snapshot(current, probe, trajectory, panel, list(range(16)), delta,
                variants=variants, method="exact", decomposition_batch_heads=args.decomposition_batch_heads,
                interaction_execution=args.execution)
            torch.cuda.synchronize(device)
            denominator = float((exact["full"]-exact["fg"]).double().square().sum())
            comparisons = []
            for method, predicted in values.items():
                exact_error = float((exact[method]-exact["full"]).double().square().sum())
                approximate_error = float((predicted-values["full"]).double().square().sum())
                comparisons.append(dict(method=method,
                    exact_score_delta_response=1-exact_error/denominator if denominator else None,
                    randomized_score_delta_response=1-approximate_error/denominator if denominator else None,
                    exact_vs_randomized_logit_max_error=float((predicted-exact[method]).abs().max())))
            report["same_snapshot_exact_full_score_oracle"] = dict(seconds=time.perf_counter()-exact_started,
                points=comparisons, exact_report=exact_report,
                scope="fixed same snapshot/panel; numerical score response only, no AUC or quality evidence",
                response_definition="1-||logit_r-logit_full||²/||logit_h_disabled-logit_full||²")
        results.append(report)
        if args.advance_events:
            start_cursor = trajectory.native.cursor
            stop = min(start_cursor+args.advance_events, len(trajectory.native.events[0]))
            until = (int(trajectory.native.events[0][stop]) if stop < len(trajectory.native.events[0])
                     else int(trajectory.native.events[0][-1])+1)
            torch.cuda.synchronize(device)
            append_started = time.perf_counter()
            trajectory.advance(until)
            torch.cuda.synchronize(device)
            append_count = trajectory.native.cursor-start_cursor
            append_seconds = time.perf_counter()-append_started
            report["causal_append_diagnostic"] = dict(events=append_count, seconds=append_seconds,
                seconds_per_event=append_seconds/append_count if append_count else None,
                inherited_rows_remaining=trajectory.source.seq_len, report=trajectory.report(),
                scope="bounded real next-event replay; no quality scores after the publishing snapshot")
        write_json(args.output / "progress.json", dict(role="numerical_performance_diagnostic",
            users=index+1, total=len(users), last=report))
        print(dict(status="e6_readiness_user", uid=uid, users=index+1,
                   seconds=report["seconds"], reserved=report["reserved_bytes"]), flush=True)
    return dict(status="complete", role=f"{len(users)}-user GPU numerical/performance readiness; no AUC computed",
        users=users, synthetic_interaction=args.synthetic_interaction,
        physical_gpu=args.gpu, identity=checkpoints, memory_estimate=memory_estimate(probe), snapshots=results,
        execution=args.execution, graph_pool=pool.report() if pool is not None else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        seconds_total=sum(report["seconds"] for report in results),
        causal_append_seconds_total=sum(report.get("causal_append_diagnostic", {}).get("seconds", 0.) for report in results),
        causal_append_events_total=sum(report.get("causal_append_diagnostic", {}).get("events", 0) for report in results),
        approximate_svd_passes_all_checked_energy_thresholds=all(point["passes_98_percent"]
            for report in results for check in report["svd_checks"] for point in check["energies"]),
        runtime_role="shape/memory/code speed only; a fitted h and qualified >=2000 feedback users are still required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--gpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-heads", type=int, default=0,
                        help="CPU same-matrix scalar/batch oracle, e.g.10 actual-size heads")
    parser.add_argument("--edge", type=int, default=1)
    parser.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--synthetic-interaction", action="store_true")
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--advance-events", type=int, default=128,
                        help="bounded causal future writes/user to measure full-E14 trajectory overhead")
    parser.add_argument("--candidate-chunk", type=int, default=230)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--variants", default="both_norm")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--verify-svd-heads", type=int, default=2)
    parser.add_argument("--decomposition-batch-heads", type=int, default=10)
    parser.add_argument("--execution", choices=("dense", "projected"), default="projected")
    parser.add_argument("--compare-decomposition", action="store_true",
                        help="GPU first real-h layer: same h serial/batch exact/random oracle and timing")
    parser.add_argument("--compare-full-score", action="store_true",
                        help="GPU first snapshot: full all-head exact vs randomized complete score response")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = numerical_sources()
    configuration = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    write_json(args.output/"configuration.json", dict(arguments=configuration, sources=source_hashes,
        role="numerical/performance diagnostic only; no population quality evidence"))
    result = cpu_check(args) if args.cpu else gpu_check(args)
    if numerical_sources() != source_hashes:
        raise ValueError("readiness numerical sources changed while running; preserve logs and use a new attempt")
    result["source_sha256"] = source_hashes
    write_json(args.output / "diagnostic.json", result)

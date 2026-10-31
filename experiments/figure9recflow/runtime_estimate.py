"""CPU-only engineering runtime projection from completed bounded benchmarks.

This is a scheduling estimate, not a performance-paper measurement or a
confidence interval. The population and methods are fixed; no result is chosen
by speed or quality. Distinct runs may cover different edges, but duplicate or
missing edges are errors. No experiment, model load, or CUDA call is started.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .common import HERE, write_json


def _json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required completed-run artifact missing: {path}")
    return json.loads(path.read_text())


def _number(value, context):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Expected a finite nonnegative value: {context}")
    return value


def _scale(seconds, sample, full, context):
    if sample <= 0:
        if full > 0:
            raise ValueError(f"Benchmark has no measured {context}; cannot extrapolate full work")
        return 0.0
    return float(seconds * full / sample)


def _prefetch_startup(log_path, edge_id, rank, outside_seconds):
    """First elapsed minus engine time includes prefetch setup and first save.

    This is a coarse fixed-startup estimate, not a precise spawn-only timer.
    Missing/invalid progress keeps the old conservative outside-user estimate.
    """
    result = dict(log=str(log_path), rank=rank, estimated_seconds=0.0,
                  status="missing_log_conservative_fallback")
    if not log_path.is_file():
        return result
    with log_path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if (not isinstance(row, dict) or row.get("edge") != edge_id or
                    row.get("rank") != rank or row.get("completed") != 1):
                continue
            try:
                elapsed = _number(row["elapsed_seconds"], "first progress elapsed")
                engine = _number(row["last_user_seconds"], "first user engine time")
            except (KeyError, TypeError, ValueError):
                result["status"] = "invalid_first_progress_conservative_fallback"
                return result
            startup = elapsed - engine
            if startup < 0 or startup > outside_seconds + max(.1, .01 * outside_seconds):
                result["status"] = "inconsistent_first_progress_conservative_fallback"
                return result
            result.update(status="estimated_from_first_completed_user",
                          estimated_seconds=min(startup, outside_seconds),
                          first_uid=row.get("uid"), first_elapsed_seconds=elapsed,
                          first_user_seconds=engine)
            return result
    result["status"] = "no_first_progress_conservative_fallback"
    return result


def _nnls(x, y):
    """Exact active-set enumeration for only three nonnegative coefficients."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    scales = np.maximum(np.max(np.abs(x), axis=0), 1.0)
    normalized = x / scales
    best = np.zeros(x.shape[1])
    error = float(y @ y)
    for size in range(1, x.shape[1] + 1):
        for columns in combinations(range(x.shape[1]), size):
            beta = np.linalg.lstsq(normalized[:, columns], y, rcond=None)[0]
            if np.any(beta < -1e-10):
                continue
            candidate = np.zeros(x.shape[1])
            candidate[list(columns)] = np.maximum(beta, 0)
            residual = y - normalized @ candidate
            rss = float(residual @ residual)
            if rss < error:
                best, error = candidate, rss
    beta = best / scales
    predicted = x @ beta
    return beta, dict(coefficients=beta.tolist(), samples=len(y),
        design_rank=int(np.linalg.matrix_rank(normalized)),
        mean_absolute_error_seconds=float(np.abs(y - predicted).mean()),
        rms_error_seconds=float(np.sqrt(error / max(1, len(y)))),
        observed_seconds=float(y.sum()), predicted_sample_seconds=float(predicted.sum()))


def _read_edge(run, plan, entry, analytic, full_uids):
    """Load small per-user result artifacts on CPU; never load checkpoints."""
    import torch

    cfg = plan["config"]
    edge_id = entry["id"]
    out = run / edge_id
    summary = _json(out / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError(f"Edge has not completed: {out}")
    for key in ("evaluation_day", "source_checkpoint", "target_checkpoint"):
        if str(entry[key]) != str(analytic[key]):
            raise ValueError(f"Benchmark/analytical edge differs: {edge_id}/{key}")
    if analytic["request_workload"]["candidate_chunk_size"] != cfg["candidate_chunk"]:
        raise ValueError(f"Candidate chunk differs from analytical workload: {edge_id}")
    chosen = np.asarray(plan["eval_uids"], np.int64)
    if not len(chosen) or not np.array_equal(chosen, full_uids[:len(chosen)]):
        raise ValueError(f"Benchmark must retain the fixed population prefix: {run}")
    if len(chosen) != plan["users"] or len(np.unique(chosen)) != len(chosen):
        raise ValueError(f"Benchmark user count/order is inconsistent: {run}")
    if cfg.get("fit_limit") != 256 or cfg.get("fit_epochs") != 8:
        raise ValueError(f"Full fitting timing (256 users, 8 epochs) is required: {run}")
    fit = _json(out / "fit_summary.json")
    expected_maps = [p for p in plan["policies"] if p["method"] == "kv_map"]
    if fit.get("diagnostic_only") or fit["actual_policies"] != expected_maps:
        raise ValueError(f"Fitting artifact is not the complete declared recipe: {out}")
    fit_seconds = _number(fit["capture"]["total_seconds"], f"{edge_id}/fit total")
    fit_capture_seconds = _number(fit["capture"]["seconds"], f"{edge_id}/fit capture")
    fit_optimization_seconds = sum(_number(p["seconds"], "map fit") for p in fit["profiles"].values())
    ranks = [_json(out / f"done_rank{rank}.json") for rank in range(len(plan["gpus"]))]
    assigned = []
    for rank, done in enumerate(ranks):
        if not done.get("completed") or done["rank"] != rank:
            raise ValueError(f"Rank has not completed: {edge_id}/{rank}")
        assigned.extend(done["uids"])
    if len(set(assigned)) != len(assigned) or set(assigned) != set(chosen.tolist()):
        raise ValueError(f"Completed ranks do not cover exactly the fixed benchmark population: {edge_id}")
    ledgers = {}
    for uid in chosen:
        path = out / "users" / f"{uid}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Completed user output missing: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["uid"] != int(uid):
            raise ValueError(f"User output identity differs: {path}")
        ledger = dict(saved["ledger"])
        for name in ("read", "append"):
            key = name + "_device_seconds"
            # Truly silent UIDs have no CUDA intervals and remain in the input.
            if key not in ledger and not ledger["has_history_consumer"]:
                ledger[key] = 0.0
            if key not in ledger:
                raise ValueError(f"Missing actual device timing: {path}/{key}")
        for name in ("initialization_seconds", "total_seconds", "read_device_seconds", "append_device_seconds"):
            ledger[name] = _number(ledger[name], f"{uid}/{name}")
        residual = (ledger["total_seconds"] - ledger["initialization_seconds"] -
                    ledger["read_device_seconds"] - ledger["append_device_seconds"])
        if residual < -max(.01, .01 * ledger["total_seconds"]):
            raise ValueError(f"Device stage timing exceeds sequential user wall time: {path}")
        ledger["host_residual_seconds"] = max(0.0, residual)
        ledgers[int(uid)] = ledger
    counts = {key: sum(x[key] for x in ledgers.values()) for key in (
        "requests", "candidates", "append_events", "ordinary_read_flops_per_policy",
        "ordinary_append_flops_per_policy", "initialization_seconds", "read_device_seconds",
        "append_device_seconds", "total_seconds", "host_residual_seconds")}
    if counts["requests"] != summary["requests"] or counts["candidates"] != summary["candidates"]:
        raise ValueError(f"Completed edge and user ledgers disagree: {edge_id}")
    counts.update(users=len(chosen), consumers=sum(bool(x["has_history_consumer"]) for x in ledgers.values()),
        history_rows=sum(x["initial_rows"] for x in ledgers.values() if x["has_history_consumer"]),
        history_squared=sum(x["initial_rows"] ** 2 for x in ledgers.values() if x["has_history_consumer"]))
    outside_seconds = outside_before_startup = 0.0
    startup_rows = []
    graph_capture = 0.0
    entry_index = next(i for i, e in enumerate(plan["selected_edges"]) if e["id"] == edge_id)
    edge_number = plan.get("edges", [None] * len(plan["selected_edges"]))[entry_index]
    for done in ranks:
        user_seconds = sum(ledgers[int(uid)]["total_seconds"] for uid in done["uids"])
        scoring = _number(done["scoring_seconds"], "rank scoring")
        if scoring + max(1.0, .01 * user_seconds) < user_seconds:
            raise ValueError(f"Rank timing omits saved user work (possibly partial resume): {edge_id}")
        rank_outside = max(0.0, scoring - user_seconds)
        outside_before_startup += rank_outside
        startup = _prefetch_startup(run / f"edge{edge_number}_rank{done['rank']}.log",
                                   edge_id, done["rank"], rank_outside)
        startup_rows.append(startup)
        outside_seconds += max(0.0, rank_outside - startup["estimated_seconds"])
        graph_capture += max((ledgers[int(uid)].get("graph_cumulative_stats", {}).get(
            "capture_seconds", 0.0) for uid in done["uids"]), default=0.0)
    counts["outside_user_seconds"] = outside_seconds
    counts["outside_user_seconds_before_startup_separation"] = outside_before_startup
    counts["prefetch_first_output_startup_seconds_sum"] = sum(x["estimated_seconds"] for x in startup_rows)
    return dict(edge_id=edge_id, run=str(run), config=cfg, analytic=analytic, ledgers=ledgers,
        counts=counts, fit_seconds=fit_seconds, fit_capture_seconds=fit_capture_seconds,
        fit_optimization_seconds=fit_optimization_seconds,
        model_load_seconds=max(_number(x["model_load_seconds"], "model loading") for x in ranks),
        graph_capture_seconds=graph_capture, ranks=ranks,
        prefetch_startup_seconds=max(x["estimated_seconds"] for x in startup_rows),
        startup_estimation_by_rank=startup_rows)


def estimate(runs, analytic_path=HERE / "runs/preparation/analytic_costs.json", *, gpus=4):
    if gpus < 1:
        raise ValueError("GPU count must be positive")
    analytic = _json(analytic_path)
    if analytic.get("status") != "complete_cpu_analytic_preparation" or analytic["population_users"] != 30000:
        raise ValueError("Need the completed full-30000 analytical workload")
    full_edges = {e["edge_id"]: e for e in analytic["edges"]}
    if len(full_edges) != 7:
        raise ValueError("Need all seven accepted release edges, without filtering")
    paths = [Path(p).resolve() for p in runs]
    if len(set(paths)) != len(paths):
        raise ValueError("A benchmark run was supplied twice")
    benchmarks, policy_reference, full_uids = {}, None, None
    for run in paths:
        plan = _json(run / "plan.json")
        completed = _json(run / "summary.json")
        if completed.get("status") != "complete_descriptive_aggregate" or plan.get("mode") != "benchmark":
            raise ValueError(f"Not a completed bounded benchmark: {run}")
        if (run / "failure.json").exists():
            raise ValueError(f"Run still has an unresolved failure record: {run}")
        if policy_reference is None:
            policy_reference = plan["policies"]
            if len(policy_reference) != 29 or policy_reference != analytic["policies"]:
                raise ValueError("Need the complete declared 29-policy workload")
            with np.load(Path(analytic["data"]) / "population.npz", allow_pickle=False) as pop:
                full_uids = np.random.default_rng(plan["config"]["population_seed"]).permutation(pop["train"])
        elif plan["policies"] != policy_reference:
            raise ValueError("Benchmark runs have different policy configurations")
        for entry in plan["selected_edges"]:
            edge_id = entry["id"]
            if edge_id in benchmarks:
                raise ValueError(f"Duplicate edge {edge_id}; explicitly provide one run, never auto-pick the fastest")
            if edge_id not in full_edges:
                raise ValueError(f"Unexpected edge: {edge_id}")
            benchmarks[edge_id] = _read_edge(run, plan, entry, full_edges[edge_id], full_uids)
    missing = [key for key in full_edges if key not in benchmarks]
    if missing:
        raise ValueError(f"Missing completed benchmark edges: {missing}")
    init_x, init_y, host_x, host_y = [], [], [], []
    for b in benchmarks.values():
        for ledger in b["ledgers"].values():
            c = int(ledger["has_history_consumer"])
            n = ledger["initial_rows"] * c
            init_x.append([c, n, n*n])
            init_y.append(ledger["initialization_seconds"])
            host_x.append([1, ledger["requests"], ledger["append_events"]])
            host_y.append(ledger["host_residual_seconds"])
    init_beta, init_fit = _nnls(init_x, init_y)
    host_beta, host_fit = _nnls(host_x, host_y)
    rows = []
    for edge_id in full_edges:
        b = benchmarks[edge_id]
        sample, full = b["counts"], b["analytic"]
        cfg, coverage, work = b["config"], full["actual_coverage"], full["request_workload"]
        graph = bool(cfg.get("cuda_graphs", False))
        read_backend = full["by_backend"][cfg["backend"]]
        append_backend = full["by_backend"][cfg["append_backend"]]
        read_key = "ordinary_graph_read_flops_per_policy" if graph else "ordinary_read_flops_per_policy"
        append_key = "ordinary_graph_append_flops_per_policy" if graph else "ordinary_append_flops_per_policy"
        # Graph setup is separately visible. Per-stage capture attribution was
        # not instrumented, so graph projections explicitly retain its inflation.
        initialization = float(init_beta @ [coverage["first_consumer_users"],
            coverage["once_repair_history_rows"], coverage["sum_repair_history_lengths_squared"]])
        read = _scale(sample["read_device_seconds"], sample["ordinary_read_flops_per_policy"],
                      read_backend[read_key], f"{edge_id} read FLOPs")
        append = _scale(sample["append_device_seconds"], sample["ordinary_append_flops_per_policy"],
                        append_backend[append_key], f"{edge_id} append FLOPs")
        host = float(host_beta @ [30000, work["requests"], coverage["appended_history_events"]])
        outside = _scale(sample["outside_user_seconds"], sample["users"], 30000, "user output/prefetch")
        init_sensitivity = {name: _scale(sample["initialization_seconds"], sample[source], target, name)
            for name, source, target in [
                ("consumer_count", "consumers", coverage["first_consumer_users"]),
                ("history_rows", "history_rows", coverage["once_repair_history_rows"]),
                ("history_squared", "history_squared", coverage["sum_repair_history_lengths_squared"]) ]}
        read_sensitivity = dict(flops=read,
            candidates=_scale(sample["read_device_seconds"], sample["candidates"], work["candidates"], "candidates"),
            requests=_scale(sample["read_device_seconds"], sample["requests"], work["requests"], "requests"))
        append_sensitivity = dict(flops=append,
            events=_scale(sample["append_device_seconds"], sample["append_events"],
                          coverage["appended_history_events"], "append events"))
        outside_sensitivity = dict(users=outside, requests=_scale(sample["outside_user_seconds"],
            sample["requests"], work["requests"], "outside-user request workload"),
            remaining_as_fixed=sample["outside_user_seconds"] * gpus / len(b["ranks"]))
        fixed = b["model_load_seconds"] + b["fit_seconds"] + b["prefetch_startup_seconds"]
        scalable = initialization + read + append + host + outside
        extremes = [fixed + (i + r + a + host + o) / gpus
                    for i in [initialization, *init_sensitivity.values()]
                    for r in read_sensitivity.values() for a in append_sensitivity.values()
                    for o in outside_sensitivity.values()]
        central = fixed + scalable / gpus
        rows.append(dict(edge_id=edge_id, evaluation_day=full["evaluation_day"], benchmark_run=b["run"],
            benchmark_users=sample["users"], benchmark_silent_users=sample["users"] - sample["consumers"],
            execution=dict(backend=cfg["backend"], append_backend=cfg["append_backend"],
                           candidate_chunk=cfg["candidate_chunk"], cuda_graphs=graph),
            measured_benchmark=sample,
            fixed_seconds=dict(model_loading=b["model_load_seconds"], fitting=b["fit_seconds"],
                prefetch_and_first_output_startup=b["prefetch_startup_seconds"],
                fitting_teacher_capture=b["fit_capture_seconds"],
                fitting_optimization_profile_sum=b["fit_optimization_seconds"],
                graph_capture_embedded_in_stage_measurements=b["graph_capture_seconds"]),
            startup_estimation_by_rank=b["startup_estimation_by_rank"],
            full_population_aggregate_worker_seconds=dict(initialization=initialization,
                read=read, append=append, host_processing=host, output_and_prefetch=outside),
            workload_ratio_sensitivity_seconds=dict(initialization=init_sensitivity,
                read=read_sensitivity, append=append_sensitivity, output_and_prefetch=outside_sensitivity),
            balanced_per_edge_wall_seconds=central,
            workload_sensitivity_wall_seconds=[min(extremes), max(extremes)],
            whole_edge_one_gpu_seconds=fixed + scalable,
            engineering_budget_seconds_25_to_40_percent=[central * 1.25, central * 1.40]))
    central = sum(row["balanced_per_edge_wall_seconds"] for row in rows)
    low = sum(row["workload_sensitivity_wall_seconds"][0] for row in rows)
    high = sum(row["workload_sensitivity_wall_seconds"][1] for row in rows)
    # An alternative matches independent whole-edge workers; do not confuse it
    # with distributing each edge's fixed population across all four GPUs.
    loads, assignments = [0.0] * gpus, [[] for _ in range(gpus)]
    for row in sorted(rows, key=lambda x: (-x["whole_edge_one_gpu_seconds"], x["edge_id"])):
        rank = int(np.argmin(loads))
        loads[rank] += row["whole_edge_one_gpu_seconds"]
        assignments[rank].append(row["edge_id"])
    limits = [
        "Engineering projection only: not measured full-run wall time, a performance-paper claim, or a statistical confidence interval.",
        "Central assumes four equally available GPUs and balanced independent users within each edge; edge loads/fits are fixed and not divided by GPU count.",
        "Initialization is nonnegative regression on consumer count, sum N, sum N^2; it includes shared experimental initialization, not a sum of independently deployed policy costs.",
        "Read/append use measured all-policy device time times matching per-policy arithmetic ratios; the common method-count factor cancels. Arithmetic ratios do not guarantee hardware speed ratios.",
        "All fixed-prefix users, including silent users, remain in the projection. No edge or timing is chosen by quality or by being fastest.",
        "First completed-user log elapsed time minus its engine time estimates fixed prefetch/startup, including the first output save's small conservative portion. The slowest measured rank startup is charged once per edge, assuming target ranks start concurrently; it is not multiplied by the full population.",
        "Host residual regression and remaining output/prefetch scaling are approximate. Missing or inconsistent first-progress logs retain the conservative per-user outside-time extrapolation and are explicitly flagged; no startup value is invented.",
        "Other process setup, CPU metric aggregation/bootstrap, filesystem contention and sustained multi-GPU slowdown are not separately benchmarked; the 25–40% margin is an engineering allowance, not a guarantee.",
    ]
    if any(b["graph_capture_seconds"] > 0 for b in benchmarks.values()):
        limits.append("Graph capture/warmup time is retained in measured stages because per-stage attribution is unavailable; graph estimates may overproject one-time capture and should not be described as steady-state acceleration.")
    return dict(status="engineering_estimate", source_runs=[str(p) for p in paths],
        analytic_costs=str(Path(analytic_path).resolve()), gpus=gpus, population_users=30000,
        edge_count=len(rows), policy_count=len(policy_reference), edges=rows,
        initialization_regression=dict(features=["consumer", "history_rows", "history_rows_squared"], **init_fit),
        host_regression=dict(features=["user", "requests", "append_events"], **host_fit),
        central_wall_seconds=central, central_wall_hours=central / 3600,
        workload_sensitivity_wall_hours=[low / 3600, high / 3600],
        engineering_budget_hours_25_to_40_percent=[central * 1.25 / 3600, central * 1.4 / 3600],
        workload_sensitivity_with_40_percent_margin_hours=high * 1.4 / 3600,
        whole_edge_parallel_sensitivity=dict(assignments=assignments,
            worker_seconds=loads, makespan_hours=max(loads) / 3600,
            engineering_budget_hours_25_to_40_percent=[max(loads) * 1.25 / 3600, max(loads) * 1.4 / 3600]),
        limitations=limits, formal_started=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--analytic-costs", type=Path, default=HERE / "runs/preparation/analytic_costs.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    result = estimate(args.runs, args.analytic_costs, gpus=args.gpus)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "edge_count", "central_wall_hours",
        "engineering_budget_hours_25_to_40_percent", "workload_sensitivity_wall_hours")}), flush=True)


if __name__ == "__main__":
    main()

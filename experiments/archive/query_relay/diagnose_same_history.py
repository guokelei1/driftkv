#!/usr/bin/env python3
"""Within-history query-gate ablation on a retained 16-candidate workload.

The dynamic writer trajectory is held fixed. Only transient recommendation
reads are replaced: each layer uses its gates averaged over this same candidate
set, at that shadow branch's own queries. No fitting or new native controls.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from hstu_kvcache.evaluation.release_context import END, RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from experiments.archive.partial_refresh.analyze import summarize
from experiments.archive.query_relay.evaluate import IDENTITY, gap, load_candidates
from hstu_kvcache.methods.cachetide.adapter import operation_gate_query
from hstu_kvcache.methods.cachetide.session import RelaySession
from hstu_kvcache.evaluation.cachetide_workload import candidate_rows
from hstu_kvcache.common.artifacts import sha256, write_json


def same_set_mean_reader(adapter, state, code_count=None, gate_query=None):
    """A read-only counterfactual, recomputing set means at every layer.

    The means have shape [batch,heads,1,rank]. A single-user set is required:
    averaging different users would introduce the confound this probe removes.
    """
    first_query = None

    def read(index, attention, q, k, v):
        nonlocal first_query
        if q.shape[0] != 1:
            raise ValueError("the diagnostic requires one fixed history per set")
        if index == 0:
            first_query = q
        layer = adapter.layers[index]
        means = None
        if code_count is None or code_count:
            gate_q = gate_query if adapter.gate_source == "item" else \
                (first_query if adapter.gate_source == "first" else q)
            if gate_q is None:
                raise ValueError("first-source shadow requires this operation's q0")
            means = tuple(values.mean(2, keepdim=True) for values in layer.query_gates(gate_q))
        return layer.read(attention, q, k, v, state.codes[index],
                          code_count=code_count, gate_constants=means)
    return read


class WithinHistoryGates:
    """Separate within-set variation from variation of set means."""
    def __init__(self):
        self.entries = {}

    @torch.no_grad()
    def add(self, version, adapter, queries, code_rows, gate_query=None):
        if adapter.gate_source == "item" and gate_query is None:
            raise ValueError("item gate statistics require the request's Current item embeddings")
        for index, (layer, q) in enumerate(zip(adapter.layers, queries, strict=True)):
            if adapter.gate_source == "first":
                q = queries[0]
            elif adapter.gate_source == "item":
                q = gate_query
            for side, values in zip(("key", "value"), layer.query_gates(q), strict=True):
                key = version, index, side
                entry = self.entries.setdefault(key, dict(active_sets=0, inactive_sets=0))
                if not code_rows:
                    entry["inactive_sets"] += 1
                    continue
                data = values[0].detach().double().cpu()  # heads,candidates,rank
                mean, variance = data.mean(1), data.var(1, unbiased=False)
                if "sum_mean" not in entry:
                    for name in ("sum_mean", "sum_mean_squared", "sum_variance", "sum_std", "nonconstant"):
                        entry[name] = torch.zeros_like(mean)
                    entry["saturated_fraction_sum"] = 0.
                entry["active_sets"] += 1
                entry["sum_mean"] += mean
                entry["sum_mean_squared"] += mean.square()
                entry["sum_variance"] += variance
                entry["sum_std"] += variance.sqrt()
                entry["nonconstant"] += (variance.sqrt() > 1e-5)
                entry["saturated_fraction_sum"] += float(((data < .05) | (data > 1.95)).double().mean())

    def result(self):
        rows = []
        for (version, layer, side), entry in sorted(self.entries.items()):
            row = dict(version=version, layer=layer, side=side,
                       active_sets=entry["active_sets"], inactive_sets=entry["inactive_sets"])
            n = entry["active_sets"]
            if n:
                means = entry["sum_mean"] / n
                within_var = entry["sum_variance"] / n
                between_var = (entry["sum_mean_squared"] / n - means.square()).clamp_min(0)
                row.update(mean_gate=means.tolist(), within_set_coordinate_rms_std=within_var.sqrt().tolist(),
                    mean_within_set_coordinate_std=float((entry["sum_std"] / n).mean()),
                    rms_within_set_coordinate_std=float(within_var.mean().sqrt()),
                    rms_between_set_mean_std=float(between_var.mean().sqrt()),
                    nonconstant_coordinate_set_fraction=float((entry["nonconstant"] / n).mean()),
                    saturated_coordinate_fraction=entry["saturated_fraction_sum"] / n)
            rows.append(row)
        return dict(rows=rows, scope="Each set shares one history/time and 16 candidates; inactive zero-code sets excluded from variation metrics")


def score_errors(actual, wanted):
    residual = actual.double() - wanted.double()
    return dict(logit_mse=float(residual.square().mean()),
                common_offset_mse=float(residual.mean(-1).square().mean()),
                centered_logit_mse=float((residual - residual.mean(-1, keepdim=True)).square().mean()))


def relative_errors(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["version"], row["round_id"]].append(row)
    errors = defaultdict(lambda: defaultdict(list))
    for (version, _), group in groups.items():
        wanted = torch.tensor([row["first_request_rebuild_logit"] for row in group], dtype=torch.float64)
        for name, field in (("dynamic", "partial_logit"), ("same_set_mean", "same_set_mean_logit")):
            actual = torch.tensor([row[field] for row in group], dtype=torch.float64)
            errors[version][name].append(score_errors(actual, wanted))
    by_release = {version: {name: {metric: float(np.mean([unit[metric] for unit in units]))
                                  for metric in units[0]} for name, units in branches.items()}
                  for version, branches in errors.items()}
    overall = {name: {metric: float(np.mean([record[name][metric] for record in by_release.values()]))
                      for metric in next(iter(by_release.values()))[name]}
               for name in ("dynamic", "same_set_mean")}
    return dict(by_release=by_release, equal_release_mean=overall,
        reference="Retained once-per-release Rebuild trajectory; this is not a per-request Fresh teacher",
        averaging="Equal 16-candidate sets within release, then equal releases")


@torch.no_grad()
def teacher_probe(args, models, adapters, weights, device):
    """Reserved held-out first-read sets against exact Current reconstruction."""
    from hstu_kvcache.evaluation.capture import capture_rows
    panel = ROOT / "results/design_two_2026_10/benchmark_panel"
    binding, groups, _ = load_panel(panel)
    metadata = json.loads((args.calibrations / "v3/calibration.json").read_text())
    uids = list(metadata["validation_uids"])
    if not set(uids) <= set(groups["validation"]):
        raise ValueError("teacher diagnostic must retain validation roles")
    histories = load_history(panel, requested_uids=uids)
    raw_v3, _, _ = capture_rows(models, histories, uids, "v3", binding["known_vocab_size"])
    units, sources = [], {}
    for version, adapter in adapters.items():
        if version == "v3":
            captured = raw_v3
            sources[version] = "Canonical V2 cache at V3 release, captured causally"
        else:
            path = args.calibrations.parent / f"source_{version}/source.pt"
            pack = torch.load(path, map_location="cpu", weights_only=False)
            if pack["settings"]["target"] != version:
                raise ValueError("teacher source pack target differs")
            previous = f"v{int(version[1:]) - 1}"
            if pack["prior_checkpoints"][previous]["sha256"] != weights[previous]["sha256"]:
                raise ValueError("teacher source pack is from a different candidate trajectory")
            captured = pack["rows"]
            sources[version] = dict(path=str(path), sha256=sha256(path))
        cut = dict(RELEASES)[version]
        for uid in uids:
            raw = captured[uid]
            cache, teacher = raw["parent"].to(device), raw["teacher"].to(device)
            items = raw.get("item_ids")
            if items is None:
                times, ids, _ = histories.rows[uid]
                stop = np.searchsorted(times, cut, side="left")
                items = torch.as_tensor(ids[:stop][-cache.seq_len:])[None]
            previous = raw.get("previous_codes")
            if previous is None:
                previous = cache.k.new_zeros(adapter.num_layers, 1, cache.seq_len, adapter.heads, adapter.rank)
            state = adapter.advance_codes(cache, previous.to(device), raw["item_features"].to(device),
                                          item_ids=items.to(device))
            candidates = raw["candidates"].to(device)[None]
            delta = torch.tensor([raw["query_delta"]], device=device)
            wanted = score(models[version], teacher, candidates, delta)[0]
            gate_query = operation_gate_query(adapter, models[version], candidates)
            dynamic = score(models[version], cache, candidates, delta,
                history_read_fn=adapter.make_history_read(state, gate_query=gate_query))[0]
            constant = score(models[version], cache, candidates, delta,
                history_read_fn=same_set_mean_reader(adapter, state, gate_query=gate_query))[0]
            units.append(dict(uid=uid, version=version,
                dynamic=score_errors(dynamic, wanted), same_set_mean=score_errors(constant, wanted)))
    by_release = {}
    for version in adapters:
        group = [unit for unit in units if unit["version"] == version]
        by_release[version] = {name: {metric: float(np.mean([unit[name][metric] for unit in group]))
                                     for metric in group[0][name]}
                              for name in ("dynamic", "same_set_mean")}
    return dict(validation_uids=uids, by_release=by_release, units=units, sources=sources,
        scope="16 reserved validation users at each release; same 16 causal calibration candidates; exact Current reconstructed teacher; no labels or fit")


@torch.no_grad()
def run(args):
    began = time.perf_counter()
    if args.users < 0 or args.start_user < 0 or not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid retained user shard")
    original = json.loads((args.reference.parent / "configuration.json").read_text())
    if original["workload"]["kind"] != "behavior" or original["workload"]["candidates"] != 16:
        raise ValueError("use an existing behavior workload with 16-candidate sets")
    panel = Path(original["panel"])
    if sha256(panel / "binding.json") != original["panel_sha256"]:
        raise ValueError("retained panel binding changed")
    binding, groups, _ = load_panel(panel)
    # A per-shard reference still records the complete frozen population. Apply
    # slicing once in that population, then require this reference to cover it.
    retained_population = original.get("population", original["selected_uids"])
    population = retained_population[:args.users] if args.users else retained_population
    selected = population[args.start_user:][args.shard_index::args.shards]
    if not selected:
        raise ValueError("empty retained user shard")
    if not set(selected) <= set(original["selected_uids"]):
        raise ValueError("retained reference does not cover the selected frozen population shard")
    versions = list(original["versions"])
    releases = tuple((version, at) for version, at in RELEASES if version in versions)
    end = RELEASES[int(versions[-1][1:]) - 2][1] if versions[-1] != "v5" else END
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = dict(calibrations=str(args.calibrations), reference=str(args.reference),
        reference_sha256=sha256(args.reference), selected_uids=selected, versions=versions,
        panel=str(panel), panel_sha256=original["panel_sha256"], workload=original["workload"],
        code_preparation=args.preparation, published_prefixes=str(args.published_prefixes) if args.published_prefixes else None,
        shards=args.shards, shard_index=args.shard_index, start_user=args.start_user,
        skip_analysis=args.skip_analysis,
        ablation="Each layer gate averaged over the same 16 candidates using the shadow branch query",
        writer="Unchanged dynamic corrected-writer trajectory; no shadow writes",
        fitting=False, labels="Analysis only", selection="First fixed UIDs from the retained run, before diagnostic outcomes",
        source_sha256={file: sha256(Path(__file__).parent / file)
                       for file in (Path(__file__).name, "session.py", "adapter.py", "evaluate.py")})
    write_json(args.output / "configuration.json", configuration)
    device = setup_device(args.gpu, args.threads)
    models = load_models(binding, device, int(versions[-1][1:]))
    adapters, weights, _ = load_candidates(args.calibrations, device, versions, groups["evaluation"])
    reference_summary = json.loads((args.reference.parent / "summary.json").read_text())
    for version in versions:
        if weights[version]["sha256"] != reference_summary["weights"][version]["sha256"]:
            raise ValueError("dynamic reference was produced by different candidate weights")
    prefixes, prefix_records = {}, {}
    if args.preparation == "compiled":
        if args.published_prefixes is None:
            raise ValueError("compiled diagnostic uses the already published full-catalogue artifacts")
        from hstu_kvcache.methods.cachetide.publication import load_prefix
        for version in versions:
            prefixes[version], prefix_records[version] = load_prefix(
                args.published_prefixes / f"{version}.pt", adapters[version], device=device)
            if prefix_records[version]["bindings"]["calibration_sha256"] != weights[version]["sha256"]:
                raise ValueError("publication and dynamic candidate differ")
    histories = load_history(panel, requested_uids=selected)
    references = {row["request_id"]: row for row in pq.read_table(args.reference,
                    filters=[("uid", "in", selected)]).to_pylist()}
    rows, stats, largest_error, expected_rows = [], WithinHistoryGates(), 0., 0
    for number, uid in enumerate(selected):
        session = RelaySession(NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1]))
        requests = candidate_rows(uid, histories.rows[uid], stride=original["workload"]["stride"],
            candidates=16, known_vocab_size=binding["known_vocab_size"], releases=releases, end=end)
        expected_rows += len(requests)
        for index, (version, at) in enumerate(releases):
            adapter = adapters[version]
            session.publish(models[version], int(version[1:]), at, adapter, prefix=prefixes.get(version))
            sets = defaultdict(list)
            for request in requests:
                if request["version"] == version:
                    sets[request["query_timestamp"], request["round_id"]].append(request)
            for (timestamp, _), pending in sorted(sets.items()):
                if len(pending) != 16:
                    raise ValueError("incomplete candidate set")
                candidates = torch.tensor([[row["item_idx"] for row in pending]], device=device)
                dynamic, trace = session.score(timestamp, candidates, trace=True)
                gate_query = operation_gate_query(adapter, models[version], candidates)
                constant = score(models[version], session.cache, candidates, gap(session.trajectory, timestamp),
                    history_read_fn=same_set_mean_reader(adapter, session.state, session.coded_rows,
                                                        gate_query=gate_query))[0]
                stats.add(version, adapter, trace.queries, session.coded_rows, gate_query=gate_query)
                for request, value, mean in zip(pending, dynamic[0].cpu().tolist(), constant[0].cpu().tolist(), strict=True):
                    prior = references[request["request_id"]]
                    if any(prior[key] != request[key] for key in IDENTITY):
                        raise ValueError("retained reference candidate identity differs")
                    error = abs(value - prior["partial_logit"])
                    largest_error = max(largest_error, error)
                    # float32 composition/kernel round-off, not permission to use a different trajectory.
                    if error > 2e-4:
                        raise ValueError(f"dynamic trajectory mismatch: {error}, uid={uid}, {version}")
                    rows.append(dict(request, partial_logit=value, same_set_mean_logit=mean,
                        reuse_logit=prior["reuse_logit"], full_logit=prior["first_request_rebuild_logit"],
                        first_request_rebuild_logit=prior["first_request_rebuild_logit"],
                        retained_dynamic_logit=prior["partial_logit"], coded_rows=session.coded_rows))
            session.advance(releases[index + 1][1] if index + 1 < len(releases) else end)
        if (number + 1) % 16 == 0 or number + 1 == len(selected):
            print(json.dumps(dict(status="same_history_diagnostic", users=number + 1, candidates=len(rows),
                                  elapsed_seconds=time.perf_counter() - began)), flush=True)
    if len(rows) != expected_rows or len(rows) != len(references):
        raise ValueError("diagnostic did not cover the selected retained request identities")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output / "scores.parquet")
    quality = constant_quality = paired = None
    if not args.skip_analysis:
        quality = summarize(table, selected, repetitions=args.bootstrap, releases=versions)
        constant_quality = summarize([dict(row, partial_logit=row["same_set_mean_logit"]) for row in rows],
                                     selected, repetitions=args.bootstrap, releases=versions)
        comparison = summarize([dict(row, full_logit=row["same_set_mean_logit"]) for row in rows],
                               selected, repetitions=args.bootstrap, releases=versions)
        paired = dict(overall=comparison["overall"]["paired"]["partial_minus_full_auc"],
                      by_release={version: record["paired"]["partial_minus_full_auc"]
                                  for version, record in comparison["by_release"].items()})
    teacher = None if args.skip_teacher else teacher_probe(args, models, adapters, weights, device)
    result = dict(status="complete", selected_uids=selected, weights=weights,
        dynamic_quality=quality, same_set_mean_quality=constant_quality,
        dynamic_minus_same_set_mean_auc=paired, gate_statistics=stats.result(),
        trajectory_score_errors=relative_errors(rows), heldout_teacher=teacher,
        checks=dict(native_controls_reused=True, no_shadow_writes=True, matched_request_identities=True,
                    dynamic_replay_max_absolute_logit_error=largest_error),
        elapsed_seconds=time.perf_counter() - began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        scope="Development same-history read-only ablation; not a new writer policy, training objective, or loaded serving performance claim",
        configuration=configuration, published_prefixes=prefix_records)
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status="complete", dynamic_minus_same_set_mean_auc=paired,
                          heldout_teacher=None if teacher is None else teacher["by_release"])), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrations", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--users", type=int, default=128)
    p.add_argument("--start-user", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--preparation", choices=("explicit", "compiled"), default="explicit")
    p.add_argument("--published-prefixes", type=Path)
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Write matched scores/statistics for a shard; defer shared full-population bootstrap")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

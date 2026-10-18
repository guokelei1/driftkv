#!/usr/bin/env python3
"""Frozen-user development replay for query gates and continuous code updates.

Native Reuse and once-per-release Rebuild scores are reused only on the original
request protocol. New workloads require independent native controls. Labels are
used solely in output analysis, never in the session or preparation policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from design_two.common import END, RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.design_three.lifecycle import LifecycleSession
from hstu_kvcache.design_two.replay import NativeTrajectory
from partial_refresh_2026_10.analyze import summarize
from query_relay_2026_10.adapter import ReleasePrefix, UnifiedRelayAdapter, operation_gate_query
from query_relay_2026_10.session import RelaySession
from unified_reuse_2026_09.common import sha256, write_json

PANEL = ROOT / "results/design_system_2026_10/phase2/representative4096/panel"
REFERENCE = ROOT / "results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_support_lazy/development4096/scores.parquet"
IDENTITY = ("uid", "version", "label", "item_idx", "query_timestamp")


def load_candidates(folder, device, versions, population, read_shared=None, read_private=None):
    adapters, records, gate_means = {}, {}, {}
    for version in versions:
        path = folder / version / "calibration.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["kind"] != UnifiedRelayAdapter.kind:
            raise ValueError(f"not a unified relay checkpoint: {path}")
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        roles = set(metadata["fit_uids"]) | set(metadata["validation_uids"])
        if roles & set(population):
            raise ValueError("evaluation overlaps calibration roles")
        adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
        adapter.load_state_dict(payload["state_dict"])
        if read_shared is not None:
            adapter.set_read_shared(read_shared)
        if read_private is not None:
            adapter.set_read_private(read_private)
        adapters[version] = adapter
        constants = payload.get("gate_constants", metadata.get("gate_constants"))
        if constants is not None:
            constants = torch.as_tensor(constants, device=device, dtype=next(adapter.parameters()).dtype)
            gate_means[version] = tuple((row[0], row[1]) for row in constants)
        records[version] = dict(path=str(path), sha256=sha256(path),
                                metadata_sha256=sha256(metadata_path), config=payload["config"],
                                gate_constants_available=constants is not None)
        if adapter.feature_mode == "origin_encoder":
            from query_relay_2026_10.fixed_features import load_bank
            bank = load_bank(adapter.feature_bank_path, adapter.feature_bank_sha256,
                             device="cpu", dtype=next(adapter.parameters()).dtype)
            records[version]["feature_bank"] = dict(path=adapter.feature_bank_path,
                sha256=adapter.feature_bank_sha256, storage_bytes=bank.storage_bytes(),
                item_table_bytes=bank.table.numel() * bank.table.element_size(),
                catalogue_items=bank.metadata["catalogue_items"],
                scope="One immutable shared asset; not per-user or per-release persistent state")
    return adapters, records, gate_means


class MappedPrefix:
    """Development-only shared item subset; raw item IDs keep their meaning."""

    def __init__(self, prefix, item_ids):
        self.prefix, self.item_ids = prefix, item_ids

    def _indices(self, items):
        items = np.asarray(items)
        positions = np.searchsorted(self.item_ids, items)
        if np.any(positions >= len(self.item_ids)) or not np.array_equal(self.item_ids[positions], items):
            raise ValueError("history item is outside the compiled development subset")
        return torch.as_tensor(positions, device=self.prefix.D.device, dtype=torch.long)

    def materialize(self, cache, anchors, items):
        return self.prefix.materialize(cache, anchors, items,
                                       table_item_ids=self._indices(items))

    def anchorize(self, cache, current, items):
        return self.prefix.anchorize(cache, current, items,
                                     table_item_ids=self._indices(items))

    def zero_anchors(self, cache, items):
        return self.prefix.zero_anchors(cache, items,
                                        table_item_ids=self._indices(items))


@torch.no_grad()
def compile_prefixes(models, adapters, histories, selected, releases, end, chunk_size):
    """Chunk item preparation; subset size is diagnostic, not deployment cost."""
    all_items = []
    for uid in selected:
        times, items, _ = histories.rows[uid]
        all_items.append(items[:int(np.searchsorted(times, end, side="left"))])
    item_ids = np.unique(np.concatenate(all_items)) if all_items else np.array([], dtype=np.int64)
    current, result, records = None, {}, {}
    for version, _ in releases:
        adapter, model = adapters[version], models[version]
        if adapter.mode not in ("A", "B"):
            raise ValueError("compiled evaluation requires A/B update checkpoints")
        if current is None:
            current = ReleasePrefix.identity(adapter, len(item_ids))
        p, q, ds = [], [], []
        for index, layer in enumerate(adapter.layers):
            # Fixed feature equality is essential for exact composition.
            old_layer = current.layers[index]
            if layer.feature_mode != old_layer.feature_mode or \
                    layer.feature_bank_sha256 != old_layer.feature_bank_sha256:
                raise ValueError("compiled releases changed their immutable feature bank")
            for j in range(len(layer.branches)):
                if not torch.equal(getattr(layer, f"native_basis_{j}"),
                                   getattr(current.layers[index], f"native_basis_{j}")):
                    raise ValueError("compiled releases changed their fixed native feature basis")
            if layer.xi_width != current.layers[index].xi_width:
                raise ValueError("compiled releases changed fixed summary width")
            rotations = layer.rotation_blocks()
            pblocks, qblocks = [], []
            for branch, a, b in zip(layer.branches, rotations, layer.native_update, strict=True):
                section = slice(branch.code_start, branch.code_start + branch.code_width)
                pblocks.append(a @ current.P[index, section, section])
                qblocks.append(a @ current.Q[index, section, section] + b)
            p.append(torch.block_diag(*pblocks))
            q.append(torch.block_diag(*qblocks))
            d = torch.empty_like(current.D[index])
            for start in range(0, len(item_ids), chunk_size):
                stop = min(start + chunk_size, len(item_ids))
                ids = torch.as_tensor(item_ids[start:stop], device=d.device)[None]
                embedding = model.lookup_item_embeddings(ids)[0]
                for branch, a in zip(layer.branches, rotations, strict=True):
                    section = slice(branch.code_start, branch.code_start + branch.code_width)
                    d[start:stop, section] = torch.einsum("cd,mdr->mcr", a,
                                                        current.D[index, start:stop, section])
                d[start:stop] += layer.item_coefficients(embedding)
            ds.append(d)
        current = ReleasePrefix(torch.stack(p), torch.stack(q), torch.stack(ds),
                                tuple(adapter.layers), current.release + 1)
        result[version] = MappedPrefix(current, item_ids)
        records[version] = dict(items=len(item_ids), storage_bytes=current.storage_bytes(),
                                scope="Observed development histories only; not full-catalogue cost")
    return result, records


class GateStatistics:
    def __init__(self):
        self.units = {}

    @torch.no_grad()
    def add(self, version, adapter, queries, code_rows, gate_query=None):
        if adapter.gate_source == "item" and gate_query is None:
            raise ValueError("item gate statistics require this request's Current item embeddings")
        for layer, (module, query) in enumerate(zip(adapter.layers, queries, strict=True)):
            if adapter.gate_source == "first":
                query = queries[0]
            elif adapter.gate_source == "item":
                query = gate_query
            for side, values in zip(("key", "value"), module.query_gates(query), strict=True):
                data = values.detach().double()
                key = version, layer, side
                count = data.shape[0] * data.shape[2]
                entry = self.units.setdefault(key, dict(count=0, active_count=0,
                    sum=torch.zeros(data.shape[1], data.shape[3], dtype=torch.float64, device=data.device),
                    squared=torch.zeros(data.shape[1], data.shape[3], dtype=torch.float64, device=data.device)))
                entry["count"] += count
                entry["active_count"] += count if code_rows else 0
                entry["sum"] += data.sum((0, 2))
                entry["squared"] += data.square().sum((0, 2))

    def result(self):
        rows = []
        for (version, layer, side), entry in sorted(self.units.items()):
            mean = (entry["sum"] / entry["count"]).cpu()
            variance = (entry["squared"] / entry["count"]).cpu().sub(mean.square()).clamp_min(0)
            std = variance.sqrt()
            rows.append(dict(version=version, layer=layer, side=side,
                queries=entry["count"], queries_with_coded_prefix=entry["active_count"],
                mean=mean.tolist(), std=std.tolist(), mean_coordinate_std=float(std.mean()),
                nonconstant_coordinate_fraction=float((std > 1e-5).double().mean())))
        return dict(rows=rows, scope="Query-gate variation; not evidence of a quality contribution")


def records(counter, names):
    return [dict(**dict(zip(names, key, strict=True)), count=int(value))
            for key, value in sorted(counter.items())]


def gap(trajectory, at):
    return torch.tensor([0 if trajectory.last_timestamp is None else int(at) - trajectory.last_timestamp],
                        dtype=torch.float32, device=trajectory.cache.k.device)


@torch.no_grad()
def run(args):
    began = time.perf_counter()
    if args.users < 0 or args.start_user < 0 or not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid frozen population slice")
    panel, groups, requests = load_panel(args.panel)
    population = list(groups["evaluation"])
    if args.users:
        population = population[:args.users]
    selected = population[args.start_user:][args.shard_index::args.shards]
    if not selected:
        raise ValueError("selected user shard is empty")
    releases = tuple(row for row in RELEASES if int(row[0][1:]) <= args.last_version)
    versions = [version for version, _ in releases]
    end = RELEASES[args.last_version - 2][1] if args.last_version < 5 else END
    original_protocol = args.workload == "feedback" and sha256(args.panel / "binding.json") == sha256(PANEL / "binding.json")
    if args.compute_new_controls and original_protocol:
        raise ValueError("original native controls are already completed; reuse their retained scores")
    if args.compute_new_controls and args.reuse_new_controls:
        raise ValueError("choose either computing or reusing matched controls")
    if args.control_cache is not None:
        if not args.compute_new_controls or original_protocol:
            raise ValueError("a control cache is used only while computing new-task controls")
        prior_settings = json.loads((args.control_cache.parent / "configuration.json").read_text())
        if prior_settings["panel_sha256"] != sha256(args.panel / "binding.json") or prior_settings["workload"]["kind"] != args.workload or prior_settings["workload"]["stride"] != args.stride or prior_settings["workload"]["candidates"] != args.candidates:
            raise ValueError("control cache is from a different registered task")
    if not args.compute_new_controls and not original_protocol:
        if not args.reuse_new_controls:
            raise ValueError("a new task requires new controls or an explicitly matched retained control")
        reference_settings = json.loads((args.reference.parent / "configuration.json").read_text())
        expected_workload = dict(kind=args.workload, stride=args.stride if args.workload == "behavior" else None,
            candidates=args.candidates if args.workload == "behavior" else None,
            task="causal next-item with uniform sampled negatives" if args.workload == "behavior" else "retained feedback labels")
        if reference_settings["workload"] != expected_workload or reference_settings["panel_sha256"] != sha256(args.panel / "binding.json"):
            raise ValueError("retained native controls belong to a different task or panel")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(selected_uids=selected, population=population, calibration_root=str(args.calibrations),
        panel=str(args.panel), panel_sha256=sha256(args.panel / "binding.json"), versions=versions,
        reference=None if args.compute_new_controls else str(args.reference),
        reference_sha256=None if args.compute_new_controls else sha256(args.reference),
        shards=args.shards, shard_index=args.shard_index, start_user=args.start_user,
        code_preparation=args.preparation, anchor_inverse=args.anchor_inverse,
        native_writes="current model with corrected history",
        storage_floats_per_layer_row=48, read_shared=args.read_shared, read_private=args.read_private,
        mean_gate_control=args.mean_gate_control, source_sha256={name: sha256(Path(__file__).parent / name)
            for name in ("evaluate.py", "session.py", "adapter.py")},
        cost_scope="Quality panel operation shapes only; fixed30k complete cost reported separately",
        role="development_exploration", no_online_labels_or_teacher=True)
    if args.anchor_inverse == "actual":
        settings["source_sha256"]["precision.py"] = sha256(Path(__file__).parent / "precision.py")
    settings["workload"] = dict(kind=args.workload, stride=args.stride if args.workload == "behavior" else None,
        candidates=args.candidates if args.workload == "behavior" else None,
        task="causal next-item with uniform sampled negatives" if args.workload == "behavior" else "retained feedback labels")
    settings["control_cache"] = None if args.control_cache is None else dict(
        path=str(args.control_cache), sha256=sha256(args.control_cache))
    settings["published_prefixes"] = None if args.published_prefixes is None else str(args.published_prefixes)
    write_json(args.output / "configuration.json", settings)
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device, last_version=args.last_version)
    adapters, weights, gate_means = load_candidates(args.calibrations, device, versions,
                                                   list(groups["evaluation"]), args.read_shared, args.read_private)
    if args.mean_gate_control and set(gate_means) != set(versions):
        raise ValueError("mean-gate control needs fit-only gate_constants in every checkpoint")
    histories = load_history(args.panel, requested_uids=selected)
    if args.workload == "behavior":
        from query_relay_2026_10.workload import candidate_rows
        requests = {uid: candidate_rows(uid, histories.rows[uid], stride=args.stride,
            candidates=args.candidates, known_vocab_size=panel["known_vocab_size"]) for uid in selected}
    if args.published_prefixes is not None:
        if args.preparation != "compiled":
            raise ValueError("published prefix artifacts require compiled preparation")
        from query_relay_2026_10.publish import load_prefix
        prefixes, prefix_records = {}, {}
        for version in versions:
            path = args.published_prefixes / f"{version}.pt"
            prefixes[version], prefix_records[version] = load_prefix(path, adapters[version], device=device)
            published = prefix_records[version]
            if published["target_version"] != int(version[1:]) or published["bindings"]["calibration_sha256"] != weights[version]["sha256"] or published["bindings"]["current_model_sha256"] != panel["models"][version]["checkpoint_sha256"]:
                raise ValueError("publication does not match the candidate and model being evaluated")
            prefix_records[version]["artifact_sha256"] = sha256(path)
    else:
        prefixes, prefix_records = ({}, {}) if args.preparation == "explicit" else compile_prefixes(
            models, adapters, histories, selected, releases, end, args.item_chunk)
    if args.anchor_inverse == "actual":
        if args.preparation != "compiled":
            raise ValueError("actual anchor inverse requires compiled preparation")
        from query_relay_2026_10.precision import StableReleasePrefix
        for version, prefix in prefixes.items():
            stable = StableReleasePrefix.wrap(prefix)
            prefixes[version] = stable
            prefix_records[version]["anchor_numerics_extension"] = stable.inverse_cost()
    references = {} if args.compute_new_controls else {row["request_id"]: row for row in pq.read_table(
        args.reference, filters=[("uid", "in", selected), ("version", "in", versions)]).to_pylist()}
    cached_controls = {} if args.control_cache is None else {row["request_id"]: row for row in pq.read_table(
        args.control_cache, filters=[("uid", "in", selected), ("version", "in", versions)]).to_pylist()}
    rows, units, counters = [], [], Counter()
    gate_stats = GateStatistics()
    combined_native, combined_code_reads = Counter(), Counter()
    for number, uid in enumerate(selected):
        make = lambda: NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1])
        session = RelaySession(make())
        reuse = make() if args.compute_new_controls else None
        rebuild = LifecycleSession(make()) if args.compute_new_controls else None
        by_release = {}
        for index, (version, at) in enumerate(releases):
            source_births = dict(Counter(int(v) for v in session.trajectory.producers))
            previous = Counter(session.costs)
            session.publish(models[version], int(version[1:]), at, adapters[version],
                            prefix=prefixes.get(version))
            if args.compute_new_controls:
                reuse.switch(models[version], int(version[1:]), at)
                rebuild.publish(models[version], int(version[1:]), at, None,
                                policy="rebuild_on_first_request")
            by_time = defaultdict(list)
            for request in requests[uid]:
                if request["version"] == version:
                    by_time[int(request["query_timestamp"])].append(request)
            first_births = None
            for timestamp, pending in sorted(by_time.items()):
                session.prepare_read(timestamp)
                if first_births is None:
                    first_births = dict(Counter(int(v) for v in session.trajectory.producers))
                if args.compute_new_controls:
                    reuse.advance(timestamp)
                    rebuild.prepare_read(timestamp)
                for offset in range(0, len(pending), args.query_batch):
                    batch = pending[offset:offset + args.query_batch]
                    candidates = torch.tensor([[row["item_idx"] for row in batch]], device=device)
                    actual, trace = session.score(timestamp, candidates, trace=True)
                    actual = actual[0]
                    if not torch.isfinite(actual).all():
                        raise RuntimeError("nonfinite relay scores")
                    item_gate = operation_gate_query(adapters[version], models[version], candidates)
                    gate_stats.add(version, adapters[version], trace.queries, session.coded_rows,
                                   gate_query=item_gate)
                    constant = None
                    if args.mean_gate_control:
                        constant = native_score(models[version], session.cache, candidates,
                            gap(session.trajectory, timestamp), history_read_fn=adapters[version].make_history_read(
                                session.state, code_count=session.coded_rows,
                                gate_constants=gate_means[version], gate_query=item_gate))[0][0].cpu().tolist()
                    if args.compute_new_controls:
                        cached_batch = all(row["request_id"] in cached_controls for row in batch)
                        if cached_batch:
                            if any(cached_controls[row["request_id"]][key] != row[key] for row in batch for key in IDENTITY):
                                raise ValueError("cached native control request identity differs")
                            reuse_values = [cached_controls[row["request_id"]]["reuse_logit"] for row in batch]
                            rebuild_values = [cached_controls[row["request_id"]]["first_request_rebuild_logit"] for row in batch]
                            counters["native_control_candidate_scores_reused"] += len(batch)
                        else:
                            reuse_values = native_score(models[version], reuse.cache, candidates,
                                                        gap(reuse, timestamp))[0][0].cpu().tolist()
                            rebuild_values = rebuild.score(timestamp, candidates)[0].cpu().tolist()
                            counters["native_control_candidate_scores_computed"] += len(batch)
                    for j, (request, value) in enumerate(zip(batch, actual.cpu().tolist(), strict=True)):
                        if args.compute_new_controls:
                            old = dict(request, reuse_logit=reuse_values[j],
                                       first_request_rebuild_logit=rebuild_values[j])
                        else:
                            old = references[request["request_id"]]
                            if any(old[key] != request[key] for key in IDENTITY):
                                raise ValueError("native reference request identity differs")
                        row = {key: request[key] for key in ("request_id", *IDENTITY)}
                        row.update(reuse_logit=old["reuse_logit"],
                            full_logit=old["first_request_rebuild_logit"], partial_logit=value,
                            first_request_rebuild_logit=old["first_request_rebuild_logit"],
                            old_method_logit=old.get("lightweight_logit"),
                            history_length=session.cache.seq_len, coded_rows=session.coded_rows)
                        if constant is not None:
                            row["mean_gate_logit"] = constant[j]
                        rows.append(row)
            until = releases[index + 1][1] if index + 1 < len(releases) else end
            session.advance(until)
            if args.compute_new_controls:
                reuse.advance(until)
                rebuild.advance(until)
            by_release[version] = dict(source_births_at_publish=source_births,
                source_births_at_first_request=first_births, requests=sum(map(len, by_time.values())),
                cost_counters=dict(session.costs - previous), final_coded_rows=session.coded_rows,
                final_history_rows=session.cache.seq_len)
        counters.update(session.costs)
        combined_native.update(session.trajectory.ledger)
        combined_code_reads.update(session.coded_read_ledger)
        units.append(dict(uid=uid, by_release=by_release, memory=dict(session.memory)))
        if (number + 1) % 16 == 0 or number + 1 == len(selected):
            print(json.dumps(dict(status="quality", users=number + 1, requests=len(rows),
                                  elapsed_seconds=time.perf_counter() - began)), flush=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output / "scores.parquet")
    quality = summarize(table, selected, repetitions=args.bootstrap, releases=versions,
                        preservation_threshold=.8) if rows else None
    if quality is not None:
        quality["estimand"] = "Equal-release mean AUC gain recovery: (Relay-Reuse)/(once-per-release Rebuild-Reuse)."
    constant_quality = dynamic_gain = None
    if rows and args.mean_gate_control:
        constant_rows = [dict(row, partial_logit=row["mean_gate_logit"]) for row in rows]
        constant_quality = summarize(constant_rows, selected, repetitions=args.bootstrap,
                                     releases=versions, preservation_threshold=.8)
        comparison = summarize([dict(row, full_logit=row["mean_gate_logit"]) for row in rows],
                               selected, repetitions=args.bootstrap, releases=versions)
        dynamic_gain = dict(overall=comparison["overall"]["paired"]["partial_minus_full_auc"],
            by_release={version: record["paired"]["partial_minus_full_auc"]
                        for version, record in comparison["by_release"].items()})
    summary = dict(status="complete", selected_uids=selected, weights=weights, quality=quality,
        shared_feature_banks={record["feature_bank"]["sha256"]: record["feature_bank"]
                              for record in weights.values() if "feature_bank" in record},
        mean_gate_quality=constant_quality,
        dynamic_minus_mean_gate_auc=dynamic_gain,
        mean_gate_control_scope="Read-only replacement on the dynamic candidate trajectory; not a separate writer policy",
        units=units, actual_service_counters=dict(counters), gate_statistics=gate_stats.result(),
        native_ledger=records(combined_native, ("action", "version", "before", "width")),
        code_read_ledger=records(combined_code_reads,
            ("kind", "version", "old_length", "queries", "coded_rows", "native_rows")),
        compiled_subset=prefix_records, elapsed_seconds=time.perf_counter() - began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        checks=dict(frozen_user_roles=True, matched_request_identities=True,
                    corrected_writes=True, new_current_rows_zero_code=True),
        cost_scope=settings["cost_scope"], source_sha256=settings["source_sha256"])
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(status="complete", users=len(selected),
                         overall=None if quality is None else quality["overall"]["paired"])), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--workload", choices=("feedback", "behavior"), default="feedback")
    p.add_argument("--stride", type=int, choices=(8, 16, 32), default=16)
    p.add_argument("--candidates", type=int, choices=(2, 16), default=16)
    p.add_argument("--reference", type=Path, default=REFERENCE)
    p.add_argument("--compute-new-controls", action="store_true",
                   help="Required only for a new request panel; original controls are reused")
    p.add_argument("--reuse-new-controls", action="store_true",
                   help="Reuse matched retained native controls for this registered task via --reference")
    p.add_argument("--control-cache", type=Path,
                   help="Reuse scored native-control batches and compute newly added releases/users")
    p.add_argument("--users", type=int, default=128, help="First fixed evaluation UIDs; 0 means all 4096")
    p.add_argument("--start-user", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", "--shard", dest="shard_index", type=int, default=0)
    p.add_argument("--last-version", type=int, choices=(3, 4, 5), default=5)
    p.add_argument("--preparation", choices=("explicit", "compiled"), default="explicit")
    p.add_argument("--anchor-inverse", choices=("transpose", "actual"), default="transpose",
                   help="Invert the saved small P blocks once per publication for compiled anchors")
    p.add_argument("--published-prefixes", type=Path,
                   help="Use full-catalogue prefix artifacts, rather than compiling a development subset")
    p.add_argument("--read-shared", type=int, choices=(0, 3, 6, 12), default=None)
    p.add_argument("--read-private", type=int, choices=(3, 6), default=None)
    p.add_argument("--mean-gate-control", action="store_true")
    p.add_argument("--item-chunk", type=int, default=16384)
    p.add_argument("--query-batch", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

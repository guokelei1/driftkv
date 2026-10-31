#!/usr/bin/env python3
"""Continuous lightweight Value correction on the frozen representative panel.

One candidate native trajectory retains real writes across all three releases.
Existing request-matched Reuse/once-rebuild scores remain immutable controls.
The quality sample never changes the independently retained 30k cost scenario.
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
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from hstu_kvcache.evaluation.release_context import (
    signature, END, RELEASES, ledger_flops, load_models, setup_device,
)
from hstu_kvcache.evaluation.panels import checked, load_history, load_panel, record
from experiments.archive.compact_read.lightweight_budget import candidate_release, scale_projection
from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueAdapter, LightweightValueState
from hstu_kvcache.methods.cachetide.lifecycle.lifecycle import LifecycleSession
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.evaluation.cost import CostModel, eager_read
from hstu_kvcache.methods.baselines.protocol.evaluate import native_backend
from hstu_kvcache.common.artifacts import sha256, write_json

PHASE = ROOT / "results/design_system_2026_10/phase2"
REFERENCE = PHASE / "representative4096/development4096"
PANEL = PHASE / "representative4096/panel"
IDENTITY_COLUMNS = ("request_id", "uid", "query_timestamp", "item_idx", "label", "edge", "version")
CONTROL_COLUMNS = ("reuse_logit", "first_request_rebuild_logit", "exact_logit", "fresh_logit")


def execution_sources():
    names = ("experiments/archive/compact_read/evaluate_lightweight.py", "experiments/archive/compact_read/lightweight_budget.py",
        "experiments/archive/compact_read/once_rebuild_accounting.py", "src/hstu_kvcache/evaluation/release_context.py",
        "src/hstu_kvcache/evaluation/panels.py", "src/hstu_kvcache/methods/cachetide/compact/lightweight.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_kernel.py", "src/hstu_kvcache/adaptation/reader.py",
        "src/hstu_kvcache/evaluation/trajectory.py", "src/hstu_kvcache/methods/cachetide/lifecycle/lifecycle.py",
        "src/hstu_kvcache/models/hstu.py", "src/hstu_kvcache/models/attention.py",
        "src/hstu_kvcache/models/state_transition.py", "src/hstu_kvcache/evaluation/binary_metrics.py")
    return {name: sha256(ROOT / name) for name in names}


class LightweightSession(LifecycleSession):
    """Reuse the existing lifecycle; only code layout and read operation change."""

    def __init__(self, trajectory, *, kernel="vector", exit_when_clear=True):
        super().__init__(trajectory, exit_when_clear=exit_when_clear)
        self.kernel = kernel

    @torch.no_grad()
    def _appended(self, old, new, events, version):
        width = len(events[0])
        self.support.append(width)
        self.costs["support_token_layers"] += width * len(self.trajectory.model.blocks)
        if self.state is not None and self.exit_when_clear and self.support.clear:
            self._retire("inherited_support_cleared", int(events[0][-1]))
        elif self.state is not None:
            entering = HSTUKVCache(new.k[:, :, -width:], new.v[:, :, -width:], width)
            encoded = self._encode(entering, events[1], "append_encode_flops")
            codes = torch.cat((self.state.codes, encoded.codes), dim=2)[:, :, -new.seq_len:].contiguous()
            self.state = LightweightValueState(codes, new.seq_len)
        self._record_memory()

    def read_callback(self, candidates, *, force_torch=False):
        if self.state is None:
            return None
        if (not force_torch and self.kernel == "vector" and self.cache.k.is_cuda
                and self.cache.k.dtype == torch.float32 and self.adapter.head_dim == 32
                and self.adapter.rank in (2, 4, 8) and 0 < self.cache.seq_len <= 1024
                and 0 < candidates.shape[1] <= 16):
            from hstu_kvcache.methods.cachetide.compact.lightweight_kernel import make_history_read
            return make_history_read(self.adapter, self.state, backend="vector")
        return self.adapter.make_history_read(self.state)

    @torch.no_grad()
    def score(self, at, candidates):
        self.prepare_read(at)
        delta = 0 if self.trajectory.last_timestamp is None else int(at) - self.trajectory.last_timestamp
        query_delta = torch.tensor([delta], dtype=torch.float32, device=self.cache.k.device)
        logits = native_score(self.trajectory.model, self.cache, candidates, query_delta,
                              history_read_fn=self.read_callback(candidates))[0]
        if self.state is not None:
            work = self.adapter.estimate_flops(tokens=self.cache.seq_len, candidates=candidates.shape[1])
            self.costs["extra_history_read_flops"] += work["extra_history_read"]
            self.costs["response_flops"] += work["candidate_reads"]
            self.costs["corrected_requests"] += candidates.numel()
        else:
            self.costs["native_requests"] += candidates.numel()
        return logits


def difference(after, before):
    return {key: int(value - before.get(key, 0)) for key, value in after.items()
            if value != before.get(key, 0)}


def load_calibrations(folder, groups, device):
    adapters, bindings = {}, {}
    for version, _ in RELEASES:
        directory = folder / version
        metadata_path, weights_path = directory / "calibration.json", directory / "calibration.pt"
        metadata = json.loads(metadata_path.read_text())
        saved = torch.load(weights_path, map_location="cpu", weights_only=False)
        if saved["kind"] != LightweightValueAdapter.kind:
            raise RuntimeError("checkpoint is not the lightweight candidate")
        if metadata.get("target", version) != version:
            raise RuntimeError("calibration target differs")
        if metadata.get("weights_sha256", sha256(weights_path)) != sha256(weights_path):
            raise RuntimeError("calibration checkpoint changed")
        fit, validation = set(metadata["fit_uids"]), set(metadata["validation_uids"])
        if fit & validation or (fit | validation) & set(groups["evaluation"]):
            raise RuntimeError("lightweight calibration overlaps evaluation")
        adapter = LightweightValueAdapter(**saved["config"])
        adapter.load_state_dict(saved["state_dict"], strict=True)
        adapters[version] = adapter.to(device).eval().requires_grad_(False)
        bindings[version] = dict(metadata=record(metadata_path), weights=record(weights_path),
            config=saved["config"], fit_uids=metadata["fit_uids"], validation_uids=metadata["validation_uids"],
            cost=metadata["cost"],
            source_trajectory_flops=metadata["cost"]["development_resource"]["source_trajectory_flops"],
            source_reconstruction_sensitivity=metadata["cost"]["source_reconstruction_sensitivity"])
    return adapters, bindings


@torch.no_grad()
def score_user(uid, events, requests, models, adapters, reference, *, query_batch=16,
               verify_controls=False, verify_state=False, kernel="vector"):
    initial = models["v2"]
    corrected = LightweightSession(NativeTrajectory(initial, 2, events, RELEASES[0][1]), kernel=kernel)
    lazy = (LifecycleSession(NativeTrajectory(initial, 2, events, RELEASES[0][1]))
            if verify_controls else None)
    by_version = defaultdict(list)
    for row in requests:
        by_version[row["version"]].append(row)
    rows, releases, checks = [], {}, Counter()
    for index, (version, at) in enumerate(RELEASES):
        model, adapter = models[version], adapters[version]
        until = RELEASES[index + 1][1] if index + 1 < len(RELEASES) else END
        before = Counter(corrected.costs)
        corrected.publish(model, int(version[1:]), at, adapter)
        if lazy is not None:
            lazy.publish(model, int(version[1:]), at, adapter, policy="rebuild_on_first_request")
        snapshots = defaultdict(list)
        query_base = 0
        for row in by_version[version]:
            if not at <= row["query_timestamp"] < until:
                raise RuntimeError("request outside its fixed release")
            snapshots[int(row["query_timestamp"])].append(row)
        for timestamp, pending in sorted(snapshots.items()):
            corrected.prepare_read(timestamp)
            if lazy is not None:
                lazy.prepare_read(timestamp)
                if corrected.trajectory.cursor != lazy.trajectory.cursor:
                    raise RuntimeError("native and rebuild controls disagree on event prefix")
            n = corrected.cache.seq_len
            backend = "triton" if n >= model.cfg.max_seq_len else "torch"
            cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, backend)
            if verify_state and corrected.state is not None:
                ids = torch.as_tensor(corrected.trajectory.retained_events[1], device=corrected.cache.k.device)[None]
                full_state = adapter.encode_cache(corrected.cache, model.lookup_item_embeddings(ids))
                error = float((full_state.codes - corrected.state.codes).abs().max())
                checks["incremental_code_max_error"] = max(checks["incremental_code_max_error"], error)
                if not torch.allclose(full_state.codes, corrected.state.codes, atol=2e-6, rtol=2e-5):
                    raise RuntimeError("append/eviction codes differ from complete re-encoding")
            for offset in range(0, len(pending), query_batch):
                batch = pending[offset:offset + query_batch]
                ids = torch.tensor([[row["item_idx"] for row in batch]], device=corrected.cache.k.device)
                delta = torch.tensor([0 if corrected.trajectory.last_timestamp is None else
                    timestamp - corrected.trajectory.last_timestamp], device=ids.device, dtype=torch.float32)
                with native_backend([model], backend):
                    actual = corrected.score(timestamp, ids)[0]
                    if verify_controls:
                        reuse = native_score(model, corrected.cache, ids, delta)[0][0]
                        once = lazy.score(timestamp, ids)[0]
                    if verify_state:
                        explicit = native_score(model, corrected.cache, ids, delta,
                            history_read_fn=corrected.read_callback(ids, force_torch=True))[0][0]
                        error = float((actual-explicit).abs().max())
                        checks["kernel_logit_max_error"] = max(checks["kernel_logit_max_error"], error)
                        if not torch.allclose(actual, explicit, atol=2e-5, rtol=2e-5):
                            raise RuntimeError("fused read differs from Torch reference")
                if not bool(torch.isfinite(actual).all()):
                    raise RuntimeError("nonfinite candidate scores")
                values = actual.cpu().tolist()
                native_values = reuse.cpu().tolist() if verify_controls else None
                once_values = once.cpu().tolist() if verify_controls else None
                query_base += eager_read(cm, n, queries=len(batch))
                for j, request in enumerate(batch):
                    old = reference[request["request_id"]]
                    if any(old[key] != request[key] for key in IDENTITY_COLUMNS):
                        raise RuntimeError("frozen reference request identity differs")
                    if verify_controls:
                        for name, value in (("reuse", native_values[j]), ("first_request_rebuild", once_values[j])):
                            key = name + "_max_abs_error"
                            checks[key] = max(checks[key], abs(value-old[name + "_logit"]))
                            if checks[key] > 2e-5:
                                raise RuntimeError(f"{name} no longer matches frozen controls: {checks[key]}")
                    rows.append({**{key: old[key] for key in (*IDENTITY_COLUMNS, *CONTROL_COLUMNS)},
                        "lightweight_logit": values[j], "history_length": n,
                        "causal_native": corrected.native_mode})
        corrected.advance(until)
        if lazy is not None:
            lazy.advance(until)
        releases[version] = dict(requests=len(by_version[version]),
            correction_costs=difference(corrected.costs, before), native_query_flops=query_base,
            native_kv_and_extra_memory=dict(corrected.memory))
    if len(rows) != len(requests):
        raise RuntimeError("missing requests in completed user trajectory")
    return rows, dict(uid=uid, requests=len(rows), releases=releases, control_checks=dict(checks),
        native_flops=ledger_flops(corrected.trajectory.ledger, initial, "triton"))


def run(args):
    started = time.perf_counter()
    panel, groups, requests = load_panel(args.panel)
    population = groups["evaluation"][:args.max_users] if args.max_users else groups["evaluation"]
    selected = population[args.shard_index::args.num_shards]
    if not selected or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("empty or invalid shard")
    reference_path = args.reference / "summary.json"
    reference = json.loads(reference_path.read_text())
    if reference["inputs"]["panel_sha256"] != sha256(args.panel / "binding.json"):
        raise RuntimeError("reference belongs to another panel")
    scores_path = checked(reference["scores"])
    adapters, calibration = load_calibrations(args.calibration_dir, groups, torch.device("cpu"))
    common = dict(panel=record(args.panel / "binding.json"), calibration=calibration,
        reference=record(reference_path), reference_scores=record(scores_path), population=population,
        query_batch=args.query_batch, kernel=args.kernel, num_shards=args.num_shards,
        verify_controls=args.verify_controls, verify_state=args.verify_state,
        population_cost=record(args.population_cost), execution_sources=execution_sources(),
        quality_scope="Fixed representative development users; includes idle users and all release windows.",
        accounting_scope="Shared calibration independent of quality sample; population cost is fixed real30k event shapes.")
    inputs = {**common, "shard_index": args.shard_index, "uids": selected}
    input_signature = signature(inputs)
    directory = args.output / f"shard_{args.shard_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "inputs.json"
    if target.exists() and json.loads(target.read_text())["signature"] != input_signature:
        raise RuntimeError("output shard belongs to different inputs")
    write_json(target, dict(signature=input_signature, inputs=inputs))
    refs = pq.read_table(scores_path)
    refs = refs.filter(pc.is_in(refs["uid"], value_set=pa.array(selected, type=refs["uid"].type)))
    refs = {row["request_id"]: row for row in refs.to_pylist()}
    histories = load_history(args.panel, requested_uids=selected)
    device = setup_device(args.gpu, 4)
    models = load_models(panel, device)
    for adapter in adapters.values():
        adapter.to(device)
    unit_paths = []
    for uid in selected:
        filename = directory / "units" / f"uid_{uid}.json"
        if filename.exists():
            saved = json.loads(filename.read_text())
            if saved["signature"] != input_signature:
                raise RuntimeError("resume unit belongs to different inputs")
            checked(saved["scores"])
            unit_paths.append(str(filename.resolve()))
            continue
        began = time.perf_counter()
        rows, unit = score_user(uid, histories.rows[uid], requests[uid], models, adapters, refs,
            query_batch=args.query_batch, verify_controls=args.verify_controls,
            verify_state=args.verify_state, kernel=args.kernel)
        torch.cuda.synchronize(device)
        filename.parent.mkdir(parents=True, exist_ok=True)
        scores_file = filename.with_suffix(".parquet")
        table = pa.Table.from_pylist(rows) if rows else pa.table({"request_id": pa.array([], type=pa.string())})
        pq.write_table(table, scores_file, compression="zstd")
        unit.update(signature=input_signature, scores=record(scores_file),
            elapsed_seconds=time.perf_counter()-began,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
        write_json(filename, unit)
        unit_paths.append(str(filename.resolve()))
        print(json.dumps(dict(uid=uid, completed=len(unit_paths), users=len(selected), requests=len(rows),
            seconds=unit["elapsed_seconds"], checks=unit["control_checks"])), flush=True)
    if execution_sources() != common["execution_sources"]:
        raise RuntimeError("numerical execution sources changed while scoring")
    write_json(directory / "shard.json", dict(status="complete", signature=input_signature,
        common_signature=signature(common), inputs=inputs, units=unit_paths,
        elapsed_seconds=time.perf_counter()-started))


def quality_summary(table):
    result, differences, gaps = {}, [], []
    for version, _ in RELEASES:
        edge = table.filter(pc.equal(table["version"], version))
        if not edge.num_rows:
            result[version] = dict(requests=0, metrics=None, once_gap_recovery_percent=None)
            continue
        labels = edge["label"].to_numpy()
        metrics = {name: binary_metrics(labels, edge[name + "_logit"].to_numpy())
                   for name in ("reuse", "first_request_rebuild", "lightweight", "exact", "fresh")}
        auc = {name: value["ROC_AUC"] for name, value in metrics.items()}
        gap = (auc["first_request_rebuild"]-auc["reuse"]
               if auc["first_request_rebuild"] is not None and auc["reuse"] is not None else None)
        gain = (auc["lightweight"]-auc["reuse"]
                if auc["lightweight"] is not None and auc["reuse"] is not None else None)
        if gap is not None and gain is not None:
            gaps.append(gap)
            differences.append(gain)
        result[version] = dict(requests=edge.num_rows, metrics=metrics, once_auc_gap=gap,
            candidate_auc_gain=gain, once_gap_recovery_percent=
                100*gain/gap if gap is not None and gap > 1e-12 and gain is not None else None)
    valid = len(gaps) == len(RELEASES) and sum(gaps) > 1e-12
    return dict(by_release=result, primary_recovery_percent=100*sum(differences)/sum(gaps) if valid else None,
        aggregation="Mean of per-release candidate-minus-Reuse AUC differences divided by mean once-rebuild-minus-Reuse differences. All releases retained; not mean recovery percentages.")


def complete_cost(common):
    costs = json.loads(checked(common["population_cost"]).read_text())
    releases, shared = {}, 0
    configs = [entry["config"] for entry in common["calibration"].values()]
    if any(config != configs[0] for config in configs):
        raise RuntimeError("this comparison requires one candidate architecture across releases")
    cfg = configs[0]
    for version, binding in common["calibration"].items():
        releases[version] = candidate_release(costs["by_release"][version]["counts_and_flops"],
            layers=cfg["num_layers"], heads=cfg["heads"], head_dim=cfg["head_dim"],
            rank=cfg["rank"], response_width=cfg["response_hidden_width"])
        releases[version]["calibration"] = binding["cost"]
        shared += binding["cost"]["calibration_flops"]
    variable = sum(row["variable_flops"] for row in releases.values())
    full = sum(row["first_request_rebuild_flops"] for row in releases.values())
    query = sum(row["extra_query_flops"] for row in releases.values())
    native_query = sum(row["native_whole_query_flops"] for row in releases.values())
    return dict(role="Analytic real30k event-shape ledger; no all-population quality or timing claim.",
        population_users=costs["users"], requests=costs["requests"], by_release=releases,
        denominator="First-request-triggered complete history rebuild once per user per release, plus native service, minus matched Reuse native service.",
        first_request_rebuild_flops=full, calibration_flops=shared, population_dependent_flops=variable,
        complete_added_flops=shared+variable, complete_added_percent=100*(shared+variable)/full,
        extra_query_percent_of_native_whole_query=100*query/native_query,
        code_storage_percent_of_native_kv=100*cfg["rank"]/(2*cfg["head_dim"]),
        causal_flags_percent_of_native_kv=100/(2*cfg["heads"]*cfg["head_dim"]*4),
        state_accounting="Conservative: encode every resident history at each release and every native append. No savings from causal retirement assumed in population ledger.",
        source_trajectory_resources={v: b["source_trajectory_flops"] for v,b in common["calibration"].items()},
        source_reconstruction_sensitivity={v: b["source_reconstruction_sensitivity"] for v,b in common["calibration"].items()},
        population_projections={str(n): scale_projection(variable=variable, full=full,
            users=costs["users"], target_users=n, shared_flops=shared) for n in (costs["users"], 100000, 1000000)},
        projection_scope="Same model, three-release horizon and per-resident real-event mix. Fixed calibration population. Extrapolated FLOPs only, not measured quality or service.")


def aggregate(args):
    shards = [json.loads(filename.read_text()) for filename in sorted(args.output.glob("shard_*/shard.json"))]
    if (not shards or len(shards) != shards[0]["inputs"]["num_shards"]
            or len({row["common_signature"] for row in shards}) != 1
            or {row["inputs"]["shard_index"] for row in shards} != set(range(len(shards)))):
        raise RuntimeError("all matching shards must be complete")
    common = {key: value for key,value in shards[0]["inputs"].items() if key not in ("shard_index", "uids")}
    units, tables = [], []
    for shard in shards:
        for filename in shard["units"]:
            unit = json.loads(Path(filename).read_text())
            if unit["signature"] != shard["signature"]:
                raise RuntimeError("unit signature differs from shard")
            table = pq.read_table(checked(unit["scores"]))
            units.append(unit)
            if table.num_rows:
                tables.append(table)
    if len({unit["uid"] for unit in units}) != len(units) or {u["uid"] for u in units} != set(common["population"]):
        raise RuntimeError("completed users do not match frozen population")
    if not tables:
        raise RuntimeError("the sample has no requests")
    table = pa.concat_tables(tables).sort_by([("query_timestamp", "ascending"), ("request_id", "ascending")])
    if len(set(table["request_id"].to_pylist())) != table.num_rows:
        raise RuntimeError("duplicated request scores")
    path = args.output / "scores.parquet"
    pq.write_table(table, path, compression="zstd")
    check_names = {key for unit in units for key in unit["control_checks"]}
    observed = {version: dict(sum_costs) for version, sum_costs in
        ((version, sum((Counter(unit["releases"][version]["correction_costs"]) for unit in units), Counter()))
         for version,_ in RELEASES)}
    result = dict(status="complete", role="lightweight_continuous_development", inputs=common,
        users=len(units), requests=table.num_rows, scores=record(path),
        units=[filename for shard in shards for filename in shard["units"]],
        control_checks={key:max(unit["control_checks"].get(key,0) for unit in units) for key in check_names},
        quality=quality_summary(table), cost=complete_cost(common), development_execution_costs=observed,
        worker_elapsed_seconds=[row["elapsed_seconds"] for row in shards],
        peak_gpu_allocated_gib=max(unit["peak_gpu_allocated_gib"] for unit in units),
        notes=["Scoring accepts no labels; labels and frozen reference scores are attached after inference.",
            "Native K/V writes are unchanged and the trajectory is not reset across releases.",
            "The cost denominator and fixed calibration do not shrink when --max-users is used.",
            "Small-sample recovery is a development signal, not acceptance or final generalization."])
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(users=result["users"], requests=result["requests"],
        recovery_percent=result["quality"]["primary_recovery_percent"],
        complete_cost_percent=result["cost"]["complete_added_percent"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--population-cost", type=Path, default=PHASE / "population_cost/summary_complete.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--kernel", choices=("vector", "torch"), default="vector")
    parser.add_argument("--verify-controls", action="store_true")
    parser.add_argument("--verify-state", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    (aggregate if args.aggregate else run)(args)

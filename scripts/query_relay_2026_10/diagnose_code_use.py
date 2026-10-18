#!/usr/bin/env python3
"""V3 code-payload controls on the unchanged corrected writer trajectory.

Zero removes the auxiliary read. Global-mean retains the dynamic reader but
replaces inherited row payloads by a single fit-only mean per layer. Native
attention weights, history lengths and code-prefix mass still vary: this is a
history-independent payload control, not an optimal query-only model.
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

from design_one.compare import FixedOrderAUC
from design_two.common import RELEASES, ledger_flops, load_models, setup_device
from design_two.prepare import load_history, load_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.design_two.replay import NativeTrajectory
from partial_refresh_2026_10.analyze import interval
from query_relay_2026_10.adapter import operation_gate_query
from query_relay_2026_10.calibration_cost import OperationCounts
from query_relay_2026_10.evaluate import IDENTITY, gap, load_candidates
from query_relay_2026_10.session import RelaySession
from query_relay_2026_10.workload import candidate_rows
from unified_reuse_2026_09.common import sha256, write_json


def code_payload_state(state, mean=None, code_count=None):
    """New transient buffer; never replace or mutate the live session state."""
    values = torch.zeros_like(state.codes)
    count = state.seq_len if code_count is None else code_count
    if mean is not None and count:
        expected = (state.codes.shape[0], state.codes.shape[-2] * state.codes.shape[-1])
        if mean.shape != expected:
            raise ValueError("global code mean must be [layers,heads*rank]")
        values[:, :, :count] = mean.to(values).reshape(
            values.shape[0], 1, 1, values.shape[-2], values.shape[-1])
    return LightweightValueState(values, state.seq_len)


@torch.no_grad()
def prepare_mean(adapter, models, metadata, fit_panel, device):
    """Native V2 at the V3 cutoff only: no teacher, labels, future or eval rows."""
    binding, roles, _ = load_panel(fit_panel)
    uids = metadata["fit_uids"]
    if len(uids) != 64 or not set(uids) <= set(roles["fit"]):
        raise ValueError("this diagnostic uses the actual retained64 fit users")
    if sha256(fit_panel / "binding.json") != metadata["panel_sha256"]:
        raise ValueError("fit-only code mean must use the fitted panel binding")
    histories = load_history(fit_panel, requested_uids=uids)
    cut = dict(RELEASES)["v3"]
    counts, native_shapes, records = OperationCounts(), Counter(), []
    total, sums, squares = 0, None, None
    for index, uid in enumerate(uids):
        trajectory = NativeTrajectory(models["v2"], 2, histories.rows[uid], cut)
        cache = trajectory.cache
        if not cache.seq_len or trajectory.last_timestamp >= cut:
            raise ValueError("fit mean requires nonempty, strictly pre-V3 native V2 history")
        items = torch.as_tensor(trajectory.retained_events[1], device=device)[None]
        features = models["v3"].lookup_item_embeddings(items)
        state = adapter.encode_initial(cache, features, item_ids=items)
        values = state.codes.double().flatten(-2)
        row_sum, row_square = values.sum((1, 2)), values.square().sum((1, 2))
        sums = row_sum if sums is None else sums + row_sum
        squares = row_square if squares is None else squares + row_square
        total += cache.seq_len
        counts.record_update(adapter, length=cache.seq_len, stage="fit_mean_encoder")
        counts.calls["fit_mean_current_item_lookup_bytes"] += features.numel() * features.element_size()
        if adapter.feature_mode == "origin_encoder":
            counts.calls["fit_mean_fixed_origin_item_bank_lookup_bytes"] += state.storage_bytes()
        counts.add("fit_mean_sum_square_reduction_flops", 3 * values.numel())
        counts.add("fit_mean_native_V2_capture_flops", sum(ledger_flops(trajectory.ledger, models["v2"], "triton").values()))
        native_shapes.update(trajectory.ledger)
        records.append(dict(uid=uid, rows=cache.seq_len, last_timestamp=trajectory.last_timestamp))
        if (index + 1) % 16 == 0:
            print(json.dumps(dict(status="fit_only_mean", users=index + 1, rows=total)), flush=True)
    mean = sums / total
    variance = (squares / total - mean.square()).clamp_min(0)
    counts.add("fit_mean_finalize_flops", 4 * mean.numel())
    return mean.to(device=device, dtype=next(adapter.parameters()).dtype), dict(
        fit_uids=uids, total_rows=total, per_user=records,
        row_weighted_mean=mean.cpu().tolist(), coordinate_variance=variance.cpu().tolist(),
        panel=str(fit_panel), panel_sha256=sha256(fit_panel / "binding.json"),
        preparation=counts.report(), native_capture_geometry=[dict(action=k[0], version=k[1],
            before_length=k[2], width=k[3], count=v) for k, v in sorted(native_shapes.items())],
        scope="Diagnostic-only preparation; actual fitted64 strict pre-V3 native V2 rows with current V3 item features; no teacher, request labels or evaluation rows; row-weighted per-layer48 mean")


def analyze(table, selected, repetitions=1000, seed=17):
    """One common UID bootstrap for all methods, AUC and offset-free metrics."""
    users = np.sort(np.asarray(selected, dtype=np.int64))
    ordered = table.sort_by([("round_id", "ascending")])
    methods = ("reuse", "rebuild", "dynamic", "zero", "global_mean")
    labels, uids = ordered["label"].to_numpy(), ordered["uid"].to_numpy()
    indices = np.searchsorted(users, uids)
    logits = {method: ordered[method + "_logit"].to_numpy().astype(np.float64) for method in methods}
    readers = [FixedOrderAUC(labels, logits[method], indices) for method in methods]
    set_labels = labels.reshape(-1, 16)
    if not np.all(set_labels.sum(1) == 1):
        raise ValueError("expected one positive per unchanged16-candidate set")
    set_users = indices.reshape(-1, 16)[:, 0]
    wanted = logits["rebuild"].reshape(-1, 16)
    totals = np.zeros((len(users), len(methods), 5), dtype=np.float64)
    for m, method in enumerate(methods):
        actual = logits[method].reshape(-1, 16)
        residual = actual - wanted
        raw, common = (residual**2).mean(1), residual.mean(1)**2
        centered = ((residual - residual.mean(1, keepdims=True))**2).mean(1)
        positive = actual[set_labels == 1][:, None]
        negative = set_labels == 0
        pair_auc = (((actual < positive) & negative).sum(1)
                    + .5 * ((actual == positive) & negative).sum(1)) / 15
        np.add.at(totals[:, m], set_users, np.stack([raw, centered, common, pair_auc, np.ones(len(wanted))], 1))

    def estimate(weights):
        sums = np.einsum("u,umk->mk", weights, totals)
        rates = sums[:, :4] / sums[:, 4:5]
        return np.asarray([reader(weights) for reader in readers]), rates

    point_auc, point_metrics = estimate(np.ones(len(users), dtype=np.int64))
    rng = np.random.default_rng(seed)
    draws = [estimate(np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users)))
             for _ in range(repetitions)]
    auc_draws, metric_draws = np.stack([x[0] for x in draws]), np.stack([x[1] for x in draws])
    metric_names = ("rebuild_raw_logit_mse", "rebuild_centered_logit_mse", "rebuild_common_offset_mse", "within_set_pair_auc")
    estimates = {method: dict(auc=interval(auc_draws[:, m], point_auc[m]),
        **{name: interval(metric_draws[:, m, k], point_metrics[m, k]) for k, name in enumerate(metric_names)})
        for m, method in enumerate(methods)}
    comparisons = {}
    dynamic = methods.index("dynamic")
    for control in ("zero", "global_mean"):
        c = methods.index(control)
        comparisons[control] = dict(auc=interval(auc_draws[:, dynamic] - auc_draws[:, c], point_auc[dynamic] - point_auc[c]),
            **{name: interval(metric_draws[:, dynamic, k] - metric_draws[:, c, k], point_metrics[dynamic, k] - point_metrics[c, k])
               for k, name in enumerate(metric_names)})
    return dict(methods=estimates, dynamic_minus_control=comparisons,
        sets=len(wanted), candidate_requests=table.num_rows,
        active_uids=len(np.unique(uids)), idle_uids=len(users) - len(np.unique(uids)),
        bootstrap=dict(seed=seed, repetitions=repetitions,
            resampling="One shared sorted-UID multiset for all methods/metrics, including idle users; equal16-candidate sets for score errors/pair-AUC; pooled candidate AUC"),
        signs="Positive AUC/pair-AUC differences favor dynamic; negative MSE differences favor dynamic",
        reference="Retained once-per-release native Rebuild, not per-request Fresh teacher")


@torch.no_grad()
def run(args):
    began = time.perf_counter()
    original = json.loads((args.reference.parent / "configuration.json").read_text())
    if original["versions"] != ["v3"] or original["workload"]["kind"] != "behavior" or original["workload"]["candidates"] != 16 or original["code_preparation"] != "explicit":
        raise ValueError("use the completed explicit V3 behavior16 reference")
    selected = original["selected_uids"]
    if len(selected) != 128:
        raise ValueError("this small diagnostic retains the original128 users, including idle users")
    panel = Path(original["panel"])
    binding, roles, _ = load_panel(panel)
    if sha256(panel / "binding.json") != original["panel_sha256"]:
        raise ValueError("retained panel changed")
    device = setup_device(args.gpu, args.threads)
    models = load_models(binding, device, 3)
    adapters, weights, _ = load_candidates(args.calibrations, device, ["v3"], selected)
    adapter = adapters["v3"]
    retained = json.loads((args.reference.parent / "summary.json").read_text())
    if weights["v3"]["sha256"] != retained["weights"]["v3"]["sha256"]:
        raise ValueError("different fitted candidate weights")
    metadata = json.loads((args.calibrations / "v3/calibration.json").read_text())
    if adapter.gate_source != "item" or set(metadata["fit_uids"]) & set(selected):
        raise ValueError("use the retained item-gate candidate and disjoint roles")
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = dict(selected_uids=selected, versions=["v3"], workload=original["workload"],
        panel=str(panel), panel_sha256=original["panel_sha256"], reference=str(args.reference),
        reference_sha256=sha256(args.reference), weights=weights, code_preparation="explicit",
        writer="Unchanged corrected dynamic item-gate trajectory; shadows never write",
        controls=dict(zero="Zero auxiliary payload; native-only reads on the same dynamic-writer cache",
            global_mean="Actual fit-only per-layer48 mean on every coded inherited prefix row; current-born suffix zero; native attention/mass preserved"),
        training=False, labels="Output analysis only; no labels in any read or mean preparation",
        source_sha256={name: sha256(Path(__file__).parent / name) for name in
            (Path(__file__).name, "adapter.py", "session.py", "calibration_cost.py")})
    write_json(args.output / "configuration.json", configuration)
    mean, mean_record = prepare_mean(adapter, models, metadata, args.fit_panel, device)
    torch.save(dict(mean=mean.cpu(), record=mean_record), args.output / "global_mean_code.pt")
    histories = load_history(panel, requested_uids=selected)
    references = {row["request_id"]: row for row in pq.read_table(args.reference).to_pylist()}
    rows, units, counts, maximum_error = [], [], OperationCounts(), 0.
    cut, end = RELEASES[0][1], RELEASES[1][1]
    for number, uid in enumerate(selected):
        session = RelaySession(NativeTrajectory(models["v2"], 2, histories.rows[uid], cut))
        session.publish(models["v3"], 3, cut, adapter)
        sets = defaultdict(list)
        for request in candidate_rows(uid, histories.rows[uid], stride=original["workload"]["stride"],
                candidates=16, known_vocab_size=binding["known_vocab_size"], releases=(("v3", cut),), end=end):
            sets[request["query_timestamp"], request["round_id"]].append(request)
        for (timestamp, _), pending in sorted(sets.items()):
            candidates = torch.tensor([[row["item_idx"] for row in pending]], device=device)
            dynamic = session.score(timestamp, candidates)[0]
            gate_query = operation_gate_query(adapter, models["v3"], candidates)
            counts.record_gate_item_input(adapter, items=candidates.numel(), stage="shared_shadow_inputs")
            controls = {}
            for name, payload, code_count in (("zero", None, 0), ("global_mean", mean, session.coded_rows)):
                state = code_payload_state(session.state, payload, code_count)
                controls[name] = score(models["v3"], session.cache, candidates, gap(session.trajectory, timestamp),
                    history_read_fn=adapter.make_history_read(state, code_count=code_count, gate_query=gate_query))[0][0]
                counts.record_query(adapter, length=session.cache.seq_len, queries=16,
                                    coded_rows=code_count, stage=name + "_shadow")
                counts.calls[name + "_transient_code_buffer_write_bytes"] += state.storage_bytes()
            counts.record_query(adapter, length=session.cache.seq_len, queries=16,
                                coded_rows=session.coded_rows, stage="dynamic_replay_query")
            for index, request in enumerate(pending):
                prior = references[request["request_id"]]
                if any(prior[key] != request[key] for key in IDENTITY):
                    raise ValueError("retained candidate identity differs")
                error = abs(float(dynamic[index]) - prior["partial_logit"])
                maximum_error = max(maximum_error, error)
                if error > 2e-4:
                    raise ValueError("dynamic writer trajectory differs from retained candidate")
                rows.append(dict(request, reuse_logit=prior["reuse_logit"],
                    rebuild_logit=prior["first_request_rebuild_logit"], dynamic_logit=float(dynamic[index]),
                    zero_logit=float(controls["zero"][index]), global_mean_logit=float(controls["global_mean"][index]),
                    retained_dynamic_logit=prior["partial_logit"], coded_rows=session.coded_rows))
        session.advance(end)
        for event in session.events:
            if event["action"] == "explicit_update":
                counts.record_update(adapter, length=event["rows"], stage="dynamic_replay_encoder")
                counts.calls["dynamic_replay_encoder_current_item_lookup_bytes"] += \
                    event["rows"] * adapter.heads * adapter.head_dim * next(adapter.parameters()).element_size()
                if adapter.feature_mode == "origin_encoder":
                    counts.calls["dynamic_replay_encoder_fixed_origin_item_bank_lookup_bytes"] += \
                        event["rows"] * adapter.num_layers * adapter.heads * adapter.rank * next(adapter.parameters()).element_size()
        for (kind, _, old_length, width, coded_rows, _), count in session.coded_read_ledger.items():
            if kind == "writer":
                for _ in range(count):
                    counts.record_event_write(adapter, old_length=old_length, width=width, coded_rows=coded_rows,
                        stage="dynamic_replay_writer", backend="triton" if old_length >= adapter.max_length else "torch")
        counts.add("dynamic_replay_native_V2_capture_flops", sum(ledger_flops(
            Counter({key: value for key, value in session.trajectory.ledger.items() if key[0] == "init"}),
            models["v2"], "triton").values()))
        units.append(dict(uid=uid, cost_counters=dict(session.costs), events=session.events,
            corrected_read_geometry=[dict(kind=k[0], version=k[1], native_prefix=k[2], queries=k[3],
                coded_prefix=k[4], native_rows=k[5], count=v) for k, v in session.coded_read_ledger.items()]))
        if (number + 1) % 16 == 0:
            print(json.dumps(dict(status="code_payload_diagnostic", users=number + 1, candidates=len(rows),
                                  elapsed_seconds=time.perf_counter() - began)), flush=True)
    if len(rows) != len(references):
        raise ValueError("diagnostic did not retain every original candidate request")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output / "scores.parquet")
    result = dict(status="complete", configuration=configuration, mean_preparation=mean_record,
        metrics=analyze(table, selected, args.bootstrap), units=units,
        diagnostic_replay_cost=counts.report(), checks=dict(no_shadow_writes=True,
            native_controls_reused=True, matched_request_identities=True,
            dynamic_replay_max_absolute_logit_error=maximum_error),
        mean_artifact_sha256=sha256(args.output / "global_mean_code.pt"),
        elapsed_seconds=time.perf_counter() - began,
        cost_scope="Fit-mean preparation and diagnostic replay are separately retained development execution; these control costs are not claimed as serving costs",
        limitation="Global mean removes row-specific compact payloads, while preserving native/history-dependent attention, code-prefix mass, current query gates and the original corrected writer; no optimal query-only baseline claim")
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status="complete", comparisons=result["metrics"]["dynamic_minus_control"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrations", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fit-panel", type=Path, default=ROOT / "results/design_two_2026_10/benchmark_panel")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=1000)
    run(p.parse_args())

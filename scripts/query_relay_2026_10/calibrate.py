#!/usr/bin/env python3
"""Small, pre-release read/score fitting for the query-relay prototypes.

First-release probes use the retained causal V2 source. Later releases require
source packs made by this method, rather than the old native-only trajectory.
No serving labels are inputs to fitting. Backbone parameters remain frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import torch

from code_structure_2026_10.structure import StructuredKVAdapter
from design_two.capture import capture_rows
from design_two.common import RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from query_relay_2026_10.adapter import UnifiedRelayAdapter, operation_gate_query
from query_relay_2026_10.calibration_cost import OperationCounts, update_forward_flops, warmstart_origin_cost
from query_relay_2026_10.session import append_corrected_event
from unified_reuse_2026_09.common import sha256, write_json

PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"
ORIGIN = ROOT / "results/code_structure_2026_10/shared12_private6/calibration/v3/calibration.pt"


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def checkpoint(adapter, path, settings, diagnostics, gate_constants, cost):
    payload = dict(kind=adapter.kind, config=adapter.get_config(),
                   state_dict={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                   gate_constants=gate_constants)
    torch.save(payload, path)
    write_json(path.with_suffix(".json"), dict(settings, status="complete",
        weights_sha256=sha256(path), diagnostics=diagnostics, cost=cost))


def attach_rollout_tail(captured, histories, cutover, steps, max_length, records):
    """Use only known events; hypothetically publish before a short history tail.

    Fresh teacher capture reconstructs raw retained gaps. These calibration
    tails use those same gaps, including the retained prefix's zero first gap.
    Ordinary service's seven-day clamp remains an explicitly separate rule.
    """
    records_by_uid = {int(row["uid"]): row for row in records}
    summary = []
    for uid, row in captured.items():
        times, items, behaviors = histories.rows[uid]
        stop = int(np.searchsorted(times, cutover, side="left"))
        length = row["parent"].seq_len
        if length < 1 or length != min(stop, max_length) or row["teacher"].seq_len != length:
            raise ValueError("source/teacher capture does not match retained pre-release history")
        if row["item_features"].shape[1] != length:
            raise ValueError("captured item features do not align with the native source")
        previous = row.get("previous_codes")
        if previous is not None and previous.shape[2] != length:
            raise ValueError("captured prior codes do not align with native source")
        record = records_by_uid[uid]
        if int(record["history_length"]) != length or int(record["last_history_timestamp"]) != int(times[stop - 1]):
            raise ValueError("source capture provenance does not match pre-release tail")
        kept_times = np.asarray(times[stop - length:stop])
        matched_ids = torch.as_tensor(np.asarray(items[stop - length:stop]),
                                      dtype=torch.long)[None]
        if "item_ids" in row and not torch.equal(row["item_ids"].cpu(), matched_ids):
            raise ValueError("captured raw item IDs differ from retained causal history")
        row["item_ids"] = matched_ids
        if length == 0 or np.any(kept_times >= cutover):
            raise ValueError("rollout calibration requires nonempty strictly pre-release history")
        deltas = np.zeros(length, dtype=np.float32)
        deltas[1:] = np.diff(kept_times)
        count = min(steps, length - 1)
        start = length - count
        row["rollout"] = dict(prefix_length=start,
            items=np.asarray(items[stop - count:stop], dtype=np.int64),
            behaviors=np.asarray(behaviors[stop - count:stop], dtype=np.int64),
            deltas=deltas[start:], timestamps=kept_times[start:])
        summary.append(dict(uid=int(uid), steps=count, prefix_length=start,
            last_timestamp=int(kept_times[-1]), target_cutover=int(cutover),
            gaps_above_serving_clamp=int(np.count_nonzero(deltas[start:] > 7 * 86400))))
    return summary


def _update_trainable(layer):
    if layer.mode == "legacy":
        parameters = (layer.input_weight, layer.input_bias)
    elif layer.mode == "C":
        parameters = (layer.input_weight, layer.input_bias, layer.previous_weight)
    else:
        parameters = (*layer.rotation, *layer.native_update,
                      layer.item_update_weight, layer.item_update_bias)
    return any(parameter.requires_grad for parameter in parameters)


@torch.no_grad()
def freeze_query_gates(adapter):
    """Independent static control; decoders and code updates remain trainable.

    Serving still executes the same RMS/tanh gate path. Zero weights make
    every gate exactly one, including during corrected behavior writes.
    """
    for layer in adapter.layers:
        for weight in (layer.key_gate_weight, layer.value_gate_weight):
            weight.zero_()
            weight.requires_grad_(False)


@torch.no_grad()
def prepare(model, captured, adapter, device, operations=None):
    rows = {}
    for uid, raw in captured.items():
        cache, teacher = raw["parent"].to(device), raw["teacher"].to(device)
        ids = raw["candidates"].to(device)[None]
        delta = torch.tensor([raw["query_delta"]], device=device)
        previous = raw.get("previous_codes")
        if previous is None:
            previous = cache.k.new_zeros(adapter.num_layers, 1, cache.seq_len,
                                         adapter.heads, adapter.rank)
        rows[uid] = dict(cache=cache, teacher=teacher,
            previous=previous.to(device), item=raw["item_features"].to(device),
            item_ids=None if raw.get("item_ids") is None else raw["item_ids"].to(device),
            ids=ids, delta=delta, wanted_logits=score(model, teacher, ids, delta)[0].detach())
        if adapter.feature_mode == "origin_encoder" and rows[uid]["item_ids"] is None:
            raise ValueError("origin encoder features require retained raw item IDs")
        if "rollout" in raw:
            tail = raw["rollout"]
            rows[uid]["rollout"] = dict(prefix_length=tail["prefix_length"],
                items=torch.as_tensor(tail["items"], device=device, dtype=torch.long),
                behaviors=torch.as_tensor(tail["behaviors"], device=device, dtype=torch.long),
                deltas=torch.as_tensor(tail["deltas"], device=device, dtype=cache.k.dtype))
        if operations is not None:
            operations.record_query(adapter, length=teacher.seq_len, queries=ids.numel(),
                                     fitted_layers=0, stage="teacher_score_prepare")
    return rows


@torch.no_grad()
def capture_local(model, adapter, rows, uids, layer_index, operations=None):
    records = []
    h, d = adapter.heads, adapter.head_dim
    for uid in uids:
        row = rows[uid]
        cache, teacher = row["cache"], row["teacher"]
        state = adapter.advance_codes(cache, row["previous"], row["item"],
                                       item_ids=row.get("item_ids"))
        gate_query = operation_gate_query(adapter, model, row["ids"])
        lower = adapter.make_history_read(state, fitted_layers=layer_index, gate_query=gate_query)
        capture = {} if gate_query is None else {"gate_q": gate_query[0]}
        if operations is not None:
            operations.record_gate_item_input(adapter, items=row["ids"].numel(), stage="local_capture")

        def read(index, attention, q, k, v):
            if adapter.gate_source == "first" and index == 0:
                capture["gate_q"] = q[0]
            if index != layer_index:
                return lower(index, attention, q, k, v)
            native = history_read(attention, q, k, v)
            wanted = history_read(attention, q, teacher.k[index], teacher.v[index])
            capture.update(q=q[0], native=native[0] / max(1, cache.seq_len),
                           wanted=wanted[0] / max(1, cache.seq_len))
            return native

        score(model, cache, row["ids"], row["delta"], history_read_fn=read)
        if operations is not None:
            operations.record_local_capture(adapter, length=cache.seq_len,
                queries=row["ids"].numel(), layer_index=layer_index)
        record = dict(uid=uid, k=cache.k[layer_index][0], v=cache.v[layer_index][0],
                      previous=row["previous"][layer_index, 0], item=row["item"][0], **capture)
        if row.get("item_ids") is not None:
            record["item_ids"] = row["item_ids"][0]
        records.append(record)
    return records


def local_batch(records, indices):
    chosen = [records[i] for i in indices]
    length = max(x["k"].shape[0] for x in chosen)
    result = {}
    keys = ("k", "v", "previous", "item")
    if "item_ids" in chosen[0]:
        keys += ("item_ids",)
    for key in keys:
        padded = []
        for row in chosen:
            value = row[key]
            tail = value.new_zeros((length - value.shape[0], *value.shape[1:]))
            padded.append(torch.cat((value, tail)))
        result[key] = torch.stack(padded)
    result["q"] = torch.stack([x["q"] for x in chosen])
    if "gate_q" in chosen[0]:
        result["gate_q"] = torch.stack([x["gate_q"] for x in chosen])
    result["wanted"] = torch.stack([x["wanted"] for x in chosen])
    result["lengths"] = torch.tensor([x["k"].shape[0] for x in chosen],
                                      device=result["k"].device)
    result["mask"] = torch.arange(length, device=result["k"].device)[None] < result["lengths"][:, None]
    return result


def fit_local(model, adapter, rows, train, validation, args, costs, operations=None):
    generator = torch.Generator().manual_seed(args.seed)
    reports = []
    for index, layer in enumerate(adapter.layers):
        records = capture_local(model, adapter, rows, train + validation, index, operations)
        residual = torch.stack([x["wanted"] - x["native"] for x in records[:len(train)]])
        scale = residual.square().mean((0, 2, 3)).sqrt().clamp_min(1e-4)
        if operations is not None:
            operations.add("local_residual_scale_flops", 3 * residual.numel() + 2 * scale.numel())
        parameters = [p for p in layer.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

        def forward(indices, backward=False):
            data = local_batch(records, indices)
            code = layer.advance(data["k"], data["v"], data["previous"], data["item"],
                                 item_ids=data.get("item_ids"))
            code = code.reshape(len(indices), data["k"].shape[1], adapter.heads, adapter.rank)
            actual = layer.read(model.blocks[index].attn, data["q"], data["k"],
                                data["v"], code, valid_mask=data["mask"][:, None, :],
                                gate_q=data.get("gate_q"))
            actual = actual / data["lengths"][:, None, None, None]
            costs["local_forward_calls"] += 1
            costs["local_forward_token_queries"] += int(data["mask"].sum()) * data["q"].shape[2]
            if operations is not None:
                operations.record_local(adapter, batch=len(indices),
                    padded_length=data["k"].shape[1], queries=data["q"].shape[2],
                    backward=backward, encoder_backward=backward and _update_trainable(layer),
                    stage="local_train" if backward else "local_validation")
            return actual, data["wanted"]

        @torch.no_grad()
        def validate():
            squared, count = 0., 0
            for start in range(len(train), len(records), args.batch_size):
                actual, wanted = forward(list(range(start, min(start + args.batch_size, len(records)))))
                squared += float((actual - wanted).square().sum())
                count += actual.numel()
            return squared / count

        best, best_epoch = validate(), 0
        initial = best
        state = copy.deepcopy(layer.state_dict())
        first_gradients = None
        history = [dict(epoch=0, validation_read_rate_mse=best)]
        for epoch in range(1, args.local_epochs + 1):
            order = torch.randperm(len(train), generator=generator).tolist()
            for start in range(0, len(order), args.batch_size):
                actual, wanted = forward(order[start:start + args.batch_size], backward=True)
                loss = ((actual - wanted) / scale[None, :, None, None]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite layer read loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if first_gradients is None:
                    first_gradients = {name: float(p.grad.norm()) if p.grad is not None else None
                                       for name, p in layer.named_parameters() if p.requires_grad}
                    if operations is not None:
                        operations.record_gradient_diagnostic(parameters=sum(p.numel() for p in parameters),
                                                               tensors=len(parameters))
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                if operations is not None:
                    operations.record_optimizer(parameters=sum(p.numel() for p in parameters), stage="local")
                costs["local_backward_calls"] += 1
            value = validate()
            history.append(dict(epoch=epoch, validation_read_rate_mse=value))
            if value < best:
                best, best_epoch, state = value, epoch, copy.deepcopy(layer.state_dict())
        layer.load_state_dict(state)
        report = dict(layer=index, selected_epoch=best_epoch, initial_mse=initial,
                      best_mse=best, history=history, first_gradients=first_gradients)
        reports.append(report)
        emit(dict(status="local_layer_complete", layer=index, selected_epoch=best_epoch,
                  initial_mse=initial, best_mse=best))
    return reports


def rollout_score(model, adapter, row, *, costs=None, operations=None, backward=False):
    """Differentiable continuation of a strictly pre-release observed tail."""
    tail = row["rollout"]
    prefix_length = tail["prefix_length"]
    original = row["cache"]
    cache = HSTUKVCache(original.k[:, :, :prefix_length],
                        original.v[:, :, :prefix_length], prefix_length)
    state = adapter.advance_codes(cache, row["previous"][:, :, :prefix_length],
                                   row["item"][:, :prefix_length],
                                   item_ids=None if row.get("item_ids") is None else
                                   row["item_ids"][:, :prefix_length])
    if operations is not None:
        operations.record_update(adapter, length=prefix_length,
                                  backward=backward and _update_trainable(adapter.layers[0]),
                                  stage="joint_rollout_encoder")
    for index in range(tail["items"].numel()):
        before = cache.seq_len
        if operations is not None:
            operations.record_event_write(adapter, old_length=before, width=1,
                coded_rows=prefix_length, backward=backward, stage="joint_rollout_write")
        cache, state, _ = append_corrected_event(model, cache, state, adapter,
            tail["items"][index:index + 1][None],
            tail["behaviors"][index:index + 1][None],
            tail["deltas"][index:index + 1][None], adapter.max_length,
            code_count=prefix_length)
        if operations is not None:
            operations.record_gate_item_input(adapter, items=1, stage="joint_rollout_write")
        if costs is not None:
            costs["joint_rollout_behavior_queries"] += 1
    actual = score(model, cache, row["ids"], row["delta"],
        history_read_fn=adapter.make_history_read(state, code_count=prefix_length,
            gate_query=operation_gate_query(adapter, model, row["ids"])))[0]
    if operations is not None:
        operations.record_gate_item_input(adapter, items=row["ids"].numel(), stage="joint_rollout_query")
        operations.record_query(adapter, length=cache.seq_len, queries=row["ids"].numel(),
            coded_rows=prefix_length, backward=backward, stage="joint_rollout_query")
        operations.record_score_loss(outputs=actual.numel(), backward=backward,
                                     stage="joint_rollout")
    if costs is not None:
        costs["joint_rollout_forward_calls"] += 1
        costs["joint_rollout_forward_token_queries"] += cache.seq_len * row["ids"].numel()
    return actual, cache, state


def fit_joint(model, adapter, rows, train, validation, args, costs, operations=None):
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate * .25)
    generator = torch.Generator().manual_seed(args.seed)

    use_rollout = getattr(args, "rollout_steps", 0) > 0

    def forward(uid, backward=False):
        row = rows[uid]
        state = adapter.advance_codes(row["cache"], row["previous"], row["item"],
                                       item_ids=row.get("item_ids"))
        actual = score(model, row["cache"], row["ids"], row["delta"],
                       history_read_fn=adapter.make_history_read(state,
                           gate_query=operation_gate_query(adapter, model, row["ids"])))[0]
        costs["joint_forward_token_queries"] += row["cache"].seq_len * row["ids"].numel()
        costs["joint_forward_calls"] += 1
        if operations is not None:
            operations.record_gate_item_input(adapter, items=row["ids"].numel(), stage="joint_static_query")
            operations.record_update(adapter, length=row["cache"].seq_len,
                                      backward=backward and _update_trainable(adapter.layers[0]),
                                      stage="joint_static_encoder")
            operations.record_query(adapter, length=row["cache"].seq_len,
                queries=row["ids"].numel(), backward=backward, stage="joint_static_query")
            operations.record_score_loss(outputs=actual.numel(), backward=backward, stage="joint_static")
        static = (actual - row["wanted_logits"]).square().mean()
        rolled = None
        if use_rollout and row["rollout"]["items"].numel():
            output, _, _ = rollout_score(model, adapter, row, costs=costs,
                                         operations=operations, backward=backward)
            rolled = (output - row["wanted_logits"]).square().mean()
            if operations is not None:
                operations.add("joint_two_view_combination_flops", 3)
        combined = static if rolled is None else .5 * static + .5 * rolled
        return combined, static, rolled

    @torch.no_grad()
    def validate():
        results = [forward(uid) for uid in validation]
        result = dict(validation_teacher_logit_mse=float(torch.stack([x[0] for x in results]).mean()),
                      validation_static_teacher_logit_mse=float(torch.stack([x[1] for x in results]).mean()))
        rolled = [x[2] for x in results if x[2] is not None]
        result["validation_rollout_teacher_logit_mse"] = None if not rolled else float(torch.stack(rolled).mean())
        return result

    initial = validate()
    best, best_epoch = initial["validation_teacher_logit_mse"], 0
    state = copy.deepcopy(adapter.state_dict())
    history = [dict(epoch=0, **initial)]
    first_gradients = None
    for epoch in range(1, args.joint_epochs + 1):
        order = torch.randperm(len(train), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            subset = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for i in subset:
                value, _, _ = forward(train[i], backward=True)
                loss = value / len(subset)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite joint score loss")
                loss.backward()
                costs["joint_backward_calls"] += 1
            if first_gradients is None:
                first_gradients = {name: float(p.grad.norm()) if p.grad is not None else None
                                   for name, p in adapter.named_parameters() if p.requires_grad}
                if operations is not None:
                    operations.record_gradient_diagnostic(parameters=sum(p.numel() for p in parameters),
                                                           tensors=len(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            if operations is not None:
                operations.record_optimizer(parameters=sum(p.numel() for p in parameters), stage="joint")
        metrics = validate()
        history.append(dict(epoch=epoch, **metrics))
        if metrics["validation_teacher_logit_mse"] < best:
            best, best_epoch, state = metrics["validation_teacher_logit_mse"], epoch, copy.deepcopy(adapter.state_dict())
        emit(dict(status="joint_epoch", epoch=epoch, **metrics, selected_epoch=best_epoch))
    adapter.load_state_dict(state)
    return dict(selected_epoch=best_epoch, history=history, first_gradients=first_gradients,
                objective_views="half_static_half_rollout" if use_rollout else "static_only")


@torch.no_grad()
def gate_statistics(model, adapter, rows, uids, operations=None):
    sums = [None] * adapter.num_layers
    squares = [None] * adapter.num_layers
    counts = [0] * adapter.num_layers
    for uid in uids:
        row = rows[uid]
        state = adapter.advance_codes(row["cache"], row["previous"], row["item"],
                                       item_ids=row.get("item_ids"))
        item_query = operation_gate_query(adapter, model, row["ids"])
        base = adapter.make_history_read(state, gate_query=item_query)
        first_query = None

        def reader(index, attention, q, k, v):
            nonlocal first_query
            if index == 0:
                first_query = q
            gate_q = item_query if adapter.gate_source == "item" else \
                (first_query if adapter.gate_source == "first" else q)
            gates = torch.stack(adapter.layers[index].query_gates(gate_q), 0).double()
            total = gates.sum((1, 3))
            total_sq = gates.square().sum((1, 3))
            sums[index] = total if sums[index] is None else sums[index] + total
            squares[index] = total_sq if squares[index] is None else squares[index] + total_sq
            counts[index] += q.shape[0] * q.shape[2]
            return base(index, attention, q, k, v)

        score(model, row["cache"], row["ids"], row["delta"], history_read_fn=reader)
        if operations is not None:
            operations.record_gate_item_input(adapter, items=row["ids"].numel(), stage="gate_statistics")
            operations.record_gate_statistics(adapter, length=row["cache"].seq_len,
                                               queries=row["ids"].numel())
    means = torch.stack([x / n for x, n in zip(sums, counts)])
    variances = torch.stack([s / n for s, n in zip(squares, counts)]) - means.square()
    if operations is not None:
        operations.add("gate_statistics_final_moment_flops", 6 * means.numel())
    return means.cpu(), variances.clamp_min(0).sqrt().cpu()


def priced_source_capture(capture_costs, pack):
    """Price retained explicit-update shapes without rerunning past candidates."""
    result = dict(capture_costs)
    if pack is None:
        return result
    geometry = pack.get("geometry", {})
    checkpoints = pack.get("prior_checkpoints", {})
    adapters, preparation = {}, 0
    for operation in geometry.get("explicit_updates", []):
        version = f"v{operation['version']}"
        if version not in adapters:
            adapters[version] = UnifiedRelayAdapter(**checkpoints[version]["config"])
        preparation += int(operation["count"]) * update_forward_flops(
            adapters[version], rows=int(operation["rows"]))
    result["source_code_update_flops"] = preparation
    return result


@torch.no_grad()
def initialize_continuation(adapter, policy):
    """Carry prior code without applying the previous release's update twice."""
    if policy == "inherit":
        return False
    if policy != "carry" or adapter.mode not in ("A", "B"):
        raise ValueError("carry initialization is defined for affine A/B updates")
    for layer in adapter.layers:
        for raw in layer.rotation:
            raw.zero_()  # Cayley(0) is identity.
        for weight in layer.native_update:
            weight.zero_()
        layer.item_update_weight.zero_()
        layer.item_update_bias.zero_()
    return True


def run(args):
    began = time.perf_counter()
    if args.freeze_gates and args.gate_only:
        raise ValueError("freeze-gates and gate-only leave no trainable control parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    panel, groups, _ = load_panel(args.panel)
    train, validation = groups["fit"][:args.fit_users], groups["validation"][:args.validation_users]
    if (set(train) | set(validation)) & set(groups["evaluation"]):
        raise RuntimeError("calibration intersects evaluation roles")
    if args.target != "v3" and args.source_pack is None:
        raise ValueError("later release requires a corrected-trajectory source pack")
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device, int(args.target[1:]))
    model = models[args.target]
    origin = torch.load(args.origin, map_location="cpu", weights_only=False)
    feature_bank_record = None
    feature_bank_created = False
    carry_initialized = False
    if origin["kind"] == UnifiedRelayAdapter.kind:
        adapter = UnifiedRelayAdapter(**origin["config"]).to(device)
        adapter.load_state_dict(origin["state_dict"])
        if adapter.mode != args.mode:
            raise ValueError("continuation checkpoint and requested update mode differ")
        if args.feature_mode == "origin_encoder" and adapter.feature_mode != "origin_encoder":
            raise ValueError("origin encoder branch must start from the structured origin checkpoint")
        carry_initialized = initialize_continuation(adapter, args.release_initialization)
        initial_gate_normalization = adapter.gate_normalization
        initial_gate_source = adapter.gate_source
        adapter.set_gate_source(args.gate_source)
        initial_read_width = [adapter.read_shared, adapter.read_private]
        changed_normalization = initial_gate_normalization != args.gate_normalization
        adapter.set_gate_normalization(args.gate_normalization)
        if changed_normalization:
            with torch.no_grad():
                for layer in adapter.layers:
                    layer.key_gate_weight.zero_()
                    layer.value_gate_weight.zero_()
    else:
        source = StructuredKVAdapter(**origin["config"]).to(device)
        source.load_state_dict(origin["state_dict"])
        bank_path = bank_hash = None
        if args.feature_mode == "origin_encoder":
            from query_relay_2026_10.fixed_features import build_bank
            bank_path = args.feature_bank or args.output.parent / "feature_bank" / "origin_encoder.pt"
            feature_bank_created = not bank_path.exists()
            origin_model = models["v3"]
            origin_items = origin_model.item_emb.weight.detach()
            feature_bank_record = build_bank(source, origin_items, bank_path,
                source_bindings=dict(origin_checkpoint_sha256=sha256(args.origin),
                    fixed_item_model="v3", fixed_model_checkpoint_sha256=panel["models"]["v3"]["checkpoint_sha256"]))
            bank_hash = feature_bank_record["sha256"]
            feature_bank_created = bool(feature_bank_record["created"])
            del origin_items
        adapter = UnifiedRelayAdapter.from_structured(source, mode=args.mode, xi_width=args.xi_width,
            gate_normalization=args.gate_normalization, feature_mode=args.feature_mode,
            feature_bank_path=None if bank_path is None else str(bank_path.resolve()),
            feature_bank_sha256=bank_hash, gate_source=args.gate_source)
        initial_gate_normalization, changed_normalization = args.gate_normalization, False
        initial_gate_source = args.gate_source
        initial_read_width = [12, 6]
        del source
    adapter.set_read_width(args.read_shared, args.read_private)
    adapter.eval()
    if args.gate_only:
        adapter.requires_grad_(False)
        for layer in adapter.layers:
            layer.key_gate_weight.requires_grad_(True)
            layer.value_gate_weight.requires_grad_(True)
    if args.freeze_gates:
        freeze_query_gates(adapter)
    settings = dict(target=args.target, mode=adapter.mode, seed=args.seed,
        fit_uids=train, validation_uids=validation, fit_users=len(train), validation_users=len(validation),
        local_epochs=args.local_epochs, joint_epochs=args.joint_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, gate_only=args.gate_only, freeze_gates=args.freeze_gates,
        read_shared=args.read_shared, read_private=args.read_private,
        initial_read_width=initial_read_width,
        gate_normalization=args.gate_normalization,
        gate_source=adapter.gate_source,
        initial_gate_source=initial_gate_source,
        gate_source_changed=initial_gate_source != adapter.gate_source,
        gate_source_switch_resets_weights=False,
        initial_gate_normalization=initial_gate_normalization,
        gate_weights_reset_for_normalization_change=changed_normalization,
        rollout_steps=args.rollout_steps,
        feature_mode=adapter.feature_mode,
        feature_bank_path=adapter.feature_bank_path,
        feature_bank_sha256=adapter.feature_bank_sha256,
        feature_bank_development_created=feature_bank_created,
        feature_bank_record=feature_bank_record,
        release_initialization=args.release_initialization,
        carry_initialization_applied=carry_initialized,
        rollout_release_hypothesis="Current release is applied just before a short, already observed strict pre-release tail.",
        rollout_time_gaps="raw retained gaps, matching raw-prefix teacher; serving independently clamps gaps to seven days",
        joint_score_views="half_static_half_rollout_same_teacher" if args.rollout_steps else "static_only",
        origin=str(args.origin), origin_sha256=sha256(args.origin),
        panel_sha256=sha256(args.panel / "binding.json"), role="development_calibration",
        source_mode="first_release_causal_V2" if args.source_pack is None else "candidate_continuous_trajectory",
        source_pack=str(args.source_pack) if args.source_pack else None,
        source_pack_sha256=sha256(args.source_pack) if args.source_pack else None,
        initial_origin_calibration_cost_is_additional=True,
        source_writes_during_capture="native_V2" if args.source_pack is None else "corrected_candidate",
        fitting_writes="differentiable_current_corrected_tail_with_zero_new_codes" if args.rollout_steps else "static_first_read_probe",
        source_state_available=not args.require_source_replay,
        native_writes_in_evaluation="corrected", new_row_code="zero_protocol_not_Fresh_claim",
        objectives=["same_query_read_MSE", "final_teacher_score_MSE"],
        execution_sources={str(p.relative_to(ROOT)): sha256(p) for p in
            (Path(__file__), ROOT / "scripts/query_relay_2026_10/adapter.py",
             ROOT / "scripts/query_relay_2026_10/session.py",
             ROOT / "scripts/query_relay_2026_10/calibration_cost.py",
             ROOT / "scripts/query_relay_2026_10/cost.py")})
    write_json(args.output / "configuration.json", settings)
    histories = load_history(args.panel, requested_uids=train + validation)
    pack = None
    if args.source_pack:
        pack = torch.load(args.source_pack, map_location="cpu", weights_only=False)
        if pack.get("settings", {}).get("target") != args.target:
            raise ValueError("source pack belongs to a different release")
        if origin["kind"] == UnifiedRelayAdapter.kind and args.target != "v3":
            previous_version = f"v{int(args.target[1:]) - 1}"
            captured_origin_hash = pack.get("prior_checkpoints", {}).get(previous_version, {}).get("sha256")
            if captured_origin_hash != sha256(args.origin):
                raise ValueError("source pack predecessor differs from the warmstart checkpoint")
        captured = {uid: pack["rows"][uid] for uid in train + validation}
        capture_costs, capture_records = pack["costs"], pack["records"]
    else:
        captured, capture_costs, capture_records = capture_rows(models, histories,
            train + validation, args.target, panel["known_vocab_size"])
    tail_records = attach_rollout_tail(captured, histories, dict(RELEASES)[args.target],
                                       args.rollout_steps, adapter.max_length, capture_records)
    capture_costs = priced_source_capture(capture_costs, pack)
    del models
    operations = OperationCounts()
    if feature_bank_created:
        operations.record_feature_bank_preparation(catalogue_items=model.cfg.num_items + 1,
            layers=adapter.num_layers, code_width=adapter.heads * adapter.rank,
            embedding_width=adapter.heads * adapter.head_dim)
    operations.record_native_capture(capture_costs, source_available=not args.require_source_replay)
    rows = prepare(model, captured, adapter, device, operations)
    del captured
    costs = Counter()
    local = fit_local(model, adapter, rows, train, validation, args, costs, operations)
    joint = fit_joint(model, adapter, rows, train, validation, args, costs, operations)
    means, std = gate_statistics(model, adapter, rows, train, operations)
    cost = operations.report(origin=warmstart_origin_cost(args.origin))
    diagnostics = dict(local=local, joint=joint, gate_mean=means.tolist(), gate_std=std.tolist(),
        gate_std_max=float(std.max()), operation_counts=dict(costs),
        capture_costs=dict(capture_costs), capture_records=capture_records,
        rollout_records=tail_records,
        elapsed_seconds=time.perf_counter() - began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        backbone_gradients_absent=all(p.grad is None for p in model.parameters()))
    checkpoint(adapter, args.output / "calibration.pt", settings, diagnostics, means, cost)
    emit(dict(status="complete", output=str(args.output), gate_std_max=float(std.max()),
              elapsed_seconds=diagnostics["elapsed_seconds"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--origin", type=Path, default=ORIGIN)
    p.add_argument("--source-pack", type=Path)
    p.add_argument("--feature-mode", choices=("native", "origin_encoder"), default="native")
    p.add_argument("--feature-bank", type=Path, help="Shared immutable origin bank; defaults beside first-release output")
    p.add_argument("--release-initialization", choices=("inherit", "carry"), default="inherit",
                   help="Continuation update initialization; first release retains its feature seed")
    p.add_argument("--target", default="v3", choices=["v3", "v4", "v5"])
    p.add_argument("--mode", default="B", choices=["legacy", "A", "B", "C"])
    p.add_argument("--xi-width", type=int, default=3)
    p.add_argument("--read-shared", type=int, default=12, choices=[0, 3, 6, 12])
    p.add_argument("--read-private", type=int, default=6, choices=[3, 6])
    p.add_argument("--gate-normalization", default="raw", choices=["raw", "rms"])
    p.add_argument("--gate-source", default="layer", choices=["layer", "first", "item"],
                   help="Gate on the layer query, native first query, or Current item embedding; native score projections stay layer-local")
    p.add_argument("--rollout-steps", type=int, default=0, choices=[0, 4])
    p.add_argument("--require-source-replay", action="store_true",
                   help="Charge source replay in primary cost; default reports reconstruction sensitivity separately")
    p.add_argument("--fit-users", type=int, default=64)
    p.add_argument("--validation-users", type=int, default=16)
    p.add_argument("--local-epochs", type=int, default=8)
    p.add_argument("--joint-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--freeze-gates", action="store_true",
                   help="Independent static effective-KV control: fix both gate weights at zero, train decoders/updaters normally, retain the executed gate FLOPs")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())

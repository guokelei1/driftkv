#!/usr/bin/env python3
"""Two frozen-backbone read corrections on the fixed pre-release Design 1 users.

Fit each layer against Full reads at the actual query produced by its fitted
lower layers. Validation is diagnostic only; training has a fixed endpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch

from design_one.calibrate import DEFAULT_CONFIG, DAY, inputs, source_hashes, timed
from design_one.release_config import CalibrationSettings, select_release_config
from evaluate_yambda500m_foundation_raw import load_histories, load_model, sha256_file
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one.nonlinear import (
    ContextKVReadViewAdapter, KVReadViewAdapter, NonlinearResponseAdapter,
)
from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from read_correction_2026_09.calibrate import capture, groups, padded_layer, stacked
from read_correction_2026_09.cost import CostModel, eager_read, teacher_history_read
from read_correction_2026_09.v2.calibrate import mixed_candidates
from selective_recompute_2026_09.calibrate import snapshot

KINDS = {"nonlinear_response": "design_one_nonlinear_response_v1",
         "kv_view": "design_one_kv_view_v1", "kv_context": "design_one_context_kv_view_v1",
         "kv_item": "design_one_item_kv_view_v1"}
WIDTHS = {"nonlinear_response": 256, "kv_view": 32, "kv_context": 32, "kv_item": 64}
EPOCHS = {"nonlinear_response": 200, "kv_view": 64, "kv_context": 32, "kv_item": 64}


def make_adapter(current, method, hidden_width=None):
    adapter_type = {"nonlinear_response": NonlinearResponseAdapter,
                    "kv_view": KVReadViewAdapter, "kv_context": ContextKVReadViewAdapter,
                    "kv_item": ItemKVReadViewAdapter}[method]
    return adapter_type(len(current.blocks), current.cfg.num_heads,
                        current.cfg.hidden_size//current.cfg.num_heads,
                        hidden_width=WIDTHS[method] if hidden_width is None else hidden_width,
                        max_length=current.cfg.max_seq_len)


def correction_cost(current, adapter, method, *, batch, queries, length, layers, current_rows=None):
    cost = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    model = sum(module.forward_flops(batch=batch, candidates=queries) if method == "nonlinear_response"
                else module.token_forward_flops(batch=batch, length=length,
                    **({"current_rows": current_rows} if current_rows is not None else {})) for module in adapter.layers[:layers])
    read = teacher_history_read(cost, length, queries=queries, batch=batch) if method != "nonlinear_response" else 0
    return model+layers*read


def prepared_override(current, adapter, cache, counts, method, fitted_layers, context=None, item_features=None,
                      producer_mask=None):
    if method == "nonlinear_response":
        return adapter.make_history_override(counts, fitted_layers=fitted_layers)
    features = {"context": context} if method == "kv_context" else {"item_features": item_features} if method == "kv_item" else {}
    if producer_mask is not None:
        features["producer_mask"] = producer_mask
    mapped = adapter.map_cache(cache, fitted_layers=fitted_layers, **features)
    override = adapter.make_history_override(current, mapped)
    return lambda layer, query, native: override(layer, query, native) if layer < fitted_layers else native


def row_features(rows, selected, method, device):
    field = {"kv_context": "context", "kv_item": "item_features"}.get(method)
    result = {field: torch.cat([rows[key][field] for key in selected], 0).to(device)} if field else {}
    if "producer_mask" in rows[selected[0]]:
        result["producer_mask"] = torch.cat([rows[key]["producer_mask"] for key in selected], 0).to(device)
    return result


@torch.no_grad()
def attach_scene_item_features(current, rows, device):
    for row in rows.values():
        row["item_features"] = current.lookup_item_embeddings(row["item_ids"].to(device)).detach().cpu()


@torch.no_grad()
def prepare_item_read_view(item_adapter, rows, uids, *, batch_size, device, costs):
    """Map each source row once; subsequent response fitting reads this view."""
    item_adapter.to(device).eval().requires_grad_(False)
    for selected in groups(uids, rows, batch_size):
        cache = stacked(rows, selected, "parent", device)
        features = torch.cat([rows[uid]["item_features"] for uid in selected], 0).to(device)
        mapped = item_adapter.map_cache(cache, features)
        for index, uid in enumerate(selected):
            rows[uid]["parent"] = type(cache)(mapped.k[:, index:index+1].cpu(),
                mapped.v[:, index:index+1].cpu(), mapped.seq_len)
        costs["frozen_item_view_flops"] += sum(layer.token_forward_flops(batch=len(selected), length=cache.seq_len)
                                              for layer in item_adapter.layers)


@torch.no_grad()
def collect(current, adapter, rows, uids, layer, method, *, batch_size, device, costs):
    """CPU tensors for one fitted layer; teacher and native share the actual q."""
    result = {}
    model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    for selected in groups(uids, rows, batch_size):
        cache = stacked(rows, selected, "parent", device)
        counts = torch.full((len(selected),), cache.seq_len, device=device)
        ids = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        dt = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        override = prepared_override(current, adapter, cache, counts, method, layer,
                                     **row_features(rows, selected, method, device))
        _, trace = score(current, cache, ids, dt, history_override=override, trace=True)
        query, native = trace.queries[layer], trace.history_heads[layer]
        teacher_k = torch.cat([rows[uid]["teacher"].k[layer] for uid in selected]).to(device)
        teacher_v = torch.cat([rows[uid]["teacher"].v[layer] for uid in selected]).to(device)
        teacher = history_read(current.blocks[layer].attn, query, teacher_k, teacher_v)
        wanted = (teacher-native)/counts[:, None, None, None]
        if not all(torch.isfinite(value).all() for value in (query, native, wanted)):
            raise RuntimeError("nonfinite same-query calibration tensors")
        for index, uid in enumerate(selected):
            result[uid] = tuple(value[index].cpu() for value in (query, native, wanted))
        b, q, n = len(selected), ids.shape[1], cache.seq_len
        costs["query_trace_flops"] += eager_read(model, n, queries=q, batch=b)
        costs["corrected_prefix_flops"] += correction_cost(current, adapter, method, batch=b, queries=q,
                                                          length=n, layers=layer)
        costs["teacher_same_query_flops"] += teacher_history_read(model, n, queries=q, batch=b)
        costs["target_normalization_flops"] += 2*wanted.numel()
    return tuple(torch.stack([result[uid][index] for uid in uids]) for index in range(3))


def aggregate_loss(prediction, wanted, counts, unit, max_length):
    """One shared scalar unit; N² weighting equals aggregate-response MSE."""
    error = ((prediction-wanted)/unit).square().mean((1, 2, 3))
    return (error*(counts/max_length).square()).mean()


def response_metrics(prediction, wanted, counts):
    error = (prediction.double()-wanted.double()).square().mean((1, 2, 3))
    baseline = wanted.double().square().mean((1, 2, 3))
    mass = counts.double().square()
    return dict(rate_mse=float(error.mean()), uncorrected_rate_mse=float(baseline.mean()),
        relative_rate_mse=float(error.mean()/baseline.mean().clamp_min(1e-30)),
        aggregate_response_mse=float((error*mass).mean()),
        uncorrected_aggregate_response_mse=float((baseline*mass).mean()),
        relative_aggregate_response_mse=float((error*mass).sum()/(baseline*mass).sum().clamp_min(1e-30)))


@torch.no_grad()
def normalize_layer(module, method, query, native, wanted, counts, keys, values, costs, features=None):
    unit = wanted.double().square().mean().sqrt().clamp_min(1e-8).float()
    if method == "nonlinear_response":
        features = module.features(query, native, counts).double()
        center, scale = features.mean((0, 1)), features.std((0, 1), correction=0).clamp_min(1e-4)
        output = torch.full((query.shape[1]*query.shape[-1],), float(unit))
        costs["normalization_statistics_flops"] += 8*features.numel()+3*wanted.numel()
    else:
        total = int(counts.sum())
        sums = torch.zeros(keys.shape[-1]*2+(features.shape[-1] if features is not None else 0), dtype=torch.double)
        squares = torch.zeros_like(sums)
        for index, (k, v, count) in enumerate(zip(keys, values, counts, strict=True)):
            parts = [k[:int(count)], v[:int(count)]]
            if features is not None:
                parts.append(features[index, :int(count)])
            joined = torch.cat(parts, -1).double()
            sums += joined.sum(0)
            squares += joined.square().sum(0)
        center = sums/total
        scale = (squares/total-center.square()).clamp_min(0).sqrt().clamp_min(1e-4)
        output = scale[:2*keys.shape[-1]]
        costs["normalization_statistics_flops"] += 6*total*sums.numel()+3*wanted.numel()
    module.set_normalization(center.float(), scale.float(), output.float())
    return unit


def compact_item_read(module, attention, query, keys, values, item_features, valid):
    """Differentiable Item read; fold the LIVE output parameters every call.

    This is the same K/V residual MLP, with its output affine map contracted
    through QK and weighted V. ELU+1 remains after the complete corrected QK.
    Native inputs and q are frozen during layerwise fitting; both affine maps
    and biases keep their original parameters, units and optimizer state.
    """
    features = torch.cat((keys, values, item_features), -1)
    normalized = (features.to(module.input_center.dtype)-module.input_center)/module.input_scale
    hidden = torch.nn.functional.silu(module.input(normalized))
    h, d, m = module.heads, module.head_dim, module.hidden_width
    # No detach or persistent folded buffer: gradients reach the original
    # output weight/bias through their frozen physical output units.
    weight = (module.output.weight*module.output_scale[:, None]).reshape(2, h, d, m)
    bias = (module.output.bias*module.output_scale).reshape(2, h, d)
    k = keys.reshape(keys.shape[0], -1, h, d).transpose(1, 2)
    v = values.reshape(values.shape[0], -1, h, d).transpose(1, 2)
    projected_query = query @ weight[0]
    key_bias = (query*bias[0][None, :, None]).sum(-1, keepdim=True)
    logits = query @ k.transpose(-2, -1)
    logits = logits + projected_query @ hidden[:, None].transpose(-2, -1) + key_bias
    weights = attention._activate(logits*attention.scale)
    weights = weights*valid[:, None, None].to(weights.dtype)
    native_read = weights @ v
    hidden_read = weights @ hidden[:, None]
    value_delta = hidden_read @ weight[1].transpose(-2, -1)
    bias_delta = weights.sum(-1, keepdim=True)*bias[1][None, :, None]
    return native_read+value_delta+bias_delta


def compact_item_fit_flops(module, *, batch, length, queries):
    """Full forward/backward estimate, including live folds and rate residual.

    Encoder inputs are frozen. Count complete factored-read backward as twice
    its forward (conservative), including native QK/AV, mask and rate units.
    Each batch also folds every output weight/bias once in EACH direction.
    Loss, optimizer and clipping remain in their existing separate ledgers.
    """
    f, m, w, h = module.input.in_features, module.hidden_width, module.width, module.heads
    rows = batch*length
    encoder_forward = rows*(2*f+2*f*m+2*m)
    encoder_backward = rows*(2*f*m+2*m)
    read = batch*queries*(length*(4*w+7*h+4*h*m)+4*w*m+7*w)
    fold = 2*w*(m+1)
    return dict(forward=encoder_forward+read+fold, backward=encoder_backward+2*read+fold)


def fit_layer(current, module, rows, train, layer, method, tensors, *, epochs, batch_size,
              device, max_length, costs, residual_rate_rms=None, learning_rate=.001, compact_training=False):
    query, native, wanted = tensors
    counts = torch.tensor([rows[uid]["parent"].seq_len for uid in train], dtype=torch.float32)
    keys, values, features = None, None, None
    if method != "nonlinear_response":
        keys, values, _ = padded_layer(rows, train, layer)
    field = {"kv_context": "context", "kv_item": "item_features"}.get(method)
    if field:
        features = keys.new_zeros((len(train), keys.shape[1], rows[train[0]][field].shape[-1]))
        for index, key in enumerate(train):
            features[index, :int(counts[index])] = rows[key][field][0]
    module.cpu()
    unit = (normalize_layer(module, method, query, native, wanted, counts, keys, values, costs, features=features)
            if residual_rate_rms is None else torch.tensor(residual_rate_rms, dtype=torch.float32))
    module.to(device).train().requires_grad_(True)
    parameter_count = sum(p.numel() for p in module.parameters())
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=.0001)
    generator = torch.Generator().manual_seed(17+layer)
    model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    training_batch = len(train) if method == "nonlinear_response" else batch_size
    n, queries = (int(counts.max()) if keys is None else keys.shape[1]), query.shape[2]
    epochs_log, steps, clipped = [], 0, 0

    def predict(indices):
        q, r, target, count = [value[indices].to(device) for value in (query, native, wanted, counts)]
        if method == "nonlinear_response":
            prediction = module.delta_rate(q, r, count)
        else:
            valid = torch.arange(n, device=device)[None] < count[:, None]
            if compact_training:
                mapped = compact_item_read(module, current.blocks[layer].attn, q, keys[indices].to(device),
                    values[indices].to(device), features[indices].to(device), valid)
            else:
                k, v = (module.map_tokens(keys[indices].to(device), values[indices].to(device), features[indices].to(device))
                        if features is not None else module.map_tokens(keys[indices].to(device), values[indices].to(device)))
                mapped = history_read(current.blocks[layer].attn, q, k, v, count=valid)
            prediction = (mapped-r)/count[:, None, None, None]
        return prediction, target, count

    def computation_cost(batch):
        if compact_training:
            return compact_item_fit_flops(module, batch=batch, length=n, queries=queries)
        result = module.fit_flops(batch=batch, candidates=queries) if method == "nonlinear_response" else module.fit_flops(batch=batch, length=n)
        if method != "nonlinear_response":
            read = teacher_history_read(model, n, queries=queries, batch=batch)
            read += batch*queries*n*current.cfg.num_heads  # padded-history mask
            read += 2*batch*queries*current.cfg.hidden_size
            result["forward"] += read
            result["backward"] += 2*read  # conservative mapped-read autograd estimate
        return result

    @torch.no_grad()
    def assess():
        predictions, objective = [], 0.
        for start in range(0, len(train), training_batch):
            selected = torch.arange(start, min(start+training_batch, len(train)))
            prediction, target, count = predict(selected)
            objective += float(aggregate_loss(prediction, target, count, unit.to(device), max_length))*len(selected)/len(train)
            predictions.append(prediction.cpu())
            costs["fitting_diagnostic_flops"] += computation_cost(len(selected))["forward"]+8*target.numel()
        return dict(objective=objective, **response_metrics(torch.cat(predictions), wanted, counts))

    initial = assess()
    for epoch in range(1, epochs+1):
        order = torch.randperm(len(train), generator=generator)
        objective_sum = 0.
        for start in range(0, len(train), training_batch):
            selected = order[start:start+training_batch]
            optimizer.zero_grad(set_to_none=True)
            prediction, target, count = predict(selected)
            objective = aggregate_loss(prediction, target, count, unit.to(device), max_length)
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite nonlinear response objective")
            objective.backward()
            grad = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
            if not torch.isfinite(grad):
                raise RuntimeError("nonfinite nonlinear response gradient")
            clipped += int(float(grad) > 1.)
            optimizer.step()
            steps += 1
            objective_sum += float(objective.detach())*len(selected)/len(train)
            costs["network_and_read_training_flops_estimate"] += sum(computation_cost(len(selected)).values())
            costs["loss_training_flops_estimate"] += 12*target.numel()+5*len(selected)
            costs["optimizer_and_clip_flops_estimate"] += 18*parameter_count
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            epochs_log.append(dict(epoch=epoch, online_training_objective=objective_sum))
    module.eval().requires_grad_(False)
    final = assess()
    return dict(layer=layer, initial_train=initial, final_train=final, epochs=epochs_log,
                optimizer_steps=steps, clipped_steps=clipped, trainable_parameters=parameter_count,
                residual_rate_rms=float(unit), actual_training_batch=training_batch,
                padded_history_length=n, fitting_users=len({rows[key].get("uid", key) for key in train}),
                fitting_scenes=len(train), fitting_queries=queries)


@torch.no_grad()
def validate(current, adapter, rows, validation, method, *, batch_size, device, costs, logit_targets=None):
    model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    outputs, counts_all, native_logits, corrected_logits, full_logits, ordered_keys = [], [], [], [], [], []
    for selected in groups(validation, rows, batch_size):
        cache, teacher = [stacked(rows, selected, name, device) for name in ("parent", "teacher")]
        counts = torch.full((len(selected),), cache.seq_len, device=device)
        ids = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        dt = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        features = row_features(rows, selected, method, device)
        override = prepared_override(current, adapter, cache, counts, method, len(current.blocks), **features)
        corrected, trace = score(current, cache, ids, dt, history_override=override, trace=True)
        if logit_targets is None:
            full, _ = score(current, teacher, ids, dt)
            native, _ = score(current, cache, ids, dt)
        else:
            full, native = [torch.stack([logit_targets[uid][index] for uid in selected]).to(device)
                            for index in (0, 1)]
        predictions, wanted = [], []
        for layer, block in enumerate(current.blocks):
            q, r = trace.queries[layer], trace.history_heads[layer]
            target = history_read(block.attn, q, teacher.k[layer], teacher.v[layer])
            predictions.append((override(layer, q, r)-r)/counts[:, None, None, None])
            wanted.append((target-r)/counts[:, None, None, None])
        outputs.append((torch.stack(predictions).cpu(), torch.stack(wanted).cpu()))
        ordered_keys.extend(selected)
        counts_all.append(counts.cpu())
        corrected_logits.append(corrected.cpu())
        native_logits.append(native.cpu())
        full_logits.append(full.cpu())
        b, q, n = len(selected), ids.shape[1], cache.seq_len
        costs["validation_native_and_corrected_trace_flops"] += (2 if logit_targets is None else 1)*eager_read(model, n, queries=q, batch=b)
        if logit_targets is None:
            costs["teacher_full_query_flops"] += eager_read(model, n, queries=q, batch=b)
        costs["teacher_same_query_flops"] += len(current.blocks)*teacher_history_read(model, n, queries=q, batch=b)
        # Two correction applications share one prepared KV cache.
        costs["validation_correction_flops"] += correction_cost(current, adapter, method, batch=b, queries=q,
            length=n, layers=len(current.blocks), current_rows=int(features["producer_mask"].sum()) if "producer_mask" in features else None)
        if method == "nonlinear_response":
            costs["validation_correction_flops"] += correction_cost(current, adapter, method, batch=b, queries=q,
                                                                   length=n, layers=len(current.blocks))
        else:
            costs["validation_correction_flops"] += len(current.blocks)*teacher_history_read(model, n, queries=q, batch=b)
        costs["validation_metric_flops"] += 12*outputs[-1][0].numel()+10*full.numel()
    prediction = torch.cat([item[0] for item in outputs], 1)
    wanted = torch.cat([item[1] for item in outputs], 1)
    count = torch.cat(counts_all)
    corrected, full, native = [torch.cat(chunks).double() for chunks in (corrected_logits, full_logits, native_logits)]
    if not all(torch.isfinite(value).all() for value in (prediction, wanted, corrected, full, native)):
        raise RuntimeError("nonfinite held-out validation output")
    result = dict(layers=[dict(layer=index, **response_metrics(prediction[index], wanted[index], count))
                        for index in range(len(current.blocks))],
        full_logit_mse=float((corrected-full).square().mean()),
        native_full_logit_mse=float((native-full).square().mean()),
        relative_full_logit_mse=float((corrected-full).square().sum()/(native-full).square().sum().clamp_min(1e-30)))
    if any("scene_kind" in rows[key] for key in ordered_keys):
        result["by_scene"] = {}
        for kind in sorted({rows[key]["scene_kind"] for key in ordered_keys}):
            indices = [index for index, key in enumerate(ordered_keys) if rows[key]["scene_kind"] == kind]
            result["by_scene"][kind] = dict(scenes=len(indices),
                layers=[dict(layer=layer, **response_metrics(prediction[layer, indices], wanted[layer, indices], count[indices]))
                        for layer in range(len(current.blocks))],
                full_logit_mse=float((corrected[indices]-full[indices]).square().mean()),
                native_full_logit_mse=float((native[indices]-full[indices]).square().mean()))
    return result


def fit(current, rows, train, validation, method, *, batch_size, device, epochs=None, costs=None, timings=None,
        hidden_width=None, adapter=None, residual_rate_rms=None, learning_rate=.001, compact_training=False):
    costs = defaultdict(int) if costs is None else costs
    timings = defaultdict(float) if timings is None else timings
    epochs = EPOCHS[method] if epochs is None else epochs
    train_uids = {rows[key].get("uid", key) for key in train}
    validation_uids = {rows[key].get("uid", key) for key in validation}
    if train_uids & validation_uids:
        raise ValueError("fitting and validation scenes must remain disjoint by UID")
    current.eval().requires_grad_(False)
    if adapter is None:
        torch.manual_seed(17)
        adapter = make_adapter(current, method, hidden_width=hidden_width)
    adapter.to(device).eval().requires_grad_(False)
    records = []
    for layer, module in enumerate(adapter.layers):
        tensors = timed(lambda: collect(current, adapter, rows, train, layer, method,
            batch_size=batch_size, device=device, costs=costs), timings, "fitting_teacher_and_query_reads", device)
        record = timed(lambda: fit_layer(current, module, rows, train, layer, method, tensors,
            epochs=epochs, batch_size=batch_size, device=device, max_length=current.cfg.max_seq_len,
            costs=costs, learning_rate=learning_rate, compact_training=compact_training,
            residual_rate_rms=None if residual_rate_rms is None else residual_rate_rms[layer]),
            timings, "network_fit", device)
        records.append(record)
        print(json.dumps({"status": "layer_fitted", "method": method, "layer": layer,
                          "initial_train": record["initial_train"], "final_train": record["final_train"]}), flush=True)
    validation_stats = timed(lambda: validate(current, adapter, rows, validation, method,
        batch_size=batch_size, device=device, costs=costs), timings, "held_out_validation", device)
    return adapter, dict(layers=records, validation=validation_stats)


def initial_calibration(path, config_path, config, versions, method="kv_view"):
    """Bind the fixed full-budget KV32 or Item-KV64 warm start before recapture."""
    metadata_path, weights_path = path/"calibration.json", path/"calibration.pt"
    metadata = json.loads(metadata_path.read_text())
    weights_hash = sha256_file(weights_path)
    _, _, _, train, validation = inputs(config_path)
    if (metadata["kind"] != KINDS[method] or metadata["status"] != "complete"
        or metadata["fit_uids"] != train or metadata["validation_uids"] != validation
        or metadata["queries_per_user"] != 16 or metadata["weights_sha256"] != weights_hash
        or metadata["configuration"]["sha256"] != sha256_file(config_path)
        or metadata["users_file_sha256"] != config["users"]["sha256"]
        or metadata["checkpoint_hashes"] != {name: versions[name]["checkpoint_sha256"] for name in ("v4", "v5")}):
        raise ValueError("initial calibration differs from the fixed full-budget KV binding")
    saved = torch.load(weights_path, map_location="cpu", weights_only=False)
    expected = dict(num_layers=6, heads=6, head_dim=32, hidden_width=64 if method == "kv_item" else 32, max_length=1024)
    if saved["kind"] != KINDS[method] or saved["adapter"]["config"] != expected:
        raise ValueError("refinement requires the fixed six-layer KV32 or Item-KV64 adapter")
    adapter_type = ItemKVReadViewAdapter if method == "kv_item" else KVReadViewAdapter
    adapter = adapter_type.from_state_dict(saved["adapter"])
    binding = dict(path=str(path.resolve()), metadata_sha256=sha256_file(metadata_path),
                   weights_sha256=weights_hash, fit_users=len(train), validation_users=len(validation),
                   calibration_flops=metadata["cost"]["calibration_flops"],
                   teacher_flops=metadata["cost"]["teacher_flops"])
    if method == "kv_item":
        binding.update(kind=metadata["kind"], adapter_config=expected, settings=metadata["settings"],
            fitting_stage=metadata.get("fitting_stage", "same_query_response_fit"),
            trainable_parameters=sum(parameter.numel() for parameter in adapter.parameters()),
            residual_rate_rms=[layer["residual_rate_rms"] for layer in metadata["diagnostics"]["layers"]])
    return adapter, metadata, binding


@torch.no_grad()
def collect_logits(current, rows, uids, *, batch_size, device, costs):
    """Actual Full/native final scores, each computed once for the fixed queries."""
    result = {}
    model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    for selected in groups(uids, rows, batch_size):
        ids = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        dt = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        scores = []
        for key, ledger in (("teacher", "teacher_full_query_flops"), ("parent", "native_target_query_flops")):
            cache = stacked(rows, selected, key, device)
            values, _ = score(current, cache, ids, dt)
            scores.append(values.detach().cpu())
            costs[ledger] += eager_read(model, cache.seq_len, queries=ids.shape[1], batch=len(selected))
        if not all(torch.isfinite(value).all() for value in scores):
            raise RuntimeError("nonfinite Full/native logit targets")
        for index, uid in enumerate(selected):
            result[uid] = tuple(values[index] for values in scores)
    return result


def joint_logit_loss(logits, teacher, unit, *, centered=False):
    residual = logits-teacher
    if centered:
        residual = residual-residual.mean()  # one mean across every logit in the actual minibatch
    return (residual/unit).square().mean()


def refine_logits(current, adapter, rows, train, validation, *, batch_size, device,
                  epochs=16, costs=None, timings=None, method="kv_view", mixed_train=None, mixed_validation=None,
                  producer_scale=False, centered=False):
    """Jointly train only the existing six KV maps against final Full logits."""
    costs = defaultdict(int) if costs is None else costs
    timings = defaultdict(float) if timings is None else timings
    current.eval().requires_grad_(False)
    adapter.to(device).train()
    if producer_scale:
        adapter.freeze_base()
    else:
        adapter.requires_grad_(True)
    target_keys = train+validation+(mixed_train or [])+(mixed_validation or [])
    if mixed_train is not None:
        if len(train) != len(mixed_train) or any(rows[p]["uid"] != rows[m]["uid"] or
                rows[p]["parent"].seq_len != rows[m]["parent"].seq_len for p, m in zip(train, mixed_train)):
            raise ValueError("paired refinement scenes must preserve UID order and history lengths")
    targets = timed(lambda: collect_logits(current, rows, target_keys,
        batch_size=batch_size, device=device, costs=costs), timings, "full_and_native_logit_targets", device)
    residual = torch.stack([targets[uid][0]-targets[uid][1] for uid in train]).double()
    unit = residual.square().mean().sqrt().clamp_min(1e-8).float().to(device)
    costs["logit_normalization_flops"] += 4*residual.numel()+1
    model = CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, "torch")
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.01 if producer_scale else .0001,
                                 weight_decay=0. if producer_scale else .0001)
    parameters = sum(p.numel() for p in trainable)
    rng = torch.Generator().manual_seed(17)

    def predict(selected):
        cache = stacked(rows, selected, "parent", device)
        counts = torch.full((len(selected),), cache.seq_len, device=device)
        ids = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        dt = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        features = row_features(rows, selected, method, device)
        override = prepared_override(current, adapter, cache, counts, method, len(current.blocks), **features)
        logits, _ = score(current, cache, ids, dt, history_override=override)
        teacher = torch.stack([targets[uid][0] for uid in selected]).to(device)
        forward = eager_read(model, cache.seq_len, queries=ids.shape[1], batch=len(selected))
        forward += correction_cost(current, adapter, method, batch=len(selected),
            queries=ids.shape[1], length=cache.seq_len, layers=len(current.blocks),
            current_rows=int(features["producer_mask"].sum()) if "producer_mask" in features else None)
        return logits, teacher, forward

    @torch.no_grad()
    def assess_train(keys=None):
        keys = train if keys is None else keys
        baseline = residual
        if keys is not train:
            baseline = torch.stack([targets[key][0]-targets[key][1] for key in keys]).double()
            costs["joint_logit_diagnostic_flops"] += 3*baseline.numel()
        squared_error, centered_error = 0., 0.
        for selected in groups(keys, rows, batch_size):
            logits, teacher, forward = predict(selected)
            difference = logits.double()-teacher.double()
            squared_error += float(difference.square().sum())
            costs["joint_logit_diagnostic_flops"] += forward+3*logits.numel()
            if centered:
                centered_error += float((difference-difference.mean()).square().sum())
                costs["centered_logit_diagnostic_flops"] += 4*difference.numel()
        error = squared_error/baseline.numel()
        result = dict(full_logit_mse=error, native_full_logit_mse=float(baseline.square().mean()),
                    relative_full_logit_mse=error/max(float(baseline.square().mean()), 1e-30),
                    objective=error/float(unit.square()))
        if centered:
            result.update(centered_logit_mse=centered_error/baseline.numel(),
                centered_objective=centered_error/baseline.numel()/float(unit.square()))
            costs["centered_logit_diagnostic_flops"] += 4
        return result

    initial = timed(assess_train, timings, "initial_joint_logit_diagnostic", device)
    initial_mixed = (timed(lambda: assess_train(mixed_train), timings, "initial_mixed_logit_diagnostic", device)
                     if mixed_train is not None else None)
    records, steps, clipped, first_gradients = [], 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        epoch_train = mixed_train if mixed_train is not None and (producer_scale or epoch % 2 == 0) else train
        order = [epoch_train[index] for index in torch.randperm(len(epoch_train), generator=rng).tolist()]
        batches = list(groups(order, rows, batch_size))
        objective_sum = 0.
        for index in torch.randperm(len(batches), generator=rng).tolist():
            selected = batches[index]
            optimizer.zero_grad(set_to_none=True)
            logits, teacher, forward = predict(selected)
            objective = joint_logit_loss(logits, teacher, unit, centered=centered)
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite joint Full-logit objective")
            # Equal-length buckets can be smaller than eight users. A fixed
            # denominator keeps each user's gradient contribution consistent.
            (objective*len(selected)/batch_size).backward()
            if steps == 0:
                first_gradients = [float(sum(p.grad.detach().square().sum() for p in layer.parameters()
                    if p.grad is not None).sqrt()) for layer in adapter.layers]
                costs["joint_gradient_diagnostic_flops"] += 3*parameters
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(gradient):
                raise RuntimeError("nonfinite joint logit gradients")
            clipped += int(float(gradient) > 1.)
            optimizer.step()
            steps += 1
            objective_sum += float(objective.detach())*len(selected)/len(train)
            costs["joint_logit_training_flops_estimate"] += 3*forward
            costs["joint_logit_loss_flops_estimate"] += 12*logits.numel()+2
            if centered:
                costs["centered_logit_training_flops_estimate"] += 4*logits.numel()
            costs["optimizer_and_clip_flops_estimate"] += 18*parameters
        if mixed_train is not None or epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            record = dict(epoch=epoch, online_training_objective=objective_sum)
            if centered:
                record["centered_training_objective"] = objective_sum
            if mixed_train is not None:
                record.update(scene_kind=rows[epoch_train[0]]["scene_kind"], users=len(epoch_train),
                              queries_per_user=rows[epoch_train[0]]["candidates"].numel(), batches=len(batches))
            records.append(record)
            print(json.dumps({"status": "joint_logit_epoch", **record}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["joint_logit_fit"] += time.perf_counter()-started
    adapter.eval().requires_grad_(False)
    final = timed(assess_train, timings, "final_joint_logit_diagnostic", device)
    validation_stats = timed(lambda: validate(current, adapter, rows, validation, method,
        batch_size=batch_size, device=device, costs=costs, logit_targets=targets),
        timings, "held_out_validation", device)
    result = dict(joint_logit=dict(initial_train=initial, final_train=final,
        epochs=records, optimizer_steps=steps, clipped_steps=clipped, trainable_parameters=parameters,
        first_step_layer_gradient_norms=first_gradients, fit_only_full_minus_native_logit_rms=float(unit)),
        validation=validation_stats)
    if mixed_train is not None:
        result["joint_logit"].update(initial_mixed_train=initial_mixed,
            final_mixed_train=timed(lambda: assess_train(mixed_train), timings, "final_mixed_logit_diagnostic", device))
        result["mixed_validation"] = timed(lambda: validate(current, adapter, rows, mixed_validation, method,
            batch_size=batch_size, device=device, costs=costs, logit_targets=targets),
            timings, "mixed_held_out_validation", device)
    return adapter, result


def save_fixed_layer0_control(output, initial, provenance):
    """Publish a fixed rule from the original artifact, with no new fit/teacher."""
    if (output/"calibration.json").exists() or (output/"calibration.pt").exists():
        raise FileExistsError("fixed-control evidence already exists")
    from hstu_kvcache.design_one.producer_scaled_item import ProducerScaledItemKVAdapter
    base, inherited, binding = initial
    control = ProducerScaledItemKVAdapter.from_item_adapter(base).eval().requires_grad_(False)
    with torch.no_grad():
        control.layers[0].native_scale.zero_()
    metadata = json.loads(json.dumps(inherited))
    metadata.pop("weights_sha256", None)
    metadata.update(kind=control.kind, adapter_config=control.get_config(), initial_calibration=binding,
        fitting_stage="fixed_layer0_native_control", execution_sources=provenance,
        fitting="no new fit or teacher; fixed native scales [0,1,1,1,1,1] on original Round11 weights",
        diagnostics_provenance="inherited Round11 pure-state diagnostics; Parent mapping is unchanged",
        producer_scale_refinement=dict(initial_alpha=[1.]*len(control.layers),
            final_alpha=[float(layer.native_scale) for layer in control.layers], trainable_parameters=0,
            frozen_base=True, training_scene=None, loss=None, fitted=False,
            rule="skip residual correction only on Current-produced layer0 rows"),
        settings=dict(epochs=0, optimizer="none", rule="fixed native layer0 scale zero, other scales one"),
        cost=dict(inherited_calibration_flops=binding["calibration_flops"], rule_publication_flops=0,
            calibration_flops=binding["calibration_flops"], teacher_flops=binding["teacher_flops"],
            scope="original Round11 calibration charged once; no new fit, captures, or teacher access",
            convention="fixed scalar assignment/copies have zero arithmetic FLOPs"))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=control.kind, adapter=control.export_state(), metadata=metadata), output/"calibration.pt")
    metadata["weights_sha256"] = sha256_file(output/"calibration.pt")
    (output/"calibration.json").write_text(json.dumps(metadata, indent=2)+"\n")


def training_epochs(args):
    """Keep historical endpoints; override only the two fresh pure fits."""
    if args.epochs is not None:
        allowed = {"kv_item": (32, 48, 64), "nonlinear_response": (100, 200)}
        if (args.epochs not in allowed.get(args.method, ()) or args.initial_calibration is not None
                or args.rolling_scenes or args.mixed_refinement or args.producer_scale_refinement
                or args.centered_logit_refinement or args.recent_only_queries
                or args.refinement_epochs is not None or args.refinement_objective != "logit"
                or args.hidden_width is not None or args.queries_per_user is not None):
            raise ValueError("--epochs supports only fresh pure Item64 (32/48/64) or Response256 (100/200), without other overrides")
    if args.small:
        return 2
    if args.initial_calibration is not None:
        return args.refinement_epochs or (8 if args.method == "kv_item" else 16)
    return args.epochs if args.epochs is not None else 32 if args.rolling_scenes else EPOCHS[args.method]


def run(args):
    began = time.perf_counter()
    epochs = training_epochs(args)
    if args.compact_training and (args.method != "kv_item" or args.initial_calibration is not None
            or args.item_read_view is not None or args.refinement_objective != "logit" or args.rolling_scenes
            or args.mixed_refinement or args.producer_scale_refinement or args.centered_logit_refinement
            or args.recent_only_queries or args.refinement_epochs is not None or args.epochs not in (None, 64)
            or args.hidden_width is not None or args.queries_per_user is not None):
        raise ValueError("--compact-training supports only fresh pure Item64 with its original64epochs/sixteen queries")
    train_limit, val_limit = (4, 2) if args.small else (None, None)
    config, stage, versions, train, validation = inputs(args.config, train_limit, val_limit)
    if args.item_read_view is not None and (args.method != "nonlinear_response" or args.initial_calibration is not None
            or args.refinement_objective != "logit" or args.rolling_scenes or args.mixed_refinement
            or args.producer_scale_refinement or args.centered_logit_refinement or args.recent_only_queries
            or args.refinement_epochs is not None or args.hidden_width is not None or args.queries_per_user is not None):
        raise ValueError("--item-read-view supports only a fresh Response256 fit on the original sixteen-query budget")
    if args.centered_logit_refinement and (args.method != "kv_item" or args.initial_calibration is None
            or args.refinement_objective != "logit" or args.mixed_refinement or args.producer_scale_refinement
            or args.rolling_scenes or args.refinement_epochs is not None or args.recent_only_queries
            or args.hidden_width is not None or args.queries_per_user is not None):
        raise ValueError("--centered-logit-refinement supports only Item-KV64 warm logit8 without other overrides")
    if args.recent_only_queries and (args.method != "kv_item" or args.initial_calibration is not None
            or args.mixed_refinement or args.producer_scale_refinement or args.rolling_scenes
            or args.hidden_width is not None or args.queries_per_user is not None or args.refinement_epochs is not None):
        raise ValueError("--recent-only-queries supports only fresh Item-KV64 with the original sixteen-query budget")
    paired_refinement = args.mixed_refinement or args.producer_scale_refinement
    if args.producer_scale_refinement and (args.method != "kv_item" or args.initial_calibration is None
            or args.refinement_objective != "logit" or args.rolling_scenes or args.refinement_epochs is not None
            or args.mixed_refinement):
        raise ValueError("--producer-scale-refinement requires Item-KV64 warm logit8 only, without other scene/epoch overrides")
    if args.mixed_refinement and (args.method != "kv_item" or args.initial_calibration is None
            or args.refinement_objective != "logit" or args.rolling_scenes or args.refinement_epochs is not None):
        raise ValueError("--mixed-refinement requires Item-KV64 warm logit8, without rolling-scenes or epoch override")
    if (args.hidden_width is not None or args.queries_per_user is not None) and (
            args.method != "kv_view" or args.rolling_scenes or args.initial_calibration is not None):
        raise ValueError("width/query overrides are only supported by fresh pure-Parent kv_view fitting")
    queries = stage["calibration_queries_per_user"] if args.queries_per_user is None else args.queries_per_user
    if args.method == "kv_item" and (args.rolling_scenes or queries != 16):
        raise ValueError("kv_item uses pure-Parent fitting with sixteen queries per UID")
    if args.refinement_objective == "response" and (args.method != "kv_item" or args.initial_calibration is None):
        raise ValueError("response refinement requires the fixed Item-KV64 warm start")
    if args.refinement_epochs is not None and (args.method != "kv_item" or args.initial_calibration is None
                                               or args.refinement_objective != "logit"):
        raise ValueError("--refinement-epochs is only supported by Item-KV64 warm-start logit refinement")
    if args.method == "kv_context" and not args.rolling_scenes:
        raise ValueError("kv_context requires the matched four rolling calibration scenes")
    if args.rolling_scenes and (args.method == "nonlinear_response" or args.initial_calibration is not None):
        raise ValueError("rolling scenes use fresh KV32/context-KV32 response fitting only")
    selected_settings = None
    hidden_width, compact_training = args.hidden_width, args.compact_training
    if args.initial_calibration is None and args.method in ("kv_item", "nonlinear_response"):
        selected_settings = select_release_config(
            release_id="medium/v4_to_v5", stage=args.method,
            fixed=CalibrationSettings(WIDTHS[args.method], epochs, compact_training))
        hidden_width = selected_settings.hidden_width
        epochs = selected_settings.epochs
        compact_training = selected_settings.compact_training
    initial = None
    item_initial = (initial_calibration(args.item_read_view, args.config, config, versions, "kv_item")
                    if args.item_read_view is not None else None)
    if args.initial_calibration is not None:
        if args.method not in ("kv_view", "kv_item"):
            raise ValueError("--initial-calibration supports only the fixed KV32 or Item-KV64 refinement")
        initial = initial_calibration(args.initial_calibration, args.config, config, versions, args.method)
    if (args.output/"calibration.pt").exists() or (args.output/"calibration.json").exists():
        raise FileExistsError("calibration output already exists; choose a new directory")
    torch.set_num_threads(args.threads)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.70, device)
        torch.cuda.reset_peak_memory_stats(device)
    costs, timings = defaultdict(int), defaultdict(float)
    if item_initial is not None:
        costs["inherited_item_calibration_flops"] = item_initial[2]["calibration_flops"]
    if initial is not None:
        costs["inherited_calibration_flops"] = initial[2]["calibration_flops"]
    parent, pp = timed(lambda: load_model(ROOT/versions["v4"]["checkpoint"], device), timings, "model_load", device)
    current, cp = timed(lambda: load_model(ROOT/versions["v5"]["checkpoint"], device), timings, "model_load", device)
    model_config = pp["config"]
    expected = dict(num_layers=6, hidden_size=192, num_heads=6, max_seq_len=1024,
                    activation="elu_plus1", block_variant="legacy", relative_position_bias=False,
                    gating="silu_gate", causal_diagonal="inclusive")
    if cp["config"] != model_config or any(model_config[key] != value for key, value in expected.items()):
        raise RuntimeError("checkpoint architecture differs from frozen Medium read operator")
    parent.eval().requires_grad_(False)
    current.eval().requires_grad_(False)
    dataset_path = ROOT/config["data"]["dataset"]["path"]
    known = int(cp.get("known_vocab_size", json.loads(dataset_path.read_text())["foundation_items"]))
    del pp, cp
    uids, cutover, maximum = train+validation, stage["days_half_open"][0]*DAY, model_config["max_seq_len"]
    history = timed(lambda: load_histories(uids, dataset_path=dataset_path, known_vocab_size=known,
        oov_buckets=model_config["num_items"]-known, start_timestamp=cutover, end_timestamp=cutover+1,
        max_history=maximum, threads=args.history_threads), timings, "history_io", device)
    histories = {uid: snapshot(history, uid, cutover, maximum) for uid in uids}
    del history
    latest_timestamp = max(int(histories[uid][0][-1]) for uid in uids)
    rows = timed(lambda: capture(parent, current, histories, uids, cutover=cutover, known=known,
        queries=queries, device=device, batch_size=args.batch_size,
        history_length=maximum, attention_backend=args.attention_backend), timings, "parent_and_teacher_capture", device)
    for uid in uids:
        rows[uid]["candidates"] = torch.from_numpy(mixed_candidates(uid, histories[uid][1], known,
            queries, seed=17, recent_budget=16 if args.recent_only_queries else None))
        if (args.method == "kv_item" or item_initial is not None) and not paired_refinement:
            ids = torch.as_tensor(histories[uid][1], dtype=torch.long, device=device)[None]
            if ids.shape[1] != rows[uid]["parent"].seq_len:
                raise RuntimeError("item features must align with every captured history row")
            with torch.no_grad():
                rows[uid]["item_features"] = current.lookup_item_embeddings(ids).detach().cpu()
    models = {name: CostModel.for_scale("medium", name) for name in ("torch", "triton")}
    captured = sum(models[args.attention_backend if rows[batch[0]]["parent"].seq_len == maximum else "torch"].full_cache(
        rows[batch[0]]["parent"].seq_len, batch=len(batch)) for batch in groups(uids, rows, args.batch_size))
    costs["parent_cache_capture_flops"] = captured
    costs["teacher_cache_capture_flops"] = captured
    history_histogram = dict(Counter(rows[uid]["parent"].seq_len for uid in uids))
    if item_initial is not None:
        timed(lambda: prepare_item_read_view(item_initial[0], rows, uids, batch_size=args.batch_size,
            device=device, costs=costs), timings, "frozen_item_read_view", device)
    train_rows, validation_rows, scene_records = train, validation, None
    mixed_train, mixed_validation = None, None
    if args.rolling_scenes or paired_refinement:
        from design_one.rolling_calibration import build_rolling_scenes
        expanded = timed(lambda: load_histories(uids, dataset_path=dataset_path, known_vocab_size=known,
            oov_buckets=model_config["num_items"]-known, start_timestamp=cutover, end_timestamp=cutover+1,
            max_history=maximum+768, threads=args.history_threads), timings, "expanded_history_io", device)
        expanded_histories = {uid: snapshot(expanded, uid, cutover, maximum+768) for uid in uids}
        del expanded
        rows, scene_records = timed(lambda: build_rolling_scenes(parent, current, rows, histories,
            expanded_histories, uids, cutover=cutover, batch_size=args.batch_size, device=device,
            max_length=maximum, attention_backend=args.attention_backend, costs=costs,
            **(dict(append_targets=(0, 768), full_queries=True) if paired_refinement else {})),
            timings, "rolling_scene_capture", device)
        train_users, validation_users = set(train), set(validation)
        train_rows = [key for key, row in rows.items() if row["uid"] in train_users]
        validation_rows = [key for key, row in rows.items() if row["uid"] in validation_users]
        if paired_refinement:
            train_rows, validation_rows = [[f"{uid}:a0" for uid in group] for group in (train, validation)]
            mixed_train, mixed_validation = [[f"{uid}:a768" for uid in group] for group in (train, validation)]
            attach_scene_item_features(current, rows, device)
            if args.producer_scale_refinement:
                for row in rows.values():
                    row["producer_mask"] = torch.tensor(row.get("producer", [4]*row["parent"].seq_len))[None] == 5
        del expanded_histories
    del parent, histories
    gc.collect()
    if initial is None:
        adapter, diagnostics = fit(current, rows, train_rows, validation_rows, args.method, batch_size=args.batch_size,
            device=device, epochs=epochs, costs=costs, timings=timings, hidden_width=hidden_width,
            compact_training=compact_training)
    elif args.refinement_objective == "response":
        adapter, diagnostics = fit(current, rows, train, validation, args.method, batch_size=args.batch_size,
            device=device, epochs=epochs, costs=costs, timings=timings, adapter=initial[0],
            residual_rate_rms=initial[2]["residual_rate_rms"], learning_rate=.0001)
    else:
        starting_adapter = initial[0]
        if args.producer_scale_refinement:
            from hstu_kvcache.design_one.producer_scaled_item import ProducerScaledItemKVAdapter
            starting_adapter = ProducerScaledItemKVAdapter.from_item_adapter(starting_adapter)
        adapter, diagnostics = refine_logits(current, starting_adapter, rows, train_rows, validation_rows, batch_size=args.batch_size,
            device=device, epochs=epochs, costs=costs, timings=timings, method=args.method,
            mixed_train=mixed_train, mixed_validation=mixed_validation, producer_scale=args.producer_scale_refinement,
            centered=args.centered_logit_refinement)
    provenance = source_hashes()
    provenance[str(Path(__file__).relative_to(ROOT))] = sha256_file(Path(__file__))
    if selected_settings is not None:
        helper = ROOT/"scripts/design_one/release_config.py"
        provenance[str(helper.relative_to(ROOT))] = sha256_file(helper)
    if args.rolling_scenes or paired_refinement:
        helper = ROOT/"scripts/design_one/rolling_calibration.py"
        provenance[str(helper.relative_to(ROOT))] = sha256_file(helper)
    distinct_caches = {id(row[key]): row[key] for row in rows.values() for key in ("parent", "teacher")}
    metadata = dict(status="complete", kind=adapter.kind, method=args.method, scale="medium", edge="v4_to_v5",
        role="development_canary" if args.small else "development_calibration",
        configuration=dict(path=str(args.config), sha256=sha256_file(args.config)),
        users_file_sha256=config["users"]["sha256"], uids=train, fit_uids=train, validation_uids=validation,
        users=len(train), budget=len(train), queries_per_user=queries,
        evaluation_users_excluded=config["users"]["groups"][stage["evaluation_group"]]["count"],
        cutover=cutover, latest_history_timestamp=latest_timestamp, history_length=maximum,
        history_length_histogram=history_histogram,
        cache_state="pure V4 strictly before release; serving applies the rule to all retained entries including native Current writes",
        candidate_rule=f"{queries//2} most recent unique known items; remaining uniform known catalog without replacement; SeedSequence([17,uid]); no feedback labels",
        history_tie_order="timestamp_mapped_item_behavior, aligned Parent and Current Full histories",
        teacher="Current Full same-window cache; each layer reads at the actual query from its frozen fitted lower layers",
        fitting="mean per-user rate residual MSE weighted by (N/1024)^2, divided by one fitting-only global residual-rate RMS squared; equivalent aggregate-response objective up to a constant",
        normalization="input means/scales fitted only on fitting users; response output uses one global rate RMS; KV output uses source coordinate std; no coordinate-dependent loss weights",
        validation_rule="fixed endpoint; separate validation users report response and Full-logit error only, with no checkpoint selection",
        kv_supervision="same-query read distillation only; no token-KV target loss or warm start",
        checkpoint_hashes={name: versions[name]["checkpoint_sha256"] for name in ("v4", "v5")},
        checkpoint_verification="sealed manifest and seal bindings checked; large payload hashes inherited",
        model_config=model_config, adapter_config=adapter.get_config(), execution_sources=provenance, diagnostics=diagnostics,
        settings=dict(seed=17, initialization="one global torch.manual_seed(17) before constructing all layers",
            shuffle_seed="17+layer", hidden_width=adapter.get_config()["hidden_width"], epochs=epochs, learning_rate=.001,
            weight_decay=.0001, gradient_clip=1., batch_size=args.batch_size,
            response_training_batch="all fitting users", attention_backend=args.attention_backend,
            device=str(device), gpu_memory_fraction=.70, torch_threads=args.threads, history_threads=args.history_threads),
        cost={**dict(costs), "calibration_flops": sum(costs.values()),
            "teacher_flops": costs["teacher_cache_capture_flops"]+costs["teacher_same_query_flops"]+costs["teacher_full_query_flops"]+costs.get("rolling_teacher_cache_flops", 0),
            "scope": "both source/teacher captures, actual-query traces and prefix corrections, teacher reads, normalization, all optimizer steps, initial/final training and held-out response/logit diagnostics",
            "convention": "multiply-add=2; dense executed shapes including padding; network forward/backward use core arithmetic helpers with frozen inputs; mapped-read forward+backward conservatively3x forward; AdamW+gradient clipping estimated18 operations/parameter/step; not measured hardware instructions"},
        timings_seconds=dict(timings), elapsed_seconds=time.perf_counter()-began,
        paired_cache_bytes=sum(cache.k.nbytes+cache.v.nbytes for cache in distinct_caches.values()),
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1<<20),
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/(1<<30) if device.type == "cuda" else 0)
    if selected_settings is not None:
        metadata["release_configuration"] = dict(
            release_id="medium/v4_to_v5", stage=args.method, **asdict(selected_settings))
    if args.method == "kv_item":
        metadata.update(input_width=3*model_config["hidden_size"],
            item_feature_source=dict(model="Current V5", parameter="item_emb",
                checkpoint_sha256=versions["v5"]["checkpoint_sha256"],
                scope="retained pre-release history IDs; one lookup per UID, reused across all layers; frozen parameters",
                lookup_tokens=sum(row["parent"].seq_len for row in rows.values()),
                lookup_bytes=sum(row["item_features"].nbytes for row in rows.values()),
                arithmetic_flops=0, history_attention=False),
            normalization="fit-only valid-token [K,V,Current item embedding] means/scales; residual output units use only source K/V std; padding excluded")
    if args.recent_only_queries:
        metadata.update(candidate_rule="16 most recent unique known items from the pre-release history in reverse event order; short histories filled uniformly without replacement using SeedSequence([17,uid]); no labels or future events",
            calibration_overrides=dict(recent_only_queries=True, recent_budget=16, default_recent_budget=8,
                queries_per_user=16, scope="fresh Item-KV64; frozen configuration hash and UID split retained"))
        metadata["settings"]["recent_only_queries"] = True
        metadata["cost"]["candidate_selection"] = "CPU integer ID filtering and RNG; excluded from arithmetic FLOPs under the existing convention"
    if args.hidden_width is not None or args.queries_per_user is not None:
        metadata["calibration_overrides"] = dict(hidden_width=args.hidden_width,
            queries_per_user=args.queries_per_user, default_hidden_width=WIDTHS[args.method],
            configured_queries_per_user=stage["calibration_queries_per_user"],
            scope="fresh pure-Parent KV fitting; frozen configuration hash and UID split retained")
    if args.epochs is not None:
        metadata["training_epoch_override"] = dict(requested=args.epochs, default=EPOCHS[args.method],
            effective=epochs, scope="fresh pure fitting stage; frozen configuration hash, UID split and queries retained")
    if compact_training:
        metadata["settings"]["compact_training"] = True
        metadata["compact_training"] = dict(
            scope="Item layer fitting and initial/final training diagnostics only; captures, corrected-prefix traces and held-out validation retain the materialized reference",
            execution="SiLU hidden64; live affine output folded through corrected QK and weighted V; activation after complete corrected QK; original parameters and AdamW settings",
            folding="every prediction rebuilds physical-unit output weights/biases with autograd; no detached or reused decoder during optimization",
            cost_symbols="B=batch, N=padded history, Q=queries, F=3W, M=hidden width, W=total head width, H=heads",
            encoder_forward="BN*(2F+2FM+2M)", encoder_backward="BN*(2FM+2M)",
            read_forward="BQ*(N*(4W+7H+4HM)+4WM+7W)", read_backward="2*read_forward (conservative)",
            live_fold_each_direction="2W*(M+1) per batch; forward and backward included in network_and_read_training_flops_estimate",
            numerical_scope="same real-valued function and gradients; floating-point reassociation can change the fresh optimization trajectory")
        metadata["cost"]["convention"] += "; compact Item training uses the declared encoder/read/live-fold formulas, including full read backward; loss/optimizer/clipping unchanged"
    if args.rolling_scenes:
        metadata.update(fit_scene_ids=train_rows, validation_scene_ids=validation_rows,
            cache_state="four pre-release scenes per UID: pure Parent and actual Current appends with rolling evictions",
            fitting="mean per-scene rate residual MSE weighted by (N/1024)^2, divided by one fit-only global residual-rate RMS squared; four equal-query scenes per UID",
            scene_protocol=dict(append_targets=[0, 128, 384, 768], fit_scenes=len(train_rows),
                validation_scenes=len(validation_rows), queries_per_scene=4, total_queries_per_uid=16,
                candidate_partition="original sixteen candidates split by state_index::4, no repeated query across scenes",
                pre_release_history_limit=maximum+768, scenes=scene_records),
            context_features=["is_parent", "old_fraction_at_write"] if args.method == "kv_context" else [],
            input_width=2*model_config["hidden_size"]+(2 if args.method == "kv_context" else 0),
            context_rule="immutable fields fixed when each retained entry was produced; pure Parent entries are [1,1]")
        metadata["settings"]["rolling_scenes"] = True
        metadata["cost"]["scope"] += "; four-scene Parent initializations, native appends, immutable context and aligned Full teacher captures all charged"
    if initial is not None:
        _, inherited, binding = initial
        label = "Item-KV64" if args.method == "kv_item" else "KV32"
        metadata.update(initial_calibration=binding, fitting_stage="joint_full_logit_refinement",
            uids=inherited["fit_uids"], fit_uids=inherited["fit_uids"], users=len(inherited["fit_uids"]),
            budget=len(inherited["fit_uids"]), validation_uids=inherited["validation_uids"],
            refinement_fit_uids=train, refinement_validation_uids=validation,
            capture_uids=uids, refinement_history_length_histogram=metadata["history_length_histogram"],
            teacher="actual Current Full final logits from the same pre-release history and sixteen candidates; targets computed once per selected UID",
            fitting=f"joint mean squared final-logit error across all six {label} networks, divided by one fit-only Full-minus-native logit RMS squared; no response auxiliary loss",
            normalization="all network input/output buffers inherited unchanged; one fitting-only logit RMS supplies a constant loss unit",
            kv_supervision=f"warm start from response-distilled {label}; refinement uses only final Full logits, no token-KV loss")
        metadata["settings"].update(initialization="exact initial-calibration checkpoint; no random reinitialization",
            shuffle_seed=17, learning_rate=.0001, response_training_batch=None,
            gradient_weighting="each equal-length batch loss multiplied by batch_users/configured_batch_size")
        metadata["settings"]["refinement_objective"] = args.refinement_objective
        if args.refinement_epochs is not None:
            metadata["refinement_epoch_override"] = dict(requested=args.refinement_epochs,
                default=8, effective=epochs, scope="Item-KV64 warm-start logit refinement")
        metadata["cost"]["refinement_flops"] = metadata["cost"]["calibration_flops"]-binding["calibration_flops"]
        metadata["cost"]["teacher_flops"] += binding["teacher_flops"]
        metadata["cost"]["scope"] = "initial calibration charged once plus every new source/teacher capture, Full/native target score, complete differentiable query path, all optimizer steps and train/held-out diagnostics"
        if args.refinement_objective == "logit":
            metadata["cost"]["convention"] = "multiply-add=2; complete query forward includes native reads, mapped KV and additional mapped reads; joint forward+backward conservatively3x complete forward including frozen foundation projections; AdamW+gradient clip estimated18 operations/parameter/step"
        else:
            metadata.update(fitting_stage="layerwise_response_refinement",
                teacher="Current Full same-window reads at actual queries from already refined lower layers; targets recaptured per layer",
                fitting="original mean per-user rate residual MSE weighted by (N/1024)^2, divided by the inherited per-layer residual-rate RMS squared",
                normalization="all network input/output buffers and per-layer residual_rate_rms inherited unchanged from the bound initial calibration",
                kv_supervision="warm start from response-distilled Item-KV64; same-query response refinement only, no token-KV or logit training loss")
            metadata["settings"].update(shuffle_seed="17+layer", response_training_batch=args.batch_size,
                gradient_weighting="original mean per-user response objective within each minibatch")
            metadata["cost"]["scope"] = "initial calibration charged once plus every new source/teacher capture, actual-query trace and corrected prefix, same-query teacher target, all layerwise optimizer steps and train/held-out diagnostics; inherited normalization incurs no new fitting statistics"
    if paired_refinement:
        metadata.update(fitting_stage="alternating_pure_mixed_logit_refinement",
            fit_scene_ids=train_rows+mixed_train, validation_scene_ids=validation_rows+mixed_validation,
            scene_protocol=dict(append_targets=[0, 768], queries_per_scene=16, queries_per_uid_per_epoch=16,
                training_rule="odd epochs pure, even epochs a768; each epoch contains one scene per fitting UID",
                pre_release_history_limit=maximum+768, scenes=scene_records),
            cache_state="paired pre-release pure Parent and real Current append768 states with rolling eviction",
            teacher="Full/native logits computed once per actual scene terminal and reused across epochs and validation",
            normalization="network input/output buffers inherited unchanged; single logit RMS uses pure fitting users only",
            fitting=f"{epochs} joint logit epochs alternating pure/mixed, {epochs//2} each, same UID/query/batch budget per epoch")
        metadata["settings"]["mixed_refinement"] = True
        metadata["item_feature_source"].update(scope="each scene actual terminal item IDs; one lookup per scene, reused across all layers",
            lookup_bytes_by_scene={kind: sum(row["item_features"].nbytes for row in rows.values() if row["scene_kind"] == kind)
                                   for kind in ("a0", "a768")})
        metadata["cost"]["scope"] += "; both scene captures/native replay/context and actual terminal teachers, both target sets and pure/mixed diagnostics; one scene per UID per epoch"
    if args.producer_scale_refinement:
        metadata.update(fitting_stage="native_producer_scale_refinement",
            fit_scene_ids=mixed_train, normalization_scene_ids=train_rows,
            producer_scale_refinement=dict(initial_alpha=[1.]*len(adapter.layers),
                final_alpha=[float(layer.native_scale.detach()) for layer in adapter.layers], trainable_parameters=len(adapter.layers),
                frozen_base=True, training_scene="a768", loss="joint_full_logit", alpha_constraint="none"),
            fitting=f"{epochs} mixed-only joint logit epochs; only one native-residual scalar per layer is trained; original Parent mapping fixed",
            normalization="original Round11 network and all normalization buffers frozen; one pure-fit logit RMS")
        metadata["settings"].update(learning_rate=.01, weight_decay=0., mixed_refinement=False, producer_scale_refinement=True)
        metadata["scene_protocol"]["training_rule"] = "every epoch a768 only; pure scenes supply the fixed RMS and diagnostics"
    if args.centered_logit_refinement:
        metadata.update(fitting_stage="centered_joint_logit_refinement",
            fitting="mean squared logit residual after subtracting its mean over every user/query logit in the actual minibatch, divided by pure-fit Full-minus-native RMS squared",
            centered_logit_refinement=dict(batch_scope="all user/query logits in an actual equal-history-length minibatch, not per-user",
                configured_batch_users=args.batch_size, queries_per_user=16,
                grouping="existing equal-history-length buckets, each at most8users; shuffled minibatch membership per epoch",
                gradient_weighting="unchanged actual_batch_users/configured_batch_size",
                diagnostic_scope="uncentered Full MSE retained; centered initial/final diagnostics use fixed UID-order buckets"))
        metadata["settings"]["centered_logit_refinement"] = True
        metadata["cost"]["convention"] += "; centering mean/subtract forward+backward estimated4 operations/logit; no new model/teacher passes"
    if item_initial is not None:
        from hstu_kvcache.design_one.composite import FrozenItemResponseAdapter
        frozen_item, inherited, binding = item_initial
        # The base named native by the shared fitter is the mapped Item view.
        validation_stats = metadata["diagnostics"]["validation"]
        validation_stats["item_baseline_full_logit_mse"] = validation_stats.pop("native_full_logit_mse")
        validation_stats["relative_to_item_baseline_full_logit_mse"] = validation_stats.pop("relative_full_logit_mse")
        adapter = FrozenItemResponseAdapter(frozen_item, adapter)
        metadata.update(kind=adapter.kind, method="frozen_item_response", adapter_config=adapter.get_config(),
            initial_item_calibration=binding, fitting_stage="response_on_frozen_item_view",
            uids=inherited["fit_uids"], fit_uids=inherited["fit_uids"], validation_uids=inherited["validation_uids"],
            users=len(inherited["fit_uids"]), budget=len(inherited["fit_uids"]),
            response_fit_uids=train, response_validation_uids=validation, capture_uids=uids,
            input_width=3*model_config["hidden_size"],
            item_feature_source=dict(model="Current V5", parameter="item_emb",
                checkpoint_sha256=versions["v5"]["checkpoint_sha256"],
                scope="retained pre-release history IDs, one lookup per captured UID; frozen parameters",
                lookup_tokens=sum(row["parent"].seq_len for row in rows.values()),
                lookup_bytes=sum(row["item_features"].nbytes for row in rows.values()),
                arithmetic_flops=0, history_attention=False),
            cache_state="original Parent cache mapped once by the bound frozen Item64; response fitting reads that mapped view",
            teacher="Current Full same-window reads at actual queries from frozen Item64 plus previously fitted lower response layers",
            fitting="fresh Response256 predicts (Full read minus Item-mapped read)/N at actual corrected queries; unchanged aggregate-response objective",
            diagnostic_baseline="Item-mapped reader, not Reuse; no extra Reuse diagnostic pass",
            normalization="fresh response coordinates and residual-rate RMS from response fitting UIDs only; all inherited Item network/norm buffers frozen")
        metadata["settings"]["item_read_view"] = str(args.item_read_view)
        metadata["cost"]["teacher_flops"] += binding["teacher_flops"]
        metadata["cost"]["scope"] = "original Item calibration once plus both new source/teacher captures, once-per-UID frozen Item mapping, all actual mapped-query/teacher reads, response fitting and diagnostics"
        component_source = Path(sys.modules[FrozenItemResponseAdapter.__module__].__file__).resolve()
        provenance[str(component_source.relative_to(ROOT))] = sha256_file(component_source)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=metadata["kind"], adapter=adapter.export_state(), metadata=metadata), args.output/"calibration.pt")
    metadata["weights_sha256"] = sha256_file(args.output/"calibration.pt")
    (args.output/"calibration.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if args.producer_scale_refinement:
        save_fixed_layer0_control(args.output/"fixed_layer0_control", initial, provenance)
    print(json.dumps(dict(status="calibration_complete", output=str(args.output),
        elapsed_seconds=metadata["elapsed_seconds"], calibration_flops=metadata["cost"]["calibration_flops"])), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=KINDS, required=True)
    parser.add_argument("--initial-calibration", type=Path,
                        help="fixed KV32 logit refinement for16epochs, or Item-KV64 refinement for8epochs")
    parser.add_argument("--refinement-objective", choices=("logit", "response"), default="logit",
                        help="warm-start objective; response is Item-KV64 only and keeps inherited normalization/loss units")
    parser.add_argument("--refinement-epochs", type=int, choices=(8, 12),
                        help="Item-KV64 warm-start logit only; default8, --small always2")
    parser.add_argument("--mixed-refinement", action="store_true",
                        help="Item-KV64 warm logit8: alternate pure/a768 epochs with16queries per UID per epoch")
    parser.add_argument("--producer-scale-refinement", action="store_true",
                        help="freeze Round11 Item64 and fit only six Current-residual scalars on a768 for8epochs")
    parser.add_argument("--recent-only-queries", action="store_true",
                        help="fresh Item-KV64 only: sixteen recent unique known history items, uniform fill if short")
    parser.add_argument("--centered-logit-refinement", action="store_true",
                        help="Item-KV64 warm logit8 only: remove one residual mean over all logits in each minibatch")
    parser.add_argument("--item-read-view", type=Path,
                        help="fresh Response256 on a bound frozen Item64 mapped history view")
    parser.add_argument("--epochs", type=int, choices=(32, 48, 64, 100, 200),
                        help="fresh pure Item64:32/48/64, Response256:100/200; defaults64/200, --small always2")
    parser.add_argument("--compact-training", action="store_true",
                        help="fresh Item64 only: differentiate factored hidden-state reads, original64epochs; --small uses2")
    parser.add_argument("--rolling-scenes", action="store_true",
                        help="four matched pre-release rolling states, four disjoint queries each; fresh KV fitting for32epochs")
    parser.add_argument("--hidden-width", type=int, choices=(32, 64),
                        help="fresh pure-Parent kv_view only; default uses the original method width")
    parser.add_argument("--queries-per-user", type=int, choices=(16, 64),
                        help="fresh pure-Parent kv_view only; explicit budget override of the frozen configuration")
    parser.add_argument("--gpu", type=int, choices=range(4))
    parser.add_argument("--small", action="store_true", help="fixed first4 fitting/2 validation users, two training epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--history-threads", type=int, default=8)
    parser.add_argument("--attention-backend", choices=("torch", "triton"), default="triton")
    run(parser.parse_args())

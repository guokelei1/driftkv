"""Fit the stronger history branch, caching frozen lower-layer query states."""
from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.read_correction import score_corrected
from hstu_kvcache.read_correction.history_conditioned.v2 import HistoryCorrectionV2
from read_correction_2026_09.calibrate import groups, stacked
from read_correction_2026_09.cost import correction_forward as base_forward
from read_correction_2026_09.v2.cost import (
    CostModel, correction_forward, eager_read, history_encoder_forward, teacher_history_read,
)


def project_context(block, hidden, keys, values):
    """Native frozen query operations that do not depend on this layer's repair."""
    normalized = block.norm(hidden)
    query, new_k, new_v = block.attn._project(normalized)
    native = history_read(block.attn, query, keys, values)
    self_heads = torch.zeros_like(native)
    if block.attn.causal_diagonal == "inclusive":
        weight = block.attn._activate((query * new_k).sum(-1, keepdim=True) * block.attn.scale)
        if block.attn.block_variant == "hstu_reference":
            weight = weight / block.attn.cfg.max_seq_len
        self_heads = weight * new_v
    return {"query": query, "native": native, "self_heads": self_heads,
            "gate": F.silu(block.gate_proj(normalized)), "residual": hidden}


def apply_context(block, context, keys, values, counts, module):
    """Finish precisely one native block after adding its fitted correction."""
    query = context["query"]
    heads = context["native"] + module(query, keys, values, counts)
    heads = heads + context["self_heads"]
    attention = block.attn._finish(heads)
    if block.block_variant == "hstu_reference":
        update = block.attn.out_proj(block.attn_output_norm(attention) * context["gate"])
    else:
        if block.gating != "silu_gate":
            raise ValueError("unexpected frozen-model gate")
        update = attention * context["gate"]
    return context["residual"] + update


def padded_history(rows, uids, layer, key):
    counts = torch.tensor([rows[uid][key].seq_len for uid in uids], dtype=torch.long)
    width = rows[uids[0]][key].k.shape[-1]
    keys = torch.zeros(len(uids), int(counts.max()), width)
    values = torch.zeros_like(keys)
    for index, uid in enumerate(uids):
        length = int(counts[index])
        keys[index, :length] = rows[uid][key].k[layer, 0]
        values[index, :length] = rows[uid][key].v[layer, 0]
    return keys, values, counts


def _batch(tensors, selected, device):
    return tuple(tensor[selected].to(device) for tensor in tensors)


@torch.no_grad()
def _residual_targets(base, queries, targets, keys, values, counts, batch_size, device):
    residuals = []
    for start in range(0, len(queries), batch_size):
        selected = slice(start, start + batch_size)
        q, target, k, v, count = _batch((queries, targets, keys, values, counts), selected, device)
        residuals.append((target - base.rate(q, k, v, count)).cpu())
    return torch.cat(residuals)


@torch.no_grad()
def _assess(module, tensors, selected, units, batch_size, device):
    squared, normalized_squared, elements = 0., 0., 0
    scale = units.to(device)[None, :, None, None]
    for start in range(0, len(selected), batch_size):
        q, target, k, v, count = _batch(tensors, selected[start:start + batch_size], device)
        error = module.residual_rate(q, k, v, count) - target
        squared += float(error.double().square().sum())
        normalized_squared += float((error / scale).double().square().sum())
        elements += error.numel()
    return {"rate_mse": squared / elements, "normalized_rate_mse": normalized_squared / elements}


def _zero_statistics(target, units):
    return {"rate_mse": float(target.double().square().mean()),
            "normalized_rate_mse": float((target / units[None, :, None, None]).double().square().mean())}


@torch.no_grad()
def validation_logits(current, rows, uids, modules, batch_size, device, *, cache_key="parent"):
    collected = {}
    for selected in groups(uids, rows, batch_size):
        cache = stacked(rows, selected, cache_key, device)
        candidates = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        deltas = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        counts = torch.full((len(selected),), cache.seq_len, device=device)
        logits = score_corrected(current, cache, candidates, deltas, modules, counts)[0].cpu()
        for index, uid in enumerate(selected):
            collected[uid] = logits[index]
    return torch.stack([collected[uid] for uid in uids])


def logit_errors(actual, teacher):
    error = actual.double() - teacher.double()
    user_mean = error.mean(1)
    return {"logit_mse": float(error.square().mean()),
            "centered_logit_mse": float((error - user_mean[:, None]).square().mean()),
            "user_mean_logit_mse": float(user_mean.square().mean())}


def fit(current, rows, uids, base_modules, config, device, scale, batch_size, *, validation_uids=None):
    """Return (all layer modules, statistics including additional fitting cost).

    Fitting users set normalization and gradients. Optional, disjoint validation
    users choose the epoch by teacher residual error, including the zero-extra
    starting point. The caller accounts for original v1 fitting and snapshots.
    Only current-layer parameters train; lower fitted states are cached exactly.
    """
    started = time.perf_counter()
    uids, validation_uids = list(uids), list(validation_uids or [])
    if set(uids).intersection(validation_uids):
        raise ValueError("history fitting and validation users must be disjoint")
    all_uids = uids + validation_uids
    if not uids or len(base_modules) != len(current.blocks):
        raise ValueError("nonempty users and one fitted base per layer are required")
    queries = len(rows[uids[0]]["candidates"])
    train_ids = torch.arange(len(uids))
    validation_ids = torch.arange(len(uids), len(all_uids))
    cost = CostModel.for_scale(scale, config["attention_backend"])
    ledger = {"query_embedding_flops": 0, "cached_prefix_native_block_flops": 0,
              "same_query_teacher_flops": 0, "base_residual_target_flops": 0,
              "history_train_flops_estimate": 0, "history_diagnostic_forward_flops": 0,
              "fitted_layer_application_flops": 0, "normalization_loss_optimizer_flops_estimate": 0,
              "final_output_validation_flops": 0}
    output_validation = None
    if validation_uids:
        teacher_logits = validation_logits(current, rows, validation_uids, [None] * len(current.blocks),
                                           batch_size, device, cache_key="teacher")
        base_logits = validation_logits(current, rows, validation_uids, base_modules, batch_size, device)
        output_validation = {"users": len(validation_uids), "uids": validation_uids,
                             "base": logit_errors(base_logits, teacher_logits)}
    hidden_chunks = []
    with torch.no_grad():
        for start in range(0, len(all_uids), batch_size):
            selected = all_uids[start:start + batch_size]
            candidates = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
            delta = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
            hidden_chunks.append(current.embed_query_tokens(candidates, delta).cpu())
            ledger["query_embedding_flops"] += cost.embedding(len(selected) * queries, query=True)
    hidden = torch.cat(hidden_chunks)
    modules, records = [], []
    epochs = int(config["history_epochs"])
    for layer, (block, base) in enumerate(zip(current.blocks, base_modules, strict=True)):
        began = time.perf_counter()
        keys, values, counts = padded_history(rows, all_uids, layer, "parent")
        teacher_k, teacher_v, teacher_counts = padded_history(rows, all_uids, layer, "teacher")
        if not torch.equal(counts, teacher_counts):
            raise ValueError("parent and teacher history lengths differ")
        context_chunks, target_chunks = {}, []
        with torch.no_grad():
            for start in range(0, len(all_uids), batch_size):
                selected = slice(start, start + batch_size)
                x, k, v, tk, tv, count = _batch((hidden, keys, values, teacher_k, teacher_v, counts), selected, device)
                context = project_context(block, x, k, v)
                target = (history_read(block.attn, context["query"], tk, tv) - context["native"]) / count[:, None, None, None]
                for name, tensor in context.items():
                    context_chunks.setdefault(name, []).append(tensor.cpu())
                target_chunks.append(target.cpu())
        context = {name: torch.cat(tensors) for name, tensors in context_chunks.items()}
        targets = torch.cat(target_chunks)
        del teacher_k, teacher_v, context_chunks, target_chunks
        # One complete native block is executed in project_context+apply_context,
        # rather than re-running already frozen lower history encoders.
        n, b = keys.shape[1], len(all_uids)
        ledger["cached_prefix_native_block_flops"] += (cost.block(b * queries, 0, residual_scales=False)
            + teacher_history_read(cost, n, queries=queries, batch=b)
            + b * queries * (4 * cost.hidden_size + 2 * cost.num_heads))
        ledger["same_query_teacher_flops"] += teacher_history_read(cost, n, queries=queries, batch=b)
        torch.manual_seed(config["seed"] + layer)
        module = HistoryCorrectionV2(
            block.attn.num_heads, block.attn.head_dim,
            encoder_width=int(block.attn.num_heads * block.attn.head_dim * config["history_encoder_width_fraction"]),
            attention_heads=config["history_attention_heads"], query_chunk=config["history_query_chunk"],
            base_width=base.width, base_token_chunk=base.token_chunk, base_query_chunk=base.query_chunk,
            freeze_base=True,
        ).to(device)
        module.initialize_history(base)
        residual = _residual_targets(module.base, context["query"], targets, keys, values, counts, batch_size, device)
        units = residual[:len(uids)].double().square().mean((0, 2, 3)).sqrt().clamp_min(1e-12).float()
        module.set_output_scale(units)
        tensors = (context["query"], residual, keys, values, counts)
        ledger["base_residual_target_flops"] += base_forward(base.get_config(), n, queries=queries,
                                                           batch=b, include_scale_add=False)
        ledger["normalization_loss_optimizer_flops_estimate"] += 4 * targets.numel() + 3 * residual[:len(uids)].numel()
        initial_train = _zero_statistics(residual[:len(uids)], units)
        initial_validation = _zero_statistics(residual[len(uids):], units) if validation_uids else None
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
        best_epoch, best_validation = 0, initial_validation
        best_objective = initial_validation["normalized_rate_mse"] if initial_validation else float("inf")
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config["history_learning_rate"],
                                      weight_decay=config["weight_decay"])
        generator = torch.Generator().manual_seed(config["seed"] + layer)
        scale_gpu = units.to(device)[None, :, None, None]
        parameter_count = sum(parameter.numel() for parameter in trainable)
        epoch_records, steps, clipped_steps = [], 0, 0
        for epoch in range(1, epochs + 1):
            order = train_ids[torch.randperm(len(uids), generator=generator)]
            total_loss = 0.
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                q, target, k, v, count = _batch(tensors, selected, device)
                optimizer.zero_grad(set_to_none=True)
                normalized_error = (module.residual_rate(q, k, v, count) - target) / scale_gpu
                loss = normalized_error.square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite H v2 normalized residual loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("nonfinite H v2 gradients")
                clipped_steps += int(float(grad_norm) > 1.)
                optimizer.step()
                total_loss += float(loss.detach()) * len(selected) / len(uids)
                steps += 1
                ledger["history_train_flops_estimate"] += 3 * history_encoder_forward(
                    module.get_config(), n, queries=queries, batch=len(selected))
                ledger["normalization_loss_optimizer_flops_estimate"] += 4 * target.numel() + 20 * parameter_count
            validation = _assess(module, tensors, validation_ids, units, batch_size, device) if validation_uids else None
            if validation:
                ledger["history_diagnostic_forward_flops"] += history_encoder_forward(
                    module.get_config(), n, queries=queries, batch=len(validation_uids))
                ledger["normalization_loss_optimizer_flops_estimate"] += 8 * residual[len(uids):].numel()
                if validation["normalized_rate_mse"] < best_objective:
                    best_objective, best_epoch, best_validation = validation["normalized_rate_mse"], epoch, validation
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
            epoch_records.append({"epoch": epoch, "train_online_normalized_rate_mse": total_loss,
                                  "validation": validation})
        if validation_uids:
            module.load_state_dict(best_state)
        else:
            best_epoch = epochs
        module.eval()
        final_train = _assess(module, tensors, train_ids, units, batch_size, device)
        ledger["history_diagnostic_forward_flops"] += history_encoder_forward(
            module.get_config(), n, queries=queries, batch=len(uids))
        ledger["normalization_loss_optimizer_flops_estimate"] += 8 * residual[:len(uids)].numel()
        next_hidden = []
        with torch.no_grad():
            for start in range(0, len(all_uids), batch_size):
                selected = slice(start, start + batch_size)
                ctx = {name: tensor[selected].to(device) for name, tensor in context.items()}
                k, v, count = _batch((keys, values, counts), selected, device)
                next_hidden.append(apply_context(block, ctx, k, v, count, module).cpu())
        hidden = torch.cat(next_hidden)
        ledger["fitted_layer_application_flops"] += correction_forward(module.get_config(), n, queries=queries, batch=b)
        modules.append(module)
        records.append({"layer": layer, "config": module.get_config(), "initial_train": initial_train,
            "initial_validation": initial_validation, "final_train": final_train,
            "selected_validation": best_validation, "selected_epoch": best_epoch, "epochs": epoch_records,
            "optimizer_steps": steps, "clipped_steps": clipped_steps,
            "trainable_parameters": parameter_count, "output_scale_per_head": units.tolist(),
            "padded_history_length": n, "fit_seconds": time.perf_counter() - began})
        print(json.dumps({"status": "history_v2_layer_fit", "users": len(uids), "layer": layer,
            "layers": len(current.blocks), "selected_epoch": best_epoch, "initial_train": initial_train,
            "final_train": final_train, "validation": best_validation,
            "fit_seconds": records[-1]["fit_seconds"]}), flush=True)
        del optimizer, trainable, context, targets, tensors, keys, values, residual, best_state
    if validation_uids:
        corrected_logits = validation_logits(current, rows, validation_uids, modules, batch_size, device)
        output_validation["v2"] = logit_errors(corrected_logits, teacher_logits)
        for uid in validation_uids:
            length = rows[uid]["parent"].seq_len
            ledger["final_output_validation_flops"] += 3 * eager_read(cost, length, queries=queries)
            ledger["final_output_validation_flops"] += sum(correction_forward(m.get_config(), length, queries=queries)
                for m in [*base_modules, *modules])
        ledger["normalization_loss_optimizer_flops_estimate"] += 20 * teacher_logits.numel()
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = "additional H v2 only; one cached native block per layer, exact same-query teacher targets, heavy backward estimated 2x forward, optimizer/clip estimated 20 operations per parameter per step; original v1 fitting and cache preparation charged by caller"
    return modules, {"layers": records, "train_uids": uids, "validation_uids": validation_uids,
        "validation_scope": "disjoint calibration users; normalization from fitting users only; no evaluation labels",
        "epoch_selection": "minimum validation normalized teacher residual MSE, including zero-extra epoch0" if validation_uids else "fixed configured epochs",
        "optimizer_steps": sum(record["optimizer_steps"] for record in records),
        "final_output_validation": output_validation,
        "epochs": epochs, "cost": ledger, "fit_seconds": time.perf_counter() - started,
        "cached_prefix": "each frozen lower-layer output computed once and copied to CPU; no repeated lower-layer encoder work"}

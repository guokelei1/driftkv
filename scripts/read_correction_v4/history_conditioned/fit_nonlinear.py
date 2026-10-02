"""Draft layerwise KV-and-read fitting of the small nonlinear token residual.

The inherited ridge map is frozen. Lower selected nonlinear corrections define
the actual query for the next layer. All caches are supplied from strictly
pre-release calibration users; the caller adds capture and inherited fit costs.
"""
from __future__ import annotations

import json
import time

import torch

from hstu_kvcache.adaptation.reader import history_read
from read_correction_2026_09.v2.history_conditioned.fit import padded_history, project_context
from read_correction_v4.cost import CostModel, correction_parts, correction_forward, teacher_history_read
from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
from read_correction_v4.cost_nonlinear import nonlinear_forward_flops, training_forward_parts, training_step_parts


def normalized_losses(module, x, base_kv, queries, target_delta, teacher_read,
                      counts, attention, read_units, query_residual=None):
    """KV residual and the native read at the same fixed actual query."""
    positions = torch.arange(x.shape[1], device=x.device)[None]
    valid = positions < counts[:, None]
    residual = module.residual_delta(x)
    kv_error = (residual - target_delta) / module.residual_scale
    kv_loss = (kv_error.square() * valid[..., None]).sum() / (valid.sum() * x.shape[-1])
    mapped = base_kv + residual * valid[..., None]
    k, v = mapped.split(module.heads * module.head_dim, -1)
    prediction = history_read(attention, queries, k, v, count=valid)
    if query_residual is not None:
        prediction = prediction + query_residual
    read_loss = ((prediction - teacher_read) / read_units[None, :, None, None]).square().mean()
    return kv_loss + read_loss, kv_loss, read_loss


def _finish_context(block, context, keys, values, counts, module):
    """Same block finish as the native reader; only its history is replaced."""
    heads = module.forward_new_read(context["query"], keys, values, counts, block.attn)
    heads = heads + context["self_heads"]
    attention = block.attn._finish(heads)
    if block.block_variant != "legacy" or block.gating != "silu_gate":
        raise ValueError("nonlinear token probe supports the frozen legacy blocks")
    return context["residual"] + attention * context["gate"]


def _batch(tensors, selected, device):
    return tuple(value[selected].to(device) for value in tensors)


@torch.no_grad()
def _assess(module, tensors, indices, block, units, batch_size, device):
    result = {"objective": 0., "normalized_kv_mse": 0., "normalized_read_mse": 0.}
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        x, base_kv, q, delta, teacher, counts, query_residual = _batch(tensors, selected, device)
        losses = normalized_losses(module, x, base_kv, q, delta, teacher, counts,
                                   block.attn, units, query_residual)
        for key, loss in zip(result, losses):
            result[key] += float(loss) * len(selected) / len(indices)
    return result


def fit_nonlinear(current, rows, train_uids, validation_uids, base_modules, *,
                  config, device, scale, batch_size=8):
    """Return (selected modules, stats); fit uses no evaluation labels.

    Config: nonlinear_epochs(default6), nonlinear_learning_rate(default.001),
    weight_decay(default.0001), seed, attention_backend; optional
    nonlinear_width overrides the default D/2 for a recorded probe.
    Includes extra per-layer preparation/training/validation arithmetic only.
    The caller charges input snapshot creation and inherited ridge fitting.
    """
    began = time.perf_counter()
    train_uids, validation_uids = list(train_uids), list(validation_uids)
    if not train_uids or not validation_uids or set(train_uids).intersection(validation_uids):
        raise ValueError("nonempty disjoint fitting/validation users are required")
    if len(base_modules) != len(current.blocks):
        raise ValueError("one frozen ridge module per layer is required")
    current.eval().requires_grad_(False)
    all_uids = train_uids + validation_uids
    train_ids, val_ids = torch.arange(len(train_uids)), torch.arange(len(train_uids), len(all_uids))
    queries = len(rows[all_uids[0]]["candidates"])
    cost = CostModel.for_scale(scale, config["attention_backend"])
    ledger = {name: 0 for name in (
        "query_embedding_flops", "cached_prefix_native_block_flops", "same_query_teacher_read_flops",
        "base_temp_kv_and_read_flops", "training_units_flops_estimate", "nonlinear_train_flops_estimate",
        "diagnostic_forward_flops", "optimizer_flops_estimate", "selected_layer_application_flops")}
    hidden_chunks = []
    with torch.no_grad():
        for start in range(0, len(all_uids), batch_size):
            selected = all_uids[start:start + batch_size]
            candidates = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
            delta = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
            hidden_chunks.append(current.embed_query_tokens(candidates, delta).cpu())
            ledger["query_embedding_flops"] += cost.embedding(len(selected) * queries, query=True)
    hidden = torch.cat(hidden_chunks)
    modules, layer_records = [], []
    for layer, (block, base) in enumerate(zip(current.blocks, base_modules, strict=True)):
        started = time.perf_counter()
        keys, values, counts = padded_history(rows, all_uids, layer, "parent")
        tk, tv, tc = padded_history(rows, all_uids, layer, "teacher")
        if not torch.equal(counts, tc):
            raise ValueError("parent and teacher token positions differ")
        n, users, d = keys.shape[1], len(all_uids), keys.shape[-1]
        context_chunks, training_chunks = {}, [[] for _ in range(6)]
        teacher_kv = torch.cat((tk, tv), -1)
        with torch.no_grad():
            for start in range(0, users, batch_size):
                selected = slice(start, start + batch_size)
                x, k, v, teacher_k, teacher_v, count = _batch((hidden, keys, values, tk, tv, counts), selected, device)
                context = project_context(block, x, k, v)
                valid = torch.arange(n, device=device)[None] < count[:, None]
                joined = torch.cat((k, v), -1)
                normalized = (joined - base.input_mean) / base.input_scale
                mapped = joined + base.token_delta(normalized) * valid[..., None]
                teacher = history_read(block.attn, context["query"], teacher_k, teacher_v, count=valid)
                mapped_k, mapped_v = mapped.split(d, -1)
                base_read = history_read(block.attn, context["query"], mapped_k, mapped_v, count=valid)
                query_extra = torch.zeros_like(base_read)
                if base.query_correction is not None:
                    query_extra = base.query_correction(context["query"], k, v, count)
                    base_read = base_read + query_extra
                for name, tensor in context.items():
                    context_chunks.setdefault(name, []).append(tensor.cpu())
                for dest, tensor in zip(training_chunks, (normalized, mapped, teacher, base_read, query_extra,
                                                          torch.cat((teacher_k, teacher_v), -1) - mapped)):
                    dest.append(tensor.cpu())
        context = {key: torch.cat(value) for key, value in context_chunks.items()}
        normalized, mapped, teacher, base_read, query_extra, residual = [torch.cat(value) for value in training_chunks]
        valid_train = torch.arange(n)[None] < counts[:len(train_uids), None]
        residual_rms = residual[:len(train_uids)][valid_train].double().square().mean(0).sqrt()
        teacher_rms = teacher_kv[:len(train_uids)][valid_train].double().square().mean(0).sqrt()
        output_units = torch.maximum(residual_rms, teacher_rms * 1e-3).clamp_min(1e-8).float()
        read_residual = teacher[:len(train_uids)] - base_read[:len(train_uids)]
        read_rms = read_residual.double().square().mean((0, 2, 3)).sqrt()
        teacher_read_rms = teacher[:len(train_uids)].double().square().mean((0, 2, 3)).sqrt()
        read_units = torch.maximum(read_rms, teacher_read_rms * 1e-3).clamp_min(1e-8).float().to(device)
        torch.manual_seed(config["seed"] + layer)
        module = NonlinearTokenReadCorrection(**base.get_config(), nonlinear_width=config.get("nonlinear_width"))
        module.to(device=device, dtype=base.map_weight.dtype).initialize_base(base)
        module.set_residual_scale(output_units)
        tensors = (normalized, mapped, context["query"], residual, teacher, counts, query_extra)
        del tk, tv, teacher_kv, training_chunks, context_chunks
        masked_read = users * queries * n * (4 * d + 4 * block.attn.num_heads)
        ledger["cached_prefix_native_block_flops"] += (cost.block(users * queries, 0, residual_scales=False)
            + teacher_history_read(cost, n, queries=queries, batch=users)
            + users * queries * (4 * d + 2 * block.attn.num_heads))
        ledger["same_query_teacher_read_flops"] += masked_read
        parts = correction_parts(base.get_config(), n, queries=queries, batch=users)
        ledger["base_temp_kv_and_read_flops"] += sum(parts.values())
        ledger["training_units_flops_estimate"] += (4 * residual[:len(train_uids)].numel()
            + 4 * teacher[:len(train_uids)].numel() + 2 * residual.numel())
        optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
            lr=config.get("nonlinear_learning_rate", .001), weight_decay=config.get("weight_decay", .0001))
        rng = torch.Generator().manual_seed(config["seed"] + layer)
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        # Includes explicit residual units; cached normalization/ridge are free
        # only here because their one actual construction is charged above.
        def diagnostic_cost(number):
            return sum(sum(training_forward_parts(module.get_config(), n, queries=queries,
                batch=min(batch_size, number - offset)).values()) for offset in range(0, number, batch_size))
        initial_train = _assess(module, tensors, train_ids, block, read_units, batch_size, device)
        best_validation = _assess(module, tensors, val_ids, block, read_units, batch_size, device)
        ledger["diagnostic_forward_flops"] += diagnostic_cost(len(train_ids)) + diagnostic_cost(len(val_ids))
        best = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
        best_epoch, best_value = 0, best_validation["objective"]
        epochs, steps, clipped = [{"epoch": 0, "train": initial_train, "validation": best_validation}], 0, 0
        for epoch in range(1, int(config.get("nonlinear_epochs", 6)) + 1):
            order = train_ids[torch.randperm(len(train_ids), generator=rng)]
            running = 0.
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                x, bk, q, target_delta, target_read, count, qr = _batch(tensors, selected, device)
                optimizer.zero_grad(set_to_none=True)
                objective, _, _ = normalized_losses(module, x, bk, q, target_delta, target_read,
                                                    count, block.attn, read_units, qr)
                if not torch.isfinite(objective):
                    raise RuntimeError("nonfinite nonlinear KV/read objective")
                objective.backward()
                grad = torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], 1.)
                if not torch.isfinite(grad):
                    raise RuntimeError("nonfinite nonlinear token gradients")
                clipped += int(float(grad) > 1.)
                optimizer.step()
                steps += 1
                running += float(objective.detach()) * len(selected) / len(train_ids)
                step_parts = training_step_parts(module.get_config(), n, queries=queries,
                    batch=len(selected), trainable_parameters=params)
                ledger["optimizer_flops_estimate"] += step_parts.pop("optimizer_estimate_flops")
                ledger["nonlinear_train_flops_estimate"] += sum(step_parts.values())
            validation = _assess(module, tensors, val_ids, block, read_units, batch_size, device)
            ledger["diagnostic_forward_flops"] += diagnostic_cost(len(val_ids))
            epochs.append({"epoch": epoch, "train_online_objective": running, "validation": validation})
            if validation["objective"] < best_value:
                best_value, best_epoch, best_validation = validation["objective"], epoch, validation
                best = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
        module.load_state_dict(best)
        module.eval()
        final_train = _assess(module, tensors, train_ids, block, read_units, batch_size, device)
        ledger["diagnostic_forward_flops"] += diagnostic_cost(len(train_ids))
        new_hidden = []
        with torch.no_grad():
            for start in range(0, users, batch_size):
                selected = slice(start, start + batch_size)
                k, v, count = _batch((keys, values, counts), selected, device)
                part_context = {name: value[selected].to(device) for name, value in context.items()}
                new_hidden.append(_finish_context(block, part_context, k, v, count, module).cpu())
        hidden = torch.cat(new_hidden)
        ledger["selected_layer_application_flops"] += (
            correction_forward(base.get_config(), n, queries=queries, batch=users)
            + nonlinear_forward_flops(module.get_config(), n, batch=users))
        record = {"layer": layer, "selected_epoch": best_epoch, "selected_validation": best_validation,
            "initial_train": initial_train, "final_train": final_train, "epochs": epochs,
            "output_scale_per_coordinate": output_units.tolist(), "read_scale_per_head": read_units.cpu().tolist(),
            "unit_floor": "max(residual RMS, .001 * teacher RMS, 1e-8), training users only",
            "optimizer_steps": steps, "clipped_steps": clipped, "trainable_parameters": params,
            "padded_history_length": n, "fit_seconds": time.perf_counter() - started}
        layer_records.append(record)
        modules.append(module)
        print(json.dumps({"status": "nonlinear_token_layer", "layer": layer,
            "selected_epoch": best_epoch, "validation": best_validation}), flush=True)
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = ("additional nonlinear fitting only; caller adds inherited ridge and snapshot capture; "
        "prefix/native and teacher reads cached once per layer; MLP+mapped-read forward/backward conservatively estimated3x; "
        "actual optimizer steps, epoch diagnostics and final selected applications counted")
    return modules, {"train_uids": train_uids, "validation_uids": validation_uids,
        "layers": layer_records, "cost": ledger, "fit_seconds": time.perf_counter() - began,
        "objective": "training-normalized token KV residual MSE plus native same-actual-query read MSE",
        "selection": "minimum independent calibration-user joint objective per layer, including frozen ridge epoch0",
        "cache_state": "pure parent at prerelease snapshot; mapped state is transient, never persisted"}

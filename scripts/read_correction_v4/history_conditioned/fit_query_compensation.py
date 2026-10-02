"""Draft same-query affine read compensation after freezing token maps."""
from __future__ import annotations

import copy
import json
import time

import torch

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.read_correction.query_only import fit_query
from read_correction_2026_09.cost import correction_forward as query_forward
from read_correction_2026_09.v2.history_conditioned.fit import padded_history, project_context
from read_correction_v4.history_conditioned.fit_nonlinear import _finish_context
from read_correction_v4.cost_nonlinear import (
    CostModel, correction_forward, query_ridge_fit, teacher_history_read,
)


def _batches(tensors, batch_size, device):
    for start in range(0, len(tensors[0]), batch_size):
        yield tuple(value[start:start + batch_size].to(device) for value in tensors)


@torch.no_grad()
def _diagnostics(correction, query, teacher, mapped, counts, batch_size, device):
    before_rate, after_rate, before_read, after_read, elements = 0., 0., 0., 0., 0
    for q, wanted, previous, count in _batches((query, teacher, mapped, counts), batch_size, device):
        delta = correction(q, None, None, count)
        corrected = previous + delta
        old_error, new_error = previous - wanted, corrected - wanted
        divisor = count[:, None, None, None]
        before_read += float(old_error.double().square().sum())
        after_read += float(new_error.double().square().sum())
        before_rate += float((old_error / divisor).double().square().sum())
        after_rate += float((new_error / divisor).double().square().sum())
        elements += old_error.numel()
    return {"before_rate_mse": before_rate / elements, "after_rate_mse": after_rate / elements,
            "before_read_mse": before_read / elements, "after_read_mse": after_read / elements}


@torch.no_grad()
def fit_query_compensation(current, rows, train, validation, modules, config, device,
                           scale, batch_size=8):
    """Fit Q residuals only; validation reports diagnostics without selection.

    Each layer's query comes from the already compensated lower-layer path.
    Token-map parameters are copied and frozen, with no further token fitting.
    The caller adds existing nonlinear fitting and snapshot preparation costs.
    """
    began = time.perf_counter()
    train, validation = list(train), list(validation)
    if not train or not validation or set(train).intersection(validation):
        raise ValueError("nonempty disjoint fitting and validation users required")
    if len(modules) != len(current.blocks):
        raise ValueError("one frozen token module per model layer required")
    current.eval().requires_grad_(False)
    selected_modules = [copy.deepcopy(module).to(device).eval().requires_grad_(False) for module in modules]
    had_query = [module.query_correction is not None for module in selected_modules]
    for module in selected_modules:
        # Fit the complete residual after mapped K/V; do not double-add a prior Q.
        module.query_correction = None
    uids = train + validation
    users, queries = len(uids), len(rows[uids[0]]["candidates"])
    cost = CostModel.for_scale(scale, config["attention_backend"])
    ledger = {name: 0 for name in (
        "query_embedding_flops", "cached_prefix_native_block_flops",
        "frozen_nonlinear_mapped_read_flops", "same_query_teacher_read_flops",
        "residual_rate_target_flops", "query_ridge_fit_flops_estimate",
        "query_compensation_diagnostic_forward_flops", "diagnostic_error_flops_estimate",
        "selected_compensated_layer_application_flops")}
    hidden = []
    for start in range(0, users, batch_size):
        group = uids[start:start + batch_size]
        candidates = torch.stack([rows[uid]["candidates"] for uid in group]).to(device)
        delta = torch.tensor([rows[uid]["query_delta"] for uid in group], device=device)
        hidden.append(current.embed_query_tokens(candidates, delta).cpu())
        ledger["query_embedding_flops"] += cost.embedding(len(group) * queries, query=True)
    hidden = torch.cat(hidden)
    records = []
    for layer, (block, module) in enumerate(zip(current.blocks, selected_modules, strict=True)):
        started = time.perf_counter()
        keys, values, counts = padded_history(rows, uids, layer, "parent")
        teacher_k, teacher_v, teacher_counts = padded_history(rows, uids, layer, "teacher")
        if not torch.equal(counts, teacher_counts) or bool((counts <= 0).any()):
            raise ValueError("parent/teacher require matching nonempty token positions")
        n, width = keys.shape[1], keys.shape[2]
        contexts, targets, mapped_reads, teacher_reads = {}, [], [], []
        for x, k, v, tk, tv, count in _batches((hidden, keys, values, teacher_k, teacher_v, counts), batch_size, device):
            context = project_context(block, x, k, v)
            mapped = module.forward_new_read(context["query"], k, v, count, block.attn)
            valid = torch.arange(n, device=device)[None] < count[:, None]
            teacher = history_read(block.attn, context["query"], tk, tv, count=valid)
            targets.append(((teacher - mapped) / count[:, None, None, None]).cpu())
            mapped_reads.append(mapped.cpu())
            teacher_reads.append(teacher.cpu())
            for name, tensor in context.items():
                contexts.setdefault(name, []).append(tensor.cpu())
        context = {name: torch.cat(parts) for name, parts in contexts.items()}
        targets, mapped, teacher = map(torch.cat, (targets, mapped_reads, teacher_reads))
        ledger["cached_prefix_native_block_flops"] += (cost.block(users * queries, 0, residual_scales=False)
            + teacher_history_read(cost, n, queries=queries, batch=users)
            + users * queries * (4 * width + 2 * block.attn.num_heads))
        ledger["frozen_nonlinear_mapped_read_flops"] += correction_forward(module.get_config(), n, queries=queries, batch=users)
        ledger["same_query_teacher_read_flops"] += users * queries * n * (4 * width + 4 * block.attn.num_heads)
        ledger["residual_rate_target_flops"] += 2 * targets.numel()
        correction, fit_stats = fit_query(context["query"][:len(train)].to(device), targets[:len(train)].to(device),
            counts[:len(train)].to(device), ridge=config.get("query_ridge", config.get("ridge", .01)))
        correction.eval().requires_grad_(False)
        module.query_correction = correction
        ledger["query_ridge_fit_flops_estimate"] += query_ridge_fit(block.attn.num_heads, block.attn.head_dim,
                                                                  len(train) * queries)
        diagnostics = {}
        for name, indices in (("train", slice(0, len(train))), ("validation", slice(len(train), users))):
            diagnostics[name] = _diagnostics(correction, context["query"][indices], teacher[indices],
                                            mapped[indices], counts[indices], batch_size, device)
        ledger["query_compensation_diagnostic_forward_flops"] += query_forward(correction.get_config(), n,
            queries=queries, batch=users, include_scale_add=True)
        # Before/after physical error, normalized error, squares and sums.
        ledger["diagnostic_error_flops_estimate"] += 12 * targets.numel()
        new_hidden = []
        for start in range(0, users, batch_size):
            select = slice(start, start + batch_size)
            part_context = {name: value[select].to(device) for name, value in context.items()}
            new_hidden.append(_finish_context(block, part_context, keys[select].to(device), values[select].to(device),
                                              counts[select].to(device), module).cpu())
        hidden = torch.cat(new_hidden)
        ledger["selected_compensated_layer_application_flops"] += correction_forward(module.get_config(), n,
            queries=queries, batch=users)
        record = {"layer": layer, "query_ridge": fit_stats, "diagnostics": diagnostics,
                  "padded_history_length": n, "replaced_prior_query_affine": had_query[layer],
                  "fit_seconds": time.perf_counter() - started}
        records.append(record)
        print(json.dumps({"status": "post_token_query_compensation", "layer": layer,
                          "diagnostics": diagnostics}), flush=True)
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = ("additional post-token Q fitting only; native context once, frozen NL mapped read once for targets "
        "and again for selected prefix propagation, teacher once, train-only FP64 ridge, cached-read Q diagnostics; "
        "caller adds snapshot preparation and prior token fitting; ridge/diagnostic arithmetic is an analytical estimate")
    return selected_modules, {"train_uids": train, "validation_uids": validation, "layers": records,
        "cost": ledger, "fit_seconds": time.perf_counter() - began,
        "target": "(same-actual-query Current Full history read minus frozen mapped-history read)/history_count",
        "selection": "none; ridge fixed prospectively, validation diagnostics never select or mix outputs",
        "frozen_token_maps": True, "cached_prefix": "one pass through already compensated lower layers"}

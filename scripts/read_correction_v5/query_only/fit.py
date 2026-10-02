"""One pass through the actual corrected query path, with train-only ridge."""
from __future__ import annotations

import json
import time

import torch

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.read_correction_v5.query_only import fit_feature_rule
from read_correction_2026_09.v2.history_conditioned.fit import padded_history, project_context, apply_context
from read_correction_2026_09.cost import CostModel, teacher_history_read
from .cost import correction_forward, feature_ridge_fit


def _batches(tensors, batch_size, device):
    for start in range(0, len(tensors[0]), batch_size):
        yield tuple(value[start:start + batch_size].to(device) for value in tensors)


@torch.no_grad()
def _diagnostics(module, query, target_rate, counts, batch_size, device):
    baseline_rate, fitted_rate, baseline_read, fitted_read, elements = 0., 0., 0., 0., 0
    for q, target, count in _batches((query, target_rate, counts), batch_size, device):
        error = module.rate(q) - target
        factor = count[:, None, None, None]
        baseline_rate += float(target.double().square().sum())
        fitted_rate += float(error.double().square().sum())
        baseline_read += float((target * factor).double().square().sum())
        fitted_read += float((error * factor).double().square().sum())
        elements += target.numel()
    return {"baseline_rate_mse": baseline_rate / elements, "fitted_rate_mse": fitted_rate / elements,
            "baseline_read_mse": baseline_read / elements, "fitted_read_mse": fitted_read / elements}


@torch.no_grad()
def fit_query_features(current, rows, train, val, cfg, device, scale, batch_size=8):
    """Return (modules, stats); costs exclude caller-owned snapshot capture.

    Validation is diagnostic only. There is one prospective feature mode and
    ridge per invocation, with no selection or blending of validation scores.
    """
    began = time.perf_counter()
    train, val = list(train), list(val)
    if not train or not val or set(train).intersection(val):
        raise ValueError("nonempty disjoint fitting and validation users required")
    mode = cfg.get("feature_mode", "head_phi")
    ridge = cfg.get("query_ridge", cfg.get("ridge", .01))
    current.eval().requires_grad_(False)
    uids = train + val
    users, queries = len(uids), len(rows[uids[0]]["candidates"])
    cost = CostModel.for_scale(scale, cfg["attention_backend"])
    ledger = {name: 0 for name in (
        "query_embedding_flops", "cached_prefix_native_block_flops", "same_query_teacher_read_flops",
        "residual_rate_target_flops", "feature_ridge_fit_flops_estimate", "diagnostic_rate_forward_flops",
        "diagnostic_error_flops_estimate", "selected_correction_application_flops")}
    hidden = []
    for start in range(0, users, batch_size):
        group = uids[start:start + batch_size]
        candidates = torch.stack([rows[uid]["candidates"] for uid in group]).to(device)
        delta = torch.tensor([rows[uid]["query_delta"] for uid in group], device=device)
        hidden.append(current.embed_query_tokens(candidates, delta).cpu())
        ledger["query_embedding_flops"] += cost.embedding(len(group) * queries, query=True)
    hidden = torch.cat(hidden)
    modules, records = [], []
    for layer, block in enumerate(current.blocks):
        started = time.perf_counter()
        keys, values, counts = padded_history(rows, uids, layer, "parent")
        teacher_k, teacher_v, teacher_counts = padded_history(rows, uids, layer, "teacher")
        if not torch.equal(counts, teacher_counts) or bool((counts <= 0).any()):
            raise ValueError("parent/teacher require matching nonempty token positions")
        n, width = keys.shape[1:]
        contexts, targets = {}, []
        for x, k, v, tk, tv, count in _batches((hidden, keys, values, teacher_k, teacher_v, counts), batch_size, device):
            context = project_context(block, x, k, v)
            valid = torch.arange(n, device=device)[None] < count[:, None]
            teacher = history_read(block.attn, context["query"], tk, tv, count=valid)
            targets.append(((teacher - context["native"]) / count[:, None, None, None]).cpu())
            for name, tensor in context.items():
                contexts.setdefault(name, []).append(tensor.cpu())
        context = {name: torch.cat(parts) for name, parts in contexts.items()}
        target = torch.cat(targets)
        ledger["cached_prefix_native_block_flops"] += (cost.block(users * queries, 0, residual_scales=False)
            + teacher_history_read(cost, n, queries=queries, batch=users)
            + users * queries * (4 * width + 2 * block.attn.num_heads))
        ledger["same_query_teacher_read_flops"] += users * queries * n * (4 * width + 4 * block.attn.num_heads)
        ledger["residual_rate_target_flops"] += 2 * target.numel()
        module, fit_stats = fit_feature_rule(context["query"][:len(train)].to(device), target[:len(train)].to(device),
            counts[:len(train)].to(device), feature_mode=mode, ridge=ridge)
        module.eval().requires_grad_(False)
        ledger["feature_ridge_fit_flops_estimate"] += feature_ridge_fit(module.get_config(), len(train) * queries)
        diagnostics = {}
        for name, indices in (("train", slice(0, len(train))), ("validation", slice(len(train), users))):
            diagnostics[name] = _diagnostics(module, context["query"][indices], target[indices], counts[indices],
                                             batch_size, device)
        ledger["diagnostic_rate_forward_flops"] += correction_forward(module.get_config(), n,
            queries=queries, batch=users, include_scale_add=False)
        ledger["diagnostic_error_flops_estimate"] += 11 * target.numel()
        new_hidden = []
        for start in range(0, users, batch_size):
            selected = slice(start, start + batch_size)
            part_context = {name: value[selected].to(device) for name, value in context.items()}
            new_hidden.append(apply_context(block, part_context, keys[selected].to(device), values[selected].to(device),
                                            counts[selected].to(device), module).cpu())
        hidden = torch.cat(new_hidden)
        ledger["selected_correction_application_flops"] += correction_forward(module.get_config(), n,
            queries=queries, batch=users)
        records.append({"layer": layer, "fit": fit_stats, "diagnostics": diagnostics,
            "padded_history_length": n, "fit_seconds": time.perf_counter() - started})
        modules.append(module)
        print(json.dumps({"status": "query_feature_fit", "feature_mode": mode, "layer": layer,
                          "diagnostics": diagnostics}), flush=True)
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = ("additional query feature fitting only; one cached native block and one same-query teacher read "
        "per layer; train-only centered FP64 ridge, validation diagnostic forward, and selected correction propagation; "
        "caller adds snapshot capture; solve and scalar diagnostic operations are analytical estimates")
    return modules, {"feature_mode": mode, "train_uids": train, "validation_uids": val,
        "counts_by_uid": {str(uid): int(rows[uid]["parent"].seq_len) for uid in uids},
        "layers": records, "cost": ledger, "fit_seconds": time.perf_counter() - began,
        "target": "(same-actual-query Current Full history read minus reused history read)/history_count",
        "selection": "none; fixed feature mode and ridge, validation diagnostics never select or mix outputs",
        "cached_prefix": "one pass through actual corrected lower layers", "features_read_kv": False}

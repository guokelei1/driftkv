"""Final-output teacher distillation of frozen query-feature basis functions."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
from torch import nn

from hstu_kvcache.read_correction import score_corrected
from read_correction_2026_09.v2.query_only.fit import (
    teacher_logits, loss_statistics, distillation_loss, assess, batch_inputs, groups,
)
from read_correction_2026_09.cost import CostModel, eager_read
from read_correction_v5.query_only.cost import correction_forward


class FeatureOutputUnits(nn.Module):
    """Train slopes/bias in fixed output units; fold units away for serving."""
    def __init__(self, module, units):
        super().__init__()
        self.module = copy.deepcopy(module)
        units = torch.as_tensor(units, device=module.weight.device, dtype=module.weight.dtype)
        if units.numel() != 1 or not bool(torch.isfinite(units).all()) or bool((units <= 0).any()):
            raise ValueError("positive finite scalar output unit required")
        units = units.reshape(())
        self.register_buffer("units", units)
        self.module.requires_grad_(False)
        self.module.weight.requires_grad_(True)
        self.module.bias.requires_grad_(True)
        with torch.no_grad():
            self.module.weight.div_(self.units)
            self.module.bias.div_(self.units)

    def forward(self, q, k, v, counts):
        return (self.module.rate(q) * self.units
                * counts.to(q.dtype)[:, None, None, None])

    def get_config(self):
        return self.module.get_config()

    def folded(self):
        result = copy.deepcopy(self.module)
        with torch.no_grad():
            result.weight.mul_(self.units)
            result.bias.mul_(self.units)
        return result.eval().requires_grad_(False)


def refine_query(current, rows, train, val, modules, cfg, device, scale, batch_size=4):
    """Refine affine heads only; return folded modules and additional-cost stats.

    The fixed teacher has no feedback labels. Training users set objective and
    parameter units; disjoint validation users select epoch, including epoch 0.
    Snapshot capture and initial feature ridge are charged by the caller.
    """
    started = time.perf_counter()
    train, val = list(train), list(val)
    if not train or not val or len(set(train + val)) != len(train + val):
        raise ValueError("unique nonempty disjoint train/validation users required")
    if len(modules) != len(current.blocks):
        raise ValueError("one query correction per model layer required")
    current.eval().requires_grad_(False)
    raw = [copy.deepcopy(module).to(device).eval().requires_grad_(False) for module in modules]
    uids = train + val
    queries = len(rows[uids[0]]["candidates"])
    cost = CostModel.for_scale(scale, cfg["attention_backend"])
    ledger = {name: 0 for name in (
        "final_teacher_read_flops",
        "teacher_objective_statistics_flops_estimate", "joint_train_flops_estimate",
        "diagnostic_forward_flops", "objective_and_optimizer_flops_estimate", "foldback_forward_flops",
        "parameter_unit_conversion_flops", "foldback_error_flops_estimate")}

    def batch_cost(selected, *, wrapped=True):
        n = rows[selected[0]]["parent"].seq_len
        backbone = eager_read(cost, n, queries=queries, batch=len(selected))
        correction = sum(correction_forward(module.get_config(), n, queries=queries, batch=len(selected))
                         for module in raw)
        if wrapped:
            correction += len(selected) * queries * sum(module.heads * module.head_dim for module in raw)
        return backbone, correction

    teachers = teacher_logits(current, rows, uids, batch_size=batch_size, device=device)
    for selected in groups(uids, rows, batch_size):
        ledger["final_teacher_read_flops"] += batch_cost(selected)[0]
    statistics = loss_statistics(torch.stack([teachers[uid] for uid in train]))
    ledger["teacher_objective_statistics_flops_estimate"] += 4 * len(train) * queries + 5 * len(train)
    units = cfg["output_units"]
    if len(units) != len(raw):
        raise ValueError("output_units requires one training-derived scalar per layer")
    units_source = "sqrt(raw-fit training baseline residual-rate MSE), floor 1e-12"
    wrappers = nn.ModuleList([FeatureOutputUnits(module, unit) for module, unit in zip(raw, units, strict=True)])
    parameter_count = sum(p.numel() for p in wrappers.parameters() if p.requires_grad)
    ledger["parameter_unit_conversion_flops"] += parameter_count

    def diagnose(selected):
        measured = assess(current, rows, selected, teachers, wrappers, statistics, batch_size=batch_size, device=device)
        for group in groups(selected, rows, batch_size):
            ledger["diagnostic_forward_flops"] += sum(batch_cost(group))
        ledger["objective_and_optimizer_flops_estimate"] += 20 * len(selected) * queries
        return measured

    initial_train, initial_validation = diagnose(train), diagnose(val)
    best_validation, best_epoch = initial_validation, 0
    best = {key: value.detach().cpu().clone() for key, value in wrappers.state_dict().items()}
    epochs = [{"epoch": 0, "train": initial_train, "validation": initial_validation}]
    optimizer = torch.optim.AdamW([p for p in wrappers.parameters() if p.requires_grad],
        lr=cfg.get("query_learning_rate", .001), weight_decay=cfg.get("weight_decay", .0001))
    rng = np.random.default_rng(cfg.get("seed", 17))
    steps, clipped = 0, 0
    for epoch in range(1, cfg.get("query_joint_epochs", 8) + 1):
        shuffled = list(train)
        rng.shuffle(shuffled)
        batches = list(groups(shuffled, rows, batch_size))
        rng.shuffle(batches)
        running = 0.
        for selected in batches:
            cache, candidates, delta, counts = batch_inputs(rows, selected, device)
            wanted = torch.stack([teachers[uid] for uid in selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            actual = score_corrected(current, cache, candidates, delta, wrappers, counts)[0]
            loss, _, _ = distillation_loss(actual, wanted, statistics)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite query refinement loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(wrappers.parameters(), 1.)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("nonfinite query refinement gradients")
            clipped += int(float(norm) > 1.)
            optimizer.step()
            steps += 1
            running += float(loss.detach()) * len(selected) / len(train)
            backbone, correction = batch_cost(selected)
            ledger["joint_train_flops_estimate"] += 2 * backbone + 3 * correction
            ledger["objective_and_optimizer_flops_estimate"] += 20 * len(selected) * queries + 20 * parameter_count
        validation = diagnose(val)
        epochs.append({"epoch": epoch, "train_online_objective": running, "validation": validation})
        if validation["objective"] < best_validation["objective"]:
            best_validation, best_epoch = validation, epoch
            best = {key: value.detach().cpu().clone() for key, value in wrappers.state_dict().items()}
        print(json.dumps({"status": "query_feature_refinement", "epoch": epoch,
            "train_online_objective": running, "validation": validation, "selected_epoch": best_epoch}), flush=True)
    wrappers.load_state_dict(best)
    final_train = diagnose(train)
    folded = [wrapper.folded() for wrapper in wrappers]
    ledger["parameter_unit_conversion_flops"] += parameter_count
    parity = 0.
    with torch.no_grad():
        for selected in groups(val, rows, batch_size):
            cache, candidates, delta, counts = batch_inputs(rows, selected, device)
            reference = score_corrected(current, cache, candidates, delta, wrappers, counts)[0]
            actual = score_corrected(current, cache, candidates, delta, folded, counts)[0]
            parity = max(parity, float((reference - actual).abs().max()))
            ledger["foldback_forward_flops"] += sum(batch_cost(selected)) + sum(batch_cost(selected, wrapped=False))
            ledger["foldback_error_flops_estimate"] += 2 * actual.numel()
    if parity > 2e-5:
        raise RuntimeError(f"folded query feature outputs differ: {parity}")
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = ("additional refinement only; no snapshot capture or original ridge; fixed teacher read once, "
        "training objective units and scalar optimizer arithmetic estimated; frozen-backbone forward+input-backward "
        "2x forward and trainable correction forward+backward 3x; wrapped output-unit multiplication explicitly counted")
    return folded, {"train_uids": train, "validation_uids": val, "epochs": epochs,
        "selected_epoch": best_epoch, "selected_validation": best_validation,
        "initial_train": initial_train, "final_train": final_train, "objective_statistics": statistics,
        "output_units_per_layer": [float(wrapper.units) for wrapper in wrappers],
        "output_units_source": units_source, "optimizer_steps": steps, "clipped_steps": clipped,
        "parameter_count": parameter_count, "foldback_max_abs_logit_error": parity,
        "fit_seconds": time.perf_counter() - started, "cost": ledger,
        "objective": "training-teacher-variance normalized within-user centered-logit and user-mean-logit MSE",
        "epoch_selection": "minimum independent validation-user teacher objective including raw epoch0; no evaluation labels",
        "trainable_parameters": "query feature affine weight and bias only; normalization and all model weights frozen"}

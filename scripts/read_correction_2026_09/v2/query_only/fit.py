"""Query-only v2: normalized affine fitting and final-output teacher distillation."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
from torch import nn

from hstu_kvcache.read_correction import fit_query, score_corrected
from read_correction_2026_09.calibrate import groups, layer_targets, stacked
from read_correction_2026_09.cost import (
    CostModel, correction_forward, eager_read, query_ridge_fit, teacher_history_read,
)


class OutputUnits(nn.Module):
    """Optimize affine parameters in head-normalized units; inference folds them."""
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer("scale", scale.to(module.weight))
        with torch.no_grad():
            self.module.weight.div_(self.scale[:, None, None])
            self.module.bias.div_(self.scale[:, None])

    def forward(self, q, k, v, counts):
        rate = self.module.rate(q)
        return rate * self.scale[None, :, None, None] * counts[:, None, None, None].to(q)

    def get_config(self):
        return self.module.get_config()

    def folded(self):
        module = copy.deepcopy(self.module).eval()
        with torch.no_grad():
            module.weight.mul_(self.scale[:, None, None])
            module.bias.mul_(self.scale[:, None])
        return module


def loss_statistics(teacher):
    """Fit normalization on training users only; validation never sets units."""
    means = teacher.double().mean(1)
    centered = teacher.double()-means[:, None]
    return {
        "within_user_variance": max(float(centered.square().mean()), 1e-8),
        "between_user_variance": max(float(means.var(correction=0)), 1e-8),
        "variance_floor": 1e-8,
    }


def distillation_loss(student, teacher, statistics):
    """Centered-logit MSE equals half the all-pairs difference-error MSE."""
    s_mean, t_mean = student.mean(1), teacher.mean(1)
    centered = ((student-s_mean[:, None])-(teacher-t_mean[:, None])).square().mean()
    means = (s_mean-t_mean).square().mean()
    total = centered/statistics["within_user_variance"] + means/statistics["between_user_variance"]
    return total, centered, means


def batch_inputs(rows, uids, device, key="parent"):
    cache = stacked(rows, uids, key, device)
    candidates = torch.stack([rows[uid]["candidates"] for uid in uids]).to(device)
    deltas = torch.tensor([rows[uid]["query_delta"] for uid in uids], device=device)
    counts = torch.full((len(uids),), cache.seq_len, device=device, dtype=torch.long)
    return cache, candidates, deltas, counts


@torch.no_grad()
def teacher_logits(current, rows, uids, *, batch_size, device):
    values = {}
    for selected in groups(uids, rows, batch_size):
        cache, candidates, deltas, counts = batch_inputs(rows, selected, device, key="teacher")
        logits = score_corrected(current, cache, candidates, deltas,
                                 [None]*len(current.blocks), counts)[0].cpu()
        for i, uid in enumerate(selected):
            values[uid] = logits[i]
    return values


@torch.no_grad()
def assess(current, rows, uids, teachers, modules, statistics, *, batch_size, device):
    result = {key: 0. for key in ("objective", "centered_logit_mse", "user_mean_logit_mse", "logit_mse")}
    for selected in groups(uids, rows, batch_size):
        cache, candidates, deltas, counts = batch_inputs(rows, selected, device)
        actual = score_corrected(current, cache, candidates, deltas, modules, counts)[0]
        target = torch.stack([teachers[uid] for uid in selected]).to(device)
        objective, centered, means = distillation_loss(actual, target, statistics)
        for key, value in zip(result, (objective, centered, means, (actual-target).square().mean())):
            result[key] += float(value)*len(selected)/len(uids)
    return result


def initialize(current, rows, uids, *, config, device, scale, batch_size):
    modules, records, units = [None]*len(current.blocks), [], []
    cost = CostModel.for_scale(scale, config["attention_backend"])
    qcount = config["calibration_queries_per_user"]
    lengths = [rows[uid]["parent"].seq_len for uid in uids]
    ledger = {"ridge_trace_flops": 0, "same_query_teacher_flops": 0,
              "ridge_fit_flops_estimate": 0, "normalization_flops": 0}
    for layer, block in enumerate(current.blocks):
        q, targets = layer_targets(current, rows, uids, modules, layer, batch_size=batch_size, device=device)
        module, stats = fit_query(q, targets, ridge=config["ridge"])
        output_scale = targets.double().square().mean((0,2,3)).sqrt().clamp_min(1e-12).float()
        ledger["ridge_trace_flops"] += sum(eager_read(cost,n,queries=qcount) for n in lengths)
        ledger["ridge_trace_flops"] += sum(correction_forward(m.get_config(),n,queries=qcount)
                                             for m in modules if m is not None for n in lengths)
        ledger["same_query_teacher_flops"] += sum(teacher_history_read(cost,n,queries=qcount) for n in lengths)
        ledger["ridge_fit_flops_estimate"] += query_ridge_fit(block.attn.num_heads, block.attn.head_dim, len(uids)*qcount)
        ledger["normalization_flops"] += 4*targets.numel()+2*len(output_scale)
        modules[layer] = module.to(device).eval()
        units.append(output_scale)
        records.append({"layer": layer, "fit": stats, "output_scale_per_head": output_scale.tolist()})
    wrappers = nn.ModuleList([OutputUnits(module, output_scale) for module,output_scale in zip(modules,units)])
    return wrappers, records, ledger


def fit_q(current, rows, uids, *, config, device, scale, batch_size, train_batch):
    """Use final 1/8 calibration UIDs for teacher-only epoch selection."""
    heldout = max(1, int(len(uids)*config["validation_fraction"]))
    train_uids, validation_uids = uids[:-heldout], uids[-heldout:]
    if not train_uids:
        raise ValueError("Q distillation needs at least two calibration users")
    start = time.perf_counter()
    modules, initialization, ledger = initialize(current, rows, train_uids, config=config,
        device=device, scale=scale, batch_size=batch_size)
    teachers = teacher_logits(current, rows, uids, batch_size=batch_size, device=device)
    statistics = loss_statistics(torch.stack([teachers[uid] for uid in train_uids]))
    cost = CostModel.for_scale(scale,config["attention_backend"])
    torch_cost = CostModel.for_scale(scale,"torch")
    qcount = config["calibration_queries_per_user"]
    ledger["parent_and_teacher_cache_flops"] = 2*sum(
        (cost if rows[uid]["parent"].seq_len == config["history_length"] else torch_cost).full_cache(rows[uid]["parent"].seq_len)
        for uid in uids)
    ledger["final_teacher_read_flops"] = sum(eager_read(cost,rows[uid]["parent"].seq_len,queries=qcount) for uid in uids)
    backbone_forward = {uid:eager_read(cost,rows[uid]["parent"].seq_len,queries=qcount) for uid in uids}
    correction_forwards = {uid:sum(correction_forward(m.get_config(),rows[uid]["parent"].seq_len,queries=qcount)
                     + qcount*current.cfg.hidden_size for m in modules) for uid in uids}
    forward = {uid:backbone_forward[uid]+correction_forwards[uid] for uid in uids}
    ledger["joint_train_flops_estimate"] = 0
    ledger["validation_forward_flops"] = 0
    ledger["objective_and_optimizer_flops_estimate"] = 0
    initial_train = assess(current,rows,train_uids,teachers,modules,statistics,batch_size=train_batch,device=device)
    best_validation = assess(current,rows,validation_uids,teachers,modules,statistics,batch_size=train_batch,device=device)
    ledger["validation_forward_flops"] += sum(forward.values())
    best = {key:value.detach().cpu().clone() for key,value in modules.state_dict().items()}
    best_epoch, best_objective = 0, best_validation["objective"]
    epochs = [{"epoch": 0, "train": initial_train, "validation": best_validation}]
    optimizer = torch.optim.AdamW(modules.parameters(),lr=config["query_learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config["seed"])
    parameter_count = sum(parameter.numel() for parameter in modules.parameters())
    steps, clipped = 0, 0
    for epoch in range(1,config["query_epochs"]+1):
        shuffled = list(train_uids)
        rng.shuffle(shuffled)
        batches = list(groups(shuffled,rows,train_batch))
        rng.shuffle(batches)
        running = 0.
        for selected in batches:
            cache,candidates,deltas,counts = batch_inputs(rows,selected,device)
            target = torch.stack([teachers[uid] for uid in selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = score_corrected(current,cache,candidates,deltas,modules,counts)[0]
            loss,_,_ = distillation_loss(predicted,target,statistics)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite Q final-output distillation loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(modules.parameters(),1.)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("nonfinite Q distillation gradients")
            clipped += int(float(grad_norm)>1.)
            optimizer.step()
            steps += 1
            running += float(loss.detach())*len(selected)/len(train_uids)
            ledger["joint_train_flops_estimate"] += sum(2*backbone_forward[uid]+3*correction_forwards[uid] for uid in selected)
            ledger["objective_and_optimizer_flops_estimate"] += 20*len(selected)*qcount+20*parameter_count
        validation = assess(current,rows,validation_uids,teachers,modules,statistics,batch_size=train_batch,device=device)
        ledger["validation_forward_flops"] += sum(forward[uid] for uid in validation_uids)
        epochs.append({"epoch":epoch,"train_online_objective":running,"validation":validation})
        if validation["objective"]<best_objective:
            best_epoch,best_objective,best_validation = epoch,validation["objective"],validation
            best = {key:value.detach().cpu().clone() for key,value in modules.state_dict().items()}
        print(json.dumps({"status":"query_distillation","users":len(uids),"epoch":epoch,
                          "train_online_objective":running,"validation":validation,"best_epoch":best_epoch}),flush=True)
    modules.load_state_dict(best)
    final_train = assess(current,rows,train_uids,teachers,modules,statistics,batch_size=train_batch,device=device)
    ledger["validation_forward_flops"] += sum(forward[uid] for uid in train_uids)
    folded = [module.folded() for module in modules]
    # Fold-back parity is checked on real validation data, with the same native reader.
    selected = next(iter(groups(validation_uids,rows,train_batch)))
    with torch.no_grad():
        cache,candidates,deltas,counts = batch_inputs(rows,selected,device)
        reference = score_corrected(current,cache,candidates,deltas,modules,counts)[0]
        actual = score_corrected(current,cache,candidates,deltas,folded,counts)[0]
        error = float((reference-actual).abs().max())
        if error > 2e-5:
            raise RuntimeError(f"normalized-output fold-back changed final scores: {error}")
    ledger["validation_forward_flops"] += 2*sum(forward[uid] for uid in selected)
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = "analytical arithmetic; frozen-backbone forward+input backward estimated as 2x forward, correction forward+backward as 3x; ridge solve and optimizer estimated"
    return folded, {"train_uids":train_uids,"validation_uids":validation_uids,
        "initialization":initialization,"objective_statistics":statistics,"epochs":epochs,
        "selected_epoch":best_epoch,"selected_validation":best_validation,"final_train":final_train,
        "initial_train":initial_train,"optimizer_steps":steps,"clipped_steps":clipped,
        "parameter_count":parameter_count,"foldback_max_abs_logit_error":error,
        "fit_seconds":time.perf_counter()-start,"cost":ledger,
        "objective":"within-user centered-logit/pairwise-difference MSE plus user-mean-logit MSE, each normalized by training-teacher variance",
        "epoch_selection":"minimum held-out calibration-user teacher objective, including ridge epoch0; no evaluation labels"}



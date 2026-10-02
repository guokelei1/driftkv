"""Fit the extra per-query full-history branch on independent calibration users."""
from __future__ import annotations

import time
import torch

from hstu_kvcache.read_correction import HistoryCorrection


def fit(query, target_rate, k, v, counts, *, initial_query, width=32, epochs=40,
        learning_rate=.003, weight_decay=.0001, batch_size=8, token_chunk=256,
        query_chunk=4, seed=17, device="cuda:0"):
    """CPU observations, bounded GPU batches; all rows train every epoch."""
    began = time.perf_counter()
    torch.manual_seed(seed)
    module = HistoryCorrection(query.shape[1], query.shape[-1], width=width,
                               query_chunk=query_chunk, token_chunk=token_chunk)
    # initialize_query copies parameters/buffers: leave the supplied Q model intact.
    module.initialize_query(initial_query)
    module.set_history_normalization(k, v, counts)
    # Adam updates parameters in absolute units. Normalize each independent head's
    # output units as well as its inputs, then fold the units into the output
    # weights after fitting. The inference formula and its FLOPs stay unchanged.
    output_scale = target_rate.detach().double().square().mean((0, 2, 3)).sqrt().clamp_min(1e-12).to(target_rate.dtype)
    normalized_target = target_rate / output_scale[None, :, None, None]
    with torch.no_grad():
        module.weight.div_(output_scale[:, None, None])
        module.bias.div_(output_scale[:, None])
    module.to(device)
    gpu_scale = output_scale.to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)

    def batches(order):
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            yield tuple(value[ids].to(device) for value in (query, normalized_target, k, v, counts))

    @torch.no_grad()
    def mse():
        squared = torch.zeros(query.shape[1], dtype=torch.float64, device=device)
        elements = 0
        for q, target, bk, bv, bc in batches(torch.arange(len(query))):
            error = ((module.rate(q, bk, bv, bc) - target).double()
                     * gpu_scale[None, :, None, None].double())
            squared += error.square().sum((0, 2, 3))
            elements += target.numel() // query.shape[1]
        per_head = (squared / elements).cpu().tolist()
        return sum(per_head) / len(per_head), per_head

    initial_mse, initial_head_mse = mse()
    epoch_mse, epoch_normalized_mse, steps = [], [], 0
    for epoch in range(epochs):
        squared, normalized_squared, elements = 0., 0., 0
        for q, target, bk, bv, bc in batches(torch.randperm(len(query), generator=generator)):
            optimizer.zero_grad(set_to_none=True)
            error = module.rate(q, bk, bv, bc) - target
            loss = error.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite history-conditioned calibration loss")
            loss.backward()
            optimizer.step()
            physical_error = error.detach().double() * gpu_scale[None, :, None, None].double()
            squared += float(physical_error.square().sum())
            normalized_squared += float(error.detach().double().square().sum())
            elements += error.numel()
            steps += 1
        epoch_mse.append(squared / elements)
        epoch_normalized_mse.append(normalized_squared / elements)
    final_mse, final_head_mse = mse()
    with torch.no_grad():
        module.weight.mul_(gpu_scale[:, None, None])
        module.bias.mul_(gpu_scale[:, None])
        module.history_weight.mul_(gpu_scale[:, None, None])
    module.eval()
    return module, {
        "initial_rate_mse": initial_mse, "final_rate_mse": final_mse,
        "initial_rate_mse_per_head": initial_head_mse, "final_rate_mse_per_head": final_head_mse,
        "epoch_online_rate_mse": epoch_mse,
        "epoch_online_normalized_rate_mse": epoch_normalized_mse,
        "output_scale_per_head": output_scale.tolist(),
        "output_normalization": "calibration target RMS per head; Q weight/bias and history output weight folded back before saving",
        "objective": "mean squared rate error in normalized output units; head-local parameters are independent",
        "epochs": epochs, "optimizer_steps": steps, "batch_size": batch_size,
        "training_rows": len(query) * query.shape[2] * epochs,
        "queries_per_user": query.shape[2], "padded_history_length": k.shape[1],
        "history_positions_per_epoch": int(counts.sum()),
        "padded_history_positions_per_epoch": len(query) * k.shape[1],
        "loss_evaluation_passes": 2, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "seed": seed,
        "fit_seconds": time.perf_counter() - began,
        "parameter_count": sum(p.numel() for p in module.parameters()),
        "output_normalization_flops": int(3*target_rate.numel() + 2*query.shape[1]
            + 2*(module.weight.numel()+module.bias.numel()) + module.history_weight.numel()),
    }

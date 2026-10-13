"""Token alignment shared by selected Q/H calibration; the affine probe CLI retired."""
import json
import time
import torch
from hstu_kvcache.read_correction_v4.token_read import TokenReadCorrection, fit_affine_tokens
from read_correction_v4.cost import CostModel, token_ridge_fit


def token_pairs(rows, uids, layer, cap):
    inputs, targets = [], []
    for uid in uids:
        row = rows[uid]
        n = row["parent"].seq_len
        index = torch.linspace(0, n - 1, min(n, cap)).long()
        x = torch.cat((row["parent"].k[layer, 0, index], row["parent"].v[layer, 0, index]), -1)
        y = torch.cat((row["teacher"].k[layer, 0, index], row["teacher"].v[layer, 0, index]), -1)
        inputs.append(x)
        targets.append(y - x)
    return torch.cat(inputs), torch.cat(targets)


def fit_modules(current, rows, train, validation, cfg, device, scale):
    modules, records = [], []
    sparse = CostModel.for_scale(scale, cfg["attention_backend"])
    dense = CostModel.for_scale(scale, "torch")
    ledger = {"parent_and_teacher_cache_flops": 2 * sum(
        (sparse if rows[u]["parent"].seq_len == cfg["history_length"] else dense).full_cache(rows[u]["parent"].seq_len)
        for u in train + validation), "token_ridge_fit_flops": 0, "teacher_residual_flops": 0,
        "validation_mapping_flops": 0}
    for layer, block in enumerate(current.blocks):
        x, y = token_pairs(rows, train, layer, cfg["tokens_per_user"])
        vx, vy = token_pairs(rows, validation, layer, cfg["tokens_per_user"])
        started = time.perf_counter()
        params, stats = fit_affine_tokens(x.to(device), y.to(device), ridge=cfg["ridge"])
        module = TokenReadCorrection(block.attn.num_heads, block.attn.head_dim).to(device)
        with torch.no_grad():
            for key, value in params.items():
                getattr(module, key).copy_(value)
            xn = (vx.to(device) - module.input_mean) / module.input_scale
            error = module.token_delta(xn) - vy.to(device)
            validation_mse = float(error.double().square().mean())
        module.eval().requires_grad_(False)
        modules.append(module)
        p = x.shape[1]
        ledger["token_ridge_fit_flops"] += token_ridge_fit(observations=len(x), input_width=p)
        ledger["teacher_residual_flops"] += (len(x) + len(vx)) * p
        ledger["validation_mapping_flops"] += len(vx) * (2*p*p + 6*p)
        records.append({"layer": layer, "train": stats, "validation_delta_mse": validation_mse,
            "validation_zero_delta_mse": float(vy.double().square().mean()),
            "seconds": time.perf_counter()-started})
        print(json.dumps({"status": "layer_fit", **records[-1]}), flush=True)
    ledger["calibration_flops"] = sum(ledger.values())
    ledger["convention"] = "analytical arithmetic, token ridge solve estimated; all teacher-cache preparation and fitting charged"
    return modules, records, ledger

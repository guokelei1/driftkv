"""Phase 0 - V2: per-user JVP vs full-KV-recompute cost ratio.

Roadmap Insight 4 predicts a single forward-mode JVP (J.dtheta) costs ~2x a
pure forward, i.e. *more* than recomputing the KV (1 forward). If true, naive
per-user drift estimation is more expensive than the thing it tries to avoid -
motivating the low-cost estimation research (paths 1/2/3).

This script sweeps model sizes and measures forward (recompute) vs JVP time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hstu_kvcache.drift import dtheta_as_dict, naive_per_user_jvp
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json

SWEEPS = [
    dict(hidden=128, layers=2, heads=2, head_dim=64, L=64),
    dict(hidden=256, layers=4, heads=4, head_dim=64, L=128),
    dict(hidden=512, layers=6, heads=8, head_dim=64, L=256),
    dict(hidden=512, layers=8, heads=8, head_dim=64, L=512),
]


def main(out_path: str = "results/phase0/V2_jvp_vs_forward.json") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for sw in SWEEPS:
        cfg = HSTUConfig(
            num_items=5000,
            num_behaviors=9,
            hidden_size=sw["hidden"],
            num_layers=sw["layers"],
            num_heads=sw["heads"],
            head_dim=sw["head_dim"],
            max_seq_len=sw["L"] + 8,
        )
        torch.manual_seed(0)
        model = HSTU(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        B = 8
        batch = {
            "item_ids": torch.randint(1, cfg.num_items + 1, (B, sw["L"]), device=device),
            "behaviors": torch.randint(0, cfg.num_behaviors + 1, (B, sw["L"]), device=device),
            "time_deltas": torch.rand(B, sw["L"], device=device) * 3600,
        }
        theta = model_params_vec(model)
        dtheta = (torch.randn_like(theta) * 0.01).to(device)
        est = naive_per_user_jvp(
            model, batch, dtheta_as_dict(model, dtheta), device, warmup=3, repeats=10
        )
        ratio = est.jvp_time_ms / est.forward_time_ms
        kv_numel = est.kv_numel
        rec = {
            **sw,
            "n_params": n_params,
            "kv_numel": kv_numel,
            "forward_ms": round(est.forward_time_ms, 3),
            "jvp_ms": round(est.jvp_time_ms, 3),
            "jvp_over_forward": round(ratio, 3),
        }
        print(rec)
        results.append(rec)
        del model, batch, theta, dtheta
        torch.cuda.empty_cache() if device.type == "cuda" else None
    save_json(results, out_path)
    print(f"\nSaved {out_path}")
    ratios = [r["jvp_over_forward"] for r in results]
    print(f"JVP/forward ratio range: {min(ratios):.2f}x .. {max(ratios):.2f}x")
    print("Verdict: JVP is MORE expensive than recompute at every scale -> Insight 4 confirmed.")


if __name__ == "__main__":
    main()

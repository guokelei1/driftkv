# HSTU KV-Cache Drift

Low-cost drift estimation for HSTU generative-recommender KV caches under streaming model updates.

## Problem

When an HSTU model is continuously retrained, each parameter update θ → θ+Δθ invalidates every user's derived KV cache `F(θ, x_u)`. Recomputing all caches costs one full forward per user. We want to estimate the drift `||F(θ+Δθ, x_u) − F(θ, x_u)||` **cheaply** (≪ one forward) to drive a three-state per-user decision: **reuse / migrate / recompute**.

The crux (roadmap Insight 4/5): a naive per-user JVP costs ~2-3× a forward, so it is *more* expensive than recomputing. The innovation must reduce that cost via cross-user sharing, offline Fisher spectra, or layer decomposition.

## Status

**Phase 0 (pre-validation) complete — gating passed.** See `experiments/phase0/SUMMARY.md`:

| Check | Result |
|---|---|
| V1 industrial streaming-update frequency | PASS (hourly-scale updates → small Δθ) |
| V2 JVP vs recompute cost | 3.2–6.4× more expensive (motive confirmed) |
| V3 cross-user J·Δθ sharing | drift *norm* predictable from 4 features (7% rel-MAE) |
| V4 stale-KV accuracy decay | gentle at small Δθ (<1.5%), steep at large (three-state viable) |

## Layout

```
src/hstu_kvcache/
  models/      HSTU (pointwise attention, first-class KV output), modular for research
  data/        KuaiRand streaming-trace loader + ML1m generative-rec loader
  streaming/   streaming trainer → θ checkpoints → Δθ → oracle recompute
  drift/       naive per-user JVP, cross-user low-rank, Fisher-spectrum
  serving/     three-state cache policy (reuse/migrate/recompute)
scripts/       v1..v4 phase-0 verification scripts
experiments/   phase0..4 result docs + artifacts
docs/          00-08 analysis & roadmap (08 is authoritative)
data/          KuaiRand logs + ML1m pilot20 (git-ignored)
```

## Setup

```bash
pip install -e .
```

## Datasets

All datasets are **git-ignored** (too large for git). Place them under `data/` as below; the original source paths on this host are shown for re-copying.

| Dataset | In-repo path | Original source (host) |
|---|---|---|
| KuaiRand-1K (ms-level logs, two 2-week windows) | `data/kuairand/` | `/home/gkl/fun-rec/data/dataset/kuairand/KuaiRand-1K/data/` |
| ML1m hard_v5 pilot20 (generative-rec format) | `data/movielens/` | `/home/gkl/LRM1/data/standardized/movielens_1m_hard_v5/pilot20` |

Copy commands:

```bash
cp -r /home/gkl/fun-rec/data/dataset/kuairand/KuaiRand-1K/data/* data/kuairand/
cp -r /home/gkl/LRM1/data/standardized/movielens_1m_hard_v5/pilot20/* data/movielens/
```

## Artifacts (git-ignored)

Model checkpoints, Δθ series, and bulky experiment outputs are **not** tracked:

- `checkpoints/` — θ₀…θ_t streaming checkpoints (`*.pt`)
- `results/runs/`, `logs/`, `wandb/` — run logs and experiment-tracking artifacts
- `data/raw|interim|processed|external/` — intermediate data caches

## Run the Phase 0 checks

```bash
python scripts/v2_jvp_vs_forward.py        # JVP vs recompute cost
python scripts/v3_cross_user_sharing.py    # cross-user low-rank drift
python scripts/v4_staleness_decay.py       # stale-KV accuracy decay curve
```

## Key design choices

- **Pointwise attention** (`models/attention.py`): `elu(QKᵀ)+1`, no softmax, multiplicative causal mask — the HSTU defining feature. Each module is standalone so roadmap U1/U2 tweaks are localised.
- **KV as first-class output**: `HSTU.compute_kv()` returns `F(θ, x_u)`; `HSTUKVCache.drift_norm()` gives ground-truth drift.
- **torch.func JVP**: `drift/jvp.py` uses forward-mode `jvp` + `functional_call` for the naive baseline and `ground_truth_drift` oracle.
- **Streaming trainer**: produces θ₀…θ_t checkpoint sequence; `dtheta_sequence()` yields the Δθ series for drift experiments.

## Roadmap

Source of truth: `docs/08_core_insights_and_roadmap.md`. Phase 0 → 1 (streaming + KV infra) → 2 (workload + existence) → 3 (drift methods) → 4 (end-to-end system), each gated.

# Fixed-endpoint cache-version matrix

> Status: exploratory metric selection and four-seed validation completed on 2026-07-23.
> This experiment strengthens the pre-design motivation; it does not establish migration-method
> generality.

## 1. Question

The earlier cumulative-age control advances the model, evaluation window, and cache age together.
It is a valid streaming trace, but a local change can reflect both cache staleness and a different
evaluation population. This follow-up fixes:

- the current model version $\theta_T$;
- the final evaluation window and user histories;
- the frozen and full-compute endpoints;

and varies only the model version $\theta_v$ used to produce the same prefix K/V:

$$
C_v(x)=F(\theta_v,x), \qquad v=T-1,\ldots,0.
$$

The current model consumes each $C_v(x)$ and the latest token. Thus every point in one curve uses
the same current model, history, target set, and users. This isolates model-version compatibility
from moving-window composition.

## 2. Cross-dataset metric

Raw BestRank maintenance is catalog- and task-scale dependent. The old endpoint values
`85.32/30.31/13.00` for KuaiRand/QB/QK therefore must not be compared as effect sizes.

For an oriented quality gain $G$, define the staleness tax

$$
\tau_G(v,T)=
\frac{G(\text{full compute},\text{reuse})}
     {G(\text{full compute},\text{frozen})}.
$$

For BestRank this is

$$
\tau_{\mathrm{BR}}=
\frac{\mathrm{BR}_{\mathrm{reuse}}-\mathrm{BR}_{\mathrm{full}}}
     {\mathrm{BR}_{\mathrm{frozen}}-\mathrm{BR}_{\mathrm{full}}}.
$$

It is the fraction of the useful streaming-training improvement forfeited by a version-stale
cache. The ratio is computed within each training seed and then summarized across seeds. It is
undefined when the full-compute streaming-value denominator is not positive. Users inside one
trained model remain diagnostics, not independent repeats.

## 3. Protocols

The coarse matrix reuses the frozen four-seed checkpoints at $T=5$:

- KuaiRand: `cache_version_matrix_v1`, 300 final-window users;
- QB fixed horizon and QK top-5k: `cache_version_matrix_full_eval_v1`, every eligible final-window
  user up to 5,000.

The fine matrix increases only temporal resolution:

- KuaiRand: one-day updates, $T=15$;
- QB/QK: the prepared ordered stream is rebinned from eight to four exposures per update,
  $T=10$;
- matrix evaluation: `cache_version_matrix_fine_full_eval_v1`;
- four training seeds per dataset.

KuaiRand fine runs retain the existing latest-sequence training protocol and are a diagnostic.
They do not supersede the all-chunks KuaiRand operating point. Tenrec age is ordinal exposure
count, not calendar time, so numeric age locations are not compared directly across datasets.

## 4. Endpoint alignment

### 4.1 Coarse fixed endpoint

| Dataset | Raw BestRank maintenance | BestRank staleness tax | MeanRank staleness tax |
|---|---:|---:|---:|
| KuaiRand | 85.32 | **0.176 `[0.115, 0.237]`** | **0.205 `[0.148, 0.263]`** |
| QB fixed horizon | 30.31 | **0.315 `[0.225, 0.404]`** | **0.160 `[0.106, 0.214]`** |
| QK top-5k | 13.00 | **0.276 `[0.168, 0.385]`** | **0.256 `[0.166, 0.346]`** |

The raw full-compute gains span 10.2x and raw maintenance gaps span 6.6x, while the primary tax
spans only 1.79x and MeanRank tax 1.60x.
Rank-utility and NDCG@100 taxes span 2.40x and 2.47x respectively, although the QK NDCG interval
includes zero. The aligned claim is therefore strongest on full-catalog rank, not every ranking
metric.

### 4.2 Fine fixed endpoint

| Dataset | BestRank staleness tax | MeanRank staleness tax |
|---|---:|---:|
| KuaiRand | **0.177 `[0.118, 0.236]`** | **0.206 `[0.148, 0.263]`** |
| QB fixed horizon | **0.350 `[0.263, 0.437]`** | **0.179 `[0.126, 0.232]`** |
| QK top-5k | **0.141 `[0.056, 0.227]`** | 0.119 `[-0.042, 0.279]` |

BestRank remains positive on all three datasets and spans 2.48x. Fine QK does not support a
positive NDCG@100 tax, and its MeanRank interval crosses zero. This negative sensitivity is
retained rather than using the fine experiment to claim metric-universal degradation.

## 5. Local degradation and fixed-window reuse

The fine curve exposes two distinct observations.

First, a substantial fraction of the final BestRank tax can arrive in one adjacent update. The
largest positive one-step change and its share of the endpoint tax are:

| Dataset | Largest-step location by seed 0/1/2/3 | Largest step by seed | Step / endpoint tax |
|---|---|---|---|
| KuaiRand | 13→14 / 13→14 / 13→14 / 13→14 | 0.044 / 0.090 / 0.074 / 0.046 | 0.31 / 0.55 / 0.32 / 0.27 |
| QB | 6→7 / 2→3 / 1→2 / 5→6 | 0.115 / 0.128 / 0.122 / 0.143 | 0.42 / 0.36 / 0.31 / 0.38 |
| QK | 8→9 / 9→10 / 4→5 / 7→8 | 0.159 / 0.097 / 0.213 / 0.557 | 1.08 / 0.62 / 1.09 / 8.28 |

KuaiRand has a common fine transition: age 13→14 adds `0.063 [0.028, 0.099]` tax and is positive
in 4/4 seeds. QB has a common age 6→7 increase of `0.094 [0.052, 0.137]`, also positive in 4/4
seeds. QK has clear within-seed jumps but their locations move; aggregating age 7→8 gives a broad
interval and only 2/4 positive directions.

Second, a fixed quality threshold is crossed at different ages. The first age at which BestRank
tax reaches 10% is:

| Dataset | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|---:|
| KuaiRand | 14 | 14 | 14 | 14 |
| QB | 2 | 1 | 5 | 4 |
| QK | 1 | 10 | 5 | 8 |

Every fine curve also contains at least two adjacent non-monotonic reversals. Cache age is
therefore not a calibrated quality state, especially on QB and QK. A universal periodic window
would either recompute unnecessarily during a benign plateau or miss an earlier bad update.

The evidence does **not** prove that every tuned fixed-window policy fails: KuaiRand is notably
stable at this operating point, and periodic recomputation remains a required baseline. The
supported claim is narrower:

> cache age alone does not provide dataset- and update-invariant quality control; observed local
> jumps motivate an update-aware, model-version-cohort trigger.

The largest-step statistic and its location were selected in exploratory analysis. They must be
frozen before a new protocol or independent dataset is used as confirmatory evidence.

## 6. Method implication

This result does not revive per-user JVP or Fisher estimation. A plausible next policy operates
once per model-version pair and can use cache age, layerwise update norms, or a small held-out
probe cache. All caches from the same old/current version cohort then receive the same
reuse/migrate/recompute action.

The immediate comparison should include:

- fixed-period full recomputation;
- an age-only threshold;
- an update-aware version-cohort trigger;
- cheap refresh, the frozen deepest-suffix candidates, and full recomputation;
- equal-cost and equal-quality operating points under a natural mixture of cache versions.

## 7. Reproduction

The fine Tenrec streams are deterministic rebinnings of the already audited prepared arrays:

```bash
python scripts/rebin_exposure_windows.py \
  --source data/processed/tenrec_qb_top50000_users5000_fixed_horizon.npz \
  --base-prefix 64 --window-size 4 --windows 12 \
  --output data/processed/tenrec_qb_top50000_users5000_fixed_horizon_fine4.npz \
  --metadata-output results/dataset_audit/tenrec_qb_top50000_users5000_fixed_horizon_fine4_prepared.json
python scripts/rebin_exposure_windows.py \
  --source data/processed/tenrec_qk_top5000_users5000.npz \
  --base-prefix 64 --window-size 4 --windows 12 \
  --output data/processed/tenrec_qk_top5000_users5000_fine4.npz \
  --metadata-output results/dataset_audit/tenrec_qk_top5000_users5000_fine4_prepared.json
```

The matrix entry point is:

```bash
python scripts/cache_version_matrix.py \
  --run-result <core-result.json> \
  --checkpoint-dir <checkpoint-directory> \
  --current-t <T> \
  --max-eval-users <N> \
  --output <per-seed-matrix.json>
```

The compact summaries are generated with:

```bash
python scripts/summarize_cache_version_matrix.py \
  --dataset 'kuairand=<kuai-matrix-glob>' \
  --dataset 'tenrec_qb=<qb-matrix-glob>' \
  --dataset 'tenrec_qk=<qk-matrix-glob>' \
  --output <summary.json>
```

Tracked evidence:

- `results/exposure/cache_age_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_fine_cross_dataset_summary.json`

Per-seed matrices include user diagnostics and are intentionally ignored by Git.

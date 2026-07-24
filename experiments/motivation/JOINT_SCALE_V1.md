# Joint data/model-scale motivation protocol

> Status: rejected after the seed-0 gate. Do not replicate or use for the primary capacity claim.
> Changing catalog size together with user count changed the task and left the largest Tenrec
> models underfed. The corrected fixed-task protocol is `CAPACITY_V2.md`.

> Frozen before seed-0 model training on 2026-07-23.

## Purpose

This protocol tests whether the complete pre-design motivation survives matched growth in training
data and model capacity. It does not evaluate cache-migration methods.

Every cell must establish the same sequence:

1. streaming updates make the current model better than frozen theta-0 serving;
2. a theta-0 prefix cache retains some of that value instead of failing immediately;
3. fresh current-model K/V recovers an additional cache-maintenance gap;
4. the gap changes across cache versions, so cache age is not assumed to be a calibrated trigger;
5. full prefix recomputation has a measured GPU cost.

## Frozen scale points

| Tier | Catalog | KuaiRand users | QB users | QK users | Model |
|---|---:|---:|---:|---:|---|
| small | 5,000 | 250 | 500 | 500 | 3L, H64, 4 heads, head dim 16 |
| medium | 20,000 | 500 | 1,000 | 1,000 | 6L, H96, 4 heads, head dim 24 |
| large | 50,000 | up to 1,000 | 2,000 | 2,000 | 12L, H192, 8 heads, head dim 24 |

The tiers are matched operating points, not a factorial model-only experiment. Both retained
training data and capacity increase from small to large. Actual users, rows, tokens, eligible
targets, and parameters must be recorded; a tier is invalid if its effective training data does
not exceed the preceding tier.

The catalog is fitted from base-only interactions. User selection never uses feedback labels or
model outcomes.

## Shared semantics

- All data enter `StreamingDataPlan` with item, behavior, label, ordinal/time, and window fields.
- The target is item `t+1` from hidden state `t`.
- Every observed exposure enters context; only observed positive feedback is a training/evaluation
  target.
- The vocabulary is frozen before the stream.
- All models use length 128, batch 32, six base epochs at `3e-4`, and two epochs per update at
  `1e-4`.
- There are 11 updates and a final unseen evaluation window.
- Evaluation uses the full fitted catalog and at most 500 active users.
- Seed 0 is the structural gate. Training seeds 1-3 are replication units, not user resamples.

## Dataset-specific ordering

KuaiRand uses its real timestamp order, 14 base dates, then 11 one-day updates. Complete base and
stream histories are chunked, so increasing the catalog and user cohort increases effective
targets rather than merely changing the embedding table.

QB and QK use the same prepared ordered-exposure representation: 64 raw base exposures and 12
four-exposure windows. The three tiers are nested top-5k/top-20k/top-50k vocabulary prefixes.
Their cohorts require at least one retained exposure in every window, then rank users by retained
base activity with user ID as the deterministic tie-break. This is a fixed-availability mechanism
cohort, not a population estimate. QB/QK official file order is ordinal and has no calendar-time
interpretation.

## Required artifacts per cell

- Core training and checkpoints theta-0 through theta-11.
- Moving-window one-step and cumulative theta-0 stale-cache measurements.
- Frozen/full-reuse/full-compute controls at theta-1/3/5/7/9/11.
- A fixed theta-11 endpoint with stale cache versions theta-10 through theta-0.
- Resident-GPU prefix operator timing at batch 32 and length 128.

Protocol strings:

- `motivation_joint_scale_v1_training`
- `motivation_joint_scale_v1_streaming_control`
- `motivation_joint_scale_v1_cache_version_matrix`
- `motivation_joint_scale_v1_operator_cost`

These artifacts cannot be pooled with the earlier aligned 6L/H96 result families.

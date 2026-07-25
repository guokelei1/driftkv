# Verified cohort migration compiler

## Status

`kuairand_long_context_4plus12_verified_compiler_v1` implements the first complete
generate–verify–publish loop for the large KuaiRand theta11/D16 endpoint. It is a two-GPU,
seed-0 adaptive design experiment. Motivation, checkpoints, serving semantics, source versions,
and the base-only 50,000-item catalog are unchanged.

The 682 evaluable users have four disjoint roles:

| Role | Users |
|---|---:|
| Full-affine fit | 40 |
| Earlier program hyperparameter selection | 60 |
| New label-free certificate | 60 |
| Final recommendation test | 522 |

The new certificate users were never used to fit the affine program or select its rank, ridge, or
attention weighting. Because earlier design rounds already inspected this seed's former test
users, the complete result remains `adaptive_seed0_exploration`; future confirmation must freeze
the compiler on a new seed or accepted external checkpoint.

## Contract

The action library contains current projection, compiled full-affine migration, structural prefix
replay at depths four and eight, and exact recomputation. Reuse supplies the zero-maintenance
reference but is not a publishable synchronization action.

For each certificate user and semantic metric, recovery closes the gap from reuse to exact. The
three error views are:

\[
e_{\mathrm{cache}}=\text{relative K/V error},\qquad
e_{\mathrm{score}}=1-\cos(s_a,s_f),\qquad
e_{\mathrm{top100}}=1-\text{overlap}_{100}.
\]

For action \(a\),

\[
\mathrm{recovery}(a)=
\frac{e_{\mathrm{reuse}}-e_a}
     {e_{\mathrm{reuse}}-e_{\mathrm{exact}}}.
\]

The frozen contract requires all three metrics to satisfy:

- recovery target at least 70%;
- a one-sided 90% bootstrap lower bound on ratio-of-means recovery of at least 70%;
- at least 80% user coverage after taking a one-sided 90% Wilson lower bound;
- measured GPU cost no greater than 30% of exact for the primary action.

Recommendation labels are unavailable to certification. The compiler selects the minimum-cost
certified action inside budget. If none exists, it selects the minimum-cost certified
budget-overflow action. Exact recomputation is the terminal fallback.

## Certificate result

| Cache age | Selected action | Cost / exact | Worst recovery lower bound | Worst coverage lower bound | Published fallback |
|---:|---|---:|---:|---:|---|
| 11 | compiled full-affine | 0.0627 | 0.8530 | 0.9224 | structural p8, exact |
| 7 | compiled full-affine | 0.0631 | 0.8373 | 0.9005 | exact |
| 1 | compiled full-affine | 0.0631 | 0.9228 | 0.9459 | structural p8, exact |

The result is selective rather than a hard-coded ladder:

- projection-only is inside budget but fails cache and top-100 fidelity at every age;
- structural p4 costs about 0.323x and fails the 70% cache contract at every age;
- structural p8 passes at ages 11 and 1 but costs about 0.549x, so it is retained only as a
  fallback;
- structural p8 fails the age-7 contract, so that plan falls directly from compiled migration to
  exact recomputation.

Thus version cohorts remain compilation and batching keys; the system does not predict that a
version is task-quality-safe to reuse. It verifies which synchronization implementation satisfies
an observable current-model semantic contract.

## Final-test result

The compiler publishes one action before accessing recommendation labels on the 522 final users.

| Cache age | Test cost / exact | K/V recovery | Worst recovery lower bound | Worst coverage lower bound | Score cosine | Top-100 overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.0638 | 0.8865 | 0.8794 | 0.9259 | 0.999770 | 0.9695 |
| 7 | 0.0640 | 0.8911 | 0.8847 | 0.9280 | 0.999913 | 0.9797 |
| 1 | 0.0641 | 0.9356 | 0.9325 | 0.9603 | 0.999994 | 0.9948 |

Every final-test lower bound exceeds its certificate requirement. The selected action also remains
well below the 30% cost budget.

At the harmful age-11 endpoint:

| Metric | Reuse | Verified migration | Fresh exact | Signed gap recovered |
|---|---:|---:|---:|---:|
| MeanRank | 10812.22 | 9297.65 | 9279.73 | 98.8% |
| AUC | 0.783816 | 0.814112 | 0.814471 | 98.8% |
| NDCG@100 | 0.008516 | 0.011043 | 0.011314 | 90.3% |
| Hit@100 | 0.141763 | 0.172414 | 0.176245 | 88.9% |

At age seven, reuse happens to outperform the exact current model on MeanRank and AUC. The
verified program restores current-model semantics and therefore does not preserve that accidental
task gain. This remains evidence that a cache system must not claim to improve the deployed model
or use task quality as a version-admission oracle.

## Implementation

The reusable implementation consists of:

- `FidelityContract`, `RecoveryCertificate`, and `ActionCertificate`;
- a serializable `VerifiedMigrationPlan` containing the selected action and ordered fallbacks;
- deterministic bootstrap and user-coverage certification;
- automatic minimum-cost selection with exact fallback;
- a two-worker compiler/evaluator that withholds recommendation labels until final test.

The three generated manifests are stored under
`checkpoints/kuairand_long_context_4plus12_exploration/seed0/verified_plans/`.

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_kuairand_long_context_verified_compiler.py
```

Result:

`results/motivation_scale/long_context_4plus12_verified_compiler_seed0.json`

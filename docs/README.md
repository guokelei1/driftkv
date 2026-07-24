# Current documentation index

This directory intentionally contains only the current research story and protocol. The precedence
order is:

1. [08_core_insights_and_roadmap.md](08_core_insights_and_roadmap.md) — research thesis, supported
   claims, open gates, and stop conditions.
2. [eval_protocol.md](eval_protocol.md) — experimental semantics and comparability rules.
3. [paper_draft_intro_motivation.md](paper_draft_intro_motivation.md) — advisor-facing narrative.
4. [../experiments/validity/README.md](../experiments/validity/README.md) — repaired motivation
   experiment and exact results.
5. [../experiments/validity/STREAMING_VALUE_CONTROL.md](../experiments/validity/STREAMING_VALUE_CONTROL.md) —
   corrected frozen/full-reuse/full-compute value chain.
6. [../experiments/validity/INTERVAL_ORACLE.md](../experiments/validity/INTERVAL_ORACLE.md) —
   terminal projection optimization, interval search, and held-out decision.
7. [../experiments/validity/LAYERWISE_METHOD.md](../experiments/validity/LAYERWISE_METHOD.md) —
   original six-layer suffix experiment and exact quality results.
8. [../experiments/scaling/SCALING_V1.md](../experiments/scaling/SCALING_V1.md) — fixed-operator
   sequence, batch, depth, update-magnitude, and MovieLens transfer results.
9. [../experiments/scaling/KUAIRAND_FACTORIAL_V1.md](../experiments/scaling/KUAIRAND_FACTORIAL_V1.md) —
   top-5k/top-20k by 6L/12L factorial stress test.
10. [../experiments/scaling/KUAIRAND_DATA_UTILIZATION_V1.md](../experiments/scaling/KUAIRAND_DATA_UTILIZATION_V1.md) —
    top-50k complete-base-chunk training and the stronger four-seed operating point.
11. [dataset_expansion_audit.md](dataset_expansion_audit.md) — Taobao target-semantic boundary and
    the leak-free Tenrec QK/QB and ZhihuRec capacity audit.
12. [../experiments/exposure/ORDERED_EXPOSURE_V1.md](../experiments/exposure/ORDERED_EXPOSURE_V1.md) —
    aligned KuaiRand/QB/QK motivation plus the separate mixed method-transfer and ZhihuRec gates.
13. [../experiments/exposure/CACHE_VERSION_MATRIX_V1.md](../experiments/exposure/CACHE_VERSION_MATRIX_V1.md) —
    fixed-current/evaluation cache-version curves, cross-dataset staleness tax, and the
    fixed-window boundary.
14. [../experiments/exposure/OPPORTUNITY_REGIME_V1.md](../experiments/exposure/OPPORTUNITY_REGIME_V1.md) —
    pre-frozen long-context QB/QK opportunity screen and its negative quality gate.
15. [../experiments/motivation/CAPACITY_V2.md](../experiments/motivation/CAPACITY_V2.md) —
    fixed-task 3x3 joint data/model-capacity motivation over four training seeds.
16. [../experiments/migration/COMPILED_LOW_RANK_V1.md](../experiments/migration/COMPILED_LOW_RANK_V1.md) —
    original aligned three-dataset compiled-projection result.
17. [../experiments/migration/PROGRESSIVE_PREFIX_REPLAY_V1.md](../experiments/migration/PROGRESSIVE_PREFIX_REPLAY_V1.md) —
    replicated no-fit structural baseline on the capacity matrix.
18. [../experiments/migration/COHORT_TIERED_MIGRATION_V1.md](../experiments/migration/COHORT_TIERED_MIGRATION_V1.md) —
    current compiled fast tier, residual structural fallback, and 27-seed capacity validation.

The background figure [streaming_training_kv_cache_background.png](streaming_training_kv_cache_background.png)
explains windows, model versions, full reuse, and full compute. It is explanatory artwork rather
than evidence.

The planning note [system_paper_candidates.md](system_paper_candidates.md) records the current
StreamKV direction—cohort migration compilation, a one-pass capsule-to-KV operator, and a
cohort-streaming multi-GPU runtime—plus older system-paper candidates and same-slot fallbacks. It
is a candidate-design document rather than a source of research claims, and remains subordinate
to the roadmap and evaluation protocol.

## Valid artifact boundary

Only these result families are current:

- `results/validity/core_seed{0,1,2,3}.json`
- `results/validity/multiseed_summary.json`
- `results/validity/core6l_seed{0,1,2,3}.json`
- `results/validity/core6l_summary.json`
- `results/validity/layerwise_seed{0,1,2,3}.json`
- `results/validity/layerwise_multiseed_summary.json`
- `results/validity/layerwise6l_seed{0,1,2,3}.json`
- `results/validity/layerwise6l_multiseed_summary.json`
- `results/validity/interval_oracle_seed0.json`
- `results/validity/interval_validation_seed{1,2,3}.json`
- `results/validity/interval_validation_summary.json`
- `results/validity/streaming_control6l_seed{0,1,2,3}.json`
- `results/validity/streaming_control6l_summary.json`
- `results/motivation_scale/capacity_v2_summary.json`
- `results/motivation_scale/design_discovery_seeds.json`
- `results/motivation_scale/progressive_prefix_replay_v1_summary.json`
- `results/motivation_scale/cohort_tiered_migration_v1_summary.json`
- `results/motivation_scale/structural_design_discovery_summary.json`
- matching `checkpoints/validity/core*_seed*/theta_*.pt`
- `results/scaling/operator_cost_seed0.json`
- `results/scaling/sequence_length_seed{0,1,2,3}.json`
- `results/scaling/update_magnitude_seed{0,1,2,3}.json`
- `results/scaling/depth{3,9}_{core,method}_seed{0,1,2,3}.json`
- `results/scaling/movielens_seed{0,1,2,3}.json`
- `results/scaling/multiaxis_summary.json`
- `results/scaling/kuairand_data_coverage.json`
- `results/scaling/kuairand_factorial_summary.json`
- `results/scaling/kuairand_data_utilization_summary.json`
- matching factorial and `top50k_*` core/method/control files
- `results/taobao/{data_audit,kuairand_matched_comparison}.json`
- `results/dataset_audit/{tenrec_qk,tenrec_qb,zhihurec}.json`
- `results/dataset_audit/*_top50000_users5000_prepared.json`
- `results/dataset_audit/tenrec_qb_top50000_users5000_fixed_horizon_prepared.json`
- `results/dataset_audit/tenrec_qk_top5000_users5000_prepared.json`
- `results/exposure/{qb,qk,zhihu}_streaming_control_summary.json`
- `results/exposure/{qb_fixed_horizon,qk_top5k}_{core,streaming_control}_summary.json`
- `results/exposure/{qb,qk}_method_summary.json`
- `results/exposure/aligned_method_gate_summary.json`
- `results/exposure/qk_top5k_aligned_method_summary.json`
- `results/exposure/{kuai,qb_fixed_horizon,qk_top5k}_allages_streaming_control_summary.json`
- `results/exposure/cache_age_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_{cross_dataset,fine_cross_dataset}_summary.json`
- `results/exposure/long_context_opportunity_summary.json`
- `results/exposure/{qb_horizon256,long_context}_operator_cost_seed0.json`
- `experiments/migration/COMPILED_LOW_RANK_V1.md`
- `experiments/migration/PROGRESSIVE_PREFIX_REPLAY_V1.md`
- `experiments/migration/COHORT_TIERED_MIGRATION_V1.md`
- matching local `results/exposure/*_seed*.json` and `checkpoints/exposure/`
- matching `checkpoints/scaling/`

## Git artifact boundary

The repository tracks the aggregate evidence needed to inspect the claims:

- `results/validity/{multiseed_summary,core6l_summary,layerwise_multiseed_summary,layerwise6l_multiseed_summary}.json`
- `results/validity/{interval_validation_summary,streaming_control6l_summary}.json`
- `results/scaling/{operator_cost_seed0,multiaxis_summary,kuairand_data_coverage}.json`
- `results/scaling/{kuairand_factorial_summary,kuairand_data_utilization_summary}.json`
- `results/taobao/{data_audit,kuairand_matched_comparison}.json`
- `results/dataset_audit/{tenrec_qk,tenrec_qb,zhihurec}.json`
- `results/dataset_audit/*_top50000_users5000_prepared.json`
- `results/dataset_audit/tenrec_qb_top50000_users5000_fixed_horizon_prepared.json`
- `results/dataset_audit/tenrec_qk_top5000_users5000_prepared.json`
- `results/exposure/{qb,qk,zhihu}_streaming_control_summary.json`
- `results/exposure/{qb_fixed_horizon,qk_top5k}_{core,streaming_control}_summary.json`
- `results/exposure/{qb,qk}_method_summary.json`
- `results/exposure/aligned_method_gate_summary.json`
- `results/exposure/qk_top5k_aligned_method_summary.json`
- `results/exposure/{kuai,qb_fixed_horizon,qk_top5k}_allages_streaming_control_summary.json`
- `results/exposure/cache_age_cross_dataset_summary.json`
- `results/exposure/cache_version_matrix_{cross_dataset,fine_cross_dataset}_summary.json`
- `results/exposure/long_context_opportunity_summary.json`
- `results/exposure/{qb_horizon256,long_context}_operator_cost_seed0.json`
- `results/motivation_scale/{capacity_v2_summary,design_discovery_seeds}.json`
- `results/motivation_scale/{progressive_prefix_replay_v1_summary,cohort_tiered_migration_v1_summary}.json`
- `results/motivation_scale/structural_design_discovery_summary.json`

The per-seed/per-user core, method, control, and oracle JSON files listed above are current local
artifacts but are ignored by Git and can be regenerated by the documented scripts. The prior
families are roughly 235 MB; the new full-active exposure runs add roughly 846 MB locally.
Checkpoints are likewise local, ignored artifacts. This distinction keeps the commit reviewable
without treating smoke or retired results as evidence.

The three-layer layerwise family is a correct sanity check. The six-layer layerwise family provides
the four-seed quality result; the interval family supersedes its legacy operator timing and tests
the suffix assumption. Scaling-v1 freezes that operator and changes one axis at a time. Its
MovieLens result is a weak/negative generality check and must not be cited as a successful
cross-dataset reproduction. The factorial and data-utilization families extend KuaiRand scale but
remain separate protocols; top-50k is not full KuaiRand. Smoke outputs are not research artifacts.
Taobao UserBehavior has been rejected as the primary second stream at the target-semantics gate
because it lacks true unclicked impressions. The ordered-exposure experiment advances the new
datasets beyond capacity audit: fixed-horizon QB and top-5k QK now reproduce the same pre-design
streaming/reuse/maintenance age logic as KuaiRand over four seeds. QB/QK remain related Tenrec
tables with ordinal rather than global time, and QB conditions on complete activity availability.
ZhihuRec is a negative maintenance boundary. Method transfer remains mixed: cheap refresh is
positive on the original QB protocol and aligned QK, while the fixed suffix fails to transfer to
aligned QB/QK. A later fit/probe/test-separated method replaces the fixed suffix with a shared
low-rank K/V residual map compiled into one prepacked projection. Its original frozen rule improves
BestRank, rank utility, and NDCG@100 over reuse in all four seeds on KuaiRand, aligned QB, and
aligned QK at `0.106-0.124x` full kernel time. The current capacity-tiered extension adds
residual-delta structural replay and full fallback. Across 27 replication seeds its 50% point costs
`0.121 [0.112, 0.130]x` full and recovers `0.587 [0.547, 0.627]` of the K/V gap, but the strict
task-quality gate passes 6/9 cells because several full-maintenance endpoints are near zero or
negative. This is an operator-level result and a motivation for cohort admission, not yet an
end-to-end serving result.
The fixed-endpoint matrix puts BestRank cache loss on one normalized scale across KuaiRand/QB/QK
and shows local, update-dependent jumps; it does not prove that every tuned periodic policy fails.
The long-context Tenrec screen confirms a larger recomputation/cheap cost separation but fails its
pre-frozen staleness-tax gate, so it is a negative boundary rather than a method setting. The
replicated KuaiRand top-50k/all-chunks cell remains the primary joint quality-cost opportunity.

Early documents `00` through `07`, phase0/JVP experiments, the old three-state serving policy,
and results under `results/phase0` or `results/streaming` were removed because they either describe
a retired route or use superseded evaluation semantics. Tracked versions remain recoverable from
Git history, but they must not be cited as current evidence.

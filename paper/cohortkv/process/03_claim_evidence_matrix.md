# Claim–evidence–design matrix

This file is the claim ledger for the manuscript. A sentence may be weaker than the corresponding
allowed claim, but never stronger.

## Evidence classes

| Code | Meaning | Permitted role |
|---|---|---|
| R | frozen multi-seed replication | primary empirical claim |
| A | adaptive seed-0 real-checkpoint result | design feasibility and controlled evidence |
| C | executable interface/correctness validation | architecture and semantic contract only |
| N | negative result | delimit the design space |
| O | open gate | limitation and future evaluation only |

## Main matrix

| ID | Research observation or claim | Evidence and protocol | Design consequence | Allowed manuscript wording | Forbidden wording |
|---|---|---|---|---|---|
| R1 | Model updates make stale HSTU K/V matter. | Six-layer validity: full fresh improves BestRank over stale reuse by 4.15 on average for one-step and 63.39 for cumulative updates; cumulative 95% CI `[41.57, 85.20]`. Training seed is the replication unit. | Treat cache as versioned derived state. | “Stale reuse can forfeit a measurable part of streaming value.” | “Every update harms every user” or production SLO claims. |
| R2 | Useful streaming learning and cache-maintenance opportunity coexist. | KuaiRand Top-50k/all-chunks: full compute over frozen 3837.67 BestRank, reuse 2952.11, maintenance 885.56; 23.1% staleness tax. | A cheaper state update can target the remaining gap. | Report the partition with protocol name and CIs. | Mix these values with latest-only or another result family. |
| R3 | The opportunity appears beyond one dataset. | Aligned theta5 BestRank full/reuse/maintenance: KuaiRand 484.34/399.02/85.32; QB 94.70/64.38/30.31; QK 47.34/34.34/13.00, all with four seeds. | Motivate a shared mechanism, not a KuaiRand-only heuristic. | “The aligned endpoints show a positive mean maintenance gap on three evaluated tables.” | “Three independent production domains”; QB and QK are related Tenrec tables with ordinal time. |
| R4 | Task-level maintenance is not monotone in model/data capacity. | Frozen 3×3 screen, four seeds per cell. Endpoint BestRank staleness tax ranges from −0.060 to 0.548; large KuaiRand 0.360, large QB 0.548, large QK −0.0048. All nine cells have positive streaming and reuse value in 4/4 seeds, but maintenance signs vary. | Do not gate updates on a predicted task gain. Separate semantic state update from task quality. | “Capacity scales cost cleanly but does not calibrate maintenance utility.” | “Larger models always need more maintenance.” |
| R5 | Age orders state drift better than task utility. | Fixed-endpoint age curves are non-monotone in every capacity cell. Long-context 4+12 seed-0 diagnostic: after removing theta0, age explains 6.15% of MeanRank variation while current version explains 60.9%. | Use explicit source/target version pair; never a universal age window. | “Age is not a calibrated task-quality trigger in the evaluated traces.” | “Every possible fixed window fails in all deployments.” |
| N1 | Per-user drift does not identify who benefits. | Drift–utility correlation 0.020, 95% CI `[−0.012, 0.052]`; JVP/Fisher route is not cheaper in this setting. | Retire per-user admission/prediction route. | One short negative-result paragraph. | Reintroduce drift/JVP/Fisher as project crux. |
| R6 | HSTU exposes a compilable migration surface. | By implementation and model definition, per-layer `K,V = Wk,Wv · Norm(x)`; shared `fresh − cheap` residual is fit and folded into one affine map. | Compile a cohort program over cached old normalized states. | State exact algebra and masking semantics. | Claim equivalence to exact current hidden propagation. |
| R7 | Compiled repair has replicated operator-level cost/fidelity value. | 27 validation runs across KuaiRand/QB/QK × small/medium/large × 3 validation seeds: all select compiled projection; mean cost 0.1211× exact, 95% CI `[0.1118, 0.1304]`; mean K/V gap recovery 0.5867, CI `[0.5466, 0.6267]`; fidelity target in 25/27. | Retain unconditional compiled fast tier. | “Compiled repair scales in measured GPU cost and K/V fidelity.” | “It guarantees task-quality improvement.” |
| R8 | Task signs usually track full maintenance but fail a strict universal gate. | Strict cell gate 6/9. Selected/full same-sign counts: BestRank 23/27, rank utility 27/27, NDCG@100 25/27. When full is positive, selected is positive 18/20, 24/24, 19/20; median positive-gap recovery 0.907/0.972/0.966. | Track current-model semantics; publish progressive fallback. | Use descriptive cross-cell language and exact denominators. | Present task quality as an admission oracle or claim 9/9 success. |
| A1 | A label-free certificate can choose a cheap cohort program. | 4+12 theta0/4/10→theta11 adaptive seed-0; disjoint 40 fit, 60 prior selection, 60 certificate, 522 final. Contract: ≥70% recovery lower bound, ≥80% coverage lower bound, 90% one-sided, cost ≤0.30×. Compiled full affine selected at all ages. | Compiler publishes selected action and ordered fallbacks; exact is terminal fallback. | “In a seed-0 design study, the frozen certificate selected…” | “The compiler is validated across seeds/datasets.” |
| A2 | Verified full-affine programs preserve high current-model semantic fidelity on the held-out final users. | Final cost 0.0638/0.0640/0.0641× exact; K/V recovery 0.8865/0.8911/0.9356 for ages 11/7/1. At age 11, signed gap recovery is 98.8% MeanRank/AUC, 90.3% NDCG@100, 88.9% Hit@100. | Use semantic probes without labels, then test task behavior separately. | Always attach “adaptive seed-0” and disjoint-split details. | Treat the 522 users as independent training replications. |
| A3 | Fusing the affine epilogue changes both kernel and complete host-boundary time. | Representative batch: FP32 reference 3.119 ms; packed FP16 0.838 ms; fused FP16 0.706 ms; FP16 relative K/V error 3.66e−4. One-GPU host path 162.5→138.1 ms, 1.176×. | Dedicated capsule-to-K/V operator with direct contiguous K/V writes. | “On the controlled seed-0 A40 benchmark…” | General hardware-independent kernel speedup. |
| A4 | The operator advantage survives matched host residency and target publication on two GPUs. | 64 real histories, assigned 13/19/32 source mix, 98,252 prefix tokens. Migration reads normalized capsules and exact reads raw histories; both sources are host-resident, and both publish complete pinned-host K/V. Fused LPT: 903.7 records/s; 1→2 GPU scaling 1.951×; 11.22× faster than independently tuned two-GPU BF16 exact. | Keep programs resident, overlap H2D/compute/D2H, bucket lengths, partition extents. | State both source representations, all included/excluded costs, and the assigned trace. | Call the inputs identical, or call the trace full-cohort, organic mixed-version, out-of-core, SSD, or online serving. |
| N2 | Jagged page compaction is not a positive result on the current long-context trace. | v3: jagged output matches dense fused values element-for-element; performance is 1.019× host and 0.984× HBM relative to one-record batches. Direct HBM is 2.159× faster than host only because the endpoint omits D2H. | Keep the lossless jagged layout as a conditional mechanism; do not claim compaction speedup. | Report as negative/sensitivity result. | “Paging accelerates CohortKV.” |
| C1 | A common destination transaction is implemented. | v4 correctness paths for HBM, DRAM, filesystem, and remote reference store; complete duplicate-free record coverage; stage/commit/abort; manifest-last visibility; readback/checksum where applicable. | One explicit destination contract and target-version manifest. | “Interface-validated” or “functional reference backend.” | Physical SSD/network throughput, cross-node durability, or production crash recovery. |
| C2 | Host-staged transformed output is bounded by waves and publication queue after capsule materialization. | v4 implementation and tests. The complete source batch sequence is still caller materialized; HBM retains complete target. | Separate transient working set from source and destination residency. | State exactly which memory is bounded. | “Fully out of core” without the source reader/full-cohort gate. |
| O1 | Full-cohort identical-boundary evaluation is missing. | No current result compares compiled and exact paths through the same v4 HBM/DRAM destination for all eligible records and 1/2/4 GPUs. | Keep engine performance claim open. | Explicit limitation and next experiment. | Fill the table with extrapolated numbers or reuse v2 as v4 evidence. |
| O2 | Physical storage evidence is missing. | POSIX backend is functional; remote uses an in-memory reference store. | Defer device/network claims. | “The interface admits these backends.” | SSD, GDS, RDMA, remote GPU, or network speedups. |
| O3 | Runtime fallback orchestration is missing. | Verified plans serialize ordered residual/structural/exact fallbacks, but v4 currently consumes compiled affine programs and does not dispatch the escalation chain. | Wire selected action and fallback order into the coordinator and destination transaction. | “The plan records ordered fallbacks; automatic dispatch is open.” | “The destination engine automatically escalates failed cohorts.” |

## Contribution-to-evaluation coverage

| Contribution | Strongest current evidence | Current strength | Missing admission gate |
|---|---|---|---|
| Cohort migration compiler | R7/R8 plus A1/A2 | replicated simple compiled operator; sophisticated verified compiler only seed-0 | freeze on new training seeds or accepted external checkpoints |
| Capsule-to-K/V operator | A3/A4 | real checkpoint, matched source residency and target publication, controlled seed-0 | more shapes/hardware and full-cohort endpoint |
| Destination-oriented engine | C1/C2 | executable architecture and correctness | automatic fallback dispatch, common compiled/exact destination path, source streaming, full cohort, 1/2/4 GPUs, physical backend |

## Global prohibited claims

- “Plain prefix/suffix is optimal.”
- “Recent-token rectangles or arbitrary intervals are the active method.”
- “Age, drift, or task quality predicts whether reuse is safe.”
- “CohortKV improves the current model’s ranking quality.”
- “Exact recomputation is a ranking-quality upper bound.”
- “The affine program is equivalent to exact current hidden propagation.”
- “The current system is production deployed or satisfies an industrial SLO.”
- “The current destination engine automatically executes the published fallback chain.”
- “v4 has measured full-cohort, SSD, network, or remote-GPU performance.”
- “The source side is already lazily streamed or globally memory bounded.”
- “The three evaluated data tables are three independent calendar-time production domains.”

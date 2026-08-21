# Yambda-50M bring-up plan

Updated: 2026-08-17

> **Historical completed plan.** 本文记录 Yambda 平台如何建立及哪些早期输出被 invalidated，不能作为当前执行入口。当前状态与授权以 [项目全程 Compact](project_compact.md) 和 [当前路线](current_route.md) 为准。

This is an implementation plan for data bring-up and a minimal HSTU workload. The original v1 calendar-time interpretation was invalidated; v2 is the timestamp-corrected canary contract and must not be used to claim formal paper results.

## Decision

Use the Yambda-50M flat `listens` table as the first external workload. Download `likes`, `dislikes`, `artist_item_mapping`, and the dataset README only as audit/support inputs. Do not download embeddings, `multi_event`, Yambda-500M/5B, VK-LSVD, or RecFlow in this phase.

The route is reasonable because the dataset has long per-user histories, timestamped listening events, explicit organic/recommendation-driven metadata, and enough scale for persistent-state canaries without making data movement the experiment. The route remains conditional on the audit establishing usable temporal coverage and future-prediction opportunity.

## Work packages

1. Acquire and pin data
   - inspect the Hub revision and file sizes with `hf download --dry-run`;
   - download the selected files into `data/raw/yambda/`;
   - record revision, file sizes, SHA-256 hashes, and schemas in a manifest;
   - keep raw files out of git and avoid derived copies by default.

2. Implement the data opportunity audit
   - verify timestamp units/range, sorting, duplicates, nulls, and the `played_ratio_pct > 50` event rule;
   - report global time span, daily density, user/item coverage, history-length quantiles, and candidate 1d/3d/7d windows;
   - report item/artist overlap, recent-vs-long-term popularity, last-item/artist hits, and organic splits without fitting a model;
   - emit `results/data_audit/yambda50m_v1/report.json` and `window_candidates.csv` only after the input manifest is complete.

3. Freeze the first workload contract
   - select one update cadence from 1d/3d/7d using only pre-release density and coverage;
   - retain approximately 300 days of base context and a 30-minute release gap only if the timestamp audit supports it;
   - define first-legal-request and 1/2/4/8/16-request dilution endpoints;
   - freeze the candidate protocol before model-quality comparisons.

4. Minimal HSTU sanity check
   - validate Random, popularity, No-history HSTU, and Full-history HSTU;
   - verify that persistent history changes outputs and that incremental append is materially cheaper than long-history recompute;
   - build only a three-version chain `theta0 -> theta1 -> theta2` after the contract is frozen;
   - compare Current Full, Previous Full, Reuse, and No-history on the same requests.

5. EvoKV base
   - add profiler/controller modules only after the workload contract exists;
   - keep no-op, selective/approximate, and exact paths explicit;
   - do not tune thresholds on future labels or revive deleted D1/D2/D3 runners.

6. θ0 canary
   - use the frozen 210-day foundation and 1-day conditional-reranking manifest;
   - run only a small history-conditioned HSTU sanity check before any full-data training;
   - require loss improvement and retain the output as canary evidence, not a quality claim.

7. Medium-scale θ0 and one-hop chain
   - train a 4-layer/H128 model with history cap 512 on `[0d, 203d)` only;
   - use `[203d, 210d)` for fixed pre-release validation/checkpoint selection;
   - evaluate only on the fresh dev/qualification manifests, never the sanity manifest;
   - after θ0 passes, run exactly two one-hop 1d edges `θ0→θ1` and `θ1→θ2` with one seed.

The formal model-sanity gates are history-conditioned gain and a usable Full path. Natural-order superiority, SASRec superiority, and positive gap on every version edge are explicitly not gates. Full is a current-model fidelity reference, not a ranking upper bound; no-op reuse is a valid result.

## Exit criteria for this phase

- raw input is reproducibly identified and hashed;
- the audit has a machine-readable report and a justified window comparison;
- one workload contract is frozen;
- the minimal HSTU path passes a canary and demonstrates history-sensitive execution;
- no formal EvoKV result is claimed before the three-version protocol is complete.

## Current audit observations

The audit found 46,467,212 listening rows, zero `(uid, timestamp)` ordering violations, and no missing day bins across the observed 0--26,000,000 timestamp-bin range. At the provisional 300-day boundary, users present in base/update/future are 373, 1552, and 2654 for 1d/3d/7d windows.

The original v1 audit multiplied timestamps by five and is retained only as invalidated historical output. In v2, timestamps are treated as seconds rounded to 5-second precision, yielding 300.93 observed days and 301 complete day bins. User-paired bootstrap audits use 12h/1d/3d/7d windows at 180/210/225/240/255/270-day anchors. The 1d item delta is positive at every anchor with adequate coverage; 3d is a useful robustness window; 7d item delta is negative or indistinguishable from zero at most anchors. v2 therefore freezes 1d as primary, 3d as robustness, and keeps 12h exploratory. These are opportunity signals, not model results.

The frozen 256-request candidate manifest uses a 210-day foundation, a 30-minute gap, a 1-day update boundary, and 100 base-popularity candidates with the observed future positive injected and marked as conditional reranking. The tiny θ0 canary reduced candidate loss from 4.6034 to 1.8086 in 30 steps and increased target margin from -0.2602 to 1.3211; this only verifies the training/scoring path and is not a formal recommendation result.

The medium θ0 screen, its first two-edge output, and both compatibility profiles are retained as invalidated development artifacts. The medium-data collator right-aligned event tokens, while the HSTU length mask and `last_hidden` select leading valid positions; those screens therefore did not consume the intended item histories. The first two-edge screen also treated the beginning of each update window as the new-model release, scoring update-window events as current-model append even though the model is released only after that window plus the 30-minute gap. The corrected run uses leading valid tokens and the prior model for every event before `theta_t` release. θ0, θ1, θ2, profiler, and oracle outputs must be regenerated before making quality, compatibility, or controller claims.

The batch-fixed θ0 passed the same history-conditioned sanity gate (qualification CE 3.5639, AUC 0.7874, NDCG@10 0.2066; Full minus empty-history target log-probability 1.0413, bootstrap 95% CI [0.9536, 1.1283]). The release-correct two-edge screen names the old `No-history` path `Current Suffix Only` and separately reports zero-post-release-append and OOV-only-append exclusions. A separate target-independent profiler manifest covers the zero-append first request using an explicit padding-item readout token for fidelity comparison only.

The resulting oracle frontier is development evidence, not a deployable request-time controller: at diagnostic Top-10 overlap-loss operating points the version-level policy must choose Exact All, while an oracle user gate can achieve the same diagnostic fidelity slice with lower recomputed-token fractions than Exact All and than length/activity/random selections. The paper boundary is now release-budgeted state convergence: the next action is to redraw the oracle as a full fidelity–exact-equivalent-work frontier, define a pre-release active-cache snapshot, and develop a pre-release risk ranker. Any controller is qualified only on a new `theta2→theta3` edge.

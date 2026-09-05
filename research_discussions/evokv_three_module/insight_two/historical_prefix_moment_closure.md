# Historical-query prefix-moment closure diagnostic

## Question and claim boundary

This diagnostic asks whether the historical Current-minus-Parent functional
difference can be formed by an all-history associative reduction rather than a
small set of token carriers. It is a falsifiable alternative to sparse causal
closure, not a new Design 1 and not a claim that affine moments themselves are
novel.

The scope is the existing Medium legacy pointwise reader: six layers, hidden
width 192, six heads, context 1024, ELU+1 attention, no relative-position bias,
and one Parent-to-Current edge at a time. The positive ELU+1 branch is affine
inside a fixed sign region. This fact is specific to the tested reader; it is
not the faithful SiLU HSTU equation and must not be generalized to softmax or
arbitrary Transformer attention.

No candidate identity, candidate score, label, future event, fitted map, token
importance score or selected token support enters construction. Every one of
the 1024 historical positions contributes to the dense reduction. Current
Exact traces are allowed only in the oracle diagnostic and never establish an
executable migration action.

## Hypothesis

For layer (l), head (h), historical query (q_i), and historical key/value
((k_j,v_j)), legacy attention is

\[
A_i=\sum_{j\leq i}(\operatorname{ELU}(q_i k_j^\top)+1)v_j.
\]

If historical queries for one user share a stable positive activation region
(m_j\in\{0,1\}), the positive response is approximated by

\[
\widehat A_i^+=B_i+s q_i M_i,
\quad
B_i=\sum_{j\leq i}m_jv_j,
\quad
M_i=\sum_{j\leq i}m_j k_j^\top v_j.
\]

((B,M)) composes by addition, so every prefix can be formed by an
associative scan. The research question is not whether this identity holds in
a fixed region—it does—but whether a small, label-free set of actual
historical query rows defines a region stable enough for paired
Current-minus-Parent response recovery and for layer-by-layer closure.

## Minimal diagnostic

Use the frozen 32 canary users and all five Medium edges only after a new
prospective contract. Do not read users 32 onward for tuning. P8 is the primary
probe count; P32 is a preregistered stability companion. Probes are fixed
equal-width historical-query endpoints, including the final history row. A
recent-tail P8 layout may be reported as a named ablation, never selected by
outcome.

For each user, edge, layer and head, run four measurements.

1. Historical activation-region stability. Trace all exact Current historical
   queries. For Current K and Parent K separately, let only causally eligible
   P8 query rows vote on each key's sign. Report agreement of this shared mask
   with every causal historical QK sign, the per-query agreement p10, agreement
   with the all-query oracle majority, unanimous-key fraction and negative-logit
   fraction. The Current and Parent masks remain separate, and their direct
   cross-version agreement is reported rather than assumed.

2. Teacher-forced paired response. Feed the same exact Current historical Q
   into Current K/V and Parent K/V. Compare the exact native response delta
   with the difference of the two shared-region prefix-moment reads. Report
   unclipped relative-L2 recovery and cosine per layer/head, plus the exact
   negative-branch response norm. This isolates representability from rollout
   error.

3. Closed prefix rollout. Starting from raw Current embeddings, derive the
   layer's P8 region from the rollout's own historical Q/K, build all-prefix
   moments, update every historical token, and pass the resulting hidden state
   to the next layer. Compare response heads, K and post-layer hidden state to
   Current Exact after every layer. A strong teacher-forced result with a
   collapsing rollout means that the representation is not causally closed.

4. Associativity and cost. Verify numerically that summaries of adjacent
   chunks combine to the full summary and that batch prefix scans match a
   sequential recurrence. Report full-token transforms, two-arm probe QK,
   two-arm moment construction, two-arm reads, sign comparisons, transient
   prefix storage and persistent delta-moment storage separately.

Aggregate per user first, then user-equal within each edge, then edge-equal.
Report every edge and layer. Do not clip recovery, remove small-gap users or
select layers after observing results.

## Decision logic

This diagnostic deliberately permits three conclusions.

- If historical regions are stable, teacher-forced paired recovery is high,
  closed rollout remains high through the final layer, and total release-time
  compute is at most 20% of Exact-All, all-history prefix closure becomes a
  Design candidate.
- If representation and closure are high but the dense token-transform floor
  is above 20%, retain the Transformer insight—distributed token error becomes
  an associative functional reduction—but reject dense closure as Design 1.
- If region agreement or closed rollout fails, reject this functional boundary
  even if an Exact-mask oracle looks good.

The second outcome is important. Replacing quadratic attention by prefix
moments does not eliminate Q/K/V, output and gate projections for all 1024
tokens. A compact persistent object therefore does not imply a cheap
constructor. The strict cost report must expose this rather than describing
the moments alone as a 1.6% design.

### Static Medium cost result

The implemented FLOP audit already resolves the Design-budget question before
any quality result is read. Under the repository's existing matmul/add counting
convention:

- Exact-All is 4,771,282,944 FLOPs per user;
- the full-token input and block-linear floor alone is 49.3160% of Exact-All;
- paired P8 prefix closure is 2,701,000,704 FLOPs, or 56.6095%; and
- paired P32 prefix closure is 2,814,246,912 FLOPs, or 58.9830%.

The persistent delta moments are only 38,016 FP32 scalars, 1.6113% of full
Current K/V, but their dense constructor is outside the 0–20% Design 1 region.
Therefore this route cannot become the primary migration design under the
current contract even if its recovery is excellent. A later GPU diagnostic is
justified only to establish or reject the Transformer insight that historical
functional error admits associative closure; it must not be presented as an
executable Design candidate without a new way to avoid full-token transforms.

## Implementation

`scripts/insight_two/historical_prefix_moment_closure.py` contains:

- exact historical layer tracing;
- fixed causal historical-query region construction and stability metrics;
- associative prefix moment build/read and segment combination;
- teacher-forced Current-minus-Parent response diagnostics;
- closed all-history prefix rollout; and
- strict Medium-compatible FLOP and storage accounting.

The implementation is CPU-unit-testable and has no data-loading or launch side
effects. A formal GPU runner, contract, thresholds and raw-result directory
must be added only after this mechanism protocol is reviewed. No GPU run is
authorized by this document.

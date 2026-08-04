# Selected cross-dataset stream checkpoint chains

Date: 2026-08-04

Status: **QK and QB development chains are selected, registered, and locally retained; QK
recursive Round A is complete and QB confirmation is next.** D2/D3 consume the selected D1 only
after the cross-dataset confirmation boundary closes. This
document replaces the completed parameter-search handoffs. It is an operational checkpoint
ledger, not a formal result protocol.

The machine-readable source is
`configs/evokv_foundation/selected_checkpoint_registry_development_v0.json`. Before any new
experiment consumes these checkpoints, run:

```bash
python scripts/verify_evokv_selected_checkpoints.py
```

The quick check hashes every manifest and compact source result and validates every payload path
and byte count. Add `--full-payload` when moving disks, copying checkpoints, or freezing a formal
round; that mode also hashes all dense, projection, embedding, bitmap, and optimizer payloads.

## 1. Why two chains are retained

The selected chains solve different experimental needs while sharing the same 24-layer,
H1536/E4096 large-core structure:

| Chain | Registered versions | Evaluated versions | Recursive edges | Role |
|---|---|---|---:|---|
| Tenrec QK LR0.15 | `theta0–theta4` | `theta1–theta4` | 3 | primary D1 design/selection chain |
| Tenrec QB `u30_e3` | `theta0–theta3` | `theta1–theta3` | 2 | locked cross-dataset D1 confirmation and later D2/D3 stressor |

The seven evaluated large models are therefore four sequential QK versions and three sequential
QB versions. They are not concurrently served models and are not seven independent seeds. At any
instant the system serves exactly one current recommendation model; version labels bind K/V
lineage across its next update. Both `theta0` checkpoints remain bootstrap/rebuild prerequisites,
but their warm-up edges are excluded from D1 evidence.

QK remains the system headline because it also owns the natural-length HET/HOM universe and the
paper-scale capacity cohorts. QB is not a replacement for QK. It is valuable because its nine
base-only feature namespaces create a separately capacity-forced embedding and its two positive
edges let us test whether a mechanism survives a different ordered-exposure table.

These are development-selected chains. They are sufficient for implementation, causal
diagnostics, baseline construction, and mechanism search. They are not yet cross-seed formal paper
evidence.

The canonical fixed-state capacity ledger is:

| Quantity | QK `theta1–4` | QB `theta1–3` |
|---|---:|---:|
| Physical embedding shape | 2,859,836×4,096 | 2,985,071×4,096 |
| Physical embedding parameters | 11,713,888,256 | 12,226,850,816 |
| Dense plus projection parameters | 291,863,040 | 291,863,040 |
| Physical fixed parameters | 12,005,751,296 | 12,518,713,856 |
| Physical fixed FP32 bytes | 48,023,005,184 | 50,074,855,424 |
| Physical fixed FP32 GiB | 44.7249 | 46.6358 |
| Optimizer-active semantic rows | 2,859,736 | 2,985,070 |
| Optimizer-active fixed FP32 bytes | 48,021,366,784 | 50,074,839,040 |

Physical counts include one padding row. Optimizer-active counts exclude padding and, on QK, 99
semantic rows without a real optimizer update. These are logical parameter bytes, not the slightly
larger serialized checkpoint-file sizes. Every evaluated version within one chain has the same
geometry and capacity.

## 2. Retained QK chain

### Model and stream

- dataset: Tenrec QK, user-local ordinal exposure order;
- model: 24L/H1536, 24 heads, E4096 owner projection, maximum context 512;
- embedding: 2,859,835 semantic rows plus one padding row, 46,855,553,024 FP32 bytes;
- stream: one `theta0→theta1` objective-alignment warm-up followed by three ordinary eight-token
  updates;
- training: 16,384 users, one epoch/update, dense/projection LR `1.5e-5`, embedding LR `1.5e-4`;
- quality: 4,096 disjoint qualification users, 999 frozen negatives, next-unseen windows;
- cache endpoint: FP16 K/V storage followed by FP32 consumption.

The ordinary Exact-over-Reuse sampled-CE gaps are:

| Edge | Gap |
|---|---:|
| `theta1→theta2` | 0.03082 |
| `theta2→theta3` | 0.01450 |
| `theta3→theta4` | 0.01095 |

Every ordinary edge has a positive record-cluster interval. This is the canonical large-QK
opportunity chain for current D1 quality work and for creating fixed D2/D3 stack revisions.

### Checkpoints

The bootstrap is retained separately:

```text
checkpoints/evokv_xp_qk_e4096_h1536/seed0/theta_0
```

Selected updates are retained under:

```text
checkpoints/evokv_xp_qk_e4096_h1536/quality_rounds/
  quality_lr_dual_20260802_round1_lr015/theta_{1,2,3,4}
```

Manifest hashes are:

```text
theta0  4ec72fea8d790f53ff6f9b27f64b8d0b59616876a7a4e3d1bae4ccfbdada2ae0
theta1  0c9ee3f92ee0fc5c0d47d5760b2b4f823dca7eb42f09cf44a828341bf70b0755
theta2  9ccfba0cfd4478803129813fc0dae97f0771badd038445b2e311751d1d2b93bd
theta3  278abc562040831692ea8b7c139750aab137f856cfb7fe0b1a87b2742b512370
theta4  65ddf21e82df4fa046e0b668b057f98af2e1c02403b0f6d667f766b2d80fde51
```

The selected update payload is about 179 GiB; the common bootstrap is about 44.7 GiB. Compact
training, quality, and binding results live under
`results/baseline_rounds/quality_chain/quality_lr_dual_20260802_round1_lr015/`.

## 3. Retained QB chain

### Model and stream

- dataset: Tenrec QB, user-local ordinal exposure order;
- frozen users: 3,500 train, 500 tuning, and 1,000 report-only qualification users;
- model: 24L/H1536, 24 heads, E4096 owner projection, maximum context 512;
- input: nine real base-only feature namespaces and item-only prediction scoring;
- embedding: 2,985,070 optimizer-active semantic rows plus one padding row;
- fixed FP32 model bytes: 50,074,839,040, above the frozen one-A40 allocatable budget;
- base: one epoch with dense/projection LR `1e-4` and embedding LR `1e-3`;
- updates: three epochs with dense/projection LR `3e-5` and embedding LR `3e-4`;
- evaluation: Frozen, current-model Reuse, and current-model Exact on the next unseen window.

`theta0→theta1` is a warm-up and is not used as a D1 opportunity edge. The retained ordinary
edges are:

| Edge | Tuning CE gap | Qualification CE gap |
|---|---:|---:|
| `theta1→theta2` | 0.02409 | 0.02246 |
| `theta2→theta3` | 0.01297 | 0.01024 |

The attempted `theta3→theta4` edge and all bounded final-update searches were negative. They are
preserved only as compact diagnostics; no theta4 payload is retained or required.

### Checkpoints

Because the family arose from a bounded screen, its physical upstream is split across two roots:

```text
theta0     checkpoints/evokv_qb_large_mf9_e4096/qb_large_round1/u15_e1/theta_0
theta1–3   checkpoints/evokv_qb_large_mf9_e4096/qb_large_round1/u30_e3/theta_{1,2,3}
```

Manifest hashes are:

```text
theta0  40cd358f20619cc99b4c21d72dfb84e23dcf34a66e0b6158ce326a65c2424f2c
theta1  66d2e7cc04f60a0ee5dd1c3fb7776f2d6bc944d0332175a5ba8e302aa1110594
theta2  ec95daf7b702a4ec1c6f2ac13640a188a7fd9775a05ef35dfe47c48c9a0ab4f5
theta3  928b4ead1e0d1acd5fea94bb419162fa34390bd41da1f777d44799479d59049c
```

Optimizer resume points after theta0, theta2, and theta3 are retained. They are useful for exact
reproduction or a predeclared continuation, but D1/D2/D3 inference experiments must not load
optimizer state. The selected checkpoint plus resume assets occupy about 193 GiB.

Compact source results are:

```text
results/baseline_rounds/qb_large/qb_large_round1/u30_e3.json
results/baseline_rounds/qb_large/qb_large_round1/u30_e3_continuation/theta2_theta3.json
```

## 4. What may vary in downstream experiments

A checkpoint fixes model parameters, embedding rows, and model-version identity. It does not fix a
single K/V workload. Downstream experiments may reuse the chain with:

- different record populations from the same frozen semantic universe;
- natural or controlled valid-history distributions up to context 512;
- different valid-byte capacity cohorts;
- different fixed D1 action budgets;
- 1/2/4-rank physical layouts derived from the same global-row identity.

Every such experiment must regenerate and bind its own raw histories, exact K/V endpoints,
ActionPlan, WavePlan/ResidencyPlan, owner map, capacity cut, quality records, and output digest.
It must not reuse a plan compiled for another edge, record set, length distribution, or checkpoint.

For current development:

- preserve the completed stateful QK `theta1→theta2→theta3→theta4` Round-A result and use
  edge-local exact-source rows only as oracle-reset diagnostics;
- freeze the QK-supported 10% RACT-KV system-design point for both QB ordinary edges;
- treat the old serialized rollout-bound field as a diagnostic rather than an accuracy theory or
  QB selection gate;
- do not tune the method on QB or resume the deleted QB theta4 search;
- keep the D1 action snapshot immutable inside a D2/D3 isolation revision;
- if mechanism discovery changes D1 actions or D2 layout, record a new `stack_revision` and rerun
  the baselines for that revision.

## 5. Rebuilding the chains

The reusable entry point is:

```bash
EVOKV_CUDA_VISIBLE_DEVICES=0,1 \
  scripts/run_evokv_selected_checkpoint_rebuild.sh qk NEW_QK_ROUND

EVOKV_CUDA_VISIBLE_DEVICES=0,1 \
  scripts/run_evokv_selected_checkpoint_rebuild.sh qb NEW_QB_ROUND
```

Set `EVOKV_PREFLIGHT_ONLY=1` to check the QB foundation, disk, and GPUs without training. QK also
passes the flag through to its existing-config runner.

The QK round reuses the registered theta0 bootstrap and rebuilds the fixed LR0.15 theta1–theta4
chain. It takes about 75–110 minutes and adds about 179 GiB. The QB round builds one coherent
theta0–theta3 root using the selected `u30_e3` policy. It takes about 70–120 minutes and adds about
195 GiB. Both refuse to overwrite an existing round, freeze input hashes, write separate logs and
machine-readable results, and persist no full K/V payload.

If the common QK theta0 has also been removed, use the explicit cold-bootstrap mode:

```bash
EVOKV_CUDA_VISIBLE_DEVICES=0,1 \
EVOKV_REBUILD_QK_BOOTSTRAP=1 \
  scripts/run_evokv_selected_checkpoint_rebuild.sh qk NEW_QK_ROUND
```

This first trains the historical H1536 source from the retained prepared corpus, expands and
trains the capacity-forced E4096 theta0, validates its active-row gate and manifest, and only then
runs the selected updates. The H1536 source payload is transient and is deleted after theta0
validation; its compact result, log, input hashes, and validation record remain. Budget about
1.5–2.5 additional hours, about 35 GiB of transient checkpoint space, and about 44.7 GiB of new
durable theta0 space. With `EVOKV_PREFLIGHT_ONLY=1`, this mode only validates prerequisites and
the larger disk envelope.

Rebuilding does not silently replace this registry. Review the new result and manifest hashes,
then update the registry in a separate revision. This prevents a rerun under changed source code
from mutating the current experimental identity.

The QB foundation itself is regenerated with:

```text
scripts/audit_evokv_qb_large_foundation.py
scripts/build_evokv_qb_large_multifield.py
src/hstu_kvcache/data/qb_large_multifield.py
src/hstu_kvcache/streaming/qb_multifield_training.py
src/hstu_kvcache/streaming/multifield_projected.py
```

The frozen foundation summaries and processed corpus/catalog paths are listed in the registry and
checked by the rebuild preflight.

## 6. Retention and cleanup

The 2026-08-03 cleanup permanently removed about 181.7 GiB:

- the rejected QB `u15_e3` checkpoint tree;
- QB `u30_e3/theta_4`, its optimizer, and orphaned edge ledgers;
- two historical fixed-512 D3-specific model trees;
- empty checkpoint directories and stale zero-byte lock files.

The deletion did not remove compact results, logs, negative-screen summaries, data manifests, or
reusable D3 mechanism code. Deleted payloads are not recoverable from a trash directory; they are
regenerable from the retained inputs and scripts. The exact path/byte ledger is in the machine
registry.

One-off QB continuation/theta4-screen launchers, their special-case summarizers, the superseded
QK dual-LR launcher, and obsolete QB horizon-extension configs were also removed. The reusable
dataset builders, trainers, D1 candidate runners, selected configs, compact negative results, and
the canonical `run_evokv_selected_checkpoint_rebuild.sh` entry remain. Completed zero-byte lock
files were removed; a future runner creates its own lock in its new result namespace.

The KuaiRand long-context and motivation-capacity checkpoint roots remain because they still bind
frozen D1 and motivation evidence. They should be reviewed only when those result families are
replaced, not deleted merely because they are not part of the QK/QB selected chain.

Never retain full old or target K/V arrays between stages. Preserve model checkpoints likely to be
reused, compact programs/plans, lineage ledgers, configurations, logs, hashes, and result
summaries. Derived 1/2/4-rank layouts are transient unless a later storage review explicitly
promotes one.

## 7. Next execution boundary

Checkpoint selection is no longer the active task. QK Round A has completed the first recursive
D1 boundary in `future_design/DESIGN1_RECURSIVE_KV_MIGRATION.md`; its 10% RACT-KV route is the
current development design point. The next work is:

1. preserve the completed v0 QK result and its historical automatic-selector outcome;
2. freeze a new QB confirmation protocol in which recursive K/V/task recovery, logical scheduled
   Exact work, and lineage are primary, while the old rollout “certificate” is diagnostic;
3. run the frozen rank-16, ridge, fit-size, sampled-token, direct-old-K/V, and 10% renewal settings
   on QB `theta1→theta2→theta3` without QB-specific tuning;
4. freeze the confirmed D1 ActionPlan/program contract before any new D2/D3 performance round.

Qualification in `11_benchmark_qualification.md` blocks formal promotion, not this mechanism and
baseline work.

The completed QK execution binding is
`configs/evokv_d1/development/qk_recursive_round_a_two_gpu_v0.json`; its compact result is under
`results/baseline_rounds/quality_chain/recursive_d1_round_a/qk_recursive_d1_round_a_20260804_round1/`.
Do not reuse its v0 protocol string for QB or formal repeats.

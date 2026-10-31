"""Explicit configuration and small atomic records, without fingerprint gates."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "experiments/recflow_full_population/data"
CHAIN = ROOT / "experiments/recflow_continuous_chain/chain_manifest.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def configuration():
    return dict(
        protocol="independent_release_once_repair_persistent_causal_fifo",
        dataset="RecFlow", model_scale="max", history_limit=1024,
        layer_intervals_one_based=[[1, 2], [6, 7], [11, 12], [1, 3], [6, 8],
                                   [11, 13], [1, 5], [9, 13], [1, 6], [1, 10], [1, 13]],
        tokens=[64, 128, 256, 512, 768],
        kv_map=[dict(fit_users=n, width=w) for n, w in
                [(32, 40), (64, 64), (96, 96), (128, 128), (192, 160), (256, 184)]],
        seed=17, population_seed=20261030, candidate_chunk=128,
        fit=dict(epochs=8, batch_rows=8192, learning_rate=.001, weight_decay=.0001,
                 beta1=.9, beta2=.999, epsilon=1e-8, normalization_floor=1e-5),
        bootstrap_replicates=1000, bootstrap_seed=17021,
        fitting_role="cache_fit", quality_role="train",
        primary_metric="request_macro_observed_positive_ndcg@10",
        secondary_metric="request_macro_observed_positive_recall@10",
        zero_positive_requests="Retained in coverage; excluded from undefined ranking metric denominator",
        inactive_users="Retained in fixed population and paired UID bootstrap",
        history_visibility="Logged exposure events with timestamp strictly before a request; same-timestamp events enter only afterwards. Exposure timestamp is the existing offline visibility assumption, not measured feedback arrival time.",
        maintenance="Include every event before the end of the complete day, including after the last score and users without scoring requests",
        arithmetic="float32 with TF32 disabled; method batching does not change individual method FLOP charges",
        hash_protection=False, automatic_successors=False,
        scope="Preparation and bounded benchmarks authorized; no automatic formal evaluation",
    )


def policies(cfg=None):
    cfg = cfg or configuration()
    rows = [dict(tag="reuse", method="reuse"), dict(tag="rebuild", method="rebuild")]
    rows += [dict(tag=f"layer_{a}_{b}", method="layer", interval=[a-1, b-1])
             for a, b in cfg["layer_intervals_one_based"]]
    rows += [dict(tag=f"{kind}_{n}", method=kind, tokens=n)
             for kind in ("tail", "deviation") for n in cfg["tokens"]]
    rows += [dict(tag=f"kv_map_u{p['fit_users']}_w{p['width']}", method="kv_map", **p)
             for p in cfg["kv_map"]]
    return rows


def ranking_metrics(ids, labels, scores, ks=(5, 10, 20)):
    """All methods rank all candidates; deterministic raw-ID tie breaking."""
    ids, labels, scores = np.asarray(ids), np.asarray(labels), np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(ids) or labels.shape != ids.shape:
        raise ValueError("Expected scores[methods,candidates] and complete candidate labels")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("Candidate duplicates or nonfinite scores")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Unknown feedback encoding")
    by_id = np.argsort(ids, kind="stable")
    order = by_id[np.argsort(-scores[:, by_id], axis=1, kind="stable")]
    top = order[:, :min(max(ks), len(ids))]
    gains = labels[top] == 1
    n_positive = int((labels == 1).sum())
    result = np.full((len(scores), len(ks)*2), np.nan, dtype=np.float64)
    if n_positive:
        discount = 1 / np.log2(np.arange(2, max(ks)+2))
        for j, k in enumerate(ks):
            gain = gains[:, :k]
            result[:, 2*j] = (gain*discount[:gain.shape[1]]).sum(1)/discount[:min(k, n_positive)].sum()
            result[:, 2*j+1] = gain.sum(1)/n_positive
    return result, top


METRIC_NAMES = [name for k in (5, 10, 20) for name in (f"ndcg@{k}", f"recall@{k}")]

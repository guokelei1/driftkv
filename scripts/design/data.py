"""Fixed disjoint Design users and existing causal Medium data primitives."""

import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from evaluate_yambda500m_foundation_raw import load_histories, load_model
from insight_two.common import (
    DATASET,
    DAY,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    checkpoint,
    load_frozen_inputs,
    verify_model_payload,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "data/manifests/evokv_design_medium_v0/split.json"
REQUEST_ROOT = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1"
V5_REQUEST_ROOT = ROOT / "data/manifests/yambda500m_medium_hstu_native_d14_v5_extension_v1"


def diagnostic_admissions(targets):
    """Reference sealed Full-only decisions; never promote a serving lineage.

V5's historical E14 tail is incomplete. Its separate Full-only record permits
the requested Design diagnostic chain, retaining that partial-tail boundary.
"""
    admissions = {}
    for target in range(1,targets+1):
        base = checkpoint(0).parent.parent / "D14"
        if target < 5:
            path = base / f"admission/v{target-1}_to_v{target}.seal.json"
            value = json.loads(path.read_text())
            assert value["reuse_unlocked"] and value["all_metric_gates_pass"]
            admissions[str(target)] = dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),serving_lineage_promoted=False)
        else:
            path = base / "v5_extension_v1/full_only/E14_partial/v4_to_v5/extension.seal.json"
            value = json.loads(path.read_text())
            report_path = path.parent/"adjudication.json"
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            assert value["adjudication_sha256"] == digest
            assert value["partial_tail_diagnostic"] and not value["serving_admission"]
            report = json.loads(report_path.read_text())
            parent = report["parent_absolute"]["hstu_native"]
            child = report["candidates"]["v5"]
            current = child["absolute"]["hstu_native"]
            ci = child["paired_release_gain"]["parent_minus_current_log_loss"]["user_cluster_bootstrap_95CI"]
            gates = dict(auc=current["ROC_AUC"]>parent["ROC_AUC"],
                log_loss=current["log_loss"]<parent["log_loss"],
                brier=current["Brier"]<=parent["Brier"],positive_ci=ci["p2_5"]>0)
            assert all(gates.values()), "V5 Full-only diagnostic gates did not pass"
            admissions[str(target)] = dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                full_only_adjudication_sha256=digest,gates=gates,
                scope="Design diagnostic sequence only; E14_partial; not serving admission",
                serving_lineage_promoted=False,partial_tail_diagnostic=True)
    return admissions


def fixed_split():
    if SPLIT_PATH.exists():
        return json.loads(SPLIT_PATH.read_text())
    uids, _, _ = load_frozen_inputs()
    users = pq.read_table(DATASET.parent / "users.parquet",
                          columns=["uid", "selector_rank", "n_theta0"]).to_pandas()
    # This conservative eligibility uses only counts predating the first release.
    eligible = users[(users.n_theta0 >= 1024) & ~users.uid.isin(uids)]
    ordered = eligible.sort_values(["selector_rank", "uid"]).uid.astype(int).tolist()
    split = dict(
        status="uid_frozen_before_first_method_output", seed=17,
        eligibility="n_theta0 >= 1024; excludes entire existing 3000-user Insight cohort",
        order="existing label-free selector_rank, uid",
        development=[int(uid) for uid in uids[:512]],
        reserved_legacy_confirmation=[int(uid) for uid in uids[512:]],
        confirmation=ordered[:2048], calibration=ordered[2048:],
        confirmation_target_outputs_read=False,
        cost_target_population=30000,
        cost_target_role="fixed Medium population; extrapolations are not measured population costs",
    )
    groups = [set(split[name]) for name in ("development", "reserved_legacy_confirmation", "confirmation", "calibration")]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i+1:])
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=2) + "\n")
    return split


def frozen_model(version, device):
    model, payload = load_model(checkpoint(version), device)
    verify_model_payload(payload)
    expected = dict(block_variant="legacy", activation="elu_plus1", relative_position_bias=False,
                    gating="silu_gate", causal_diagonal="inclusive")
    assert all(payload["config"][key] == value for key, value in expected.items())
    model.requires_grad_(False)
    return model


def stratified_calibration(split, count):
    """Sample existing calibration users by pre-release population history."""
    users = pq.read_table(DATASET.parent / "users.parquet", columns=["uid", "n_theta0"]).to_pandas()
    users["stratum"] = np.searchsorted([256, 1024, 4096], users.n_theta0, side="right")
    population = users.groupby("stratum").size().reindex(range(4), fill_value=0).to_numpy()
    allocation = count * population / population.sum()
    quota = np.floor(allocation).astype(int)
    for i in np.argsort(-(allocation-quota), kind="stable")[:count-int(quota.sum())]:
        quota[i] += 1
    available = users[users.uid.isin(split["calibration"])]
    groups = []
    for stratum, n in enumerate(quota):
        candidates = available[available.stratum == stratum].uid.astype(int).tolist()
        candidates.sort(key=lambda uid: hashlib.sha256(f"evokv-calibration-stratified:17:{uid}".encode()).digest())
        assert len(candidates) >= n
        groups.append(candidates[:n])
    # The early lifetime subset also covers every history stratum.
    chosen = [uid for row in zip_longest(*groups) for uid in row if uid is not None]
    assert len(chosen) == len(set(chosen)) == count
    assert not set(chosen) & (set(split["development"]) | set(split["confirmation"]))
    return chosen


def histories(uids, end_day):
    return load_histories(uids, oov_buckets=OOV_BUCKETS, dataset_path=DATASET,
                          known_vocab_size=KNOWN_ITEMS, end_timestamp=end_day * DAY,
                          threads=8)


def prefix(history, uid, timestamp, device):
    items, actions, times = history.prefix(uid, timestamp, 1024)
    if not len(items):
        raise ValueError("selected Design user has no causal prefix")
    deltas = np.zeros(len(times), dtype=np.float32)
    deltas[1:] = np.diff(times)
    return (torch.tensor(items[None], device=device),
            torch.tensor(actions[None], device=device),
            torch.tensor(deltas[None], device=device), times)


def calibration_candidates(history, uid, cutover, count=16):
    items = history.prefix(uid, cutover, 1024)[0]
    recent = list(dict.fromkeys(int(i) for i in items[::-1] if 0 < i < KNOWN_ITEMS))[:count // 2]
    rng = np.random.default_rng(17 + uid)
    while len(recent) < count:
        candidate = int(rng.integers(1, KNOWN_ITEMS))
        if candidate not in recent:
            recent.append(candidate)
    return recent


def quality_requests(uids, start, stop):
    path = (V5_REQUEST_ROOT if start >= 287*DAY else REQUEST_ROOT) / "requests_quality.parquet"
    table = pq.read_table(path, filters=[("uid", "in", uids),
                            ("time_block", "=", "matrix_horizon"),
                            ("target_known", "=", True),
                            ("query_timestamp", ">=", start),
                            ("query_timestamp", "<", stop)],
                          columns=["request_id", "uid", "query_timestamp", "item_idx", "label"])
    rows = table.to_pandas().sort_values(["uid", "query_timestamp", "item_idx", "request_id"])
    return {uid: group.to_dict("records") for uid, group in rows.groupby("uid", sort=False)}

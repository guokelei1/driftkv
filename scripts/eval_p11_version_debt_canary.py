#!/usr/bin/env python3
"""Evaluate direct age-2 and true recursive mixed lineage at theta2 cutover."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import adjudicate_p9_heldout_rolling_quality as quality
import eval_p8_release_raw as p8raw
import eval_p9_cutover_profiler_raw as profiler
import train_p7_theta0 as p7
from hstu_kvcache.models import append_with_rolling_cap


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_0_version_debt_contract_v1.yaml"
POPULATION = ROOT / "data/manifests/p9_full_population_v1"
OUTPUT = ROOT / "results/p11/p11_0_version_debt_canary_v1"
CUTOVER1 = 19958400
CUTOVER2 = 21168000
ACTIONS = ("one_hop", "direct_age2", "recursive_mixed")


def validate() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    paths = {
        "full_stack_freeze_sha256": ROOT / "configs/contracts/p10_6_full_stack_freeze_v1.yaml",
        "r1_edge2_raw_seal_sha256": ROOT / "results/p8/r1_edge2/raw_score_seal_v1.json",
        "r1_edge2_adjudication_sha256": ROOT / "results/p8/r1_edge2/hs_adjudication_v1.json",
        "edge1_manifest_sha256": POPULATION / "edge1/manifest.json",
        "edge2_manifest_sha256": POPULATION / "edge2/manifest.json",
        "edge2_probe_sha256": POPULATION / "edge2/cutover_probes.parquet",
    }
    for key, path in paths.items():
        if p7.sha256_file(path) != contract["inputs"][key]:
            raise RuntimeError(f"P11 input mismatch: {key}")
    return contract


def raw_events(reader, start: int, end: int, uid: int):
    if start == end:
        return []
    table = reader.rows(start, end)
    if not np.all(table["uid"].to_numpy(zero_copy_only=False) == uid):
        raise RuntimeError("recursive suffix crossed uid")
    return list(zip(
        table["timestamp"].to_pylist(), table["item_id"].to_pylist(), table["is_organic"].to_pylist(), strict=True
    ))


@torch.no_grad()
def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = validate()
    model_name, seed = contract["canary"]["model"], int(contract["canary"]["seed"])
    edge1 = {int(row["uid"]): row for row in pq.read_table(POPULATION / "edge1/states.parquet").to_pylist()}
    edge2 = {int(row["uid"]): row for row in pq.read_table(POPULATION / "edge2/states.parquet").to_pylist()}
    probes = {int(row["uid"]): row for row in pq.read_table(POPULATION / "edge2/cutover_probes.parquet").to_pylist()}
    uids = sorted(set(edge1) & set(edge2), key=lambda uid: hashlib.sha256(str(uid).encode()).digest())[: int(contract["canary"]["users"])]
    checkpoint2 = p8raw.TRAIN_ROOT / "r1_edge2" / f"{model_name}_seed{seed}" / "selected.pt"
    theta2, child2 = p8raw.load_model(checkpoint2, torch.device("cuda:0"))
    checkpoint1 = ROOT / child2["parent_checkpoint"]
    theta1, child1 = p8raw.load_model(checkpoint1, torch.device("cuda:0"))
    checkpoint0 = ROOT / child1["parent_checkpoint"]
    theta0, _ = p8raw.load_model(checkpoint0, torch.device("cuda:0"))
    reader = profiler.RawStateReader()
    rows = []
    max_self = 0.0
    suffix_events = []
    for uid in uids:
        row1, row2 = dict(edge1[uid]), dict(edge2[uid])
        row1["cutover"], row2["cutover"] = CUTOVER1, CUTOVER2
        items1, behaviors1, deltas1, last1 = profiler.state_tensors(reader, [row1], torch.device("cuda:0"))
        items2, behaviors2, deltas2, last2 = profiler.state_tensors(reader, [row2], torch.device("cuda:0"))
        cache0_at1 = theta0.compute_kv(items1, behaviors1, deltas1)
        suffix = raw_events(reader, int(row1["raw_prefix_end_exclusive"]), int(row2["raw_prefix_end_exclusive"]), uid)
        suffix_events.append(len(suffix))
        recursive = cache0_at1
        if suffix:
            timestamps = np.asarray([event[0] for event in suffix], dtype=np.int64)
            if np.any(np.diff(timestamps) < 0) or timestamps[0] < last1[0] or timestamps[-1] >= CUTOVER2:
                raise RuntimeError("recursive suffix is not chronological pre-cutover data")
            delta = np.zeros(len(suffix), dtype=np.float32)
            delta[0] = np.clip(timestamps[0] - last1[0], 0, 7 * 86400)
            if len(suffix) > 1:
                delta[1:] = np.diff(timestamps).clip(0, 7 * 86400)
            suffix_tensors = (
                torch.tensor([[event[1] for event in suffix]], dtype=torch.long, device="cuda:0"),
                torch.tensor([[1 + (1 - event[2]) for event in suffix]], dtype=torch.long, device="cuda:0"),
                torch.tensor(delta[None, :], dtype=torch.float32, device="cuda:0"),
            )
            recursive = append_with_rolling_cap(theta1, recursive, *suffix_tensors, max_length=512)
        if recursive.seq_len != int(row2["effective_prefix_length"]):
            raise RuntimeError(f"recursive final length mismatch for uid {uid}")
        current_cache = theta2.compute_kv(items2, behaviors2, deltas2)
        cache1 = theta1.compute_kv(items2, behaviors2, deltas2)
        cache0_direct = theta0.compute_kv(items2, behaviors2, deltas2)
        candidates = torch.tensor([probes[uid]["candidate_ids"]], dtype=torch.long, device="cuda:0")
        query_delta = torch.tensor([CUTOVER2 - last2[0]], dtype=torch.float32, device="cuda:0").clamp(0, 7 * 86400)
        lengths = torch.tensor([current_cache.seq_len], dtype=torch.long, device="cuda:0")
        query_type = torch.tensor([2], dtype=torch.long, device="cuda:0")
        def score(cache):
            return theta2.score_cc_reuse(cache, candidates, query_delta, prefix_lengths=lengths, query_type_ids=query_type).float()[0]
        current = score(current_cache)
        max_self = max(max_self, float((score(current_cache) - current).abs().max()))
        for action, cache in (("one_hop", cache1), ("direct_age2", cache0_direct), ("recursive_mixed", recursive)):
            action_score = score(cache)
            mse = float(torch.mean((action_score - current) ** 2))
            js = float(np.mean(quality.bernoulli_js(action_score.cpu().numpy(), current.cpu().numpy())))
            rows.append({
                "uid": uid, "action": action, "suffix_events": len(suffix),
                "mse": mse, "bernoulli_js": js,
                "max_abs_logit_shift": float((action_score - current).abs().max()),
            })
    if max_self > 1e-12 or not all(np.isfinite(row["mse"]) and np.isfinite(row["bernoulli_js"]) for row in rows):
        raise RuntimeError("P11 canary numeric gate failed")
    OUTPUT.mkdir(parents=True)
    raw_path = OUTPUT / "state_metrics.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw_path, compression="zstd")
    summaries = []
    for action in ACTIONS:
        values = [row for row in rows if row["action"] == action]
        summaries.append({
            "action": action,
            "mean_MSE": float(np.mean([row["mse"] for row in values])),
            "mean_JS": float(np.mean([row["bernoulli_js"] for row in values])),
            "P95_JS": float(np.quantile([row["bernoulli_js"] for row in values], 0.95)),
        })
    payload = {
        "status": "P11_0_version_debt_recursive_lineage_canary_passed",
        "users": len(uids), "suffix_events_total": int(sum(suffix_events)),
        "suffix_events_P50": float(np.median(suffix_events)), "suffix_events_max": int(max(suffix_events)),
        "summaries": summaries,
        "max_CurrentExact_self_logit_difference": max_self,
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": p7.sha256_file(raw_path),
        "contract_sha256": p7.sha256_file(CONTRACT),
        "theta0_sha256": p7.sha256_file(checkpoint0), "theta1_sha256": p7.sha256_file(checkpoint1),
        "theta2_sha256": p7.sha256_file(checkpoint2),
        "full_matrix_authorized": True, "blind_edge_executed": False,
    }
    (OUTPUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

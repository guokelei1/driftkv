"""Shared bounded rolling snapshots; Q and H retain separate read operators."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hstu_kvcache.read_correction import score_corrected
from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
from hstu_kvcache.read_correction_v4.token_read import score_token_corrected
from hstu_kvcache.training import collate_foundation_batch
from read_correction_v4.cost_nonlinear import correction_forward as history_cost
from read_correction_v4.evaluate_full import HISTOGRAMS
from read_correction_2026_09.cost import CostModel
from selective_recompute_2026_09.evaluate import all_snapshots, prefix_events


def load_policies(entries, device):
    import json
    from hstu_kvcache.read_correction_v5.query_only.core import QueryFeatureCorrection
    from read_correction_v5.query_only.cost import correction_forward as query_cost
    policies = []
    loaded = {}
    for entry in entries:
        folder = Path(entry["calibration_dir"])
        if str(folder) not in loaded:
            artifact = torch.load(folder / "calibration.pt", map_location="cpu", weights_only=False)
            cls = QueryFeatureCorrection if artifact["kind"] == "query_features_v5" else NonlinearTokenReadCorrection
            if artifact["kind"] not in ("query_features_v5", "token_read_nonlinear_v4"):
                raise ValueError("unsupported v5 candidate kind")
            modules = []
            for layer in artifact["modules"]:
                module = cls(**layer["config"]).to(device)
                module.load_state_dict(layer["state_dict"])
                modules.append(module.eval().requires_grad_(False))
            loaded[str(folder)] = modules
        policies.append({**entry, "modules": loaded[str(folder)],
            "calibration": json.loads((folder / "calibration.json").read_text()),
            "cost_fn": query_cost if entry["method"] == "query_only" else history_cost})
    return policies


@torch.inference_mode()
def score_unit(current, parent, history, by_user, uids, cutover, policies, cfg, scale, device, *, verify):
    records = {p["tag"]: [] for p in policies}
    stats = {name: Counter() for name in HISTOGRAMS}
    controls = {"requests": 0, "reuse_max_abs_logit_error": 0.,
        "full_max_abs_logit_error": 0., "identity_max_abs_logit_error": 0.}
    full = [u for u in uids if len(prefix_events(history.rows[u], cutover, cfg["history_length"])) == cfg["history_length"]]
    full_set, size = set(full), cfg["cohort_sizes"][scale]
    cohorts = [full[i:i+size] for i in range(0, len(full), size)] + [[u] for u in uids if u not in full_set]
    cost = CostModel.for_scale(scale, cfg["attention_backend"])
    for snap, _ in all_snapshots(cohorts, by_user, history, parent, current, cutover,
        cfg["query_batches"][scale], stats, cost, cfg["append_band_size"]):
        cache, requests = snap.state.cache, snap.requests
        n = cache.seq_len
        counts = torch.full((len(requests),), n, device=device)
        inherited = [max(0, n-int(r["append_count_since_cutover"])) for r in requests]
        old_counts = torch.tensor(inherited, device=device)
        if verify:
            native = current.observe_cc_reuse(cache, snap.candidates, snap.query_deltas)[0][:, 0]
            identity = score_corrected(current, cache, snap.candidates, snap.query_deltas,
                [None]*len(current.blocks), counts)[0][:, 0]
            batch = collate_foundation_batch([{**r, "weight": r.get("weight", 1.)} for r in requests],
                history, device=device, max_history=cfg["history_length"])
            actual_full = current.observe_cc_full(batch.item_ids, batch.behaviors, batch.time_deltas,
                batch.candidate_ids, batch.query_time_deltas, lengths=batch.lengths)[0][:, 0]
            for key, actual, expected in (
                ("reuse_max_abs_logit_error", native, native.new_tensor([r["reuse_logit"] for r in requests])),
                ("full_max_abs_logit_error", actual_full, native.new_tensor([r["full_logit"] for r in requests])),
                ("identity_max_abs_logit_error", identity, native)):
                controls[key] = max(controls[key], float((actual-expected).abs().max()))
            controls["requests"] += len(requests)
        for policy in policies:
            modules = policy["modules"]
            if policy["method"] == "query_only":
                logits = score_corrected(current, cache, snap.candidates, snap.query_deltas, modules, counts)[0][:, 0]
            else:
                logits = score_token_corrected(current, cache, snap.candidates, snap.query_deltas, modules, counts,
                    old_counts=old_counts if policy["variant"] == "map_old_prefix" else None)[0][:, 0]
            values = logits.float().cpu().tolist()
            if not np.isfinite(values).all():
                raise RuntimeError("nonfinite v5 predictions")
            extra = sum(policy["cost_fn"](m.get_config(), n) for m in modules)
            for row, value, old_count in zip(requests, values, inherited, strict=True):
                records[policy["tag"]].append({key: row[key] for key in (
                    "request_id", "uid", "label", "query_timestamp", "full_logit", "reuse_logit")}
                    | {"hstu_logit": value, "correction_flops": extra, "history_length": n,
                       "inherited_count": old_count, "append_count_since_cutover": row["append_count_since_cutover"]})
    if any(v > 2e-5 for k, v in controls.items() if k.endswith("error")):
        raise RuntimeError(f"v5 native controls differ: {controls}")
    return records, {name: {str(k): int(v) for k,v in values.items()} for name,values in stats.items()}, controls

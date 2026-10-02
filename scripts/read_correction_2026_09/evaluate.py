"""Transient corrections on the sealed rolling-Reuse timeline."""
from collections import Counter

import numpy as np
import torch

from hstu_kvcache.read_correction import score_corrected
from hstu_kvcache.training import collate_foundation_batch
from selective_recompute_2026_09.evaluate import all_snapshots, prefix_events
from read_correction_2026_09.cost import correction_forward


@torch.inference_mode()
def score_unit(uids, by_user, history, parent, current, cutover, *, corrections,
               cost_model, cohort_size, query_batch, verify=False, append_band_size=32):
    """All methods/budgets see the same cache; correction never writes it."""
    rows = {method: [] for method in corrections}
    stats = {name: Counter() for name in (
        "full_history_hist", "append_prefix_hist", "initial_history_hist",
        "torch_full_history_hist", "torch_append_prefix_hist",
        "band_append_hist", "torch_band_append_hist")}
    controls = {"reuse_max_abs_logit_error": 0.0, "full_max_abs_logit_error": 0.0,
                "zero_correction_max_abs_logit_error": 0.0, "requests": 0}
    full = [uid for uid in uids if len(prefix_events(history.rows[uid], cutover,
                                                   current.cfg.max_seq_len)) == current.cfg.max_seq_len]
    full_set = set(full)
    cohorts = [full[i:i + cohort_size] for i in range(0, len(full), cohort_size)]
    cohorts += [[uid] for uid in uids if uid not in full_set]
    for snap, active_cost in all_snapshots(cohorts, by_user, history, parent, current,
                                           cutover, query_batch, stats, cost_model, append_band_size):
        cache = snap.state.cache
        n = cache.seq_len
        counts = torch.full((len(snap.requests),), n, device=snap.candidates.device)
        if verify:
            native = current.observe_cc_reuse(cache, snap.candidates, snap.query_deltas)[0][:, 0]
            zero = score_corrected(current, cache, snap.candidates, snap.query_deltas,
                                   [None] * len(current.blocks), counts)[0][:, 0]
            full_batch = collate_foundation_batch(
                [{**r, "weight": r.get("weight", 1.0)} for r in snap.requests], history,
                device=snap.candidates.device, max_history=current.cfg.max_seq_len)
            actual_full = current.observe_cc_full(full_batch.item_ids, full_batch.behaviors,
                full_batch.time_deltas, full_batch.candidate_ids, full_batch.query_time_deltas,
                lengths=full_batch.lengths)[0][:, 0]
            expected_reuse = native.new_tensor([r["reuse_logit"] for r in snap.requests])
            expected_full = actual_full.new_tensor([r["full_logit"] for r in snap.requests])
            for key, difference in (
                ("reuse_max_abs_logit_error", native - expected_reuse),
                ("zero_correction_max_abs_logit_error", zero - native),
                ("full_max_abs_logit_error", actual_full - expected_full),
            ):
                controls[key] = max(controls[key], float(difference.abs().max()))
            controls["requests"] += len(snap.requests)
        pending = []
        for method, variants in corrections.items():
            for budget, modules in variants.items():
                logits = score_corrected(current, cache, snap.candidates, snap.query_deltas, modules, counts)[0][:, 0]
                extra = sum(correction_forward(module.get_config(), n) for module in modules)
                pending.append((method, int(budget), extra, logits))
        values = torch.stack([item[-1] for item in pending]).float().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite correction logits")
        for (method, budget, extra, _), logits in zip(pending, values):
            rows[method].extend({
                "request_id": request["request_id"], "uid": int(request["uid"]),
                "query_timestamp": int(request["query_timestamp"]), "budget": budget,
                "hstu_logit": float(logit), "history_length": n,
                "correction_flops": extra, "total_flops": extra,
            } for request, logit in zip(snap.requests, logits))
    if verify:
        for key, value in controls.items():
            if key.endswith("error") and value > 2e-5:
                raise RuntimeError(f"reference control failed: {key}={value}")
    return rows, {name: dict(value) for name, value in stats.items()}, controls

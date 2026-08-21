#!/usr/bin/env python3
"""Small deterministic HSTU/cache-lineage canary; no dataset training."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from hstu_kvcache.models import HSTU, HSTUConfig


def main() -> None:
    torch.manual_seed(37)
    cfg = HSTUConfig(
        num_items=128,
        num_behaviors=4,
        hidden_size=32,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
        input_dropout=0.0,
    )
    theta0 = HSTU(cfg).eval()
    theta_identity = copy.deepcopy(theta0).eval()
    theta1 = copy.deepcopy(theta0).eval()
    with torch.no_grad():
        theta1.blocks[0].attn.q_proj.weight.add_(0.01)

    prefix_items = torch.randint(1, cfg.num_items, (3, 8))
    prefix_behaviors = torch.randint(0, cfg.num_behaviors, (3, 8))
    prefix_deltas = torch.rand(3, 8) * 3600
    suffix_items = torch.randint(1, cfg.num_items, (3, 3))
    suffix_behaviors = torch.randint(0, cfg.num_behaviors, (3, 3))
    suffix_deltas = torch.rand(3, 3) * 3600
    full_items = torch.cat([prefix_items, suffix_items], dim=1)
    full_behaviors = torch.cat([prefix_behaviors, suffix_behaviors], dim=1)
    full_deltas = torch.cat([prefix_deltas, suffix_deltas], dim=1)

    old_prefix = theta0.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    identity_prefix = theta_identity.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    identity_suffix_hidden, identity_append = theta_identity.forward_with_cache(
        identity_prefix, suffix_items, suffix_behaviors, suffix_deltas,
    )
    identity_full_hidden, identity_full = theta_identity(
        full_items, full_behaviors, full_deltas, return_kv=True,
    )
    current_full = theta1.compute_kv(full_items, full_behaviors, full_deltas)
    reused_suffix_hidden, reused = theta1.forward_with_cache(
        old_prefix, suffix_items, suffix_behaviors, suffix_deltas
    )
    exact_suffix_hidden, exact = theta1.forward_with_cache(
        theta1.compute_kv(prefix_items, prefix_behaviors, prefix_deltas),
        suffix_items, suffix_behaviors, suffix_deltas,
    )
    no_history_hidden, _ = theta1(
        suffix_items, suffix_behaviors, suffix_deltas, return_kv=False
    )
    candidates = torch.arange(1, 9).repeat(3, 1)
    full_hidden, _ = theta1(full_items, full_behaviors, full_deltas)
    full_scores = theta1.score_candidates(full_hidden, candidates, torch.full((3,), 11))
    no_history_scores = theta1.score_candidates(
        no_history_hidden, candidates, torch.full((3,), 3)
    )
    identity_full_scores = theta_identity.score_candidates(
        identity_full_hidden, candidates, torch.full((3,), 11)
    )
    identity_reuse_scores = theta_identity.score_hidden(
        identity_suffix_hidden[:, -1, :], candidates
    )

    output_cfg = copy.deepcopy(cfg)
    output_cfg.tie_item_embeddings = False
    output_theta0 = HSTU(output_cfg).eval()
    output_theta1 = copy.deepcopy(output_theta0).eval()
    with torch.no_grad():
        output_theta1.output_emb.weight[1].add_(0.01)
    output_prefix = output_theta0.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    output_new_prefix = output_theta1.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
    output_old_hidden, _ = output_theta1.forward_with_cache(
        output_prefix, suffix_items, suffix_behaviors, suffix_deltas,
    )
    output_exact_hidden, _ = output_theta1.forward_with_cache(
        output_new_prefix, suffix_items, suffix_behaviors, suffix_deltas,
    )

    append_matrix = {}
    for append_len in (1, 2, 4, 16):
        append_items = torch.randint(1, cfg.num_items, (3, append_len))
        append_behaviors = torch.randint(0, cfg.num_behaviors, (3, append_len))
        append_deltas = torch.rand(3, append_len) * 3600
        exact_items = torch.cat([prefix_items, append_items], dim=1)
        exact_behaviors = torch.cat([prefix_behaviors, append_behaviors], dim=1)
        exact_deltas = torch.cat([prefix_deltas, append_deltas], dim=1)
        exact_cache = theta0.compute_kv(exact_items, exact_behaviors, exact_deltas)
        _, appended_cache = theta0.forward_with_cache(
            theta0.compute_kv(prefix_items, prefix_behaviors, prefix_deltas),
            append_items, append_behaviors, append_deltas,
        )
        append_matrix[str(append_len)] = exact_cache.difference_metrics(appended_cache)
    result = {
        "cache_shape": list(current_full.k.shape),
        "append_seq_len": reused.seq_len,
        "append_matches_exact_model": current_full.difference_metrics(exact),
        "old_to_current_prefix_difference": old_prefix.difference_metrics(
            theta1.compute_kv(prefix_items, prefix_behaviors, prefix_deltas)
        ),
        "reuse_vs_exact_suffix_hidden_rms": float((reused_suffix_hidden - exact_suffix_hidden).pow(2).mean().sqrt()),
        "full_vs_no_history_score_rms": float((full_scores - no_history_scores).pow(2).mean().sqrt().detach()),
        "reuse_vs_exact_cache_rms": reused.difference_metrics(exact),
        "identity_transition": {
            "prefix": identity_prefix.difference_metrics(old_prefix),
            "append": identity_append.difference_metrics(identity_full),
            "hidden_rms": float((identity_suffix_hidden - identity_full_hidden[:, -3:, :]).pow(2).mean().sqrt().detach()),
            "score_rms": float((identity_reuse_scores - identity_full_scores).pow(2).mean().sqrt().detach()),
        },
        "output_only_transition": {
            "cache": output_prefix.difference_metrics(output_new_prefix),
            "hidden_rms": float((output_old_hidden - output_exact_hidden).pow(2).mean().sqrt()),
        },
        "append_length_matrix": append_matrix,
    }
    assert reused.seq_len == 11
    assert result["append_matches_exact_model"]["k_rms"] < 1e-5
    assert result["full_vs_no_history_score_rms"] > 1e-6
    assert result["identity_transition"]["prefix"]["k_rms"] < 1e-8
    assert result["identity_transition"]["append"]["k_rms"] < 1e-5
    assert result["identity_transition"]["hidden_rms"] < 1e-5
    assert result["identity_transition"]["score_rms"] < 1e-5
    assert result["output_only_transition"]["cache"]["k_rms"] < 1e-8
    assert result["output_only_transition"]["hidden_rms"] < 1e-5
    output = Path("results/data_audit/yambda50m_v2/cache_identity_numeric_floor.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

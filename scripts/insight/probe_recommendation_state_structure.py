#!/usr/bin/env python3
"""Label-free recommendation-state observations over the fixed v0..v5 chain.

The experiment is prospective and diagnostic.  It does not train a model,
read request labels, fit target Current K/V, or promote a cache lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from hstu_kvcache.models import HSTUKVCache  # noqa: E402


CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_small_hstu_native_recommendation_state_structure_v1.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1"
)
DATASET = ROOT / "data/processed/yambda500m_unified_v1/scales/small/dataset.json"
USERS = ROOT / "data/processed/yambda500m_unified_v1/scales/small/users.parquet"
KNOWN_ITEMS = 781_678
DAY = 86_400
CUTOVER_DAYS = (231, 245, 259, 273, 287)
POPULATION = 3_000
HISTORY = 512
RECENT = 128
SHARED_ITEMS = 1_024
RECENT_CANDIDATES = 16
OLD_CANDIDATES = 16
NOVEL_CANDIDATES = 32
SPLIT_NAMESPACE = "evokv:recommendation-state-structure:v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(version: int) -> Path:
    if version == 0:
        return (
            ROOT
            / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"
        )
    return (
        ROOT
        / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
        / f"train_14d/checkpoints/v{version}/checkpoint_100.pt"
    )


def split_for_uid(uid: int) -> int:
    material = f"{SPLIT_NAMESPACE}:{int(uid)}".encode("ascii")
    return hashlib.sha256(material).digest()[0] & 1


def verify_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    for name in ("dataset", "users", "item_mapping"):
        path = ROOT / frozen[name]["path"]
        if sha256(path) != frozen[name]["sha256"]:
            raise RuntimeError(f"frozen {name} differs from the prospective contract")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        if ROOT / record["path"] != checkpoint(version):
            raise RuntimeError(f"v{version} checkpoint path differs from the contract")
        if sha256(checkpoint(version)) != record["sha256"]:
            raise RuntimeError(f"v{version} checkpoint differs from the contract")
    return contract


def select_population(count: int) -> pd.DataFrame:
    dataset = json.loads(DATASET.read_text())
    listens = (DATASET.parent / dataset["shared_listens_glob"]).resolve()
    connection = duckdb.connect()
    frame = connection.execute(
        """
        WITH eligible AS (
          SELECT u.uid, u.selector_rank, count(*) AS events_before_first_cutover
          FROM read_parquet(?) u JOIN read_parquet(?) l USING(uid)
          WHERE l.timestamp < ?
          GROUP BY u.uid, u.selector_rank
          HAVING count(*) >= ?
        )
        SELECT * FROM eligible ORDER BY selector_rank, uid LIMIT ?
        """,
        [str(USERS), str(listens), CUTOVER_DAYS[0] * DAY, HISTORY, count],
    ).fetchdf()
    connection.close()
    if len(frame) != count:
        raise RuntimeError(f"only {len(frame)} eligible users for requested population {count}")
    frame["split"] = ["fit" if split_for_uid(uid) == 0 else "held_out" for uid in frame.uid]
    return frame


def histories_at_cutover(history, uids: np.ndarray, cutover: int) -> tuple[np.ndarray, ...]:
    timestamps, items, behaviors = [], [], []
    for uid in uids:
        prefix_items, prefix_behaviors, prefix_timestamps = history.prefix(
            int(uid), cutover, HISTORY
        )
        if len(prefix_items) != HISTORY:
            raise RuntimeError(f"selected uid {uid} lacks a full cutover history")
        timestamps.append(prefix_timestamps)
        items.append(prefix_items)
        behaviors.append(prefix_behaviors)
    ts = np.stack(timestamps).astype(np.int64, copy=False)
    item = np.stack(items).astype(np.int64, copy=False)
    action = np.stack(behaviors).astype(np.int64, copy=False)
    delta = np.zeros_like(ts, dtype=np.float32)
    delta[:, 1:] = ts[:, 1:] - ts[:, :-1]
    query_delta = (cutover - ts[:, -1]).astype(np.float32)
    return ts, item, action, delta, query_delta


def select_shared_items(
    items: np.ndarray,
    splits: np.ndarray,
    count: int,
    minimum_split_users: int = 8,
    require_count: bool = True,
) -> np.ndarray:
    users = np.repeat(np.arange(len(items), dtype=np.int64), items.shape[1])
    values = items.reshape(-1)
    valid = (values > 0) & (values < KNOWN_ITEMS)
    composite = users[valid] * KNOWN_ITEMS + values[valid]
    unique = np.unique(composite)
    item_ids = unique % KNOWN_ITEMS
    user_ids = unique // KNOWN_ITEMS
    fit = np.bincount(item_ids[splits[user_ids] == 0], minlength=KNOWN_ITEMS)
    held = np.bincount(item_ids[splits[user_ids] == 1], minlength=KNOWN_ITEMS)
    total = fit + held
    candidates = np.flatnonzero(
        (fit >= minimum_split_users) & (held >= minimum_split_users)
    )
    order = np.lexsort((candidates, -total[candidates], -np.minimum(fit[candidates], held[candidates])))
    selected = candidates[order[:count]]
    if require_count and len(selected) != count:
        raise RuntimeError(f"only {len(selected)} items have cross-user support")
    if len(selected) < 1:
        raise RuntimeError("no item has the required cross-user support")
    return selected.astype(np.int64, copy=False)


def _capture_hook(storage: dict[str, torch.Tensor], name: str, transform=None):
    def hook(_module, _inputs, output):
        value = output if transform is None else transform(output)
        storage[name] = value

    return hook


@torch.inference_mode()
def trace_model(model, item_ids, behaviors, time_deltas):
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for layer, block in enumerate(model.blocks):
        prefix = f"layer{layer}"
        handles.extend(
            [
                block.norm.register_forward_hook(_capture_hook(captured, f"{prefix}.norm")),
                block.attn.q_proj.register_forward_hook(_capture_hook(captured, f"{prefix}.q")),
                block.attn.k_proj.register_forward_hook(_capture_hook(captured, f"{prefix}.k")),
                block.attn.v_proj.register_forward_hook(_capture_hook(captured, f"{prefix}.v")),
                block.attn.out_proj.register_forward_pre_hook(
                    lambda _module, inputs, key=f"{prefix}.av": captured.__setitem__(key, inputs[0])
                ),
                block.attn.register_forward_hook(
                    _capture_hook(captured, f"{prefix}.attention", lambda value: value[0])
                ),
                block.gate_proj.register_forward_hook(
                    _capture_hook(captured, f"{prefix}.gate", lambda value: F.silu(value))
                ),
                block.register_forward_hook(
                    _capture_hook(captured, f"{prefix}.hidden", lambda value: value[0])
                ),
            ]
        )
    try:
        captured["item_embedding"] = model.lookup_item_embeddings(item_ids)
        embedded = model.embed_inputs(item_ids, behaviors, time_deltas)
        captured["combined_input"] = embedded
        _, cache = model.forward_embedded(
            embedded, return_kv=True, return_hidden=False
        )
    finally:
        for handle in handles:
            handle.remove()
    assert cache is not None
    for layer in range(len(model.blocks)):
        captured[f"layer{layer}.update"] = (
            captured[f"layer{layer}.attention"] * captured[f"layer{layer}.gate"]
        )
    return captured, cache


class DeltaAccumulator:
    def __init__(self, items: int, actions: int, minimum: int = 8) -> None:
        self.items = items
        self.actions = actions
        self.groups = items * actions
        self.minimum = minimum
        self.stages: dict[str, dict[str, list[torch.Tensor]]] = {}

    def add(
        self,
        name: str,
        values: torch.Tensor,
        typed_group: torch.Tensor,
        user_splits: torch.Tensor,
    ) -> None:
        dimension = values.shape[-1]
        flat_groups = typed_group.reshape(-1)
        valid = flat_groups >= 0
        if not bool(valid.any()):
            return
        batch = values.shape[0]
        owner = torch.arange(batch, device=values.device)[:, None].expand_as(typed_group).reshape(-1)
        composite = owner[valid] * self.groups + flat_groups[valid]
        unique, inverse = torch.unique(composite, sorted=True, return_inverse=True)
        sums = torch.zeros((len(unique), dimension), dtype=torch.float32, device=values.device)
        sums.index_add_(0, inverse, values.reshape(-1, dimension)[valid].float())
        counts = torch.bincount(inverse, minlength=len(unique)).to(torch.float32)
        means = sums / counts[:, None]
        groups = unique % self.groups
        owners = unique // self.groups
        splits = user_splits[owners]
        if name not in self.stages:
            self.stages[name] = {
                "sum": [torch.zeros((self.groups, dimension)), torch.zeros((self.groups, dimension))],
                "norm": [torch.zeros(self.groups), torch.zeros(self.groups)],
                "count": [torch.zeros(self.groups, dtype=torch.int64), torch.zeros(self.groups, dtype=torch.int64)],
            }
        stats = self.stages[name]
        for split in (0, 1):
            mask = splits == split
            if not bool(mask.any()):
                continue
            group_cpu = groups[mask].cpu()
            value_cpu = means[mask].cpu()
            stats["sum"][split].index_add_(0, group_cpu, value_cpu)
            stats["norm"][split].index_add_(0, group_cpu, value_cpu.square().sum(dim=1))
            stats["count"][split].index_add_(
                0, group_cpu, torch.ones(len(group_cpu), dtype=torch.int64)
            )

    @staticmethod
    def _sse(norm: np.ndarray, sums: np.ndarray, counts: np.ndarray, means: np.ndarray) -> float:
        return float(np.sum(norm - 2.0 * np.sum(means * sums, axis=1) + counts * np.sum(means * means, axis=1)))

    def rows(self, edge: str) -> list[dict[str, Any]]:
        output = []
        group_items = np.repeat(np.arange(self.items), self.actions)
        for stage, stats in self.stages.items():
            train_sum = stats["sum"][0].numpy().astype(np.float64)
            test_sum = stats["sum"][1].numpy().astype(np.float64)
            train_norm = stats["norm"][0].numpy().astype(np.float64)
            test_norm = stats["norm"][1].numpy().astype(np.float64)
            train_count = stats["count"][0].numpy().astype(np.float64)
            test_count = stats["count"][1].numpy().astype(np.float64)
            valid = (train_count >= self.minimum) & (test_count >= self.minimum)
            if not np.any(valid):
                raise RuntimeError(f"no held-out typed groups for {edge} {stage}")
            dimension = train_sum.shape[1]
            item_sum = train_sum.reshape(self.items, self.actions, dimension).sum(axis=1)
            item_count = train_count.reshape(self.items, self.actions).sum(axis=1)
            item_mean = item_sum / np.maximum(item_count[:, None], 1.0)
            typed_mean = train_sum / np.maximum(train_count[:, None], 1.0)
            global_mean = train_sum.sum(axis=0) / train_count.sum()
            zero_sse = float(test_norm[valid].sum())
            global_means = np.broadcast_to(global_mean, (int(valid.sum()), dimension))
            item_means = item_mean[group_items[valid]]
            typed_means = typed_mean[valid]
            global_sse = self._sse(test_norm[valid], test_sum[valid], test_count[valid], global_means)
            item_sse = self._sse(test_norm[valid], test_sum[valid], test_count[valid], item_means)
            typed_sse = self._sse(test_norm[valid], test_sum[valid], test_count[valid], typed_means)
            output.append(
                {
                    "edge": edge,
                    "stage": stage,
                    "dimension": dimension,
                    "qualified_item_action_groups": int(valid.sum()),
                    "qualified_items": int(np.unique(group_items[valid]).size),
                    "held_out_user_item_action_samples": int(test_count[valid].sum()),
                    "held_out_delta_energy": zero_sse,
                    "global_version_shift_R2": 1.0 - global_sse / zero_sse,
                    "item_centroid_R2": 1.0 - item_sse / zero_sse,
                    "item_excess_R2_over_global": 1.0 - item_sse / global_sse,
                    "item_action_R2": 1.0 - typed_sse / zero_sse,
                    "item_action_increment_over_item": 1.0 - typed_sse / item_sse,
                }
            )
        return output


def _pair_by_tiers(items: np.ndarray, actions: np.ndarray, tiers: list[np.ndarray]) -> np.ndarray:
    unmatched = set(range(len(items)))
    pairs: list[tuple[int, int]] = []
    for keys in tiers:
        groups: dict[int, list[int]] = defaultdict(list)
        for position in sorted(unmatched):
            groups[int(keys[position])].append(position)
        for key in sorted(groups):
            values = groups[key]
            for offset in range(0, len(values) - 1, 2):
                left, right = values[offset], values[offset + 1]
                pairs.append((left, right))
                unmatched.remove(left)
                unmatched.remove(right)
    remaining = sorted(unmatched)
    if len(remaining) % 2:
        raise RuntimeError("pairing left an odd number of occurrences")
    pairs.extend(zip(remaining[::2], remaining[1::2], strict=True))
    pairs = sorted(pairs, key=lambda pair: (max(pair), min(pair)))
    if len(pairs) * 2 != len(items) or len({value for pair in pairs for value in pair}) != len(items):
        raise RuntimeError("pairing is not a partition")
    return np.asarray(pairs, dtype=np.int64)


def pairings(items: np.ndarray, actions: np.ndarray, action_slots: int) -> dict[str, np.ndarray]:
    positional = np.arange(len(items), dtype=np.int64).reshape(-1, 2)
    same_item = _pair_by_tiers(items, actions, [items])
    typed = _pair_by_tiers(
        items,
        actions,
        [items * action_slots + actions, items, actions],
    )
    return {"positional_pairs": positional, "same_item_pairs": same_item, "typed_pairs": typed}


def pairing_record(edge: str, uid: int, path: str, pairs: np.ndarray, items, actions) -> dict[str, Any]:
    left, right = pairs[:, 0], pairs[:, 1]
    return {
        "edge": edge,
        "uid": uid,
        "path": path,
        "same_item_pair_fraction": float(np.mean(items[left] == items[right])),
        "same_action_pair_fraction": float(np.mean(actions[left] == actions[right])),
        "same_item_action_pair_fraction": float(
            np.mean((items[left] == items[right]) & (actions[left] == actions[right]))
        ),
        "mean_pair_position_distance": float(np.mean(np.abs(right - left))),
    }


def _most_recent_unique(values: np.ndarray, count: int, excluded: set[int]) -> list[int]:
    selected = []
    for value in values[::-1]:
        item = int(value)
        if item <= 0 or item >= KNOWN_ITEMS or item in excluded:
            continue
        selected.append(item)
        excluded.add(item)
        if len(selected) == count:
            break
    return selected


def candidate_panel(
    items: np.ndarray, *, allow_canary_novel_fallback: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    counts = np.bincount(items.reshape(-1), minlength=KNOWN_ITEMS)[:KNOWN_ITEMS]
    observed = np.flatnonzero(counts > 0)
    observed = observed[observed > 0]
    order = np.lexsort((observed, -counts[observed]))
    bank = observed[order[: min(len(observed), 8_192)]].tolist()
    panels, modes = [], []
    audit = {
        "minimum_recent_unique": HISTORY,
        "minimum_old_only_unique": HISTORY,
        "minimum_novel_bank": HISTORY,
        "minimum_selected_recent": RECENT_CANDIDATES,
        "minimum_selected_old": OLD_CANDIDATES,
        "maximum_selected_novel": 0,
    }
    for row in items:
        recent_set = {int(value) for value in row[-RECENT:] if 0 < int(value) < KNOWN_ITEMS}
        full_set = {int(value) for value in row if 0 < int(value) < KNOWN_ITEMS}
        recent_available = len(recent_set)
        old_available = len(
            {int(value) for value in row[:-RECENT] if 0 < int(value) < KNOWN_ITEMS} - recent_set
        )
        novel_available = sum(int(value) not in full_set for value in bank)
        audit["minimum_recent_unique"] = min(audit["minimum_recent_unique"], recent_available)
        audit["minimum_old_only_unique"] = min(audit["minimum_old_only_unique"], old_available)
        audit["minimum_novel_bank"] = min(audit["minimum_novel_bank"], novel_available)
        used: set[int] = set()
        recent = _most_recent_unique(row[-RECENT:], RECENT_CANDIDATES, used)
        old = _most_recent_unique(row[:-RECENT], OLD_CANDIDATES, used | recent_set)
        novel_count = RECENT_CANDIDATES + OLD_CANDIDATES + NOVEL_CANDIDATES - len(recent) - len(old)
        novel = [int(value) for value in bank if int(value) not in full_set][:novel_count]
        if len(novel) < novel_count and allow_canary_novel_fallback:
            novel_set = set(novel)
            for value in range(1, KNOWN_ITEMS):
                if value not in full_set and value not in novel_set:
                    novel.append(value)
                    novel_set.add(value)
                    if len(novel) == novel_count:
                        break
        if len(novel) != novel_count:
            raise RuntimeError(
                "a selected user cannot fill the fixed 64-candidate bank: "
                f"recent={len(recent)}, old={len(old)}, novel={len(novel)}/{novel_count}"
            )
        audit["minimum_selected_recent"] = min(audit["minimum_selected_recent"], len(recent))
        audit["minimum_selected_old"] = min(audit["minimum_selected_old"], len(old))
        audit["maximum_selected_novel"] = max(audit["maximum_selected_novel"], len(novel))
        panels.append(recent + old + novel)
        modes.append(
            ["recent_repeat"] * len(recent)
            + ["old_only_repeat"] * len(old)
            + ["novel_to_prefix"] * len(novel)
        )
    return np.asarray(panels, dtype=np.int64), np.asarray(modes), audit


def compact_cache(prefix: HSTUKVCache, carriers: HSTUKVCache, endpoints: torch.Tensor) -> HSTUKVCache:
    layers, batch, _, width = carriers.k.shape
    absolute = endpoints + (HISTORY - RECENT)
    index = absolute[None, :, :, None].expand(layers, batch, endpoints.shape[1], width)
    selected_k = carriers.k.gather(2, index)
    selected_v = carriers.v.gather(2, index) * 2.0
    return HSTUKVCache(
        k=torch.cat([prefix.k[:, :, : HISTORY - RECENT], selected_k], dim=2),
        v=torch.cat([prefix.v[:, :, : HISTORY - RECENT], selected_v], dim=2),
        seq_len=HISTORY - RECENT + endpoints.shape[1],
    )


def dense_tail_cache(parent: HSTUKVCache, current: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=torch.cat([parent.k[:, :, : HISTORY - RECENT], current.k[:, :, -RECENT:]], dim=2),
        v=torch.cat([parent.v[:, :, : HISTORY - RECENT], current.v[:, :, -RECENT:]], dim=2),
        seq_len=HISTORY,
    )


@torch.inference_mode()
def score_cache(model, cache: HSTUKVCache, candidates, query_deltas) -> torch.Tensor:
    return model.score_cc_reuse(cache, candidates, query_deltas)


@torch.inference_mode()
def candidate_read_trace(model, cache: HSTUKVCache, candidates, query_deltas):
    if model.cfg.relative_position_bias:
        raise ValueError("the fixed influence trace requires no relative-position bias")
    batch, candidate_count = candidates.shape
    x = model.embed_query_tokens(candidates, query_deltas).reshape(
        batch * candidate_count, 1, model.cfg.hidden_size
    )
    influences = []
    for layer, block in enumerate(model.blocks):
        attention = block.attn
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = attention._project(x_norm)
        length = cache.k.shape[2]
        cached_k = cache.k[layer].repeat_interleave(candidate_count, dim=0)
        cached_v = cache.v[layer].repeat_interleave(candidate_count, dim=0)
        cached_k = cached_k.view(-1, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        cached_v = cached_v.view(-1, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        prefix_weights = attention._activate(
            torch.matmul(q, cached_k.transpose(-2, -1)) * attention.scale
        )
        value_energy = cached_v.float().square().sum(dim=-1)
        influence = torch.sqrt(
            (prefix_weights[:, :, 0, :].float().square() * value_energy).sum(dim=1).clamp_min(0.0)
        )
        influences.append(influence.reshape(batch, candidate_count, length))
        output = torch.matmul(prefix_weights, cached_v)
        if attention.causal_diagonal == "inclusive":
            self_weight = attention._activate((q * k_new).sum(dim=-1, keepdim=True) * attention.scale)
            output = output + self_weight * v_new
        attention_out = attention._finish(output)
        if block.block_variant == "hstu_reference":
            assert block.attn_output_norm is not None
            update = block.attn.out_proj(
                block.attn_output_norm(attention_out) * F.silu(block.gate_proj(x_norm))
            )
        else:
            update = attention_out * F.silu(block.gate_proj(x_norm))
        x = residual + update
    readout = model.final_norm(x).reshape(batch, candidate_count, model.cfg.hidden_size)
    scores = model.cc_score_head(readout).squeeze(-1)
    return scores, readout, influences


def spectral_metrics(matrix: torch.Tensor) -> dict[str, np.ndarray]:
    gram = torch.matmul(matrix.float(), matrix.float().transpose(-1, -2))
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0.0).flip(-1)
    total = eigen.sum(dim=-1)
    probability = eigen / total[:, None].clamp_min(1e-20)
    entropy = -(probability * probability.clamp_min(1e-20).log()).sum(dim=-1)
    cumulative = probability.cumsum(dim=-1)
    rank90 = (cumulative < 0.9).sum(dim=-1) + 1
    nonzero = total > 1e-16
    return {
        "top_direction_energy_fraction": torch.where(nonzero, probability[:, 0], 0.0).cpu().numpy(),
        "effective_rank": torch.where(nonzero, entropy.exp(), 0.0).cpu().numpy(),
        "rank90": torch.where(nonzero, rank90, 0).cpu().numpy(),
        "frobenius_energy": total.cpu().numpy(),
    }


def rank_correlation(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    ref_rank = torch.argsort(torch.argsort(reference, dim=1), dim=1).float()
    other_rank = torch.argsort(torch.argsort(other, dim=1), dim=1).float()
    ref_rank -= ref_rank.mean(dim=1, keepdim=True)
    other_rank -= other_rank.mean(dim=1, keepdim=True)
    return (ref_rank * other_rank).sum(dim=1) / torch.sqrt(
        ref_rank.square().sum(dim=1) * other_rank.square().sum(dim=1)
    ).clamp_min(1e-12)


def coreset_rows(edge, uids, exact, reuse, paths: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    exact_probability = torch.sigmoid(exact.float())
    reuse_gap = torch.abs(torch.sigmoid(reuse.float()) - exact_probability).mean(dim=1)
    exact_top1 = exact.argmax(dim=1)
    exact_top10 = exact.topk(10, dim=1).indices
    output = []
    for name, scores in paths.items():
        probability_gap = torch.abs(torch.sigmoid(scores.float()) - exact_probability).mean(dim=1)
        top10 = scores.topk(10, dim=1).indices
        overlap = (top10[:, :, None] == exact_top10[:, None, :]).any(dim=2).sum(dim=1).float() / 10.0
        recovery = 1.0 - probability_gap / reuse_gap.clamp_min(1e-12)
        correlation = rank_correlation(exact, scores)
        for index, uid in enumerate(uids):
            output.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    "path": name,
                    "mean_abs_logit_gap": float(torch.abs(scores[index] - exact[index]).mean()),
                    "mean_abs_probability_gap": float(probability_gap[index]),
                    "output_gap_recovery_over_reuse": float(recovery[index]),
                    "top1_agreement": int(scores[index].argmax() == exact_top1[index]),
                    "top10_overlap": float(overlap[index]),
                    "rank_correlation": float(correlation[index]),
                }
            )
    return output


def candidate_rows(edge, uids, modes, exact_scores, reuse_scores, exact_readout, reuse_readout,
                   exact_influences, reuse_influences) -> list[dict[str, Any]]:
    output = []
    readout_spectrum = spectral_metrics(exact_readout - reuse_readout)
    mode_names = ("recent_repeat", "old_only_repeat", "novel_to_prefix")
    score_gap = torch.abs(exact_scores - reuse_scores)
    mode_gap = {name: [] for name in mode_names}
    for user_index in range(len(uids)):
        for name in mode_names:
            mask = torch.as_tensor(modes[user_index] == name, device=score_gap.device)
            mode_gap[name].append(
                float(score_gap[user_index, mask].mean()) if bool(mask.any()) else float("nan")
            )
    for layer, (exact_raw, reuse_raw) in enumerate(zip(exact_influences, reuse_influences, strict=True)):
        exact = exact_raw / exact_raw.sum(dim=2, keepdim=True).clamp_min(1e-20)
        reuse = reuse_raw / reuse_raw.sum(dim=2, keepdim=True).clamp_min(1e-20)
        exact_spectrum = spectral_metrics(exact)
        delta_spectrum = spectral_metrics(exact - reuse)
        support = {}
        for name in mode_names:
            values = []
            for user_index in range(len(uids)):
                mask = torch.as_tensor(modes[user_index] == name, device=exact.device)
                probability = exact[user_index, mask]
                entropy = -(probability * probability.clamp_min(1e-20).log()).sum(dim=1)
                values.append(float(entropy.exp().mean()) if bool(mask.any()) else float("nan"))
            support[name] = values
        for index, uid in enumerate(uids):
            output.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    "layer": layer,
                    "candidate_influence_top_direction_fraction": float(exact_spectrum["top_direction_energy_fraction"][index]),
                    "candidate_influence_effective_rank": float(exact_spectrum["effective_rank"][index]),
                    "candidate_influence_rank90": int(exact_spectrum["rank90"][index]),
                    "exact_minus_reuse_influence_effective_rank": float(delta_spectrum["effective_rank"][index]),
                    "exact_minus_reuse_influence_rank90": int(delta_spectrum["rank90"][index]),
                    "exact_minus_reuse_influence_energy": float(delta_spectrum["frobenius_energy"][index]),
                    "exact_minus_reuse_readout_effective_rank": float(readout_spectrum["effective_rank"][index]),
                    "exact_minus_reuse_readout_rank90": int(readout_spectrum["rank90"][index]),
                    "recent_repeat_effective_support": float(support["recent_repeat"][index]),
                    "old_only_repeat_effective_support": float(support["old_only_repeat"][index]),
                    "novel_to_prefix_effective_support": float(support["novel_to_prefix"][index]),
                    "recent_repeat_abs_logit_shift": mode_gap["recent_repeat"][index],
                    "old_only_repeat_abs_logit_shift": mode_gap["old_only_repeat"][index],
                    "novel_to_prefix_abs_logit_shift": mode_gap["novel_to_prefix"][index],
                }
            )
    return output


def summarize(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, sort=True)[values].agg(["mean", "median"]).reset_index()


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(value) for value in column if str(value)) if isinstance(column, tuple) else column
        for column in frame.columns
    ]
    return frame


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(delta: pd.DataFrame, coreset: pd.DataFrame, candidate: pd.DataFrame,
                  population: pd.DataFrame, runtime: float) -> str:
    delta_focus = delta[delta.stage.isin(["item_embedding", "combined_input", "layer0.k", "layer1.k", "layer2.k", "layer3.k", "layer3.update"])]
    coreset_focus = coreset[
        ["edge", "path", "mean_abs_probability_gap_mean", "output_gap_recovery_over_reuse_mean", "top10_overlap_mean", "rank_correlation_mean"]
    ]
    candidate_focus = candidate[
        ["edge", "layer", "candidate_influence_top_direction_fraction_mean", "candidate_influence_effective_rank_mean", "candidate_influence_rank90_median", "exact_minus_reuse_influence_rank90_median", "novel_to_prefix_effective_support_mean", "recent_repeat_effective_support_mean"]
    ]
    lines = [
        "# Recommendation-state structure across v0..v5",
        "",
        f"Scope: {len(population):,} fixed users ({100 * len(population) / 10000:.1f}% of the frozen Small population), five adjacent edges, 512 pre-cutover events per user, and 64 label-free candidate probes per user. No request label was read.",
        "",
        "## Cross-user state-delta factorization",
        "",
        *markdown_table(delta_focus, ["edge", "stage", "qualified_items", "held_out_user_item_action_samples", "global_version_shift_R2", "item_centroid_R2", "item_excess_R2_over_global", "item_action_R2", "item_action_increment_over_item"]),
        "",
        "Centroids are fitted on one deterministic UID split and evaluated on disjoint users. Item and item-action columns therefore measure cross-user generalization, not within-sample reconstruction.",
        "",
        "## Matched-budget semantic coreset",
        "",
        *markdown_table(coreset_focus, list(coreset_focus.columns)),
        "",
        "Every compact path retains the same Parent old-384 prefix, 64 Current carriers for recent-128 evidence, and represented mass two. Only the label-free pairing rule changes.",
        "",
        "## Candidate-bank influence subspace",
        "",
        *markdown_table(candidate_focus, list(candidate_focus.columns)),
        "",
        "The fixed 64-candidate panel takes up to 16 known recent repeats and 16 known old-only repeats, then fills with known novel-to-prefix items. This keeps high-OOV/low-repeat users in scope. Probes are not sampled negatives and are never joined to labels. Influence uses the norm of each position's pointwise-attention value contribution before candidate-wise normalization.",
        "",
        f"Elapsed wall time: {runtime / 60.0:.1f} minutes.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-users", type=int, default=POPULATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_users < 1 or args.max_users > POPULATION:
        raise ValueError(f"max-users must be in 1..{POPULATION}")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    contract = verify_contract()
    started = time.time()
    population = select_population(args.max_users)
    if args.max_users == POPULATION and len(population) != contract["scope"]["users"]:
        raise RuntimeError("formal population count differs from the contract")
    uids = population.uid.to_numpy(dtype=np.int64)
    splits = np.asarray([split_for_uid(uid) for uid in uids], dtype=np.int64)
    device = torch.device(args.device)

    probe, payload = load_model(checkpoint(1), device)
    oov_buckets = int(payload["config"]["num_items"]) - KNOWN_ITEMS
    item_embedding_rows = int(payload["config"]["num_items"]) + 1
    action_slots = int(payload["config"]["num_behaviors"]) + 1
    del probe
    torch.cuda.empty_cache()
    print(json.dumps({"phase": "load_histories", "users": len(uids)}), flush=True)
    history = load_histories(uids.tolist(), oov_buckets=oov_buckets)

    delta_rows: list[dict[str, Any]] = []
    coreset_user_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    panel_audits: dict[str, Any] = {}
    trace_errors: dict[str, float] = {}

    for edge, cutover_day in enumerate(CUTOVER_DAYS):
        edge_name = f"v{edge}_to_v{edge + 1}"
        print(json.dumps({"phase": "edge_start", "edge": edge_name}), flush=True)
        cutover = cutover_day * DAY
        _, item_np, action_np, delta_np, query_delta_np = histories_at_cutover(history, uids, cutover)
        minimum_group_samples = 8 if args.max_users == POPULATION else 1
        shared = select_shared_items(
            item_np,
            splits,
            min(SHARED_ITEMS, max(1, args.max_users // 2)),
            minimum_split_users=minimum_group_samples,
            require_count=args.max_users == POPULATION,
        )
        item_to_group = np.full(item_embedding_rows, -1, dtype=np.int32)
        item_to_group[shared] = np.arange(len(shared), dtype=np.int32)
        panel_np, modes_np, panel_audit = candidate_panel(
            item_np, allow_canary_novel_fallback=args.max_users != POPULATION
        )
        panel_audits[edge_name] = panel_audit

        all_endpoints: dict[str, np.ndarray] = {
            "positional_pairs": np.empty((len(uids), RECENT // 2), dtype=np.int64),
            "same_item_pairs": np.empty((len(uids), RECENT // 2), dtype=np.int64),
            "typed_pairs": np.empty((len(uids), RECENT // 2), dtype=np.int64),
        }
        for user_index, uid in enumerate(uids):
            recent_items = item_np[user_index, -RECENT:]
            recent_actions = action_np[user_index, -RECENT:]
            for name, pairs in pairings(recent_items, recent_actions, action_slots).items():
                all_endpoints[name][user_index] = np.sort(pairs.max(axis=1))
                pairing_rows.append(
                    pairing_record(edge_name, int(uid), name, pairs, recent_items, recent_actions)
                )

        parent, _ = load_model(checkpoint(edge), device)
        current, _ = load_model(checkpoint(edge + 1), device)
        accumulator = DeltaAccumulator(
            len(shared), action_slots, minimum=minimum_group_samples
        )
        item_map_gpu = torch.as_tensor(item_to_group, device=device)

        for start in range(0, len(uids), args.batch_size):
            stop = min(start + args.batch_size, len(uids))
            batch_uids = uids[start:stop]
            items = torch.as_tensor(item_np[start:stop], dtype=torch.long, device=device)
            actions = torch.as_tensor(action_np[start:stop], dtype=torch.long, device=device)
            deltas = torch.as_tensor(delta_np[start:stop], dtype=torch.float32, device=device)
            query_deltas = torch.as_tensor(query_delta_np[start:stop], dtype=torch.float32, device=device)
            candidates = torch.as_tensor(panel_np[start:stop], dtype=torch.long, device=device)
            group = item_map_gpu[items]
            typed_group = torch.where(group >= 0, group * action_slots + actions, -1)
            batch_splits = torch.as_tensor(splits[start:stop], dtype=torch.long, device=device)

            parent_trace, parent_cache = trace_model(parent, items, actions, deltas)
            current_trace, current_cache = trace_model(current, items, actions, deltas)
            if parent_trace.keys() != current_trace.keys():
                raise RuntimeError("Parent and Current trace stages differ")
            for stage in parent_trace:
                accumulator.add(
                    stage,
                    current_trace[stage].float() - parent_trace[stage].float(),
                    typed_group,
                    batch_splits,
                )

            exact_scores, exact_readout, exact_influence = candidate_read_trace(
                current, current_cache, candidates, query_deltas
            )
            reuse_scores, reuse_readout, reuse_influence = candidate_read_trace(
                current, parent_cache, candidates, query_deltas
            )
            if edge_name not in trace_errors:
                builtin_exact, _ = current.observe_cc_reuse(current_cache, candidates, query_deltas)
                builtin_reuse, _ = current.observe_cc_reuse(parent_cache, candidates, query_deltas)
                error = max(
                    float(torch.max(torch.abs(builtin_exact - exact_scores)).detach()),
                    float(torch.max(torch.abs(builtin_reuse - reuse_scores)).detach()),
                )
                trace_errors[edge_name] = error
                if error > 2e-5:
                    raise RuntimeError(f"candidate influence trace changes model scores by {error}")

            dense_scores = score_cache(
                current,
                dense_tail_cache(parent_cache, current_cache), candidates, query_deltas
            )
            path_scores = {"parent_reuse": reuse_scores, "dense_current_tail128": dense_scores}
            for name in ("positional_pairs", "same_item_pairs", "typed_pairs"):
                endpoints = torch.as_tensor(all_endpoints[name][start:stop], dtype=torch.long, device=device)
                cache = compact_cache(parent_cache, current_cache, endpoints)
                path_scores[name] = score_cache(current, cache, candidates, query_deltas)
            coreset_user_rows.extend(
                coreset_rows(edge_name, batch_uids, exact_scores, reuse_scores, path_scores)
            )
            subspace_rows.extend(
                candidate_rows(
                    edge_name,
                    batch_uids,
                    modes_np[start:stop],
                    exact_scores,
                    reuse_scores,
                    exact_readout,
                    reuse_readout,
                    exact_influence,
                    reuse_influence,
                )
            )
            del parent_trace, current_trace, parent_cache, current_cache
            del exact_scores, exact_readout, exact_influence, reuse_scores, reuse_readout, reuse_influence

        delta_rows.extend(accumulator.rows(edge_name))
        del parent, current, accumulator
        torch.cuda.empty_cache()
        print(json.dumps({"phase": "edge_complete", "edge": edge_name}), flush=True)

    delta_frame = pd.DataFrame(delta_rows)
    coreset_frame = pd.DataFrame(coreset_user_rows)
    pairing_frame = pd.DataFrame(pairing_rows)
    subspace_frame = pd.DataFrame(subspace_rows)
    coreset_summary = flatten_columns(
        summarize(
            coreset_frame,
            ["edge", "path"],
            ["mean_abs_probability_gap", "output_gap_recovery_over_reuse", "top1_agreement", "top10_overlap", "rank_correlation"],
        )
    )
    pairing_summary = flatten_columns(
        summarize(
            pairing_frame,
            ["edge", "path"],
            ["same_item_pair_fraction", "same_action_pair_fraction", "same_item_action_pair_fraction", "mean_pair_position_distance"],
        )
    )
    subspace_summary = flatten_columns(
        summarize(
            subspace_frame,
            ["edge", "layer"],
            [
                "candidate_influence_top_direction_fraction",
                "candidate_influence_effective_rank",
                "candidate_influence_rank90",
                "exact_minus_reuse_influence_effective_rank",
                "exact_minus_reuse_influence_rank90",
                "exact_minus_reuse_readout_effective_rank",
                "exact_minus_reuse_readout_rank90",
                "recent_repeat_effective_support",
                "old_only_repeat_effective_support",
                "novel_to_prefix_effective_support",
                "recent_repeat_abs_logit_shift",
                "old_only_repeat_abs_logit_shift",
                "novel_to_prefix_abs_logit_shift",
            ],
        )
    )
    runtime = time.time() - started
    summary = {
        "status": "recommendation_state_structure_observation_complete",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "users": len(population),
        "population_fraction": len(population) / 10_000,
        "fit_users": int((population.split == "fit").sum()),
        "held_out_users": int((population.split == "held_out").sum()),
        "edges": [f"v{edge}_to_v{edge + 1}" for edge in range(5)],
        "history_positions_per_user_edge": HISTORY,
        "candidate_probes_per_user_edge": RECENT_CANDIDATES + OLD_CANDIDATES + NOVEL_CANDIDATES,
        "labels_read": False,
        "candidate_probe_negative_semantics": False,
        "candidate_trace_max_abs_score_error": trace_errors,
        "candidate_panel_audit": panel_audits,
        "elapsed_seconds": runtime,
    }

    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    population.to_parquet(partial / "population.parquet", index=False)
    delta_frame.to_csv(partial / "state_delta_factorization.csv", index=False)
    coreset_frame.to_parquet(partial / "semantic_coreset_user_metrics.parquet", index=False)
    coreset_summary.to_csv(partial / "semantic_coreset_summary.csv", index=False)
    pairing_frame.to_parquet(partial / "pairing_user_metrics.parquet", index=False)
    pairing_summary.to_csv(partial / "pairing_summary.csv", index=False)
    subspace_frame.to_parquet(partial / "candidate_subspace_user_metrics.parquet", index=False)
    subspace_summary.to_csv(partial / "candidate_subspace_summary.csv", index=False)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (partial / "report.md").write_text(
        render_report(delta_frame, coreset_summary, subspace_summary, population, runtime)
    )
    os.replace(partial, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

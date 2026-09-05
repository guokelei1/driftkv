from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from hstu_kvcache.models import HSTUKVCache


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_medium_hstu_native_insight1_locality_v1.yaml"
DATASET = ROOT / "data/processed/yambda500m_unified_v1/scales/medium/dataset.json"
USERS = ROOT / "data/processed/yambda500m_unified_v1/scales/medium/users.parquet"
INPUT_MANIFEST = ROOT / "data/manifests/yambda500m_medium_insight1_locality_v1"
RESULT_ROOT = ROOT / "results/yambda500m_medium_seed17/insight1_locality_v1"

DAY = 86_400
CUTOVER_DAYS = (231, 245, 259, 273, 287)
EDGES = tuple(f"v{index}_to_v{index + 1}" for index in range(5))
HISTORY = 1024
LAYERS = 6
CANDIDATES = 64
POPULATION = 3000
KNOWN_ITEMS = 1_380_509
OOV_BUCKETS = 256
RECENT = 256

SELECTOR_NAMES = (
    "ATTN_MASS",
    "READ_NORM",
    "PERSISTENCE",
    "KV_DRIFT",
    "READ_DELTA",
)
TOKEN_BUDGETS = {
    "p10": (102, ("ATTN_MASS", "READ_NORM", "PERSISTENCE", "KV_DRIFT", "READ_DELTA")),
    "p20": (205, ("ATTN_MASS", "READ_NORM", "KV_DRIFT", "READ_DELTA")),
    "p40": (410, ("READ_NORM", "READ_DELTA")),
    "p80": (819, ("READ_DELTA",)),
}


@dataclass(frozen=True)
class LocalityConfig:
    config_id: str
    family: str
    cost: float
    budget: str
    layers: tuple[int, ...] = ()
    interval: tuple[int, int] | None = None
    selector: str | None = None
    positions: int | None = None


def _layer_configs() -> list[LocalityConfig]:
    groups = (
        (0,), (1,), (2,), (3,), (4,), (5,),
        (0, 1), (2, 3), (4, 5),
        (0, 1, 2), (2, 3, 4), (3, 4, 5),
        (0, 1, 2, 3), (2, 3, 4, 5),
    )
    output = []
    for group in groups:
        one_based = "_".join(str(value + 1) for value in group)
        output.append(
            LocalityConfig(
                config_id=f"layer_{one_based}",
                family="layer",
                cost=len(group) / LAYERS,
                budget=f"k{len(group)}",
                layers=group,
            )
        )
    return output


def _window_configs() -> list[LocalityConfig]:
    intervals = (
        (896, 1024), (768, 896), (640, 768),
        (768, 1024), (512, 768), (256, 512),
        (512, 1024), (256, 1024),
    )
    return [
        LocalityConfig(
            config_id=f"window_{stop - start}_{start}_{stop}",
            family="window",
            cost=(stop - start) / HISTORY,
            budget=f"w{stop - start}",
            interval=(start, stop),
        )
        for start, stop in intervals
    ]


def _token_configs() -> list[LocalityConfig]:
    output = []
    for budget, (positions, selectors) in TOKEN_BUDGETS.items():
        for selector in selectors:
            output.append(
                LocalityConfig(
                    config_id=f"token_{budget}_{selector.lower()}",
                    family="token",
                    cost=positions / HISTORY,
                    budget=budget,
                    selector=selector,
                    positions=positions,
                )
            )
    return output


LOCALITY_CONFIGS = tuple(_layer_configs() + _window_configs() + _token_configs())
PATH_IDS = ("reuse", "current_exact") + tuple(config.config_id for config in LOCALITY_CONFIGS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(version: int) -> Path:
    if version == 0:
        return RESULT_ROOT.parent / "full_reuse_matrix_v1/shared_v0/checkpoint_100.pt"
    if version == 5:
        return RESULT_ROOT.parent / "full_reuse_matrix_v1/D14/v5_extension_v1/checkpoint/checkpoint_100.pt"
    return RESULT_ROOT.parent / f"full_reuse_matrix_v1/D14/checkpoints/v{version}/checkpoint_100.pt"


def verify_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    frozen = contract["frozen_inputs"]
    for name in ("design", "dataset", "users", "item_mapping"):
        record = frozen[name]
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen input differs from contract: {name}")
    for section in ("checkpoints", "checkpoint_seals"):
        for version in range(6):
            record = frozen[section][f"v{version}"]
            path = ROOT / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"frozen {section} v{version} differs from contract")
            if section == "checkpoints" and path != checkpoint(version):
                raise RuntimeError(f"checkpoint path differs for v{version}")
    if len(LOCALITY_CONFIGS) != 34 or len(PATH_IDS) != 36:
        raise RuntimeError("frozen configuration count differs from 34 locality paths")
    return contract


def histories_at_cutover(history, uids: np.ndarray, cutover: int) -> tuple[np.ndarray, ...]:
    timestamps, items, behaviors = [], [], []
    for uid in uids:
        prefix_items, prefix_behaviors, prefix_timestamps = history.prefix(
            int(uid), cutover, HISTORY
        )
        if len(prefix_items) != HISTORY:
            raise RuntimeError(f"selected uid {uid} lacks a full 1024-event prefix")
        timestamps.append(prefix_timestamps)
        items.append(prefix_items)
        behaviors.append(prefix_behaviors)
    timestamp_array = np.stack(timestamps).astype(np.int64, copy=False)
    item_array = np.stack(items).astype(np.int64, copy=False)
    behavior_array = np.stack(behaviors).astype(np.int64, copy=False)
    deltas = np.zeros_like(timestamp_array, dtype=np.float32)
    deltas[:, 1:] = timestamp_array[:, 1:] - timestamp_array[:, :-1]
    query_deltas = (cutover - timestamp_array[:, -1]).astype(np.float32)
    return timestamp_array, item_array, behavior_array, deltas, query_deltas


def _most_recent_unique(values: np.ndarray, count: int, excluded: set[int]) -> list[int]:
    selected: list[int] = []
    for raw in values[::-1]:
        item = int(raw)
        if item <= 0 or item >= KNOWN_ITEMS or item in excluded:
            continue
        selected.append(item)
        excluded.add(item)
        if len(selected) == count:
            break
    return selected


def candidate_panel(items: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    counts = np.bincount(items.reshape(-1), minlength=KNOWN_ITEMS)[:KNOWN_ITEMS]
    observed = np.flatnonzero(counts > 0)
    observed = observed[observed > 0]
    bank_order = np.lexsort((observed, -counts[observed]))
    bank = observed[bank_order[: min(len(observed), 16_384)]].tolist()
    panels: list[list[int]] = []
    modes: list[list[int]] = []
    audit = {
        "minimum_recent_unique": HISTORY,
        "minimum_old_only_unique": HISTORY,
        "minimum_novel_bank": HISTORY,
        "minimum_selected_recent": 16,
        "minimum_selected_old": 16,
        "maximum_selected_novel": 0,
    }
    for row in items:
        recent_set = {int(value) for value in row[-RECENT:] if 0 < int(value) < KNOWN_ITEMS}
        full_set = {int(value) for value in row if 0 < int(value) < KNOWN_ITEMS}
        old_set = {int(value) for value in row[:-RECENT] if 0 < int(value) < KNOWN_ITEMS}
        audit["minimum_recent_unique"] = min(audit["minimum_recent_unique"], len(recent_set))
        audit["minimum_old_only_unique"] = min(
            audit["minimum_old_only_unique"], len(old_set - recent_set)
        )
        audit["minimum_novel_bank"] = min(
            audit["minimum_novel_bank"], sum(int(value) not in full_set for value in bank)
        )
        used: set[int] = set()
        recent = _most_recent_unique(row[-RECENT:], 16, used)
        old = _most_recent_unique(row[:-RECENT], 16, used | recent_set)
        novel_count = CANDIDATES - len(recent) - len(old)
        novel = [int(value) for value in bank if int(value) not in full_set][:novel_count]
        if len(novel) != novel_count:
            raise RuntimeError(
                f"candidate panel cannot be filled: recent={len(recent)} old={len(old)} "
                f"novel={len(novel)}/{novel_count}"
            )
        audit["minimum_selected_recent"] = min(audit["minimum_selected_recent"], len(recent))
        audit["minimum_selected_old"] = min(audit["minimum_selected_old"], len(old))
        audit["maximum_selected_novel"] = max(audit["maximum_selected_novel"], len(novel))
        panels.append(recent + old + novel)
        modes.append([0] * len(recent) + [1] * len(old) + [2] * len(novel))
    panel_array = np.asarray(panels, dtype=np.int64)
    mode_array = np.asarray(modes, dtype=np.uint8)
    if panel_array.shape != (len(items), CANDIDATES) or mode_array.shape != panel_array.shape:
        raise RuntimeError("candidate panel shape differs from [users,64]")
    return panel_array, mode_array, audit


def load_input_manifest(
    *, verify_frozen: bool = True
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    contract = (
        verify_contract()
        if verify_frozen
        else yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    )
    descriptor_path = INPUT_MANIFEST / "manifest.json"
    if not descriptor_path.is_file():
        raise RuntimeError("Insight 1 input manifest has not been prepared")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor["contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("input manifest was built from another contract")
    for name, record in descriptor["artifacts"].items():
        path = INPUT_MANIFEST / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"input manifest artifact differs: {name}")
    population = np.load(INPUT_MANIFEST / "population.npz", allow_pickle=False)
    panels = np.load(INPUT_MANIFEST / "candidate_panels.npz", allow_pickle=False)
    uids = population["uids"].astype(np.int64, copy=False)
    candidates = panels["candidates"].astype(np.int64, copy=False)
    modes = panels["modes"].astype(np.uint8, copy=False)
    if uids.shape != (POPULATION,):
        raise RuntimeError("input population does not contain 3000 users")
    if candidates.shape != (len(EDGES), POPULATION, CANDIDATES):
        raise RuntimeError("candidate panels differ from [5,3000,64]")
    if not np.array_equal(panels["uids"], uids):
        raise RuntimeError("candidate panels and population UID order differ")
    return contract, uids, candidates, modes


def _l1_history(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def _prefix_step(model, block, x: torch.Tensor, cache: HSTUKVCache, layer: int, candidates: int):
    batch_candidates = x.shape[0]
    if batch_candidates % candidates:
        raise ValueError("query batch is not divisible by candidate count")
    batch = batch_candidates // candidates
    attention = block.attn
    residual = x
    x_norm = block.norm(x)
    q, k_new, v_new = attention._project(x_norm)
    length = cache.k.shape[2]
    q_group = q[:, :, 0, :].reshape(batch, candidates, attention.num_heads, attention.head_dim)
    cached_k = cache.k[layer].reshape(batch, length, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    cached_v = cache.v[layer].reshape(batch, length, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    raw = torch.einsum("bchd,bhld->bchl", q_group, cached_k) * attention.scale
    bias = attention._relative_position_bias(
        torch.tensor([length], device=x.device),
        torch.arange(length, device=x.device),
        raw.dtype,
    )
    if bias is not None:
        raw = raw + bias.permute(0, 2, 1, 3)
    weights = attention._activate(raw)
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights)
    prefix_heads = torch.einsum("bchl,bhld->bchd", weights, cached_v).reshape(
        batch_candidates, attention.num_heads, 1, attention.head_dim
    )
    if attention.causal_diagonal == "inclusive":
        self_weight = (q * k_new).sum(dim=-1, keepdim=True) * attention.scale
        self_bias = attention._relative_position_bias(
            torch.tensor([length], device=x.device),
            torch.tensor([length], device=x.device),
            self_weight.dtype,
        )
        if self_bias is not None:
            self_weight = self_weight + self_bias
        self_weight = attention._activate(self_weight)
        if attention.block_variant == "hstu_reference":
            self_weight = self_weight / attention.cfg.max_seq_len
        prefix_heads = prefix_heads + attention.attn_dropout(self_weight) * v_new
    attention_out = attention._finish(prefix_heads)
    if block.block_variant == "hstu_reference":
        if block.attn_output_norm is None:
            raise RuntimeError("HSTU reference block lacks attention output norm")
        update = attention.out_proj(
            block.attn_output_norm(attention_out) * F.silu(block.gate_proj(x_norm))
        )
    elif block.gating == "silu_gate":
        update = attention_out * F.silu(block.gate_proj(x_norm))
    elif block.gating == "glu":
        update = attention_out * torch.sigmoid(block.gate_proj(x_norm))
    elif block.gating == "ffn":
        update = block.fc2(F.silu(block.fc1(x_norm)) * block.fc3(x_norm))
    else:
        update = attention_out
    return residual + update, weights, cached_v


@torch.inference_mode()
def token_importance_scores(
    model,
    parent_cache: HSTUKVCache,
    current_cache: HSTUKVCache,
    candidates: torch.Tensor,
    query_deltas: torch.Tensor,
    candidate_chunk: int,
) -> dict[str, torch.Tensor]:
    if parent_cache.k.shape != current_cache.k.shape:
        raise ValueError("Parent and Current cache shapes differ")
    if parent_cache.k.shape[0] != LAYERS or parent_cache.k.shape[2] != HISTORY:
        raise ValueError("selector requires a 6-layer 1024-position cache")
    batch = candidates.shape[0]
    accumulators = {
        name: torch.zeros(batch, HISTORY, dtype=torch.float32, device=candidates.device)
        for name in ("ATTN_MASS", "READ_NORM", "PERSISTENCE", "READ_DELTA")
    }
    for start in range(0, candidates.shape[1], candidate_chunk):
        stop = min(start + candidate_chunk, candidates.shape[1])
        candidate_slice = candidates[:, start:stop]
        count = stop - start
        x_parent = model.embed_query_tokens(candidate_slice, query_deltas).reshape(
            batch * count, 1, model.cfg.hidden_size
        )
        x_current = x_parent.clone()
        for layer, block in enumerate(model.blocks):
            x_parent, parent_weights, parent_values = _prefix_step(
                model, block, x_parent, parent_cache, layer, count
            )
            x_current, current_weights, current_values = _prefix_step(
                model, block, x_current, current_cache, layer, count
            )
            accumulators["ATTN_MASS"] += _l1_history(parent_weights).sum(dim=(1, 2))
            parent_value_norm = parent_values.float().square().sum(dim=-1).sqrt()
            parent_read = parent_weights.float() * parent_value_norm[:, None, :, :]
            accumulators["READ_NORM"] += _l1_history(parent_read).sum(dim=(1, 2))
            top = parent_weights.topk(TOKEN_BUDGETS["p10"][0], dim=-1).indices
            persistence = torch.zeros_like(parent_weights, dtype=torch.float32)
            persistence.scatter_(-1, top, 1.0)
            accumulators["PERSISTENCE"] += persistence.sum(dim=(1, 2))
            p = parent_weights.float()
            c = current_weights.float()
            p_norm2 = parent_values.float().square().sum(dim=-1)
            c_norm2 = current_values.float().square().sum(dim=-1)
            cross = (parent_values.float() * current_values.float()).sum(dim=-1)
            delta2 = (
                p.square() * p_norm2[:, None, :, :]
                + c.square() * c_norm2[:, None, :, :]
                - 2.0 * p * c * cross[:, None, :, :]
            ).clamp_min(0.0)
            accumulators["READ_DELTA"] += _l1_history(delta2.sqrt()).sum(dim=(1, 2))
    normalizer = float(candidates.shape[1] * model.cfg.num_heads * model.cfg.num_layers)
    for name in accumulators:
        accumulators[name] /= normalizer
    drift = (
        (current_cache.k.float() - parent_cache.k.float()).square().sum(dim=-1).sqrt()
        + (current_cache.v.float() - parent_cache.v.float()).square().sum(dim=-1).sqrt()
    )
    accumulators["KV_DRIFT"] = _l1_history(drift).mean(dim=0)
    if set(accumulators) != set(SELECTOR_NAMES):
        raise RuntimeError("selector score set differs from the frozen contract")
    return accumulators


def stable_topk_mask(scores: torch.Tensor, positions: int) -> torch.Tensor:
    if scores.ndim != 2 or not 1 <= positions <= scores.shape[1]:
        raise ValueError("top-k mask received invalid shape or cardinality")
    reversed_scores = torch.flip(scores, dims=(-1,))
    order_reversed = torch.argsort(reversed_scores, dim=-1, descending=True, stable=True)
    order = scores.shape[1] - 1 - order_reversed[:, :positions]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, order, True)
    if not bool((mask.sum(dim=1) == positions).all()):
        raise RuntimeError("token mask cardinality differs from budget")
    return mask


def token_masks(selector_scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for config in LOCALITY_CONFIGS:
        if config.family != "token":
            continue
        assert config.selector is not None and config.positions is not None
        output[config.config_id] = stable_topk_mask(
            selector_scores[config.selector], config.positions
        )
    if len(output) != 12:
        raise RuntimeError("token mask configuration count differs from 12")
    return output


def config_mask(
    config: LocalityConfig,
    batch: int,
    device: torch.device,
    selected_tokens: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = torch.zeros(LAYERS, batch, HISTORY, dtype=torch.bool, device=device)
    if config.family == "layer":
        mask[list(config.layers), :, :] = True
    elif config.family == "window":
        if config.interval is None:
            raise RuntimeError("window config lacks an interval")
        start, stop = config.interval
        mask[:, :, start:stop] = True
    elif config.family == "token":
        token_mask = selected_tokens[config.config_id]
        if token_mask.shape != (batch, HISTORY):
            raise RuntimeError("token mask shape differs from [batch,1024]")
        mask = token_mask.unsqueeze(0).expand(LAYERS, -1, -1)
    else:
        raise ValueError(f"unknown locality family: {config.family}")
    expected = round(config.cost * LAYERS * HISTORY)
    if not bool((mask.sum(dim=(0, 2)) == expected).all()):
        raise RuntimeError(f"mask cardinality differs for {config.config_id}")
    return mask


def hybrid_cache(
    parent: HSTUKVCache,
    current: HSTUKVCache,
    mask: torch.Tensor,
) -> HSTUKVCache:
    if parent.k.shape != current.k.shape or parent.v.shape != current.v.shape:
        raise ValueError("Parent and Current cache shapes differ")
    if mask.shape != parent.k.shape[:3]:
        raise ValueError("hybrid mask must have shape [layers,batch,history]")
    expanded = mask.unsqueeze(-1)
    return HSTUKVCache(
        k=torch.where(expanded, current.k, parent.k),
        v=torch.where(expanded, current.v, parent.v),
        seq_len=parent.seq_len,
    )


@torch.inference_mode()
def score_cache_chunked(model, cache, candidates, query_deltas, chunk: int) -> torch.Tensor:
    output = []
    for start in range(0, candidates.shape[1], chunk):
        stop = min(start + chunk, candidates.shape[1])
        output.append(model.score_cc_reuse(cache, candidates[:, start:stop], query_deltas))
    return torch.cat(output, dim=1)


def config_records() -> list[dict[str, Any]]:
    records = []
    for index, config in enumerate(LOCALITY_CONFIGS, start=2):
        records.append(
            {
                "path_index": index,
                "config_id": config.config_id,
                "family": config.family,
                "budget": config.budget,
                "cost": config.cost,
                "layers": list(config.layers),
                "interval": list(config.interval) if config.interval else None,
                "selector": config.selector,
                "positions": config.positions,
            }
        )
    return records

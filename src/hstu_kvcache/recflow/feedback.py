"""Request-grouped binary feedback on the existing Yambda-shaped backbone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def grouped_scores(model, items, behaviors, deltas, lengths, candidates, query_deltas):
    """One differentiable prefix, independent queries; no candidate cross-attention.

    Uses legacy/inclusive/no-position-bias only. A single item-table lookup
    avoids two separately materialized full-table gradients during backward.
    Padding is masked at input and after every residual layer.
    """
    batch, width = items.shape
    count = candidates.shape[1]
    joined = torch.cat((items, candidates), dim=1)
    if model.training and torch.is_grad_enabled() and getattr(model, "direct_embedding_gradient", False):
        from .feedback_embedding import lookup
        vectors = lookup(model.item_emb.weight, joined)
    else:
        vectors = model.lookup_item_embeddings(joined)
    valid = torch.arange(width, device=items.device)[None] < lengths[:, None]
    prefix = model.combine_input_features(vectors[:, :width], behaviors, deltas)
    prefix = prefix * valid[..., None]
    query = model.embed_query_tokens(candidates, query_deltas, item_vectors=vectors[:, width:])
    for block in model.blocks:
        prefix, (keys, values) = block(prefix, return_kv=True)
        prefix = prefix * valid[..., None]
        normalized = block.norm(query)
        attn = block.attn
        q, self_k, self_v = attn._project(normalized)
        k = keys.view(batch, width, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = values.view(batch, width, attn.num_heads, attn.head_dim).transpose(1, 2)
        weights = F.elu(torch.matmul(q, k.transpose(-2, -1)) * attn.scale) + 1
        weights = weights * valid[:, None, None, :]
        read = torch.matmul(weights, v)
        # Each query sees its own K/V once; it never sees another candidate.
        self_score = torch.matmul(q.unsqueeze(-2), self_k.unsqueeze(-1)).squeeze(-1)
        self_weight = F.elu(self_score * attn.scale) + 1
        read = read + self_weight * self_v
        output = attn._finish(read)
        query = query + output * F.silu(block.gate_proj(normalized))
    return model.cc_score_head(model.final_norm(query)).reshape(batch, count)


class FeedbackData:
    """Compact CPU metadata and GPU-resident causal event arrays."""

    def __init__(self, root, device):
        root = Path(root)
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        original = Path(self.manifest["source_root"])
        self.requests = np.load(original / "requests.npy", mmap_mode="r")
        self.population = dict(np.load(original / "population.npz"))
        self.source_events = np.load(original / "events.npy", mmap_mode="r")
        self.mapped_items = np.load(root / "items.npy", mmap_mode="r")
        self.valid = self.requests["history_stop"] > self.requests["history_start"]
        self.counts = self.requests["event_stop"] - self.requests["event_start"]
        self.lengths = np.minimum(self.requests["history_stop"] - self.requests["history_start"], 1024)
        self.device = torch.device(device)
        def resident(array, dtype):
            return torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=self.device)
        req = self.requests
        self.items = resident(self.mapped_items, torch.int32)
        self.behaviors = resident(self.source_events["behavior"], torch.uint8)
        self.deltas = resident(np.load(root / "deltas.npy", mmap_mode="r"), torch.float32)
        self.starts = resident(req["event_start"], torch.int64)
        self.counts_gpu = resident(self.counts, torch.int32)
        self.stops = resident(req["history_stop"], torch.int64)
        self.lengths_gpu = resident(self.lengths, torch.int32)
        gaps = (req["ts"] - self.source_events["ts"][np.maximum(req["history_stop"] - 1, 0)]) / 1000
        self.gaps = resident(gaps, torch.float32)
        self.weights = torch.zeros(len(req), device=self.device)

    def indices(self, role, first_day, last_day, users=None):
        req = self.requests
        mask = self.valid & (req["role"] == role) & (req["day"] >= first_day) & (req["day"] <= last_day)
        if users is not None:
            mask &= np.isin(req["uid"], users)
        return np.flatnonzero(mask)

    def set_training_weights(self, indices):
        _, inverse = np.unique(self.requests["uid"][indices], return_inverse=True)
        per_user = np.bincount(inverse, weights=self.counts[indices])
        weights = len(indices) / (len(per_user) * per_user[inverse])
        self.weights.zero_()
        self.weights[torch.as_tensor(indices, device=self.device)] = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        return dict(requests=len(indices), users=len(per_user), targets=int(self.counts[indices].sum()))

    def batch(self, indices, batch_size, width=None, candidates=None, denominator=None):
        real = len(indices)
        width = width or int(self.lengths[indices].max())
        candidates = candidates or int(self.counts[indices].max())
        padded = np.resize(indices if real else np.array([0]), batch_size)
        ix = torch.as_tensor(padded, device=self.device)
        active = torch.arange(batch_size, device=self.device) < real
        lengths = self.lengths_gpu[ix].long()
        positions = self.stops[ix, None] - lengths[:, None] + torch.arange(width, device=self.device)[None]
        valid = torch.arange(width, device=self.device)[None] < lengths[:, None]
        positions = positions.clamp(0, len(self.items) - 1)
        items = torch.where(valid, self.items[positions], 0).long()
        behaviors = torch.where(valid, self.behaviors[positions].long() + 1, 0)
        deltas = torch.where(valid, self.deltas[positions], 0)
        offsets = torch.arange(candidates, device=self.device)[None]
        targets = (self.starts[ix, None] + offsets).clamp_max(len(self.items) - 1)
        target_mask = (offsets < self.counts_gpu[ix, None]) & active[:, None]
        candidate_ids = torch.where(target_mask, self.items[targets], 0).long()
        labels = (self.behaviors[targets] & 1).float()
        weights = target_mask * self.weights[ix, None] / (denominator or real)
        return (items, behaviors, deltas, lengths, candidate_ids, self.gaps[ix], labels, weights, target_mask)

    def embedding_rows(self, indices, width):
        """Exact union of all looked-up rows, including PAD, for gradient SUM.

        The dense AdamW gradient is zero outside this union. Communicating
        these rows is equivalent to communicating the entire dense table.
        """
        lengths = self.lengths[indices]
        positions = self.requests["history_stop"][indices, None] - lengths[:, None] + np.arange(width)[None]
        valid = np.arange(width)[None] < lengths[:, None]
        history = self.mapped_items[positions[valid]]
        starts = self.requests["event_start"][indices]
        candidates = np.concatenate([self.mapped_items[a:a + self.counts[i]] for a, i in zip(starts, indices)])
        return np.unique(np.concatenate((history, candidates, np.array([0], dtype=np.int32))))


def feedback_loss(model, batch):
    logits = grouped_scores(model, *batch[:6])
    loss = F.binary_cross_entropy_with_logits(logits.float(), batch[6], reduction="none")
    return (loss * batch[7]).sum(), logits

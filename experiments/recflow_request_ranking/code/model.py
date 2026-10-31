"""Feature-aware causal Transformer with one shared history per request."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from hstu_kvcache.models.hstu import HSTU
from hstu_kvcache.models.kv_cache import HSTUKVCache

from .configuration import backbone_config, load_proposal
from .shared_reader import score_shared_cache


class ItemFeatures(nn.Module):
    def __init__(self, item_map, explicit_ids, buckets, hidden_size, embedding_dim=64):
        super().__init__()
        self.explicit_ids, self.buckets = explicit_ids, buckets
        # Identical immutable lookup on all releases; do not copy into each
        # checkpoint or broadcast this large buffer at every DDP forward.
        self.register_buffer("item_map", torch.as_tensor(np.array(item_map, copy=True)), persistent=False)
        self.item = nn.Embedding(explicit_ids+buckets+1, embedding_dim, padding_idx=0)
        self.author = nn.Embedding(262144+1, 16, padding_idx=0)
        self.category_one = nn.Embedding(4096+1, 8, padding_idx=0)
        self.category_two = nn.Embedding(16384+1, 16, padding_idx=0)
        self.numeric = nn.Linear(1, 8)
        self.projection = nn.Linear(embedding_dim+48, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        for table in [self.item, self.author, self.category_one, self.category_two]:
            nn.init.normal_(table.weight, std=.02)
            with torch.no_grad():
                table.weight[0].zero_()

    @staticmethod
    def hash(ids, buckets):
        return ((ids ^ (ids >> 16))*2654435761 + 17).remainder(buckets)

    def forward(self, features):
        raw = features.long()
        valid = raw[..., 0] >= 0
        safe = raw.clamp_min(0)
        mapped = self.map_ids(raw[..., 0])
        vectors = [self.item(mapped), self.author(self.hash(safe[..., 1], 262144)+1),
                   self.category_one(safe[..., 2].remainder(4096)+1),
                   self.category_two(safe[..., 3].remainder(16384)+1),
                   self.numeric(torch.log1p(safe[..., 4:5].float()/1000)/10)]
        return self.norm(self.projection(torch.cat(vectors, dim=-1))) * valid.unsqueeze(-1)

    def map_ids(self, raw_ids):
        valid = raw_ids >= 0
        item_id = raw_ids.long().clamp_min(0)
        mapped = self.item_map[item_id.clamp_max(len(self.item_map)-1)].long()
        mapped = torch.where(item_id < len(self.item_map), mapped,
                             self.explicit_ids+1+self.hash(item_id, self.buckets))
        mapped = torch.where(valid, mapped, 0)
        return mapped


class RequestRanker(nn.Module):
    def __init__(self, vocabulary_root, scale="pilot", config=None):
        super().__init__()
        self.proposal = load_proposal() if config is None else config
        self.scale = scale
        root = Path(vocabulary_root)
        self.vocabulary = json.loads((root / "manifest.json").read_text())
        cfg = backbone_config(self.proposal, scale, num_items=1)
        self.backbone = HSTU(cfg)
        # This parameter is bypassed by external vectors. A frozen dummy avoids
        # an unused trainable parameter hanging ordinary DDP reductions.
        self.backbone.item_emb.requires_grad_(False)
        self.features = ItemFeatures(np.load(root / "item_map.npy", mmap_mode="r"),
                                     self.vocabulary["explicit_ids"], self.vocabulary["hash_buckets"],
                                     cfg.hidden_size,
                                     self.proposal["architecture"][scale]["item_embedding_dimension"])
        self.chunk_size = self.proposal["execution"]["candidate_chunk_size"]
        self.checkpoint_queries = self.proposal["execution"]["checkpoint_candidate_chunks"]

    def encode_history(self, batch, *, reference=False):
        model = self.backbone
        vectors = self.features(batch["history_features"])
        hidden = model.combine_input_features(vectors, batch["behaviors"], batch["deltas"])
        if reference:
            return model.forward_embedded(hidden, return_kv=True, return_hidden=False,
                                          lengths=batch["lengths"])[1]
        # The last layer's history OUTPUT is never read: queries need its K/V.
        # Skip only that dead attention computation; preserve native cache math.
        valid = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < batch["lengths"][:, None]
        hidden = hidden * valid.unsqueeze(-1)
        pairs = []
        for layer, block in enumerate(model.blocks):
            if layer == len(model.blocks)-1:
                normalized = block.norm(hidden)
                pairs.append((F.silu(block.attn.k_proj(normalized)), F.silu(block.attn.v_proj(normalized))))
            else:
                hidden, kv = block(hidden, return_kv=True)
                pairs.append(kv)
                hidden = hidden * valid.unsqueeze(-1)
        return HSTUKVCache.from_layer_list(pairs, seq_len=hidden.shape[1])

    def read(self, batch, cache):
        vectors = self.features(batch["candidate_features"])
        ids = batch["candidate_features"][..., 0].long().clamp_min(0)
        lengths, query_deltas = batch["lengths"], batch["query_deltas"]

        def chunk_read(chunk_ids, chunk_vectors, keys, values):
            return score_shared_cache(self.backbone, HSTUKVCache(keys, values, cache.seq_len),
                                      chunk_ids, query_deltas, lengths, item_vectors=chunk_vectors, validate=False)

        output = []
        for start in range(0, ids.shape[1], self.chunk_size):
            args = (ids[:, start:start+self.chunk_size], vectors[:, start:start+self.chunk_size], cache.k, cache.v)
            if self.training and self.checkpoint_queries and torch.is_grad_enabled():
                output.append(checkpoint(chunk_read, *args, use_reentrant=False))
            else:
                output.append(chunk_read(*args))
        return torch.cat(output, dim=1)

    def forward(self, batch):
        return self.read(batch, self.encode_history(batch))

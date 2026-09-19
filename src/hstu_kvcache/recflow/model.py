"""Three-token, catalog-constrained generation on the existing HSTU backbone.

History consists only of real exposure tokens. The transient suffix is
``[REC], category_1, category_2``; its states predict the two categories and
then the video. Training and inference normalize over exactly the same legal
children. Category pairs, rather than category-2 alone, identify leaf groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hstu_kvcache.models.hstu import HSTU, HSTUConfig
from hstu_kvcache.models.kv_cache import HSTUKVCache


class RecFlowGenerator(nn.Module):
    """Known IDs are 1..K; PAD=0 and OOV=K+1 have paths [-1, -1].

    ``cfg.num_items`` is the maximum item ID, including OOV, as in HSTU.
    ``cfg.max_seq_len`` must allow the history plus three transient tokens.
    ``item_paths`` has shape [K+2, 2]; raw nonnegative category IDs are compacted
    here once. A video's path is fixed by the data preparation cutoff.
    ``query_time_deltas`` is request time minus last history-event time, in
    seconds. It enters only the REC token; omitted gaps default to zero.
    ``history_categories`` optionally adds fixed catalog category embeddings
    to known history items, preserving one token per exposure and the same
    parameter tables. The default retains the original ID-only item features.
    """

    def __init__(
        self, cfg: HSTUConfig, item_paths: torch.Tensor, history_categories: bool = False,
    ) -> None:
        super().__init__()
        self.history_categories = history_categories
        raw_paths = torch.as_tensor(item_paths, dtype=torch.long, device="cpu")
        if raw_paths.shape != (cfg.num_items + 1, 2):
            raise ValueError("item_paths must include PAD, known items, and OOV")
        if cfg.causal_diagonal != "inclusive":
            raise ValueError("three-token decoder requires inclusive causal attention")
        if bool((raw_paths[1:-1] < 0).any()) or bool((raw_paths[[0, -1]] != -1).any()):
            raise ValueError("known items need paths; PAD and OOV must have no path")
        paths = torch.full_like(raw_paths, -1)
        c1_values, paths[1:-1, 0] = torch.unique(raw_paths[1:-1, 0], return_inverse=True)
        c2_values, paths[1:-1, 1] = torch.unique(raw_paths[1:-1, 1], return_inverse=True)
        self.num_c1, self.num_c2 = len(c1_values), len(c2_values)
        self.num_known = len(paths) - 2
        if not self.num_known:
            raise ValueError("catalog must contain a known item")
        self.backbone = HSTU(cfg)
        h = cfg.hidden_size
        self.rec_token = nn.Parameter(torch.empty(h))
        self.c1_embedding = nn.Embedding(self.num_c1, h)
        self.c2_embedding = nn.Embedding(self.num_c2, h)
        self.c1_head = nn.Linear(h, self.num_c1)
        self.c2_head = nn.Linear(h, self.num_c2)
        self.leaf_bias = nn.Parameter(torch.zeros(cfg.num_items + 1))
        nn.init.normal_(self.rec_token, std=0.02)
        nn.init.normal_(self.c1_embedding.weight, std=0.02)
        nn.init.normal_(self.c2_embedding.weight, std=0.02)
        child_mask = torch.zeros(self.num_c1, self.num_c2, dtype=torch.bool)
        child_mask[paths[1:-1, 0], paths[1:-1, 1]] = True
        pair_keys = paths[1:-1, 0] * self.num_c2 + paths[1:-1, 1]
        order = torch.argsort(pair_keys, stable=True)
        keys, counts = torch.unique_consecutive(pair_keys[order], return_counts=True)
        self._leaf_spans: dict[int, tuple[int, int]] = {}
        start = 0
        for key, count in zip(keys.tolist(), counts.tolist(), strict=True):
            self._leaf_spans[key] = (start, start + count)
            start += count
        self.register_buffer("item_paths", paths)
        self.register_buffer("c2_child_mask", child_mask)
        self.register_buffer("leaf_order", order + 1)

    def _transient(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.backbone.input_dropout(self.backbone.in_proj(vectors))

    def _history_item_vectors(self, item_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.backbone.lookup_item_embeddings(item_ids)
        if self.history_categories:
            paths = self.item_paths[item_ids]
            known = (paths[..., 0] >= 0)[..., None]
            categories = self.c1_embedding(paths[..., 0].clamp_min(0))
            categories = categories + self.c2_embedding(paths[..., 1].clamp_min(0))
            vectors = vectors + categories * known
        return vectors

    def _embed_history(self, item_ids, behaviors, time_deltas) -> torch.Tensor:
        if not self.history_categories:
            return self.backbone.embed_inputs(item_ids, behaviors, time_deltas)
        return self.backbone.combine_input_features(
            self._history_item_vectors(item_ids), behaviors, time_deltas,
        )

    def _request_vectors(self, batch: int, query_time_deltas: torch.Tensor | None) -> torch.Tensor:
        if query_time_deltas is None:
            query_time_deltas = torch.zeros(batch, device=self.rec_token.device)
        if query_time_deltas.shape != (batch,):
            raise ValueError("query_time_deltas must have shape [B]")
        # The request time is known at prediction. Category/video identity is not.
        gaps = query_time_deltas.to(device=self.rec_token.device, dtype=torch.float32)
        return self.rec_token[None] + self.backbone.temporal_enc(gaps)

    def _leaves(self, pair_key: int) -> torch.Tensor:
        lo, hi = self._leaf_spans[pair_key]
        return self.leaf_order[lo:hi]

    def _leaf_log_probs(self, hidden: torch.Tensor, leaves: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, self.backbone.item_emb.weight[leaves], self.leaf_bias[leaves])
        return F.log_softmax(logits.float(), dim=-1)

    def _c2_log_probs(self, hidden: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
        logits = self.c2_head(hidden).float()
        return F.log_softmax(logits.masked_fill(~self.c2_child_mask[c1], -torch.inf), dim=-1)

    def _teacher_states(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        query_time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bool(((targets < 1) | (targets > self.num_known)).any()):
            raise ValueError("teacher targets must be known positive item IDs")
        batch, width = item_ids.shape
        if width + 3 > self.backbone.cfg.max_seq_len:
            raise ValueError("max_seq_len must include the three transient tokens")
        lengths = lengths.to(device=item_ids.device, dtype=torch.long)
        paths = self.item_paths[targets]
        suffix = self._transient(torch.stack([
            self._request_vectors(batch, query_time_deltas),
            self.c1_embedding(paths[:, 0]),
            self.c2_embedding(paths[:, 1]),
        ], dim=1))
        # Scatter suffix at each true prefix end, never after its right padding.
        prefix = self._embed_history(item_ids, behaviors, time_deltas)
        valid = torch.arange(width, device=item_ids.device)[None] < lengths[:, None]
        x = F.pad(prefix * valid[..., None], (0, 0, 0, 3))
        indices = lengths[:, None] + torch.arange(3, device=item_ids.device)[None]
        x = x.scatter(1, indices[..., None].expand(-1, -1, x.shape[-1]), suffix)
        hidden, _ = self.backbone.forward_embedded(x, lengths=lengths + 3)
        return hidden.gather(1, indices[..., None].expand(-1, -1, hidden.shape[-1]))

    def teacher_log_probs(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        query_time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return each target's three conditional log probabilities, [B, 3]."""
        hidden = self._teacher_states(
            item_ids, behaviors, time_deltas, lengths, targets, query_time_deltas,
        )
        paths = self.item_paths[targets]
        first = F.log_softmax(self.c1_head(hidden[:, 0]).float(), dim=-1)
        second = self._c2_log_probs(hidden[:, 1], paths[:, 0])
        rows = torch.arange(len(targets), device=targets.device)
        leaf_lp = torch.zeros(len(targets), device=hidden.device, dtype=torch.float32)
        pair_keys = paths[:, 0] * self.num_c2 + paths[:, 1]
        keys = torch.unique(pair_keys).tolist()
        leaf_groups = [self._leaves(key) for key in keys]
        # Gather the batch's disjoint leaf groups together. Backward then
        # scatters into the full item table once, rather than once per pair.
        batch_leaves = torch.cat(leaf_groups)
        batch_weights = self.backbone.item_emb.weight[batch_leaves]
        batch_bias = self.leaf_bias[batch_leaves]
        offset = 0
        for key, leaves in zip(keys, leaf_groups, strict=True):
            selected = torch.nonzero(pair_keys == key, as_tuple=True)[0]
            local_targets = torch.searchsorted(leaves, targets[selected])
            stop = offset + len(leaves)
            logits = F.linear(hidden[selected, 2], batch_weights[offset:stop], batch_bias[offset:stop])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            leaf_lp = leaf_lp.index_copy(0, selected, log_probs.gather(1, local_targets[:, None])[:, 0])
            offset = stop
        return torch.stack([first[rows, paths[:, 0]], second[rows, paths[:, 1]], leaf_lp], dim=-1)

    def loss_per_example(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        query_time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum token NLL for one positive per row; caller averages by request."""
        return -self.teacher_log_probs(
            item_ids, behaviors, time_deltas, lengths, targets, query_time_deltas,
        ).sum(-1)

    @staticmethod
    def _select_cache(cache: HSTUKVCache, rows: torch.Tensor, length: int | None = None) -> HSTUKVCache:
        stop = cache.seq_len if length is None else length
        return HSTUKVCache(cache.k[:, rows, :stop], cache.v[:, rows, :stop], stop)

    def _prefix_cache(self, item_ids, behaviors, time_deltas, lengths) -> HSTUKVCache:
        if item_ids.shape[1] + 3 > self.backbone.cfg.max_seq_len:
            raise ValueError("max_seq_len must include the three transient tokens")
        if self.history_categories:
            return self.backbone.compute_kv_from_item_embeddings(
                self._history_item_vectors(item_ids), behaviors, time_deltas, lengths=lengths,
            )
        return self.backbone.compute_kv(item_ids, behaviors, time_deltas, lengths=lengths)

    def _start(
        self, cache: HSTUKVCache, request_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, HSTUKVCache]:
        hidden, rec_cache = self.backbone.forward_with_cache_embedded(
            cache, self._transient(request_vector[:, None])
        )
        return F.log_softmax(self.c1_head(hidden[:, 0]).float(), dim=-1)[0], rec_cache

    def _advance_c1(self, cache: HSTUKVCache, c1: torch.Tensor):
        branches = self._select_cache(cache, torch.zeros_like(c1))
        hidden, branches = self.backbone.forward_with_cache_embedded(
            branches, self._transient(self.c1_embedding(c1)[:, None])
        )
        return self._c2_log_probs(hidden[:, 0], c1), branches

    @torch.no_grad()
    def score_items(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        candidate_ids: torch.Tensor,
        query_time_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact joint log probabilities for [B, C] candidates; PAD/OOV=-inf.

        Candidate subsets affect neither the category nor the leaf denominators.
        The real prefix is encoded once, and transient branches are discarded.
        """
        was_training = self.training
        self.eval()
        try:
            prefix = self._prefix_cache(item_ids, behaviors, time_deltas, lengths)
            requests = self._request_vectors(len(item_ids), query_time_deltas)
            output = torch.full(candidate_ids.shape, -torch.inf, device=item_ids.device)
            for row, length in enumerate(lengths.tolist()):
                ids = candidate_ids[row]
                valid = (ids >= 1) & (ids <= self.num_known)
                if not bool(valid.any()):
                    continue
                valid_cols = torch.nonzero(valid, as_tuple=True)[0]
                targets = ids[valid]
                paths = self.item_paths[targets]
                selected_c1 = torch.unique(paths[:, 0])
                cache = self._select_cache(prefix, torch.tensor([row], device=ids.device), length)
                first, rec_cache = self._start(cache, requests[row:row + 1])
                second, c1_cache = self._advance_c1(rec_cache, selected_c1)
                keys = paths[:, 0] * self.num_c2 + paths[:, 1]
                unique_keys = torch.unique(keys)
                branch_c1 = torch.div(unique_keys, self.num_c2, rounding_mode="floor")
                branch_c2 = unique_keys % self.num_c2
                c1_rows = torch.searchsorted(selected_c1, branch_c1)
                hidden, _ = self.backbone.forward_with_cache_embedded(
                    self._select_cache(c1_cache, c1_rows),
                    self._transient(self.c2_embedding(branch_c2)[:, None]),
                )
                for branch, key in enumerate(unique_keys.tolist()):
                    cols = torch.nonzero(keys == key, as_tuple=True)[0]
                    leaves = self._leaves(key)
                    leaf_lp = self._leaf_log_probs(hidden[branch, 0], leaves)
                    values = first[branch_c1[branch]] + second[c1_rows[branch], branch_c2[branch]]
                    values = values + leaf_lp[torch.searchsorted(leaves, targets[cols])]
                    output[row, valid_cols[cols]] = values
            return output
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_topk(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        k: int = 100,
        beam_width: int = 100,
        query_time_deltas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Free constrained beam search, ranking the sum of three log probs.

        Each category stage keeps ``beam_width`` cumulative paths. The final
        stage scores all legal leaves of retained paths. With a beam covering
        every category pair this equals exhaustive full-catalog ranking. Narrow
        beams may return fewer than k videos, padded with ID 0 and score -inf.
        The decoder never receives target items or target categories.
        """
        if min(k, beam_width) < 1:
            raise ValueError("k and beam_width must be positive")
        was_training = self.training
        self.eval()
        try:
            prefix = self._prefix_cache(item_ids, behaviors, time_deltas, lengths)
            requests = self._request_vectors(len(item_ids), query_time_deltas)
            k = min(k, self.num_known)
            out_ids = torch.zeros((len(item_ids), k), dtype=torch.long, device=item_ids.device)
            out_scores = torch.full(out_ids.shape, -torch.inf, device=item_ids.device)
            for row, length in enumerate(lengths.tolist()):
                cache = self._select_cache(prefix, torch.tensor([row], device=item_ids.device), length)
                first, rec_cache = self._start(cache, requests[row:row + 1])
                c1 = first.topk(min(beam_width, self.num_c1)).indices
                second, c1_cache = self._advance_c1(rec_cache, c1)
                pair_scores = first[c1, None] + second
                legal = torch.nonzero(torch.isfinite(pair_scores.flatten()), as_tuple=True)[0]
                selected = pair_scores.flatten()[legal].topk(min(beam_width, len(legal))).indices
                flat = legal[selected]
                c1_rows = torch.div(flat, self.num_c2, rounding_mode="floor")
                c2 = flat % self.num_c2
                hidden, _ = self.backbone.forward_with_cache_embedded(
                    self._select_cache(c1_cache, c1_rows),
                    self._transient(self.c2_embedding(c2)[:, None]),
                )
                score_parts, id_parts = [], []
                keys = c1[c1_rows] * self.num_c2 + c2
                for branch, key in enumerate(keys.tolist()):
                    leaves = self._leaves(key)
                    leaf_lp = self._leaf_log_probs(hidden[branch, 0], leaves)
                    best = leaf_lp.topk(min(k, len(leaves)))
                    score_parts.append(best.values + pair_scores[c1_rows[branch], c2[branch]])
                    id_parts.append(leaves[best.indices])
                scores, ids = torch.cat(score_parts), torch.cat(id_parts)
                best = scores.topk(min(k, len(scores)))
                out_ids[row, :len(best.indices)] = ids[best.indices]
                out_scores[row, :len(best.indices)] = best.values
            return out_ids, out_scores
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_exact_topk(
        self,
        item_ids: torch.Tensor,
        behaviors: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        k: int = 100,
        branch_batch_size: int = 32,
        query_time_deltas: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        """Full-catalog Top-K using category-pair probability upper bounds.

        Every leaf log probability is nonpositive, so its pair's cumulative
        log probability bounds its final score from above. Expand pairs in
        descending bound order, stopping only once the best remaining bound
        is below the current Kth item. This changes search, not model scores.
        Batched suffix expansion limits temporary K/V memory at long contexts.
        Use FP32 for exhaustive-ranking comparisons: BF16 rounding can change
        near-tied scores when the decoder branch batch shape changes.
        """
        if min(k, branch_batch_size) < 1:
            raise ValueError("k and branch_batch_size must be positive")
        was_training = self.training
        self.eval()
        try:
            prefix = self._prefix_cache(item_ids, behaviors, time_deltas, lengths)
            requests = self._request_vectors(len(item_ids), query_time_deltas)
            k = min(k, self.num_known)
            out_ids = torch.zeros((len(item_ids), k), dtype=torch.long, device=item_ids.device)
            out_scores = torch.full(out_ids.shape, -torch.inf, device=item_ids.device)
            expanded = []
            for row, length in enumerate(lengths.tolist()):
                cache = self._select_cache(prefix, torch.tensor([row], device=item_ids.device), length)
                first, rec_cache = self._start(cache, requests[row:row + 1])
                c1 = first.argsort(descending=True)
                second, c1_cache = self._advance_c1(rec_cache, c1)
                pair_scores = first[c1, None] + second
                legal = torch.nonzero(torch.isfinite(pair_scores.flatten()), as_tuple=True)[0]
                order = pair_scores.flatten()[legal].argsort(descending=True)
                flat = legal[order]
                bounds = pair_scores.flatten()[flat]
                best_ids = out_ids.new_empty(0)
                best_scores = out_scores.new_empty(0)
                cursor = 0
                while cursor < len(flat):
                    if len(best_scores) == k and bool(bounds[cursor] < best_scores[-1]):
                        break
                    stop = min(cursor + branch_batch_size, len(flat))
                    chunk = flat[cursor:stop]
                    c1_rows = torch.div(chunk, self.num_c2, rounding_mode="floor")
                    c2 = chunk % self.num_c2
                    hidden, temporary = self.backbone.forward_with_cache_embedded(
                        self._select_cache(c1_cache, c1_rows),
                        self._transient(self.c2_embedding(c2)[:, None]),
                    )
                    # These cache branches never become persistent history.
                    del temporary
                    score_parts, id_parts = [best_scores], [best_ids]
                    keys = c1[c1_rows] * self.num_c2 + c2
                    for branch, key in enumerate(keys.tolist()):
                        leaves = self._leaves(key)
                        leaf_lp = self._leaf_log_probs(hidden[branch, 0], leaves)
                        top = leaf_lp.topk(min(k, len(leaves)))
                        score_parts.append(top.values + bounds[cursor + branch])
                        id_parts.append(leaves[top.indices])
                    scores, ids = torch.cat(score_parts), torch.cat(id_parts)
                    best = scores.topk(min(k, len(scores)))
                    best_scores, best_ids = best.values, ids[best.indices]
                    cursor = stop
                out_ids[row], out_scores[row] = best_ids, best_scores
                expanded.append(cursor)
            if return_stats:
                return out_ids, out_scores, {
                    "expanded_pairs": expanded, "total_pairs": len(self._leaf_spans),
                    "branch_batch_size": branch_batch_size,
                }
            return out_ids, out_scores
        finally:
            self.train(was_training)

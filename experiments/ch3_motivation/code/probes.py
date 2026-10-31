"""Candidate/row correction probes for the Chapter 3 development study.

The backbone is frozen. ``f`` is materialized once into a cache; ``g`` and
``h`` modify only the history read. The candidate's native self-attention is
left to adaptation.reader, including the actual checkpoint's activation,
scaling, diagonal convention, gates and final score head.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from hstu_kvcache.adaptation.reader import history_read, read_embedded, score as native_score
from hstu_kvcache.models import HSTUKVCache


@dataclass
class ProbeConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    item_dim: int
    hidden_width: int = 64
    depth: int = 2
    include_item: bool = True
    normalize: bool = True
    query_chunk: int = 16
    row_chunk: int = 128

    @classmethod
    def from_model(cls, model, **kwargs):
        head_dim = model.blocks[0].attn.head_dim
        return cls(model.cfg.num_layers, model.cfg.num_heads, head_dim,
                   model.cfg.hidden_size // model.cfg.num_heads, **kwargs)

    @property
    def row_dim(self):
        return 2 * self.head_dim + (self.item_dim if self.include_item else 0)


class HeadMLP(nn.Module):
    """Independent head MLPs, batched over users, candidates and rows.

    Inputs are [B,H,...,input_width]. There is no rank constraint on the
    candidate-row interaction; hidden_width is an ordinary nonlinear capacity.
    """
    def __init__(self, heads, input_width, output_width, hidden_width, depth):
        super().__init__()
        if depth < 1:
            raise ValueError("MLP depth must be positive")
        widths = [input_width] + [hidden_width] * depth + [output_width]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for n_in, n_out in zip(widths[:-1], widths[1:], strict=True):
            weight = nn.Parameter(torch.empty(heads, n_in, n_out))
            nn.init.normal_(weight, std=n_in ** -.5)
            self.weights.append(weight)
            self.biases.append(nn.Parameter(torch.zeros(heads, n_out)))
        nn.init.zeros_(self.weights[-1])

    def forward(self, x):
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = torch.einsum("bh...i,hio->bh...o", x, weight)
            x = x + bias.reshape(1, bias.shape[0], *([1] * (x.ndim - 3)), -1)
            if index + 1 != len(self.weights):
                x = torch.nn.functional.silu(x)
        return x


class ProbeLadder(nn.Module):
    def __init__(self, cfg: ProbeConfig, stage="f"):
        super().__init__()
        if stage not in ("f", "fg", "fgh"):
            raise ValueError("probe stage must be f, fg or fgh")
        if min(cfg.query_chunk, cfg.row_chunk) < 1:
            raise ValueError("query and row chunks must be positive")
        self.cfg, self.stage = cfg, stage
        args = (cfg.num_heads, 2 * cfg.head_dim, cfg.hidden_width, cfg.depth)
        self.f = nn.ModuleList(HeadMLP(args[0], cfg.row_dim, *args[1:])
                               for _ in range(cfg.num_layers))
        self.g = nn.ModuleList(HeadMLP(args[0], cfg.head_dim, *args[1:])
                               for _ in range(cfg.num_layers)) if stage != "f" else None
        self.h = nn.ModuleList(HeadMLP(args[0], cfg.head_dim + cfg.row_dim, *args[1:])
                               for _ in range(cfg.num_layers)) if stage == "fgh" else None
        self.register_buffer("row_mean", torch.zeros(cfg.num_layers, cfg.num_heads, cfg.row_dim))
        self.register_buffer("row_scale", torch.ones_like(self.row_mean))
        self.register_buffer("query_scale", torch.ones(cfg.num_layers, cfg.num_heads, cfg.head_dim))
        self.register_buffer("statistics_initialized", torch.tensor(False))

    def expanded(self, stage):
        """Nested initialization: old modules identical, new output exactly zero."""
        order = {"f": 0, "fg": 1, "fgh": 2}
        if stage not in order or order[stage] < order[self.stage]:
            raise ValueError("nested expansion cannot remove a probe term")
        result = ProbeLadder(copy.deepcopy(self.cfg), stage).to(self.row_mean)
        result.load_state_dict(self.state_dict(), strict=False)
        return result

    def _heads(self, x):
        batch, length, width = x.shape
        if width != self.cfg.num_heads * self.cfg.head_dim:
            raise ValueError("cache width differs from probe heads and head dimension")
        return x.reshape(batch, length, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)

    def _items(self, model, item_ids, item_embeddings=None):
        if not self.cfg.include_item:
            return None
        embedded = (model.lookup_item_embeddings(item_ids) if item_embeddings is None
                    else item_embeddings).detach()
        batch, length, width = embedded.shape
        if width != self.cfg.num_heads * self.cfg.item_dim:
            raise ValueError("item features must split evenly across configured heads")
        return embedded.reshape(batch, length, self.cfg.num_heads, self.cfg.item_dim).transpose(1, 2)

    def _raw_rows(self, layer, cache, items):
        parts = [self._heads(cache.k[layer]), self._heads(cache.v[layer])]
        if items is not None:
            parts.append(items)
        return torch.cat(parts, -1)

    def row_features(self, model, source_cache, item_ids, layer, *, item_embeddings=None):
        """Fixed row features for E6 and direct reader checks, [B,H,N,F]."""
        return self._normalize_rows(layer, self._raw_rows(
            layer, source_cache, self._items(model, item_ids, item_embeddings)))

    def prepare_read_features(self, model, source_cache, item_ids, *, item_embeddings=None):
        """Fixed, reusable per-row inputs for all h reads within the release.

        Slice [:,:,dropped:] on each tensor when inherited rows are evicted.
        No trainable MLP is evaluated here, so this is also safe during fitting.
        """
        items = self._items(model, item_ids, item_embeddings)
        return tuple(self._normalize_rows(layer, self._raw_rows(layer, source_cache, items))
                     for layer in range(self.cfg.num_layers))

    def _normalize_rows(self, layer, rows):
        if not self.cfg.normalize:
            return rows
        return ((rows - self.row_mean[layer][None, :, None]) /
                self.row_scale[layer][None, :, None])

    def _output_scale(self, layer, ndim):
        if not self.cfg.normalize:
            return 1.
        scale = self.row_scale[layer, :, :2 * self.cfg.head_dim]
        return scale.reshape(1, self.cfg.num_heads, *([1] * (ndim - 3)), -1)

    def query_correction(self, layer, q):
        if self.g is None:
            return q.new_zeros(*q.shape[:-1], 2 * self.cfg.head_dim)
        normalized = q / self.query_scale[layer][None, :, None] if self.cfg.normalize else q
        value = self.g[layer](normalized)
        return value * self._output_scale(layer, value.ndim)

    def interaction_correction(self, layer, q, rows):
        """Full-capacity h output [B,H,Q,N,2D], with no imposed low-rank form.

        Callers should chunk Q/N. ``rows`` is the normalized row_features()
        output. Output units are the original K/V units.
        """
        if self.h is None:
            raise ValueError("interaction correction requires stage fgh")
        normalized = q / self.query_scale[layer][None, :, None] if self.cfg.normalize else q
        nq, nr = q.shape[2], rows.shape[2]
        features = torch.cat((normalized[:, :, :, None].expand(-1, -1, -1, nr, -1),
                              rows[:, :, None].expand(-1, -1, nq, -1, -1)), -1)
        value = self.h[layer](features)
        return value * self._output_scale(layer, value.ndim)

    def prepare_cache(self, model, parent_cache, item_ids, *, item_embeddings=None, read_features=None):
        """Materialize f once per inherited row. Does not mutate native input."""
        features = (self.prepare_read_features(model, parent_cache, item_ids, item_embeddings=item_embeddings)
                    if read_features is None else read_features)
        keys, values = [], []
        for layer in range(self.cfg.num_layers):
            delta = self.f[layer](features[layer])
            delta = delta * self._output_scale(layer, delta.ndim)
            dk, dv = delta.chunk(2, -1)
            flatten = lambda x: x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.cfg.num_heads*self.cfg.head_dim)
            keys.append(parent_cache.k[layer] + flatten(dk))
            values.append(parent_cache.v[layer] + flatten(dv))
        return HSTUKVCache(torch.stack(keys), torch.stack(values), parent_cache.seq_len)

    def make_reader(self, model, parent_cache, item_ids, prepared_cache=None, *, item_embeddings=None,
                    read_features=None, interaction_override=None):
        """Build a history callback. Native candidate self remains in reader.py.

        parent_cache supplies the immutable inherited row inputs to h. The
        cache supplied to score contains f's materialization. A native suffix
        appended after preparation receives no f/g/h correction. On FIFO
        eviction the caller must slice parent_cache and item_ids together.
        """
        features = read_features
        if self.h is not None and features is None:
            features = self.prepare_read_features(model, parent_cache, item_ids, item_embeddings=item_embeddings)

        def read(layer, attention, q, k, v):
            if attention.position_bias is not None:
                raise ValueError("Chapter 3 probe assumes no relative position bias")
            if self.stage == "f":
                return history_read(attention, q, k, v)
            inherited = parent_cache.seq_len
            if k.shape[1] < inherited:
                raise ValueError("slice inherited inputs together after FIFO eviction")
            kh, vh = self._heads(k), self._heads(v)
            rows = features[layer] if self.h is not None else None
            outputs = []
            for query_start in range(0, q.shape[2], self.cfg.query_chunk):
                qc = q[:, :, query_start:query_start + self.cfg.query_chunk]
                gk, gv = self.query_correction(layer, qc).chunk(2, -1)
                result = torch.zeros_like(qc)
                for start in range(0, inherited, self.cfg.row_chunk):
                    end = min(start + self.cfg.row_chunk, inherited)
                    ks, vs = kh[:, :, start:end], vh[:, :, start:end]
                    if self.h is None:
                        dots = qc @ ks.transpose(-2, -1) + (qc * gk).sum(-1, keepdim=True)
                        hv = None
                    else:
                        interaction = (self.interaction_correction(layer, qc, rows[:, :, start:end])
                                       if interaction_override is None else
                                       interaction_override(layer, qc, rows[:, :, start:end], query_start, start))
                        hk, hv = interaction.chunk(2, -1)
                        dots = (qc @ ks.transpose(-2, -1) +
                                (qc * gk).sum(-1, keepdim=True) +
                                (qc[:, :, :, None] * hk).sum(-1))
                    weights = attention._activate(dots * attention.scale)
                    if attention.block_variant == "hstu_reference":
                        weights = weights / attention.cfg.max_seq_len
                    weights = attention.attn_dropout(weights)
                    result = result + weights @ vs + weights.sum(-1, keepdim=True) * gv
                    if hv is not None:
                        result = result + (weights[..., None] * hv).sum(-2)
                if k.shape[1] > inherited:
                    result = result + history_read(attention, qc, k[:, inherited:], v[:, inherited:])
                outputs.append(result)
            return torch.cat(outputs, 2)

        return read

    def score(self, model, parent_cache, item_ids, candidates, query_delta, *, trace=False,
              prepared_cache=None, item_embeddings=None, read_features=None, interaction_override=None):
        features = read_features
        if self.h is not None and features is None:
            features = self.prepare_read_features(model, parent_cache, item_ids, item_embeddings=item_embeddings)
        prepared = (self.prepare_cache(model, parent_cache, item_ids,
                                       item_embeddings=item_embeddings, read_features=features)
                    if prepared_cache is None else prepared_cache)
        return native_score(model, prepared, candidates, query_delta, trace=trace,
                            history_read_fn=self.make_reader(model, parent_cache, item_ids, prepared,
                                item_embeddings=item_embeddings, read_features=features,
                                interaction_override=interaction_override))

    def append(self, model, cache, source_cache, source_item_ids, item, behavior, delta,
               *, max_length=1024, trace=False, read_features=None):
        """One causal behavior write through this method's corrected history.

        Returns (cache, surviving_source_cache, surviving_source_ids, trace).
        f's cache is supplied by the caller and is NEVER prepared again.
        Eviction occurs before encoding, exactly as the native rolling append.
        New rows are native outputs of corrected reads and receive no extra
        f/g/h correction until another release.
        """
        if item.shape[-1] != 1:
            raise ValueError("append accepts one causal behavior per call")
        retained = min(cache.seq_len, max_length - 1)
        dropped = cache.seq_len - retained
        cache = HSTUKVCache(cache.k[:, :, -retained:] if retained else cache.k[:, :, :0],
                           cache.v[:, :, -retained:] if retained else cache.v[:, :, :0], retained)
        source_dropped = min(dropped, source_cache.seq_len)
        source_cache = HSTUKVCache(source_cache.k[:, :, source_dropped:],
                                  source_cache.v[:, :, source_dropped:], source_cache.seq_len-source_dropped)
        source_item_ids = source_item_ids[:, source_dropped:]
        if read_features is not None:
            read_features = tuple(rows[:, :, source_dropped:] for rows in read_features)
        result = read_embedded(model, cache, model.embed_inputs(item, behavior, delta), trace=trace,
                               history_read_fn=self.make_reader(model, source_cache, source_item_ids, cache,
                                                                read_features=read_features))
        cache = HSTUKVCache(torch.cat((cache.k, result.new_kv.k), 2),
                           torch.cat((cache.v, result.new_kv.v), 2), retained+1)
        return cache, source_cache, source_item_ids, result

    @torch.no_grad()
    def fit_input_statistics(self, model, samples):
        """Freeze publishing-time scales using fitting users only."""
        samples = list(samples)
        total = torch.zeros_like(self.row_mean)
        squares = torch.zeros_like(total)
        query_squares = torch.zeros_like(self.query_scale)
        row_count = query_count = 0
        for sample in samples:
            sample = sample_to_device(sample, self.row_mean.device)
            source = sample["parent_cache"]
            items = self._items(model, sample["item_ids"], sample.get("item_embeddings"))
            _, trace = native_score(model, source, sample["candidates"], sample["query_delta"], trace=True)
            for layer, q in enumerate(trace.queries):
                rows = self._raw_rows(layer, source, items)
                total[layer] += rows.sum((0, 2))
                squares[layer] += rows.square().sum((0, 2))
                query_squares[layer] += q.square().sum((0, 2))
            row_count += source.k.shape[1] * source.seq_len
            query_count += sample["candidates"].numel()
        if not row_count or not query_count:
            raise ValueError("fitting statistics require nonempty histories and candidates")
        self.row_mean.copy_(total / row_count)
        self.row_scale.copy_((squares / row_count - self.row_mean.square()).clamp_min(1e-8).sqrt())
        self.query_scale.copy_((query_squares / query_count).clamp_min(1e-8).sqrt())
        self.statistics_initialized.fill_(True)

    def forward_flops(self, history_length, queries, *, batch=1):
        """Analytical executed-shape probe arithmetic (multiply-add = 2).

        Backbone query FLOPs and teacher reconstruction are separate driver
        components. Activation/division count as one elementary operation.
        This is an arithmetic model, not measured GPU instruction counts.
        """
        c, n, q, b = self.cfg, int(history_length), int(queries), int(batch)

        def mlp(width):
            widths = [width] + [c.hidden_width] * c.depth + [2*c.head_dim]
            return sum(2*i*o + o for i, o in zip(widths[:-1], widths[1:], strict=True)) + 2*c.depth*c.hidden_width

        lh, d = c.num_layers*c.num_heads, c.head_dim
        row_norm = 2*c.row_dim if c.normalize else 0
        out_norm = 2*d if c.normalize else 0
        f = b*n*lh*(row_norm + mlp(c.row_dim) + out_norm + 2*d)
        g = h = 0
        if self.g is not None:
            # g network; query-key shift, weight sum and weighted value shift.
            g = b*q*lh*((d if c.normalize else 0) + mlp(d) + out_norm + 2*d + n + 2*d)
        if self.h is not None:
            h = (b*n*lh*row_norm + b*q*lh*(d if c.normalize else 0) +
                 b*q*n*lh*(mlp(d+c.row_dim) + out_norm + 4*d))
        return dict(f_prepare_flops=int(f), g_read_extra_flops=int(g),
                    h_read_extra_flops=int(h), read_extra_flops=int(g+h),
                    total_correction_forward_flops=int(f+g+h))


def sample_to_device(sample, device):
    """Only the current user's caches need to reside on the model device."""
    used = {"parent_cache", "teacher_cache", "item_ids", "item_embeddings", "candidates", "query_delta",
            "teacher_scores", "read_scales", "score_scale"}
    return {name: value.to(device) if name in used and isinstance(value, (torch.Tensor, HSTUKVCache)) else value
            for name, value in sample.items()}


def native_query_flops(model, *, history_length, queries, batch=1):
    """Dense legacy reader arithmetic, with actual head geometry and self rule."""
    w, h = model.cfg.hidden_size, model.cfg.num_heads
    d = model.blocks[0].attn.head_dim
    inner, rows = h*d, int(batch)*int(queries)
    nf = model.cfg.temporal_num_freqs
    total = rows*(4*nf + 4*nf*w + 2*w*w + 3*w) + 3*nf
    for block in model.blocks:
        total += rows*(4*w+2 + 6*w*inner + 2*inner*w + 2*w*w + 4*w)
        total += rows*int(history_length)*(4*inner+3*h)
        if block.attn.causal_diagonal == "inclusive":
            total += rows*(4*inner+3*h)
        if block.block_variant == "hstu_reference":
            # Extra layer norm and fixed attention divisor, explicitly shown.
            total += rows*(5*inner+2) + rows*(int(history_length)+1)*h
    return int(total + rows*(4*w+2 + 2*w+1))


SCALE_VERSION = "fit_pool_equal_user_length_normalized_v1"


@torch.no_grad()
def _prepare_target_moments(model, samples):
    """Cache teacher targets and per-user sufficient statistics, never val weights."""
    device = next(model.parameters()).device
    result = []
    for original in samples:
        sample = sample_to_device(original, device)
        if (sample.get("target_moment_version") == SCALE_VERSION and
                all(key in sample for key in ("teacher_scores", "read_moment_sum", "score_moment_sum", "scale_users"))):
            result.append(dict(original))
            continue
        if "teacher_scores" in sample:
            target = sample["teacher_scores"]
        else:
            target, _ = native_score(model, sample["teacher_cache"], sample["candidates"], sample["query_delta"])
        old_scores, trace = native_score(model, sample["parent_cache"], sample["candidates"], sample["query_delta"], trace=True)
        moments = []
        length = max(1, sample["parent_cache"].seq_len)
        for layer, q in enumerate(trace.queries):
            wanted = history_read(model.blocks[layer].attn, q,
                                  sample["teacher_cache"].k[layer], sample["teacher_cache"].v[layer])
            actual = trace.history_heads[layer]
            error = (wanted-actual)/length
            # Each user has equal weight, regardless of history length and
            # candidate count. Within a user average its candidate/dim entries.
            moments.append(error.square().mean((2, 3)).sum(0))
        destination = original["parent_cache"].k.device
        result.append({**original, "teacher_scores": target.detach().to(destination),
                       "read_moment_sum": torch.stack(moments).to(destination),
                       "score_moment_sum": (target-old_scores).square().mean(1).sum().to(destination),
                       "scale_users": sample["candidates"].shape[0],
                       "target_moment_version": SCALE_VERSION})
    return result


def _fit_pool_scales(samples):
    users = sum(int(sample["scale_users"]) for sample in samples)
    if not users:
        raise ValueError("fit-pool scales require nonempty fitting users")
    read = sum(sample["read_moment_sum"].detach().cpu().double() for sample in samples)/users
    score = sum(sample["score_moment_sum"].detach().cpu().double() for sample in samples)/users
    return read.sqrt().clamp_min(1e-4).float(), score.sqrt().clamp_min(1e-4).float(), users


def _apply_fit_scales(samples, read, score):
    return [{**sample,
             "read_scales": read[:, None, :, None, None].to(sample["parent_cache"].k.device),
             "score_scale": score.to(sample["parent_cache"].k.device),
             "loss_scale_version": SCALE_VERSION} for sample in samples]


@torch.no_grad()
def prepare_training_samples(model, samples):
    """Cache targets and bind scales from this FITTING pool, equal weight/user.

    fit_probe always calls bind_fit_scales afterward, so separately prepared
    validation data cannot supply its own loss denominator. Cached legacy
    per-user scales are overwritten; old logits can still be reused.
    """
    prepared = _prepare_target_moments(model, samples)
    read, score, _ = _fit_pool_scales(prepared)
    return _apply_fit_scales(prepared, read, score)


@torch.no_grad()
def bind_fit_scales(model, train_samples, validation_samples):
    """Use ONE fit-only scale on all fitting and validation users."""
    train = _prepare_target_moments(model, list(train_samples))
    validation = _prepare_target_moments(model, list(validation_samples))
    read, score, users = _fit_pool_scales(train)
    metadata = dict(version=SCALE_VERSION, source="fitting_users_only", users=users,
        aggregation="equal-user RMS; within each user mean over candidates and head dimensions",
        history_normalization="divide both historical reads by their aligned history length before loss",
        read_scales=read.tolist(), score_scale=float(score), minimum_scale=1e-4,
        validation_uses_fitting_scales=True)
    return _apply_fit_scales(train, read, score), _apply_fit_scales(validation, read, score), metadata


def probe_loss(model, probe, sample, *, read_weight=1., score_weight=1.):
    """Read alignment at the student's own queries plus final-score matching."""
    if sample.get("loss_scale_version") != SCALE_VERSION:
        raise ValueError("bind fixed fit-pool scales before computing probe loss")
    prediction, student = probe.score(model, sample["parent_cache"], sample["item_ids"],
                                      sample["candidates"], sample["query_delta"], trace=True,
                                      item_embeddings=sample.get("item_embeddings"))
    if "teacher_scores" in sample:
        target = sample["teacher_scores"]
    else:
        with torch.no_grad():
            target, _ = native_score(model, sample["teacher_cache"], sample["candidates"],
                                     sample["query_delta"])
    raw_score_mse = (prediction-target).square().mean()
    score_scale = sample["score_scale"]
    score_mse = raw_score_mse / score_scale.square()
    read_errors = []
    for layer, (q, actual) in enumerate(zip(student.queries, student.history_heads, strict=True)):
        # The rebuilt read uses the SAME query, so this isolates read error
        # rather than mixing it with changes in the teacher's lower layers.
        with torch.no_grad():
            wanted = history_read(model.blocks[layer].attn, q.detach(),
                                  sample["teacher_cache"].k[layer], sample["teacher_cache"].v[layer])
            scale = sample["read_scales"][layer]
        length = max(1, sample["parent_cache"].seq_len)
        read_errors.append(((actual/length - wanted/length)/scale).square().mean())
    read_mse = torch.stack(read_errors).mean()
    return read_weight * read_mse + score_weight * score_mse, {
        "read_mse": read_mse.detach(), "score_mse": score_mse.detach(),
        "raw_score_mse": raw_score_mse.detach()}


def fit_probe(model, probe, train_samples: Iterable[Mapping], validation_samples: Iterable[Mapping], *,
              steps=100, learning_rate=1e-3, read_weight=1., score_weight=1., seed=17,
              patience=30, validation_every=10, weight_decay=1e-4, gradient_clip=5., progress_callback=None):
    """Fit only probe parameters, select on validation, and restore the best.

    A step consumes one sample (one user/history and its candidates). Calling
    again on stage f supplies the equal-extra-training f control. User roles,
    sample construction, three-seed loops and 64/128/256 sweeps live in the
    driver; this routine never samples evaluation users.
    """
    train, validation = list(train_samples), list(validation_samples)
    if not train or not validation:
        raise ValueError("separate nonempty fitting and validation samples are required")
    if steps < 0 or validation_every < 1:
        raise ValueError("steps must be nonnegative and validation_every positive")
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    calls, arithmetic = Counter(), Counter()

    def charge(sample, stage, *, backward=False, targets=False):
        cache = sample["parent_cache"]
        batch, queries = sample["candidates"].shape
        n = cache.seq_len
        native = native_query_flops(model, history_length=n, queries=queries, batch=batch)
        read = batch*queries*n*model.cfg.num_layers*(4*model.cfg.num_heads*model.blocks[0].attn.head_dim+3*model.cfg.num_heads)
        if targets:
            queries_executed = 1 + int("teacher_scores" not in sample)
            arithmetic[stage+"_native_query_flops"] += queries_executed*native
            arithmetic[stage+"_teacher_history_read_flops"] += read
            calls[stage+"_native_queries"] += queries_executed*batch*queries
        else:
            correction = probe.forward_flops(n, queries, batch=batch)["total_correction_forward_flops"]
            arithmetic[stage+"_native_query_flops"] += native
            arithmetic[stage+"_probe_forward_flops"] += correction
            arithmetic[stage+"_same_query_teacher_read_flops"] += read
            calls[stage+"_native_queries"] += batch*queries
            calls[stage+"_probe_rows"] += batch*n
            calls[stage+"_candidate_row_pairs"] += batch*queries*n
            if backward:
                # Same convention as existing calibration ledgers: estimate
                # backward as 2x forward, including the frozen query path.
                arithmetic[stage+"_backward_flops_estimate"] += 2*(native+correction)
                calls[stage+"_backward_calls"] += 1
        calls[stage+"_sample_calls"] += 1

    if probe.cfg.normalize and not bool(probe.statistics_initialized):
        probe.fit_input_statistics(model, train)
        for sample in train:
            batch, queries = sample["candidates"].shape
            arithmetic["statistics_native_query_flops"] += native_query_flops(
                model, history_length=sample["parent_cache"].seq_len, queries=queries, batch=batch)
            calls["statistics_native_queries"] += batch*queries
    for sample in train+validation:
        if sample.get("target_moment_version") != SCALE_VERSION:
            charge(sample, "targets", targets=True)
    train, validation, fit_scales = bind_fit_scales(model, train, validation)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history = []
    training_window = Counter()
    window_steps = 0

    @torch.no_grad()
    def validate(step):
        losses, reads, scores, raw_scores = [], [], [], []
        for sample in validation:
            charge(sample, "validation")
            sample = sample_to_device(sample, device)
            value, parts = probe_loss(model, probe, sample, read_weight=read_weight, score_weight=score_weight)
            losses.append(float(value))
            reads.append(float(parts["read_mse"]))
            scores.append(float(parts["score_mse"]))
            raw_scores.append(float(parts["raw_score_mse"]))
        row = dict(step=step, validation_loss=sum(losses)/len(losses),
                   validation_read_mse=sum(reads)/len(reads), validation_score_mse=sum(scores)/len(scores),
                   validation_raw_score_mse=sum(raw_scores)/len(raw_scores))
        history.append(row)
        return row["validation_loss"]

    best_loss = validate(0)
    best = {name: value.detach().clone() for name, value in probe.state_dict().items()}
    best_step, last_step = 0, 0
    order = torch.randperm(len(train), generator=generator).tolist()
    for step in range(1, steps + 1):
        if (step - 1) % len(train) == 0:
            order = torch.randperm(len(train), generator=generator).tolist()
        sample = train[order[(step - 1) % len(train)]]
        charge(sample, "training", backward=True)
        sample = sample_to_device(sample, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = probe_loss(model, probe, sample, read_weight=read_weight, score_weight=score_weight)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite probe fitting loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), gradient_clip)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite probe fitting gradient")
        optimizer.step()
        training_window.update(training_loss=float(loss.detach()),
            training_read_mse=float(parts["read_mse"]), training_score_mse=float(parts["score_mse"]),
            training_raw_score_mse=float(parts["raw_score_mse"]), gradient_norm=float(norm))
        window_steps += 1
        last_step = step
        if step % validation_every == 0 or step == steps:
            value = validate(step)
            history[-1].update({name: value/window_steps for name,value in training_window.items()})
            training_window.clear()
            window_steps = 0
            if value < best_loss:
                best_loss, best_step = value, step
                best = {name: tensor.detach().clone() for name, tensor in probe.state_dict().items()}
            if progress_callback is not None:
                progress_callback(dict(history[-1], best_step=best_step, best_validation_loss=best_loss))
            if patience and step - best_step >= patience:
                break
    probe.load_state_dict(best)
    return dict(stage=probe.stage, config=asdict(probe.cfg), seed=seed,
                best_step=best_step, completed_steps=last_step, best_validation_loss=best_loss,
                training_users=len(train), validation_users=len(validation), history=history,
                objective=dict(read_weight=read_weight, score_weight=score_weight),
                fit_pool_loss_scales=fit_scales,
                teacher_preparation_users=len(train) + len(validation),
                operation_counts=dict(calls), arithmetic=dict(arithmetic),
                total_calibration_flops_estimate=sum(arithmetic.values()),
                arithmetic_convention="multiply-add=2; elementary operations=1; backward estimated as 2x forward; source/teacher cache reconstruction charged separately")

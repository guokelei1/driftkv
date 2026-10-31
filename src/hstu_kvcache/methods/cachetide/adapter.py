"""Query-dependent reads and small, composable release updates.

The only per-row state is the explicitly sized shared/private code. Fixed native features are recomputed
when codes change; they are never cached. The A/B update is affine in the old
code and these immutable features, while the serving read is nonlinear in Q.
This is a research reference, not a fused or concurrent serving implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hstu_kvcache.methods.cachetide.structure import StructuredKVLayer
from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.methods.cachetide.features import encoder_digest, load_bank


def _flat(state, code_shape=(6, 8)):
    codes = state.codes if isinstance(state, LightweightValueState) else state
    return codes.flatten(-2) if len(code_shape) == 2 and codes.shape[-2:] == code_shape else codes


def _state(flat, heads=6, rank=8, *, code_shape=None):
    shape = (heads, rank) if code_shape is None else code_shape
    return LightweightValueState(flat.reshape(*flat.shape[:-1], *shape), flat.shape[2])


def _rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def _block_apply(rows, matrix, branches, *, inverse=False):
    pieces = []
    for branch in branches:
        section = slice(branch.code_start, branch.code_start + branch.code_width)
        block = matrix[section, section]
        pieces.append(rows[..., section] @ (block if inverse else block.T))
    return torch.cat(pieces, -1)


def item_gate_query(model, item_ids):
    """Transient Current item content, retaining every embedding coordinate.

    The retained model uses hidden_size == heads * head_dim. This lookup adds
    memory traffic but no arithmetic projection or persistent row state.
    """
    if item_ids is None or item_ids.ndim != 2:
        raise ValueError("item gates require this operation's [batch,queries] item IDs")
    values = model.lookup_item_embeddings(item_ids)
    heads = model.cfg.num_heads
    if values.shape[-1] % heads:
        raise ValueError("item gates need [batch,queries] IDs and the retained head geometry")
    return values.reshape(*item_ids.shape, heads, values.shape[-1] // heads).transpose(1, 2)


def operation_gate_query(adapter, model, item_ids):
    return item_gate_query(model, item_ids) if adapter.gate_source == "item" else None


class RelayLayer(StructuredKVLayer):
    """Shared12/private6 layout; frozen native features share its branch layout."""

    def __init__(self, heads, head_dim, rank, mode, xi_width, read_shared=12,
                 gate_normalization="raw", read_private=6, feature_mode="native",
                 feature_bank_path=None, feature_bank_sha256=None, layer_index=0,
                 gate_source="layer", layout="shared12_private6"):
        super().__init__(heads, head_dim, rank, layout)
        self.stored_shared = self.branches[0].code_width
        self.stored_private = self.branches[1].code_width
        self.mode = mode
        self.feature_mode = feature_mode
        self.feature_bank_path = None if feature_bank_path is None else str(feature_bank_path)
        self.feature_bank_sha256 = feature_bank_sha256
        self.layer_index = layer_index
        self.gate_source = gate_source
        self.xi_width = 1 if mode == "A" else xi_width
        if self.xi_width not in (1, 2, 3):
            raise ValueError("the probe uses constant plus at most two native summaries")
        self.key_gate_weight = nn.Parameter(torch.zeros(heads, head_dim, self.accessible_rank))
        self.value_gate_weight = nn.Parameter(torch.zeros_like(self.key_gate_weight))
        # Decoder/gate parameters keep all 18 slots for checkpoint compatibility.
        # Only the selected shared slots and six private slots execute in reads.
        self.register_buffer("read_slots", torch.arange(self.accessible_rank), persistent=False)
        self.read_private = read_private
        self.set_read_shared(read_shared)
        self.set_gate_normalization(gate_normalization)
        self.rotation = nn.ParameterList([
            nn.Parameter(torch.zeros(b.code_width, b.code_width)) for b in self.branches])
        self.native_update = nn.ParameterList([
            nn.Parameter(torch.eye(b.code_width)) for b in self.branches])
        self.item_update_weight = nn.Parameter(torch.zeros(self.item_width,
                                                          self.code_width, self.xi_width))
        self.item_update_bias = nn.Parameter(torch.zeros(self.code_width, self.xi_width))
        self.previous_weight = nn.Parameter(torch.zeros(self.code_width, self.code_width))
        for j, branch in enumerate(self.branches):
            width = 2 * len(branch.heads) * self.head_dim
            self.register_buffer(f"native_basis_{j}", torch.zeros(width, branch.code_width))
        if mode in ("A", "B"):
            self.input_weight.requires_grad_(False)
            self.input_bias.requires_grad_(False)
            self.previous_weight.requires_grad_(False)
        elif mode in ("legacy", "C"):
            self.rotation.requires_grad_(False)
            self.native_update.requires_grad_(False)
            self.item_update_weight.requires_grad_(False)
            self.item_update_bias.requires_grad_(False)
            if mode == "legacy":
                self.previous_weight.requires_grad_(False)
        else:
            raise ValueError(f"unknown relay update {mode}")

    def set_read_shared(self, width):
        self.set_read_width(width, self.read_private)

    def set_read_private(self, width):
        self.set_read_width(self.read_shared, width)

    def set_read_width(self, shared, private):
        if (shared not in (0, 3, 6, 12) or private not in (3, 6)
                or shared > self.stored_shared or private > self.stored_private):
            raise ValueError("read width exceeds the stored shared/private layout")
        self.read_shared, self.read_private = shared, private
        self.accessible_rank = shared + private
        self.read_slots = torch.cat((torch.arange(shared, device=self.access_indices.device),
                                     torch.arange(self.stored_shared, self.stored_shared + private,
                                                  device=self.access_indices.device)))

    def set_gate_normalization(self, mode):
        if mode not in ("raw", "rms"):
            raise ValueError("query gate normalization must be raw or rms")
        self.gate_normalization = mode

    def accessible_codes(self, codes):
        flat = _flat(codes, self.code_shape)
        selected = self.access_indices[:, self.read_slots]
        return flat[..., selected].transpose(1, 2)

    @torch.no_grad()
    def initialize_features(self):
        """Use the originating encoder's directions, independent of later fits."""
        native, item, bias = self.effective_weights()
        for j, weights in enumerate(native):
            basis = weights / weights.square().sum(0, keepdim=True).sqrt().clamp_min(1e-6)
            getattr(self, f"native_basis_{j}").copy_(basis)
        if self.feature_mode == "origin_encoder":
            self.item_update_weight.zero_()
            self.item_update_bias.zero_()
        else:
            self.item_update_weight[:, :, 0].copy_(item)
            self.item_update_bias[:, 0].copy_(bias)

    def fixed_features(self, k, v, item_ids=None):
        eta, xi = [], []
        bank = None
        if self.feature_mode == "origin_encoder":
            if item_ids is None:
                raise ValueError("origin encoder features require raw item IDs")
            bank = load_bank(self.feature_bank_path, self.feature_bank_sha256,
                             device=self.input_weight.device, dtype=self.input_weight.dtype)
            item_ids = torch.as_tensor(item_ids, device=self.input_weight.device, dtype=torch.long)
        for j, branch in enumerate(self.branches):
            raw = self.kv_features(k, v, branch).to(self.input_weight.dtype)
            normalized = _rms(raw) if bank is None or self.xi_width > 1 else None
            if bank is None:
                eta.append(F.silu(normalized @ getattr(self, f"native_basis_{j}")))
            else:
                eta.append(raw @ bank.native_weights[self.layer_index][j])
            half = raw.shape[-1] // 2
            parts = [torch.ones_like(raw[..., :1])]
            if self.xi_width >= 2:
                parts.append(normalized[..., :half].mean(-1, keepdim=True))
            if self.xi_width >= 3:
                parts.append(normalized[..., half:].mean(-1, keepdim=True))
            # Expansion is temporary. Each branch uses one immutable summary.
            summary = torch.cat(parts, -1)
            xi.append(summary.unsqueeze(-2).expand(*summary.shape[:-1],
                                                  branch.code_width, self.xi_width))
        eta = torch.cat(eta, -1)
        if bank is not None:
            eta = F.silu((eta + bank.table[self.layer_index][item_ids])
                         + bank.bias[self.layer_index])
        return eta, torch.cat(xi, -2)

    def rotation_blocks(self):
        rotations = []
        for raw in self.rotation:
            skew = (raw - raw.T) * .5
            eye = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
            rotations.append(torch.linalg.solve(eye + skew, eye - skew))
        return rotations

    def update_matrices(self):
        return torch.block_diag(*self.rotation_blocks()), torch.block_diag(*self.native_update)

    def item_coefficients(self, item_features):
        return torch.einsum("...e,ecr->...cr", item_features.to(self.item_update_weight.dtype),
                            self.item_update_weight) + self.item_update_bias

    def advance(self, k, v, previous, item_features, item_ids=None):
        previous = _flat(previous, self.code_shape).to(self.input_weight.dtype)
        if self.mode == "legacy":
            return _flat(self.encode(k, v, item_features), self.code_shape)
        if self.mode == "C":
            native_weights, item_weight, bias = self.effective_weights()
            native = torch.cat([
                self.kv_features(k, v, branch).to(item_weight.dtype) @ weight
                for branch, weight in zip(self.branches, native_weights, strict=True)], -1)
            raw = native + item_features.to(item_weight.dtype) @ item_weight + bias
            return previous + F.silu(raw + previous @ self.previous_weight.T)
        eta, xi = self.fixed_features(k, v, item_ids)
        native = []
        for branch, a, b in zip(self.branches, self.rotation_blocks(), self.native_update,
                                 strict=True):
            section = slice(branch.code_start, branch.code_start + branch.code_width)
            native.append(previous[..., section] @ a.T + eta[..., section] @ b.T)
        coefficients = self.item_coefficients(item_features)
        item = coefficients[..., 0] if self.xi_width == 1 else (coefficients * xi).sum(-1)
        return torch.cat(native, -1) + item

    def query_gates(self, q):
        gate_q = _rms(q) if self.gate_normalization == "rms" else q
        key = 1 + torch.tanh(gate_q @ self.key_gate_weight[..., self.read_slots].to(q.dtype)[None])
        value = 1 + torch.tanh(gate_q @ self.value_gate_weight[..., self.read_slots].to(q.dtype)[None])
        return key, value

    def read(self, attention, q, k, v, codes, *, valid_mask=None, code_count=None,
             gate_constants=None, gate_q=None):
        if attention.position_bias is not None or attention.block_variant != "legacy" \
                or attention.cfg.activation != "elu_plus1":
            raise ValueError("relay probe uses legacy ELU+1 without position bias")
        if attention.training and getattr(attention.attn_dropout, "p", 0):
            raise ValueError("the calibration probe requires zero attention dropout")
        b, n, _ = k.shape
        keys = k.to(q.dtype).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        values = v.to(q.dtype).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        count = codes.shape[1] if code_count is None else code_count
        scores = q @ keys.transpose(-2, -1)
        if count:
            z = self.accessible_codes(codes[:, :count].to(q.dtype))
            if gate_constants is None and gate_q is None:
                if self.gate_source == "first" and self.layer_index:
                    raise ValueError("an upper-layer first-source read requires the operation's q0")
                if self.gate_source == "item":
                    raise ValueError("an item-source read requires this operation's Current item embedding")
            if gate_q is not None:
                if gate_q.shape != q.shape:
                    raise ValueError("gate item/query rows must match this operation's native query geometry")
                gate_q = gate_q.to(q)
            gk, gv = self.query_gates(q if gate_q is None else gate_q) \
                if gate_constants is None else gate_constants
            if gate_constants is not None:
                if gk.shape[-1] == self.key_decoder.shape[1]:
                    gk, gv = gk[..., self.read_slots], gv[..., self.read_slots]
                gk, gv = (gate.to(q).reshape(1, self.heads, 1, self.accessible_rank)
                          for gate in (gk, gv))
            dk = self.key_decoder[:, self.read_slots].to(q.dtype)
            projected = (q @ dk.transpose(-2, -1)[None]) * gk
            delta = projected @ z.transpose(-2, -1)
            scores = torch.cat((scores[..., :count] + delta, scores[..., count:]), -1)
        weights = attention._activate(scores * attention.scale)
        if valid_mask is not None:
            mask = valid_mask.to(q)
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            weights = weights * mask
        weights = attention.attn_dropout(weights)
        native = weights @ values
        if not count:
            return native
        # Known-zero current-version suffixes need no code arithmetic. A fused
        # implementation can accumulate these two sums in one history scan.
        latent = weights[..., :count] @ z
        return native + (latent * gv) @ self.decoder[:, self.read_slots].to(q.dtype)[None]


@dataclass
class ReleasePrefix:
    """Shared release state; P/Q/D contain no per-user native features."""

    P: torch.Tensor  # [layers,48,48]
    Q: torch.Tensor  # [layers,48,48]
    D: torch.Tensor  # [layers,catalog,48,xi]
    layers: tuple[RelayLayer, ...]
    release: int = 0

    @classmethod
    def identity(cls, adapter, catalog_size):
        reference = adapter.layers[0].input_weight
        eye = torch.eye(adapter.code_width, device=reference.device,
                        dtype=reference.dtype).repeat(adapter.num_layers, 1, 1)
        d = reference.new_zeros(adapter.num_layers, catalog_size,
                                adapter.code_width, adapter.layers[0].xi_width)
        return cls(eye, torch.zeros_like(eye), d, tuple(adapter.layers))

    def _offset(self, cache, item_ids, table_item_ids=None):
        rows = []
        for index, layer in enumerate(self.layers):
            eta, xi = layer.fixed_features(cache.k[index], cache.v[index], item_ids)
            item = self.D[index][item_ids if table_item_ids is None else table_item_ids]
            item_term = item[..., 0] if layer.xi_width == 1 else (item * xi).sum(-1)
            rows.append(_block_apply(eta, self.Q[index], layer.branches) + item_term)
        return torch.stack(rows)

    def materialize(self, cache, anchors, item_ids, *, table_item_ids=None):
        flat = _flat(anchors, self.layers[0].code_shape).to(self.P)
        current = torch.stack([_block_apply(flat[index], self.P[index], layer.branches)
                               for index, layer in enumerate(self.layers)])
        current = current + self._offset(cache, item_ids, table_item_ids)
        return _state(current.to(cache.k.dtype), code_shape=self.layers[0].code_shape)

    def anchorize(self, cache, current, item_ids, *, table_item_ids=None):
        rhs = _flat(current, self.layers[0].code_shape).to(self.P) - self._offset(cache, item_ids, table_item_ids)
        # P is a product of block orthogonal transforms, inverse = transpose.
        anchors = torch.stack([_block_apply(rhs[index], self.P[index], layer.branches,
                                            inverse=True)
                               for index, layer in enumerate(self.layers)])
        return _state(anchors.to(cache.k.dtype), code_shape=self.layers[0].code_shape)

    def zero_anchors(self, cache, item_ids, *, table_item_ids=None):
        zeros = self.P.new_zeros(cache.k.shape[0], cache.k.shape[1], cache.seq_len,
                                self.layers[0].code_width)
        return self.anchorize(cache, _state(zeros, code_shape=self.layers[0].code_shape),
                              item_ids, table_item_ids=table_item_ids)

    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.P, self.Q, self.D))


class UnifiedRelayAdapter(nn.Module):
    kind = "query_relay_shared12_private6_v1"

    def __init__(self, num_layers, heads=6, head_dim=32, rank=8, max_length=1024,
                 mode="B", xi_width=3, read_shared=12, gate_normalization="raw", read_private=6,
                 feature_mode="native", feature_bank_path=None, feature_bank_sha256=None,
                 gate_source="layer", layout="shared12_private6"):
        super().__init__()
        self.num_layers, self.heads, self.head_dim = num_layers, heads, head_dim
        self.rank, self.max_length, self.mode = rank, max_length, mode
        self.layout = layout
        self.response_hidden_width, self.response = 0, None
        self.xi_width = 1 if mode == "A" else xi_width
        self.read_shared = read_shared
        self.read_private = read_private
        self.gate_normalization = gate_normalization
        if gate_source not in ("layer", "first", "item"):
            raise ValueError("query gates use layer, first, or Current item inputs")
        self.gate_source = gate_source
        if feature_mode not in ("native", "origin_encoder"):
            raise ValueError("fixed feature mode must be native or origin_encoder")
        if feature_mode == "origin_encoder" and (mode not in ("A", "B")
                or feature_bank_path is None or feature_bank_sha256 is None):
            raise ValueError("origin_encoder requires A/B and an immutable bank path plus SHA256")
        self.feature_mode = feature_mode
        self.feature_bank_path = None if feature_bank_path is None else str(feature_bank_path)
        self.feature_bank_sha256 = feature_bank_sha256
        self.layers = nn.ModuleList([
            RelayLayer(heads, head_dim, rank, mode, self.xi_width, read_shared,
                       gate_normalization, read_private, feature_mode,
                       self.feature_bank_path, feature_bank_sha256, index, gate_source, layout)
            for index in range(num_layers)])
        self.accessible_rank = self.layers[0].accessible_rank
        self.code_width = self.layers[0].code_width
        self.code_shape = self.layers[0].code_shape

    def get_config(self):
        return dict(num_layers=self.num_layers, heads=self.heads, head_dim=self.head_dim,
                    rank=self.rank, max_length=self.max_length, mode=self.mode,
                    xi_width=self.xi_width, read_shared=self.read_shared,
                    gate_normalization=self.gate_normalization, read_private=self.read_private,
                    feature_mode=self.feature_mode, feature_bank_path=self.feature_bank_path,
                    feature_bank_sha256=self.feature_bank_sha256, gate_source=self.gate_source,
                    layout=self.layout)

    def set_read_shared(self, width):
        self.set_read_width(width, self.read_private)

    def set_read_private(self, width):
        self.set_read_width(self.read_shared, width)

    def set_read_width(self, shared, private):
        for layer in self.layers:
            layer.set_read_width(shared, private)
        self.read_shared, self.read_private = shared, private
        self.accessible_rank = self.layers[0].accessible_rank

    def set_gate_normalization(self, mode):
        for layer in self.layers:
            layer.set_gate_normalization(mode)
        self.gate_normalization = mode

    def set_gate_source(self, source):
        if source not in ("layer", "first", "item"):
            raise ValueError("query gates use layer, first, or Current item inputs")
        for layer in self.layers:
            layer.gate_source = source
        self.gate_source = source

    @classmethod
    @torch.no_grad()
    def from_structured(cls, source, *, mode="B", xi_width=3, read_shared=12,
                        gate_normalization="raw", read_private=6, feature_mode="native",
                        feature_bank_path=None, feature_bank_sha256=None, gate_source="layer"):
        if source.layout not in ("shared12_private6", "shared6_private3"):
            raise ValueError("relay warmstart requires an explicit shared/private adapter")
        reference = next(source.parameters())
        result = cls(source.num_layers, source.heads, source.head_dim, source.rank,
                     source.max_length, mode=mode, xi_width=xi_width,
                     read_shared=read_shared, gate_normalization=gate_normalization,
                     read_private=read_private, feature_mode=feature_mode,
                     feature_bank_path=feature_bank_path,
                     feature_bank_sha256=feature_bank_sha256, gate_source=gate_source,
                     layout=source.layout).to(reference)
        if feature_mode == "origin_encoder":
            bank = load_bank(feature_bank_path, feature_bank_sha256,
                             device="cpu", dtype=reference.dtype)
            if bank.metadata["encoder_sha256"] != encoder_digest(source):
                raise ValueError("origin encoder bank and warmstart encoder differ")
        result.load_state_dict(source.state_dict(), strict=False)
        for layer in result.layers:
            layer.initialize_features()
        return result

    def encode_initial(self, cache, item_features, item_ids=None):
        zeros = cache.k.new_zeros(self.num_layers, cache.k.shape[1], cache.seq_len,
                                  *self.code_shape)
        return self.advance_codes(cache, zeros, item_features, item_ids)

    def encode_cache(self, cache, item_features, item_ids=None):
        return self.encode_initial(cache, item_features, item_ids)

    def advance_codes(self, cache, previous, item_features, item_ids=None):
        codes = previous.codes if isinstance(previous, LightweightValueState) else previous
        rows = [layer.advance(cache.k[index], cache.v[index], codes[index], item_features, item_ids)
                for index, layer in enumerate(self.layers)]
        return _state(torch.stack(rows).to(cache.k.dtype), code_shape=self.code_shape)

    def make_history_read(self, state, *, valid_mask=None, gate_constants=None, code_count=None,
                          fitted_layers=None, gate_query=None):
        if self.gate_source == "item" and gate_query is None and gate_constants is None:
            raise ValueError("item-source reads require the operation's Current item embedding")
        stop = self.num_layers if fitted_layers is None else fitted_layers
        first_query = None

        def read(index, attention, q, k, v):
            nonlocal first_query
            if self.gate_source == "first" and index == 0:
                # This tensor predates the operation's first history read.
                # Preserve each candidate/event row and its gradient graph.
                first_query = q
            if index >= stop:
                return history_read(attention, q, k, v)
            if k.shape[1] < state.seq_len:
                raise ValueError("relay code prefix cannot exceed the native history")
            constants = None if gate_constants is None else gate_constants[index]
            return self.layers[index].read(attention, q, k, v, state.codes[index],
                                           valid_mask=valid_mask, gate_constants=constants,
                                           code_count=code_count,
                                           gate_q=first_query if self.gate_source == "first" else gate_query)
        return read

    @torch.no_grad()
    def compile_release(self, prefix, catalog_features):
        if self.mode not in ("A", "B"):
            raise ValueError("only the affine A/B update can be compiled")
        if prefix is None:
            prefix = ReleasePrefix.identity(self, len(catalog_features))
        p, q, d = [], [], []
        for index, layer in enumerate(self.layers):
            if layer.xi_width != prefix.layers[index].xi_width:
                raise ValueError("composition requires the same frozen native summaries")
            if layer.feature_mode != prefix.layers[index].feature_mode:
                raise ValueError("composition requires the same fixed feature mode")
            if layer.feature_mode == "origin_encoder":
                if layer.feature_bank_sha256 != prefix.layers[index].feature_bank_sha256:
                    raise ValueError("composition requires the same immutable origin encoder bank")
            else:
                for j in range(len(layer.branches)):
                    if not torch.equal(getattr(layer, f"native_basis_{j}"),
                                       getattr(prefix.layers[index], f"native_basis_{j}")):
                        raise ValueError("composition requires the same frozen native features")
            pblocks, qblocks, dblocks = [], [], []
            for branch, a, b in zip(layer.branches, layer.rotation_blocks(),
                                     layer.native_update, strict=True):
                section = slice(branch.code_start, branch.code_start + branch.code_width)
                pblocks.append(a @ prefix.P[index, section, section])
                qblocks.append(a @ prefix.Q[index, section, section] + b)
                dblocks.append(torch.einsum("cd,mdr->mcr", a,
                                            prefix.D[index, :, section]))
            p.append(torch.block_diag(*pblocks))
            q.append(torch.block_diag(*qblocks))
            d.append(torch.cat(dblocks, -2) + layer.item_coefficients(catalog_features))
        return ReleasePrefix(torch.stack(p), torch.stack(q), torch.stack(d),
                             tuple(self.layers), prefix.release + 1)

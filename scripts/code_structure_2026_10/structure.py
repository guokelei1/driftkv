"""Shared/private code layouts with the same 48 persistent floats per row.

Each layout contains the original six independent rank8 encoders exactly.
Weights operate on native K/V and the complete Current item embedding after
the original fixed normalization is folded. Branches avoid computing forbidden
zero connections. Only the read temporarily gathers each head's accessible
coordinates; the stored state remains [layer, owner, row, 6, 8].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hstu_kvcache.design_one.lightweight import LightweightValueState


@dataclass(frozen=True)
class Branch:
    heads: tuple[int, ...]
    code_start: int
    code_width: int


def layout_spec(layout, heads=6, rank=8):
    if (heads, rank) != (6, 8):
        raise ValueError("this experiment uses six native heads and 48 code floats")
    if layout == "group2":
        branches = [Branch((h, h+1), h*rank, 2*rank) for h in range(0, heads, 2)]
        owners = [(h, r) for h in range(heads) for r in range(rank)]
        access = [list(range((h//2)*16, (h//2+1)*16)) for h in range(heads)]
    elif layout in ("shared6_private7", "shared12_private6"):
        shared = 6 if layout == "shared6_private7" else 12
        private = rank-shared//heads
        branches = [Branch(tuple(range(heads)), 0, shared)]
        branches += [Branch((h,), shared+h*private, private) for h in range(heads)]
        owners = [(h, r) for h in range(heads) for r in range(private, rank)]
        owners += [(h, r) for h in range(heads) for r in range(private)]
        access = [list(range(shared))+list(range(shared+h*private, shared+(h+1)*private))
                  for h in range(heads)]
    else:
        raise ValueError(f"unknown code layout: {layout}")
    return branches, owners, access


class StructuredKVLayer(nn.Module):
    def __init__(self, heads, head_dim, rank, layout):
        super().__init__()
        self.heads, self.head_dim, self.rank = heads, head_dim, rank
        self.layout = layout
        self.branches, self.owners, access = layout_spec(layout, heads, rank)
        self.code_width = heads*rank
        self.accessible_rank = len(access[0])
        self.item_width = heads*head_dim
        self.weight_sizes = [(2*len(b.heads)*head_dim+self.item_width)*b.code_width
                             for b in self.branches]
        self.input_weight = nn.Parameter(torch.zeros(sum(self.weight_sizes)))
        self.input_bias = nn.Parameter(torch.zeros(self.code_width))
        self.decoder = nn.Parameter(torch.zeros(heads, self.accessible_rank, head_dim))
        self.key_decoder = nn.Parameter(torch.zeros_like(self.decoder))
        self.register_buffer("access_indices", torch.tensor(access, dtype=torch.long))
        self.register_buffer("weight_coordinate_scale", torch.ones_like(self.input_weight))
        self.register_buffer("weight_coordinate_center", torch.zeros_like(self.input_weight))
        self.register_buffer("bias_coordinate_scale", torch.ones_like(self.input_bias))
        self.register_buffer("value_coordinate_scale", torch.ones(heads, 1, 1))
        self.register_buffer("key_coordinate_scale", torch.ones(heads, 1, 1))

    def weight_blocks(self, weight=None):
        flat = self.input_weight if weight is None else weight
        return [piece.reshape(2*len(branch.heads)*self.head_dim+self.item_width,
                              branch.code_width)
                for piece, branch in zip(flat.split(self.weight_sizes), self.branches, strict=True)]

    def effective_weights(self, weight=None, bias=None):
        blocks = self.weight_blocks(weight)
        kv = [block[:2*len(branch.heads)*self.head_dim]
              for block, branch in zip(blocks, self.branches, strict=True)]
        item = torch.cat([block[2*len(branch.heads)*self.head_dim:]
                          for block, branch in zip(blocks, self.branches, strict=True)], dim=-1)
        return kv, item, self.input_bias if bias is None else bias

    def kv_features(self, k, v, branch):
        shape = (*k.shape[:-1], self.heads, self.head_dim)
        keys, values = k.reshape(shape), v.reshape(shape)
        # Serving uses all selected Keys followed by all selected Values.
        return torch.cat((keys[..., list(branch.heads), :].flatten(-2),
                          values[..., list(branch.heads), :].flatten(-2)), dim=-1)

    def encode(self, k, v, item_features, weight=None, bias=None):
        kv_weights, item_weight, effective_bias = self.effective_weights(weight, bias)
        native = torch.cat([self.kv_features(k, v, branch).to(item_weight.dtype) @ w
                            for branch, w in zip(self.branches, kv_weights, strict=True)], dim=-1)
        projected = item_features.to(item_weight.dtype) @ item_weight
        result = F.silu(native+projected+effective_bias)
        return result.reshape(*k.shape[:-1], self.heads, self.rank).to(k.dtype)

    def accessible_codes(self, codes):
        flat = codes.flatten(-2)
        return flat[..., self.access_indices].transpose(1, 2)

    def read(self, attention, q, k, v, codes, decoder=None, key_decoder=None):
        if (attention.position_bias is not None or attention.training
                or attention.block_variant != "legacy" or attention.cfg.activation != "elu_plus1"):
            raise ValueError("structured reads require frozen legacy ELU+1 attention")
        b, n, _ = k.shape
        keys = k.to(q.dtype).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        values = v.to(q.dtype).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        z = self.accessible_codes(codes.to(q.dtype))
        dk = self.key_decoder if key_decoder is None else key_decoder
        dv = self.decoder if decoder is None else decoder
        projected = q @ dk.to(q.dtype).transpose(-2, -1)[None]
        scores = q @ keys.transpose(-2, -1)+projected @ z.transpose(-2, -1)
        weights = attention.attn_dropout(attention._activate(scores*attention.scale))
        aggregate = weights @ torch.cat((values, z), dim=-1)
        native, latent = aggregate.split((self.head_dim, self.accessible_rank), dim=-1)
        return native+latent @ dv.to(q.dtype)[None]

    @torch.no_grad()
    def embed(self, original):
        d = self.head_dim
        effective = original.input_weight/original.input_scale[..., None]
        bias = original.input_bias-(original.input_center[..., None]*effective).sum(-2)
        origin_weight_scale = original.input_weight.square().mean((-2, -1)).sqrt().clamp_min(1e-6)
        origin_bias_rms = original.input_bias.square().mean(-1).sqrt()
        origin_bias_scale = torch.where(origin_bias_rms > 0, origin_bias_rms, origin_weight_scale)
        blocks, scale_blocks = self.weight_blocks(), self.weight_blocks(self.weight_coordinate_scale)
        center_blocks = self.weight_blocks(self.weight_coordinate_center)
        self.input_weight.zero_()
        for branch, block, scales, centers in zip(self.branches, blocks, scale_blocks, center_blocks, strict=True):
            native_width = 2*len(branch.heads)*d
            for column in range(branch.code_width):
                global_index = branch.code_start+column
                owner, coordinate = self.owners[global_index]
                slot = branch.heads.index(owner)
                value_start = len(branch.heads)*d+slot*d
                block[slot*d:(slot+1)*d, column].copy_(effective[owner, :d, coordinate])
                block[value_start:value_start+d, column].copy_(effective[owner, d:2*d, coordinate])
                block[native_width:, column].copy_(effective[owner, 2*d:, coordinate])
                self.input_bias[global_index].copy_(bias[owner, coordinate])
                self.bias_coordinate_scale[global_index].copy_(origin_bias_scale[owner])
                # A unit normalized-weight coordinate is divided by the source
                # input scale, including the newly opened cross-head entries.
                for source_slot, source_head in enumerate(branch.heads):
                    source_value_start = len(branch.heads)*d+source_slot*d
                    scales[source_slot*d:(source_slot+1)*d, column].copy_(
                        origin_weight_scale[owner]/original.input_scale[source_head, :d])
                    scales[source_value_start:source_value_start+d, column].copy_(
                        origin_weight_scale[owner]/original.input_scale[source_head, d:2*d])
                    centers[source_slot*d:(source_slot+1)*d, column].copy_(original.input_center[source_head, :d])
                    centers[source_value_start:source_value_start+d, column].copy_(original.input_center[source_head, d:2*d])
                scales[native_width:, column].copy_(
                    origin_weight_scale[owner]/original.input_scale[owner, 2*d:])
                centers[native_width:, column].copy_(original.input_center[owner, 2*d:])
        self.decoder.zero_(); self.key_decoder.zero_()
        for h in range(self.heads):
            for slot, global_index in enumerate(self.access_indices[h].tolist()):
                owner, coordinate = self.owners[global_index]
                if owner == h:
                    self.decoder[h, slot].copy_(original.decoder[h, coordinate])
                    self.key_decoder[h, slot].copy_(original.key_decoder[h, coordinate])
        # Preserve source decoder scales; zero padding must not shrink them.
        self.value_coordinate_scale.copy_(original.decoder.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6))
        self.key_coordinate_scale.copy_(original.key_decoder.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6))


class StructuredKVAdapter(nn.Module):
    kind = "design_one_structured_full_item_kv_v1"

    def __init__(self, num_layers, heads, head_dim, rank=8, response_hidden_width=0,
                 max_length=1024, layout="group2"):
        super().__init__()
        if response_hidden_width:
            raise ValueError("structured codes use no Response module")
        self.num_layers, self.heads, self.head_dim = num_layers, heads, head_dim
        self.rank, self.response_hidden_width, self.max_length = rank, 0, max_length
        self.layout = layout
        self.layers = nn.ModuleList([StructuredKVLayer(heads, head_dim, rank, layout)
                                     for _ in range(num_layers)])
        self.response = None
        self.accessible_rank = self.layers[0].accessible_rank

    def get_config(self):
        return dict(num_layers=self.num_layers, heads=self.heads, head_dim=self.head_dim,
                    rank=self.rank, response_hidden_width=0, max_length=self.max_length, layout=self.layout)

    @classmethod
    @torch.no_grad()
    def from_adapter(cls, original, layout):
        reference = next(original.parameters())
        result = cls(**original.get_config(), layout=layout).to(device=reference.device, dtype=reference.dtype)
        for target, source in zip(result.layers, original.layers, strict=True):
            target.embed(source)
        return result

    def encode_cache(self, cache, item_features):
        codes = [layer.encode(cache.k[i], cache.v[i], item_features)
                 for i, layer in enumerate(self.layers)]
        return LightweightValueState(torch.stack(codes), cache.seq_len)

    def make_history_read(self, state, counts=None, fitted_layers=None, strengths=None):
        if strengths is not None:
            raise ValueError("release strengths must already be folded")
        prefix = self.num_layers if fitted_layers is None else fitted_layers
        def read(index, attention, q, k, v):
            if index >= prefix:
                from hstu_kvcache.adaptation.reader import history_read
                return history_read(attention, q, k, v)
            return self.layers[index].read(attention, q, k, v, state.codes[index])
        return read

    def effective_weights(self):
        """Serving split; KV branches are independent, Current item is shared.

        kv_weights[j]: [layers, 2*len(branch.heads)*head_dim, branch.code_width]
        item_weight: [layers, heads*head_dim, heads*rank]
        bias: [layers, heads*rank]. No normalization remains to fold.
        """
        rows = [layer.effective_weights() for layer in self.layers]
        return dict(branches=[dict(heads=list(b.heads), code_start=b.code_start, code_width=b.code_width)
                              for b in self.layers[0].branches],
                    kv_weights=[torch.stack([row[0][j] for row in rows])
                                for j in range(len(self.layers[0].branches))],
                    item_weight=torch.stack([row[1] for row in rows]),
                    bias=torch.stack([row[2] for row in rows]))

    def row_kv_flops(self):
        branches = self.layers[0].branches
        return self.num_layers*(sum(4*len(b.heads)*self.head_dim*b.code_width for b in branches)
                                +2*self.heads*self.rank)

    def estimate_flops(self, *, batch=1, tokens=1, candidates=1):
        h, d, r, l = self.heads, self.head_dim, self.accessible_rank, self.num_layers
        # Actual encode: branch KV products, one complete-item product, then
        # KV+item+bias and SiLU. A served precomputed item term includes bias.
        encode = self.row_kv_flops()+l*(2*(h*d)*(h*self.rank)+h*self.rank)
        read = l*batch*candidates*(4*h*d*r+4*h*tokens*r+h*tokens+h*d)
        return dict(token_transform=int(batch*tokens*encode),
                    extra_history_read=int(read), candidate_reads=0)

    def packing_bytes(self, *, batch=1, tokens=1, element_size=4):
        per_layer = batch*tokens*self.heads*(self.head_dim+self.accessible_rank)*element_size
        return dict(total_written=self.num_layers*per_layer, peak_scratch=per_layer)


class _DecoderCoordinates(nn.Module):
    def __init__(self, origin, scale):
        super().__init__()
        self.register_buffer("origin", origin)
        self.register_buffer("scale", scale)
        self.delta = nn.Parameter(torch.zeros_like(origin))

    def decoders(self):
        return self.origin+self.scale*self.delta


class StructuredKVRefinement(nn.Module):
    """Offline coordinates only; all selected changes fold into the adapter."""
    def __init__(self, adapter):
        super().__init__()
        for name, values in (
            ("weight_origin", [x.input_weight for x in adapter.layers]),
            ("bias_origin", [x.input_bias for x in adapter.layers]),
            ("weight_scale", [x.weight_coordinate_scale for x in adapter.layers]),
            ("weight_center", [x.weight_coordinate_center for x in adapter.layers]),
            ("bias_scale", [x.bias_coordinate_scale for x in adapter.layers]),
            ("key_origin", [x.key_decoder for x in adapter.layers]),
            ("key_scale", [x.key_coordinate_scale for x in adapter.layers])):
            self.register_buffer(name, torch.stack([v.detach().clone() for v in values]))
        self.weight_delta = nn.Parameter(torch.zeros_like(self.weight_origin))
        self.bias_delta = nn.Parameter(torch.zeros_like(self.bias_origin))
        self.key_delta = nn.Parameter(torch.zeros_like(self.key_origin))
        self.decoder = _DecoderCoordinates(torch.stack([x.decoder.detach().clone() for x in adapter.layers]),
                                          torch.stack([x.value_coordinate_scale.detach().clone() for x in adapter.layers]))
        self.weight_sizes = adapter.layers[0].weight_sizes
        self.branches = adapter.layers[0].branches
        self.head_dim = adapter.head_dim
        self.item_width = adapter.heads*adapter.head_dim
        self.coordinate_centering_forward_flops = 2*self.weight_delta.numel()+self.bias_delta.numel()

    def encoders(self):
        increment = self.weight_scale*self.weight_delta
        centered = increment*self.weight_center
        shifts = [piece.reshape(self.weight_delta.shape[0],
                                2*len(branch.heads)*self.head_dim+self.item_width,
                                branch.code_width).sum(1)
                  for piece, branch in zip(centered.split(self.weight_sizes, dim=-1), self.branches, strict=True)]
        # Folded weights serve uncentered inputs; this bias term makes their
        # trainable increments operate on the original centered coordinates.
        return (self.weight_origin+increment,
                self.bias_origin+self.bias_scale*self.bias_delta-torch.cat(shifts, dim=-1))

    def key_decoders(self):
        return self.key_origin+self.key_scale*self.key_delta

    def encode_cache(self, adapter, cache, item_features):
        weights, biases = self.encoders()
        codes = [layer.encode(cache.k[i], cache.v[i], item_features, weights[i], biases[i])
                 for i, layer in enumerate(adapter.layers)]
        return LightweightValueState(torch.stack(codes), cache.seq_len)

    def make_history_read(self, adapter, state):
        values, keys = self.decoder.decoders(), self.key_decoders()
        def read(index, attention, q, k, v):
            return adapter.layers[index].read(attention, q, k, v, state.codes[index],
                                              decoder=values[index], key_decoder=keys[index])
        return read

    def identity_penalty(self):
        parameters = tuple(self.parameters())
        return sum(p.square().sum() for p in parameters)/sum(p.numel() for p in parameters)

    def coordinate_rms(self):
        return self.identity_penalty().sqrt()

    @torch.no_grad()
    def fold_into(self, adapter):
        weights, biases = self.encoders()
        values, keys = self.decoder.decoders(), self.key_decoders()
        for i, layer in enumerate(adapter.layers):
            layer.input_weight.copy_(weights[i]); layer.input_bias.copy_(biases[i])
            layer.decoder.copy_(values[i]); layer.key_decoder.copy_(keys[i])


# Short name used by the calibration entry.
StructuredRefinement = StructuredKVRefinement


@torch.no_grad()
def expansion_diagnostics(adapter):
    """Measure whether fitting used connections absent from the rank8 origin.

    'Owner' labels refer only to the original embedding of the baseline code,
    not a constraint or semantic assignment after training. Decoder matrix rank
    and nonzero cross weights describe expression use, not ranking quality.
    """
    layers = []
    for index, layer in enumerate(adapter.layers):
        cross_encoder_sq = 0.
        cross_encoder_count = 0
        for branch, block in zip(layer.branches, layer.weight_blocks(), strict=True):
            for column in range(branch.code_width):
                owner, _ = layer.owners[branch.code_start+column]
                for slot, source in enumerate(branch.heads):
                    if source == owner:
                        continue
                    value_start = len(branch.heads)*layer.head_dim+slot*layer.head_dim
                    values = torch.cat((block[slot*layer.head_dim:(slot+1)*layer.head_dim, column],
                                        block[value_start:value_start+layer.head_dim, column]))
                    cross_encoder_sq += float(values.square().sum())
                    cross_encoder_count += values.numel()
        heads = []
        cross_decoder_sq = 0.
        for head in range(layer.heads):
            matrix = torch.cat((layer.key_decoder[head], layer.decoder[head]), dim=-1)
            singular = torch.linalg.svdvals(matrix)
            rank = int(torch.linalg.matrix_rank(matrix))
            slots = [slot for slot, coordinate in enumerate(layer.access_indices[head].tolist())
                     if layer.owners[coordinate][0] != head]
            if slots:
                cross_decoder_sq += float(matrix[slots].square().sum())
            heads.append(dict(head=head, combined_key_value_decoder_rank=rank,
                              singular_values=singular.cpu().tolist()))
        all_decoder_sq = float(layer.decoder.square().sum()+layer.key_decoder.square().sum())
        all_encoder_sq = float(layer.input_weight.square().sum())
        layers.append(dict(layer=index, accessible_rank=layer.accessible_rank,
                           encoder_cross_head_connections=cross_encoder_count,
                           encoder_cross_head_l2=cross_encoder_sq**.5,
                           encoder_cross_head_fraction_l2=(cross_encoder_sq/max(all_encoder_sq, 1e-30))**.5,
                           decoder_off_origin_head_l2=cross_decoder_sq**.5,
                           decoder_off_origin_head_fraction_l2=(cross_decoder_sq/max(all_decoder_sq, 1e-30))**.5,
                           heads=heads))
    return dict(layout=adapter.layout, stored_floats_per_layer_row=adapter.heads*adapter.rank,
                baseline_per_head_combined_decoder_rank_upper_bound=8, layers=layers,
                interpretation="Connection/rank diagnostics only; quality requires the fixed paired panel.")

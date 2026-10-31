"""Explicit inference configuration over the frozen query-relay semantics.

The reference adapter and synchronous RelaySession remain unchanged. This
view changes only operation preparation and the implementation of history
reads. Training must continue to use the original differentiable adapter.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ImplementationConfig:
    reader_backend: str = "triton"
    pack_item_gates: bool = True
    row_chunk: int = 128

    def __post_init__(self):
        if self.reader_backend not in ("torch", "triton"):
            raise ValueError("choose the torch or triton backend explicitly")
        if self.row_chunk < 1:
            raise ValueError("row_chunk must be positive")


class InferenceAdapter:
    """Read-only view; no parameter updates, implicit backend fallback or router."""

    @torch.no_grad()
    def __init__(self, adapter, config=None):
        self.original = adapter
        self.config = config or ImplementationConfig()
        if adapter.training or any(p.requires_grad for p in adapter.parameters()):
            raise ValueError("freeze and eval the original adapter before configuring inference")
        if (adapter.heads not in (6, 10) or adapter.layout != "shared12_private6"
                or (adapter.mode, adapter.gate_source, adapter.gate_normalization,
                    adapter.head_dim, adapter.read_shared,
                    adapter.read_private) != ("A", "item", "rms", 32, 3, 3)):
            raise ValueError("implementation targets A/item/RMS/H6-or-H10/D32/read3+3")
        if not adapter.layers:
            raise ValueError("adapter needs a layer")
        self.counters = Counter()
        # Publish-sized immutable derived weights. Each layer retains its own
        # K/V gates; only the common operation input and GEMM are shared.
        rows = []
        for layer in adapter.layers:
            rows.append(torch.cat((layer.key_gate_weight[..., layer.read_slots],
                                   layer.value_gate_weight[..., layer.read_slots]), -1))
        packed = torch.stack(rows).permute(1, 2, 0, 3).contiguous()
        self.gate_weights = packed.flatten(-2).detach()
        self.decoders = tuple((layer.key_decoder[:, layer.read_slots].contiguous().detach(),
                               layer.decoder[:, layer.read_slots].contiguous().detach())
                              for layer in adapter.layers)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def describe(self):
        return dict(configuration=asdict(self.config), reference=self.original.get_config(),
                    extra_persistent_row_bytes=0,
                    derived_gate_weight_bytes=self.gate_weights.numel()*self.gate_weights.element_size(),
                    derived_decoder_bytes=sum(x.numel()*x.element_size()
                        for pair in self.decoders for x in pair),
                    gates="one operation RMS and packed all-layer K/V projection; independent per candidate",
                    training="original differentiable adapter; this view is inference-only",
                    concurrency="original synchronous session; no concurrent shared-model hook guarantee",
                    implicit_fallback=False)

    def prepare_gates(self, gate_query):
        if torch.is_grad_enabled():
            raise ValueError("configured inference requires torch.no_grad()")
        if gate_query.ndim != 4 or gate_query.shape[1] != self.heads or gate_query.shape[-1] != self.head_dim:
            raise ValueError("item query must have [batch,heads,queries,head_dim] geometry")
        if gate_query.dtype != torch.float32 or gate_query.device != self.gate_weights.device:
            raise ValueError("packed item gates require matching FP32 operation and published weights")
        # RMS is candidate/head-local: no mixing of candidate rows, timestamps,
        # users or future events takes place in this shared operation.
        normalized = gate_query * torch.rsqrt(gate_query.square().mean(-1, keepdim=True) + 1e-6)
        gates = 1 + torch.tanh(normalized @ self.gate_weights[None])
        b, h, q, _ = gates.shape
        gates = gates.reshape(b, h, q, self.num_layers, 2, self.accessible_rank)
        self.counters["packed_gate_operations"] += 1
        self.counters["packed_gate_query_rows"] += b*q
        self.counters["rms_layer_recomputations_avoided"] += self.num_layers-1
        return tuple((gates[..., index, 0, :], gates[..., index, 1, :])
                     for index in range(self.num_layers))

    def make_history_read(self, state, *, valid_mask=None, gate_constants=None,
                          code_count=None, fitted_layers=None, gate_query=None):
        if torch.is_grad_enabled():
            raise ValueError("configured inference requires torch.no_grad()")
        if fitted_layers is not None and fitted_layers != self.num_layers:
            raise ValueError("partial-layer calibration uses the original differentiable adapter")
        if gate_query is None and gate_constants is None:
            raise ValueError("item-source reads require this operation's Current item embedding")
        count = state.seq_len if code_count is None else code_count
        prepared = None
        if self.config.pack_item_gates and count and gate_constants is None:
            prepared = self.prepare_gates(gate_query)

        def read(index, attention, q, k, v):
            from hstu_kvcache.methods.cachetide.read import fused_relay_read
            if k.shape[1] < state.seq_len:
                raise ValueError("code buffer cannot exceed native history")
            out = fused_relay_read(self.layers[index], attention, q, k, v,
                state.codes[index], valid_mask=valid_mask, code_count=code_count,
                gate_constants=None if gate_constants is None else gate_constants[index],
                gate_q=gate_query,
                prepared_gates=None if prepared is None else prepared[index],
                prepared_decoders=self.decoders[index],
                backend=self.config.reader_backend)
            self.counters[self.config.reader_backend+"_history_calls"] += 1
            self.counters["coded_history_calls" if count else "native_only_history_calls"] += 1
            return out

        return read


def configure_prefix(prefix, config=None, *, feature_fn=None):
    """Bind a shared prefix to bounded-scratch in-place user conversions."""
    from hstu_kvcache.methods.cachetide.state import InplaceReleasePrefix
    config = config or ImplementationConfig()
    return InplaceReleasePrefix.wrap(prefix, row_chunk=config.row_chunk, feature_fn=feature_fn)

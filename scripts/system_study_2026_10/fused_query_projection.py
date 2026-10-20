"""A serving adapter that computes per-layer query projections in the scan."""
import torch

from implementation_2026_10.backend import InferenceAdapter
from implementation_2026_10.reader import _validate, _mask_geometry
from implementation_2026_10.reader_kernel import launch_read as native_read
from system_study_2026_10.fused_query_projection_kernel import launch_read


class QueryProjectionAdapter(InferenceAdapter):
    def make_history_read(self, state, *, valid_mask=None, gate_constants=None,
                          code_count=None, fitted_layers=None, gate_query=None):
        if torch.is_grad_enabled() or gate_constants is not None or fitted_layers is not None:
            raise ValueError("this prototype uses the frozen full-layer item-gated inference path")
        count = state.seq_len if code_count is None else code_count
        gates = self.prepare_gates(gate_query) if count else None

        def read(index, attention, q, k, v):
            codes, c = _validate(self.layers[index], attention, q, k, v, state.codes[index], count)
            b, h, m, _ = q.shape
            mask, strides = _mask_geometry(valid_mask, b, h, m, k.shape[1], q.device)
            self.counters["triton_history_calls"] += 1
            self.counters["coded_history_calls" if c else "native_only_history_calls"] += 1
            if k.shape[1] == 0 or m == 0:
                return torch.zeros_like(q)
            if not c:
                return native_read(q, k, v, codes, q, q, q, mask, strides,
                                   code_count=0, scale=float(attention.scale))
            gk, gv = gates[index]
            dk, dv = self.decoders[index]
            return launch_read(q, k, v, codes, gk, gv, dk, dv, mask, strides,
                               code_count=c, scale=float(attention.scale))
        return read

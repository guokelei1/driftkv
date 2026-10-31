#!/usr/bin/env python3
"""Use existing causal row support to gate the retained rank8 correction.

The native writes, fitted weights and attention traversal are unchanged. Codes
are multiplied by the existing layer/row flags during initialization/appends.
Verification recomputes the same masked state, rather than an unmasked adapter.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import torch

from experiments.archive.compact_read import evaluate_lightweight_fullitem_kv_dispatch as dispatch
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.common.artifacts import sha256

base = dispatch.kv.base
original_score_user = base.score_user
original_candidate_release = dispatch.kv.candidate_release
original_sources = dispatch.execution_sources
verify_masked_state = False


def mask_codes(state, flags):
    if flags.shape != (state.codes.shape[0], state.seq_len):
        raise RuntimeError('causal flags and encoded rows differ')
    mask = torch.as_tensor(flags.copy(), device=state.codes.device)[:, None, :, None, None]
    state.codes.mul_(mask)
    return state


class SupportMaskedSession(dispatch.kv.LazyKVSession):
    last_session = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_checks = Counter()
        self.last_verified = None
        SupportMaskedSession.last_session = self

    @torch.no_grad()
    def _encode(self, cache, items, cost_key):
        state = super()._encode(cache, items, cost_key)
        flags = self.support.flags[:, -cache.seq_len:]
        self.costs['support_mask_flops'] += state.codes.numel()
        self.costs['support_mask_transfer_bytes'] += flags.nbytes
        self.costs['masked_clear_layer_rows'] += int((~flags).sum())
        return mask_codes(state, flags)

    @torch.no_grad()
    def prepare_read(self, at):
        super().prepare_read(at)
        stamp = self.trajectory.version, self.trajectory.cursor
        if verify_masked_state and self.state is not None and stamp != self.last_verified:
            ids = torch.as_tensor(self.trajectory.retained_events[1], device=self.cache.k.device)[None]
            features = self.trajectory.model.lookup_item_embeddings(ids)
            reference = mask_codes(self.adapter.encode_cache(self.cache, features), self.support.flags)
            error = float((reference.codes - self.state.codes).abs().max())
            self.probe_checks['masked_incremental_code_max_error'] = max(
                self.probe_checks['masked_incremental_code_max_error'], error)
            if not torch.allclose(reference.codes, self.state.codes, atol=2e-6, rtol=2e-5):
                raise RuntimeError('masked maintenance differs from full masked encoding')
            self.last_verified = stamp

    @torch.no_grad()
    def score(self, at, candidates):
        actual = super().score(at, candidates)
        if verify_masked_state and self.state is not None:
            delta = 0 if self.trajectory.last_timestamp is None else int(at) - self.trajectory.last_timestamp
            wanted = score(self.trajectory.model, self.cache, candidates,
                torch.tensor([delta], dtype=torch.float32, device=self.cache.k.device),
                history_read_fn=self.read_callback(candidates, force_torch=True))[0]
            error = float((actual - wanted).abs().max())
            self.probe_checks['masked_kernel_logit_max_error'] = max(
                self.probe_checks['masked_kernel_logit_max_error'], error)
            if not torch.allclose(actual, wanted, atol=2e-5, rtol=2e-5):
                raise RuntimeError('masked fused read differs from Torch')
        return actual


def score_user(*args, verify_state=False, **kwargs):
    global verify_masked_state
    verify_masked_state = verify_state
    rows, meta = original_score_user(*args, verify_state=False, **kwargs)
    meta['control_checks'].update(SupportMaskedSession.last_session.probe_checks)
    return rows, meta


def candidate_release(counts, **kwargs):
    row = original_candidate_release(counts, **kwargs)
    tokens = counts['lazy_initial_encode_tokens'] + counts['lazy_append_encode_tokens']
    work = tokens * kwargs['layers'] * kwargs['heads'] * kwargs['rank']
    row['components_flops']['row_support_mask'] = work
    row['variable_flops'] += work
    row['variable_percent_of_once_rebuild'] = 100 * row['variable_flops'] / row['first_request_rebuild_flops']
    row['shared_work_budget_at_20_percent_flops'] -= work
    row['support_mask_transfer_bytes'] = tokens * kwargs['layers']
    row['code_policy'] += ' Multiply each new code by its existing layer/row inherited-support flag; retained masks do not change.'
    return row


def execution_sources():
    name = 'experiments/archive/compact_read/evaluate_lightweight_support_mask.py'
    return {**original_sources(), name: sha256(ROOT / name)}


def bind_candidate():
    dispatch.bind_candidate()
    base.LightweightSession = SupportMaskedSession
    base.score_user = score_user
    base.candidate_release = candidate_release
    base.execution_sources = execution_sources


if __name__ == '__main__':
    dispatch.kv.bind_candidate = bind_candidate
    dispatch.kv.main()

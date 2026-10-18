#!/usr/bin/env python3
"""Frozen-weight equivalence and conversion scratch-memory probe, no training.

Real rows are already retained, strictly pre-release calibration-source rows.
Long repeated shapes are explicitly memory stress cases, not quality users or
additional evidence for the 20%/80% target. GPU peak numbers concern this code
conversion call only; no loaded-service latency/throughput claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch

from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from implementation_2026_10.state import InplaceReleasePrefix
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.precision import load_stable_prefix
from query_relay_2026_10.fixed_features import load_bank


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def memory_call(prefix, action, cache, initial, ids, expected):
    state = LightweightValueState(initial.clone(), cache.seq_len)
    pointer = state.codes.data_ptr()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = getattr(prefix, action)(cache, state, ids)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    error = (out.codes - expected).abs()
    torch.testing.assert_close(out.codes, expected, atol=1e-5, rtol=1e-5)
    return dict(peak_extra_allocated_bytes=int(peak), same_state_object=out is state,
                same_code_storage=out.codes.data_ptr() == pointer,
                max_abs=float(error.max()) if error.numel() else 0.,
                expected_rms=float(expected.square().mean().sqrt()),
                relative_l2=float(torch.linalg.vector_norm(error.double()) /
                    torch.linalg.vector_norm(expected.double()).clamp_min(1e-12)))


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    calibration = args.branch / "calibration/v5/calibration.pt"
    publication = args.branch / "publication/v5.pt"
    source = args.branch / "source_v5/source.pt"
    payload = torch.load(calibration, map_location="cpu", weights_only=False)
    adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
    adapter.load_state_dict(payload["state_dict"])
    prefix, metadata = load_stable_prefix(publication, adapter, device=device)
    pack = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    uid = next(iter(pack["rows"])) if args.source_uid is None else args.source_uid
    row = pack["rows"][uid]
    origin = row["parent"]
    n0 = origin.seq_len
    feature_fn = None
    if args.prepared_features:
        from functools import partial
        from implementation_2026_10.prepare import fixed_features_from_bank
        bank = load_bank(adapter.feature_bank_path, adapter.feature_bank_sha256,
                         device=device, dtype=next(adapter.parameters()).dtype)
        feature_fn = partial(fixed_features_from_bank, bank=bank)
    records = []
    for length in args.lengths:
        index = torch.arange(length) % n0
        cache = HSTUKVCache(origin.k[:, :, index].to(device),
                           origin.v[:, :, index].to(device), length)
        ids = row["item_ids"][:, index].to(device)
        # Retained calibration capture stores this as a packed tensor.
        initial = row["previous_codes"][:, :, index].to(device)
        # New current-version suffixes are represented by zero in the same
        # buffer; retain this distinction in the numerical conversion check.
        initial[:, :, -min(length, 17):] = 0
        state = LightweightValueState(initial, length)
        expected_anchor = prefix.anchorize(cache, state, ids).codes
        anchor_pointer = initial.data_ptr()
        anchor = InplaceReleasePrefix.wrap(prefix, row_chunk=args.chunks[0], feature_fn=feature_fn)
        scratch = LightweightValueState(initial.clone(), length)
        anchor.anchorize(cache, scratch, ids)
        expected_current = prefix.materialize(cache,
            LightweightValueState(expected_anchor, length), ids).codes
        # Warm encoder bank, BLAS setup and every implemented path before
        # allocator measurements. The large read-only shared tables stay live.
        for chunk in args.chunks:
            fast = InplaceReleasePrefix.wrap(prefix, row_chunk=chunk, feature_fn=feature_fn)
            temporary = LightweightValueState(initial.clone(), length)
            fast.anchorize(cache, temporary, ids)
            fast.materialize(cache, temporary, ids)
            del temporary
        result = dict(length=length, source_uid=int(uid), real_history_length=n0,
            scope="retained real rows" if length <= n0 else "repeated-row memory stress",
            native_kv_bytes=cache.k.numel() * cache.k.element_size() * 2,
            compact_bytes=initial.numel() * initial.element_size(),
            reference={}, chunks=[])
        result["reference"]["anchorize"] = memory_call(prefix, "anchorize", cache,
            initial, ids, expected_anchor)
        result["reference"]["materialize"] = memory_call(prefix, "materialize", cache,
            expected_anchor, ids, expected_current)
        for chunk in args.chunks:
            fast = InplaceReleasePrefix.wrap(prefix, row_chunk=chunk, feature_fn=feature_fn)
            entry = dict(row_chunk=chunk, geometry=fast.geometry())
            entry["anchorize"] = memory_call(fast, "anchorize", cache, initial,
                ids, expected_anchor)
            entry["materialize"] = memory_call(fast, "materialize", cache,
                expected_anchor, ids, expected_current)
            for action in ("anchorize", "materialize"):
                assert entry[action]["same_state_object"] and entry[action]["same_code_storage"]
            result["chunks"].append(entry)
        assert initial.data_ptr() == anchor_pointer
        records.append(result)
        print(json.dumps(result), flush=True)
        del cache, ids, initial, expected_anchor, expected_current, state, scratch
    output = dict(kind="implementation_chunked_inplace_state_v1", records=records,
        settings=dict(chunks=args.chunks, lengths=args.lengths, dtype="torch.float32",
            tf32=False, current_release=5, shared=12, private=6, heads=6, layers=6,
            prepared_features=args.prepared_features, source_uid=int(uid),
            no_training=True, no_quality_population_or_labels=True),
        prefix_inverse=prefix.inverse_cost(), shared_publication_bytes=prefix.storage_bytes(),
        publication_metadata=metadata,
        bindings={str(path.relative_to(ROOT)):sha(path) for path in
            (calibration, publication, source, Path(__file__), Path(__file__).with_name("state.py"))},
        device=dict(name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
        interpretation="Same mathematical conversion, one persistent code buffer. Peak extra allocated bytes include temporary tensors and reference output; they exclude resident input/native/shared tables. This is a synchronous inference probe, not a loaded serving benchmark. Chunking does not reduce the conversion's theoretical FLOPs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", type=Path, default=ROOT / "results/query_relay_2026_10/probes/A_itemgate6_rms_roll4")
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 8192, 32768])
    p.add_argument("--chunks", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--source-uid", type=int)
    p.add_argument("--prepared-features", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results/implementation_2026_10/state/gpu2.json")
    return p


if __name__ == "__main__":
    run(parser().parse_args())

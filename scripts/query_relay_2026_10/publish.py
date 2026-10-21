#!/usr/bin/env python3
"""Compile one full-catalogue A/B release into a reviewable local artifact.

This synchronous prototype keeps at most the previous and new D tables on the
producer device. All model item IDs, including row zero and the last embedding
row, are prepared. There is no request-population or future-item subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch

from design_two.common import setup_device
from evaluate_yambda500m_foundation_raw import load_model
from query_relay_2026_10.adapter import ReleasePrefix, UnifiedRelayAdapter
from query_relay_2026_10.fixed_features import load_bank
from unified_reuse_2026_09.common import sha256, write_json

KIND = "query_relay_full_catalogue_release_v1"
PANEL = ROOT / "results/design_two_2026_10/benchmark_panel/binding.json"


def feature_signature(adapter):
    """Only immutable feature semantics are required to match across releases."""
    result = dict(feature_mode=adapter.feature_mode, xi_width=adapter.layers[0].xi_width,
                  layers=adapter.num_layers, heads=adapter.heads, head_dim=adapter.head_dim,
                  code_width=adapter.code_width, layout=adapter.layout)
    if adapter.feature_mode == "origin_encoder":
        result["feature_bank_sha256"] = adapter.feature_bank_sha256
    else:
        h = hashlib.sha256()
        for layer in adapter.layers:
            for j in range(len(layer.branches)):
                value = getattr(layer, f"native_basis_{j}").detach().cpu().contiguous()
                h.update(str(value.dtype).encode())
                h.update(str(tuple(value.shape)).encode())
                h.update(value.view(torch.uint8).numpy().tobytes())
        result["native_basis_sha256"] = h.hexdigest()
    return result


def load_prefix(path, adapter, *, device=None, expected_sha256=None):
    """Restore tensors and attach CURRENT layers after checking frozen features."""
    path = Path(path).resolve()
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("published prefix SHA256 changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    metadata = payload["metadata"]
    if metadata["kind"] != KIND:
        raise ValueError("not a full-catalogue release artifact")
    if metadata["feature_signature"] != feature_signature(adapter):
        raise ValueError("published prefix and current adapter have different fixed features")
    reference = next(adapter.parameters())
    target = reference.device if device is None else torch.device(device)
    prefix = ReleasePrefix(payload["P"].to(device=target, dtype=reference.dtype),
        payload["Q"].to(device=target, dtype=reference.dtype),
        payload["D"].to(device=target, dtype=reference.dtype), tuple(adapter.layers),
        int(metadata["composed_releases"]))
    if prefix.D.shape[1] != metadata["catalogue_table_rows"]:
        raise ValueError("published catalogue tensor and metadata disagree")
    return prefix, metadata


@torch.no_grad()
def compile_full_catalogue(adapter, model, previous=None, *, chunk_size=16384):
    """Same block arithmetic as the subset compiler, using original model IDs."""
    if adapter.mode not in ("A", "B"):
        raise ValueError("full publication requires a composable A/B updater")
    if chunk_size <= 0:
        raise ValueError("item chunk must be positive")
    reference = next(adapter.parameters())
    rows = len(model.item_emb.weight)
    code = adapter.code_width
    xi = adapter.layers[0].xi_width
    if rows != model.cfg.num_items + 1:
        raise ValueError("model item-ID range and embedding rows disagree")
    if previous is not None:
        if previous.D.shape != (adapter.num_layers, rows, code, xi):
            raise ValueError("previous publication has a different catalogue/summary shape")
        old = previous.layers[0]
        # load_prefix already checks the saved feature signature. Direct callers
        # also get protection against incompatible in-memory references.
        if old.feature_mode != adapter.feature_mode or old.xi_width != xi:
            raise ValueError("previous publication changed fixed feature semantics")
        for index, layer in enumerate(adapter.layers):
            prior = previous.layers[index]
            if layer.feature_mode == "origin_encoder":
                if layer.feature_bank_sha256 != prior.feature_bank_sha256:
                    raise ValueError("previous publication changed the immutable item bank")
            elif any(not torch.equal(getattr(layer, f"native_basis_{j}"),
                                     getattr(prior, f"native_basis_{j}"))
                     for j in range(len(layer.branches))):
                raise ValueError("previous publication changed fixed native feature basis")
    d = torch.empty((adapter.num_layers, rows, code, xi),
                    device=reference.device, dtype=reference.dtype)
    p, q, work = [], [], dict(item_projection_flops=0, history_composition_flops=0,
                              matrix_composition_flops=0, cayley_preparation_flops_estimate=0)
    for index, layer in enumerate(adapter.layers):
        rotations = layer.rotation_blocks()
        widths = [branch.code_width for branch in layer.branches]
        work["cayley_preparation_flops_estimate"] += sum(
            (8 * width**3 + 2) // 3 + 6 * width**2 for width in widths)
        pblocks, qblocks = [], []
        for branch, a, b in zip(layer.branches, rotations, layer.native_update, strict=True):
            section = slice(branch.code_start, branch.code_start + branch.code_width)
            if previous is None:
                # A@I and A@0+B need no arithmetic in this first publication.
                pblocks.append(a.clone())
                qblocks.append(b.clone())
            else:
                pblocks.append(a @ previous.P[index, section, section])
                qblocks.append(a @ previous.Q[index, section, section] + b)
                work["matrix_composition_flops"] += 4 * branch.code_width**3 + branch.code_width**2
        p.append(torch.block_diag(*pblocks))
        q.append(torch.block_diag(*qblocks))
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            ids = torch.arange(start, stop, device=reference.device)[None]
            features = model.lookup_item_embeddings(ids)[0]
            current = layer.item_coefficients(features)
            work["item_projection_flops"] += (stop - start) * (
                2 * layer.item_width * code * xi + code * xi)
            if previous is not None:
                for branch, a in zip(layer.branches, rotations, strict=True):
                    section = slice(branch.code_start, branch.code_start + branch.code_width)
                    d[index, start:stop, section] = torch.einsum("cd,mdr->mcr", a,
                        previous.D[index, start:stop, section])
                d[index, start:stop] += current
                work["history_composition_flops"] += (stop - start) * (
                    2 * sum(width**2 for width in widths) * xi + code * xi)
            else:
                d[index, start:stop].copy_(current)
    prefix = ReleasePrefix(torch.stack(p), torch.stack(q), d, tuple(adapter.layers),
                           1 if previous is None else previous.release + 1)
    return prefix, work


def save_publication(path, prefix, adapter, model, *, target_version, work,
                     bindings, previous_metadata=None):
    """Save one CPU pt and its JSON index; retain the previous artifact separately."""
    path = Path(path).resolve()
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    if prefix.D.is_cuda:
        torch.cuda.synchronize(prefix.D.device)
    rows = len(model.item_emb.weight)
    bank = None
    if adapter.feature_mode == "origin_encoder":
        frozen = load_bank(adapter.feature_bank_path, adapter.feature_bank_sha256,
                           device="cpu", dtype=next(adapter.parameters()).dtype)
        if frozen.table.shape[1] != rows:
            raise ValueError("immutable feature bank does not cover the full catalogue")
        bank = dict(path=adapter.feature_bank_path, sha256=adapter.feature_bank_sha256,
                    storage_bytes=frozen.storage_bytes(),
                    table_bytes=frozen.table.numel() * frozen.table.element_size(),
                    catalogue_table_rows=frozen.table.shape[1], scope="One fixed bank across all releases")
    table_bytes = prefix.D.numel() * prefix.D.element_size()
    matrix_bytes = sum(value.numel() * value.element_size() for value in (prefix.P, prefix.Q))
    metadata = dict(kind=KIND, target_version=int(target_version),
        composed_releases=prefix.release, catalogue_table_rows=rows,
        model_num_items=model.cfg.num_items, raw_model_item_id_range_inclusive=[0, rows - 1],
        item_id_semantics="Original model item_idx, including padding row0; no subset remapping, no external raw_item_id",
        adapter_config=adapter.get_config(), feature_signature=feature_signature(adapter),
        feature_bank=bank, dtype=str(prefix.D.dtype), tensor_shapes=dict(
            P=list(prefix.P.shape), Q=list(prefix.Q.shape), D=list(prefix.D.shape)),
        tensor_bytes=dict(item_table=table_bytes, composition_matrices=matrix_bytes,
                          total=table_bytes + matrix_bytes),
        actual_compilation_components_flops=dict(work), actual_compilation_flops=sum(work.values()),
        producer_device=str(prefix.D.device), producer_had_full_device_table=True,
        producer_had_full_cuda_table=prefix.D.is_cuda,
        centralized_compile_then_four_copies=dict(preparation_flops=sum(work.values()),
            additional_publication_transfer_bytes=3 * (table_bytes + matrix_bytes),
            resident_copies=4, all_copies_payload_bytes=4 * (table_bytes + matrix_bytes),
            scope="Transfer is a payload estimate; no external broadcast or loaded-serving benchmark executed"),
        independent_four_replica_preparation_upper_bound_flops=4 * sum(work.values()),
        bindings=dict(bindings), previous_target_version=None if previous_metadata is None
            else previous_metadata["target_version"],
        source_sha256=sha256(Path(__file__)))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(metadata=metadata, P=prefix.P.detach().cpu(),
                    Q=prefix.Q.detach().cpu(), D=prefix.D.detach().cpu()), path)
    index = dict(metadata, artifact_path=str(path), artifact_sha256=sha256(path),
                 file_bytes=path.stat().st_size)
    write_json(path.with_suffix(".json"), index)
    return index


def run(args):
    began = time.perf_counter()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    target = int(args.target_version)
    if (target == 3) != (args.previous is None):
        raise ValueError("V3 starts the chain; V4/V5 require the preceding publication")
    device = setup_device(args.gpu, args.threads)
    calibration = args.calibrations / f"v{target}" / "calibration.pt"
    calibration_metadata = calibration.with_suffix(".json")
    calibration_digest = sha256(calibration)
    if calibration_metadata.exists():
        learned = json.loads(calibration_metadata.read_text())
        if learned.get("target", f"v{target}") != f"v{target}":
            raise ValueError("calibration checkpoint targets a different release")
        if learned.get("weights_sha256", calibration_digest) != calibration_digest:
            raise ValueError("calibration checkpoint and metadata SHA256 disagree")
    payload = torch.load(calibration, map_location="cpu", weights_only=False)
    if payload["kind"] != UnifiedRelayAdapter.kind:
        raise ValueError("publication requires a query-relay checkpoint")
    adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
    adapter.load_state_dict(payload["state_dict"])
    panel = json.loads(args.panel.read_text())
    model_record = panel["models"][f"v{target}"]
    model_path = ROOT / model_record["checkpoint"]
    if sha256(model_path) != model_record["checkpoint_sha256"]:
        raise ValueError("Current model checkpoint changed")
    model, model_payload = load_model(model_path, device)
    del model_payload
    previous, previous_metadata = None, None
    bindings = dict(calibration_path=str(calibration.resolve()), calibration_sha256=calibration_digest,
                    current_model_path=str(model_path.resolve()), current_model_sha256=sha256(model_path),
                    panel_sha256=sha256(args.panel))
    if calibration_metadata.exists():
        bindings["calibration_metadata_sha256"] = sha256(calibration_metadata)
    if args.previous is not None:
        bindings["previous_artifact_path"] = str(args.previous.resolve())
        bindings["previous_artifact_sha256"] = sha256(args.previous)
        previous, previous_metadata = load_prefix(args.previous, adapter, device=device,
            expected_sha256=bindings["previous_artifact_sha256"])
        if previous_metadata["target_version"] != target - 1:
            raise ValueError("previous publication is not the adjacent release")
    prefix, work = compile_full_catalogue(adapter, model, previous, chunk_size=args.item_chunk)
    del previous
    metadata = save_publication(args.output, prefix, adapter, model, target_version=target,
        work=work, bindings=bindings, previous_metadata=previous_metadata)
    print(json.dumps(dict(status="complete_full_catalogue_publication", output=str(args.output),
        target_version=target, catalogue_table_rows=metadata["catalogue_table_rows"],
        table_bytes=metadata["tensor_bytes"]["item_table"], compilation_flops=metadata["actual_compilation_flops"],
        artifact_sha256=metadata["artifact_sha256"], elapsed_seconds=time.perf_counter() - began)), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrations", type=Path, required=True)
    p.add_argument("--target-version", type=int, choices=(3, 4, 5), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--item-chunk", type=int, default=16384)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

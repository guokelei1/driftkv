#!/usr/bin/env python3
"""Equivalence and isolated preparation probe; no fitting or quality selection."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src")]

import torch

from evaluate_yambda500m_foundation_raw import load_model
from hstu_kvcache.models import HSTUKVCache
from implementation_2026_10.prepare import (compile_item_tables, native_feature_chunks,
    project_fixed_item_terms)
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.fixed_features import load_bank
from query_relay_2026_10.publish import compile_full_catalogue
from unified_reuse_2026_09.common import sha256, write_json

BASE = ROOT/"results/query_relay_2026_10/probes/A_itemgate6_rms_roll4"


class EmbeddingModel:
    """Keep the actual lookup operation for a selected real-item subset."""
    def __init__(self, embeddings):
        self.item_emb = type("Embedding", (), {"weight": embeddings})()
        self.cfg = type("Configuration", (), {"num_items": len(embeddings)-1})()

    def lookup_item_embeddings(self, ids):
        return self.item_emb.weight[ids]


def difference(actual, expected, *, chunk_size=16384):
    maximum, squared, reference_squared = 0., 0., 0.
    # Catalogue/row axis is one for every table checked here.
    for start in range(0, actual.shape[1], chunk_size):
        a = actual[:, start:start+chunk_size]
        b = expected[:, start:start+chunk_size].to(a)
        delta = (a-b).double()
        maximum = max(maximum, float(delta.abs().max()))
        squared += float(delta.square().sum())
        reference_squared += float(b.double().square().sum())
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    return dict(max_abs=maximum, relative_l2=(squared/max(reference_squared,1e-30))**.5)


def timed(function, *, repeats=5):
    for _ in range(2):
        value = function()
        del value
    torch.cuda.synchronize()
    observations = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        value = function()
        stop.record(); stop.synchronize()
        observations.append(start.elapsed_time(stop))
        del value
    return dict(observations_ms=observations, median_ms=statistics.median(observations),
                warmups=2, repeats=repeats,
                scope="Isolated synchronous CUDA operations; no loaded-serving throughput or p99")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output/"summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.70, device)
    freeze_path = BASE/"freeze.json"
    freeze = json.loads(freeze_path.read_text())
    binding = dict(freeze_sha256=sha256(freeze_path), source_sha256={
        str(path.relative_to(ROOT)):sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("prepare.py"),
            ROOT/"scripts/query_relay_2026_10/adapter.py",
            ROOT/"scripts/query_relay_2026_10/publish.py")}, releases={})
    report = dict(kind="current_scheme_preparation_equivalence", device=str(device),
        gpu_name=torch.cuda.get_device_name(device), dtype="float32", tf32=False,
        chunk_size=args.chunk_size, full_catalogue=args.full_catalogue,
        seed=20261015, bindings=binding, releases={},
        exclusions="No training, quality selection, new labels, old publication replacement, or row eta persistence")
    generator = torch.Generator().manual_seed(20261015)
    catalogue_rows = freeze["fixed_feature_bank"]["catalogue_rows"]
    ids = torch.unique(torch.cat((torch.tensor([0,catalogue_rows-1]),
        torch.randint(1,catalogue_rows-1,(args.subset_rows-2,),generator=generator)))).to(device)
    report["real_item_subset"] = dict(requested_rows=args.subset_rows, unique_rows=len(ids),
        includes_padding_and_last=True, selection="Seeded uniform IDs, no request/user-dependent selection")
    previous = None
    for version in (3,4,5):
        name = f"v{version}"
        record = freeze["release_bindings"][name]
        checkpoint, model_path = Path(record["calibration_path"]), Path(record["model_path"])
        assert sha256(checkpoint) == record["calibration_sha256"]
        assert sha256(model_path) == record["model_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
        adapter.load_state_dict(payload["state_dict"])
        model, model_payload = load_model(model_path, device)
        model.eval().requires_grad_(False)
        del model_payload, payload
        binding["releases"][name] = dict(calibration_sha256=record["calibration_sha256"],
            model_sha256=record["model_sha256"])
        subset = EmbeddingModel(model.item_emb.weight[ids])
        subset_prefix, subset_work = compile_item_tables(adapter, subset, chunk_size=1024)
        subset_reference, subset_reference_work = compile_full_catalogue(adapter, subset, chunk_size=1024)
        assert subset_work["arithmetic"] == subset_reference_work
        subset_diff = difference(subset_prefix.D, subset_reference.D)
        timings = dict(reference=timed(lambda:compile_full_catalogue(adapter,subset,chunk_size=1024)),
                       batched=timed(lambda:compile_item_tables(adapter,subset,chunk_size=1024)))
        result = dict(real_subset_D_difference=subset_diff, subset_work=subset_work,
                      subset_isolated_timing=timings)
        del subset_prefix, subset_reference
        if version == 3:
            bank_record = freeze["fixed_feature_bank"]
            bank = load_bank(bank_record["path"],bank_record["sha256"],device=device,dtype=torch.float32)
            subset_bank, subset_bank_work = project_fixed_item_terms(adapter, subset.item_emb.weight,
                                                                    chunk_size=1024)
            result["subset_phi_difference"] = difference(subset_bank.table,bank.table[:,ids])
            result["subset_phi_work"] = subset_bank_work
            del subset_bank
            if args.full_catalogue:
                begin = time.perf_counter()
                optimized_bank, bank_work = project_fixed_item_terms(adapter, model.item_emb.weight,
                                                                      chunk_size=args.chunk_size)
                torch.cuda.synchronize()
                report["fixed_phi"] = dict(work=bank_work,
                    difference=difference(optimized_bank.table,bank.table),
                    one_observation_elapsed_seconds=time.perf_counter()-begin,
                    bank_sha256=bank_record["sha256"],
                    scope="Immutable all-catalogue table; prepared once, not once per release")
            else:
                optimized_bank = bank
            pack_path = BASE/"source_v4/source.pt"
            pack = torch.load(pack_path,map_location="cpu",weights_only=False,mmap=True)
            feature_checks = []
            for uid in list(pack["rows"])[:8]:
                row = pack["rows"][uid]
                cache = row["parent"]
                cache = HSTUKVCache(cache.k.to(device),cache.v.to(device),cache.seq_len)
                item_ids = row["item_ids"].to(device)
                chunks = list(native_feature_chunks(adapter,cache,item_ids,optimized_bank,chunk_size=257))
                actual = torch.cat([part[2] for part in chunks],2)
                expected = torch.stack([layer.fixed_features(cache.k[i],cache.v[i],item_ids)[0]
                                         for i,layer in enumerate(adapter.layers)])
                feature_checks.append(dict(uid=int(uid),rows=cache.seq_len,
                    difference=difference(actual.transpose(1,2),expected.transpose(1,2))))
            report["actual_native_features"] = dict(source_pack_sha256=sha256(pack_path),
                users=feature_checks, scope="8 frozen fit-state trajectories; equivalence only, no quality optimization",
                row_eta_flops=6*(2*(384*12+6*64*6)+5*48),
                avoided_naive_row_item_projection_flops=2*6*192*48,
                eta_persistent_bytes=0,
                note="Per-row saving is relative to recomputing the item projection, not a new saving over the existing shared-bank method")
            del pack, chunks, actual, expected, cache, optimized_bank, bank
        if args.full_catalogue:
            begin = time.perf_counter()
            prefix, work = compile_item_tables(adapter,model,previous,chunk_size=args.chunk_size)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            reference_path = BASE/"publication"/f"{name}.pt"
            reference_metadata = json.loads(reference_path.with_suffix(".json").read_text())
            assert sha256(reference_path) == reference_metadata["artifact_sha256"]
            old = torch.load(reference_path,map_location="cpu",weights_only=True,mmap=True)
            assert work["arithmetic"] == old["metadata"]["actual_compilation_components_flops"]
            torch.testing.assert_close(prefix.P,old["P"].to(device),atol=0,rtol=0)
            torch.testing.assert_close(prefix.Q,old["Q"].to(device),atol=0,rtol=0)
            result["full_catalogue"] = dict(work=work, D_difference=difference(prefix.D,old["D"]),
                P_Q_bitwise_equal=True, original_publication_sha256=reference_metadata["artifact_sha256"],
                original_compilation_flops=old["metadata"]["actual_compilation_flops"],
                one_observation_elapsed_seconds=elapsed,
                scope="Fresh V3-to-V5 optimized compilation, each from optimized prior; no table copied from retained publication")
            del old
            previous = prefix
        report["releases"][name] = result
        write_json(args.output/"progress.json",report)
        print(json.dumps(dict(status="completed",release=name,subset_max_abs=subset_diff["max_abs"],
            full=result.get("full_catalogue",{}).get("D_difference"))),flush=True)
        del model, subset, adapter
        gc.collect()
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["arithmetic_conclusion"] = "All-layer batching preserves shared-preparation FLOPs; it reduces repeated embedding reads and operation count. Full catalogue has no projection misses, so batch dedup is not counted again."
    write_json(summary_path,report)
    print(json.dumps(dict(status="complete",summary=str(summary_path))),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--subset-rows",type=int,default=4096)
    p.add_argument("--chunk-size",type=int,default=16384)
    p.add_argument("--full-catalogue",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

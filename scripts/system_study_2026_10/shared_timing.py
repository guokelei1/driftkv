#!/usr/bin/env python3
"""Measure actual full-catalogue preparation and retain calibration timer scopes.

Uses frozen weights and existing numerical implementations. No fitting, labels,
or user trajectories are added. Four replica copies are sequential observations
on ONE GPU, not a measurement of a four-GPU concurrent deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch
from hstu_kvcache.models import HSTU, HSTUConfig
from implementation_2026_10.backend import InferenceAdapter
from implementation_2026_10.prepare import compile_item_tables, project_fixed_item_terms
from implementation_2026_10.run_prepare import difference
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.precision import StableReleasePrefix
from unified_reuse_2026_09.common import sha256, write_json

BASE = ROOT / "results/query_relay_2026_10/probes/A_itemgate6_rms_roll4"


def calibration_evidence():
    phases = []
    for version in (3, 4, 5):
        path = BASE / f"calibration/v{version}/calibration.json"
        record = json.loads(path.read_text())
        phases.append(dict(name="relay_calibration", version=f"v{version}",
            wall_seconds=record["diagnostics"]["elapsed_seconds"], evidence=str(path),
            evidence_sha256=sha256(path), timing_kind="retained_actual_wall_observation",
            fit_users=record["fit_users"], validation_users=record["validation_users"],
            local_epochs=record["local_epochs"], joint_epochs=record["joint_epochs"],
            seed=record["seed"], rollout_steps=record["rollout_steps"],
            scope="run entry through final gate diagnostics: inputs/model/history loads, preparation, local and joint fitting; excludes final checkpoint save/hash",
            source_teacher="V3 source and teacher captured inside this timer" if version == 3 else "external source+teacher pack loaded here; its generation is a separate phase",
            excluded="fixed feature-bank creation, inherited origin training, foundation training, publication compile and replica distribution",
            replicas=1, frequency="once per publication"))
        if version > 3:
            path = BASE / f"source_v{version}/summary.json"
            source = json.loads(path.read_text())
            phases.append(dict(name="source_teacher_capture", version=f"v{version}",
                wall_seconds=source["elapsed_seconds"], evidence=str(path), evidence_sha256=sha256(path),
                timing_kind="retained_actual_wall_observation",
                scope="run entry: model/prior method/history loads, causal source replay and current teacher, source pack serialization and hash; excludes final summary write",
                already_in_calibration_timer=False, replicas=1,
                frequency="once per publication in recorded offline replay workflow"))
    path = ROOT / "results/code_structure_2026_10/shared12_private6/calibration/v3/calibration.json"
    origin = json.loads(path.read_text())
    phases.append(dict(name="structured_origin_joint", version="origin_v3",
        wall_seconds=origin["elapsed_seconds"], evidence=str(path), evidence_sha256=sha256(path),
        timing_kind="retained_actual_wall_observation", replicas=1, frequency="once per origin",
        scope="model/input loads, repeated source+teacher capture, layout conversion, joint16, checks, weight save/hash; inherits separately trained local32",
        overlap_warning="Includes repeated teacher/source reconstruction versus local32 origin; actual separate process expense, not minimal resident implementation cost"))
    path = ROOT / "results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration/v3/calibration.json"
    original = json.loads(path.read_text())
    local = sum(row.get("read_capture_and_fit_seconds", 0) for row in original["timings"])
    unused_joint = sum(row.get("fitting_seconds", 0) for row in original["timings"] if row.get("stage") == "joint")
    phases.append(dict(name="inherited_local32_origin_prefix", version="origin_v3",
        wall_seconds=None, lower_seconds=local, upper_seconds=original["elapsed_seconds"]-unused_joint,
        timing_kind="censored_prefix_bounds_not_confidence_interval", evidence=str(path), evidence_sha256=sha256(path),
        replicas=1, frequency="once per origin",
        scope="Lower: measured six layer capture+local fits; upper: original entire process minus its unused joint fitting timer. Necessary input/model/source/teacher loads and read-only checkpoint save lack a separate prefix timer.",
        original_whole_process_seconds=original["elapsed_seconds"], unused_joint_fit_seconds=unused_joint,
        inherited_checkpoint=str(path.parent / "read_only/calibration.pt")))
    return dict(phases=phases,
        interpretation="These are offline single-run wall times on the historical machine, not GPU busy-seconds, replica-multiplied cost or latency distributions. Their input/replay costs are real but repeated in separate commands; a resident optimized workflow may reuse them.",
        missing="Exact local32 prefix wall time and final relay checkpoint serialization were not separately measured. Foundation training is common model production and excluded. No new fitting is performed.")


class Timer:
    def __init__(self, report, output, device):
        self.report, self.output, self.device = report, output, device

    def measure(self, name, function, *, version=None, category="shared_compute", gpu=False, **scope):
        torch.cuda.synchronize(self.device)
        if gpu:
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            # Initialize events outside timing to avoid confusing their first-use cost.
            start.record(); stop.record(); stop.synchronize()
        begin = time.perf_counter()
        if gpu:
            start.record()
        result = function()
        if gpu:
            stop.record(); stop.synchronize()
        wall = time.perf_counter() - begin
        record = dict(name=name, version=version, category=category, wall_seconds=wall,
            cuda_event_ms=start.elapsed_time(stop) if gpu else None, repetitions=1, **scope)
        self.report["phases"].append(record)
        write_json(self.output / "progress.json", self.report)
        print(json.dumps(record), flush=True)
        return result


def bytes_of(tensors):
    return sum(x.numel()*x.element_size() for x in tensors)


def distribute(tensors, timer, name, version, replicas):
    size = bytes_of(tensors)
    host = timer.measure(name+"_d2h", lambda: [x.detach().to("cpu", copy=True) for x in tensors],
        version=version, category="distribution", gpu=True, tensor_bytes=size,
        scope="GPU publisher to pageable CPU allocations; includes allocations and synchronous copies", replicas=1)
    for replica in range(replicas):
        copied = timer.measure(name+"_h2d", lambda: [x.to(timer.device, copy=True) for x in host],
            version=version, category="distribution", gpu=True, tensor_bytes=size, replica_index=replica,
            replicas=replicas, scope="One actual pageable CPU to GPU copy; all replica observations sequentially reuse the same GPU, not concurrent four-device wall time")
        del copied
    del host


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError("use a new attempt directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    context_start = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device); torch.empty(1, device=device); torch.cuda.synchronize()
    report = dict(kind="actual_shared_preparation_timing", started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_name=torch.cuda.get_device_name(), device=str(device), dtype="float32", tf32=False,
        chunk_size=args.chunk_size, replicas=args.replicas, phases=[], checks={},
        context_startup_seconds=time.perf_counter()-context_start,
        timing_scope="Single-process actual synchronized API wall time; CUDA event spans include launch gaps, not pure GPU busy time. One fresh full-catalogue generation per version, no claim about timing variance.",
        storage_scope="OS page cache uncontrolled, no cache flush: file load is actual observed CPU deserialization, not guaranteed physical disk cold-read latency.",
        exclusions="No foundation/method retraining; no user C or requests R, queue/network transfer, artifact persistence/fsync or concurrent replica deployment. Verification/hashing excluded from phase times.",
        bindings={"freeze": str(BASE/"freeze.json"), "freeze_sha256": sha256(BASE/"freeze.json"),
            "entry_sha256": sha256(Path(__file__))})
    write_json(args.output / "calibration_timing.json", calibration_evidence())
    timer = Timer(report, args.output, device)
    freeze = json.loads((BASE/"freeze.json").read_text())
    previous = None
    for version in (3, 4, 5):
        name = f"v{version}"
        binding = freeze["release_bindings"][name]
        path = Path(binding["model_path"])
        payload = timer.measure("foundation_checkpoint_cpu_load", lambda: torch.load(path, map_location="cpu", weights_only=False),
            version=name, category="common_model_startup", file_bytes=path.stat().st_size,
            scope="Existing base model also required by Reuse/Rebuild; not all method-added cost")
        if payload["progress"] != 1.0 or not payload["status"].startswith("formal_"):
            raise ValueError("not a complete foundation checkpoint")
        model = timer.measure("foundation_cpu_construct", lambda: HSTU(HSTUConfig(**payload["config"])),
            version=name, category="common_model_startup", scope="Same constructor as existing load_model")
        model = timer.measure("foundation_initial_to_gpu", lambda: model.to(device), version=name,
            category="common_model_startup", gpu=True, scope="Existing loader transfers initialized model before loading weights")
        timer.measure("foundation_weights_h2d", lambda: model.load_state_dict(payload["model"]),
            version=name, category="common_model_startup", gpu=True, tensor_bytes=bytes_of(payload["model"].values()))
        model.eval().requires_grad_(False)
        del payload
        checkpoint = Path(binding["calibration_path"])
        state = timer.measure("adapter_checkpoint_cpu_load", lambda: torch.load(checkpoint, map_location="cpu", weights_only=False),
            version=name, category="method_input_load", file_bytes=checkpoint.stat().st_size)
        def make_adapter():
            adapter = UnifiedRelayAdapter(**state["config"]).to(device).eval().requires_grad_(False)
            adapter.load_state_dict(state["state_dict"])
            return adapter
        adapter = timer.measure("adapter_construct_and_h2d", make_adapter, version=name,
            category="method_input_load", gpu=True, tensor_bytes=bytes_of(state["state_dict"].values()))
        del state
        if version == 3:
            bank, bank_work = timer.measure("fixed_bank_generate", lambda: project_fixed_item_terms(
                adapter, model.item_emb.weight, chunk_size=args.chunk_size), version="origin_v3", gpu=True,
                frequency="once per immutable origin", scope="Fresh complete catalogue projection, no precomputed phi loaded")
            report["fixed_bank_work"] = bank_work
            distribute([bank.table, bank.bias]+[w for layer in bank.native_weights for w in layer],
                       timer, "fixed_bank", "origin_v3", args.replicas)
            verify_start = time.perf_counter()
            reference = torch.load(freeze["fixed_feature_bank"]["path"], map_location="cpu", weights_only=True, mmap=True)
            report["checks"]["fixed_bank"] = difference(bank.table, reference["table"])
            report["checks"]["fixed_bank"]["verification_wall_seconds_excluded"] = time.perf_counter()-verify_start
            del reference, bank
        prefix, work = timer.measure("release_table_compile", lambda: compile_item_tables(
            adapter, model, previous, chunk_size=args.chunk_size), version=name, gpu=True,
            frequency="once per publication", scope="Fresh P/Q/D generation, V4/V5 from newly generated prior publication")
        stable = timer.measure("small_block_inverse", lambda: StableReleasePrefix.wrap(prefix), version=name,
            gpu=True, frequency="once per publication", scope="Inverse actual FP32 P in FP64 then retain FP32 inverse")
        reader = timer.measure("reader_pack", lambda: InferenceAdapter(adapter), version=name,
            gpu=True, frequency="once per publication")
        tensors = [prefix.P, prefix.Q, prefix.D]
        tensors += [x for blocks in stable.inverse_blocks for x in blocks]
        tensors += list(adapter.state_dict().values())
        tensors += [reader.gate_weights]+[x for pair in reader.decoders for x in pair]
        distribute(tensors, timer, "publication", name, args.replicas)
        report.setdefault("release_work", {})[name] = work
        verify_start = time.perf_counter()
        old_path = BASE / f"publication/{name}.pt"
        old = torch.load(old_path, map_location="cpu", weights_only=True, mmap=True)
        assert work["arithmetic"] == old["metadata"]["actual_compilation_components_flops"]
        torch.testing.assert_close(prefix.P, old["P"].to(device), atol=0, rtol=0)
        torch.testing.assert_close(prefix.Q, old["Q"].to(device), atol=0, rtol=0)
        report["checks"][name] = dict(D=difference(prefix.D, old["D"]), P_Q_bitwise_equal=True,
            arithmetic_equal=True, verification_wall_seconds_excluded=time.perf_counter()-verify_start)
        # Hash only after measured input load so hashing cannot prewarm that load.
        assert sha256(checkpoint) == binding["calibration_sha256"]
        assert sha256(path) == binding["model_sha256"]
        report["bindings"][name] = binding
        previous = prefix
        del tensors, old, model, adapter, reader, stable
        gc.collect()
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["aggregates"] = dict(
        resident_shared_compute_seconds=sum(p["wall_seconds"] for p in report["phases"] if p["category"] == "shared_compute"),
        method_input_load_seconds=sum(p["wall_seconds"] for p in report["phases"] if p["category"] == "method_input_load"),
        common_model_startup_seconds=sum(p["wall_seconds"] for p in report["phases"] if p["category"] == "common_model_startup"),
        distribution_serial_seconds=sum(p["wall_seconds"] for p in report["phases"] if p["category"] == "distribution"),
        composition_warning="Historical calibration includes its own input loads. Add resident generation and required distribution to those recorded workflows; do not also add this standalone foundation reload without explicitly assuming separate preparation commands. Replicas multiply transfers, not the single publication fit/compile.")
    write_json(args.output / "summary.json", report)
    print(json.dumps(dict(status="complete", summary=str(args.output/"summary.json"), aggregates=report["aggregates"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=16384)
    p.add_argument("--replicas", type=int, default=4)
    run(p.parse_args())

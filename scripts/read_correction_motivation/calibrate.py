#!/usr/bin/env python3
"""Supplement the fixed H-v4 map_all curve with C64 and C256 calibrations.

Only calibration_users changes. The original pure-Parent capture, uniform
catalog queries, ridge initialization and six-epoch nonlinear fitter are reused.
Evaluation remains a separate invocation of read_correction_v4/evaluate_full.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from read_correction_v4 import probe
from read_correction_v4.common import PANEL_ROOT, RESERVATIONS, edge_name, sha256, write_json
from read_correction_v4.cost_nonlinear import correction_forward
from read_correction_v4.probe_nonlinear import fit, nonlinear_configuration

FROZEN_SOURCES = probe.sources
KIND = "token_read_nonlinear_v4"
DEFAULT_OUTPUT = ROOT / "results/read_correction_2026_09/motivation/history_conditioned/calibration"


def configuration(budget):
    if budget not in (64, 128, 256):
        raise ValueError("the fixed curve uses C64, C128 and C256")
    return {**nonlinear_configuration(), "calibration_users": budget}


def sources():
    return {**FROZEN_SOURCES(), str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__)}


def expected_inputs(scale, edge, budget):
    binding_path = PANEL_ROOT / scale / edge_name(edge) / "binding.json"
    reservation_path = RESERVATIONS / scale / edge_name(edge) / "calibration_users.json"
    binding = json.loads(binding_path.read_text())
    reserved = json.loads(reservation_path.read_text())
    binding_hash = sha256(binding_path)
    if reserved["source_binding"]["sha256"] != binding_hash:
        raise RuntimeError("calibration reservation no longer matches the original panel")
    uids = reserved["uids"][:budget + 16]
    if len(uids) != budget + 16 or len(set(uids)) != len(uids):
        raise RuntimeError("insufficient or repeated independent calibration UIDs")
    if binding["panels"]["evaluation"]["users"] != 3000:
        raise RuntimeError("the retained evaluation panel must contain 3000 users")
    # Check the actual UID file without reading evaluation labels or predictions.
    import pyarrow.parquet as pq
    panel_users = binding["files"]["evaluation_users"]
    panel_users_path = ROOT / panel_users["path"]
    if sha256(panel_users_path) != panel_users["sha256"]:
        raise RuntimeError("original evaluation UID file changed")
    evaluation_uids = set(pq.read_table(panel_users_path, columns=["uid"])["uid"].to_pylist())
    if len(evaluation_uids) != 3000 or evaluation_uids.intersection(uids):
        raise RuntimeError("calibration and original evaluation UIDs are not disjoint")
    return {
        "panel_binding_sha256": binding_hash,
        "calibration_users_sha256": sha256(reservation_path),
        "checkpoint_hashes": {key: binding["sources"][key]["sha256"] for key in ("parent", "current")},
        "uids": uids[:budget], "validation_uids": uids[budget:],
    }


def validate_record(path, config, expected, source_hashes):
    saved = json.loads(path.read_text())
    for key, value in {"status": "complete", "settings": config,
                       "execution_sources": source_hashes, **expected}.items():
        if saved.get(key) != value:
            raise RuntimeError(f"retained calibration differs in {key}: {path}")
    weights = path.with_suffix(".pt")
    if not weights.exists() or sha256(weights) != saved["weights_sha256"]:
        raise RuntimeError(f"retained calibration weights differ: {weights}")
    import torch
    artifact = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    if artifact["kind"] != KIND or any(row["config"]["query_affine"] for row in artifact["modules"]):
        raise RuntimeError("the retained artifact is not the fixed nonlinear H without query compensation")
    return saved


def run(args):
    cfg = configuration(args.budget)
    source_hashes = sources()
    expected = expected_inputs(args.scale, args.edge, args.budget)
    budget_root = args.output_root.resolve() / f"c{args.budget}"
    output = budget_root / args.scale / edge_name(args.edge)
    record_path = output / "calibration.json"
    invocation = {"method": "history_conditioned", "variant": "map_all", "kind": KIND,
        "scale": args.scale, "edge": edge_name(args.edge), "budget": args.budget,
        "settings": cfg, "execution_sources": source_hashes, **expected,
        "reference_configuration": nonlinear_configuration(),
        "changed_settings": [] if args.budget == 128 else ["calibration_users"],
        "fit_entrypoint": "read_correction_v4.probe_nonlinear.fit",
        "uid_rule": "first C reservation UIDs for fit; next 16 for validation; original 3000-user evaluation panel",
        "evaluation_command": ["scripts/read_correction_v4/evaluate_full.py", "--scale", args.scale,
            "--edge", str(args.edge), "--calibration-dir", str(output), "--variants", "map_all"]}
    if record_path.exists():
        saved = validate_record(record_path, cfg, expected, source_hashes)
        print(json.dumps({"status": "calibration_already_complete", "path": str(record_path),
            "weights_sha256": saved["weights_sha256"]}), flush=True)
        return
    if (output / "calibration.pt").exists():
        raise RuntimeError("weights exist without a verifiable calibration record; retain the interrupted artifact")
    request_path = output / "invocation.json"
    if request_path.exists() and json.loads(request_path.read_text()) != invocation:
        raise RuntimeError("interrupted invocation has different settings, UIDs or sources")
    if args.check_only:
        print(json.dumps({"status": "inputs_checked_without_gpu", "output": str(output),
            "budget": args.budget, "train_users": len(expected["uids"]),
            "validation_users": len(expected["validation_uids"]),
            "changed_settings": invocation["changed_settings"], "settings": cfg}), flush=True)
        return
    if args.budget == 128:
        raise ValueError("C128 is retained under v4/nonlinear_probe; use --check-only rather than refit it")
    if args.gpu is None:
        raise ValueError("--gpu is required for fitting")
    output.mkdir(parents=True, exist_ok=True)
    write_json(request_path, invocation)
    started = time.perf_counter()
    old_sources = probe.sources
    try:
        # Process-local provenance hook: the frozen fitting implementation is unchanged.
        probe.sources = lambda: source_hashes
        probe.run(argparse.Namespace(scale=args.scale, edge=args.edge, gpu=args.gpu,
            output_root=budget_root, calibration_only=True), fitter=fit, config=cfg,
            artifact_kind=KIND, correction_cost=correction_forward)
    finally:
        probe.sources = old_sources
    if sources() != source_hashes:
        raise RuntimeError("execution sources changed while calibration ran")
    saved = validate_record(record_path, cfg, expected, source_hashes)
    import torch
    device = torch.device(f"cuda:{args.gpu}")
    saved["budget_extension"] = {"method": "history_conditioned", "variant": "map_all",
        "budget": args.budget, "changed_settings": ["calibration_users"],
        "reference": "scripts/read_correction_v4/probe_nonlinear.py:nonlinear_configuration",
        "invocation_sha256": sha256(request_path), "gpu": args.gpu,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
    write_json(record_path, saved)
    print(json.dumps({"status": "complete", "output": str(output),
        **saved["budget_extension"], "weights_sha256": saved["weights_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--budget", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true", help="validate configuration and UID/panel inputs; never open CUDA")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

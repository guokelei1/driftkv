#!/usr/bin/env python3
"""Fit isolated Q-feature or mixed-history candidates on causal user histories."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch
from read_correction_2026_09.cost import CostModel
from read_correction_2026_09.v2.calibrate import load_data
from read_correction_v4.probe import fit_modules
from read_correction_v4.history_conditioned.fit_nonlinear import fit_nonlinear
from read_correction_v5.common import OUTPUT, PANEL_ROOT, RESERVATIONS, configuration, edge_name, sha256, sources, write_json
from read_correction_v5.history_conditioned.capture import load_mixed_data
from read_correction_v5.query_only.fit import fit_query_features


def directory(root, method, state, feature, budget, scale, edge):
    return root / method / state / feature / f"c{budget}" / scale / edge_name(edge)


def capture_cost(rows, uids, scale, cfg):
    sparse = CostModel.for_scale(scale, cfg["attention_backend"])
    dense = CostModel.for_scale(scale, "torch")
    lengths = Counter(rows[u]["parent"].seq_len for u in uids)
    batch = cfg["capture_batches"][scale]
    one = sum((sparse if n == cfg["history_length"] else dense).full_cache(n, batch=min(batch, count-start))
        for n,count in lengths.items() for start in range(0, count, batch))
    return {"parent_full_flops": one, "teacher_full_flops": one, "append_flops": 0, "total_flops": 2*one}


def save_fit(output, modules, *, metadata, method, kind, scale, edge, budget, train,
             validation, settings, ledger, fitted, source_hashes, elapsed):
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "modules": [{"config": m.get_config(),
        "state_dict": {k:v.detach().cpu() for k,v in m.state_dict().items()}} for m in modules]}, output / "calibration.pt")
    write_json(output / "calibration.json", {**metadata, "status": "complete", "method": method, "kind": kind,
        "scale": scale, "edge": edge_name(edge), "budget": budget, "users": budget, "uids": train,
        "validation_uids": validation, "settings": settings, "cost": ledger, "fit": fitted,
        "execution_sources": source_hashes, "weights_sha256": sha256(output / "calibration.pt"), "elapsed_seconds": elapsed})


def run(args):
    cfg = {**configuration(), "state_mode": args.state_mode,
        "candidate_mode": "uniform_known" if args.method == "history_conditioned" else "mixed_recent_uniform"}
    state_name = args.state_mode
    if args.mix_profile == "old_heavy":
        if args.state_mode != "mixed":
            raise ValueError("old_heavy profile requires genuine mixed capture")
        cfg.update(mixed_append_targets=[0, 0, 256, 512], mix_profile="old_heavy")
        state_name += "_old_heavy"
    if args.query_joint_epochs:
        if args.method != "query_only":
            raise ValueError("joint query refinement is for query-only candidates")
        cfg.update(query_joint_epochs=args.query_joint_epochs, query_learning_rate=.001,
            query_output_units_rule="sqrt per-layer training target rate MSE, floor1e-12")
    budgets = sorted(set(args.budgets))
    features = args.feature_modes if args.method == "query_only" else ["nonlinear"]
    if args.state_mode == "mixed" and len(budgets) != 1:
        raise ValueError("mixed calibration uses one fixed user budget per capture")
    cfg["calibration_users"] = max(budgets)
    panel_path = PANEL_ROOT / args.scale / edge_name(args.edge) / "binding.json"
    pending = []
    for budget in budgets:
        for feature in features:
            stored_feature = feature + (f"_joint{args.query_joint_epochs}" if args.query_joint_epochs else "")
            output = directory(args.output_root, args.method, state_name, stored_feature, budget, args.scale, args.edge)
            path = output / "calibration.json"
            if path.exists():
                record = json.loads(path.read_text())
                expected = {**cfg, "calibration_users": budget, "feature_mode": feature}
                if record["settings"] != expected or record["panel_binding_sha256"] != sha256(panel_path) or record["weights_sha256"] != sha256(output / "calibration.pt"):
                    raise RuntimeError("existing candidate calibration differs")
                # Orchestration can gain new entry points; fitted operators and
                # fitting implementations must retain their original meaning.
                fit_prefixes = ("src/hstu_kvcache/", "scripts/read_correction_v5/query_only/",
                    "scripts/read_correction_v5/history_conditioned/", "scripts/read_correction_v4/history_conditioned/")
                for path, expected_hash in record["execution_sources"].items():
                    if path.startswith(fit_prefixes) and sha256(ROOT / path) != expected_hash:
                        raise RuntimeError(f"retained fitting implementation changed: {path}")
                print(json.dumps({"status": "calibration_already_complete", "output": str(output)}), flush=True)
            else:
                pending.append((budget, feature, output))
    if not pending:
        return
    started = time.perf_counter()
    source_hashes = sources()
    loader_args = argparse.Namespace(scale=args.scale, edge=args.edge, gpu=args.gpu,
        panel_root=PANEL_ROOT, reservations=RESERVATIONS, budgets=[max(budgets)+cfg["validation_users"]])
    loader = load_mixed_data if args.state_mode == "mixed" else load_data
    current, rows, uids, device, metadata = loader(loader_args, cfg)
    if args.method == "history_conditioned" and args.state_mode == "pure":
        from read_correction_2026_09.calibrate import candidates
        panel = json.loads(panel_path.read_text())
        checkpoint = torch.load(ROOT / panel["sources"]["current"]["path"], map_location="cpu", weights_only=False, mmap=True)
        dataset = json.loads((ROOT / panel["sources"]["dataset"]["path"]).read_text())
        known = int(checkpoint.get("known_vocab_size", dataset["foundation_items"]))
        del checkpoint
        for uid in uids:
            rows[uid]["candidates"] = torch.from_numpy(candidates(uid, known, cfg["calibration_queries_per_user"]))
        metadata["candidate_rule"] = "uniform known catalog without replacement; SeedSequence([17,uid]); no feedback labels"
        metadata["cache_state"] = "pure parent at prerelease snapshot; mapped state transient, never persisted"
    for budget, feature, output in pending:
        local_cfg = {**cfg, "calibration_users": budget, "feature_mode": feature}
        train, validation = uids[:budget], uids[budget:budget+cfg["validation_users"]]
        captured = metadata["capture_cost"] if args.state_mode == "mixed" else capture_cost(rows, train+validation, args.scale, cfg)
        if args.method == "query_only":
            modules, fitted = fit_query_features(current, rows, train, validation, local_cfg, device, args.scale, batch_size=8)
            ledger = {"capture": captured, "query_fitting": fitted["cost"],
                "calibration_flops": captured["total_flops"]+fitted["cost"]["calibration_flops"]}
            layers, artifact_kind = fitted, "query_features_v5"
            if args.query_joint_epochs:
                raw_output = directory(args.output_root, args.method, state_name, feature, budget, args.scale, args.edge)
                raw_settings = {k:v for k,v in local_cfg.items() if k not in (
                    "query_joint_epochs", "query_learning_rate", "query_output_units_rule")}
                if not (raw_output / "calibration.json").exists():
                    save_fit(raw_output, modules, metadata=metadata, method=args.method, kind=artifact_kind,
                        scale=args.scale, edge=args.edge, budget=budget, train=train, validation=validation,
                        settings=raw_settings, ledger=ledger, fitted=fitted, source_hashes=source_hashes,
                        elapsed=time.perf_counter()-started)
                from read_correction_v5.query_only.refine import refine_query
                units = [max(1e-12, math.sqrt(layer["diagnostics"]["train"]["baseline_rate_mse"])) for layer in fitted["layers"]]
                modules, refined = refine_query(current, rows, train, validation, modules,
                    {**local_cfg, "output_units": units}, device, args.scale, batch_size=4)
                layers = {"ridge": fitted, "joint_refinement": refined}
                ledger = {"ridge_and_capture": ledger, "refinement": refined["cost"],
                    "calibration_flops": ledger["calibration_flops"]+refined["cost"]["calibration_flops"]}
        else:
            base, base_layers, base_cost = fit_modules(current, rows, train, validation, local_cfg, device, args.scale)
            original_capture = base_cost.pop("parent_and_teacher_cache_flops")
            base_cost["capture"] = captured
            base_cost["calibration_flops"] += captured["total_flops"] - original_capture
            modules, fitted = fit_nonlinear(current, rows, train, validation, base,
                config=local_cfg, device=device, scale=args.scale, batch_size=8)
            fitted["cache_state"] = metadata["cache_state"]
            ledger = {"affine_and_capture": base_cost, "nonlinear": fitted["cost"],
                "calibration_flops": base_cost["calibration_flops"]+fitted["cost"]["calibration_flops"]}
            layers, artifact_kind = {"affine": base_layers, "nonlinear": fitted}, "token_read_nonlinear_v4"
            del base
        save_fit(output, modules, metadata=metadata, method=args.method, kind=artifact_kind,
            scale=args.scale, edge=args.edge, budget=budget, train=train, validation=validation,
            settings=local_cfg, ledger=ledger, fitted=layers, source_hashes=source_hashes,
            elapsed=time.perf_counter()-started)
        print(json.dumps({"status": "calibration_complete", "method": args.method, "feature": feature,
            "budget": budget, "output": str(output), "calibration_flops": ledger["calibration_flops"]}), flush=True)
        del modules, fitted, layers
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1,6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--method", choices=("query_only", "history_conditioned"), required=True)
    parser.add_argument("--state-mode", choices=("pure", "mixed"), required=True)
    parser.add_argument("--feature-modes", nargs="+", choices=("head_phi", "cross_phi"), default=["head_phi", "cross_phi"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[128])
    parser.add_argument("--mix-profile", choices=("balanced", "old_heavy"), default="balanced")
    parser.add_argument("--query-joint-epochs", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    run(parser.parse_args())

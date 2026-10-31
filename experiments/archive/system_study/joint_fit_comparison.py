#!/usr/bin/env python3
"""One local fit, then original/grouped joint fits from the same V4 endpoint.

Execution comparison on the original calibration roles. Experimental weights
are isolated; the frozen candidate and evaluation population are not modified.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]

import torch
from hstu_kvcache.evaluation.release_context import RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.calibration import (
    attach_rollout_tail, checkpoint, fit_local, fit_joint, gate_statistics, initialize_continuation,
    prepare, priced_source_capture,
)
from hstu_kvcache.methods.cachetide.calibration_cost import OperationCounts, warmstart_origin_cost
from hstu_kvcache.methods.cachetide.features import load_bank
from experiments.archive.system_study.joint_batch import fit_joint_batched
from experiments.archive.system_study.joint_batch_probe import difference
from hstu_kvcache.common.artifacts import sha256, write_json


def run(cli):
    began = time.perf_counter()
    config = json.loads(cli.config.read_text())
    phases, device = [], None

    @contextmanager
    def phase(name, branch="common", scope=""):
        if device is not None:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        yield
        if device is not None:
            torch.cuda.synchronize(device)
        record = dict(name=name, branch=branch, wall_seconds=time.perf_counter()-started,
                      scope=scope)
        phases.append(record)
        write_json(cli.output / "phases.json", phases)
        print(json.dumps(dict(status="phase", **record)), flush=True)

    with phase("configuration_and_binding_checks"):
        metadata_path = ROOT / config["calibration_metadata"]
        metadata = json.loads(metadata_path.read_text())
        panel_path = ROOT / config["panel"]
        panel, roles, _ = load_panel(panel_path)
        train, validation = metadata["fit_uids"], metadata["validation_uids"]
        if train != roles["fit"][:64] or validation != roles["validation"][:16]:
            raise ValueError("original calibration role/order changed")
        if (set(train) | set(validation)) & set(roles["evaluation"]):
            raise ValueError("calibration/evaluation intersection")
        for key in ("target", "seed", "local_epochs", "joint_epochs", "batch_size",
                    "learning_rate", "rollout_steps", "release_initialization"):
            if config[key] != metadata[key]:
                raise ValueError(f"original recipe changed: {key}")
        origin_path, source_path = ROOT / metadata["origin"], ROOT / metadata["source_pack"]
        for path, digest in ((origin_path, metadata["origin_sha256"]),
                             (source_path, metadata["source_pack_sha256"]),
                             (panel_path / "binding.json", metadata["panel_sha256"])):
            if sha256(path) != digest:
                raise ValueError(f"original binding changed: {path}")
        settings = dict(config=config, pid=os.getpid(), fit_uids=train, validation_uids=validation,
            origin=str(origin_path), origin_sha256=metadata["origin_sha256"],
            source_pack=str(source_path), source_pack_sha256=metadata["source_pack_sha256"],
            original_calibration_metadata=str(metadata_path), metadata_sha256=sha256(metadata_path),
            sources={str(path.relative_to(ROOT)):sha256(path) for path in (
                Path(__file__), ROOT / "experiments/archive/system_study/joint_batch.py",
                ROOT / "src/hstu_kvcache/methods/cachetide/calibration.py")},
            scope="V4 calibration execution only; local once, two joint branches from identical endpoint. No evaluation or candidate replacement.")
        write_json(cli.output / "configuration.json", settings)
        args = SimpleNamespace(**{key:metadata[key] for key in (
            "seed", "local_epochs", "joint_epochs", "batch_size", "learning_rate", "rollout_steps")})
        torch.manual_seed(args.seed)
    with phase("device_context"):
        device = setup_device(cli.gpu, config["threads"])
    with phase("foundation_model_load", scope="Original loader loads V2/V3/V4; includes CPU reads, hashes, construction and H2D. Frozen models, no backbone training."):
        models = load_models(panel, device, 4)
        model = models["v4"]
    with phase("origin_load_and_carry_initialize"):
        origin = torch.load(origin_path, map_location="cpu", weights_only=False)
        adapter = UnifiedRelayAdapter(**origin["config"]).to(device)
        adapter.load_state_dict(origin["state_dict"])
        initialize_continuation(adapter, metadata["release_initialization"])
        adapter.set_gate_source(metadata["gate_source"])
        adapter.set_gate_normalization(metadata["gate_normalization"])
        adapter.set_read_width(metadata["read_shared"], metadata["read_private"])
        adapter.eval()
        if adapter.mode != metadata["mode"] or metadata["gate_only"] or metadata["freeze_gates"]:
            raise ValueError("this probe requires the original full-parameter A recipe")
    with phase("source_pack_load_and_tail", scope="Existing source is loaded, not regenerated; includes causal history load and four-event pre-release tail attachment."):
        histories = load_history(panel_path, requested_uids=train+validation)
        pack = torch.load(source_path, map_location="cpu", weights_only=False)
        if (pack["settings"]["target"] != "v4" or
            pack["prior_checkpoints"]["v3"]["sha256"] != metadata["origin_sha256"]):
            raise ValueError("source predecessor binding changed")
        captured = {uid:pack["rows"][uid] for uid in train+validation}
        tail_records = attach_rollout_tail(captured, histories, dict(RELEASES)["v4"],
            args.rollout_steps, adapter.max_length, pack["records"])
        capture_costs = priced_source_capture(pack["costs"], pack)
        del histories, pack, models
    common_operations = OperationCounts()
    common_operations.record_native_capture(capture_costs, source_available=True)
    with phase("source_h2d_and_teacher_query_prepare"):
        rows = prepare(model, captured, adapter, device, common_operations)
        del captured
    with phase("immutable_bank_load_and_h2d", scope="Existing fixed bank, hash checked and transferred once; no bank generation. Separated from original fit_local first-use load."):
        load_bank(adapter.feature_bank_path, adapter.feature_bank_sha256,
                  device=device, dtype=next(adapter.parameters()).dtype)
    common_costs = Counter()
    with phase("local_fit"):
        local = fit_local(model, adapter, rows, train, validation, args,
                          common_costs, common_operations)
    with phase("local_snapshot", "experiment_only", "Fork/snapshot cost solely for this comparison; not charged to either single-branch deployment."):
        local_state = {name:value.detach().cpu().clone() for name,value in adapter.state_dict().items()}
        torch.save(dict(config=adapter.get_config(), state_dict=local_state), cli.output / "local_endpoint.pt")
        local_hash = sha256(cli.output / "local_endpoint.pt")
    reports, final_states = {}, {}
    trainable_names = {name for name,p in adapter.named_parameters() if p.requires_grad}
    common_seconds = sum(p["wall_seconds"] for p in phases if p["branch"] == "common")
    for name, fitter in (("serial", fit_joint), ("batched", fit_joint_batched)):
        output = cli.output / name
        output.mkdir()
        with phase("restore_identical_endpoint", "experiment_only", name):
            adapter.load_state_dict(local_state)
            adapter.zero_grad(set_to_none=True)
            torch.manual_seed(args.seed)
            if any(not torch.equal(v.detach().cpu(), local_state[k]) for k,v in adapter.state_dict().items()):
                raise AssertionError("joint branch did not start from identical local state")
        costs, operations = Counter(), OperationCounts()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        with phase("joint_fit", name, "Includes initial+four epoch validations, independent fresh Adam, original seed/order/8-user boundaries; no warmup fit."):
            joint = fitter(model, adapter, rows, train, validation, args, costs, operations)
        joint_peak = torch.cuda.max_memory_allocated(device)
        with phase("post_gate_statistics", name):
            means, std = gate_statistics(model, adapter, rows, train, operations)
        total_operations = copy.deepcopy(common_operations)
        total_operations.components.update(operations.components)
        total_operations.calls.update(operations.calls)
        total_costs = common_costs + costs
        cost = total_operations.report(origin=warmstart_origin_cost(origin_path))
        diagnostics = dict(local=local, joint=joint, gate_mean=means.tolist(), gate_std=std.tolist(),
            gate_std_max=float(std.max()), operation_counts=dict(total_costs),
            capture_costs=capture_costs, rollout_records=tail_records,
            local_endpoint_sha256=local_hash, experimental_weights_not_selected=True)
        with phase("checkpoint_save_and_hash", name, "New experimental adapter artifact, CPU copy, save and hash; no fsync guarantee."):
            checkpoint(adapter, output / "calibration.pt", settings, diagnostics, means, cost)
        with phase("final_state_copy", "experiment_only", name):
            final_states[name] = {k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        reports[name] = dict(joint=joint, logical_joint_counts=dict(costs),
            joint_and_post_operations=operations.report(origin={}),
            deployment_operations=cost, joint_peak_allocated_bytes=joint_peak,
            joint_incremental_peak_bytes=joint_peak-baseline_memory,
            recorded_calibration_seconds=common_seconds + sum(p["wall_seconds"] for p in phases if p["branch"]==name),
            checkpoint=str(output / "calibration.pt"))
        write_json(output / "report.json", reports[name])
    with phase("comparison", "experiment_only"):
        reference, candidate = final_states["serial"], final_states["batched"]
        per_parameter = {name:difference(reference[name],candidate[name]) for name in trainable_names}
        flat_ref = torch.cat([reference[n].flatten() for n in sorted(trainable_names)])
        flat_got = torch.cat([candidate[n].flatten() for n in sorted(trainable_names)])
        parameter_diff = difference(flat_ref,flat_got)
        frozen_equal = all(torch.equal(reference[n],candidate[n]) for n in reference if n not in trainable_names)
        ledger_equal = (reports["serial"]["logical_joint_counts"] == reports["batched"]["logical_joint_counts"]
            and reports["serial"]["joint_and_post_operations"] == reports["batched"]["joint_and_post_operations"])
        backbone_clean = all(p.grad is None for p in model.parameters())
        write_json(cli.output / "parameter_differences.json", dict(trainable=parameter_diff,
            per_parameter=per_parameter, frozen_state_equal=frozen_equal))
        if not ledger_equal or not frozen_equal or not backbone_clean:
            raise AssertionError(dict(ledger_equal=ledger_equal,frozen_equal=frozen_equal,backbone_clean=backbone_clean))
    summary = dict(status="complete", phases=phases, branches=reports,
        trainable_parameter_difference=parameter_diff, logical_operation_ledger_equal=ledger_equal,
        frozen_state_equal=frozen_equal, backbone_gradients_absent=backbone_clean,
        selected_epochs={k:r["joint"]["selected_epoch"] for k,r in reports.items()},
        joint_speed_ratio=next(p["wall_seconds"] for p in phases if p["branch"]=="serial" and p["name"]=="joint_fit")/
            next(p["wall_seconds"] for p in phases if p["branch"]=="batched" and p["name"]=="joint_fit"),
        whole_experiment_seconds=time.perf_counter()-began,
        source_generation=dict(rerun=False, historical_wall_seconds=33.801780375652015,
            evidence="results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/source_v4/summary.json",
            scope="Prior causal source+teacher capture workflow. Retain separately in complete S; current source load cannot replace generation."),
        scope="Single V4 execution comparison, one observed run per branch, serial then batched. Includes input loads/CPU-GPU moves in phases, no cold-I/O or variance guarantee. Local and fixed-bank load shared once. Neither two branches nor experiment-only fork/check costs constitute one deployment.",
        complete_S_boundary="Can replace only the measured V4 calibration workflow in S. Origin/local32, other releases, external source generation, fresh bank/publication generation/distribution remain separate. No 30k serving, quality evaluation, or candidate promotion.")
    write_json(cli.output / "summary.json", summary)
    print(json.dumps(dict(status="complete", selected_epochs=summary["selected_epochs"],
        joint_speed_ratio=summary["joint_speed_ratio"], parameters=parameter_diff,
        calibration_seconds={k:r["recorded_calibration_seconds"] for k,r in reports.items()})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    sys.stdout = sys.stderr = (cli.output / "run.log").open("a", buffering=1)
    try:
        run(cli)
    except BaseException as error:
        write_json(cli.output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
        raise

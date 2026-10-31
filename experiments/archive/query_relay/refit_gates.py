#!/usr/bin/env python3
"""Gate-only fitting of within-candidate-set teacher score differences.

The trained code updater, decoders and frozen recommendation model remain
fixed. Both static and corrected-write rollout views use the same centered
score MSE; no variance regularizer, future request labels or extra target is
introduced. A later release requires this branch's own predecessor trajectory.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import torch

from hstu_kvcache.evaluation.capture import capture_rows
from hstu_kvcache.evaluation.release_context import RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter, operation_gate_query
from hstu_kvcache.methods.cachetide.calibration import (
    attach_rollout_tail, checkpoint, gate_statistics, prepare, priced_source_capture, rollout_score,
)
from hstu_kvcache.methods.cachetide.calibration_cost import OperationCounts
from hstu_kvcache.common.artifacts import sha256, write_json

PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def center(scores):
    return scores - scores.mean(-1, keepdim=True)


def centered_mse(student, teacher):
    return (center(student) - center(teacher)).square().mean()


def freeze_except_gates(adapter):
    adapter.requires_grad_(False)
    for layer in adapter.layers:
        layer.key_gate_weight.requires_grad_(True)
        layer.value_gate_weight.requires_grad_(True)
    return [parameter for parameter in adapter.parameters() if parameter.requires_grad]


def fit_centered(model, adapter, rows, train, validation, args, operations):
    parameters = freeze_except_gates(adapter)
    adapter.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    # The static updater has no trainable parameters or gate-dependent inputs.
    # Prepare its codes once instead of re-running it in every refit epoch.
    with torch.no_grad():
        for row in rows.values():
            row["fixed_state"] = adapter.advance_codes(row["cache"], row["previous"],
                row["item"], item_ids=row.get("item_ids"))
            row["centered_target"] = center(row["wanted_logits"])
            operations.record_update(adapter, length=row["cache"].seq_len,
                                      stage="refit_static_prepare")
            operations.add("refit_target_center_flops", 2 * row["wanted_logits"].numel() + 1)
    costs = Counter()

    def forward(uid, backward=False):
        row = rows[uid]
        static = score(model, row["cache"], row["ids"], row["delta"],
            history_read_fn=adapter.make_history_read(row["fixed_state"],
                gate_query=operation_gate_query(adapter, model, row["ids"])))[0]
        operations.record_gate_item_input(adapter, items=row["ids"].numel(), stage="refit_static_query")
        operations.record_query(adapter, length=row["cache"].seq_len,
            queries=row["ids"].numel(), backward=backward, stage="refit_static_query")
        operations.record_score_loss(outputs=static.numel(), backward=backward, stage="refit_static")
        operations.add("refit_static_student_center_flops", (4 if backward else 2) * static.numel() + 2)
        static_loss = (center(static) - row["centered_target"]).square().mean()
        rolled_loss = None
        rolled = None
        if args.rollout_steps and row["rollout"]["items"].numel():
            rolled, _, _ = rollout_score(model, adapter, row, costs=costs,
                                          operations=operations, backward=backward)
            rolled_loss = (center(rolled) - row["centered_target"]).square().mean()
            operations.add("refit_rollout_student_center_flops", (4 if backward else 2) * rolled.numel() + 2)
            operations.add("refit_two_view_combination_flops", 3)
        loss = static_loss if rolled_loss is None else .5 * static_loss + .5 * rolled_loss
        raw = (static - row["wanted_logits"]).square().mean()
        if rolled is not None:
            raw = .5 * raw + .5 * (rolled - row["wanted_logits"]).square().mean()
        operations.add("refit_raw_mse_diagnostic_flops", 3 * static.numel() * (2 if rolled is not None else 1))
        costs["refit_static_forward_calls"] += 1
        return loss, static_loss, rolled_loss, raw

    @torch.no_grad()
    def validate():
        values = [forward(uid) for uid in validation]
        rolled = [row[2] for row in values if row[2] is not None]
        return dict(validation_centered_logit_mse=float(torch.stack([row[0] for row in values]).mean()),
            validation_static_centered_logit_mse=float(torch.stack([row[1] for row in values]).mean()),
            validation_rollout_centered_logit_mse=None if not rolled else float(torch.stack(rolled).mean()),
            validation_raw_logit_mse=float(torch.stack([row[3] for row in values]).mean()))

    initial = validate()
    best, best_epoch = initial["validation_centered_logit_mse"], 0
    selected = copy.deepcopy(adapter.state_dict())
    history = [dict(epoch=0, **initial)]
    first_gradients = None
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            subset = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in subset:
                loss = forward(train[index], backward=True)[0] / len(subset)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite centered gate loss")
                loss.backward()
                costs["refit_backward_calls"] += 1
            if first_gradients is None:
                first_gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                    for name, parameter in adapter.named_parameters() if parameter.requires_grad}
                operations.record_gradient_diagnostic(parameters=sum(p.numel() for p in parameters),
                                                       tensors=len(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            operations.record_optimizer(parameters=sum(p.numel() for p in parameters), stage="refit")
        metrics = validate()
        history.append(dict(epoch=epoch, **metrics))
        if metrics["validation_centered_logit_mse"] < best:
            best, best_epoch = metrics["validation_centered_logit_mse"], epoch
            selected = copy.deepcopy(adapter.state_dict())
        emit(dict(status="centered_gate_epoch", epoch=epoch, selected_epoch=best_epoch, **metrics))
    adapter.load_state_dict(selected)
    return dict(selected_epoch=best_epoch, history=history, first_gradients=first_gradients,
        operation_counts=dict(costs), selection_metric="heldout centered score MSE",
        objective="Within each user's fixed candidate set, student and teacher each subtract their own mean.",
        views="half_static_half_rollout" if args.rollout_steps else "static_only")


def cumulative_cost(parent, stage, parent_path, parent_hash):
    """Same-release refit must retain its earlier local/joint fitting expense."""
    if not parent or "calibration_flops" not in parent:
        raise ValueError("parent checkpoint lacks calibration cost evidence")
    result = dict(stage)
    result["calibration_flops"] = int(parent["calibration_flops"]) + int(stage["calibration_flops"])
    prior_with_origin = parent.get("calibration_with_initial_origin_flops")
    result["calibration_with_initial_origin_flops"] = None if prior_with_origin is None else \
        int(prior_with_origin) + int(stage["calibration_flops"])
    prior_source = parent.get("source_reconstruction_sensitivity", {}).get("additional_flops", 0)
    result["source_reconstruction_sensitivity"] = dict(
        additional_flops=int(prior_source) + stage["source_reconstruction_sensitivity"]["additional_flops"],
        scope="Parent calibration and refit reconstruction sensitivities; required work remains in primary.")
    result["warmstart_origin"] = parent.get("warmstart_origin")
    result["parent_calibration"] = dict(path=str(parent_path), sha256=parent_hash,
        calibration_flops=int(parent["calibration_flops"]), cost=parent)
    result["refit_stage_cost"] = stage
    result["scope"] = "Parent's same-release local/joint fit plus this gate-only refit; previous releases paid separately in chain."
    return result


def run(args):
    began = time.perf_counter()
    torch.manual_seed(args.seed)
    panel, groups, _ = load_panel(args.panel)
    train = list(groups["fit"][:args.fit_users])
    validation = list(groups["validation"][:args.validation_users])
    if not train or not validation or set(train) & set(validation) or \
            (set(train) | set(validation)) & set(groups["evaluation"]):
        raise ValueError("refit requires disjoint frozen fit/validation roles")
    payload = torch.load(args.origin, map_location="cpu", weights_only=False)
    metadata = json.loads(args.origin.with_suffix(".json").read_text())
    if payload["kind"] != UnifiedRelayAdapter.kind:
        raise ValueError("refit requires an already fitted unified checkpoint")
    target = args.target or metadata["target"]
    if metadata["target"] != target:
        raise ValueError("parent checkpoint belongs to a different target release")
    if set(train) - set(metadata["fit_uids"]) or set(validation) - set(metadata["validation_uids"]):
        raise ValueError("refit users must retain their parent fit/validation roles")
    pack = None
    previous = args.previous_checkpoint
    if target != "v3":
        if args.source_pack is None:
            raise ValueError("later gate refit requires its own corrected source trajectory")
        if previous is None:
            previous = args.output.parent / f"v{int(target[1:]) - 1}" / "calibration.pt"
        pack = torch.load(args.source_pack, map_location="cpu", weights_only=False)
        if pack.get("settings", {}).get("target") != target:
            raise ValueError("source pack belongs to a different release")
        predecessor = f"v{int(target[1:]) - 1}"
        recorded = pack.get("prior_checkpoints", {}).get(predecessor, {}).get("sha256")
        if recorded != sha256(previous):
            raise ValueError("source pack does not use this branch's predecessor checkpoint")
    elif args.source_pack is not None:
        raise ValueError("the first gate-refit release uses the canonical V2 source")
    args.output.mkdir(parents=True, exist_ok=False)
    origin_hash = sha256(args.origin)
    settings = dict(target=target, mode=payload["config"]["mode"], seed=args.seed,
        fit_uids=train, validation_uids=validation, role="pre_release_gate_only_refit",
        origin=str(args.origin), origin_sha256=origin_hash,
        parent_checkpoint=str(args.origin), parent_checkpoint_sha256=origin_hash,
        previous_checkpoint=None if previous is None else str(previous),
        previous_checkpoint_sha256=None if previous is None else sha256(previous),
        panel_sha256=sha256(args.panel / "binding.json"),
        source_pack=None if args.source_pack is None else str(args.source_pack),
        source_pack_sha256=None if args.source_pack is None else sha256(args.source_pack),
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
        rollout_steps=args.rollout_steps, selection_metric="heldout centered score MSE",
        trainable="query key/value gates only; updater, decoder, backbone frozen",
        objective="same final-score alignment using per-user candidate-set centered logits",
        objectives_added=0, score_views="half_static_half_rollout" if args.rollout_steps else "static_only",
        no_future_events_or_request_labels=True, source_state_available=not args.require_source_replay,
        execution_sources={str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), ROOT / "src/hstu_kvcache/methods/cachetide/calibration.py",
             ROOT / "src/hstu_kvcache/methods/cachetide/adapter.py", ROOT / "src/hstu_kvcache/methods/cachetide/session.py",
             ROOT / "src/hstu_kvcache/methods/cachetide/calibration_cost.py")})
    write_json(args.output / "configuration.json", settings)
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device, last_version=int(target[1:]))
    model = models[target]
    adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval()
    adapter.load_state_dict(payload["state_dict"])
    histories = load_history(args.panel, requested_uids=train + validation)
    if pack is None:
        captured, capture_costs, capture_records = capture_rows(models, histories, train + validation,
                                                               target, panel["known_vocab_size"])
    else:
        captured = {uid: pack["rows"][uid] for uid in train + validation}
        capture_costs, capture_records = pack["costs"], pack["records"]
    tails = attach_rollout_tail(captured, histories, dict(RELEASES)[target], args.rollout_steps,
                               adapter.max_length, capture_records)
    capture_costs = priced_source_capture(capture_costs, pack)
    del models, pack
    operations = OperationCounts()
    operations.record_native_capture(capture_costs, source_available=not args.require_source_replay)
    rows = prepare(model, captured, adapter, device, operations)
    del captured
    diagnostics = fit_centered(model, adapter, rows, train, validation, args, operations)
    means, std = gate_statistics(model, adapter, rows, train, operations)
    diagnostics.update(gate_mean=means.tolist(), gate_std=std.tolist(), gate_std_max=float(std.max()),
        capture_records=capture_records, capture_costs=capture_costs, rollout_records=tails,
        backbone_gradients_absent=all(parameter.grad is None for parameter in model.parameters()),
        non_gate_gradients_absent=all(parameter.grad is None for name, parameter in adapter.named_parameters()
                                    if "gate_weight" not in name),
        elapsed_seconds=time.perf_counter() - began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    stage_cost = operations.report()
    cost = cumulative_cost(metadata.get("cost"), stage_cost, args.origin, origin_hash)
    checkpoint(adapter, args.output / "calibration.pt", settings, diagnostics, means, cost)
    emit(dict(status="complete", output=str(args.output), selected_epoch=diagnostics["selected_epoch"],
              centered_mse=diagnostics["history"][diagnostics["selected_epoch"]]["validation_centered_logit_mse"],
              calibration_flops=cost["calibration_flops"]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--origin", type=Path, required=True, help="Already fitted checkpoint at the target release")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--target", choices=("v3", "v4", "v5"), default=None)
    p.add_argument("--source-pack", type=Path)
    p.add_argument("--previous-checkpoint", type=Path,
                   help="Own branch predecessor; defaults to sibling vN/calibration.pt")
    p.add_argument("--fit-users", type=int, default=64)
    p.add_argument("--validation-users", type=int, default=16)
    p.add_argument("--epochs", type=int, choices=(1, 8, 16), default=8)
    p.add_argument("--rollout-steps", type=int, choices=(0, 4), default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=.00025)
    p.add_argument("--require-source-replay", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

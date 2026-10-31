"""CPU correction of the retained full-item fgh calibration FLOPs ledger.

The affected reference score shares normalized rows between f and h, while
forward_flops charges that normalization twice. Keep the checkpoint untouched
and subtract only the duplicated term, including the declared 2x-forward
backward estimate. This does not discount any executed training or validation.
"""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import torch

from hstu_kvcache.common.artifacts import sha256

ROOT = Path(__file__).resolve().parents[3]
CORE_SOURCE = "experiments/ch3_motivation/code/probes.py"
AFFECTED_CORE_SHA256 = "6ab7246f3f4d2d2867ca3e69f2db85f8513b1ff6ae3d3983cf5c76aac2ede934"
VARIANT_SOURCE = "experiments/ch3_motivation/code/probe_variants.py"
AFFECTED_VARIANT_SHA256 = "a251cf74b39f524440edd620aa462b3a1fe6f55a5694c0a3e4a6e04c5e4db607"
LABEL_STAGES = dict(f="f", fg="fg", fgh="fgh", f_extended="f", fg_extended="fg")


# Original audit plus the import-only relocation; unknown future implementations
# must still fail the fixed source gate. Historical checkpoint records stay intact.
RELOCATED_VARIANT_SHA256 = "b54fe13c01e28a40dddec66c91358c561dfad17bdd2b098bcef7286d05585fef"


def audited_sources(sources):
    current = (sources.get(CORE_SOURCE), sources.get(VARIANT_SOURCE))
    original = (sources.get("scripts/ch3_motivation/probes.py"),
                sources.get("scripts/ch3_motivation/probe_variants.py"))
    return current in {
        (AFFECTED_CORE_SHA256, AFFECTED_VARIANT_SHA256),
        (AFFECTED_CORE_SHA256, RELOCATED_VARIANT_SHA256),
    } or original == (AFFECTED_CORE_SHA256, AFFECTED_VARIANT_SHA256)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"missing or invalid nonnegative integer {name}")
    return value


def correction_record(payload):
    """Return raw/deduction/corrected costs from one actual typed payload.

    No missing row count is inferred from steps, Q, or an average history.
    The source SHA prevents applying this correction to a future fixed ledger.
    """
    metadata, config = payload["metadata"], payload["config"]
    identity = metadata["identity"]
    label, stage = metadata["method_label"], payload["stage"]
    if (payload.get("format") != "ch3_probe_variant_v1" or label not in LABEL_STAGES or
            LABEL_STAGES[label] != stage or metadata["stage"] != stage):
        raise ValueError("correction requires the actual typed probe stage/branch")
    raw = _integer(metadata.get("cumulative_calibration_flops_estimate"), "raw cumulative FLOPs")
    own = _integer(metadata.get("total_calibration_flops_estimate"), "raw stage FLOPs")
    if own > raw:
        raise ValueError("raw stage cost exceeds its cumulative cost")
    deductions = dict(training_forward=0, validation_forward=0, training_backward_estimate=0)
    unit = 0
    if label == "fgh":
        if payload.get("variant") != "full_item":
            raise ValueError("this correction has been audited for full_item fgh only")
        sources = identity.get("numerical_sources", {})
        if not audited_sources(sources):
            raise ValueError("fgh calibration sources are not the audited affected core/variant SHA256")
        counts = metadata.get("operation_counts", {})
        train_rows = _integer(counts.get("training_probe_rows"), "training_probe_rows")
        val_rows = _integer(counts.get("validation_probe_rows"), "validation_probe_rows")
        steps = _integer(metadata.get("completed_steps"), "completed_steps")
        for key in ("training_sample_calls", "training_backward_calls"):
            if _integer(counts.get(key), key) != steps:
                raise ValueError("training calls do not match the declared per-user fitting steps")
        for key in ("num_layers", "num_heads", "head_dim", "item_dim"):
            if _integer(config.get(key), key) == 0:
                raise ValueError("positive actual probe dimensions are required")
        if type(config.get("normalize")) is not bool or type(config.get("include_item")) is not bool:
            raise ValueError("actual normalization/item configuration is required")
        row_dim = 2*config["head_dim"] + (config["item_dim"] if config["include_item"] else 0)
        unit = 2*row_dim*config["num_layers"]*config["num_heads"] if config["normalize"] else 0
        deductions = dict(training_forward=unit*train_rows, validation_forward=unit*val_rows,
                          training_backward_estimate=2*unit*train_rows)
    deduction = sum(deductions.values())
    if deduction > own:
        raise ValueError("normalization deduction exceeds the raw stage cost")
    return dict(format="probe_calibration_cost_correction_v1", edge=identity["edge"], seed=identity["seed"],
        method_label=label, variant=payload.get("variant"), config=dict(config),
        raw_cumulative_calibration_flops_estimate=raw, raw_stage_calibration_flops_estimate=own,
        shared_row_normalization_deduction_flops=deduction,
        deduction_components_flops=deductions, duplicated_forward_flops_per_probe_row=unit,
        corrected_cumulative_calibration_flops_estimate=raw-deduction,
        corrected_stage_calibration_flops_estimate=own-deduction,
        formula="2*row_dim*layers*heads*(3*training_probe_rows+validation_probe_rows) for normalized full_item fgh; other listed branches=0",
        backward_convention="retain the original backward estimate of 2x student forward",
        fitting_identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        fitting_identity=identity,
        correction_source_sha256={str(Path(__file__).relative_to(ROOT)):sha256(Path(__file__)),
            "src/hstu_kvcache/common/artifacts.py":sha256(ROOT/"src/hstu_kvcache/common/artifacts.py")},
        original_metadata_and_weights_modified=False)


def checkpoint_correction(path, *, expected_sha256=None):
    """Read a frozen checkpoint on CPU; never construct a probe or use a GPU."""
    path = Path(path).resolve()
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("calibration checkpoint SHA256 differs")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    record = correction_record(payload)
    record.update(checkpoint=str(path), checkpoint_sha256=digest)
    return record


def apply_to_costs(costs, correction):
    """Return explicit raw/corrected E5-shaped accounts without mutating input.

    E6's already separated accounts can be joined as
    ``dict(actual_event, candidate_reads_230=analytic230)`` for this call.
    Deployment preparation/read/writer/setup terms and both denominators stay
    unchanged. A mismatched or already corrected calibration field is refused.
    """
    raw = correction["raw_cumulative_calibration_flops_estimate"]
    deduction = correction["shared_row_normalization_deduction_flops"]
    corrected = copy.deepcopy(costs)
    for account in (corrected, corrected["candidate_reads_230"]):
        if account["calibration_flops"] != raw:
            raise ValueError("cost account calibration does not match the raw checkpoint ledger")
        total = _integer(account["complete_added_flops"], "complete added FLOPs")
        denominator = _integer(account["once_rebuild_flops"], "once Rebuild FLOPs")
        if total < deduction or denominator == 0:
            raise ValueError("invalid complete cost or Rebuild denominator")
        account.update(calibration_flops=raw-deduction, complete_added_flops=total-deduction,
            percent_of_once_rebuild=100*(total-deduction)/denominator)
    # The optional projected/raw execution comparison also contains the same
    # fixed calibration fee, so retain its execution delta after subtraction.
    if "comparison_to_raw_graph" in corrected:
        comparison = corrected["comparison_to_raw_graph"]
        for key in ("complete_added_flops", "normalized_230_complete_added_flops"):
            comparison[key] = _integer(comparison[key], key)-deduction
            if comparison[key] < 0:
                raise ValueError("raw graph comparison total is smaller than the deduction")
    return dict(raw=copy.deepcopy(costs), corrected=corrected, calibration_correction=copy.deepcopy(correction))


def audit_checkpoints(paths, rebuild_summaries):
    """Audit the actual three-stage ladders and fixed2000 reference costs.

    This is an analytical preparation record, not quality evidence. Resident
    thresholds retain the observed average history/first-consumer rate and
    omit every deployment term, so they are only calibration-only lower bounds.
    """
    baselines = {}
    for path in rebuild_summaries:
        path = Path(path).resolve()
        summary = json.loads(path.read_text())
        costs = summary["costs"]["actual_event_ledger"]
        edge = summary["identity"]["settings"]["edge"]
        if (costs["users"] != 2000 or edge in baselines or
                summary["edge"] != f"V{edge-1}->V{edge}"):
            raise ValueError("require one unchanged2000-user Rebuild reference per edge")
        baselines[edge] = dict(path=str(path), sha256=sha256(path),
            users=2000, once_rebuild_flops=costs["once_rebuild_flops"],
            models=summary["identity"]["checkpoint_sha256"])
    rows, chains = [], {}
    for path in paths:
        row = checkpoint_correction(path)
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        metadata = payload["metadata"]
        counts, arithmetic = metadata["operation_counts"], metadata["arithmetic"]
        if sum(arithmetic.values()) != row["raw_stage_calibration_flops_estimate"]:
            raise ValueError("raw stage arithmetic does not sum to the stored total")
        row["raw_stage_arithmetic"] = dict(arithmetic)
        row["standalone_shared_preparation_flops"] = _integer(
            metadata.get("source_teacher_and_target_preparation_flops"), "shared preparation FLOPs")
        row["calibration_preparation"] = metadata["calibration_preparation"]
        rounds, users = len(metadata["history"]), metadata["validation_users"]
        if (counts["validation_sample_calls"] != rounds*users or
                counts["training_sample_calls"] != metadata["completed_steps"]):
            raise ValueError("executed training/validation counters differ from recorded history")
        val = sum(value for key,value in arithmetic.items() if key.startswith("validation_"))
        row["executed_validation"] = dict(rounds=rounds, users_per_round=users,
            validation_every_steps=metadata["identity"]["validation_every"], includes_step0=True,
            validation_sample_calls=counts["validation_sample_calls"],
            validation_native_candidate_queries=counts["validation_native_queries"],
            training_sample_calls=counts["training_sample_calls"], completed_steps=metadata["completed_steps"],
            best_step=metadata["best_step"], raw_validation_forward_flops=val,
            corrected_validation_forward_flops=val-row["deduction_components_flops"]["validation_forward"],
            early_stop_work_retained=True)
        baseline = baselines[row["edge"]]
        if baseline["models"] != row["fitting_identity"]["models"]:
            raise ValueError("Rebuild reference and fitted probe model identities differ")
        denominator = _integer(baseline["once_rebuild_flops"], "once Rebuild FLOPs")
        if denominator == 0:
            raise ValueError("positive fixed2000 Rebuild reference is required")
        corrected = row["corrected_cumulative_calibration_flops_estimate"]
        ceil_ratio = lambda numerator,denominator:(numerator+denominator-1)//denominator
        row["linear_amortization_lower_bound"] = dict(reference=baseline,
            raw_calibration_percent_of_fixed2000_rebuild=100*row["raw_cumulative_calibration_flops_estimate"]/denominator,
            corrected_calibration_percent_of_fixed2000_rebuild=100*corrected/denominator,
            residents_for_calibration_100_percent=ceil_ratio(2000*corrected,denominator),
            residents_for_calibration_20_percent=ceil_ratio(10000*corrected,denominator),
            assumption="fixed calibration; resident-average Rebuild cost/history/first-consumer rate equals the original2000-user ledger",
            deployment_flops_included=False, measured_population_runtime=False,
            replaces_main2000_cost_denominator=False)
        key = row["fitting_identity_sha256"]
        stages = chains.setdefault(key,{})
        if row["method_label"] in stages:
            raise ValueError("duplicate calibration branch")
        stages[row["method_label"]] = row
        rows.append(row)
    if not rows:
        raise ValueError("actual checkpoints are required; no empty final cost audit is produced")
    for stages in chains.values():
        if set(stages) != {"f","fg","fgh"}:
            raise ValueError("this audit expects the complete three-stage ladder")
        previous = stages["f"]["raw_cumulative_calibration_flops_estimate"]-stages["f"]["raw_stage_calibration_flops_estimate"]
        if any(row["standalone_shared_preparation_flops"] != previous for row in stages.values()):
            raise ValueError("the retained ladder does not contain exactly one shared preparation fee")
        for label in ("f","fg","fgh"):
            row = stages[label]
            if row["raw_cumulative_calibration_flops_estimate"] != previous+row["raw_stage_calibration_flops_estimate"]:
                raise ValueError("cumulative calibration repeats or omits a predecessor stage")
            previous = row["raw_cumulative_calibration_flops_estimate"]
    if any(row["correction_source_sha256"] != rows[0]["correction_source_sha256"] for row in rows):
        raise ValueError("correction sources changed during the CPU audit")
    return dict(format="probe_calibration_cost_audit_v1", status="complete_cpu_audit", records=rows,
        stage_records=len(rows), complete_ladders=len(chains), source_sha256=rows[0]["correction_source_sha256"],
        standalone_preparation_charged_once=True,
        scope="actual retained development weights; no quality metric, new fitting, GPU use, or final configuration claim",
        original_artifacts_modified=False, main2000_denominator_changed=False)

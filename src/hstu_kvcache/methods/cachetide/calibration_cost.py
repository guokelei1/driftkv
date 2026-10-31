"""Small analytical counters for the actual query-relay calibration calls.

Multiply-add is two FLOPs. Dense padded shapes are charged. As in the retained
local/joint calibration, backward through a trainable computation or frozen
backbone is conservatively estimated as twice its complete forward arithmetic.
This is an arithmetic convention, not a hardware-instruction measurement.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]

from hstu_kvcache.methods.cachetide.cost import read_flops, origin_feature_bank_cost
from hstu_kvcache.evaluation.cost import CostModel, eager_read, teacher_history_read


def _geometry(adapter):
    layer = adapter.layers[0]
    return dict(mode=adapter.mode, layers=adapter.num_layers, heads=adapter.heads,
                dim=adapter.head_dim, rank=layer.accessible_rank,
                gate_normalization=getattr(layer, "gate_normalization", "raw"),
                feature_mode=getattr(layer, "feature_mode", "native"),
                code=layer.code_width, item=layer.item_width, xi=layer.xi_width,
                widths=tuple(branch.code_width for branch in layer.branches),
                native_widths=tuple(2 * len(branch.heads) * adapter.head_dim for branch in layer.branches))


def update_forward_flops(adapter, *, rows, layers=None, cached_item_coefficients=False):
    """advance_codes arithmetic; features/coefficients are not presumed cached."""
    g = _geometry(adapter)
    count = g["layers"] if layers is None else layers
    c, e, r = g["code"], g["item"], g["xi"]
    native_weights = sum(n * w for n, w in zip(g["native_widths"], g["widths"]))
    blocks = sum(w * w for w in g["widths"])
    native_forward = 2 * native_weights
    if g["mode"] in ("legacy", "C"):
        item = 0 if cached_item_coefficients else 2 * e * c + c
        previous = 2 * c * c + 2 * c if g["mode"] == "C" else 0
        return count * rows * (native_forward + item + 4 * c + previous)
    norm = sum(4 * n + 2 for n in g["native_widths"])
    if g["feature_mode"] == "origin_encoder" and r == 1:
        norm = 0
    # Origin item lookup and separate bias preserve the warmstart's arithmetic.
    eta = native_forward + (5 if g["feature_mode"] == "origin_encoder" else 3) * c
    summaries = (r - 1) * sum(n // 2 + 1 for n in g["native_widths"])
    item = 0 if cached_item_coefficients else 2 * e * c * r + c * r
    # Two block products, H*xi reduction, and addition of all three terms.
    per_row = norm + eta + summaries + 4 * blocks + item + 2 * c * r + 2 * c
    # Cayley solve with matrix RHS: leading 8/3 n^3 plus pointwise skew/eye.
    matrix_construction = sum((8 * w**3 + 2) // 3 + 6 * w**2 for w in g["widths"])
    return count * (rows * per_row + matrix_construction)


def corrected_query_flops(adapter, *, length, queries, batch=1,
                          coded_rows=None, fitted_layers=None):
    g = _geometry(adapter)
    cm = CostModel(g["heads"] * g["dim"], g["layers"], g["heads"], "torch")
    native = eager_read(cm, length, queries=queries, batch=batch)
    coded = length if coded_rows is None else coded_rows
    layers = g["layers"] if fitted_layers is None else fitted_layers
    extra = 0 if coded == 0 else read_flops(pairs=batch * queries * coded,
        queries=batch * queries, head_rank=g["rank"], layers=layers,
        heads=g["heads"], head_dim=g["dim"], gate_normalization=g["gate_normalization"])["total"]
    return native, extra


def warmstart_origin_cost(path):
    """Original trained origin charged once; candidate predecessors paid in chain.

    A missing cost is left unknown, never silently converted to zero.
    """
    path = Path(path).resolve()
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    candidate = metadata.get("kind") == "query_relay_shared12_private6_v1" or \
                "operation_counts" in metadata.get("diagnostics", {})
    cost = metadata.get("cost")
    if candidate:
        return dict(path=str(path), metadata_path=str(metadata_path),
            charge_separately=False, calibration_flops=0,
            scope="Previous candidate release is paid once in the continuous-chain aggregate; do not recursively double-charge it.")
    if cost is None or "calibration_flops" not in cost:
        return dict(path=str(path), metadata_path=str(metadata_path),
                    charge_separately=True, calibration_flops=None,
                    scope="Original warmstart training cost is unknown and must be supplied before a complete-cost claim.")
    sensitivity = cost.get("source_reconstruction_sensitivity", {})
    return dict(path=str(path), metadata_path=str(metadata_path), charge_separately=True,
        calibration_flops=int(cost["calibration_flops"]),
        source_reconstruction_sensitivity_flops=int(sensitivity.get("additional_flops", 0)),
        scope="This originating release only, paid once per selected chain. Unused V4/V5 historical calibrations are not included.")


class OperationCounts:
    """Add FLOPs at the existing forward/backward boundaries, without hooks."""

    def __init__(self):
        self.components = Counter()
        self.calls = Counter()

    def add(self, name, flops):
        self.components[name] += int(flops)

    def record_gate_item_input(self, adapter, *, items, stage="query"):
        """Executed extra item lookup, memory-only; no feature is persisted.

        Call at the actual preparation boundary, including zero-code reads if
        the implementation performs the lookup. Existing gate/RMS arithmetic
        is already charged by the read counter and is not added again.
        """
        if getattr(adapter, "gate_source", "layer") != "item":
            return 0
        payload = int(items) * adapter.heads * adapter.head_dim \
            * next(adapter.parameters()).element_size()
        self.calls[stage + "_gate_item_lookup_calls"] += 1
        self.calls[stage + "_gate_item_lookup_items"] += int(items)
        self.calls[stage + "_gate_item_lookup_bytes"] += payload
        return payload

    def record_update(self, adapter, *, length, batch=1, layers=None,
                      backward=False, cached_item_coefficients=False, stage="update"):
        forward = update_forward_flops(adapter, rows=batch * length, layers=layers,
                                        cached_item_coefficients=cached_item_coefficients)
        self.add(stage + "_forward_flops", forward)
        if backward:
            self.add(stage + "_backward_flops_estimate", 2 * forward)
        self.calls[stage + "_rows"] += batch * length
        return forward

    def record_query(self, adapter, *, length, queries, batch=1, coded_rows=None,
                     fitted_layers=None, backward=False, stage="query"):
        native, correction = corrected_query_flops(adapter, length=length, queries=queries,
            batch=batch, coded_rows=coded_rows, fitted_layers=fitted_layers)
        self.add(stage + "_native_forward_flops", native)
        self.add(stage + "_correction_forward_flops", correction)
        if backward:
            self.add(stage + "_frozen_backbone_backward_flops_estimate", 2 * native)
            self.add(stage + "_correction_backward_flops_estimate", 2 * correction)
        self.calls[stage + "_candidate_queries"] += batch * queries
        self.calls[stage + "_dense_token_queries"] += batch * queries * length
        return native + correction

    def record_local(self, adapter, *, batch, padded_length, queries,
                     backward=False, encoder_backward=None, stage="local"):
        """One layer's fresh encoder and actual dense local read, including mask."""
        g = _geometry(adapter)
        rows, q = batch * padded_length, batch * queries
        pairs = rows * queries
        encode = self.record_update(adapter, length=padded_length, batch=batch,
            layers=1, backward=backward if encoder_backward is None else encoder_backward,
            stage=stage + "_encoder")
        native = pairs * (4 * g["heads"] * g["dim"] + 4 * g["heads"])
        correction = read_flops(pairs=pairs, queries=q, head_rank=g["rank"],
            layers=1, heads=g["heads"], head_dim=g["dim"],
            gate_normalization=g["gate_normalization"])["total"]
        outputs = q * g["heads"] * g["dim"]
        self.add(stage + "_native_read_forward_flops", native)
        self.add(stage + "_correction_read_forward_flops", correction)
        self.add(stage + "_rate_normalization_flops", outputs)
        self.add(stage + "_read_loss_flops", (8 if backward else 3) * outputs)
        if backward:
            self.add(stage + "_native_read_backward_flops_estimate", 2 * native)
            self.add(stage + "_correction_read_backward_flops_estimate", 2 * correction)
            self.add(stage + "_rate_backward_flops_estimate", 2 * outputs)
        self.calls[stage + "_dense_padded_token_queries"] += pairs
        return encode + native + correction + outputs

    def record_local_capture(self, adapter, *, length, queries, layer_index):
        """Actual capture_local runs all layer updates plus a complete query."""
        g = _geometry(adapter)
        self.record_update(adapter, length=length, stage="local_capture_encoder")
        self.record_query(adapter, length=length, queries=queries,
                          fitted_layers=layer_index, stage="local_capture_query")
        cm = CostModel(g["heads"] * g["dim"], g["layers"], g["heads"], "torch")
        self.add("local_capture_same_query_teacher_read_flops",
                 teacher_history_read(cm, length, queries=queries))
        self.add("local_capture_rate_normalization_flops", 2 * g["heads"] * queries * g["dim"])

    def record_score_loss(self, *, outputs, backward=False, stage="joint"):
        self.add(stage + "_score_loss_flops", (6 if backward else 3) * outputs)

    def record_optimizer(self, *, parameters, stage="optimizer"):
        self.add(stage + "_adam_and_clip_flops_estimate", 18 * parameters + 10)
        self.calls[stage + "_steps"] += 1

    def record_gradient_diagnostic(self, *, parameters, tensors=1):
        self.add("gradient_norm_diagnostic_flops", 2 * parameters + tensors)

    def record_native_capture(self, capture_costs, *, source_available=True):
        self.add("teacher_cache_capture_flops", capture_costs.get("teacher_cache_capture_flops", 0))
        source = capture_costs.get("source_trajectory_flops", 0)
        # Retained source assumes actual service state is available; report
        # development execution and the reconstruction sensitivity separately.
        self.calls["source_trajectory_development_flops"] += int(source)
        if not source_available:
            self.add("required_source_trajectory_flops", source)
        else:
            self.calls["omitted_source_reconstruction_flops"] += int(source)
        # Corrected source replay may also contain preparation/read costs. The
        # caller supplies these explicitly; unknown components are not dropped.
        for key, value in capture_costs.items():
            if key not in ("teacher_cache_capture_flops", "source_trajectory_flops"):
                self.calls["source_extra_" + key] += int(value)
                if not source_available and key.endswith("_flops"):
                    self.add("required_source_" + key, value)
                elif source_available and key.endswith("_flops"):
                    self.calls["omitted_source_reconstruction_flops"] += int(value)

    def record_feature_bank_preparation(self, *, catalogue_items, layers=6,
                                        code_width=48, embedding_width=192):
        """Record actual one-time asset construction, outside per-release fit.

        Complete deployment price charges this shared bank explicitly, including
        preparation replicas. Do not add it again to per-release calibration.
        """
        cost = origin_feature_bank_cost(catalogue_items, layers=layers,
            code_width=code_width, embedding_width=embedding_width, replicas=1)
        self.calls["immutable_feature_bank_preparation_flops"] += cost["preparation_flops"]
        self.calls["immutable_feature_bank_payload_bytes"] += cost["payload_bytes_per_replica"]
        return cost

    def record_event_write(self, adapter, *, old_length, width, coded_rows,
                           backward=False, stage="write_unroll", backend="torch"):
        g = _geometry(adapter)
        cm = CostModel(g["heads"] * g["dim"], g["layers"], g["heads"], backend)
        native = cm.band_append(old_length, width, window_size=adapter.max_length)
        correction = 0 if coded_rows == 0 else read_flops(pairs=width * coded_rows,
            queries=width, head_rank=g["rank"], layers=g["layers"],
            heads=g["heads"], head_dim=g["dim"], gate_normalization=g["gate_normalization"])["total"]
        self.add(stage + "_native_forward_flops", native)
        self.add(stage + "_correction_forward_flops", correction)
        if backward:
            self.add(stage + "_frozen_backbone_backward_flops_estimate", 2 * native)
            self.add(stage + "_correction_backward_flops_estimate", 2 * correction)
        self.calls[stage + "_behavior_queries"] += width
        self.calls[stage + "_dense_prefix_pairs"] += width * coded_rows

    def record_gate_statistics(self, adapter, *, length, queries):
        g = _geometry(adapter)
        self.record_update(adapter, length=length, stage="gate_statistics_encoder")
        self.record_query(adapter, length=length, queries=queries, stage="gate_statistics_query")
        # Statistics callback computes gates again before the actual read.
        gate = g["layers"] * queries * (4 * g["heads"] * g["dim"] * g["rank"]
                                        + 4 * g["heads"] * g["rank"])
        if g["gate_normalization"] == "rms":
            gate += g["layers"] * queries * g["heads"] * (4 * g["dim"] + 2)
        reduce = g["layers"] * 6 * queries * g["heads"] * g["rank"]
        self.add("gate_statistics_duplicate_gate_and_reduction_flops", gate + reduce)

    def report(self, *, origin=None):
        components = dict(self.components)
        stage = sum(components.values())
        origin_charge = 0 if origin is None or not origin.get("charge_separately") else origin["calibration_flops"]
        source = self.calls.get("omitted_source_reconstruction_flops", 0)
        origin_source = 0 if origin is None else origin.get("source_reconstruction_sensitivity_flops", 0)
        return dict(components=components, calibration_flops=stage,
            stage_calibration_flops=stage, warmstart_origin=origin,
            calibration_with_initial_origin_flops=None if origin_charge is None else stage + origin_charge,
            source_reconstruction_sensitivity=dict(additional_flops=source + origin_source,
                scope="Only reconstruction omitted from primary. Required source work already charged is not added twice."),
            operation_counts=dict(self.calls),
            immutable_feature_bank_preparation=dict(
                actual_construction_flops=self.calls.get("immutable_feature_bank_preparation_flops", 0),
                actual_payload_bytes=self.calls.get("immutable_feature_bank_payload_bytes", 0),
                cost_owner="Complete serving price includes the once-per-chain shared bank and replica policy; excluded from this per-release fit total to avoid double charging."),
            convention="MAC=2; elementary pointwise=1; padded dense shapes charged. Forward arithmetic declared from current adapter. Trainable forward and frozen-backbone backward conservatively estimated2x; no loaded serving timing claim.",
            source_extra_scope="Supplied source_extra_* FLOPs join the reconstruction sensitivity or, if source_available=False, the primary cost. Other shared-calibration work is added explicitly at its call site.")

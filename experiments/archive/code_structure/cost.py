#!/usr/bin/env python3
"""Price shared codes on the retained fixed30k event ledger.

The preview replaces the original joint16 arithmetic with the proposed folded
encoder. A completed run instead supplies its actual calibration JSONs; no
preview estimate is then used for calibration. Native trajectories, static4k
IDs and lazy full-refresh lifecycle must match the retained service counts.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration"
FACTORED = ROOT / "results/partial_refresh_2026_10/round2/factored_cost/summary.json"
REFERENCE = ROOT / "results/partial_refresh_2026_10/population_cost_raw/summary.json"
LAYOUTS = {
    "headlocal8": dict(head_rank=8, kv_weights=18_432, encoder_weights=73_728),
    "group2": dict(head_rank=16, kv_weights=36_864, encoder_weights=92_160),
    "shared6_private7": dict(head_rank=13, kv_weights=29_952, encoder_weights=85_248),
    "shared12_private6": dict(head_rank=18, kv_weights=41_472, encoder_weights=96_768),
}
VERSIONS = ("v3", "v4", "v5")
OLD_ROW_FLOPS = 166_464
ITEM_WEIGHTS = 55_296
ITEM_FLOPS = 110_880
CODE_WIDTH = 288
OLD_PARAMETERS = 92_448


def read(path):
    return json.loads(Path(path).read_text())


def preview_calibration(layout):
    """Same local32 origins and original joint16 schedule, prior to execution."""
    spec = LAYOUTS[layout]
    # Structure.encode runs KV + item, then adds bias and applies SiLU. Serving
    # has already included bias in the item table and needs only two row ops.
    train_row = 2 * spec["encoder_weights"] + 3 * CODE_WIDTH
    parameters = spec["encoder_weights"] + CODE_WIDTH + 2 * 6 * 6 * spec["head_rank"] * 32
    delta_rank = spec["head_rank"] - 8
    by_release = {}
    encoder_keys = ("frozen_code_preparation_flops", "execution_check_reencoding_flops",
        "joint_validation_encoder_flops", "joint_training_encoder_forward_flops",
        "joint_training_encoder_backward_flops_estimate")
    # Counts are derivable from retained original components; these diagnostics
    # are estimates of a replacement run, never represented as measured work.
    parameter_rules = {
        "joint_coordinate_statistics_flops": (4, 288),
        "execution_check_coordinate_flops": (4, 0),
        "joint_validation_coordinates_flops": (2, 0),
        "joint_coordinate_diagnostics_flops": (2, 4),
        "joint_coordinates_forward_backward_flops": (3, 0),
        "joint_identity_prior_flops": (6, 4),
        "joint_optimizer_clip_flops_estimate": (18, 10),
        "first_gradient_diagnostic_flops": (2, 4),
        "publication_joint_fold_flops": (2, 0),
    }
    for version in VERSIONS:
        metadata = read(ORIGINAL / version / "calibration.json")
        local = read(ORIGINAL / version / "read_only/calibration.json")["cost"]["calibration_flops"]
        original = metadata["cost"]["stage_components"]["joint_refinement"]
        joint = deepcopy(original)
        for key in encoder_keys:
            rows = original[key] // OLD_ROW_FLOPS
            if rows * OLD_ROW_FLOPS != original[key]:
                raise RuntimeError(f"original encoder count is not integral: {version}/{key}")
            joint[key] = rows * train_row
        lengths = {row["uid"]: row["history_length"] for row in metadata["capture_records"]}
        train = metadata["fit_uids"]
        validation = metadata["validation_uids"]
        for query_key, rows, users in (
            ("joint_training_query_forward_flops", 16 * sum(lengths[u] for u in train), 16 * len(train)),
            ("joint_training_query_backward_flops_estimate", 32 * sum(lengths[u] for u in train), 32 * len(train)),
            ("joint_validation_query_flops", 3 * sum(lengths[u] for u in validation), 3 * len(validation)),
            ("execution_check_query_flops", 4 * lengths[validation[0]], 4),
        ):
            joint[query_key] += 6 * 4 * 6 * delta_rank * 16 * (rows + 32 * users)
        for key, (multiplier, constant) in parameter_rules.items():
            unit = multiplier * OLD_PARAMETERS + constant
            count, remainder = divmod(original[key], unit)
            if remainder:
                raise RuntimeError(f"original parameter operation count differs: {version}/{key}")
            joint[key] = count * (multiplier * parameters + constant)
        train_calls = original["joint_coordinates_forward_backward_flops"] // (3 * OLD_PARAMETERS)
        validation_calls = original["joint_validation_coordinates_flops"] // (2 * OLD_PARAMETERS)
        centering = 2 * spec["encoder_weights"] + CODE_WIDTH
        # Three non-training calls are the two refined publication checks and
        # final folding. Training includes forward and twice-forward backward.
        joint["joint_coordinate_centering_flops"] = centering * (3 * train_calls + validation_calls + 3)
        # The initial head-local affine is folded once before layout conversion.
        joint["original_local_encoder_normalization_fold_flops"] = 221_184
        by_release[version] = dict(inherited_local_calibration_flops=local,
            replacement_joint_components=joint, replacement_joint_flops=sum(joint.values()),
            calibration_flops=local + sum(joint.values()),
            source_reconstruction_sensitivity_flops=metadata["cost"]["source_reconstruction_sensitivity"]["additional_flops"])
    return dict(mode="analytical_preview_not_execution", training_encoder_flops_per_row=train_row,
        trainable_parameters=parameters, by_release=by_release,
        calibration_flops=sum(row["calibration_flops"] for row in by_release.values()),
        source_reconstruction_sensitivity_flops=sum(row["source_reconstruction_sensitivity_flops"] for row in by_release.values()),
        assumption="Replaces joint16 after original local32. Folded KV and complete item projections run for every row at each training forward, followed by two additions and SiLU; no trainable-code or cross-step item reuse. Centered normalized-coordinate increments are explicitly charged. Same original validation/check schedule and normalized-coordinate prior. Actual calibration JSON replaces this preview.")


def actual_calibration(directory):
    by_release = {}
    for version in VERSIONS:
        path = Path(directory) / version / "calibration.json"
        metadata = read(path)
        if metadata["status"] != "complete":
            raise RuntimeError(f"incomplete calibration: {path}")
        cost = metadata["cost"]
        total = cost["calibration_flops"]
        if sum(cost["components"].values()) != total:
            raise RuntimeError(f"calibration component sum differs: {path}")
        inherited = read(ORIGINAL / version / "read_only/calibration.json")["cost"]["calibration_flops"]
        if cost["inherited_local_calibration_flops"] != inherited:
            raise RuntimeError(f"local32 preparation charge differs: {path}")
        source = cost["source_reconstruction_sensitivity"]["additional_flops"]
        by_release[version] = dict(calibration_path=str(path.resolve()),
            inherited_local_calibration_flops=inherited, calibration_flops=total,
            source_reconstruction_sensitivity_flops=source)
    return dict(mode="completed_calibration_components", by_release=by_release,
        calibration_flops=sum(row["calibration_flops"] for row in by_release.values()),
        source_reconstruction_sensitivity_flops=sum(row["source_reconstruction_sensitivity_flops"] for row in by_release.values()))


def price(layout, calibration, *, replicas=4, include_controls=True):
    spec = LAYOUTS[layout]
    retained = read(FACTORED)
    reference = read(REFERENCE)
    original = retained["policies"]["full_static4096"]
    baseline = reference["policies"]["original_masked_full_reference"]
    releases = {}
    row_flops = 2 * spec["kv_weights"] + 2 * CODE_WIDTH
    for version, row in original["by_release"].items():
        counts, cache = row["counts"], row["cache_counts"]
        encoder_rows = counts["first_read_encode_rows"] + counts["append_encode_rows"]
        components = dict(
            initial_row_kv=counts["initial_encode_rows"] * row_flops,
            refresh_row_kv=counts["refresh_encode_rows"] * row_flops,
            append_row_kv=counts["append_encode_rows"] * row_flops,
            first_read_uncached_items=cache["first_read_unique_item_misses"] * ITEM_FLOPS,
            append_uncached_items=cache["append_unique_item_misses"] * ITEM_FLOPS,
            static_table_preparation=replicas * row["maintenance_components_flops"]["static_table_preparation"],
            fixed_normalization_fold=0,
            encoder_support_mask=encoder_rows * CODE_WIDTH)
        # Per layer/head: QDk, projected-Q @ code, attention @ code, latent Dv;
        # score/output additions remain the same as the original reader.
        query = 6 * ((4 * 6 * spec["head_rank"] + 6) * counts["corrected_token_query_pairs"]
            + (4 * 6 * 32 * spec["head_rank"] + 6 * 32) * counts["corrected_requests"])
        releases[version] = dict(counts=counts, cache_counts=cache,
            maintenance_components_flops=components, maintenance_flops=sum(components.values()),
            query_extra_flops=query, static_table_payload_bytes_per_worker=row["shared_projection_table_bytes"],
            encoder_parameter_payload_bytes=(spec["encoder_weights"] + CODE_WIDTH) * 4,
            key_value_decoder_payload_bytes=2 * 6 * 6 * spec["head_rank"] * 32 * 4)
    maintenance = sum(row["maintenance_flops"] for row in releases.values())
    query = sum(row["query_extra_flops"] for row in releases.values())
    complete = calibration["calibration_flops"] + maintenance + query
    denominator = reference["once_rebuild_flops"]
    source = calibration["source_reconstruction_sensitivity_flops"]
    report = dict(layout=layout, mode=calibration["mode"], calibration=calibration,
        serving_assumption="Same fixed30k native event ledger, full lazy initialization/refresh and active appends, original support mask. Folded encoder parameters are published; no normalization fold at serving. Same static4096 admission IDs, unique item projections per actual encode call, no admission of misses.",
        users=retained["users"], active_request_users=retained["active_request_users"],
        requests=retained["requests"], preparation_replicas=replicas, by_release=releases,
        stored_floats_per_layer_row=48, visible_floats_per_head=spec["head_rank"],
        row_kv_flops=row_flops, unique_item_projection_flops=ITEM_FLOPS,
        native_kv_bytes_per_row=9216, code_bytes_per_row=1152, support_flag_bytes_per_row=6,
        code_percent_of_native_kv=12.5, codes_and_support_flags_percent_of_native_kv=100 * 1158 / 9216,
        shared_current_item_table_payload_bytes_per_release_per_worker=4718592,
        shared_current_item_table_payload_bytes_per_release_all_workers=4718592 * replicas,
        maintenance_flops=maintenance, extra_query_flops=query,
        shared_calibration_flops=calibration["calibration_flops"], complete_added_flops=complete,
        once_rebuild_flops=denominator, complete_percent_of_once_rebuild=100 * complete / denominator,
        complete_20_percent_budget_flops=denominator * .2,
        remaining_20_percent_budget_flops=denominator * .2 - complete,
        source_reconstruction_sensitivity_flops=source,
        complete_with_source_reconstruction_percent=100 * (complete + source) / denominator,
        native_query_flops=baseline["native_query_flops"],
        native_history_attention_flops=baseline["native_history_attention_flops"],
        extra_percent_of_native_whole_query=100 * query / baseline["native_query_flops"],
        extra_percent_of_native_history_attention=100 * query / baseline["native_history_attention_flops"],
        original_static4096_four_worker_complete_percent=original["independent_preparation_sensitivity"]["4"]["complete_percent_of_once_rebuild"],
        notes=["Only completed calibration mode is final cost evidence; preview is a capacity prefilter.",
            "Exploratory fitting sweeps and duplicate GPU capture execution are development resources. The selected production calibration charges the complete inherited local32 path and replacement joint path once.",
            "Shared payload reports tensors only. Item IDs, Python dictionaries and transient per-head code gathering are separate bytes. No loaded serving time/p99 claim.",
            "The service ledger applies only if the candidate uses these same lifecycle/encode-call shapes. GPU execution counters must be checked before calling it a completed candidate cost."])
    if include_controls:
        preparation = original["independent_preparation_sensitivity"]["4"]["one_replica_preparation_flops"]
        original_complete = original["complete_added_flops"] + (replicas - 1) * preparation
        fair = price("headlocal8", preview_calibration("headlocal8"),
                     replicas=replicas, include_controls=False)
        report["original_static4096_fixed_weights_control"] = dict(
            mode="retained_original_calibration_and_selected_weights",
            preparation_replicas=replicas, complete_added_flops=original_complete,
            complete_percent_of_once_rebuild=100 * original_complete / denominator,
            quality="Retained selected head-local rank8 weights. No control refitting or new quality claim.")
        report["fair_folded_headlocal8_joint16_preview"] = {key: fair[key] for key in (
            "mode", "shared_calibration_flops", "maintenance_flops", "extra_query_flops",
            "complete_added_flops", "complete_percent_of_once_rebuild",
            "complete_with_source_reconstruction_percent", "extra_percent_of_native_whole_query")}
        report["fair_folded_headlocal8_joint16_preview"]["scope"] = (
            "Theoretical arithmetic for the same original local32 plus joint16 trajectory, "
            "expressed through folded head-local encoders and centered coordinates. "
            "A fairness cost reference, not an executed refit or a new quality result.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=tuple(LAYOUTS))
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calibration_root and not args.layout:
        parser.error("completed calibration requires --layout")
    if args.replicas < 1:
        parser.error("--replicas must be positive")
    layouts = (args.layout,) if args.layout else tuple(LAYOUTS)
    reports = {layout: price(layout,
        actual_calibration(args.calibration_root) if args.calibration_root else preview_calibration(layout),
        replicas=args.replicas) for layout in layouts}
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({layout: {key: report[key] for key in (
        "mode", "complete_percent_of_once_rebuild", "complete_with_source_reconstruction_percent",
        "shared_calibration_flops", "maintenance_flops", "extra_query_flops",
        "extra_percent_of_native_whole_query", "remaining_20_percent_budget_flops",
        "original_static4096_fixed_weights_control", "fair_folded_headlocal8_joint16_preview")}
        for layout, report in reports.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

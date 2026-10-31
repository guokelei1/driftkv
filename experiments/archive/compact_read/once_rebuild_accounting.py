#!/usr/bin/env python3
"""Re-account retained evidence against one reconstruction per user/release.

Reads completed CPU population ledgers and matched development scores only.
The original Fresh-based reports remain unchanged as historical evidence.
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
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT / "results/design_system_2026_10/phase2"
VERSIONS = ("v3", "v4", "v5")
EDGES = ("v2_to_v3", "v3_to_v4", "v4_to_v5")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator > 0 else None


def quality(analysis):
    edges = analysis["metrics"]["edges"]
    output = {}
    for reference in ("first_request_rebuild", "exact"):
        edge_results = {}
        for edge in EDGES:
            row = edges[edge]
            reuse = row["methods"]["reuse"]["ROC_AUC"]
            gap = row["methods"][reference]["ROC_AUC"] - reuse
            edge_results[edge] = dict(requests=row["requests"],
                active_users=row["active_users"], reference_minus_reuse_auc=gap,
                methods={name: dict(auc=m["ROC_AUC"], log_loss=m["log_loss"],
                    auc_gain_vs_reuse=m["ROC_AUC"]-reuse,
                    gap_recovery_percent=percent(m["ROC_AUC"]-reuse, gap))
                    for name, m in row["methods"].items()})
        mean_gap = sum(e["reference_minus_reuse_auc"] for e in edge_results.values()) / 3
        methods = {}
        for name in analysis["methods"]:
            gain = sum(e["methods"][name]["auc_gain_vs_reuse"] for e in edge_results.values()) / 3
            methods[name] = dict(mean_release_auc_gain=gain,
                release_equal_gap_recovery_percent=percent(gain, mean_gap),
                mean_release_log_loss=sum(e["methods"][name]["log_loss"] for e in edge_results.values()) / 3,
                target_80_margin=gain-.8*mean_gap,
                point_recovery_target_met=mean_gap > 0 and gain >= .8*mean_gap)
        output[reference] = dict(mean_release_reference_minus_reuse_auc=mean_gap,
            by_release=edge_results, methods=methods)
    return dict(scope="Same representative 4096 development users and 51731 requests; no population quality extrapolation.",
        aggregation="Ratio of equal-release mean AUC differences, using all three releases; not mean recovery and not pooled AUC.",
        primary_reference="first_request_rebuild", sensitivity_reference="exact",
        references=output,
        uncertainty="This accounting change recomputes point ratios. Retained paired AUC/log-loss difference intervals remain valid; Fresh-denominator recovery intervals are not relabeled as once-rebuild intervals.")


def query_arithmetic(counts, *, layers, width, heads, compact_width):
    # Invert the executed compact estimator's linear-in-(requests*history)
    # formula. This extracts exact native history attention without assuming
    # every request has a full 1024-token history or ignoring batch call costs.
    requests = counts["requests"]
    native_per_token = 4*width + 3*heads
    compact_per_token = 4*heads*compact_width + 3*heads
    compact_per_query = 4*width*compact_width + 5*width
    numerator = counts["extra_history_read_flops"] - layers*requests*compact_per_query
    divisor = layers*(native_per_token+compact_per_token)
    token_queries, remainder = divmod(numerator, divisor)
    if remainder or token_queries < 0:
        raise RuntimeError("compact shape ledger cannot reproduce its attention term")
    native_attention = layers*token_queries*native_per_token
    extra = sum(counts.get(k, 0) for k in
        ("lifecycle_extra_history_read_flops", "lifecycle_response_flops", "coordination_flops"))
    native_query = counts["native_query_flops"]
    return dict(requests=requests, token_query_pairs=token_queries,
        mean_history_length=token_queries/requests,
        native_whole_query_flops=native_query,
        native_history_attention_flops=native_attention,
        extra_query_flops=extra,
        extra_query_components_flops=dict(
            duplicated_native_history_attention=native_attention,
            compact_contractions_and_bias=counts["lifecycle_extra_history_read_flops"]-native_attention,
            response=counts["lifecycle_response_flops"],
            coordination=counts.get("coordination_flops", 0)),
        native_whole_query_flops_per_request=native_query/requests,
        native_history_attention_flops_per_request=native_attention/requests,
        extra_query_flops_per_request=extra/requests,
        extra_percent_of_native_whole_query=percent(extra, native_query),
        extra_percent_of_native_history_attention=percent(extra, native_attention),
        complete_corrected_query_over_native_whole_query=(native_query+extra)/native_query,
        note="Actual retained implementation still executes the first native read plus corrected attention. No credit for unexecuted native-read elision or kernel optimization.")


def cost_release(release, calibrations):
    counts = release["counts_and_flops"]
    lazy, eager = counts["first_request_rebuild_flops"], counts["exact_rebuild_flops"]
    policies = {}
    for name, calibration in calibrations.items():
        components = dict(fixed_complete_calibration_flops=calibration,
            fixed_projection_setup_flops=release["shared_projection_setup_flops"])
        prefix = "" if name == "d1" else "lifecycle_"
        components.update(
            release_user_encoding_flops=counts[prefix+"initial_encode_flops"],
            appended_user_encoding_flops=counts[prefix+"append_encode_flops"],
            extra_query_history_read_flops=counts[prefix+"extra_history_read_flops"],
            extra_query_response_flops=counts[prefix+"response_flops"],
            extra_query_coordination_flops=0 if name in ("d1", "d1_causal_exit") else counts["coordination_flops"])
        total = sum(components.values())
        recorded = release["policies"].get(name)
        if recorded and recorded["incremental_vs_reuse_flops"] != total:
            raise RuntimeError(f"component decomposition disagrees with completed population ledger: {name}")
        policies[name] = dict(components_flops=components,
            incremental_vs_reuse_flops=total,
            percent_of_first_request_rebuild=percent(total, lazy),
            percent_of_all_resident_release_rebuild=percent(total, eager))
    return dict(requests=counts["requests"],
        first_request_rebuild_minus_reuse_flops=lazy,
        all_resident_release_rebuild_minus_reuse_flops=eager,
        common_incremental_native_writes_flops=counts["native_append_flops"],
        common_native_queries_flops=counts["native_query_flops"], policies=policies)


def scenario_projection(components, first, eager, users, target_users):
    # Only a conditional algebraic sensitivity, not sampled-population evidence.
    fixed = components["fixed_complete_calibration_flops"] + components["fixed_projection_setup_flops"]
    population_work = sum(components.values())-fixed
    multiplier = target_users/users
    total = fixed+multiplier*population_work
    return dict(resident_users=target_users, conditional_incremental_flops=total,
        percent_of_first_request_rebuild=percent(total, multiplier*first),
        percent_of_all_resident_release_rebuild=percent(total, multiplier*eager),
        quality=None,
        assumptions="Same model/context, three release windows, activity/write/request/history mix per resident, fixed calibration users/epochs, and fixed four projection replicas. This is an analytic population projection, not measured users or quality.")


def reassess(costs, analysis, config, read, panel_config):
    base_metas, strength_metas = {}, {}
    for version in VERSIONS:
        base_metas[version] = read(ROOT/config["calibration_dir"]/version/"calibration.json")
        strength_metas[version] = read(ROOT/config["strengths_dir"]/version/"strengths.json")
    modes = config.get("strength_modes", {"task_causal_exit":"task", "read_causal_exit":"read"})
    by_release, shape_counts = {}, Counter()
    for version in VERSIONS:
        base, strengths = base_metas[version], strength_metas[version]
        calibration = {name: base["cost"]["calibration_flops"] for name in ("d1", "d1_causal_exit")}
        calibration.update({name: strengths["cost_by_mode"][mode]["calibration_flops"] for name, mode in modes.items()})
        for name, value in calibration.items():
            if analysis["costs"]["by_release"][version]["policies"][name]["complete_calibration_flops"] != value:
                raise RuntimeError(f"analysis and calibration metadata disagree: {version}/{name}")
        by_release[version] = cost_release(costs["by_release"][version], calibration)
        shape_counts.update(costs["by_release"][version]["counts_and_flops"])
    first = sum(r["first_request_rebuild_minus_reuse_flops"] for r in by_release.values())
    eager = sum(r["all_resident_release_rebuild_minus_reuse_flops"] for r in by_release.values())
    totals = {}
    for name in by_release["v3"]["policies"]:
        components = Counter()
        for row in by_release.values():
            components.update(row["policies"][name]["components_flops"])
        total = sum(components.values())
        variable = total-components["fixed_complete_calibration_flops"]-components["fixed_projection_setup_flops"]
        totals[name] = dict(components_flops=dict(components), incremental_vs_reuse_flops=total,
            percent_of_first_request_rebuild=percent(total, first),
            percent_of_all_resident_release_rebuild=percent(total, eager),
            first_request_cost_ceiling_20_met=total <= .2*first,
            variable_service_floor_percent_of_first_request_rebuild=percent(variable, first),
            variable_service_floor_percent_of_all_resident_release_rebuild=percent(variable, eager),
            fixed_calibration_population_projection={str(n):scenario_projection(components, first, eager, costs["users"], n)
                for n in (costs["users"], 1000000)})
    calibration_users = {}
    for version in VERSIONS:
        base, strengths = base_metas[version], strength_metas[version]
        calibration_users[version] = dict(d1_fit=len(base["fit_uids"]), d1_validation=len(base["validation_uids"]),
            d2_fit=len(strengths["fit_uids"]), d2_validation=len(strengths["validation_uids"]),
            all_calibration_unique_uids=len(set(base["fit_uids"]+base["validation_uids"]+strengths["fit_uids"]+strengths["validation_uids"])))
    widths = {m["settings"]["item_width"] for m in base_metas.values()}
    if len(widths) != 1:
        raise RuntimeError("this diagnostic expects one compact width")
    query = query_arithmetic(shape_counts, layers=panel_config["num_layers"], width=panel_config["hidden_size"],
        heads=panel_config["num_heads"], compact_width=widths.pop())
    return dict(primary_method=config["primary_method"],
        actual_service_population=dict(users=costs["users"], requests=costs["requests"],
            active_users=costs["active_users"], days=costs["days"], model_chain=costs["model_chain"],
            context=costs["context"], query_batch=costs["query_batch"], append_band=costs["append_band_size"]),
        calibration_users=calibration_users,
        population_formula="C(N)=K+P+(N/30000)*S; B_first(N)=(N/30000)*B_first(30000). K is complete fixed calibration, P shared projection setup, S actual user encoding plus extra reads. N-axis projection assumes identical event mix and does not predict quality.",
        first_request_rebuild_minus_reuse_flops=first,
        all_resident_release_rebuild_minus_reuse_flops=eager,
        primary_reference_vs_reuse_percent=100.0,
        eager_reference_percent_of_primary=percent(eager, first),
        common_v2_initialization_flops=costs["common_v2_initialization_flops"],
        common_v2_initialization_note="Pre-existing state common to all methods; excluded from post-release incremental numerator and denominator.",
        by_release=by_release, totals=totals, query_cost=query, quality=quality(analysis))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PHASE/"once_rebuild_reassessment.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {}

    def read(path):
        path = Path(path).resolve()
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    panel = read(ROOT/"results/design_two_2026_10/benchmark_panel/binding.json")
    inputs = {
        "original_task_read": (PHASE/"population_cost/summary_complete.json",
            PHASE/"representative4096/development4096/analysis.json", PHASE/"configuration.json"),
        "final_logit_round": (PHASE/"final_logit_round/population_cost/summary.json",
            PHASE/"final_logit_round/development4096/analysis.json", PHASE/"final_logit_round/configuration.json"),
    }
    rounds = {}
    first_shapes = None
    first_controls = None
    for name, (cost_path, analysis_path, config_path) in inputs.items():
        costs, analysis, config = read(cost_path), read(analysis_path), read(config_path)
        if costs["protocol"]["sha256"] != digest(config_path):
            raise RuntimeError("population ledger protocol binding differs")
        if costs["users"] != 30000 or analysis["users"] != 4096 or analysis["requests"] != 51731:
            raise RuntimeError("unexpected retained experiment scope")
        shapes = {v: costs["by_release"][v]["counts_and_flops"] for v in VERSIONS}
        controls = {e: {m:analysis["metrics"]["edges"][e]["methods"][m]
                    for m in ("reuse", "first_request_rebuild", "exact", "fresh", "d1")} for e in EDGES}
        if first_shapes is not None and (shapes != first_shapes or controls != first_controls):
            raise RuntimeError("rounds must share exactly the retained workload and quality controls")
        first_shapes, first_controls = shapes, controls
        rounds[name] = reassess(costs, analysis, config, read, panel["model_config"])
    report = dict(status="complete", role="Correction of the primary comparator; retained source outputs are unchanged.",
        primary_definition="Full rebuilds each user's retained history once at its first request after each model release, then uses normal incremental writes and query service. Reuse is zero added work; first-request rebuild is 100 percent.",
        sensitivity_definition="Rebuild every resident user's history at release, including users who never query; reported separately as eager Exact, not substituted to obtain a favorable percentage.",
        excluded_primary="Per-request Fresh reconstruction remains only a historical diagnostic, never the main 100-percent cost or quality reference.",
        cost_aggregation="Sum complete incremental FLOPs over all three releases before dividing by the sum of once-rebuild FLOPs. Ordinary native writes and query service cancel against Reuse.",
        evidence_scope="Actual complete 30000-user event-shape FLOPs ledger; quality on fixed representative4096 development users. Hypothetical1M is a labeled conditional arithmetic projection only.",
        rounds=rounds, source_sha256=sources,
        generator=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
        conclusion="Neither existing full candidate meets the corrected primary 20-percent compute ceiling or 80-percent quality point target. Fixed-calibration amortization alone cannot fix the primary cost ceiling: variable service exceeds20percent even as population grows.",
        paper_claim_validated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({name:{method: dict(cost_percent=value["percent_of_first_request_rebuild"],
        recovery_percent=row["quality"]["references"]["first_request_rebuild"]["methods"][method]["release_equal_gap_recovery_percent"])
        for method,value in row["totals"].items()} for name,row in rounds.items()}, indent=2))


if __name__ == "__main__":
    main()

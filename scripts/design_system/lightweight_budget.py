#!/usr/bin/env python3
"""Prospective CPU arithmetic for one native read plus narrow value correction.

Read retained 30k event shapes; never read labels, fit models, or change old
results. This screens an explicitly specified forward path, not its quality.
Unknown shared calibration cost stays unknown instead of being counted as zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from once_rebuild_accounting import query_arithmetic

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "results/design_system_2026_10/phase2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_flops(width, hidden):
    """Existing NonlinearResponseLayer.forward_flops for one query/layer."""
    if hidden == 0:
        return 0
    inputs = 2 * width
    # Normalize input, affine/bias/SiLU/output scale, r/N, N*rate+r.
    return 2*inputs + 2*hidden*(inputs+width) + 2*hidden + 5*width


def candidate_release(counts, *, layers, heads, head_dim, rank, response_width):
    width = heads * head_dim
    old_shapes = query_arithmetic(counts, layers=layers, width=width,
                                  heads=heads, compact_width=64)
    queries = counts["requests"]
    pairs = old_shapes["token_query_pairs"]
    # A row supplies the head's old K, old V and Current item embedding slice.
    inputs = 3 * head_dim
    encode_per_token = layers*heads*(2*inputs + 2*inputs*rank + 2*rank)
    components = dict(
        resident_release_initialization=counts["initial_history_length"]*encode_per_token,
        observed_append_encoding=counts["appends"]*encode_per_token,
        query_added_weighted_value=layers*2*heads*rank*pairs,
        query_low_rank_decode=layers*queries*2*heads*rank*head_dim,
        query_value_residual_add=layers*queries*width,
        query_response=layers*queries*response_flops(width, response_width),
    )
    extra_query = sum(value for key, value in components.items() if key.startswith("query_"))
    variable = sum(components.values())
    full = counts["first_request_rebuild_flops"]
    return dict(components_flops=components, variable_flops=variable,
        requests=queries, token_query_pairs=pairs,
        encoded_initial_tokens=counts["initial_history_length"],
        encoded_append_tokens=counts["appends"],
        first_request_rebuild_flops=full,
        variable_percent_of_once_rebuild=100*variable/full,
        extra_query_flops=extra_query,
        extra_percent_of_native_whole_query=100*extra_query/counts["native_query_flops"],
        extra_percent_of_native_history_attention=100*extra_query/old_shapes["native_history_attention_flops"],
        native_whole_query_flops=counts["native_query_flops"],
        native_history_attention_flops=old_shapes["native_history_attention_flops"],
        shared_work_budget_at_20_percent_flops=.2*full-variable)


def scale_projection(*, variable, full, users, target_users, shared_flops):
    scale = target_users/users
    scaled_variable, scaled_full = scale*variable, scale*full
    return dict(resident_users=target_users,
        first_request_rebuild_flops=scaled_full,
        population_dependent_flops=scaled_variable,
        population_dependent_percent_of_once_rebuild=100*scaled_variable/scaled_full,
        maximum_shared_work_flops_at_20_percent=.2*scaled_full-scaled_variable,
        supplied_shared_work_flops=shared_flops,
        complete_added_flops=None if shared_flops is None else shared_flops+scaled_variable,
        complete_percent_of_once_rebuild=None if shared_flops is None else
            100*(shared_flops+scaled_variable)/scaled_full,
        quality=None)


def build_report(costs, reassessment, panel, *, shared_flops, element_bytes):
    config = panel["model_config"]
    layers, heads, width = (config[k] for k in ("num_layers", "num_heads", "hidden_size"))
    head_dim = config.get("head_dim") or width//heads
    if heads*head_dim != width:
        raise ValueError("unexpected head dimensions")
    full = sum(row["counts_and_flops"]["first_request_rebuild_flops"]
               for row in costs["by_release"].values())
    retained = reassessment["rounds"]["original_task_read"]
    if full != retained["first_request_rebuild_minus_reuse_flops"]:
        raise RuntimeError("once-rebuild reports disagree")
    candidates = []
    for rank in (2, 4, 8):
        for hidden in (0, 8, 16):
            releases = {version: candidate_release(row["counts_and_flops"], layers=layers,
                heads=heads, head_dim=head_dim, rank=rank, response_width=hidden)
                for version, row in costs["by_release"].items()}
            components = Counter()
            for release in releases.values():
                components.update(release["components_flops"])
            variable = sum(components.values())
            extra_query = sum(r["extra_query_flops"] for r in releases.values())
            native_query = sum(r["native_whole_query_flops"] for r in releases.values())
            native_attention = sum(r["native_history_attention_flops"] for r in releases.values())
            pairs = sum(r["token_query_pairs"] for r in releases.values())
            initial = sum(r["encoded_initial_tokens"] for r in releases.values())
            appends = sum(r["encoded_append_tokens"] for r in releases.values())
            # Same stored item embedding is shared across layers. Count payload
            # lookup bytes; transfers/cache-line amplification require timing.
            lookup_bytes = (initial+appends)*width*element_bytes
            # The retained ledger records token*query pairs, not token*call
            # counts. With at most Q queries/call, actual cat work lies here.
            # cat reads source V+code and writes an equally large temporary;
            # attention subsequently reads that temporary at least once.
            packed_upper = layers*pairs*(width+heads*rank)*element_bytes
            packed_lower = packed_upper/costs["query_batch"]
            native_cache_bytes = 2*layers*costs["context"]*width*element_bytes
            code_bytes = layers*costs["context"]*heads*rank*element_bytes
            candidates.append(dict(name=f"value_p{rank}_response_h{hidden}",
                head_rank=rank, response_hidden_width=hidden,
                by_release=releases,
                components_flops=dict(components),
                variable_flops=variable,
                variable_percent_of_once_rebuild=100*variable/full,
                query=dict(extra_flops=extra_query,
                    extra_percent_of_native_whole_query=100*extra_query/native_query,
                    extra_percent_of_native_history_attention=100*extra_query/native_attention,
                    native_whole_query_flops=native_query,
                    native_history_attention_flops=native_attention),
                shared_work=dict(fixed_population=True, calibration_users=None,
                    calibration_flops=None, supplied_complete_shared_work_flops=shared_flops,
                    maximum_at_20_percent_flops=.2*full-variable,
                    description="K covers teacher access, fitting, validation/selection and shared setup of all deployed modules over all three releases. K and its calibration population are not inferred from the serving population or assumed zero."),
                storage=dict(code_relative_to_native_kv_percent=100*code_bytes/native_cache_bytes,
                    code_bytes_per_full_user=code_bytes, native_kv_bytes_per_full_user=native_cache_bytes,
                    full_history_temporary_cat_bytes_per_layer_per_owner=
                        costs["context"]*(width+heads*rank)*element_bytes,
                    scope="Codes only. Existing native K/V and row IDs are retained; no expanded transformed K/V is persisted. Additional lifecycle metadata or shared model parameters are not included in this user-code ratio."),
                byte_work=dict(current_item_embedding_payload_lookup_bytes=lookup_bytes,
                    initialization_code_writes=initial*layers*heads*rank*element_bytes,
                    append_code_writes=appends*layers*heads*rank*element_bytes,
                    cat_source_read_plus_temporary_write_bytes_bounds=[2*packed_lower, 2*packed_upper],
                    packed_av_payload_read_bytes_bounds=[packed_lower, packed_upper],
                    scope="Payload arithmetic for the explicit torch.cat implementation; not measured HBM traffic. Bounds reflect unknown query grouping within max batch32. No fused-kernel or latency credit."),
                population_projection={str(n):scale_projection(variable=variable, full=full,
                    users=costs["users"], target_users=n, shared_flops=shared_flops)
                    for n in (costs["users"], 100000, 1000000)},
                quality=None))
    return dict(status="prospective_arithmetic_only",
        primary_comparator="Each user rebuilds once at first request per release, then normal native service; Reuse is zero added work.",
        formula="C(N)=K+(N/30000)*S; B(N)=(N/30000)*B(30000). K is fixed shared calibration/setup and S includes user encoding, appends and extra query work.",
        population_projection_assumptions="Same three release windows, model, context and per-resident event/query/history mix. Analytical extrapolation only; no quality, latency or million-user evidence.",
        architecture=dict(layers=layers, heads=heads, head_dim=head_dim,
            encoder="Per head: z=SiLU(A * normalize([old_K;old_V;Current_item_head_slice])+b), A:3d->p.",
            read="Native QK and activation exactly once; one weights @ cat(V,z), then z-read @ B and native-read addition. B:p->d has no bias. No key correction or second historical QK.",
            response="Optional existing NonlinearResponseLayer([q; corrected_read/N]), 2D->h->D, with h=0 disabling it.",
            accounting_convention="Multiply-add=2 FLOPs; elementary activation=1, matching retained ledgers. Counts include normalization, biases, residual additions and Response length scaling.",
            state_policy="Conservative executed-shape policy: encode every resident history at release and encode every observed append. No lazy initialization, retirement or inactive-user savings are assumed.",
            dtype_element_bytes=element_bytes),
        workload={k:costs[k] for k in ("users", "requests", "active_users", "days", "model_chain", "context", "query_batch")},
        first_request_rebuild_flops=full,
        shared_calibration_flops=None,
        candidates=candidates,
        limits=["No training success, quality target or complete-method acceptance is claimed.",
            "The candidate requires a single native attention execution; a callback that repeats native QK/AV would invalidate these estimates.",
            "Unknown calibration and any subsequently added coordination/lifecycle computation must be charged before complete cost can be evaluated.",
            "Small FLOPs alone do not establish small latency: concatenation copies and kernel launches remain in the initial implementation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/
        "results/design_system_2026_10/lightweight_read/prospective_cost.json")
    parser.add_argument("--shared-work-tflops", type=float,
        help="Optional independently measured/estimated total calibration+setup cost K over all releases; never inferred from N.")
    parser.add_argument("--element-bytes", type=int, default=4, choices=(2, 4))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shared_work_tflops is not None and args.shared_work_tflops < 0:
        raise ValueError("shared work cannot be negative")
    paths = [PHASE/"population_cost/summary_complete.json",
             PHASE/"once_rebuild_reassessment.json",
             ROOT/"results/design_two_2026_10/benchmark_panel/binding.json"]
    costs, reassessment, panel = (json.loads(path.read_text()) for path in paths)
    report = build_report(costs, reassessment, panel,
        shared_flops=None if args.shared_work_tflops is None else args.shared_work_tflops*1e12,
        element_bytes=args.element_bytes)
    report["source_sha256"] = {str(path):sha256(path) for path in paths}
    report["code_sha256"] = {str(path):sha256(path) for path in
        (Path(__file__).resolve(), Path(__file__).with_name("once_rebuild_accounting.py"),
         ROOT/"src/hstu_kvcache/design_one/nonlinear.py",
         ROOT/"src/hstu_kvcache/design_one/lightweight.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps([dict(name=r["name"], variable_cost_percent=r["variable_percent_of_once_rebuild"],
        query_extra_percent=r["query"]["extra_percent_of_native_whole_query"],
        shared_work_headroom_tflops=r["shared_work"]["maximum_at_20_percent_flops"]/1e12)
        for r in report["candidates"]], indent=2))


if __name__ == "__main__":
    main()

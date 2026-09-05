#!/usr/bin/env python3
"""Run the formal 32-user paired functional-delta canary.

This canary deliberately separates three scientific objects:

* ``representation_*`` rows use complete Current Exact state and test whether
  the positive-region response delta is a compact functional representation;
* ``oracle_*`` rows use Current Exact upper-layer K/V only at selected carrier
  positions and test whether the delta is compressible before construction;
* ``native_parent_conditioned_*`` and ``native_causal_closure_*`` rows are a
  matched executable ablation.  Only the latter lets earlier paired
  functional deltas causally form later Current carriers.

The recursive dependency closure is the Design candidate.  Affine moments and
landmark selection remain diagnostics rather than novelty claims.  Current
Exact is instantiated only after every legal memory has been built and is used
solely for evaluation and explicitly typed oracle rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402

from insight_two.causal_delta_closure import (  # noqa: E402
    build_causal_delta_closure,
    build_native_pair_memory,
)
from insight_two.common import (  # noqa: E402
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HELDOUT_INDICES,
    HISTORY,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    RESULT_ROOT,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_model_payload,
)
from insight_two.common import (  # noqa: E402
    verify_contract as verify_boundary_contract,
)
from insight_two.cone_response_memory import intervene_cone_response_memory  # noqa: E402
from insight_two.paired_region_delta import (  # noqa: E402
    build_paired_region_delta_memory,
    causal_delta_closure_cost,
    certify_nested_moment_disagreement,
    exact_cache_samples,
    project_full_current_layer0,
    select_legal_layer0_address_landmarks,
    trace_history_item_region_queries,
)
from insight_two.signed_response_memory import (  # noqa: E402
    intervene_oracle_signed_response_memory,
)

CONTRACT = (
    ROOT / "configs/contracts/"
    "yambda500m_medium_legacy_pointwise_insight2_paired_functional_delta_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "diagnostic_paired_functional_delta_v1"
CANARY_USERS = 32
PRIMARY_PROBES = 8
REPRESENTATION_PROBES = (8, 32)
CARRIER_COUNTS = (64, 128)
EXPECTED_LAYERS = 6
EXPECTED_HEADS = 6
EXPECTED_HIDDEN = 192
EXPECTED_MOMENT_SCALARS = 38_016
METHODS_PER_USER_EDGE = 12
SCORE_METRIC_COLUMNS = (
    "reuse_logit_gap",
    "observed_logit_gap",
    "logit_gap_recovery",
    "reuse_probability_gap",
    "observed_probability_gap",
    "probability_gap_recovery",
    "bernoulli_js_to_exact",
    "top1_agreement",
    "top10_overlap",
    "rank_correlation",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cost_grid(*, context: int = HISTORY) -> list[dict[str, Any]]:
    return [
        causal_delta_closure_cost(
            layers=EXPECTED_LAYERS,
            hidden=EXPECTED_HIDDEN,
            heads=EXPECTED_HEADS,
            context=context,
            carriers=carriers,
            recursive_delta=recursive,
        )
        for carriers in CARRIER_COUNTS
        for recursive in (False, True)
    ]


def verify_contract() -> tuple[dict[str, Any], str]:
    """Verify the prospective contract once it has been reviewed and frozen."""

    verify_boundary_contract()
    if not CONTRACT.is_file():
        raise FileNotFoundError(f"paired functional-delta contract has not been frozen: {CONTRACT}")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    scope = contract["scope"]
    if scope["edges"] != list(EDGES) or scope["cutover_days"] != list(CUTOVER_DAYS):
        raise RuntimeError("paired-delta edge scope differs")
    if int(scope["history_positions"]) != HISTORY:
        raise RuntimeError("paired-delta history width differs")
    if scope["attention_activation"] != "elu_plus1":
        raise RuntimeError("paired-delta activation differs")
    if scope["block_variant"] != "legacy" or bool(scope["relative_position_bias"]):
        raise RuntimeError("paired-delta legacy/no-bias architecture differs")

    population = contract["population"]
    if population["canary_indices_half_open"] != [0, CANARY_USERS]:
        raise RuntimeError("paired-delta canary population differs")
    if population["confirmation_indices_half_open"] != [512, 3000]:
        raise RuntimeError("paired-delta confirmation holdout differs")

    construction = contract["construction"]
    if tuple(construction["probe_counts"]) != REPRESENTATION_PROBES:
        raise RuntimeError("paired-delta probe-count axis differs")
    if int(construction["primary_probe_count"]) != PRIMARY_PROBES:
        raise RuntimeError("paired-delta primary probe count differs")
    if tuple(construction["carrier_counts"]) != CARRIER_COUNTS:
        raise RuntimeError("paired-delta carrier-count axis differs")
    if construction["probe_source"] != "fixed_equal_width_history_item_ids":
        raise RuntimeError("paired-delta probe source differs")

    heldout = contract["candidate_evaluation"]
    if tuple(heldout["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("paired-delta held-out panel differs")
    if heldout["construction_candidates"] is not None:
        raise RuntimeError("paired-delta construction must be candidate-free")

    boundary = contract["claim_boundary"]
    required_false = (
        "moments_sampling_or_clustering_is_novelty",
        "design1_admitted_by_this_canary",
    )
    if any(boundary.get(name) is not False for name in required_false):
        raise RuntimeError("paired-delta claim boundary is too broad")
    if boundary.get("legal_path_is_nonrecursive_lower_bound") is not True:
        raise RuntimeError("legal Parent-conditioned path must remain a lower bound")
    if boundary.get("recursive_causal_closure_implemented") is not True:
        raise RuntimeError("paired-delta contract must type the recursive closure")

    execution = contract["execution"]
    if execution["GPUs"] != [0, 1, 2, 3]:
        raise RuntimeError("paired-delta GPU allowlist differs")
    if int(execution["canary_users"]) != CANARY_USERS:
        raise RuntimeError("paired-delta canary user count differs")
    if contract["outputs"]["root"] != OUTPUT_ROOT.relative_to(ROOT).as_posix():
        raise RuntimeError("paired-delta output root differs")

    expected_costs = {
        (int(row["carriers"]), bool(row["recursive_delta"])): row
        for row in contract["theoretical_compute"]["grid"]
    }
    actual_costs = {
        (int(row["carriers"]), bool(row["recursive_delta"])): row for row in _cost_grid()
    }
    if expected_costs.keys() != actual_costs.keys():
        raise RuntimeError("paired-delta cost carrier axis differs")
    for key, actual in actual_costs.items():
        expected = expected_costs[key]
        for name in (
            "total_generation_flops_per_user",
            "full_recompute_flops_per_user",
            "over_full_fraction",
        ):
            if expected[name] != actual[name]:
                raise RuntimeError(f"paired-delta {key} theoretical cost differs: {name}")

    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen paired-delta input differs: {path}")
    return contract, sha256_file(CONTRACT)


def verify_model(model, payload: dict[str, Any]) -> None:
    verify_model_payload(payload)
    cfg = model.cfg
    expected = {
        "hidden_size": EXPECTED_HIDDEN,
        "num_layers": EXPECTED_LAYERS,
        "num_heads": EXPECTED_HEADS,
        "max_seq_len": HISTORY,
    }
    for name, value in expected.items():
        if int(getattr(cfg, name)) != value:
            raise RuntimeError(f"model {name} differs: {getattr(cfg, name)}")
    if cfg.activation != "elu_plus1" or cfg.block_variant != "legacy":
        raise RuntimeError("model is not the sealed legacy ELU+1 path")
    if bool(cfg.relative_position_bias):
        raise RuntimeError("model unexpectedly has relative-position bias")
    if model.training:
        raise RuntimeError("loaded model must be in eval mode")


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("paired functional-delta canary requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"paired functional-delta canary requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def _memory_row(
    *,
    edge: str,
    uid: int,
    method: str,
    evidence_class: str,
    probes: int,
    carriers: int,
    persistent_scalars: int,
    persistent_metadata_bytes: int,
    materialized_scalars: int,
    full_kv_scalars: int,
    current_exact_for_construction: bool,
    constructor_is_legal: bool,
    recursive_causal_closure: bool,
    theoretical_cost: dict[str, Any] | None,
    exact_scores: torch.Tensor,
    reuse_scores: torch.Tensor,
    observed_scores: torch.Tensor,
) -> dict[str, Any]:
    return {
        "edge": edge,
        "uid": uid,
        "source": "heldout_odd32",
        "method": method,
        "evidence_class": evidence_class,
        "probe_count": probes,
        "carrier_count": carriers,
        "persistent_incremental_scalars": persistent_scalars,
        "persistent_metadata_bytes": persistent_metadata_bytes,
        "persistent_incremental_bytes_fp32": (4 * persistent_scalars + persistent_metadata_bytes),
        "persistent_ratio_to_full_KV": (4 * persistent_scalars + persistent_metadata_bytes)
        / (4 * full_kv_scalars),
        "materialized_intervention_scalars": materialized_scalars,
        "materialized_intervention_ratio_to_full_KV": (materialized_scalars / full_kv_scalars),
        "current_exact_for_construction": current_exact_for_construction,
        "current_exact_for_evaluation": True,
        "constructor_is_legal": constructor_is_legal,
        "recursive_causal_closure": recursive_causal_closure,
        "theoretical_neural_compute_fraction": (
            None
            if theoretical_cost is None
            else theoretical_cost.get(
                "neural_over_full_fraction", theoretical_cost["over_full_fraction"]
            )
        ),
        "theoretical_selection_compute_fraction": (
            None
            if theoretical_cost is None
            else theoretical_cost.get("selection_over_full_fraction", 0.0)
        ),
        "theoretical_total_compute_fraction": (
            None if theoretical_cost is None else theoretical_cost["over_full_fraction"]
        ),
        "within_20_percent_total_compute": (
            None if theoretical_cost is None else theoretical_cost["over_full_fraction"] <= 0.20
        ),
        "carrier_position_sum": (
            None if theoretical_cost is None else theoretical_cost["carrier_position_sum"]
        ),
        "cost_semantics": (
            "not_applicable_oracle_or_baseline"
            if theoretical_cost is None
            else theoretical_cost["cost_semantics"]
        ),
        **metrics_row(score_metrics(exact_scores, reuse_scores, observed_scores)),
    }


def _cache_relative_l2(reference: torch.Tensor, observed: torch.Tensor) -> float:
    reference = reference.float().reshape(-1)
    observed = observed.float().reshape(-1)
    return float(
        torch.linalg.vector_norm(observed - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-20)
    )


@torch.inference_mode()
def evaluate_user(
    *,
    uid: int,
    edge: str,
    parent,
    current,
    items: torch.Tensor,
    behaviors: torch.Tensor,
    deltas: torch.Tensor,
    query_delta: torch.Tensor,
    heldout: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run the matched independent-versus-causal-closure experiment.

    All executable paths are constructed before ``Current Exact`` exists.
    R64 is the only primary budget point; R128 is diagnostic only.  The
    affine compiler consumes legal closure carriers but is not itself the
    causal mechanism or the novelty claim.
    """

    parent_cache = parent.compute_kv(items, behaviors, deltas)
    source_length = parent_cache.seq_len
    if source_length < max(CARRIER_COUNTS) or source_length % max(REPRESENTATION_PROBES):
        raise RuntimeError("history width does not support the carrier/probe grid")
    parent_before = (parent_cache.k.clone(), parent_cache.v.clone())

    probes_by_count = {
        probes: trace_history_item_region_queries(
            current, parent_cache, items, query_delta, probe_count=probes
        )
        for probes in REPRESENTATION_PROBES
    }
    layer0 = project_full_current_layer0(current, parent_cache, items, behaviors, deltas)
    selections = {
        carriers: select_legal_layer0_address_landmarks(layer0, parent_cache, sample_count=carriers)
        for carriers in CARRIER_COUNTS
    }
    nested = bool(
        torch.equal(
            selections[64].selected_positions,
            selections[128].selected_positions[:64],
        )
    )
    if not nested:
        raise RuntimeError("R64 landmarks are not the R128 nested prefix")

    # Same model, positions, masses and Parent control; only recursive_delta
    # changes.  These builds cannot receive an Exact Current cache.
    independent = {
        carriers: build_causal_delta_closure(
            current,
            parent_cache,
            items,
            behaviors,
            deltas,
            selections[carriers],
            current_layer0=layer0,
            recursive_delta=False,
        )
        for carriers in CARRIER_COUNTS
    }
    closure = {
        carriers: build_causal_delta_closure(
            current,
            parent_cache,
            items,
            behaviors,
            deltas,
            selections[carriers],
            current_layer0=layer0,
            recursive_delta=True,
        )
        for carriers in CARRIER_COUNTS
    }
    exact_layer0_ablation = build_causal_delta_closure(
        current,
        parent_cache,
        items,
        behaviors,
        deltas,
        selections[64],
        current_layer0=layer0,
        recursive_delta=True,
        layer0_prefix_mode="exact_current",
    )
    affine = {
        carriers: build_paired_region_delta_memory(
            current,
            closure[carriers].current_carriers,
            parent_cache,
            probes_by_count[PRIMARY_PROBES],
            closure[carriers].partition.positions,
            closure[carriers].partition.masses.to(device=items.device, dtype=parent_cache.v.dtype),
        )
        for carriers in CARRIER_COUNTS
    }
    affine_certificate = certify_nested_moment_disagreement(
        current, probes_by_count[PRIMARY_PROBES], affine[64], affine[128]
    )
    costs = {
        (carriers, recursive): causal_delta_closure_cost(
            layers=len(current.blocks),
            hidden=current.cfg.hidden_size,
            heads=current.cfg.num_heads,
            context=source_length,
            carriers=carriers,
            recursive_delta=recursive,
            carrier_position_sum=int(closure[carriers].partition.positions.long().sum().item()),
            temporal_freqs=current.cfg.temporal_num_freqs,
        )
        for carriers in CARRIER_COUNTS
        for recursive in (False, True)
    }

    # Exact state is evaluation-only for legal rows and construction-only for
    # explicitly typed representation/carrier oracles below this line.
    exact_cache = current.compute_kv(items, behaviors, deltas)
    exact_before = (exact_cache.k.clone(), exact_cache.v.clone())
    exact_scores, _ = current.observe_cc_reuse(exact_cache, heldout, query_delta)
    reuse_scores, _ = current.observe_cc_reuse(parent_cache, heldout, query_delta)
    full_kv_scalars = exact_cache.k.numel() + exact_cache.v.numel()
    full_positions = torch.arange(source_length, device=items.device)
    full_weights = torch.ones(source_length, device=items.device, dtype=exact_cache.v.dtype)
    representation = {
        probes: build_paired_region_delta_memory(
            current,
            exact_cache,
            parent_cache,
            probes_by_count[probes],
            full_positions,
            full_weights,
        )
        for probes in REPRESENTATION_PROBES
    }
    carrier_oracle = {
        carriers: build_native_pair_memory(
            exact_cache_samples(exact_cache, closure[carriers].partition.positions),
            parent_cache,
            closure[carriers].partition,
        )
        for carriers in CARRIER_COUNTS
    }

    observed: dict[str, torch.Tensor] = {}
    for probes, memory in representation.items():
        name = f"representation_full_affine_bulk_P{probes}"
        observed[name] = intervene_cone_response_memory(
            current, parent_cache, memory, heldout, query_delta
        ).scores
    for carriers, memory in carrier_oracle.items():
        name = f"carrier_oracle_native_R{carriers}"
        observed[name] = intervene_oracle_signed_response_memory(
            current, parent_cache, memory, heldout, query_delta
        ).scores
    for carriers, built in independent.items():
        name = f"native_parent_conditioned_R{carriers}"
        observed[name] = intervene_oracle_signed_response_memory(
            current, parent_cache, built.memory, heldout, query_delta
        ).scores
    for carriers, built in closure.items():
        name = f"native_causal_closure_R{carriers}"
        observed[name] = intervene_oracle_signed_response_memory(
            current, parent_cache, built.memory, heldout, query_delta
        ).scores
    observed["native_exact_layer0_closure_R64"] = intervene_oracle_signed_response_memory(
        current,
        parent_cache,
        exact_layer0_ablation.memory,
        heldout,
        query_delta,
    ).scores
    for carriers, memory in affine.items():
        name = f"closure_affine_compiler_R{carriers}_P8"
        observed[name] = intervene_cone_response_memory(
            current, parent_cache, memory, heldout, query_delta
        ).scores

    def add_row(
        name: str,
        evidence_class: str,
        *,
        probes: int,
        carriers: int,
        persistent: int,
        materialized: int,
        uses_exact: bool,
        legal: bool,
        recursive: bool,
        cost: dict[str, Any] | None,
    ) -> dict[str, Any]:
        native_carrier_state = evidence_class in {
            "carrier_state_oracle",
            "legal_independent_ablation",
            "legal_recursive_candidate",
            "legal_layer0_consistency_ablation",
        }
        return _memory_row(
            edge=edge,
            uid=uid,
            method=name,
            evidence_class=evidence_class,
            probes=probes,
            carriers=carriers,
            persistent_scalars=persistent,
            persistent_metadata_bytes=(16 * carriers if native_carrier_state else 0),
            materialized_scalars=materialized,
            full_kv_scalars=full_kv_scalars,
            current_exact_for_construction=uses_exact,
            constructor_is_legal=legal,
            recursive_causal_closure=recursive,
            theoretical_cost=cost,
            exact_scores=exact_scores,
            reuse_scores=reuse_scores,
            observed_scores=reuse_scores if name == "Current_Reuse" else observed[name],
        )

    records = [
        add_row(
            "Current_Reuse",
            "serving_baseline",
            probes=0,
            carriers=0,
            persistent=0,
            materialized=0,
            uses_exact=False,
            legal=True,
            recursive=False,
            cost=None,
        )
    ]
    for probes, memory in representation.items():
        name = f"representation_full_affine_bulk_P{probes}"
        records.append(
            add_row(
                name,
                "representation_oracle",
                probes=probes,
                carriers=source_length,
                persistent=memory.stored_scalars,
                materialized=memory.stored_scalars,
                uses_exact=True,
                legal=False,
                recursive=False,
                cost=None,
            )
        )
    for carriers, memory in carrier_oracle.items():
        name = f"carrier_oracle_native_R{carriers}"
        incremental = 2 * len(current.blocks) * carriers * current.cfg.hidden_size
        records.append(
            add_row(
                name,
                "carrier_state_oracle",
                probes=0,
                carriers=carriers,
                persistent=incremental,
                materialized=memory.keys.numel() + memory.signed_values.numel(),
                uses_exact=True,
                legal=False,
                recursive=False,
                cost=None,
            )
        )
    for carriers, built in independent.items():
        name = f"native_parent_conditioned_R{carriers}"
        records.append(
            add_row(
                name,
                "legal_independent_ablation",
                probes=0,
                carriers=carriers,
                persistent=built.current_carriers.k.numel() + built.current_carriers.v.numel(),
                materialized=built.memory.keys.numel() + built.memory.signed_values.numel(),
                uses_exact=False,
                legal=True,
                recursive=False,
                cost=costs[(carriers, False)],
            )
        )
    for carriers, built in closure.items():
        name = f"native_causal_closure_R{carriers}"
        records.append(
            add_row(
                name,
                "legal_recursive_candidate",
                probes=0,
                carriers=carriers,
                persistent=built.current_carriers.k.numel() + built.current_carriers.v.numel(),
                materialized=built.memory.keys.numel() + built.memory.signed_values.numel(),
                uses_exact=False,
                legal=True,
                recursive=True,
                cost=costs[(carriers, True)],
            )
        )
    records.append(
        add_row(
            "native_exact_layer0_closure_R64",
            "legal_layer0_consistency_ablation",
            probes=0,
            carriers=64,
            persistent=exact_layer0_ablation.current_carriers.k.numel()
            + exact_layer0_ablation.current_carriers.v.numel(),
            materialized=exact_layer0_ablation.memory.keys.numel()
            + exact_layer0_ablation.memory.signed_values.numel(),
            uses_exact=False,
            legal=True,
            recursive=True,
            # This locally exact layer-0 reader is not the primary executable
            # and receives no budget admission in this canary.
            cost=None,
        )
    )
    for carriers, memory in affine.items():
        name = f"closure_affine_compiler_R{carriers}_P8"
        records.append(
            add_row(
                name,
                "legal_affine_compiler_ablation",
                probes=PRIMARY_PROBES,
                carriers=carriers,
                persistent=memory.stored_scalars,
                materialized=memory.stored_scalars,
                uses_exact=False,
                legal=True,
                recursive=True,
                # Compiler overhead is intentionally not hidden inside the
                # native closure budget.  This row makes no compute admission.
                cost=None,
            )
        )

    projected_layer0_error = float(
        max(
            torch.max(torch.abs(layer0.k - exact_cache.k[0])).item(),
            torch.max(torch.abs(layer0.v - exact_cache.v[0])).item(),
        )
    )
    carrier_errors: dict[tuple[int, str, str], float] = {}
    closure_difference: dict[int, float] = {}
    for carriers in CARRIER_COUNTS:
        exact_selected = exact_cache_samples(exact_cache, closure[carriers].partition.positions)
        for kind, built in (
            ("independent", independent[carriers]),
            ("closure", closure[carriers]),
        ):
            carrier_errors[(carriers, kind, "layer0")] = float(
                max(
                    torch.max(torch.abs(built.current_carriers.k[0] - exact_selected.k[0])).item(),
                    torch.max(torch.abs(built.current_carriers.v[0] - exact_selected.v[0])).item(),
                )
            )
            carrier_errors[(carriers, kind, "upper")] = _cache_relative_l2(
                torch.cat((exact_selected.k[1:], exact_selected.v[1:]), dim=0),
                torch.cat(
                    (built.current_carriers.k[1:], built.current_carriers.v[1:]),
                    dim=0,
                ),
            )
        closure_difference[carriers] = _cache_relative_l2(
            torch.cat(
                (
                    independent[carriers].current_carriers.k[1:],
                    independent[carriers].current_carriers.v[1:],
                ),
                dim=0,
            ),
            torch.cat(
                (
                    closure[carriers].current_carriers.k[1:],
                    closure[carriers].current_carriers.v[1:],
                ),
                dim=0,
            ),
        )
    exact_selected64 = exact_cache_samples(exact_cache, exact_layer0_ablation.partition.positions)
    exact_layer0_ablation_upper_l2 = _cache_relative_l2(
        torch.cat((exact_selected64.k[1:], exact_selected64.v[1:]), dim=0),
        torch.cat(
            (
                exact_layer0_ablation.current_carriers.k[1:],
                exact_layer0_ablation.current_carriers.v[1:],
            ),
            dim=0,
        ),
    )

    diagnostics: dict[str, Any] = {
        "edge": edge,
        "uid": uid,
        "probe_source": "fixed_equal_width_history_item_ids",
        "construction_candidates_read": False,
        "recursive_causal_closure_implemented": True,
        "R64_is_primary_budget_point": True,
        "R128_is_diagnostic_only": True,
        "primary_layer0_prefix_mode": "paired_closure",
        "exact_layer0_prefix_is_ablation_only": True,
        "exact_layer0_ablation_upper_relative_l2_to_Exact": (exact_layer0_ablation_upper_l2),
        "R64_prefix_of_R128": nested,
        "full_layer0_projection_max_abs_error": projected_layer0_error,
        "closure_affine_R64_R128_certificate_relative_l2": (affine_certificate.relative_l2),
        "closure_affine_R64_R128_certificate_cosine": affine_certificate.cosine,
    }
    for carriers in CARRIER_COUNTS:
        partition = closure[carriers].partition
        coverage = closure[carriers].represented_prefix_fractions.float()
        for recursive, label in ((False, "independent"), (True, "closure")):
            cost = costs[(carriers, recursive)]
            diagnostics.update(
                {
                    f"R{carriers}_{label}_neural_compute_fraction": float(
                        cost["neural_over_full_fraction"]
                    ),
                    f"R{carriers}_{label}_selection_compute_fraction": float(
                        cost["selection_over_full_fraction"]
                    ),
                    f"R{carriers}_{label}_total_compute_fraction": float(
                        cost["total_over_full_fraction"]
                    ),
                    f"R{carriers}_{label}_within_20_percent": bool(
                        cost["within_20_percent_at_reported_position_sum"]
                    ),
                    f"R{carriers}_{label}_within_20_for_all_position_sets": bool(
                        cost["within_20_percent_for_all_unique_position_sets"]
                    ),
                }
            )
        diagnostics.update(
            {
                f"R{carriers}_cluster_mass_sum": int(partition.masses.sum().item()),
                f"R{carriers}_cluster_mass_min": int(partition.masses.min().item()),
                f"R{carriers}_carrier_position_sum": int(partition.positions.sum().item()),
                f"R{carriers}_represented_prefix_coverage_mean": float(coverage.mean()),
                f"R{carriers}_represented_prefix_coverage_min": float(coverage.min()),
                f"R{carriers}_represented_prefix_coverage_p10": float(
                    torch.quantile(coverage, 0.10)
                ),
                f"R{carriers}_independent_layer0_max_abs_error": carrier_errors[
                    (carriers, "independent", "layer0")
                ],
                f"R{carriers}_closure_layer0_max_abs_error": carrier_errors[
                    (carriers, "closure", "layer0")
                ],
                f"R{carriers}_independent_upper_relative_l2_to_Exact": carrier_errors[
                    (carriers, "independent", "upper")
                ],
                f"R{carriers}_closure_upper_relative_l2_to_Exact": carrier_errors[
                    (carriers, "closure", "upper")
                ],
                f"R{carriers}_closure_vs_independent_upper_relative_l2": (
                    closure_difference[carriers]
                ),
            }
        )

    score_values = [
        value
        for record in records
        for name, value in record.items()
        if name in SCORE_METRIC_COLUMNS
    ]
    numeric_diagnostics = [
        value
        for name, value in diagnostics.items()
        if name not in {"edge", "probe_source"} and not isinstance(value, bool)
    ]
    correctness = {
        "edge": edge,
        "uid": uid,
        "legal_construction_completed_before_Current_Exact": True,
        "legal_constructor_Current_Exact_argument": False,
        "construction_candidates_read": False,
        "recursive_causal_closure_implemented": True,
        "primary_layer0_prefix_mode_is_paired_closure": bool(
            all(built.layer0_prefix_mode == "paired_closure" for built in closure.values())
            and exact_layer0_ablation.layer0_prefix_mode == "exact_current"
        ),
        "finite_all_paths": bool(
            all(
                torch.isfinite(scores).all()
                for scores in [exact_scores, reuse_scores, *observed.values()]
            )
            and all(np.isfinite(value) for value in score_values)
            and all(np.isfinite(value) for value in numeric_diagnostics)
        ),
        "parent_cache_unchanged": bool(
            torch.equal(parent_cache.k, parent_before[0])
            and torch.equal(parent_cache.v, parent_before[1])
        ),
        "exact_cache_unchanged": bool(
            torch.equal(exact_cache.k, exact_before[0])
            and torch.equal(exact_cache.v, exact_before[1])
        ),
        "R64_prefix_of_R128": nested,
        "cluster_mass_sums_equal_history": bool(
            all(built.partition.masses.sum().item() == source_length for built in closure.values())
        ),
        "cluster_masses_positive": bool(
            all(torch.all(built.partition.masses > 0) for built in closure.values())
        ),
        "represented_prefix_coverage_in_unit_interval": bool(
            all(
                torch.all(built.represented_prefix_fractions >= 0)
                and torch.all(built.represented_prefix_fractions <= 1)
                for built in closure.values()
            )
        ),
        "full_layer0_projection_within_tolerance": projected_layer0_error <= 2e-5,
        "all_legal_carrier_layer0_within_tolerance": all(
            value <= 2e-5
            for (carriers, kind, stage), value in carrier_errors.items()
            if stage == "layer0"
        ),
        "native_cost_rows_complete": all(
            record["theoretical_total_compute_fraction"] is not None
            for record in records
            if record["evidence_class"]
            in {"legal_independent_ablation", "legal_recursive_candidate"}
        ),
        "persistent_scalars_exact": all(
            memory.stored_scalars == EXPECTED_MOMENT_SCALARS
            for memory in (*representation.values(), *affine.values())
        ),
        "method_count": len(records),
    }
    return records, diagnostics, correctness


def _validate_combined_rows(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    correctness: pd.DataFrame,
    *,
    users: int,
) -> bool:
    expected_user_edges = users * len(EDGES)
    if len(metrics) != expected_user_edges * METHODS_PER_USER_EDGE:
        raise RuntimeError("paired-delta metric row count differs")
    if len(diagnostics) != expected_user_edges or len(correctness) != expected_user_edges:
        raise RuntimeError("paired-delta diagnostic row count differs")
    if metrics[["edge", "uid", "method"]].duplicated().any():
        raise RuntimeError("duplicate paired-delta metric key")
    if diagnostics[["edge", "uid"]].duplicated().any():
        raise RuntimeError("duplicate paired-delta diagnostic key")

    independent = metrics[metrics.evidence_class == "legal_independent_ablation"]
    closure = metrics[metrics.evidence_class == "legal_recursive_candidate"]
    compiler = metrics[metrics.evidence_class == "legal_affine_compiler_ablation"]
    layer0_ablation = metrics[metrics.evidence_class == "legal_layer0_consistency_ablation"]
    oracle = metrics[metrics.evidence_class.isin(["representation_oracle", "carrier_state_oracle"])]
    evidence_typing = bool(
        len(independent) == expected_user_edges * len(CARRIER_COUNTS)
        and len(closure) == expected_user_edges * len(CARRIER_COUNTS)
        and len(compiler) == expected_user_edges * len(CARRIER_COUNTS)
        and len(layer0_ablation) == expected_user_edges
        and independent.constructor_is_legal.all()
        and closure.constructor_is_legal.all()
        and compiler.constructor_is_legal.all()
        and layer0_ablation.constructor_is_legal.all()
        and not independent.current_exact_for_construction.any()
        and not closure.current_exact_for_construction.any()
        and not compiler.current_exact_for_construction.any()
        and not layer0_ablation.current_exact_for_construction.any()
        and not independent.recursive_causal_closure.any()
        and closure.recursive_causal_closure.all()
        and compiler.recursive_causal_closure.all()
        and layer0_ablation.recursive_causal_closure.all()
        and not oracle.constructor_is_legal.any()
        and oracle.current_exact_for_construction.all()
        and not oracle.recursive_causal_closure.any()
    )
    return bool(
        evidence_typing
        and correctness.finite_all_paths.all()
        and correctness.legal_construction_completed_before_Current_Exact.all()
        and not correctness.legal_constructor_Current_Exact_argument.any()
        and not correctness.construction_candidates_read.any()
        and correctness.recursive_causal_closure_implemented.all()
        and correctness.primary_layer0_prefix_mode_is_paired_closure.all()
        and correctness.parent_cache_unchanged.all()
        and correctness.exact_cache_unchanged.all()
        and correctness.R64_prefix_of_R128.all()
        and correctness.cluster_mass_sums_equal_history.all()
        and correctness.cluster_masses_positive.all()
        and correctness.full_layer0_projection_within_tolerance.all()
        and correctness.represented_prefix_coverage_in_unit_interval.all()
        and correctness.all_legal_carrier_layer0_within_tolerance.all()
        and correctness.native_cost_rows_complete.all()
        and correctness.persistent_scalars_exact.all()
        and (correctness.method_count == METHODS_PER_USER_EDGE).all()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary",), default="canary")
    args = parser.parse_args()
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()

    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            _, contract_hash = verify_contract()
            verification[0] = {"ok": True, "contract_sha256": contract_hash}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract verification failed: {verification[0]['error']}")
    contract_hash = str(verification[0]["contract_sha256"])

    all_uids, all_candidates, _ = load_frozen_inputs()
    selected_indices = np.arange(CANARY_USERS, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = OUTPUT_ROOT / args.scope
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": contract_hash,
                "scope": args.scope,
                "users": CANARY_USERS,
                "edges": list(EDGES),
                "probe_counts": list(REPRESENTATION_PROBES),
                "primary_probe_count": PRIMARY_PROBES,
                "carrier_counts": list(CARRIER_COUNTS),
                "probe_source": "fixed_equal_width_history_item_ids",
                "construction_candidates": None,
                "evaluation_candidate_indices": list(HELDOUT_INDICES),
                "labels_read": False,
                "representation_oracle_uses_Current_Exact": True,
                "legal_Parent_conditioned_path_uses_Current_Exact": False,
                "legal_path_is_nonrecursive_lower_bound": True,
                "recursive_causal_closure_implemented": True,
                "primary_layer0_prefix_mode": "paired_closure",
                "exact_Current_layer0_prefix_mode": "R64_ablation_only",
                "R64_is_only_primary_budget_point": True,
                "R128_is_diagnostic_only": True,
                "moments_sampling_or_clustering_is_novelty": False,
            },
        )
        atomic_json(
            partial / "theoretical_compute.json",
            {
                "grid_at_default_position_sum": _cost_grid(),
                "actual_per_user_position_sums_are_in_raw_rows": True,
                "scope": "native_independent_and_recursive_closure",
                "diagnostic_affine_compiler_cost_not_admitted": True,
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    histories = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=8,
    )
    metric_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0

    for edge_index, edge in enumerate(EDGES):
        edge_started = time.perf_counter()
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model(parent, parent_payload)
        verify_model(current, current_payload)
        _, items_np, behaviors_np, deltas_np, query_np = histories_at_cutover(
            histories, local_uids, CUTOVER_DAYS[edge_index] * DAY
        )
        heldout_np = all_candidates[edge_index, local_indices][:, HELDOUT_INDICES]
        torch.cuda.reset_peak_memory_stats(device)

        for offset, uid_value in enumerate(local_uids):
            items = torch.as_tensor(items_np[offset : offset + 1], dtype=torch.long, device=device)
            behaviors = torch.as_tensor(
                behaviors_np[offset : offset + 1], dtype=torch.long, device=device
            )
            deltas = torch.as_tensor(
                deltas_np[offset : offset + 1], dtype=torch.float32, device=device
            )
            query_delta = torch.as_tensor(
                query_np[offset : offset + 1], dtype=torch.float32, device=device
            )
            heldout = torch.as_tensor(
                heldout_np[offset : offset + 1], dtype=torch.long, device=device
            )
            metrics, diagnostics, correctness = evaluate_user(
                uid=int(uid_value),
                edge=edge,
                parent=parent,
                current=current,
                items=items,
                behaviors=behaviors,
                deltas=deltas,
                query_delta=query_delta,
                heldout=heldout,
            )
            for row in metrics:
                row["user_index"] = int(local_indices[offset])
            diagnostics["user_index"] = int(local_indices[offset])
            correctness["user_index"] = int(local_indices[offset])
            metric_records.extend(metrics)
            diagnostic_records.append(diagnostics)
            correctness_records.append(correctness)
            peak_allocated_mib = max(
                peak_allocated_mib,
                torch.cuda.max_memory_allocated(device) / (1 << 20),
            )
            peak_reserved_mib = max(
                peak_reserved_mib,
                torch.cuda.max_memory_reserved(device) / (1 << 20),
            )

        seconds = time.perf_counter() - edge_started
        edge_records.append({"edge": edge, "users": len(local_uids), "seconds": seconds})
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    "edge": edge,
                    "users": len(local_uids),
                    "seconds": seconds,
                }
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload
        torch.cuda.empty_cache()

    pd.DataFrame(metric_records).to_parquet(rank_output / "metrics.parquet", index=False)
    pd.DataFrame(diagnostic_records).to_parquet(rank_output / "diagnostics.parquet", index=False)
    pd.DataFrame(correctness_records).to_parquet(rank_output / "correctness.parquet", index=False)
    atomic_json(
        rank_output / "summary.json",
        {
            "rank": rank,
            "uids": local_uids.tolist(),
            "edge_records": edge_records,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "labels_read": False,
        },
    )
    dist.barrier()

    if rank == 0:
        metrics = pd.concat(
            [pd.read_parquet(partial / f"rank{shard}/metrics.parquet") for shard in range(world)],
            ignore_index=True,
        ).sort_values(["edge", "user_index", "method"])
        diagnostics = pd.concat(
            [
                pd.read_parquet(partial / f"rank{shard}/diagnostics.parquet")
                for shard in range(world)
            ],
            ignore_index=True,
        ).sort_values(["edge", "user_index"])
        correctness = pd.concat(
            [
                pd.read_parquet(partial / f"rank{shard}/correctness.parquet")
                for shard in range(world)
            ],
            ignore_index=True,
        ).sort_values(["edge", "user_index"])
        passed = _validate_combined_rows(metrics, diagnostics, correctness, users=CANARY_USERS)
        combined = {
            "metrics.parquet": metrics,
            "diagnostics.parquet": diagnostics,
            "correctness.parquet": correctness,
        }
        artifacts: dict[str, Any] = {}
        for name, frame in combined.items():
            path = partial / name
            frame.to_parquet(path, index=False)
            artifacts[name] = {
                "rows": len(frame),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        rank_summaries = [
            json.loads((partial / f"rank{shard}/summary.json").read_text(encoding="utf-8"))
            for shard in range(world)
        ]
        summary = {
            "status": (
                "paired_functional_delta_instrumentation_passed"
                if passed
                else "paired_functional_delta_instrumentation_failed"
            ),
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": CANARY_USERS,
            "edges": list(EDGES),
            "labels_read": False,
            "construction_candidates_read": False,
            "representation_oracle_uses_Current_Exact": True,
            "constructor_oracle_uses_Current_Exact_upper_layers": True,
            "legal_Parent_conditioned_path_uses_Current_Exact": False,
            "legal_path_is_nonrecursive_lower_bound": True,
            "recursive_causal_closure_implemented": True,
            "primary_layer0_prefix_mode": "paired_closure",
            "design1_gate": "pending_adjudication_closure_gain_required",
            "novelty_gate": "not_adjudicated_moments_and_selection_are_not_novelty",
            "elapsed_seconds": max(float(row["elapsed_seconds"]) for row in rank_summaries),
            "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in rank_summaries),
            "peak_reserved_mib": max(float(row["peak_reserved_mib"]) for row in rank_summaries),
            "artifacts": artifacts,
        }
        atomic_json(partial / "summary.json", summary)
        partial.replace(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

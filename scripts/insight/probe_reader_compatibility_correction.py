#!/usr/bin/env python3
"""Prospective stage localization for the HSTU reader compatibility correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from probe_candidate_shared_causal import (  # noqa: E402
    exposed_canary_groups,
    rolling_caches_for_exposed_group,
)
from probe_recommendation_state_structure import (  # noqa: E402
    CUTOVER_DAYS,
    DAY,
    HISTORY,
    KNOWN_ITEMS,
    candidate_panel,
    checkpoint,
    histories_at_cutover,
)
from reader_compatibility_correction import STAGES, trace_reader_correction  # noqa: E402
from candidate_shared_causal import nested_width_indices  # noqa: E402
from probe_candidate_shared_causal import rank_correlation  # noqa: E402


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_reader_compatibility_correction_v1.yaml"
POPULATION = ROOT / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/population.parquet"
OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_reader_compatibility_correction_v1"
WIDTHS = (64, 32, 16, 8)
REAL_WIDTHS = (16, 8, 4, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text())
    for name, record in contract["frozen_inputs"].items():
        if name == "checkpoints":
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen {name} differs from the prospective contract")
        required = record.get("required_status")
        if required and json.loads(path.read_text())["status"] != required:
            raise RuntimeError(f"frozen {name} status differs")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"v{version} checkpoint differs")
    return contract


def _score_rows(
    *, edge: str, source: str, identifiers: np.ndarray, width: int, trace
) -> list[dict[str, Any]]:
    exact = trace.exact_scores.float()
    reuse = trace.reuse_scores.float()
    exact_probability = torch.sigmoid(exact)
    reuse_gap = torch.abs(torch.sigmoid(reuse) - exact_probability).mean(dim=1)
    exact_top1 = exact.argmax(dim=1)
    output = []
    paths = {"reuse": reuse, **trace.stage_scores}
    for stage, scores in paths.items():
        gap = torch.abs(torch.sigmoid(scores.float()) - exact_probability).mean(dim=1)
        correlation = rank_correlation(exact, scores.float())
        for index, identifier in enumerate(identifiers):
            output.append({
                "edge": edge,
                "source": source,
                "identifier": str(identifier),
                "width": width,
                "stage": stage,
                "mean_abs_logit_gap": float(torch.abs(scores[index] - exact[index]).mean()),
                "mean_abs_probability_gap": float(gap[index]),
                "gap_recovery_over_reuse": float(
                    0.0 if stage == "reuse" else 1.0 - gap[index] / reuse_gap[index].clamp_min(1e-12)
                ),
                "top1_agreement": int(scores[index].argmax() == exact_top1[index]),
                "rank_correlation": float(correlation[index]),
                "reuse_probability_gap": float(reuse_gap[index]),
            })
    return output


def _energy_rows(
    *, edge: str, source: str, identifiers: np.ndarray, width: int, trace
) -> list[dict[str, Any]]:
    output = []
    for metrics in trace.energy_metrics:
        for index, identifier in enumerate(identifiers):
            output.append({
                "edge": edge,
                "source": source,
                "identifier": str(identifier),
                "width": width,
                "stage": metrics["stage"],
                "layer": int(metrics["layer"]),
                "signed_shared_energy_fraction": float(metrics["shared_energy_fraction"][index]),
                "signed_residual_energy_fraction": float(metrics["residual_energy_fraction"][index]),
                "orthogonality_error": float(metrics["orthogonality_error"][index]),
                "total_energy": float(metrics["total_energy"][index]),
            })
    return output


def _evaluate(
    *, current, exact_cache, reuse_cache, candidates, query_deltas, edge, source,
    identifiers, width, score_records, energy_records, correctness_records,
) -> None:
    trace = trace_reader_correction(
        current, exact_cache, reuse_cache, candidates, query_deltas
    )
    score_records.extend(_score_rows(
        edge=edge, source=source, identifiers=identifiers, width=width, trace=trace
    ))
    energy_records.extend(_energy_rows(
        edge=edge, source=source, identifiers=identifiers, width=width, trace=trace
    ))
    correctness_records.append({
        "edge": edge,
        "source": source,
        "width": width,
        **trace.correctness,
    })


def _report(summary: dict[str, Any], score: pd.DataFrame, energy: pd.DataFrame) -> str:
    scores = (
        score[score.stage != "reuse"]
        .groupby(["source", "edge", "width", "stage"], as_index=False)
        .agg(mean_gap_recovery=("gap_recovery_over_reuse", "mean"))
    )
    energies = (
        energy.groupby(["source", "edge", "width", "stage"], as_index=False)
        .agg(mean_shared_energy=("signed_shared_energy_fraction", "mean"))
    )
    focus_score = scores[scores.width == scores.groupby("source").width.transform("max")]
    focus_energy = energies[energies.width == energies.groupby("source").width.transform("max")]
    lines = [
        "# Reader compatibility-correction stage localization",
        "",
        f"Scope: `{summary['scope']}`; correctness: **{'PASS' if summary['correctness']['passed'] else 'FAIL'}**.",
        "",
        "The correction is a same-request signed oracle derived from coherent Current-Exact and Parent-Reuse reader traces. It is not a materialized history basis or an executable action.",
        "",
        "## Largest-width score-gap recovery",
        "",
        "| source | edge | stage | recovery |",
        "| --- | --- | --- | ---: |",
    ]
    for row in focus_score.itertuples(index=False):
        lines.append(f"| {row.source} | {row.edge} | {row.stage} | {row.mean_gap_recovery:.6f} |")
    lines.extend([
        "",
        "## Largest-width signed shared energy",
        "",
        "| source | edge | stage | shared energy |",
        "| --- | --- | --- | ---: |",
    ])
    for row in focus_energy.itertuples(index=False):
        lines.append(f"| {row.source} | {row.edge} | {row.stage} | {row.mean_shared_energy:.6f} |")
    lines.extend([
        "",
        f"Maximum native/full reconstruction error: {summary['correctness']['max_abs_error']:.8g}.",
        "No label was read.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("canary", "formal-controlled"), default="canary")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUT / ("canary_stage" if args.scope == "canary" else "formal_controlled_stage")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    started = time.time()
    contract = verify_contract()
    population = pd.read_parquet(POPULATION)
    uids = population.uid.to_numpy(dtype=np.int64)
    if len(uids) != 3000:
        raise RuntimeError("frozen population count differs")
    selected = uids if args.scope == "formal-controlled" else uids[:32]
    if args.scope == "formal-controlled":
        canary = json.loads((OUTPUT / "canary_stage/summary.json").read_text())
        if not canary["correctness"]["passed"] or canary["contract_sha256"] != sha256(CONTRACT):
            raise RuntimeError("stage canary did not unlock formal controlled scope")
    exposed = exposed_canary_groups(set(map(int, uids))) if args.scope == "canary" else {}

    device = torch.device(args.device)
    probe, payload = load_model(checkpoint(1), device)
    oov_buckets = int(payload["config"]["num_items"]) - KNOWN_ITEMS
    del probe
    torch.cuda.empty_cache()
    history = load_histories(uids.tolist(), oov_buckets=oov_buckets)
    score_records: list[dict[str, Any]] = []
    energy_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []

    for edge_index, cutover_day in enumerate(CUTOVER_DAYS):
        edge = f"v{edge_index}_to_v{edge_index + 1}"
        print(json.dumps({"phase": "edge_start", "edge": edge}), flush=True)
        cutover = cutover_day * DAY
        _, all_items, _, _, _ = histories_at_cutover(history, uids, cutover)
        full_panel, _, _ = candidate_panel(all_items)
        _, item_np, action_np, delta_np, query_delta_np = histories_at_cutover(
            history, selected, cutover
        )
        panel_np = full_panel[: len(selected)]
        parent, _ = load_model(checkpoint(edge_index), device)
        current, _ = load_model(checkpoint(edge_index + 1), device)
        controlled_widths = WIDTHS if args.scope == "formal-controlled" else (64, 16)
        for start in range(0, len(selected), args.batch_size):
            stop = min(start + args.batch_size, len(selected))
            items = torch.as_tensor(item_np[start:stop], dtype=torch.long, device=device)
            actions = torch.as_tensor(action_np[start:stop], dtype=torch.long, device=device)
            deltas = torch.as_tensor(delta_np[start:stop], dtype=torch.float32, device=device)
            query_deltas = torch.as_tensor(query_delta_np[start:stop], dtype=torch.float32, device=device)
            exact_cache = current.compute_kv(items, actions, deltas)
            reuse_cache = parent.compute_kv(items, actions, deltas)
            for width in controlled_widths:
                indices = nested_width_indices(64, width)
                candidates = torch.as_tensor(
                    panel_np[start:stop, indices], dtype=torch.long, device=device
                )
                _evaluate(
                    current=current, exact_cache=exact_cache, reuse_cache=reuse_cache,
                    candidates=candidates, query_deltas=query_deltas, edge=edge,
                    source="controlled", identifiers=selected[start:stop], width=width,
                    score_records=score_records, energy_records=energy_records,
                    correctness_records=correctness_records,
                )

        if args.scope == "canary":
            group = exposed[edge]
            exact_cache, reuse_cache, query_delta = rolling_caches_for_exposed_group(
                history, group["uid"], group["query_timestamp"], cutover,
                parent, current,
            )
            identifier = np.asarray([f"{group['uid']}:{group['query_timestamp']}"])
            for width in REAL_WIDTHS:
                indices = nested_width_indices(16, width)
                candidates = torch.as_tensor(
                    group["items"][indices][None], dtype=torch.long, device=device
                )
                _evaluate(
                    current=current, exact_cache=exact_cache, reuse_cache=reuse_cache,
                    candidates=candidates,
                    query_deltas=torch.tensor([query_delta], device=device),
                    edge=edge, source="real_exposed_canary", identifiers=identifier,
                    width=width, score_records=score_records,
                    energy_records=energy_records, correctness_records=correctness_records,
                )
        del parent, current
        torch.cuda.empty_cache()
        print(json.dumps({"phase": "edge_complete", "edge": edge}), flush=True)

    score = pd.DataFrame(score_records)
    energy = pd.DataFrame(energy_records)
    correctness = pd.DataFrame(correctness_records)
    max_error = float(correctness[
        ["native_exact", "native_reuse", "final_full_delta", "layer_stage_full_delta"]
    ].max().max())
    passed = max_error <= contract["focused_canary"]["correctness_max_abs_error"]
    summary = {
        "status": f"reader_correction_{args.scope.replace('-', '_')}_{'passed' if passed else 'failed'}",
        "scope": args.scope,
        "contract_sha256": sha256(CONTRACT),
        "controlled_users": len(selected),
        "controlled_edges": 5,
        "controlled_widths": list(controlled_widths),
        "real_exposed_groups": 5 if args.scope == "canary" else 0,
        "real_exposed_widths": list(REAL_WIDTHS) if args.scope == "canary" else [],
        "labels_read": False,
        "correctness": {"passed": passed, "max_abs_error": max_error},
        "elapsed_seconds": time.time() - started,
    }
    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    score.to_parquet(partial / "score_interventions.parquet", index=False)
    energy.to_parquet(partial / "stage_energy.parquet", index=False)
    correctness.to_csv(partial / "correctness.csv", index=False)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (partial / "report.md").write_text(_report(summary, score, energy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

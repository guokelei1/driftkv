#!/usr/bin/env python3
"""Four-GPU anchor-to-heldout rank-0 functional-boundary canary."""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight.reader_compatibility_correction import (  # noqa: E402
    STAGES,
    intervene_reader_correction,
    trace_reader_correction,
)
from insight_one_locality.common import histories_at_cutover  # noqa: E402
from insight_two.common import (  # noqa: E402
    ANCHOR_INDICES,
    CANARY_USERS,
    CONTRACT,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HELDOUT_INDICES,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    RESULT_ROOT,
    STAGE_PRESENTATION,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_contract,
    verify_model_payload,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("the frozen canary requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"the frozen canary requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def correction_numel(values: tuple[torch.Tensor, ...]) -> int:
    return sum(value[0].numel() for value in values)


def main() -> None:
    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()

    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            verify_contract()
            verification[0] = {"ok": True}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract verification failed: {verification[0]['error']}")

    all_uids, all_candidates, all_modes = load_frozen_inputs()
    selected_indices = np.arange(CANARY_USERS, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = RESULT_ROOT / "canary_rank0"
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": sha256_file(CONTRACT),
                "users": CANARY_USERS,
                "edges": list(EDGES),
                "anchor_indices": list(ANCHOR_INDICES),
                "heldout_indices": list(HELDOUT_INDICES),
                "stages": list(STAGES),
                "stage_presentation": STAGE_PRESENTATION,
                "labels_read": False,
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    print(
        json.dumps({"phase": "load_histories", "rank": rank, "users": len(local_uids)}),
        flush=True,
    )
    history_started = time.perf_counter()
    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=8,
    )
    history_seconds = time.perf_counter() - history_started
    score_records: list[dict[str, Any]] = []
    energy_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0

    for edge_index, edge in enumerate(EDGES):
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        edge_started = time.perf_counter()
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model_payload(parent_payload)
        verify_model_payload(current_payload)
        _, items_np, actions_np, deltas_np, query_deltas_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * DAY
        )
        candidate_np = all_candidates[edge_index, local_indices]
        mode_np = all_modes[edge_index, local_indices]
        torch.cuda.reset_peak_memory_stats(device)

        for local_offset, uid in enumerate(local_uids):
            user_started = time.perf_counter()
            items = torch.as_tensor(
                items_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            actions = torch.as_tensor(
                actions_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            deltas = torch.as_tensor(
                deltas_np[local_offset : local_offset + 1], dtype=torch.float32, device=device
            )
            query_delta = torch.as_tensor(
                query_deltas_np[local_offset : local_offset + 1],
                dtype=torch.float32,
                device=device,
            )
            anchor_candidates = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, ANCHOR_INDICES],
                dtype=torch.long,
                device=device,
            )
            heldout_candidates = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, HELDOUT_INDICES],
                dtype=torch.long,
                device=device,
            )
            parent_cache = parent.compute_kv(items, actions, deltas)
            current_cache = current.compute_kv(items, actions, deltas)
            anchor_trace = trace_reader_correction(
                current,
                current_cache,
                parent_cache,
                anchor_candidates,
                query_delta,
                verify_full_delta=True,
            )
            exact_heldout = current.score_cc_reuse(
                current_cache, heldout_candidates, query_delta
            )
            reuse_heldout = current.score_cc_reuse(
                parent_cache, heldout_candidates, query_delta
            )
            parent_exact_heldout = parent.score_cc_reuse(
                parent_cache, heldout_candidates, query_delta
            )

            baseline = score_metrics(exact_heldout, reuse_heldout, reuse_heldout)
            score_records.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    "source": "heldout",
                    "stage": "reuse",
                    "presentation": "Current_reader_Parent_state",
                    "correction_numel_fp32_per_user": 0,
                    "anchor_mode_0": int((mode_np[local_offset, ANCHOR_INDICES] == 0).sum()),
                    "anchor_mode_1": int((mode_np[local_offset, ANCHOR_INDICES] == 1).sum()),
                    "anchor_mode_2": int((mode_np[local_offset, ANCHOR_INDICES] == 2).sum()),
                    "heldout_mode_0": int((mode_np[local_offset, HELDOUT_INDICES] == 0).sum()),
                    "heldout_mode_1": int((mode_np[local_offset, HELDOUT_INDICES] == 1).sum()),
                    "heldout_mode_2": int((mode_np[local_offset, HELDOUT_INDICES] == 2).sum()),
                    "parent_exact_mean_abs_logit_gap": float(
                        torch.abs(parent_exact_heldout.float() - exact_heldout.float())
                        .mean()
                        .detach()
                    ),
                    **metrics_row(baseline),
                }
            )
            anchor_reuse_metrics = score_metrics(
                anchor_trace.exact_scores,
                anchor_trace.reuse_scores,
                anchor_trace.reuse_scores,
            )
            score_records.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    "source": "anchor_same_bank",
                    "stage": "reuse",
                    "presentation": "Current_reader_Parent_state",
                    "correction_numel_fp32_per_user": 0,
                    "anchor_mode_0": int((mode_np[local_offset, ANCHOR_INDICES] == 0).sum()),
                    "anchor_mode_1": int((mode_np[local_offset, ANCHOR_INDICES] == 1).sum()),
                    "anchor_mode_2": int((mode_np[local_offset, ANCHOR_INDICES] == 2).sum()),
                    "heldout_mode_0": int((mode_np[local_offset, HELDOUT_INDICES] == 0).sum()),
                    "heldout_mode_1": int((mode_np[local_offset, HELDOUT_INDICES] == 1).sum()),
                    "heldout_mode_2": int((mode_np[local_offset, HELDOUT_INDICES] == 2).sum()),
                    "parent_exact_mean_abs_logit_gap": float("nan"),
                    **metrics_row(anchor_reuse_metrics),
                }
            )

            for stage in STAGES:
                heldout_scores, _ = intervene_reader_correction(
                    current,
                    parent_cache,
                    heldout_candidates,
                    query_delta,
                    stage=stage,
                    corrections=anchor_trace.corrections[stage],
                )
                heldout_metrics = score_metrics(
                    exact_heldout, reuse_heldout, heldout_scores
                )
                anchor_metrics = score_metrics(
                    anchor_trace.exact_scores,
                    anchor_trace.reuse_scores,
                    anchor_trace.stage_scores[stage],
                )
                common = {
                    "edge": edge,
                    "uid": int(uid),
                    "stage": stage,
                    "presentation": STAGE_PRESENTATION[stage],
                    "correction_numel_fp32_per_user": correction_numel(
                        anchor_trace.corrections[stage]
                    ),
                    "anchor_mode_0": int((mode_np[local_offset, ANCHOR_INDICES] == 0).sum()),
                    "anchor_mode_1": int((mode_np[local_offset, ANCHOR_INDICES] == 1).sum()),
                    "anchor_mode_2": int((mode_np[local_offset, ANCHOR_INDICES] == 2).sum()),
                    "heldout_mode_0": int((mode_np[local_offset, HELDOUT_INDICES] == 0).sum()),
                    "heldout_mode_1": int((mode_np[local_offset, HELDOUT_INDICES] == 1).sum()),
                    "heldout_mode_2": int((mode_np[local_offset, HELDOUT_INDICES] == 2).sum()),
                    "parent_exact_mean_abs_logit_gap": float("nan"),
                }
                score_records.append(
                    {**common, "source": "heldout", **metrics_row(heldout_metrics)}
                )
                score_records.append(
                    {
                        **common,
                        "source": "anchor_same_bank",
                        **metrics_row(anchor_metrics),
                    }
                )
                del heldout_scores

            for metrics in anchor_trace.energy_metrics:
                stage = str(metrics["stage"])
                energy_records.append(
                    {
                        "edge": edge,
                        "uid": int(uid),
                        "stage": stage,
                        "presentation": STAGE_PRESENTATION[stage],
                        "layer": int(metrics["layer"]),
                        "candidate_mean_energy_fraction": float(
                            metrics["shared_energy_fraction"][0]
                        ),
                        "candidate_centered_energy_fraction": float(
                            metrics["residual_energy_fraction"][0]
                        ),
                        "orthogonality_error": float(metrics["orthogonality_error"][0]),
                        "total_energy": float(metrics["total_energy"][0]),
                    }
                )
            correctness_records.append(
                {
                    "edge": edge,
                    "uid": int(uid),
                    **anchor_trace.correctness,
                    "finite_exact": bool(torch.isfinite(exact_heldout).all()),
                    "finite_reuse": bool(torch.isfinite(reuse_heldout).all()),
                    "anchor_heldout_item_overlap": int(
                        len(
                            set(anchor_candidates[0].tolist())
                            & set(heldout_candidates[0].tolist())
                        )
                    ),
                    "user_seconds": time.perf_counter() - user_started,
                }
            )
            peak_allocated_mib = max(
                peak_allocated_mib, torch.cuda.max_memory_allocated(device) / (1 << 20)
            )
            peak_reserved_mib = max(
                peak_reserved_mib, torch.cuda.max_memory_reserved(device) / (1 << 20)
            )
            del (
                items,
                actions,
                deltas,
                query_delta,
                anchor_candidates,
                heldout_candidates,
                parent_cache,
                current_cache,
                anchor_trace,
                exact_heldout,
                reuse_heldout,
                parent_exact_heldout,
            )

        edge_seconds = time.perf_counter() - edge_started
        edge_records.append({"edge": edge, "users": len(local_uids), "seconds": edge_seconds})
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "rank": rank,
                    "edge": edge,
                    "users": len(local_uids),
                    "seconds": edge_seconds,
                }
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload
        torch.cuda.empty_cache()

    score_path = rank_output / "score_records.parquet"
    energy_path = rank_output / "energy_records.parquet"
    correctness_path = rank_output / "correctness.parquet"
    pd.DataFrame(score_records).to_parquet(score_path, index=False)
    pd.DataFrame(energy_records).to_parquet(energy_path, index=False)
    pd.DataFrame(correctness_records).to_parquet(correctness_path, index=False)
    rank_summary = {
        "status": "rank0_canary_rank_complete",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "uids": local_uids.tolist(),
        "labels_read": False,
        "history_seconds": history_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "edges": edge_records,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (score_path, energy_path, correctness_path)
        },
    }
    atomic_json(rank_output / "summary.json", rank_summary)
    dist.barrier()

    if rank == 0:
        summaries = [
            json.loads((partial / f"rank{value}/summary.json").read_text(encoding="utf-8"))
            for value in range(world)
        ]
        correctness = pd.concat(
            [
                pd.read_parquet(partial / f"rank{value}/correctness.parquet")
                for value in range(world)
            ],
            ignore_index=True,
        )
        observed_uids = [uid for summary in summaries for uid in summary["uids"]]
        max_error = float(
            correctness[
                ["native_exact", "native_reuse", "final_full_delta", "layer_stage_full_delta"]
            ].max().max()
        )
        gate = verify_contract()["gates"]["instrumentation"]
        passed = (
            len(observed_uids) == CANARY_USERS
            and len(set(observed_uids)) == CANARY_USERS
            and len(correctness) == CANARY_USERS * len(EDGES)
            and max_error <= float(gate["native_and_reconstructed_max_abs_logit_error"])
            and bool(correctness["finite_exact"].all())
            and bool(correctness["finite_reuse"].all())
            and int(correctness["anchor_heldout_item_overlap"].max()) == 0
        )
        elapsed = max(float(summary["elapsed_seconds"]) for summary in summaries)
        summary = {
            "status": "rank0_canary_complete" if passed else "rank0_canary_invalid",
            "passed": passed,
            "contract_sha256": sha256_file(CONTRACT),
            "users": CANARY_USERS,
            "edges": list(EDGES),
            "user_edges": len(correctness),
            "labels_read": False,
            "max_correctness_error": max_error,
            "elapsed_seconds": elapsed,
            "peak_allocated_mib": max(summary["peak_allocated_mib"] for summary in summaries),
            "peak_reserved_mib": max(summary["peak_reserved_mib"] for summary in summaries),
            "rough_512_user_seconds": elapsed * 512 / CANARY_USERS,
            "rank_summaries": [f"rank{value}/summary.json" for value in range(world)],
        }
        atomic_json(partial / "summary.json", summary)
        partial.rename(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

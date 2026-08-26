#!/usr/bin/env python3
"""Focused probe for the EvoKV typed-state refinement algebra.

The probe evaluates CAST/PATCH/GROUP/SCALE as diagnostic state semantics.  It
does not extend the frozen executable action catalog, fit target K/V, schedule
from labels, or launch training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_first_pass import MANIFEST, load_edge, markdown_table  # noqa: E402
from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from probe_kv_mechanism import checkpoint  # noqa: E402

from hstu_kvcache.evaluation import bernoulli_js, binary_metrics, stable_log_loss  # noqa: E402
from hstu_kvcache.models import HSTUKVCache, hybrid_tail_refresh, truncate_cache  # noqa: E402
from hstu_kvcache.training import collate_foundation_batch  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_refinement_algebra_v1"
PREFIX = 512
TAIL = 128
KEEP = PREFIX - TAIL
CARRIER_COUNTS = (8, 16, 32, 64, 128)


def joint_cast_maps(parent, current) -> list[torch.Tensor]:
    """Parameter-only joint K/V coordinate maps; no target state is fitted."""
    maps = []
    for parent_block, current_block in zip(parent.blocks, current.blocks, strict=True):
        parent_projection = torch.cat(
            [parent_block.attn.k_proj.weight.T.float(), parent_block.attn.v_proj.weight.T.float()],
            dim=1,
        )
        current_projection = torch.cat(
            [
                current_block.attn.k_proj.weight.T.float(),
                current_block.attn.v_proj.weight.T.float(),
            ],
            dim=1,
        )
        norm_scale = current_block.norm.weight.float() / parent_block.norm.weight.float().clamp_min(
            1e-8
        )
        maps.append(
            torch.linalg.pinv(parent_projection) @ torch.diag(norm_scale) @ current_projection
        )
    return maps


@torch.inference_mode()
def cast_state(cache: HSTUKVCache, maps: list[torch.Tensor]) -> HSTUKVCache:
    """CAST a typed state without changing its support or represented mass."""
    translated_k, translated_v = [], []
    width = cache.k.shape[-1]
    for layer, mapping in enumerate(maps):
        source = torch.cat([cache.k[layer].float(), cache.v[layer].float()], dim=-1)
        target = source @ mapping
        translated_k.append(target[..., :width].to(cache.k.dtype))
        translated_v.append(target[..., width:].to(cache.v.dtype))
    return HSTUKVCache(
        k=torch.stack(translated_k),
        v=torch.stack(translated_v),
        seq_len=cache.seq_len,
    )


def union_segments(prefix: HSTUKVCache, suffix: HSTUKVCache) -> HSTUKVCache:
    """Materialize an ordered UNION view for two disjoint contiguous segments."""
    if prefix.k.shape[:2] != suffix.k.shape[:2] or prefix.k.shape[-1] != suffix.k.shape[-1]:
        raise ValueError("UNION segment shapes differ")
    return HSTUKVCache(
        k=torch.cat([prefix.k, suffix.k], dim=2),
        v=torch.cat([prefix.v, suffix.v], dim=2),
        seq_len=prefix.seq_len + suffix.seq_len,
    )


def tail_segment(cache: HSTUKVCache, start: int = KEEP) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k[:, :, start:, :],
        v=cache.v[:, :, start:, :],
        seq_len=cache.seq_len - start,
    )


def gather_cache(cache: HSTUKVCache, indices: torch.Tensor) -> HSTUKVCache:
    if indices.ndim != 1:
        raise ValueError("shared GROUP indices must be one-dimensional")
    expanded = indices.view(1, 1, -1, 1).expand(
        cache.k.shape[0], cache.k.shape[1], indices.numel(), cache.k.shape[-1]
    )
    return HSTUKVCache(
        k=torch.gather(cache.k, 2, expanded),
        v=torch.gather(cache.v, 2, expanded),
        seq_len=indices.numel(),
    )


def group_indices(carriers: int, device: torch.device) -> torch.Tensor:
    if TAIL % carriers:
        raise ValueError("carrier count must divide tail width")
    endpoints = torch.arange(
        KEEP + TAIL // carriers - 1,
        PREFIX,
        TAIL // carriers,
        dtype=torch.long,
        device=device,
    )
    return torch.cat([torch.arange(KEEP, device=device), endpoints])


@torch.inference_mode()
def group_then_patch(
    current,
    base_cache: HSTUKVCache,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    time_deltas: torch.Tensor,
    carriers: int,
) -> HSTUKVCache:
    """GROUP raw evidence first, then materialize only its Current carriers."""
    if TAIL % carriers:
        raise ValueError("carrier count must divide tail width")
    step = TAIL // carriers
    positions = torch.arange(step - 1, TAIL, step, device=item_ids.device)
    embedded = current.embed_inputs(
        item_ids[:, -TAIL:].index_select(1, positions),
        behaviors[:, -TAIL:].index_select(1, positions),
        time_deltas[:, -TAIL:].index_select(1, positions),
    )
    _, state = current.forward_with_cache_embedded(truncate_cache(base_cache, KEEP), embedded)
    return state


def patch_then_group(patched_cache: HSTUKVCache, carriers: int) -> HSTUKVCache:
    """Materialize the dense Current tail first, then retain GROUP carriers."""
    return gather_cache(patched_cache, group_indices(carriers, patched_cache.k.device))


def scale_mass(cache: HSTUKVCache, factor: float, start: int = KEEP) -> HSTUKVCache:
    """Diagnostic encoding of SCALE; the runtime contract should store mass as metadata."""
    values = cache.v.clone()
    values[:, :, start:] *= factor
    return HSTUKVCache(k=cache.k.clone(), v=values, seq_len=cache.seq_len)


def payload_residual(
    target: HSTUKVCache, base: HSTUKVCache, start: int = KEEP
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a base-conditioned diagnostic PATCH payload."""
    return (
        target.k[:, :, start:, :].float() - base.k[:, :, start:, :].float(),
        target.v[:, :, start:, :].float() - base.v[:, :, start:, :].float(),
    )


def apply_payload_residual(
    base: HSTUKVCache,
    residual: tuple[torch.Tensor, torch.Tensor],
    start: int = KEEP,
) -> HSTUKVCache:
    """Apply a diagnostic additive PATCH to one payload scope."""
    k, v = base.k.clone(), base.v.clone()
    k[:, :, start:, :] += residual[0].to(k.dtype)
    v[:, :, start:, :] += residual[1].to(v.dtype)
    return HSTUKVCache(k=k, v=v, seq_len=base.seq_len)


def max_cache_error(left: HSTUKVCache, right: HSTUKVCache) -> float:
    return max(
        float((left.k.float() - right.k.float()).abs().max()),
        float((left.v.float() - right.v.float()).abs().max()),
    )


@torch.inference_mode()
def score(current, cache: HSTUKVCache, batch) -> np.ndarray:
    logits = current.score_cc_reuse(cache, batch.candidate_ids, batch.query_time_deltas)[:, 0]
    return logits.float().cpu().numpy()


def probability(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def cost_rows(layers: int = 4) -> list[dict]:
    rows = [
        {
            "plan": "CAST(all)",
            "raw_tokens": 0,
            "recomputed_token_layers": 0,
            "state_token_layers_read": layers * PREFIX,
            "state_token_layers_written": layers * PREFIX,
            "persistent_positions": PREFIX,
        },
        {
            "plan": "PATCH_exact(tail128)",
            "raw_tokens": TAIL,
            "recomputed_token_layers": layers * TAIL,
            "state_token_layers_read": layers * KEEP,
            "state_token_layers_written": layers * TAIL,
            "persistent_positions": PREFIX,
        },
        {
            "plan": "CAST(prefix)+PATCH_exact(tail128)",
            "raw_tokens": TAIL,
            "recomputed_token_layers": layers * TAIL,
            "state_token_layers_read": layers * PREFIX,
            "state_token_layers_written": layers * PREFIX,
            "persistent_positions": PREFIX,
        },
    ]
    for carriers in CARRIER_COUNTS:
        rows.extend(
            [
                {
                    "plan": f"GROUP({TAIL}->{carriers})->PATCH->SCALE",
                    "raw_tokens": carriers,
                    "recomputed_token_layers": layers * carriers,
                    "state_token_layers_read": layers * KEEP,
                    "state_token_layers_written": layers * carriers,
                    "persistent_positions": KEEP + carriers,
                },
                {
                    "plan": f"PATCH_dense->GROUP({TAIL}->{carriers})->SCALE",
                    "raw_tokens": TAIL,
                    "recomputed_token_layers": layers * TAIL,
                    "state_token_layers_read": layers * KEEP,
                    "state_token_layers_written": layers * carriers,
                    "persistent_positions": KEEP + carriers,
                },
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-requests", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--edges", type=int, nargs="*", default=list(range(5)))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if not args.edges or any(edge not in range(5) for edge in args.edges):
        raise ValueError("edges must be drawn from 0..4")
    device = torch.device(args.device)

    manifest = pq.read_table(MANIFEST / "requests_quality.parquet").to_pandas()
    by_request = manifest.set_index("request_id").to_dict("index")
    labels = dict(zip(manifest["request_id"], manifest["label"], strict=True))
    selections = {}
    for edge in args.edges:
        rows = load_edge(edge, labels)
        rows = rows[(rows["append_count_since_cutover"] == 0) & (rows["history_length"] == PREFIX)]
        rows = rows.sort_values(["uid", "query_timestamp", "request_id"])
        rows = rows.groupby("uid", sort=True).head(1).head(args.max_requests).copy()
        selections[edge] = rows
    uids = sorted({int(uid) for rows in selections.values() for uid in rows["uid"]})

    probe, payload = load_model(checkpoint(1), device)
    oov_buckets = int(payload["config"]["num_items"]) - 781678
    del probe
    torch.cuda.empty_cache()
    history = load_histories(uids, oov_buckets=oov_buckets)

    path_rows, cast_patch_rows = [], []
    baseline_errors: dict[str, dict[str, float]] = {}
    for edge, selected in selections.items():
        edge_name = f"v{edge}_to_v{edge + 1}"
        parent, _ = load_model(checkpoint(edge), device)
        current, _ = load_model(checkpoint(edge + 1), device)
        cast_maps = joint_cast_maps(parent, current)
        requests = [
            {**by_request[request_id], "request_id": request_id, "weight": 1.0}
            for request_id in selected["request_id"]
        ]
        path_logits: dict[str, list[float]] = {}
        dce_errors, reconstruction_errors = [], []
        for start in range(0, len(requests), args.batch_size):
            request_batch = requests[start : start + args.batch_size]
            batch = collate_foundation_batch(request_batch, history, device=device)
            if not bool((batch.lengths == PREFIX).all()):
                raise RuntimeError("refinement probe requires full 512-token prefixes")

            parent_cache = parent.compute_kv(batch.item_ids, batch.behaviors, batch.time_deltas)
            current_cache = current.compute_kv(batch.item_ids, batch.behaviors, batch.time_deltas)
            cast_cache = cast_state(parent_cache, cast_maps)
            patch_parent = hybrid_tail_refresh(
                current, parent_cache, batch.item_ids, batch.behaviors, batch.time_deltas, TAIL
            )
            cast_then_patch = hybrid_tail_refresh(
                current, cast_cache, batch.item_ids, batch.behaviors, batch.time_deltas, TAIL
            )

            # CAST on a scope that an exact PATCH fully overwrites is dead work.
            cast_prefix_parent_tail = union_segments(
                truncate_cache(cast_cache, KEEP), tail_segment(parent_cache)
            )
            cast_prefix_then_patch = hybrid_tail_refresh(
                current,
                cast_prefix_parent_tail,
                batch.item_ids,
                batch.behaviors,
                batch.time_deltas,
                TAIL,
            )
            dce_errors.append(max_cache_error(cast_then_patch, cast_prefix_then_patch))

            cast_residual = payload_residual(cast_then_patch, cast_cache)
            reconstructed = apply_payload_residual(cast_cache, cast_residual)
            reconstruction_errors.append(max_cache_error(reconstructed, cast_then_patch))

            # Test whether a PATCH generated against Parent is portable after CAST.
            parent_residual = payload_residual(patch_parent, parent_cache)
            residual_transfer = apply_payload_residual(cast_cache, parent_residual)

            observed = {
                "reuse_parent": score(current, parent_cache, batch),
                "current_exact": score(current, current_cache, batch),
                "cast_all": score(current, cast_cache, batch),
                "patch_tail128": score(current, patch_parent, batch),
                "cast_then_patch_tail128": score(current, cast_then_patch, batch),
                "parent_patch_residual_on_cast": score(current, residual_transfer, batch),
            }

            for carriers in CARRIER_COUNTS:
                group_patch = group_then_patch(
                    current,
                    parent_cache,
                    batch.item_ids,
                    batch.behaviors,
                    batch.time_deltas,
                    carriers,
                )
                patch_group = patch_then_group(patch_parent, carriers)
                mass = TAIL / carriers
                observed[f"group_patch_c{carriers}"] = score(current, group_patch, batch)
                observed[f"group_patch_scale_c{carriers}"] = score(
                    current, scale_mass(group_patch, mass), batch
                )
                observed[f"patch_group_c{carriers}"] = score(current, patch_group, batch)
                observed[f"patch_group_scale_c{carriers}"] = score(
                    current, scale_mass(patch_group, mass), batch
                )

            for name, values in observed.items():
                path_logits.setdefault(name, []).extend(values.tolist())

        logits = {
            name: np.asarray(values, dtype=np.float64) for name, values in path_logits.items()
        }
        baseline_errors[edge_name] = {
            "current": float(
                np.max(np.abs(logits["current_exact"] - selected["current_exact_logit"].to_numpy()))
            ),
            "reuse": float(
                np.max(np.abs(logits["reuse_parent"] - selected["reuse_logit"].to_numpy()))
            ),
        }
        if max(baseline_errors[edge_name].values()) > 2e-5:
            raise RuntimeError("refinement baseline replay differs from sealed rolling raw")

        target = np.asarray([request["label"] for request in requests], dtype=np.int64)
        exact = logits["current_exact"]
        exact_probability = probability(exact)
        reuse_gap = float(np.mean(np.abs(probability(logits["reuse_parent"]) - exact_probability)))
        exact_loss = stable_log_loss(exact, target)
        for path, values in logits.items():
            gap = float(np.mean(np.abs(probability(values) - exact_probability)))
            path_loss = stable_log_loss(values, target)
            metrics = binary_metrics(target, values)
            path_rows.append(
                {
                    "edge": edge_name,
                    "path": path,
                    "requests": len(values),
                    "mean_abs_probability_gap": gap,
                    "output_gap_recovery": 1.0 - gap / reuse_gap,
                    "path_minus_exact_log_loss": float((path_loss - exact_loss).mean()),
                    "ROC_AUC": metrics["ROC_AUC"],
                    "mean_bernoulli_js": float(bernoulli_js(values, exact).mean()),
                }
            )

        edge_recovery = {
            row["path"]: row["output_gap_recovery"] for row in path_rows if row["edge"] == edge_name
        }
        cast_patch_rows.append(
            {
                "edge": edge_name,
                "cast_recovery": edge_recovery["cast_all"],
                "patch_recovery": edge_recovery["patch_tail128"],
                "cast_then_patch_recovery": edge_recovery["cast_then_patch_tail128"],
                "increment_over_best": edge_recovery["cast_then_patch_tail128"]
                - max(edge_recovery["cast_all"], edge_recovery["patch_tail128"]),
                "parent_residual_on_cast_recovery": edge_recovery["parent_patch_residual_on_cast"],
                "cast_shadowed_scope_max_kv_error": max(dce_errors),
                "base_conditioned_patch_reconstruction_error": max(reconstruction_errors),
            }
        )
        del parent, current
        torch.cuda.empty_cache()

    path_frame = pd.DataFrame(path_rows)
    cast_patch_frame = pd.DataFrame(cast_patch_rows)
    recovery_lookup = path_frame.set_index(["edge", "path"])["output_gap_recovery"]
    order_rows, scale_rows, density_rows = [], [], []
    for edge_name in cast_patch_frame["edge"]:
        for carriers in CARRIER_COUNTS:
            gp = recovery_lookup[(edge_name, f"group_patch_c{carriers}")]
            gps = recovery_lookup[(edge_name, f"group_patch_scale_c{carriers}")]
            pg = recovery_lookup[(edge_name, f"patch_group_c{carriers}")]
            pgs = recovery_lookup[(edge_name, f"patch_group_scale_c{carriers}")]
            order_rows.append(
                {
                    "edge": edge_name,
                    "carriers": carriers,
                    "group_then_patch_scaled_recovery": gps,
                    "patch_then_group_scaled_recovery": pgs,
                    "patch_then_group_minus_group_then_patch": pgs - gps,
                }
            )
            scale_rows.extend(
                [
                    {
                        "edge": edge_name,
                        "order": "group_then_patch",
                        "carriers": carriers,
                        "unscaled_recovery": gp,
                        "scaled_recovery": gps,
                        "scale_increment": gps - gp,
                    },
                    {
                        "edge": edge_name,
                        "order": "patch_then_group",
                        "carriers": carriers,
                        "unscaled_recovery": pg,
                        "scaled_recovery": pgs,
                        "scale_increment": pgs - pg,
                    },
                ]
            )
        for order in ("group_patch_scale", "patch_group_scale"):
            previous = None
            for carriers in CARRIER_COUNTS:
                recovery = recovery_lookup[(edge_name, f"{order}_c{carriers}")]
                density_rows.append(
                    {
                        "edge": edge_name,
                        "order": order,
                        "carriers": carriers,
                        "carrier_density": carriers / TAIL,
                        "represented_mass": TAIL / carriers,
                        "recovery": recovery,
                        "increment_from_previous_density": (
                            recovery - previous if previous is not None else float("nan")
                        ),
                    }
                )
                previous = recovery

    order_frame = pd.DataFrame(order_rows)
    scale_frame = pd.DataFrame(scale_rows)
    density_frame = pd.DataFrame(density_rows)
    cost_frame = pd.DataFrame(cost_rows())
    args.output.mkdir(parents=True)
    path_frame.to_csv(args.output / "path_recovery.csv", index=False)
    cast_patch_frame.to_csv(args.output / "cast_patch_decomposition.csv", index=False)
    order_frame.to_csv(args.output / "group_patch_order.csv", index=False)
    scale_frame.to_csv(args.output / "scale_ablation.csv", index=False)
    density_frame.to_csv(args.output / "carrier_density_frontier.csv", index=False)
    cost_frame.to_csv(args.output / "cost_shape.csv", index=False)

    scale64 = scale_frame[scale_frame["carriers"] == 64]
    order64 = order_frame[order_frame["carriers"] == 64]
    density_increments = density_frame.dropna(subset=["increment_from_previous_density"])
    summary = {
        "status": "typed_state_refinement_algebra_probe_complete",
        "scope": f"up to {args.max_requests} append-free full-prefix requests per selected edge; Small seed17",
        "requests_per_edge": {
            f"v{edge}_to_v{edge + 1}": len(rows) for edge, rows in selections.items()
        },
        "baseline_replay_max_abs_error": baseline_errors,
        "protocol": (
            "diagnostic only; parameter-only CAST; raw-history PATCH; no target-KV fit, "
            "score mixing, label-driven selection, or scale action change"
        ),
        "cast_patch_increment_range": [
            float(cast_patch_frame["increment_over_best"].min()),
            float(cast_patch_frame["increment_over_best"].max()),
        ],
        "cast_shadowed_scope_max_kv_error": float(
            cast_patch_frame["cast_shadowed_scope_max_kv_error"].max()
        ),
        "base_conditioned_patch_reconstruction_max_error": float(
            cast_patch_frame["base_conditioned_patch_reconstruction_error"].max()
        ),
        "scale64_increment_range": [
            float(scale64["scale_increment"].min()),
            float(scale64["scale_increment"].max()),
        ],
        "group_patch_order64_abs_difference_mean": float(
            order64["patch_then_group_minus_group_then_patch"].abs().mean()
        ),
        "positive_density_increments": int(
            (density_increments["increment_from_previous_density"] >= 0).sum()
        ),
        "density_increments": int(len(density_increments)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Typed-state refinement algebra probe",
        "",
        "All paths are diagnostic interventions. Output-gap recovery is not rolling recommendation-quality recovery.",
        "",
        "## CAST/PATCH decomposition",
        "",
        *markdown_table(cast_patch_frame, list(cast_patch_frame.columns)),
        "",
        "## GROUP/PATCH order",
        "",
        *markdown_table(order_frame, list(order_frame.columns)),
        "",
        "## SCALE ablation",
        "",
        *markdown_table(scale_frame, list(scale_frame.columns)),
        "",
        "## Carrier-density frontier",
        "",
        *markdown_table(density_frame, list(density_frame.columns)),
        "",
        "## Structural cost shape",
        "",
        *markdown_table(cost_frame, list(cost_frame.columns)),
        "",
        "The report intentionally does not freeze a residual threshold, carrier-density contract, or budget compiler. Those require held-out rolling quality and a target-free residual estimator.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

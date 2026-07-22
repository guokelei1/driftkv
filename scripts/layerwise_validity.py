"""Validity-v1 evaluation for cheap K/V projection plus top-N full suffix layers."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from motivation_validity import (
    STANDARD_LOGS,
    eval_batches,
    move_batch,
    ranking_metrics,
    seed_everything,
)

from hstu_kvcache.data import StreamingDataPlan
from hstu_kvcache.migration import (
    capture_layerwise_state,
    extra_state_numel,
    migrate_legacy_suffix_cache,
    sample_relative_cache_error,
)
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.streaming import model_params_vec
from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-result")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model-ts", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--stale-modes",
        nargs="+",
        choices=["one_step", "cumulative_theta0"],
        default=["one_step", "cumulative_theta0"],
    )
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.run_result is None:
        args.run_result = f"results/validity/core_seed{args.seed}.json"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/validity/core_seed{args.seed}"
    if args.output is None:
        args.output = f"results/validity/layerwise_seed{args.seed}.json"


def make_model(metadata: dict, num_items: int, num_behaviors: int, device: str) -> HSTU:
    cfg = HSTUConfig(
        num_items=num_items,
        num_behaviors=num_behaviors,
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        num_heads=metadata["num_heads"],
        head_dim=metadata["head_dim"],
        max_seq_len=metadata["seq_len"],
        activation="relu",
    )
    return HSTU(cfg).to(device)


def load_model(
    metadata: dict,
    num_items: int,
    num_behaviors: int,
    device: str,
    checkpoint_dir: str,
    version: int,
) -> HSTU:
    model = make_model(metadata, num_items, num_behaviors, device)
    state = torch.load(
        Path(checkpoint_dir) / f"theta_{version}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def reconstruct_eval_samples(
    plan: StreamingDataPlan,
    model_ts: list[int],
    window_days: int,
    max_users: int,
) -> dict[int, list[dict]]:
    samples = {}
    wanted = set(model_ts)
    for version in range(1, max(model_ts) + 1):
        start = (version - 1) * window_days
        end = version * window_days
        if end >= len(plan.stream_dates):
            break
        for date in plan.stream_dates[start:end]:
            plan.ingest_day(date)
        if version in wanted:
            samples[version] = plan.get_eval_set(plan.stream_dates[end], max_users)
    return samples


def timed_call(fn, device: torch.device, repeats: int):
    if device.type == "cuda":
        with torch.cuda.device(device):
            fn()
            torch.cuda.synchronize(device)
            times = []
            value = None
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                value = fn()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
        return value, float(np.median(times))
    fn()
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return value, float(np.median(times))


def layer_relative_errors(old: HSTUKVCache, fresh: HSTUKVCache) -> torch.Tensor:
    delta = (old.k.float() - fresh.k.float()).square().sum(dim=(2, 3))
    delta += (old.v.float() - fresh.v.float()).square().sum(dim=(2, 3))
    scale = fresh.k.float().square().sum(dim=(2, 3))
    scale += fresh.v.float().square().sum(dim=(2, 3))
    return delta.sqrt() / scale.sqrt().clamp_min(1e-12)


def config_name(top_n: int, num_layers: int) -> str:
    if top_n == 0:
        return "cheap_all"
    if top_n == num_layers:
        return "recompute"
    return f"cheap_plus_top{top_n}_full"


def summarize(
    per_user: list[dict],
    timing: dict[str, float],
    cache_error: dict[str, list[float]],
    extra_numel: dict[str, int],
    cache_numel: int,
) -> dict:
    metric_names = (
        "mrr",
        "ndcg@10",
        "ndcg@100",
        "hit@10",
        "hit@100",
        "best_rank",
        "mean_rank",
        "rank_utility",
    )
    configs = list(per_user[0]["configs"])
    reuse = "reuse"
    output = {}
    for name in configs:
        metrics = {
            metric: float(np.mean([row["configs"][name][metric] for row in per_user]))
            for metric in metric_names
        }
        gains = {}
        for metric in metric_names:
            current = np.array([row["configs"][name][metric] for row in per_user])
            baseline = np.array([row["configs"][reuse][metric] for row in per_user])
            if metric in ("best_rank", "mean_rank"):
                gain = baseline - current
            else:
                gain = current - baseline
            gains[metric] = float(gain.mean())
        output[name] = {
            "metrics": metrics,
            "gain_over_reuse": gains,
            "cache_error_rel": float(np.mean(cache_error[name])),
            "migration_ms_per_user": timing[name] / len(per_user),
            "migration_ratio_to_recompute": 0.0,
            "extra_state_numel_per_user": extra_numel[name] // len(per_user),
            "extra_state_ratio_to_kv": extra_numel[name] / max(cache_numel, 1),
            "extra_state_fp16_bytes_per_user": 2 * extra_numel[name] // len(per_user),
        }
    full_time = timing["recompute"]
    for value in output.values():
        value["migration_ratio_to_recompute"] = (
            value["migration_ms_per_user"]
            / max(output["recompute"]["migration_ms_per_user"], 1e-12)
        )
    fresh = {
        metric: float(np.mean([row["fresh"][metric] for row in per_user]))
        for metric in metric_names
    }
    parity = max(row["fresh_incremental_parity_max_abs"] for row in per_user)
    return {
        "fresh_full": fresh,
        "configs": output,
        "fresh_incremental_parity_max_abs": parity,
        "total_recompute_migration_ms": full_time,
    }


@torch.inference_mode()
def evaluate_pair(
    current: HSTU,
    old: HSTU,
    samples: list[dict],
    metadata: dict,
    device: torch.device,
    timing_repeats: int,
) -> dict:
    num_layers = len(current.blocks)
    all_items = torch.arange(1, current.cfg.num_items + 1, device=device)
    per_user = []
    timing = {"reuse": 0.0}
    cache_error = {"reuse": []}
    extra_numel = {"reuse": 0}
    cache_numel = 0
    layer_errors = []
    for selected, full_cpu, prefix_cpu, suffix_cpu in eval_batches(
        samples,
        metadata["seq_len"],
        metadata["batch_size"],
    ):
        full = move_batch(full_cpu, device)
        prefix = move_batch(prefix_cpu, device)
        suffix = move_batch(suffix_cpu, device)
        full_hidden, _ = current(
            full["item_ids"],
            full["behaviors"],
            full["time_deltas"],
            lengths=full["lengths"],
        )
        fresh_hidden = current.last_hidden(full_hidden, full["lengths"])
        old_state = capture_layerwise_state(
            old,
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            prefix["lengths"],
        )
        fresh_cache = current.compute_kv(
            prefix["item_ids"],
            prefix["behaviors"],
            prefix["time_deltas"],
            lengths=prefix["lengths"],
        )
        fresh_incremental, _ = current.forward_with_cache(
            fresh_cache,
            suffix["item_ids"],
            suffix["behaviors"],
            suffix["time_deltas"],
        )
        parity = (fresh_incremental[:, 0] - fresh_hidden).abs().amax(dim=1)
        fresh_scores = current.item_emb.score(
            fresh_hidden,
            all_items.unsqueeze(0).expand(len(selected), -1),
        )
        layer_errors.append(layer_relative_errors(old_state.kv, fresh_cache).cpu().numpy())
        cache_numel += fresh_cache.k.numel() + fresh_cache.v.numel()

        caches = {"reuse": old_state.kv}
        for top_n in range(num_layers + 1):
            name = config_name(top_n, num_layers)
            fn = partial(
                migrate_legacy_suffix_cache,
                current,
                old_state,
                prefix["item_ids"],
                prefix["behaviors"],
                prefix["time_deltas"],
                top_n_full=top_n,
            )
            cache, elapsed = timed_call(fn, device, timing_repeats)
            caches[name] = cache
            timing[name] = timing.get(name, 0.0) + elapsed
            extra_numel[name] = extra_numel.get(name, 0) + extra_state_numel(
                old_state,
                top_n,
            )

        batch_metrics = {}
        for name, cache in caches.items():
            migrated_hidden, _ = current.forward_with_cache(
                cache,
                suffix["item_ids"],
                suffix["behaviors"],
                suffix["time_deltas"],
            )
            scores = current.item_emb.score(
                migrated_hidden[:, 0],
                all_items.unsqueeze(0).expand(len(selected), -1),
            )
            batch_metrics[name] = [
                ranking_metrics(scores[row], selected[row]["pos_items"])
                for row in range(len(selected))
            ]
            errors = sample_relative_cache_error(cache, fresh_cache).cpu().tolist()
            cache_error.setdefault(name, []).extend(errors)

        for row, sample in enumerate(selected):
            per_user.append(
                {
                    "user_id": int(sample["history"]["user_id"]),
                    "history_length": int(full["lengths"][row].item()),
                    "fresh": ranking_metrics(fresh_scores[row], sample["pos_items"]),
                    "fresh_incremental_parity_max_abs": float(parity[row].item()),
                    "configs": {
                        name: batch_metrics[name][row]
                        for name in batch_metrics
                    },
                }
            )

    layer_error_array = np.concatenate(layer_errors, axis=1)
    summary = summarize(per_user, timing, cache_error, extra_numel, cache_numel)
    summary["per_layer_stale_cache_error_rel"] = layer_error_array.mean(axis=1).tolist()
    summary["per_layer_error_order"] = np.argsort(
        -layer_error_array.mean(axis=1)
    ).tolist()
    return {"summary": summary, "per_user": per_user}


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(args.seed)
    metadata_result = json.loads(Path(args.run_result).read_text())
    metadata = metadata_result["args"]
    plan = StreamingDataPlan.from_csvs(
        STANDARD_LOGS,
        base_num_days=metadata["base_days"],
        max_seq_len=metadata["seq_len"],
        max_items=metadata["max_items"],
        fit_vocabulary_on_base=True,
    )
    plan.init_base()
    samples = reconstruct_eval_samples(
        plan,
        args.model_ts,
        metadata["stream_window_days"],
        metadata["max_eval_users"],
    )
    result = {
        "protocol": "layerwise_validity_v1",
        "operator": {
            "cheap": "cached Norm(x_old) projected by current Wk/Wv",
            "top_n_full": "current full blocks over the deepest contiguous N-layer suffix, starting from the cached split hidden state",
            "serving": "migrated prefix cache plus latest behavior token under the current model",
        },
        "seed": args.seed,
        "source_run": args.run_result,
        "checkpoint_dir": args.checkpoint_dir,
        "model_ts": args.model_ts,
        "stale_modes": args.stale_modes,
        "pairs": [],
    }
    for model_t in args.model_ts:
        current = load_model(
            metadata,
            plan.num_items,
            plan.num_behaviors,
            args.device,
            args.checkpoint_dir,
            model_t,
        )
        for mode in args.stale_modes:
            stale_t = model_t - 1 if mode == "one_step" else 0
            old = load_model(
                metadata,
                plan.num_items,
                plan.num_behaviors,
                args.device,
                args.checkpoint_dir,
                stale_t,
            )
            current_vec = model_params_vec(current).detach()
            old_vec = model_params_vec(old).detach()
            dtheta = float((current_vec - old_vec).norm() / old_vec.norm())
            pair = evaluate_pair(
                current,
                old,
                samples[model_t],
                metadata,
                torch.device(args.device),
                args.timing_repeats,
            )
            pair.update(
                {
                    "mode": mode,
                    "model_t": model_t,
                    "stale_t": stale_t,
                    "dtheta_rel": dtheta,
                    "eval_date": plan.stream_dates[model_t * metadata["stream_window_days"]],
                    "n_users": len(pair["per_user"]),
                }
            )
            result["pairs"].append(pair)
            save_json(result, args.output)
            configs = pair["summary"]["configs"]
            print(
                f"seed={args.seed} mode={mode} theta={stale_t}->{model_t} "
                f"n={pair['n_users']} dtheta={dtheta:.5f}",
                flush=True,
            )
            for name, value in configs.items():
                print(
                    f"  {name:>22} time={value['migration_ratio_to_recompute']:.3f} "
                    f"kv_err={value['cache_error_rel']:.4f} "
                    f"rank_gain={value['gain_over_reuse']['best_rank']:.2f} "
                    f"ndcg100_gain={value['gain_over_reuse']['ndcg@100']:.5f}",
                    flush=True,
                )
            del old
        del current
    save_json(result, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

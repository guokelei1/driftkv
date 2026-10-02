#!/usr/bin/env python3
"""Complete-epoch, single-seed chronological development on frozen time windows."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from development_probe import (
    ROOT,
    HSTUConfig,
    PreparedRecFlow,
    ProbeData,
    RecFlowGenerator,
    evaluate,
    save_json,
)

WINDOWS = {
    "A": ("train_1_18", "19_21", None),
    "B": ("update_19_21", "22_24", "A"),
    "C": ("update_22_24", "25_27", "B"),
    "D": ("update_21_21", "22_22", "C"),
    "E": ("update_22_22", "23_23", "D"),
    "F": ("update_23_23", "24_24", "E"),
}


def phase_windows(settings, phase):
    window = settings["windows"][phase]
    fit = window["fit"]
    future = window.get("evaluate", window.get("compare_parent_and_current"))
    if not (fit[0] <= fit[1] < future[0] <= future[1]):
        raise ValueError("Training must finish before the complete future evaluation window.")
    prefix = "train" if phase == "A" else "update"
    return f"{prefix}_{fit[0]}_{fit[1]}", f"{future[0]}_{future[1]}", fit, future


def check_parent_days(previous_fit, fit, same_phase):
    if (same_phase and previous_fit != fit) or (not same_phase and previous_fit[1] >= fit[0]):
        raise ValueError("Checkpoint training days overlap the new update or change a resumed window.")


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def epoch_batches(known, batch_size, rng, day_blocks=None):
    """Visit every eligible request once, retaining the final partial batch."""
    if day_blocks is None:
        order = rng.permutation(known)
    else:
        # Shuffle within each three-day block, then concatenate chronologically
        # before batching. One block consumes exactly the original RNG draw.
        order = np.concatenate([rng.permutation(known[day_blocks == block])
                                for block in np.unique(day_blocks)])
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def train_epoch(model, dataset, known, optimizer, args, phase, epoch, out, day_blocks=None):
    # Epoch-addressed RNG makes continuation reproducible without relying on
    # how much evaluation ran between complete epochs.
    rng_seed = args.seed + 100_000 * (ord(phase) - ord("A")) + epoch
    rng = np.random.default_rng(rng_seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    begin = time.monotonic()
    processed, weighted_loss, steps = 0, 0.0, 0
    order_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    for indices in epoch_batches(known, args.batch_size, rng, day_blocks):
        batch = dataset.batch(indices, args.device, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            losses = model.loss_per_example(**batch)
            loss = losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {phase} epoch{epoch}; epoch is incomplete.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradient in {phase} epoch{epoch}; epoch is incomplete.")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        processed += len(indices)
        steps += 1
        weighted_loss += float(loss.detach()) * len(indices)
        order_hash.update(indices.tobytes())
        target_hash.update(batch["targets"].detach().cpu().numpy().tobytes())
        if steps == 1 or steps % args.log_every == 0:
            progress = dict(phase=phase, epoch=epoch, step=steps,
                requests=processed, expected_requests=len(known),
                request_mean_loss=weighted_loss / processed,
                seconds=time.monotonic() - begin)
            save_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert processed == len(known), "Incomplete epoch must never qualify a model."
    elapsed = time.monotonic() - begin
    return dict(phase=phase, epoch=epoch, complete=True, rng_seed=rng_seed,
        training_order=args.training_order,
        eligible_requests=len(known), processed_requests=processed, optimizer_steps=steps,
        request_mean_loss=weighted_loss / processed, seconds=elapsed,
        requests_per_second=processed / elapsed,
        order_sha256=order_hash.hexdigest(), sampled_targets_sha256=target_hash.hexdigest(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None)


def train_distributed_epoch(model, dataset, known, optimizer, args, phase, epoch, out, day_blocks=None):
    """Adapt the verified global-batch DDP helper to the existing epoch record."""
    from distributed_probe import train_epoch_ddp

    rank = dist.get_rank()
    rng_seed = args.seed + 100_000 * (ord(phase) - ord("A")) + epoch
    rng = np.random.default_rng(rng_seed)
    order_hash, target_hash = hashlib.sha256(), hashlib.sha256()

    def global_batches():
        for indices in epoch_batches(known, args.batch_size, rng, day_blocks):
            batch = dataset.batch(indices, "cpu", rng)
            order_hash.update(indices.tobytes())
            target_hash.update(batch["targets"].numpy().tobytes())
            yield batch

    def progress_callback(progress):
        progress.update(phase=phase, epoch=epoch, expected_requests=len(known))
        save_json(out / "progress.json", progress)
        print(json.dumps(progress), flush=True)

    steps = (len(known) + args.batch_size - 1) // args.batch_size
    result = train_epoch_ddp(model, optimizer, global_batches() if rank == 0 else None,
        steps, torch.device(args.device), clip_norm=args.clip_norm,
        progress_callback=progress_callback if rank == 0 else None, log_every=args.log_every)
    assert result["global_requests"] == len(known), "Incomplete epoch must never qualify a model."
    record = dict(phase=phase, epoch=epoch, complete=True, rng_seed=rng_seed,
        training_order=args.training_order,
        eligible_requests=len(known), processed_requests=result["global_requests"],
        optimizer_steps=result["global_steps"], request_mean_loss=result["mean_training_loss"],
        seconds=result["seconds"], requests_per_second=result["requests_per_second"],
        order_sha256=order_hash.hexdigest(), sampled_targets_sha256=target_hash.hexdigest(),
        peak_allocated_bytes=max(row["peak_allocated_bytes"] for row in result["rank_memory"]),
        peak_reserved_bytes=max(row["peak_reserved_bytes"] for row in result["rank_memory"]),
        rank_memory=result["rank_memory"], world_size=dist.get_world_size(),
        global_batch_size=args.batch_size)
    if rank == 0:
        assert target_hash.hexdigest() == result["target_sha256"]
        save_json(out / "progress.json", record)
        print(json.dumps(record), flush=True)
    return record


@torch.no_grad()
def fixed_training_nll(model, dataset, known, args, phase):
    """Training-only fixed requests/targets; future labels never choose epochs."""
    seed = args.seed + 100_000 * (ord(phase) - ord("A")) + 70_000
    rng = np.random.default_rng(seed)
    indices = rng.permutation(known)[:args.training_probe_requests]
    model.eval()
    total = 0.0
    targets = hashlib.sha256()
    for start in range(0, len(indices), args.batch_size):
        selected = indices[start:start + args.batch_size]
        batch = dataset.batch(selected, args.device, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            losses = model.loss_per_example(**batch)
        total += float(losses.float().sum())
        targets.update(batch["targets"].cpu().numpy().tobytes())
    return dict(requests=len(indices), mean_nll=total / len(indices), rng_seed=seed,
        indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        sampled_targets_sha256=targets.hexdigest(), precision="BF16 autocast, FP32 loss")


def score_window(model, dataset, panels, args, phase, window, out, config, checkpoint=None):
    if args.eval_devices and not config["canary"]:
        from parallel_evaluate import parallel_evaluate

        # Separate evaluation workers cannot reuse this process's cached blocks.
        # Keep live model/optimizer state but release unused training/NLL storage.
        torch.cuda.empty_cache()
        # The canary's in-memory 8/4 panels deliberately use the serial path;
        # passing the original frozen file would expand them to 3072/768.
        return parallel_evaluate(checkpoint, args.panels, window, out, phase, args.eval_devices)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "configuration.json", config)
    evaluation = evaluate(model, dataset, panels[f"eval_{window}"], args, phase, out,
                          sampled_indices=panels[f"sampled_{window}"])
    save_json(out / "summary.json", dict(configuration=config, **{phase: evaluation}))
    return evaluation


def reuse_parent_evaluation(directory, checkpoint_hash, panels, args, window):
    """Avoid repeating the exact frozen A/day20 evaluation across LR probes."""
    from evaluation_pair import load_evaluation

    directory = Path(directory).resolve()
    phase, previous, proof = load_evaluation(directory)
    if (previous["evaluation_checkpoint_sha256"] != checkpoint_hash
            or previous["evaluation_window"] != window
            or previous["evaluation_canary_limit"] is not None):
        raise ValueError("Reused parent evaluation has another checkpoint/window or is a canary.")
    for key in ("candidate_seed", "decoder", "beam_width", "eval_precision", "sampled_distractors"):
        if previous[key] != getattr(args, key):
            raise ValueError(f"Reused parent evaluation differs in {key}.")
    full = panels[f"eval_{window}"]
    sample = full[np.isin(full, panels[f"sampled_{window}"])]
    for name, expected in (("full_catalog", full), ("sampled", sample)):
        with np.load(proof["panel_evidence"][name]["path"]) as saved:
            if not np.array_equal(saved["indices"], expected):
                raise ValueError("Reused parent evaluation has another request panel/order.")
    return proof["evaluation"], dict(directory=str(directory), phase=phase,
        summary_sha256=proof["input_summary_sha256"], panels=proof["panel_evidence"],
        checkpoint_sha256=checkpoint_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=WINDOWS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, help="Total complete epochs for this phase, including resumed epochs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-devices", nargs="+", help="Optional parallel evaluation CUDA devices, interpreted within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--parent-evaluation", type=Path,
                        help="Reuse a completed evaluation of the exact parent on the same future panel")
    parser.add_argument("--canary", action="store_true", help="One complete tiny subset epoch; never a qualified A/B/C")
    cli = parser.parse_args()
    settings = json.loads(cli.config.read_text())
    args = argparse.Namespace(**settings["arguments"])
    args.training_order = settings["arguments"].get("training_order", "shuffle")
    if args.training_order not in ("shuffle", "three_day_blocks"):
        parser.error("training_order must be shuffle or three_day_blocks")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    args.device = f"cuda:{local_rank}" if distributed else cli.device
    args.eval_devices = cli.eval_devices
    control_group = None
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(args.device), timeout=timedelta(hours=2))
        # CPU waits let independent evaluation workers use the same GPUs.
        # A pending NCCL barrier can keep a GPU kernel busy for the whole
        # rank0 evaluation. Training collectives still use the NCCL group.
        control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    epochs = cli.epochs or args.epochs
    if args.seed != 17 or epochs < 1:
        parser.error("This development protocol uses seed17 and positive complete epoch counts.")
    if cli.phase != "A" and cli.checkpoint is None:
        parser.error("Updates must start from their trained predecessor checkpoint.")
    if cli.parent_evaluation and (cli.phase == "A" or cli.canary):
        parser.error("Parent evaluation reuse is only for full update development runs.")
    if rank == 0:
        if cli.output.exists():
            parser.error("Use a fresh output directory to preserve prior development outcomes.")
        cli.output.mkdir(parents=True)
    if distributed:
        dist.barrier(group=control_group)
    torch.set_num_threads(4 if distributed else 8)
    torch.manual_seed(args.seed)
    prepared = PreparedRecFlow(args.data)
    dataset = ProbeData(prepared, args.catalog_size, args.cohort_users, args.context)
    manifest_path = Path(args.panels).with_name("summary.json")
    manifest = json.loads(manifest_path.read_text())
    if digest(args.panels) != manifest["arrays_file_sha256"]:
        raise ValueError("Frozen window panel file changed.")
    catalog_hash = hashlib.sha256(dataset.raw_ids.tobytes()).hexdigest()
    cohort_hash = hashlib.sha256(np.asarray(dataset.uids).tobytes()).hexdigest()
    if (catalog_hash != manifest["catalog_sha256"] or cohort_hash != manifest["cohort_sha256"]):
        raise ValueError("Window manifest and configured cohort/catalog differ.")
    panels = dict(np.load(args.panels))
    train_key, window, fit_days, evaluation_days = phase_windows(settings, cli.phase)
    predecessor = WINDOWS[cli.phase][2]
    train_indices = panels[train_key]
    for indices, days in ((train_indices, fit_days), (panels[f"eval_{window}"], evaluation_days)):
        actual_days = prepared.requests["day"][indices]
        if not ((actual_days >= days[0]) & (actual_days <= days[1])).all():
            raise ValueError("Frozen request indices fall outside the configured time window.")
    known = np.array([i for i in train_indices if len(dataset.targets(i)[1])], dtype=np.int64)
    if hashlib.sha256(known.tobytes()).hexdigest() != manifest["training"][train_key]["known_training_indices_sha256"]:
        raise ValueError("The complete eligible training window differs from its audit.")
    if cli.canary:
        known = known[:2 * args.batch_size + 7]
        epochs = 1
        for key in list(panels):
            if key.startswith("eval_"):
                panels[key] = panels[key][:8]
                panels[key.replace("eval_", "sampled_")] = panels[key][::2]
    day_blocks = ((prepared.requests["day"][known].astype(np.int64) - 1) // 3
                  if args.training_order == "three_day_blocks" else None)
    cfg = HSTUConfig(num_items=dataset.oov, num_behaviors=4,
        num_prediction_items=len(dataset.raw_ids), hidden_size=args.hidden,
        num_layers=args.layers, num_heads=args.heads, max_seq_len=args.context + 3,
        block_variant="hstu_reference", activation="silu", gating="silu_gate",
        relative_position_bias=True, input_dropout=0.0, attn_dropout=0.0)
    model = RecFlowGenerator(cfg, torch.tensor(dataset.paths), history_categories=False).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    loaded = None
    checkpoint_hash = None
    if cli.checkpoint:
        loaded = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
        previous = loaded["configuration"]
        if (previous["model"] != dataclasses.asdict(cfg) or previous["catalog_sha256"] != catalog_hash
                or previous["cohort_sha256"] != cohort_hash or (previous.get("canary", False) and not cli.canary)):
            raise ValueError("Checkpoint architecture/cohort/catalog/canary scope mismatch.")
        if loaded["phase"] not in (cli.phase, predecessor):
            raise ValueError("Checkpoint would break the chronological predecessor lineage.")
        previous_fit = previous.get("training_window_days")
        if previous_fit is None:
            # Existing six-layer A checkpoints predate explicit day metadata.
            previous_settings = Path(previous["settings_path"])
            if digest(previous_settings) != previous["settings_sha256"]:
                raise ValueError("Historical checkpoint settings changed.")
            previous_fit = json.loads(previous_settings.read_text())["windows"][loaded["phase"]]["fit"]
        check_parent_days(previous_fit, fit_days, loaded["phase"] == cli.phase)
        start_epoch = loaded["epoch"] if loaded["phase"] == cli.phase else 0
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        checkpoint_hash = digest(cli.checkpoint)
        del loaded
    if start_epoch >= epochs:
        raise ValueError("No new complete epoch requested.")
    optimizer_steps_before = sorted({int(value["step"].item()) for value in optimizer.state.values() if "step" in value})
    distributed_model = None
    if distributed:
        from distributed_probe import wrap

        distributed_model = wrap(model, torch.device(args.device))
        torch.cuda.empty_cache()
    sources = [Path(__file__).resolve(), ROOT / "scripts/recflow/development_probe.py",
        ROOT / "scripts/recflow/random_baseline.py", ROOT / "src/hstu_kvcache/recflow/model.py",
        ROOT / "src/hstu_kvcache/recflow/data.py", ROOT / "src/hstu_kvcache/recflow/metrics.py"]
    if distributed:
        sources.append(ROOT / "scripts/recflow/distributed_probe.py")
    if args.eval_devices:
        sources.append(ROOT / "scripts/recflow/parallel_evaluate.py")
    if cli.parent_evaluation:
        sources.extend([ROOT / "scripts/recflow/evaluation_pair.py", ROOT / "scripts/recflow/window_comparison.py"])
    config = dict(vars(args), model=dataclasses.asdict(cfg),
        parameters=sum(p.numel() for p in model.parameters()),
        phase=cli.phase, requested_complete_epochs=epochs, start_epoch=start_epoch,
        training_window_days=fit_days, evaluation_window_days=evaluation_days,
        training_order_blocks=([{ "days": [int(block) * 3 + 1, int(block) * 3 + 3],
            "requests": int((day_blocks == block).sum())} for block in np.unique(day_blocks)]
            if day_blocks is not None else None),
        world_size=world, global_batch_size=args.batch_size,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        optimizer_state_steps_before=optimizer_steps_before,
        optimizer_learning_rates=[group["lr"] for group in optimizer.param_groups],
        parallel_evaluation_memory_note="Evaluation worker peaks exclude the waiting training ranks' resident model, optimizer and DDP state; total physical GPU usage includes both.",
        canary=cli.canary, history_categories=False,
        catalog_items=len(dataset.raw_ids), catalog_sha256=catalog_hash, cohort_sha256=cohort_hash,
        settings_path=str(cli.config), settings_sha256=digest(cli.config),
        window_manifest_sha256=digest(manifest_path), panels_sha256=digest(args.panels),
        source_checkpoint=str(cli.checkpoint) if cli.checkpoint else None,
        source_checkpoint_sha256=checkpoint_hash,
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
        torch_version=str(torch.__version__),
        scope="Tiny canary, not A/B/C qualification" if cli.canary else "Complete-epoch development, fixed seed17; no formal admission/final-user outcomes")
    if rank == 0:
        save_json(cli.output / "configuration.json", config)
        print(json.dumps(config), flush=True)
        pre_nll = fixed_training_nll(model, dataset, known, args, cli.phase)
        record = dict(configuration=config, complete=False,
            training_window=train_key, all_positive_requests=len(train_indices),
            eligible_requests=len(known), audited_eligible_requests=manifest["training"][train_key]["known_positive_requests"],
            pre_training_nll=pre_nll, epochs=[])
        save_json(cli.output / "summary.json", record)
        if predecessor and start_epoch == 0:
            name = f"parent_{predecessor}_d{window}"
            if cli.parent_evaluation:
                record["parent_evaluation"], record["parent_evaluation_reuse"] = reuse_parent_evaluation(
                    cli.parent_evaluation, checkpoint_hash, panels, args, window)
            else:
                record["parent_evaluation"] = score_window(model, dataset, panels, args, name, window,
                    cli.output / name, config, checkpoint=cli.checkpoint)
            save_json(cli.output / "summary.json", record)
    if distributed:
        dist.barrier(group=control_group)
    for epoch in range(start_epoch + 1, epochs + 1):
        if distributed:
            train = train_distributed_epoch(distributed_model, dataset, known, optimizer, args, cli.phase, epoch, cli.output, day_blocks)
        else:
            train = train_epoch(model, dataset, known, optimizer, args, cli.phase, epoch, cli.output, day_blocks)
        if distributed:
            # Nonzero ranks wait on CPU Gloo while rank0 evaluates separately.
            torch.cuda.empty_cache()
        if rank == 0:
            nll = fixed_training_nll(model, dataset, known, args, cli.phase)
            entry = dict(training=train, fixed_training_nll=nll)
            record["epochs"].append(entry)
            # Save the unwrapped generator and real optimizer, independent of
            # the number of training ranks used by this or the next phase.
            epoch_dir = cli.output / f"{cli.phase}_epoch{epoch}"
            epoch_dir.mkdir()
            checkpoint = epoch_dir / "checkpoint.pt"
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                configuration=config, phase=cli.phase, epoch=epoch), checkpoint)
            save_json(epoch_dir / "training.json", entry)
            if epoch in args.evaluation_epochs or epoch == epochs:
                label = f"{cli.phase}_epoch{epoch}_d{window}"
                entry["evaluation"] = score_window(model, dataset, panels, args, label, window,
                    epoch_dir / "evaluation", dict(config, completed_epoch=epoch), checkpoint=checkpoint)
            save_json(cli.output / "summary.json", record)
        if distributed:
            dist.barrier(group=control_group)
    if rank == 0:
        record["complete"] = True
        save_json(cli.output / "summary.json", record)
        print(json.dumps(dict(phase=cli.phase, complete=True, epochs=epochs, output=str(cli.output))), flush=True)
    if distributed:
        dist.barrier(group=control_group)
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

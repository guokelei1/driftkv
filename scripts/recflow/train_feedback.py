#!/usr/bin/env python3
"""202-only Large/Max V0-V5: grouped feedback, row-gradient SUM, epoch recovery."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.recflow.feedback import FeedbackData, feedback_loss, grouped_scores
from hstu_kvcache.training.recovery import cpu_copy, digest


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def setup():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    return rank, world, torch.device("cuda", local)


def barrier(world):
    if world > 1:
        dist.barrier()


def read_run_config(path, scale=None):
    config = json.loads(Path(path).read_text())
    if "models" in config:
        if scale not in config["models"]:
            raise ValueError("Select --scale large or max for the sequential plan")
        selected = config.pop("models")[scale]
        config.update(selected)
    config["scale"] = scale or config.get("scale", "large")
    return config


def version_schedule(design, config):
    stop = int(config["stop_after"][1:])
    return [(int(v["name"][1:]), tuple(v["fit_days_inclusive"]),
        config["base_epochs"] if v["name"] == "v0" else config["update_epochs"],
        design["training"]["foundation_learning_rate"] if v["name"] == "v0" else design["training"]["update_learning_rate"])
        for v in design["training"]["versions"] if int(v["name"][1:]) <= stop]


def build_model(design, device, scale="large"):
    torch.manual_seed(design["training"]["seed"])
    spec = design["scales"][scale]
    cfg = HSTUConfig(**design["shared_model"], num_layers=spec["num_layers"],
        num_items=spec["known_videos"] + design["catalog"]["oov_buckets"])
    # Initialize directly on CUDA; avoids a 3GB temporary CPU-to-GPU staging copy.
    with torch.device(device):
        model = HSTU(cfg)
    assert sum(p.numel() for p in model.parameters()) == {"large": 730211201, "max": 1149285121}[scale]
    return model, cfg


def epoch_order(data, indices, seed, batch_size, bucket=True):
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    if not bucket:
        return order
    # Random windows, then random minibatch order within each length-sorted window.
    # Window size is divisible by global batch, so only the final batch is short.
    output = []
    for offset in range(0, len(order), batch_size * 64):
        part = order[offset:offset + batch_size * 64]
        part = part[np.argsort(data.lengths[part], kind="stable")]
        full = len(part) // batch_size * batch_size
        blocks = part[:full].reshape(-1, batch_size)
        if len(blocks):
            output.append(blocks[rng.permutation(len(blocks))].reshape(-1))
        if full < len(part):
            output.append(part[full:])
    return np.concatenate(output)


def shape_for(data, indices):
    length = int(data.lengths[indices].max())
    width = next(n for n in (128, 256, 512, 768, 1024) if n >= length)
    count = int(data.counts[indices].max())
    candidates = next(n for n in (8, 16, 32) if n >= count)
    return width, candidates


class Backward:
    """Capture only forward/backward; AdamW and distributed SUM stay explicit."""

    def __init__(self, model, graphs):
        self.model = model
        self.graphs = graphs
        self.cache = {}
        self.previous_rows = None
        self.pool = torch.cuda.graph_pool_handle() if graphs else None

    def eager(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            loss, _ = feedback_loss(self.model, batch)
        loss.backward()
        return loss.detach()

    def __call__(self, batch):
        if not self.graphs:
            return self.eager(batch)
        key = (batch[0].shape, batch[4].shape)
        if key not in self.cache:
            # Capture preparation must not change parameters, optimizer or RNG.
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            static = tuple(x.clone() for x in batch)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.model.zero_grad(set_to_none=False)
                    self.eager(static)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.model.zero_grad(set_to_none=False)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                loss = self.eager(static)
            self.model.zero_grad(set_to_none=False)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            self.cache[key] = (graph, static, loss)
        graph, static, loss = self.cache[key]
        for dest, source in zip(static, batch):
            dest.copy_(source)
        graph.replay()
        return loss


def sum_gradients(model, rows, world, dense=False):
    embedding = model.item_emb.weight.grad
    if dense:
        if world > 1:
            dist.all_reduce(embedding)
        selected = None
    else:
        selected = embedding.index_select(0, rows)
        if world > 1:
            dist.all_reduce(selected)
    remaining = [p.grad for p in model.parameters() if p is not model.item_emb.weight and p.grad is not None]
    if world > 1:
        flat = torch._utils._flatten_dense_tensors(remaining)
        dist.all_reduce(flat)
        for target, value in zip(remaining, torch._utils._unflatten_dense_tensors(flat, remaining)):
            target.copy_(value)
    return selected, remaining


def train_step(model, optimizer, backward, data, global_indices, rank, world, microbatch, dense=False, shape=None):
    width, candidates = shape or shape_for(data, global_indices)
    local = global_indices[rank::world]
    if getattr(model, "direct_embedding_gradient", False) and backward.previous_rows is not None:
        # Only these rows were nonzero after the previous completed step.
        model.item_emb.weight.grad.index_fill_(0, backward.previous_rows, 0)
        others = [p.grad for p in model.parameters() if p is not model.item_emb.weight and p.grad is not None]
        torch._foreach_zero_(others)
    else:
        optimizer.zero_grad(set_to_none=False)
    loss = torch.zeros((), device=data.device)
    # Global denominator gives exact request/user weighting after SUM across ranks.
    for offset in range(0, max(1, len(local)), microbatch):
        batch = data.batch(local[offset:offset + microbatch], microbatch, width, candidates, denominator=len(global_indices))
        loss = loss + backward(batch).detach()
    rows = torch.as_tensor(data.embedding_rows(global_indices, width), dtype=torch.long, device=data.device)
    selected, others = sum_gradients(model, rows, world, dense=dense)
    if dense:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    else:
        # Exact dense gradient norm: all omitted embedding rows are zero.
        norms = torch._foreach_norm(others, 2)
        norm = torch.linalg.vector_norm(torch.stack((*norms, selected.norm(2))), 2)
        coefficient = torch.clamp(1.0 / (norm + 1e-6), max=1.0)
        selected.mul_(coefficient)
        torch._foreach_mul_(others, coefficient)
        model.item_emb.weight.grad.index_copy_(0, rows, selected)
    optimizer.step()
    backward.previous_rows = rows
    return loss, norm, len(rows)


def rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def save_recovery(model, optimizer, folder, cursor, binding, rank, world):
    started = time.perf_counter()
    barrier(world)
    states = [None] * world
    local = rng_state()
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        # Different cursor commits at an epoch boundary never replace an
        # already committed generation with the same optimizer-step count.
        tag = f"_v{cursor.get('version', 0)}_e{cursor.get('epoch', 0)}_b{cursor.get('next_batch', 0):06d}"
        target = folder / f"step_{cursor['completed_steps']:09d}{tag}.pt"
        temp = target.with_suffix(".pt.partial")
        payload = dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
            cursor=cursor, binding=binding, rng_by_rank=states)
        with temp.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        marker = target.with_suffix(".json")
        checksum = digest(target)
        atomic_json(marker, dict(file=target.name, bytes=target.stat().st_size, sha256=checksum, cursor=cursor,
            binding=binding, save_seconds=time.perf_counter() - started))
        for old in sorted(folder.glob("step_*.json"))[:-2]:
            old.with_suffix(".pt").unlink(missing_ok=True)
            old.unlink()
        print(json.dumps(dict(recovery_saved=str(target), cursor=cursor)), flush=True)
    barrier(world)


def load_recovery(model, optimizer, folder, binding, rank, world):
    folder = Path(folder)
    markers = list(folder.glob("step_*.json")) + list(folder.parent.glob("v[0-5][12]/checkpoint.json"))
    records = [(path, json.loads(path.read_text())) for path in markers]
    latest, marker = max(records, key=lambda item: (
        item[1]["cursor"]["completed_steps"], item[1]["cursor"].get("version", 0),
        item[1]["cursor"].get("epoch", 0), item[1]["cursor"].get("next_batch", 0)))
    if marker["binding"] != binding:
        raise ValueError("Resume inputs, settings, world size or source code changed")
    path = latest.parent / marker["file"]
    if rank == 0 and (path.stat().st_size != marker["bytes"] or digest(path) != marker["sha256"]):
        raise ValueError("Recovery file hash mismatch")
    if rank == 0:
        # An interruption may occur after a permanent epoch is committed but
        # before its selected-version alias is written.
        for endpoint, record in records:
            if record.get("final_epoch") and record["binding"] == binding:
                link_selected_endpoint(endpoint.parent / record["file"], record)
    barrier(world)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng_by_rank"][rank])
    return payload["cursor"]


def link_selected_endpoint(path, marker):
    alias = path.parent.parent / marker["version"] / "checkpoint.pt"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() or not os.path.samefile(path, alias):
        temporary = alias.with_suffix(".pt.partial")
        temporary.unlink(missing_ok=True)
        os.link(path, temporary)
        os.replace(temporary, alias)
    atomic_json(alias.with_suffix(".json"), dict(marker, path=str(alias), selected_epoch=marker["label"]))


def save_endpoint(model, cfg, optimizer, output, version, epoch, final, parent, cursor, binding, rank, world):
    started = time.perf_counter()
    barrier(world)
    states = [None] * world
    local = rng_state()
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        label = f"{version}{epoch}"
        path = Path(output) / label / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.with_suffix(".json").exists():
            raise FileExistsError(f"Permanent epoch already committed: {path}; resume instead")
        temporary = path.with_suffix(".pt.partial")
        payload = dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
            rng_by_rank=states, config=vars(cfg), version=version, epoch=epoch, label=label,
            parent_epoch=parent, task="effective_view_binary_feedback", cursor=cursor, binding=binding)
        steps = sorted({int(state["step"]) for state in payload["optimizer"]["state"].values()})
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        checksum = digest(path)
        marker = dict(file=path.name, path=str(path), bytes=path.stat().st_size, sha256=checksum,
            cursor=cursor, binding=binding, label=label, parent_epoch=parent, optimizer_state_steps=steps,
            version=version, final_epoch=final,
            parameters=sum(p.numel() for p in model.parameters()), save_seconds=time.perf_counter() - started)
        atomic_json(path.with_suffix(".json"), marker)
        if final:
            # Selected-version names share the permanent epoch's file.
            link_selected_endpoint(path, marker)
        print(json.dumps(dict(permanent_epoch_saved=label, parent=parent, optimizer_state_steps=steps)), flush=True)
    barrier(world)


def benchmark(model, optimizer, data, cfg, output, rank, world):
    indices = data.indices(0, 1, 18)
    data.set_training_weights(indices)
    order = epoch_order(data, indices, 17, cfg["global_batch"])
    # Predeclared mixture of history lengths; no test users or quality metrics.
    positions = np.linspace(0, min(4096, len(order) // cfg["global_batch"] - 1), 12).astype(int)
    batches = [order[p * cfg["global_batch"]:(p + 1) * cfg["global_batch"]] for p in positions]
    results = []
    for graphs, dense, direct in [(False, True, False), (False, False, False), (False, False, True), (True, False, True)]:
        model.direct_embedding_gradient = direct
        torch.cuda.reset_peak_memory_stats()
        backward = Backward(model, graphs)
        for batch in batches:
            train_step(model, optimizer, backward, data, batch, rank, world, cfg["microbatch"], dense)
        barrier(world)
        torch.cuda.synchronize()
        start = time.perf_counter()
        values = []
        for batch in batches[4:]:
            loss, norm, rows = train_step(model, optimizer, backward, data, batch, rank, world, cfg["microbatch"], dense)
            values.append((loss, norm, rows))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if world > 1:
            elapsed = torch.tensor(seconds, device=data.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            seconds = elapsed.item()
        assert all(torch.isfinite(v[0]).item() and torch.isfinite(v[1]).item() for v in values)
        row = dict(cuda_graphs=graphs, dense_embedding_communication=dense, world_size=world,
            direct_embedding_gradient=direct, global_batch=cfg["global_batch"],
            timed_steps=len(values), seconds_per_step=seconds / len(values), requests_per_second=len(values) * cfg["global_batch"] / seconds,
            average_communicated_embedding_rows=float(np.mean([v[2] for v in values])),
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
        results.append(row)
        if rank == 0:
            atomic_json(Path(output) / "summary.json", dict(status="development_speed_probe", hostname=socket.gethostname(), cells=results, quality_claim=False))
            print(json.dumps(row), flush=True)
        del backward
        torch.cuda.empty_cache()
    return results


def profile(model, optimizer, data, cfg, design, output, rank, world):
    """Measure every actual (padded history, candidate count) training shape."""
    model.direct_embedding_gradient = True
    backward = Backward(model, True)
    cells, schedules = {}, {}
    for number, days, epochs, _ in version_schedule(design, cfg):
        version = f"v{number}"
        indices = data.indices(0, *days)
        data.set_training_weights(indices)
        order = epoch_order(data, indices, 17, cfg["global_batch"])
        frequencies, examples = {}, {}
        for offset in range(0, len(order), cfg["global_batch"]):
            batch = order[offset:offset + cfg["global_batch"]]
            shape = shape_for(data, batch)
            frequencies[shape] = frequencies.get(shape, 0) + 1
            examples.setdefault(shape, batch)
        seconds = 0
        for shape, frequency in sorted(frequencies.items()):
            batch = examples[shape]
            # Each version's weights/order remain training-only probe inputs.
            for _ in range(2):
                train_step(model, optimizer, backward, data, batch, rank, world, cfg["microbatch"])
            torch.cuda.synchronize()
            barrier(world)
            begin = time.perf_counter()
            for _ in range(6):
                loss, norm, rows = train_step(model, optimizer, backward, data, batch, rank, world, cfg["microbatch"])
            torch.cuda.synchronize()
            elapsed = torch.tensor((time.perf_counter() - begin) / 6, device=data.device)
            if world > 1:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
            measured = elapsed.item()
            seconds += frequency * measured * epochs
            key = f"{version}_n{shape[0]}_c{shape[1]}"
            cells[key] = dict(width=shape[0], candidates=shape[1], batches_per_epoch=frequency,
                seconds_per_step=measured, communicated_embedding_rows=rows)
            if rank == 0:
                print(json.dumps(dict(profile_cell=key, **cells[key])), flush=True)
        schedules[version] = dict(epochs=epochs, steps=epochs * sum(frequencies.values()),
            seconds=seconds, requests=len(indices), targets=int(data.counts[indices].sum()))
    if rank == 0:
        total = sum(row["seconds"] for row in schedules.values())
        atomic_json(Path(output) / "summary.json", dict(status="complete_input_shape_weighted_speed_profile",
            scale=cfg["scale"], world_size=world, global_batch=cfg["global_batch"], cuda_graphs=True, direct_embedding_gradient=True,
            schedules=schedules, cells=cells, estimated_training_hours=total / 3600,
            planning_hours_with_25percent_margin=[total / 3600, total / 3600 * 1.25],
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
            scope="measured warmed steps weighted by every configured version's real batch shapes; startup/checkpoint/evaluation excluded; no quality result"))


def memory_probe(model, optimizer, data, cfg, output, binding, rank, world, resume):
    """All 15 padded shapes, mixed replays, full save and fresh-process restore."""
    if resume:
        load_recovery(model, optimizer, Path(output) / "recovery", binding, rank, world)
    backward = Backward(model, True)
    indices = data.indices(0, 1, 18)
    data.set_training_weights(indices)
    batches = {}
    for width in (128, 256, 512, 768, 1024):
        for candidates in (8, 16, 32):
            eligible = indices[(data.lengths[indices] <= width) & (data.counts[indices] <= candidates)]
            chosen = eligible[np.linspace(0, len(eligible) - 1, cfg["global_batch"]).astype(int)]
            batches[width, candidates] = chosen
    control = [(1024, 32), (768, 16)]
    records = []

    def observe(phase):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        values = torch.tensor([torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
            torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), total - free],
            dtype=torch.float64, device=data.device)
        if world > 1:
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        row = dict(phase=phase, allocated_gb=values[0].item() / 1e9, reserved_gb=values[1].item() / 1e9,
            peak_allocated_gb=values[2].item() / 1e9, peak_reserved_gb=values[3].item() / 1e9,
            driver_free_gb=(total - values[4].item()) / 1e9, cached_graphs=len(backward.cache))
        records.append(row)
        if rank == 0:
            atomic_json(Path(output) / ("resume_memory.json" if resume else "memory.json"), records)
            print(json.dumps(row), flush=True)

    observe("after_fresh_process_restore" if resume else "initial")
    # A real restarted process has an empty graph cache. Test its first two
    # steps against the controls continued immediately after the saved state.
    if resume:
        losses = []
        for shape in control:
            loss, norm, _ = train_step(model, optimizer, backward, data, batches[shape], rank, world, cfg["microbatch"], shape=shape)
            assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
            losses.append(loss.item())
        expected = json.loads((Path(output) / "continuation.json").read_text())["losses_by_rank"][rank]
        error = float(np.max(np.abs(np.array(losses) - expected)))
        if world > 1:
            value = torch.tensor(error, device=data.device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            error = value.item()
        assert error <= 2e-4, (losses, expected)
        observe("restored_control_steps")
    for shape, batch in batches.items():
        for _ in range(2):
            loss, norm, _ = train_step(model, optimizer, backward, data, batch, rank, world, cfg["microbatch"], shape=shape)
        assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
        observe(f"captured_n{shape[0]}_c{shape[1]}")
    shapes = list(batches)
    for step in range(120):
        shape = shapes[(step * 7) % len(shapes)]
        loss, norm, _ = train_step(model, optimizer, backward, data, batches[shape], rank, world, cfg["microbatch"], shape=shape)
        if (step + 1) % 20 == 0:
            assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
            observe(f"mixed_replay_{step + 1}")
    if not resume:
        save_recovery(model, optimizer, Path(output) / "recovery",
            dict(completed_steps=150, version=0, epoch=0, next_batch=150), binding, rank, world)
        observe("after_full_checkpoint_save")
        losses = []
        for shape in control:
            loss, _, _ = train_step(model, optimizer, backward, data, batches[shape], rank, world, cfg["microbatch"], shape=shape)
            losses.append(loss.item())
        all_losses = [None] * world
        if world > 1:
            dist.all_gather_object(all_losses, losses)
        else:
            all_losses[0] = losses
        if rank == 0:
            atomic_json(Path(output) / "continuation.json", dict(losses_by_rank=all_losses))
    if rank == 0:
        name = "summary.json" if resume else "save_summary.json"
        atomic_json(Path(output) / name, dict(status="passed", scale=cfg["scale"], world_size=world,
            global_batch=cfg["global_batch"], microbatch=cfg["microbatch"], all_15_shapes=True,
            mixed_replay_steps=120, full_checkpoint_saved=not resume, fresh_process_resumed=resume,
            max_resumed_loss_difference=error if resume else None, records=records,
            scope="development inputs only; finite bounded mixed-shape replay and full recovery, not a complete long training run"))


def verify(model, optimizer, data, config, output, binding, rank, world):
    indices = data.indices(0, 1, 18)
    data.set_training_weights(indices)
    batch = epoch_order(data, indices, 17, config["global_batch"])[:config["global_batch"]]
    backward = Backward(model, config["cuda_graphs"])
    model.direct_embedding_gradient = False
    initial = cpu_copy(model.state_dict())
    initial_rng = rng_state()
    train_step(model, optimizer, backward, data, batch, rank, world, config["microbatch"], dense=True)
    dense_result = cpu_copy(model.state_dict())
    dense_moments = cpu_copy(optimizer.state_dict())
    model.load_state_dict(initial)
    optimizer.state.clear()
    model.direct_embedding_gradient = True
    backward = Backward(model, config["cuda_graphs"])
    restore_rng(initial_rng)
    train_step(model, optimizer, backward, data, batch, rank, world, config["microbatch"])
    communication_error = max((value.detach().cpu() - dense_result[name]).abs().max().item() for name, value in model.state_dict().items())
    communication_moment_error = max((value.detach().cpu() - dense_moments["state"][key][name]).abs().max().item()
        for key, state in optimizer.state_dict()["state"].items() for name, value in state.items() if torch.is_tensor(value))
    del initial, dense_result, dense_moments
    save_recovery(model, optimizer, Path(output) / "recovery", dict(completed_steps=1), binding, rank, world)
    losses = []
    for _ in range(2):
        loss, _, _ = train_step(model, optimizer, backward, data, batch, rank, world, config["microbatch"])
        losses.append(loss.item())
    final = cpu_copy(model.state_dict())
    moments = cpu_copy(optimizer.state_dict())
    load_recovery(model, optimizer, Path(output) / "recovery", binding, rank, world)
    # Rebuild all CUDA graphs, as a restarted process would; restore is not
    # allowed to depend on a surviving graph cache or gradient buffer contents.
    backward = Backward(model, config["cuda_graphs"])
    resumed = []
    for _ in range(2):
        loss, _, _ = train_step(model, optimizer, backward, data, batch, rank, world, config["microbatch"])
        resumed.append(loss.item())
    error = max((value.detach().cpu() - final[name]).abs().max().item() for name, value in model.state_dict().items())
    moment_error = max((value.detach().cpu() - moments["state"][key][name]).abs().max().item()
        for key, state in optimizer.state_dict()["state"].items() for name, value in state.items() if torch.is_tensor(value))
    passed = max(error, moment_error, communication_error, communication_moment_error) <= 2e-5 and np.max(np.abs(np.array(losses) - resumed)) <= 2e-4
    if world > 1:
        flags = torch.tensor([int(not passed), error, moment_error, communication_error, communication_moment_error], device=data.device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        passed = not bool(flags[0].item())
        error, moment_error, communication_error, communication_moment_error = flags[1:].tolist()
    if rank == 0:
        atomic_json(Path(output) / "summary.json", dict(status="passed" if passed else "failed", full_model_scale=config["scale"],
            control_losses=losses, resumed_losses=resumed, max_parameter_difference=error,
            max_optimizer_difference=moment_error, dense_vs_row_parameter_difference=communication_error,
            dense_vs_row_optimizer_difference=communication_moment_error, world_size=world, cuda_graphs=config["cuda_graphs"]))
    assert passed, (losses, resumed, error, moment_error)


def train(model, model_cfg, optimizer, data, config, design, output, binding, rank, world, resume):
    cursor = dict(version=0, epoch=0, next_batch=0, completed_steps=0)
    if resume:
        cursor = load_recovery(model, optimizer, Path(output) / "recovery", binding, rank, world)
    backward = Backward(model, config["cuda_graphs"])
    last_save = time.monotonic()
    start = time.perf_counter()
    for version, days, epochs, lr in version_schedule(design, config):
        if version < cursor["version"]:
            continue
        indices = data.indices(0, *days)
        population = data.set_training_weights(indices)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for epoch in range(epochs):
            if version == cursor["version"] and epoch < cursor["epoch"]:
                continue
            order = epoch_order(data, indices, design["training"]["seed"] + version * 1000 + epoch, config["global_batch"])
            count = math.ceil(len(order) / config["global_batch"])
            first = cursor["next_batch"] if version == cursor["version"] and epoch == cursor["epoch"] else 0
            for step in range(first, count):
                batch = order[step * config["global_batch"]:(step + 1) * config["global_batch"]]
                loss, norm, _ = train_step(model, optimizer, backward, data, batch, rank, world, config["microbatch"])
                cursor = dict(version=version, epoch=epoch, next_batch=step + 1, completed_steps=cursor["completed_steps"] + 1)
                if step % 100 == 0 or step + 1 == count:
                    values = torch.stack((loss, norm))
                    if world > 1:
                        dist.all_reduce(values[:1])
                    if not torch.isfinite(values).all().item():
                        raise FloatingPointError("Nonfinite loss/gradient: stop without overwriting recovery")
                    save = torch.tensor(int(rank == 0 and (cursor["completed_steps"] == 101 or time.monotonic() - last_save >= config["checkpoint_seconds"])), device=data.device)
                    if world > 1:
                        dist.broadcast(save, src=0)
                    if rank == 0:
                        progress = dict(status="training", version=f"v{version}", epoch=epoch + 1, epoch_steps=step + 1,
                            steps_per_epoch=count, cursor=cursor, loss=values[0].item(), grad_norm=values[1].item(),
                            elapsed_seconds=time.perf_counter() - start, training_population=population, world_size=world,
                            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
                        atomic_json(Path(output) / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                    if save.item():
                        save_recovery(model, optimizer, Path(output) / "recovery", cursor, binding, rank, world)
                        last_save = time.monotonic()
            cursor.update(epoch=epoch + 1, next_batch=0)
            parent = f"v{version}{epoch}" if epoch else (
                f"v{version - 1}{config['base_epochs'] if version == 1 else config['update_epochs']}" if version else None)
            if epoch + 1 == epochs:
                cursor.update(version=version + 1, epoch=0)
            save_endpoint(model, model_cfg, optimizer, output, f"v{version}", epoch + 1,
                epoch + 1 == epochs, parent, cursor.copy(), binding, rank, world)
            # The permanent payload is itself a complete recovery point.
            last_save = time.monotonic()
    if rank == 0:
        atomic_json(Path(output) / "progress.json", dict(status=f"training_complete_stopped_after_{config['stop_after']}",
            scale=config["scale"], cursor=cursor, binding=binding,
            elapsed_seconds=time.perf_counter() - start, next="paired future-day evaluation of adjacent versions"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["benchmark", "profile", "memory", "verify", "train"], required=True)
    parser.add_argument("--scale", choices=["large", "max"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_run_config(args.config, args.scale)
    if args.mode == "train" and not config.get("training_authorized", False):
        raise RuntimeError("User requires waiting after the speed/ETA report; formal training is not authorized")
    rank, world, device = setup()
    assert socket.gethostname() == "g202", "User explicitly requires machine 202"
    assert world == len(config["physical_gpus"]), "GPU count must match the frozen run configuration"
    design = json.loads(Path(config["design"]).read_text())
    if rank == 0:
        manifest = json.loads((Path(config["data"]) / "manifest.json").read_text())
        for root, files in [(Path(manifest["source_root"]), manifest["source_inputs"]), (Path(config["data"]), manifest["outputs"])]:
            for name, record in files.items():
                path = root / name
                if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError(f"Frozen input changed: {path}")
    barrier(world)
    data = FeedbackData(config["data"], device)
    assert data.manifest["known_videos"] == design["scales"][config["scale"]]["known_videos"]
    assert data.manifest["oov_buckets"] == design["catalog"]["oov_buckets"]
    binding = dict(configuration_sha256=digest(args.config), design_sha256=digest(config["design"]),
        scale=config["scale"],
        data_manifest_sha256=digest(Path(config["data"]) / "manifest.json"), torch_version=str(torch.__version__),
        world_size=world, source_sha256={path: digest(path) for path in [__file__, "src/hstu_kvcache/recflow/feedback.py", "src/hstu_kvcache/recflow/feedback_embedding.py", "src/hstu_kvcache/recflow/feedback_embedding_cuda.py", "src/hstu_kvcache/models/hstu.py", "src/hstu_kvcache/models/block.py", "src/hstu_kvcache/models/embeddings.py", "src/hstu_kvcache/models/rmsnorm.py", "src/hstu_kvcache/models/attention.py", "src/hstu_kvcache/models/triton_attention.py"]})
    model, model_cfg = build_model(design, device, config["scale"])
    model.direct_embedding_gradient = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=design["training"]["foundation_learning_rate"],
        weight_decay=design["training"]["weight_decay"], fused=True)
    torch.manual_seed(17 + rank * 1000)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        filename = "resume_configuration.json" if args.resume else "configuration.json"
        atomic_json(args.output / filename, dict(config=config, binding=binding, model=vars(model_cfg), mode=args.mode))
    if args.mode == "benchmark":
        benchmark(model, optimizer, data, config, args.output, rank, world)
    elif args.mode == "profile":
        profile(model, optimizer, data, config, design, args.output, rank, world)
    elif args.mode == "memory":
        memory_probe(model, optimizer, data, config, args.output, binding, rank, world, args.resume)
    elif args.mode == "verify":
        verify(model, optimizer, data, config, args.output, binding, rank, world)
    else:
        train(model, model_cfg, optimizer, data, config, design, args.output, binding, rank, world, args.resume)
    barrier(world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

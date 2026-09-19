#!/usr/bin/env python3
"""Single-node DDP numerical/resource canary; never a learning experiment.

Run with CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2.
The global batch remains 128. Fixed catalog buffers do not need per-step sync.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import time
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
    save_json,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class LossWrapper(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, **batch):
        return self.generator.loss_per_example(**batch)


def wrap(model, device):
    # HSTU query/readout parameters are unused by this generator. The used
    # parameter set is constant even though target leaf groups change.
    return DDP(LossWrapper(model), device_ids=[device.index], broadcast_buffers=False,
               static_graph=True, gradient_as_bucket_view=True)


def broadcast_batch(batch, device):
    """Rank0 owns the frozen global inputs/targets; distribute no extra rows."""
    rank, world = dist.get_rank(), dist.get_world_size()
    shape = torch.tensor(batch['item_ids'].shape if rank == 0 else [0, 0],
                         dtype=torch.long, device=device)
    dist.broadcast(shape, src=0)
    count, width = shape.tolist()
    if count < world:
        raise ValueError('This helper requires at least one real row per rank; current epoch tails satisfy this.')
    lo, hi = (count * rank // world, count * (rank + 1) // world)
    local = {}
    for key, dtype, size in (
        ('item_ids', torch.long, (count, width)), ('behaviors', torch.long, (count, width)),
        ('time_deltas', torch.float32, (count, width)), ('lengths', torch.long, (count,)),
        ('query_time_deltas', torch.float32, (count,)), ('targets', torch.long, (count,)),
    ):
        values = batch[key].to(device) if rank == 0 else torch.empty(size, dtype=dtype, device=device)
        dist.broadcast(values, src=0)
        local[key] = values[lo:hi]
    return local, count


def train_epoch_ddp(model, optimizer, global_batches, global_steps, device, bf16=True, clip_norm=1.0,
                    progress_callback=None, log_every=0):
    """Consume one rank0 global-batch iterator, e.g. a complete shuffled epoch.

    Other ranks pass None. There is no sampler padding or timed early exit.
    Rank0 can lazily prepare batches with the same epoch RNG as window_chain.
    The caller supplies the exact number of batches, including the real tail.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    batches = iter(global_batches) if rank == 0 else None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    processed, loss_sum = 0, torch.zeros((), device=device, dtype=torch.float64)
    target_hash = hashlib.sha256()
    for step in range(global_steps):
        batch = next(batches) if rank == 0 else None
        if rank == 0:
            target_hash.update(batch['targets'].numpy().tobytes())
        local, count = broadcast_batch(batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            losses = model(**local)
            # DDP averages gradients over ranks. This reproduces sum(global
            # losses)/globalN even when ranks own different tail-batch sizes.
            loss = losses.sum() * (world / count)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite distributed loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite distributed gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        processed += count
        loss_sum += losses.detach().double().sum()
        if log_every and (step == 0 or (step + 1) % log_every == 0):
            global_loss = loss_sum.clone()
            dist.all_reduce(global_loss)
            if rank == 0 and progress_callback is not None:
                progress_callback(dict(step=step + 1, requests=processed,
                    request_mean_loss=global_loss.item() / processed,
                    seconds=time.monotonic() - started))
    dist.all_reduce(loss_sum)
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.monotonic() - started, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    memory = [None] * world
    dist.all_gather_object(memory, dict(rank=rank,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    return dict(global_steps=global_steps, global_requests=processed,
                seconds=elapsed.item(), requests_per_second=processed / elapsed.item(),
                mean_training_loss=loss_sum.item() / processed,
                target_sha256=target_hash.hexdigest() if rank == 0 else None, rank_memory=memory)


def numerical_canary(device):
    """Two unequal local-batch updates against the actual generator on one GPU."""
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(17)
    cfg = HSTUConfig(num_items=7, num_behaviors=4, hidden_size=16, num_layers=2,
        num_heads=2, max_seq_len=11, input_dropout=0.0, attn_dropout=0.0,
        block_variant='hstu_reference', activation='silu', relative_position_bias=True)
    paths = torch.tensor([[-1,-1],[10,100],[10,100],[10,200],[20,200],[20,300],[20,300],[-1,-1]])
    generator = RecFlowGenerator(cfg, paths).to(device)
    reference = copy.deepcopy(generator) if rank == 0 else None
    model = wrap(generator, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=1e-4) if rank == 0 else None
    reports = []
    for count in ((5, 3) if world == 2 else (5, 7)):
        rng = np.random.default_rng(17 + count)
        batch = dict(item_ids=torch.tensor(rng.integers(1, 7, (count, 5))),
                     behaviors=torch.tensor(rng.integers(0, 4, (count, 5))),
                     time_deltas=torch.tensor(rng.uniform(0, 30, (count, 5)), dtype=torch.float32),
                     lengths=torch.tensor([5, 3, 4, 2, 1, 4, 2][:count]),
                     query_time_deltas=torch.tensor(rng.uniform(0, 20, count), dtype=torch.float32),
                     targets=torch.tensor([1, 4, 2, 6, 3, 5, 1][:count]))
        if rank == 0:
            reference_optimizer.zero_grad(set_to_none=True)
            reference.loss_per_example(**{k:v.to(device) for k,v in batch.items()}).mean().backward()
        optimizer.zero_grad(set_to_none=True)
        local, global_count = broadcast_batch(batch if rank == 0 else None, device)
        (model(**local).sum() * (world / global_count)).backward()
        grad_error = 0.0
        if rank == 0:
            for (name, actual), (_, expected) in zip(generator.named_parameters(), reference.named_parameters(), strict=True):
                if expected.grad is None:
                    assert actual.grad is None, name
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5, msg=name)
                    grad_error = max(grad_error, float((actual.grad - expected.grad).abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if rank == 0:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_optimizer.step()
            update_error = 0.0
            for (name, actual), (_, expected) in zip(generator.named_parameters(), reference.named_parameters(), strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5, msg=name)
                update_error = max(update_error, float((actual - expected).detach().abs().max()))
            reports.append(dict(global_count=count,
                local_counts=[count*(r+1)//world-count*r//world for r in range(world)],
                max_gradient_abs_error=grad_error, max_parameter_abs_error=update_error))
        dist.barrier()
    return dict(passed=True, precision='FP32', optimizer='AdamW lr=.001 wd=1e-4 clip=1',
                cases=reports, weighting='local_loss.sum() * world_size / globalN before DDP averaging')


def make_model(dataset, device, args):
    torch.manual_seed(17)
    cfg = HSTUConfig(num_items=dataset.oov, num_behaviors=4, num_prediction_items=len(dataset.raw_ids),
        hidden_size=args.hidden, num_layers=args.layers, num_heads=args.heads, max_seq_len=1027,
        block_variant='hstu_reference', activation='silu', gating='silu_gate',
        relative_position_bias=True, input_dropout=0.0, attn_dropout=0.0)
    return RecFlowGenerator(cfg, torch.tensor(dataset.paths)).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--skip-single', action='store_true',
                        help='Measure distributed resources only when global128 may not fit one GPU')
    args = parser.parse_args()
    rank, local_rank, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world not in (2, 4):
        raise ValueError('This resource canary uses the authorized two or four GPUs')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    numerical = numerical_canary(device)
    gc.collect()
    torch.cuda.empty_cache()
    if rank == 0:
        save_json(args.output / 'numerical.json', numerical)
        print(json.dumps(numerical), flush=True)
    prepared = PreparedRecFlow(ROOT / 'data/processed/recflow_v1')
    dataset = ProbeData(prepared, 1000000, 512, 1024)
    batches, preparation_seconds = None, None
    if rank == 0:
        known = np.array([i for i in dataset.indices(1,18,None) if len(dataset.targets(i)[1])], dtype=np.int64)
        rng = np.random.default_rng(17)
        order = rng.permutation(known)[:14*128]
        start = time.monotonic()
        batches = [dataset.batch(order[i:i+128], 'cpu', rng) for i in range(0,len(order),128)]
        preparation_seconds = time.monotonic() - start
        sources = [Path(__file__).resolve(), ROOT/'scripts/recflow/development_probe.py',
                   ROOT/'src/hstu_kvcache/recflow/model.py', ROOT/'src/hstu_kvcache/recflow/data.py']
        config = dict(scope='Numerical/resource canary only; no learning or admission evidence.',
            physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), world_size=world, global_batch_size=128,
            local_batch_size=128 // world, training_seed=17, layers=args.layers,
            hidden=args.hidden, heads=args.heads, context=1024,
            catalog_items=len(dataset.raw_ids), cohort_users=512, warmup_steps=2, measured_steps=12,
            skip_single_gpu_baseline=args.skip_single,
            precision='BF16 autocast; FP32 parameters/loss/AdamW states',
            data_preparation_seconds_14_batches=preparation_seconds,
            frozen_request_indices_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        save_json(args.output / 'configuration.json', config)
    # Controlled single-GPU timing on the same physical rank0 device and exact
    # frozen global batches. The second rank remains idle during this baseline.
    single = None
    if rank == 0 and not args.skip_single:
        model = make_model(dataset, device, args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        for step, batch in enumerate(batches):
            if step == 2:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = model.loss_per_example(**{k:v.to(device) for k,v in batch.items()}).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.monotonic()-start
        single = dict(seconds=elapsed, requests=12*128, requests_per_second=12*128/elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    dist.barrier()
    generator = make_model(dataset,device,args)
    model = wrap(generator, device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    train_epoch_ddp(model,optimizer,batches[:2] if rank==0 else None,2,device)
    measured = train_epoch_ddp(model,optimizer,batches[2:] if rank==0 else None,12,device)
    # A separate synchronized all-reduce measures dense item-gradient traffic;
    # it is diagnostic bandwidth cost, not assumed unhidden training overhead.
    dense = torch.zeros_like(generator.backbone.item_emb.weight)
    dist.all_reduce(dense)
    torch.cuda.synchronize(device)
    dist.barrier()
    start = time.monotonic()
    for _ in range(6):
        dist.all_reduce(dense)
    torch.cuda.synchronize(device)
    comm_seconds = (time.monotonic()-start)/6
    if rank == 0:
        measured['requests_per_second_including_proportional_preparation'] = 1536 / (measured['seconds'] + preparation_seconds*12/14)
        result = dict(configuration=config, numerical=numerical, single_gpu_matched=single,
            ddp=measured, parameters=sum(p.numel() for p in generator.parameters()),
            dense_embedding_gradient=dict(bytes=dense.numel()*dense.element_size(),
                isolated_all_reduce_seconds=comm_seconds, repeated_measurements=6,
                caveat='Standalone synchronized cost; DDP may overlap communication with backward.'),
            speedup_vs_matched_single=(measured['requests_per_second']/single['requests_per_second']
                                      if single else None),
            limitations='Frozen CPU batches; broadcast/device transfer and optimizer included, input preparation reported separately. Warmup2 + measured12 updates are resource-only. No model checkpoint or quality evaluation is saved.')
        save_json(args.output/'summary.json',result)
        print(json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

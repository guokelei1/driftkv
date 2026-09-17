#!/usr/bin/env python3
"""Bounded four-rank synthetic save/restart equivalence and cost probe."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training.recovery import load_recovery, save_recovery

ROOT = Path(__file__).resolve().parents[2]


def equal(a, b):
    if torch.is_tensor(a):
        # GPU embedding backward may differ by a rounding bit across processes.
        return torch.allclose(a, b, rtol=1e-6, atol=1e-8) if a.is_floating_point() else torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['reference', 'interrupt', 'resume'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-model', action='store_true')
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=torch.device(f'cuda:{rank}'))
    spec = importlib.util.spec_from_file_location('trainer', ROOT / 'scripts/train_yambda500m_foundation_fsdp.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    cfg = HSTUConfig(num_items=3530906 if args.max_model else 128,
                     hidden_size=320 if args.max_model else 32,
                     num_layers=16 if args.max_model else 2,
                     num_heads=10 if args.max_model else 2,
                     max_seq_len=1024 if args.max_model else 16,
                     num_behaviors=4, input_dropout=0.1,
                     num_query_types=3, query_type_id=2,
                     num_query_actions=1, query_action_id=0)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    model = FSDP(trainer.FoundationForward(HSTU(cfg)), device_id=rank,
                 use_orig_params=True, sync_module_states=True, limit_all_gathers=True,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                reduce_dtype=torch.float32, buffer_dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    binding = {'probe': 'Max recovery equivalence', 'full_model': args.max_model}
    split = args.output / 'split'
    start = 0
    load_seconds = None
    torch.cuda.reset_peak_memory_stats()
    if args.mode == 'resume':
        started = time.perf_counter()
        start = load_recovery(model, optimizer, split / 'step_000000002', binding)
        load_seconds = time.perf_counter() - started
    end = 2 if args.mode == 'interrupt' else 4
    batch = 20 if args.max_model else 2
    for step in range(start, end):
        g = torch.Generator().manual_seed(1000 + step * 4 + rank)
        items = torch.randint(1, cfg.num_items, (batch, cfg.max_seq_len), generator=g).cuda()
        behaviors = torch.ones_like(items)
        deltas = torch.zeros_like(items, dtype=torch.float32)
        candidates = torch.randint(1, cfg.num_items, (batch, 1), generator=g).cuda()
        logits = model(items, behaviors, deltas, candidates,
                       torch.zeros(batch, device='cuda'),
                       torch.full((batch,), cfg.max_seq_len, device='cuda'))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    save_seconds = save_recovery(model, optimizer,
                                 args.output / 'reference' if args.mode == 'reference' else split,
                                 end, binding)
    if args.mode == 'resume':
        reference = torch.load(args.output / f'reference/step_000000004/rank_{rank}.pt', map_location='cpu', weights_only=False)
        actual = torch.load(split / f'step_000000004/rank_{rank}.pt', map_location='cpu', weights_only=False)
        for key in ['parameters', 'buffers', 'optimizer', 'torch_rng', 'cuda_rng']:
            assert equal(reference[key], actual[key]), key
    metrics = [None] * dist.get_world_size()
    dist.all_gather_object(metrics, {'rank': rank, 'save_seconds': save_seconds,
                                    'load_seconds': load_seconds,
                                    'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
    if rank == 0:
        report = {'mode': args.mode, 'status': 'passed', 'full_Max_model': args.max_model,
                  'numerical_restart_equivalence': args.mode == 'resume',
                  'comparison_rtol': 1e-6, 'comparison_atol': 1e-8, 'rank_metrics': metrics}
        (args.output / f'{args.mode}.json').write_text(json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

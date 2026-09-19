#!/usr/bin/env python3
"""Bounded, development-only generative retrieval and parent/current probes."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hstu_kvcache.models.hstu import HSTUConfig  # noqa: E402
from hstu_kvcache.recflow.data import PreparedRecFlow  # noqa: E402
from hstu_kvcache.recflow.metrics import (  # noqa: E402
    aggregate_metrics,
    random_expected_metrics,
    request_metrics,
    sample_candidates,
)
from hstu_kvcache.recflow.model import RecFlowGenerator  # noqa: E402


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ProbeData:
    def __init__(self, prepared, catalog_size, cohort_users, context):
        self.prepared = prepared
        self.context = context
        self.uids = prepared.cohort(min_history=1024, limit=cohort_users)
        catalog = prepared.catalog
        count = np.asarray(catalog["development_exposure_count"])
        n = len(catalog["raw_item_ids"])
        # Selection only uses initial development exposure frequency, then raw
        # ID breaks ties. It does not inspect any future relevance labels.
        select = np.lexsort((catalog["raw_item_ids"], -count))[:min(catalog_size or n, n)]
        select = np.sort(select)
        self.raw_ids = np.asarray(catalog["raw_item_ids"])[select]
        self.paths = np.full((len(select) + 2, 2), -1, dtype=np.int64)
        self.paths[1:-1, 0] = np.asarray(catalog["c1"])[select]
        self.paths[1:-1, 1] = np.asarray(catalog["c2"])[select]
        self.oov = len(select) + 1
        self.mapping = np.full(n + 2, self.oov, dtype=np.int64)
        self.mapping[0] = 0
        self.mapping[select + 1] = np.arange(1, len(select) + 1)
        self.catalog_set = set(map(int, self.raw_ids))
        self.popularity = np.asarray(catalog["development_effective_count"])[select].astype(float)
        self.popular_ranking = self.raw_ids[np.lexsort((self.raw_ids, -self.popularity))[:100]].tolist()

    def indices(self, first_day, last_day, limit):
        return np.asarray(self.prepared.request_indices(
            day_start=first_day, day_end=last_day, uids=self.uids, limit=limit,
            positive_only=True,
        ), dtype=np.int64)

    def targets(self, index):
        raw, mapped = self.prepared.targets(int(index))
        local = self.mapping[np.asarray(mapped, dtype=np.int64)]
        return np.asarray(raw), local[local != self.oov]

    def batch(self, indices, device, rng=None):
        count = len(indices)
        items = np.zeros((count, self.context), dtype=np.int64)
        behaviors = np.zeros_like(items)
        deltas = np.zeros((count, self.context), dtype=np.float32)
        lengths = np.zeros(count, dtype=np.int64)
        query_gaps = np.zeros(count, dtype=np.float32)
        targets = []
        for row, index in enumerate(indices):
            ii, bb, tt = self.prepared.history(int(index), max_length=self.context)
            length = len(ii)
            items[row, :length] = self.mapping[np.asarray(ii, dtype=np.int64)]
            behaviors[row, :length] = bb
            deltas[row, :length] = tt
            lengths[row] = length
            if length:
                request = self.prepared.requests[index]
                previous_ts = self.prepared.events[int(request['history_stop'])-1]['ts']
                query_gaps[row] = (int(request['ts'])-int(previous_ts))/1000.0
            if rng is not None:
                _, positives = self.targets(index)
                targets.append(int(rng.choice(positives)))
        width = max(int(lengths.max()), 1)
        inputs = dict(item_ids=torch.from_numpy(items[:, :width]).to(device),
            behaviors=torch.from_numpy(behaviors[:, :width]).to(device),
            time_deltas=torch.from_numpy(deltas[:, :width]).to(device),
            lengths=torch.from_numpy(lengths).to(device),
            query_time_deltas=torch.from_numpy(query_gaps).to(device))
        if rng is not None:
            inputs['targets'] = torch.tensor(targets, device=device)
        return inputs


def train_phase(model, dataset, indices, optimizer, args, phase, out):
    known = np.array([i for i in indices if len(dataset.targets(i)[1])], dtype=np.int64)
    if not len(known):
        raise RuntimeError('The selected training requests have no in-catalog targets.')
    rng = np.random.default_rng(args.seed + (1000 if phase == 'current' else 0))
    begin = time.monotonic()
    losses = []
    processed = 0
    steps = args.current_steps if phase == 'current' else args.steps
    model.train()
    optimizer.zero_grad(set_to_none=True)
    order = rng.permutation(known)
    cursor = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
    for step in range(steps):
        if cursor >= len(order):
            order = rng.permutation(known)
            cursor = 0
        batch_indices = order[cursor:cursor + args.batch_size]
        cursor += len(batch_indices)
        batch = dataset.batch(batch_indices, args.device, rng)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
            loss = model.loss_per_example(**batch).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite {phase} loss at step {step}.')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f'Nonfinite {phase} gradients at step {step}.')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        processed += len(batch_indices)
        losses.append(float(loss.detach()))
        if (step + 1) % args.log_every == 0 or step == 0:
            record = dict(phase=phase,step=step + 1,loss=float(np.mean(losses[-args.log_every:])),
                elapsed_seconds=time.monotonic() - begin,requests=processed)
            print(json.dumps(record),flush=True)
            save_json(out / 'progress.json',record)
        if time.monotonic() - begin >= args.max_train_seconds:
            print(json.dumps(dict(phase=phase,stop='development_time_budget')),flush=True)
            break
    elapsed = time.monotonic() - begin
    summary = dict(phase=phase,selected_requests=len(indices),known_target_requests=len(known),
        optimizer_steps=len(losses),processed_requests=processed,seconds=elapsed,
        requests_per_second=processed/elapsed,first_loss=float(np.mean(losses[:min(20,len(losses))])),
        final_loss=float(np.mean(losses[-min(20,len(losses)): ])),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if args.device.startswith('cuda') else None)
    save_json(out / f'{phase}_training.json',summary)
    return summary


@torch.no_grad()
def evaluate(model, dataset, indices, args, phase, out, sampled_indices=None):
    """Free retrieval on indices; optional sampled diagnostics on a subset."""
    indices = np.asarray(indices, dtype=np.int64)
    selected_sampled = indices if sampled_indices is None else np.asarray(sampled_indices, dtype=np.int64)
    if len(np.unique(selected_sampled)) != len(selected_sampled) or not np.isin(selected_sampled, indices).all():
        raise ValueError('Sampled diagnostic indices must be a unique subset of the full evaluation panel.')
    sampled_set = set(map(int, selected_sampled))
    model.eval()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    rows = []
    baseline_rows = []
    sampled = {}
    random_rows = []
    sampled_random = {}
    uids = []
    days = []
    sampled_order, sampled_uids, sampled_days, sampled_counts = [], [], [], []
    sampled_pool_sizes = {}
    raw_rankings = []
    start = time.monotonic()
    # Deliberately keep variable-length inference simple. Generation shares
    # a request's historical cache across its category branches in the model.
    for number, index in enumerate(indices):
        batch = dataset.batch([index], args.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=args.device.startswith('cuda') and args.eval_precision == 'bf16'):
            if args.decoder == 'exact':
                predicted, _ = model.generate_exact_topk(**batch,k=100,branch_batch_size=32)
            else:
                predicted, _ = model.generate_topk(**batch,k=100,beam_width=args.beam_width)
        mapped = predicted[0].detach().cpu().numpy()
        mapped = mapped[(mapped > 0) & (mapped < dataset.oov)]
        ranking = dataset.raw_ids[mapped - 1].tolist()
        positives, _ = dataset.targets(index)
        row = request_metrics(ranking, positives, dataset.catalog_set)
        rows.append(row)
        random_rows.append(random_expected_metrics(
            int(row['positives']), int(row['known_positives']), len(dataset.raw_ids)))
        baseline_rows.append(request_metrics(dataset.popular_ranking, positives, dataset.catalog_set))
        request = dataset.prepared.requests[index]
        uids.append(int(request['uid']))
        days.append(int(request['day']))
        raw_rankings.append(dict(index=int(index),uid=int(request['uid']),request_id=int(request['request_id']),
            positive_ids=positives.tolist(),ranked_ids=ranking))
        if int(index) in sampled_set and args.sampled_distractors:
            sampled_order.append(int(index))
            sampled_uids.append(int(request['uid']))
            sampled_days.append(int(request['day']))
            sampled_counts.append(row)
        for count in args.sampled_distractors if int(index) in sampled_set else ():
            for strategy in ('uniform', 'popularity'):
                key = f'{strategy}_{count}'
                candidate_raw = sample_candidates(dataset.raw_ids, positives, count, args.candidate_seed,
                    int(request['request_id']),dataset.popularity if strategy=='popularity' else None)
                candidate_local = np.searchsorted(dataset.raw_ids, candidate_raw) + 1
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,
                                    enabled=args.device.startswith('cuda') and args.eval_precision == 'bf16'):
                    scores = model.score_items(**batch,candidate_ids=torch.tensor(candidate_local[None,:],device=args.device))
                # Positive injection must not also give positives priority on
                # equal scores (BF16 ties are common). Raw ID is label-free.
                order = np.lexsort((candidate_raw, -scores[0].float().cpu().numpy()))
                sampled.setdefault(key,[]).append(request_metrics(candidate_raw[order].tolist(),positives,dataset.catalog_set))
                sampled_random.setdefault(key,[]).append(random_expected_metrics(
                    int(row['positives']), int(row['known_positives']), len(candidate_raw)))
                sampled_pool_sizes.setdefault(key, []).append(len(candidate_raw))
        if (number + 1) % 25 == 0:
            print(json.dumps(dict(phase=phase,evaluated=number+1,total=len(indices),seconds=time.monotonic()-start)),flush=True)
    summary = dict(phase=phase,full_catalog=aggregate_metrics(rows,uids),
        initial_popularity=aggregate_metrics(baseline_rows,uids),
        sampled_candidate_diagnostics={key:aggregate_metrics(value,sampled_uids) for key,value in sampled.items()},
        random_expected=dict(full_catalog=aggregate_metrics(random_rows,uids),
            sampled_candidate_diagnostics={key:aggregate_metrics(value,sampled_uids) for key,value in sampled_random.items()}),
        random_definition='Uniform video permutation of the same allowed catalog or sampled pool; OOV positives remain in denominators.',
        seconds=time.monotonic()-start,beam_width=args.beam_width,
        retrieval_scope=(f'Exact bound-pruned search over the configured {len(dataset.raw_ids)}-item catalog.'
                         if args.decoder == 'exact' else
                         f'Free beam search over the configured {len(dataset.raw_ids)}-item catalog; not exhaustive top-k.'),
        evaluation_precision=args.eval_precision,
        request_selection='Fixed development cohort; stable-hash positive-request panel, including all-OOV requests.',
        candidate_seed=args.candidate_seed,sampled_tie_break='raw_item_id ascending',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if args.device.startswith('cuda') else None)
    sampled_order = np.asarray(sampled_order, dtype=np.int64)
    summary['full_request_panel'] = dict(requests=len(indices),
        indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    summary['sampled_request_panel'] = dict(requests=len(sampled_order),
        indices_sha256=hashlib.sha256(sampled_order.tobytes()).hexdigest(),
        selection='Full panel' if sampled_indices is None else 'Supplied fixed subset of the full panel')
    # Each daily and pooled diagnostic uses its own request/user denominator.
    summary['per_day'] = {}
    for day in sorted(set(days)):
        full_rows = [i for i, value in enumerate(days) if value == day]
        sample_rows = [i for i, value in enumerate(sampled_days) if value == day]
        full_uids = [uids[i] for i in full_rows]
        day_sample_uids = [sampled_uids[i] for i in sample_rows]
        summary['per_day'][str(day)] = dict(
            full_catalog=aggregate_metrics([rows[i] for i in full_rows], full_uids),
            initial_popularity=aggregate_metrics([baseline_rows[i] for i in full_rows], full_uids),
            sampled_candidate_diagnostics={key:aggregate_metrics([value[i] for i in sample_rows],day_sample_uids)
                for key,value in sampled.items()},
            random_expected=dict(
                full_catalog=aggregate_metrics([random_rows[i] for i in full_rows],full_uids),
                sampled_candidate_diagnostics={key:aggregate_metrics([value[i] for i in sample_rows],day_sample_uids)
                    for key,value in sampled_random.items()}))
    np.savez_compressed(out / f'{phase}_request_metrics.npz',indices=indices,uids=np.array(uids),
        **{key:np.array([r[key] for r in rows]) for key in rows[0]} if rows else {})
    if args.sampled_distractors:
        np.savez_compressed(out / f'{phase}_sampled_request_metrics.npz',
            indices=sampled_order,uids=np.asarray(sampled_uids,dtype=np.int64),
            positives=np.asarray([r['positives'] for r in sampled_counts]),
            known_positives=np.asarray([r['known_positives'] for r in sampled_counts]),
            **{f'{key}__{metric}':np.asarray([r[metric] for r in value])
                for key,value in sampled.items() for metric in value[0] if '@' in metric},
            **{f'{key}__candidate_count':np.asarray(value,dtype=np.int64)
                for key,value in sampled_pool_sizes.items()})
    # This compact raw ranking panel is local runtime evidence, not tracked.
    (out / f'{phase}_rankings.json').write_text(json.dumps(raw_rankings)+'\n')
    save_json(out / f'{phase}_evaluation.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/recflow_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--candidate-seed',type=int,default=17,help='Fixed independently of the training repeat seed')
    p.add_argument('--catalog-size',type=int,default=50000,help='0 means complete initial catalog')
    p.add_argument('--cohort-users',type=int,default=256)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--heads',type=int,default=3)
    p.add_argument('--train-requests',type=int,default=20000)
    p.add_argument('--eval-requests',type=int,default=128)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--current-steps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=0.001)
    p.add_argument('--beam-width',type=int,default=100)
    p.add_argument('--decoder',choices=('beam','exact'),default='beam')
    p.add_argument('--eval-precision',choices=('fp32','bf16'),default='fp32')
    p.add_argument('--history-categories',action='store_true')
    p.add_argument('--max-train-seconds',type=float,default=600)
    p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--sampled-distractors',type=int,nargs='*',default=[])
    p.add_argument('--run-current',action='store_true')
    p.add_argument('--skip-day19',action='store_true',help='Skip base-window evaluation in a repeated update probe')
    p.add_argument('--save-checkpoint',action='store_true')
    p.add_argument('--evaluate-checkpoint',type=Path,help='Evaluate a saved development model without retraining')
    p.add_argument('--eval-day',type=int,default=19,help='Day for checkpoint-only evaluation')
    p.add_argument('--long-contract',type=Path,help='Prospective population-development settings; explicit launch is still required')
    p.add_argument('--check-contract',action='store_true',help='Validate prospective settings and exit before any training')
    args=p.parse_args()
    if args.long_contract:
        contract=json.loads(args.long_contract.read_text())
        for name, expected in contract['arguments'].items():
            actual=getattr(args,name)
            if isinstance(actual,Path):
                actual=str(actual)
            if actual != expected:
                p.error(f'Prospective setting differs: {name}={actual!r}, expected {expected!r}.')
        for name, expected in contract['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                p.error(f'Prospective source changed: {name}.')
        if hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest() != contract['data_manifest_sha256']:
            p.error('Prepared data manifest differs from the prospective setting.')
        if args.check_contract:
            print(json.dumps(dict(contract=str(args.long_contract),settings_match=True,
                scope=contract['scope'],resource_estimate=contract['resource_estimate'],
                training_launched=False),indent=2))
            return
    elif args.check_contract:
        p.error('--check-contract requires --long-contract.')
    if args.max_train_seconds > 1200 and not args.long_contract:
        p.error('This development entry point caps each phase at 20 minutes; prepare a separate long-job contract for larger execution.')
    if args.decoder == 'exact' and args.eval_precision != 'fp32':
        p.error('Use FP32 for exact ranking; BF16 branch batch shapes can change close scores.')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():
        p.error('Use a new output directory to retain previous outcomes.')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    prepared=PreparedRecFlow(args.data)
    dataset=ProbeData(prepared,args.catalog_size,args.cohort_users,args.context)
    cfg=HSTUConfig(num_items=dataset.oov,num_behaviors=4,num_prediction_items=len(dataset.raw_ids),
        hidden_size=args.hidden,num_layers=args.layers,num_heads=args.heads,max_seq_len=args.context+3,
        block_variant='hstu_reference',activation='silu',gating='silu_gate',relative_position_bias=True,
        input_dropout=0.0,attn_dropout=0.0)
    model=RecFlowGenerator(cfg,torch.tensor(dataset.paths),history_categories=args.history_categories).to(args.device)
    parameters=sum(v.numel() for v in model.parameters())
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(model=dataclasses.asdict(cfg),parameters=parameters,
        scope='Development-only; no formal model admission or final-user evaluation.',
        catalog_sha256=hashlib.sha256(dataset.raw_ids.tobytes()).hexdigest(),
        cohort_sha256=hashlib.sha256(np.asarray(dataset.uids).tobytes()).hexdigest(),
        cohort_size=len(dataset.uids),catalog_items=len(dataset.raw_ids),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT/'src/hstu_kvcache/recflow/model.py',
                         ROOT/'src/hstu_kvcache/recflow/data.py',ROOT/'src/hstu_kvcache/recflow/metrics.py')},
        torch_version=str(torch.__version__))
    if args.evaluate_checkpoint:
        # This option consumes this runner's own locally generated checkpoints.
        saved=torch.load(args.evaluate_checkpoint,map_location=args.device,weights_only=False)
        previous=saved['configuration']
        if (previous['model'] != config['model'] or previous['catalog_sha256'] != config['catalog_sha256']
                or previous.get('history_categories',False) != args.history_categories):
            p.error('Checkpoint architecture and catalog must match this evaluation.')
        model.load_state_dict(saved['model'])
        config['checkpoint_sha256']=hashlib.sha256(args.evaluate_checkpoint.read_bytes()).hexdigest()
        config['checkpoint_training_configuration']=previous
    save_json(args.output/'configuration.json',config)
    print(json.dumps(config),flush=True)
    evaluation_day=args.eval_day if args.evaluate_checkpoint else 19
    evaluation=dataset.indices(evaluation_day,evaluation_day,args.eval_requests)
    result=dict(configuration=config)
    if args.evaluate_checkpoint:
        if args.run_current:
            p.error('A saved development checkpoint has no optimizer; evaluation only.')
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
        train=dataset.indices(1,18,args.train_requests)
        result['parent_train']=train_phase(model,dataset,train,optimizer,args,'parent',args.output)
    if args.save_checkpoint and args.run_current:
        torch.save(dict(model=model.state_dict(),configuration=config,phase='parent'),args.output/'parent_checkpoint.pt')
    if not args.skip_day19:
        phase=f'checkpoint_day{evaluation_day}' if args.evaluate_checkpoint else 'parent_day19'
        result[phase]=evaluate(model,dataset,evaluation,args,phase,args.output)
    if args.run_current:
        future=dataset.indices(22,22,args.eval_requests)
        result['parent_day22']=evaluate(model,dataset,future,args,'parent_day22',args.output)
        incremental=dataset.indices(19,21,args.train_requests)
        result['current_train']=train_phase(model,dataset,incremental,optimizer,args,'current',args.output)
        result['current_day22']=evaluate(model,dataset,future,args,'current_day22',args.output)
    if args.save_checkpoint:
        torch.save(dict(model=model.state_dict(),configuration=config,phase='current' if args.run_current else 'parent'),
                   args.output/'development_checkpoint.pt')
    save_json(args.output/'summary.json',result)


if __name__=='__main__':
    main()

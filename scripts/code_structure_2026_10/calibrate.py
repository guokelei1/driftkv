#!/usr/bin/env python3
"""Replace joint16 after the retained local32, without fitting foundation models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import torch
from code_structure_2026_10.structure import StructuredKVAdapter, StructuredKVRefinement
from design_system.calibrate_lightweight_fullitem import save_adapter
from design_system.calibrate_lightweight_fullitem_joint import fit_joint
from design_system.calibrate_lightweight_decoder import prepare_rows
from design_two.capture import capture_rows
from design_two.common import load_models, setup_device
from design_two.prepare import PANEL, load_history, load_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from unified_reuse_2026_09.common import sha256, write_json

ORIGINAL = ROOT / 'results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration'


def run(args):
    began = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    panel, groups, _ = load_panel(args.panel)
    directory = ORIGINAL / args.target / 'read_only'
    old = json.loads((directory / 'calibration.json').read_text())
    if old['fit_uids'] != groups['fit'] or old['validation_uids'] != groups['validation']:
        raise RuntimeError('retained local calibration roles differ')
    train, validation = list(groups['fit']), list(groups['validation'])
    if args.small:
        train, validation = train[:4], validation[:2]
    if (set(train) | set(validation)) & set(groups['evaluation']):
        raise RuntimeError('calibration overlaps evaluation')
    epochs = 4 if args.small else 16
    checks = [0, 1, 2, 4] if args.small else [0, 8, 16]
    source_paths = [Path(__file__), Path(__file__).with_name('structure.py'),
        ROOT / 'scripts/design_system/calibrate_lightweight_fullitem_joint.py']
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    write_json(args.output / 'configuration.json', dict(target=args.target, layout=args.layout,
        fit_uids=train, validation_uids=validation, seed=17, epochs=epochs, checks=checks,
        batch_size=8, learning_rate=.001, identity_weight=.001,
        origin=str(directory), origin_sha256=sha256(directory / 'calibration.pt'),
        panel_sha256=sha256(args.panel / 'binding.json'), execution_sources=sources,
        role='development_canary' if args.small else 'development_calibration',
        native_writes='unchanged', storage_floats_per_layer_row=48,
        lineage='retained local32 plus replacement joint16; original joint16 is not used',
        cost_scope='inherited local32 plus replacement joint plus complete serving; source reconstruction separate'))
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device, int(args.target[1:]))
    captured, resource, records = capture_rows(models,
        load_history(args.panel, requested_uids=train + validation), train + validation,
        args.target, panel['known_vocab_size'])
    model = models[args.target]
    del models
    payload = torch.load(directory / 'calibration.pt', map_location='cpu', weights_only=False)
    original = LightweightFullItemKVAdapter(**payload['config']).to(device).eval()
    original.load_state_dict(payload['state_dict'])
    original.requires_grad_(False)
    adapter = StructuredKVAdapter.from_adapter(original, args.layout).eval().requires_grad_(False)
    costs = defaultdict(int)
    costs['structure_conversion_flops'] += 3 * sum(x.input_weight.numel() for x in original.layers)
    rows, lookup_bytes = prepare_rows(model, adapter, captured, train + validation, device, costs)
    for uid, row in rows.items():
        row['item_features'] = captured[uid]['item_features'].to(device)
    refiner = StructuredKVRefinement(adapter).to(device)
    size = sum(p.numel() for p in refiner.parameters())
    costs['joint_coordinate_statistics_flops'] += 4 * size
    probe = rows[validation[0]]
    cache = probe['cache']
    original_k, original_v = cache.k.clone(), cache.v.clone()

    @torch.no_grad()
    def check(refined):
        state = (refiner.encode_cache(adapter, cache, probe['item_features']) if refined else
            adapter.encode_cache(cache, probe['item_features']))
        read = (refiner.make_history_read(adapter, state) if refined else adapter.make_history_read(state))
        actual = score(model, cache, probe['candidates'], probe['query_delta'], history_read_fn=read)[0]
        from read_correction_2026_09.cost import CostModel, eager_read
        extra = adapter.estimate_flops(tokens=cache.seq_len, candidates=probe['candidates'].shape[1])
        costs['execution_check_encoder_flops'] += extra['token_transform']
        costs['execution_check_query_flops'] += extra['extra_history_read'] + eager_read(
            CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, 'torch'),
            cache.seq_len, queries=probe['candidates'].shape[1])
        if refined:
            costs['execution_check_coordinates_flops'] += 2 * size
        return actual, state.codes

    origin = check(False)
    zero = check(True)
    identity_error = max(float((a - b).abs().max()) for a, b in zip(origin, zero))
    if identity_error > 2e-5:
        raise RuntimeError('zero refinement changed the structured origin')
    diagnostics = fit_joint(model, adapter, refiner, rows, train, validation,
        epochs=epochs, batch_size=8, checks=checks, costs=costs)
    centering = refiner.coordinate_centering_forward_flops
    train_calls = costs['joint_coordinates_forward_backward_flops'] // (3 * size)
    validation_calls = costs['joint_validation_coordinates_flops'] // (2 * size)
    costs['joint_coordinate_centering_flops'] += centering * (3 * train_calls + validation_calls + 3)
    before = check(True)
    refiner.fold_into(adapter)
    costs['publication_joint_fold_flops'] += 2 * size
    after = check(False)
    fold_error = max(float((a - b).abs().max()) for a, b in zip(before, after))
    costs['first_gradient_diagnostic_flops'] += 2 * size + len(tuple(refiner.parameters()))
    costs['execution_check_comparison_flops'] += 6 * (origin[0].numel() + origin[1].numel()) + 8
    if fold_error > 2e-5 or not torch.equal(cache.k, original_k) or not torch.equal(cache.v, original_v):
        raise RuntimeError('publication changed function or native state')
    save_adapter(adapter, args.output / 'calibration.pt')
    local = old['cost']['components']
    components = defaultdict(int, local)
    for key, value in costs.items():
        components[key] += value
    write_json(args.output / 'calibration.json', dict(status='complete', target=args.target,
        kind=adapter.kind, config=adapter.get_config(), fit_uids=train, validation_uids=validation,
        weights_sha256=sha256(args.output / 'calibration.pt'), capture_records=records,
        diagnostics=diagnostics, execution_sources=sources,
        checks=dict(zero_coordinate_max_error=identity_error, publication_max_error=fold_error,
            native_state_unchanged=True, no_foundation_training=True),
        cost=dict(components=dict(components), calibration_flops=sum(components.values()),
            inherited_local_calibration_flops=sum(local.values()), replacement_joint_flops=sum(costs.values()),
            stage_components=dict(inherited_local_calibration=local, replacement_joint=dict(costs)),
            development_resource=dict(replayed_source_trajectory_flops=resource['source_trajectory_flops'],
                repeated_teacher_prefix_capture_flops=resource['teacher_cache_capture_flops']),
            source_reconstruction_sensitivity=dict(
                additional_flops=old['cost']['source_reconstruction_sensitivity']['additional_flops'],
                calibration_if_source_reconstructed=sum(components.values()) +
                    old['cost']['source_reconstruction_sensitivity']['additional_flops']),
            item_lookup_bytes=lookup_bytes,
            convention='Local32 includes the same teacher prefix once; repeated capture is development work. Query/encoder backward charged twice forward. Training never reuses codes across optimizer steps.'),
        elapsed_seconds=time.perf_counter() - began,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30))
    print(json.dumps(dict(status='complete', target=args.target, layout=args.layout,
        selected_epoch=diagnostics['selected_epoch'], calibration_flops=sum(components.values()))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', required=True, choices=['group2', 'shared6_private7', 'shared12_private6'])
    p.add_argument('--target', required=True, choices=['v3', 'v4', 'v5'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel', type=Path, default=PANEL)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--small', action='store_true')
    run(p.parse_args())

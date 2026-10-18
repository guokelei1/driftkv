#!/usr/bin/env python3
"""Fixed validation-user connection interventions; no fitting or request labels."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import torch
from code_structure_2026_10.structure import StructuredKVAdapter
from design_two.capture import capture_rows
from design_two.common import RELEASES, load_models, setup_device
from design_two.prepare import PANEL, load_panel, load_history
from hstu_kvcache.adaptation.reader import score
from unified_reuse_2026_09.common import write_json


@torch.no_grad()
def remove_connections(adapter, *, encoder=False, decoder=False):
    result = copy.deepcopy(adapter)
    for layer in result.layers:
        if decoder:
            for head in range(layer.heads):
                slots = [slot for slot, index in enumerate(layer.access_indices[head].tolist())
                    if layer.owners[index][0] != head]
                layer.decoder[head, slots] = 0
                layer.key_decoder[head, slots] = 0
        if encoder:
            blocks = layer.weight_blocks()
            centers = layer.weight_blocks(layer.weight_coordinate_center)
            for branch, block, center in zip(layer.branches, blocks, centers, strict=True):
                for column in range(branch.code_width):
                    output = branch.code_start + column
                    owner = layer.owners[output][0]
                    for slot, head in enumerate(branch.heads):
                        if head == owner:
                            continue
                        for start in [slot * layer.head_dim, (len(branch.heads) + slot) * layer.head_dim]:
                            rows = slice(start, start + layer.head_dim)
                            layer.input_bias[output] += (center[rows, column] * block[rows, column]).sum()
                            block[rows, column] = 0
    return result


@torch.no_grad()
def run(args):
    panel, groups, _ = load_panel(PANEL)
    users = list(groups['validation'])
    device = setup_device(args.gpu)
    models = load_models(panel, device)
    histories = load_history(PANEL, requested_uids=users)
    result = {}
    for version, _ in RELEASES:
        payload = torch.load(args.calibrations / version / 'calibration.pt', map_location='cpu', weights_only=False)
        adapter = StructuredKVAdapter(**payload['config']).to(device).eval().requires_grad_(False)
        adapter.load_state_dict(payload['state_dict'])
        candidates = dict(full=adapter,
            no_cross_decoder=remove_connections(adapter, decoder=True),
            no_cross_encoder=remove_connections(adapter, encoder=True),
            no_cross_either=remove_connections(adapter, encoder=True, decoder=True))
        captured, costs, _ = capture_rows(models, histories, users, version, panel['known_vocab_size'])
        rows = []
        for uid in users:
            raw = captured[uid]
            cache, teacher = raw['parent'].to(device), raw['teacher'].to(device)
            item = raw['item_features'].to(device)
            ids = raw['candidates'].to(device)[None]
            delta = torch.tensor([raw['query_delta']], device=device)
            wanted = score(models[version], teacher, ids, delta)[0]
            metrics = {}
            for name, candidate in candidates.items():
                state = candidate.encode_cache(cache, item)
                actual = score(models[version], cache, ids, delta,
                    history_read_fn=candidate.make_history_read(state))[0]
                error = actual - wanted
                metrics[name] = dict(logit_mse=float(error.square().mean()),
                    centered_logit_mse=float((error - error.mean(-1, keepdim=True)).square().mean()),
                    mean_logit_error=float(error.mean()))
            rows.append(dict(uid=uid, metrics=metrics))
        result[version] = dict(users=rows, means={name: {
            metric: sum(r['metrics'][name][metric] for r in rows) / len(rows)
            for metric in ['logit_mse', 'centered_logit_mse', 'mean_logit_error']}
            for name in candidates}, repeated_capture_development_flops=costs)
        print(json.dumps(dict(version=version, means=result[version]['means'])), flush=True)
    write_json(args.output, dict(status='complete', validation_uids=users, by_release=result,
        scope='Diagnostic intervention on the same16 validation users used for epoch selection. No fitting, labels, quality claim, or independent confirmation. Complete query is rerun after every intervention.',
        semantics='Owner is the initialization mapping. Encoder cut removes centered cross-native-KV contributions and shifts folded bias; complete Current item embedding remains. Decoder cut removes access to other initialization owners. Both-cut is not a refitted optimal rank8 control.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibrations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    run(p.parse_args())

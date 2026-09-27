#!/usr/bin/env python3
"""Check the selected Max checkpoint on synthetic full-length inputs, without fitting."""
import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_hstu_backends import backend, delta, digest, capture_gradients, gradient_delta
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.backend_info import attention_backend_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-epochs', type=int, default=1)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = HSTUConfig(**saved['config'])
    assert cfg.num_layers == 16 and saved['training_epochs_completed'] == args.expected_epochs
    model = HSTU(cfg)
    model.load_state_dict(saved['model'], strict=True)
    model.cuda().eval()
    # Exercise the loaded 16-layer backbone and scoring head; embedding lookup is unchanged.
    x = torch.randn(2, 1024, cfg.hidden_size, device='cuda') * .1
    target = torch.tensor([0., 1.], device='cuda')
    kernel = importlib.import_module('hstu_kvcache.models.triton_attention')
    report = {'checkpoint_sha256': digest(args.checkpoint), 'strict_load': True,
              'scope': 'synthetic embedded inputs; loaded Max backbone/head; dropout disabled; no held-out labels or optimizer updates',
              'attention_execution': attention_backend_info(), 'precisions': {}}
    for dtype, tolerance in [(torch.float32, 1e-4), (torch.bfloat16, .03)]:
        model.to(dtype=dtype)
        snapshots = []
        for name in ['torch', 'triton']:
            backend(model, name)
            model.zero_grad(set_to_none=True)
            inputs = x.to(dtype).detach().requires_grad_(True)
            with patch.object(kernel, 'triton_attention', wraps=kernel.triton_attention) as calls:
                hidden, _ = model.forward_embedded(inputs)
                logits = model.cc_score_head(hidden[:, -1]).flatten().float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                call_count = calls.call_count
            snapshots.append({'hidden': hidden.detach().cpu(), 'logits': logits.detach().cpu(),
                              'loss': loss.detach().cpu(), 'input_gradient': inputs.grad.cpu(),
                              'gradients': capture_gradients(model), 'triton_calls': call_count})
        a, b = snapshots
        comparisons = {k: delta(a[k], b[k]) for k in ['hidden', 'logits', 'loss', 'input_gradient']}
        comparisons['gradients'] = gradient_delta(a['gradients'], b['gradients'])
        comparisons['triton_calls'] = b['triton_calls']
        comparisons['relative_l2_tolerance'] = tolerance
        comparisons['passed'] = (b['triton_calls'] == 16 and
            all(comparisons[k]['finite'] and comparisons[k]['relative_l2'] < tolerance
                for k in ['hidden', 'logits', 'loss', 'input_gradient']) and
            comparisons['gradients']['finite'] and comparisons['gradients']['worst_relative_l2'] < tolerance)
        report['precisions'][str(dtype)] = comparisons
        del snapshots, a, b, hidden, logits, loss, inputs
    report['passed'] = all(v['passed'] for v in report['precisions'].values())
    report['script_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {m: v for m, v in p.items() if m != 'gradients'}
                      for k, p in report['precisions'].items()}, indent=2), flush=True)
    assert report['passed'], 'Max backend parity check failed; retain report and stop'


if __name__ == '__main__':
    main()

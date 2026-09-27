"""Layer-sized translation fitting and dynamic-depth snapshot evaluation."""

from __future__ import annotations

import time

import numpy as np
import torch

from design import bias_read_probe
from design.run_shared_read_probe import device_parameters
from design.run_unified_auc import TAILS, MAP_KS, take_events
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.baselines import layer_recompute as lr
from hstu_kvcache.baselines.kv_translate import KVTranslator, fit_affine_ridge
from hstu_kvcache.baselines.kv_translate.core import _features
from hstu_kvcache.baselines.tail_recompute import recompute_tail


@torch.no_grad()
def fit_translators_streamed(source, target, num_heads, device, ks=(1, 2, 3, 4), ridge=.01):
    """Same selection and ridge formulas; keep the complete paired KV on CPU.

    The original selector materializes both entire K/V pairs in FP64. Here its
    identical (K/V, target layer, source layer, head) loop moves only the active
    layer pair. Final full-width maps still call the original affine fitter.
    """
    assert source.k.device.type == target.k.device.type == "cpu"
    assert source.k.shape == source.v.shape == target.k.shape == target.v.shape
    layers, _, _, width = source.k.shape
    head_width = width // num_heads
    scores = torch.zeros(layers, layers, dtype=torch.float64, device=device)
    for kind, (source_values, target_values) in enumerate(((source.k, target.k), (source.v, target.v))):
        print(f"Translate: rank {'K' if kind == 0 else 'V'} source layers", flush=True)
        for target_layer in range(layers):
            y = target_values[target_layer].flatten(0, 1).to(device=device, dtype=torch.float64)
            for source_layer in range(layers):
                x = source_values[source_layer].flatten(0, 1).to(device=device, dtype=torch.float64)
                for head in range(num_heads):
                    columns = slice(head*head_width, (head+1)*head_width)
                    weight, bias = fit_affine_ridge(x[:, columns], y[:, columns], ridge=0)
                    truth = y[:, columns]
                    prediction = x[:, columns] @ weight + bias
                    total = (truth-truth.mean(0)).square().sum()
                    residual = (truth-prediction).square().sum()
                    r2 = torch.where(total > 0, 1-residual/total.clamp_min(torch.finfo(total.dtype).tiny),
                                     torch.zeros_like(total))
                    scores[target_layer, source_layer] += r2/(2*num_heads)
                del x, weight, bias, truth, prediction, total, residual, r2
            del y
    ranking = scores.argsort(dim=1, descending=True, stable=True)[:, :max(ks)].cpu()
    scores = scores.cpu()
    fitted = {}
    for k in ks:
        print(f"Translate: fit full-width maps with {k} source layers", flush=True)
        selected = ranking[:, :k].clone()
        parameters = []
        for source_values, target_values in ((source.k, target.k), (source.v, target.v)):
            weights, biases = [], []
            for layer, source_layers in enumerate(selected):
                x = _features(source_values, source_layers).flatten(0, 1).to(device)
                y = target_values[layer].flatten(0, 1).to(device)
                weight, bias = fit_affine_ridge(x, y, ridge)
                weights.append(weight.cpu())
                biases.append(bias.cpu())
                del x, y, weight, bias
            parameters.extend((torch.stack(weights), torch.stack(biases)))
        fitted[k] = KVTranslator(selected, *parameters, scores, "in_sample")
    return fitted


def compare_translators(actual, reference):
    differences = {}
    for k in actual:
        assert torch.equal(actual[k].source_layers.cpu(), reference[k].source_layers.cpu())
        for name in ("selection_scores", "k_weight", "k_bias", "v_weight", "v_bias"):
            first, second = getattr(actual[k], name).cpu(), getattr(reference[k], name).cpu()
            torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
            differences[f"{k}/{name}"] = float((first-second).abs().max())
    return dict(status="passed", maximum_parameter_difference=max(differences.values()),
                per_parameter_difference=differences, selected_layers_identical=True)


@torch.inference_mode()
def evaluate(parent, current, arrays, uids, rows, artifact, translators, batch_size, chunk, device):
    """The original unified evaluator with layer counts from the actual model."""
    layers = len(current.blocks)
    paths = ["parent", "reuse", "exact", *[f"droid_{k}" for k in range(1, layers+1)],
             *[f"tail_{n}" for n in TAILS], *[f"translate_{k}" for k in MAP_KS], *artifact["rules"]]
    values = {name: np.full(len(rows), np.nan, dtype=np.float32) for name in paths}
    by_uid = {int(uid): frame.index.to_numpy() for uid, frame in rows.groupby("uid", sort=False)}
    rules = {name: device_parameters(parameters, device) for name, parameters in artifact["rules"].items()}
    checks, counts = [], np.zeros(len(uids), dtype=np.int64)
    began = time.perf_counter()
    for begin in range(0, len(uids), batch_size):
        end = min(begin+batch_size, len(uids))
        events = take_events(arrays, begin, end, device)
        state, exact = lr.capture_state(parent, *events), current.compute_kv(*events)
        groups = [by_uid.get(int(uid), np.empty(0, dtype=np.int64)) for uid in uids[begin:end]]
        counts[begin:end] = [len(group) for group in groups]
        lengths = torch.full((end-begin,), float(state.cache.seq_len), device=device)
        maximum, chunks = max(map(len, groups), default=0), []
        for offset in range(0, maximum, chunk):
            width = min(chunk, maximum-offset)
            item = np.ones((end-begin, width), dtype=np.int64)
            delta = np.ones((end-begin, width), dtype=np.float32)
            indices = np.full((end-begin, width), -1, dtype=np.int64)
            for local, group in enumerate(groups):
                take = group[offset:offset+width]
                indices[local, :len(take)] = take
                if len(take):
                    item[local, :len(take)] = rows.loc[take, "item_idx"].to_numpy(dtype=np.int64)
                    delta[local, :len(take)] = rows.loc[take, "query_timestamp"].to_numpy()-arrays[0][begin+local, -1]
            chunks.append((torch.as_tensor(item, device=device), torch.as_tensor(delta, device=device), indices))

        def observe(name, model, cache, parameters=None):
            override = bias_read_probe.make_history_override(parameters, lengths) if parameters is not None else None
            for panel, delta, indices in chunks:
                prediction, _ = score(model, cache, panel, delta, history_override=override)
                mask = indices >= 0
                values[name][indices[mask]] = prediction.cpu().numpy()[mask]

        observe("parent", parent, state.cache)
        observe("reuse", current, state.cache)
        observe("exact", current, exact)
        for count, interval in artifact["intervals"].items():
            migrated = lr.recompute_interval(current, state, *events, tuple(interval))
            if begin == 0 and int(count) == layers:
                torch.testing.assert_close(migrated.cache.k, exact.k, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(migrated.cache.v, exact.v, atol=2e-5, rtol=2e-5)
                checks.append("full_interval_matches_exact_KV")
            observe(f"droid_{count}", current, migrated.cache)
            del migrated
        for n in TAILS:
            migrated = recompute_tail(current, state.cache, *events, n)
            if begin == 0 and n == state.cache.seq_len:
                torch.testing.assert_close(migrated.k, exact.k, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(migrated.v, exact.v, atol=2e-5, rtol=2e-5)
                checks.append("full_tail_matches_exact_KV")
            observe(f"tail_{n}", current, migrated)
            del migrated
        for k, translator in translators.items():
            migrated = translator.apply(state.cache)
            observe(f"translate_{k}", current, migrated)
            del migrated
        for name, parameters in rules.items():
            observe(name, current, state.cache, parameters)
        del state, exact, events, lengths, chunks
        if end % 256 < batch_size or end == len(uids):
            print(f"  {end}/{len(uids)} snapshots; {time.perf_counter()-began:.1f}s", flush=True)
    assert sum(counts) == len(rows) and all(np.isfinite(value).all() for value in values.values())
    last = {int(uid): int(stamps[-1]) for uid, stamps in zip(uids, arrays[0], strict=True)}
    raw = rows.assign(prefix_last_timestamp=rows.uid.map(last), prefix_length=1024, **values)
    assert (raw.prefix_last_timestamp < raw.query_timestamp).all()
    return raw, dict(checks=checks, snapshot_users=len(uids), feedback_users=len(by_uid),
        no_feedback_users=len(uids)-len(by_uid), requests=len(rows),
        per_user_requests=counts.tolist(), elapsed_seconds=time.perf_counter()-began)

#!/usr/bin/env python3
"""One causal release rule: calibrate layer strength, then select a translator.

The two existing source/time designs are fitted on the same 64 users. Separate
pre-release users set layer strength and compare actual corrected forwards.
No quality labels, future requests, model-ID branches or mixed logits are used.
"""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from design.data import DAY, ROOT, diagnostic_admissions, fixed_split, frozen_model, histories
from design.run import (
    initialize_calibration,
    make_scenes,
    new_translator,
    replay_calibration,
    timed,
    training_state,
    write_json,
)
from insight_two.common import CUTOVER_DAYS

from hstu_kvcache.adaptation import inference
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def scale_layers(translator, strength):
    """Use one scale after each layer map, preserving FP32 cancellation."""
    translator.response_strength.mul_(strength)


@torch.no_grad()
def choose(current, scenes, variants, strength_uids, batch_size):
    anchor = next(iter(variants.values()))
    states = [training_state(s, current, anchor) for s in scenes]
    sources = [state.pack_source(anchor.target) for state in states]
    features = torch.stack([anchor.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    group_counts = Counter(s.uid for s in scenes)
    weights = features.new_tensor([1 / group_counts[s.uid] for s in scenes])
    strength_mask = torch.tensor([s.uid in strength_uids for s in scenes], device=features.device)
    candidates_per_time = scenes[0].candidates.shape[1] // 4
    phi = torch.stack([current.temporal_enc.features(s.query_delta[:, ::candidates_per_time])[0] for s in scenes])
    groups = defaultdict(list)
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
    batches = [indices[start:start+batch_size] for indices in groups.values()
               for start in range(0, len(indices), batch_size)]

    def inputs(batch):
        source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                            torch.cat([states[i].cache.v for i in batch], 1), states[batch[0]].cache.seq_len)
        teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                             torch.cat([scenes[i].teacher.v for i in batch], 1), source.seq_len)
        candidates = torch.cat([scenes[i].candidates for i in batch])
        delta = torch.cat([scenes[i].query_delta for i in batch])
        return source, teacher, candidates, delta

    rates = features.new_zeros(len(scenes), 4, anchor.layers, anchor.width)
    exact = features.new_zeros(len(scenes), scenes[0].candidates.shape[1])
    reuse = torch.zeros_like(exact)
    for batch in batches:
        source, teacher, candidates, delta = inputs(batch)
        rates[batch] = inference.calibration_rates(current, source, teacher, candidates, delta, counts[batch],
            features.new_zeros(len(batch), anchor.layers, anchor.width),
            features.new_zeros(len(batch), anchor.layers, anchor.time_dim, anchor.width), 4)
        exact[batch] = score(current, teacher, candidates, delta)[0]
        reuse[batch] = score(current, source, candidates, delta)[0]
    # All scenes belonging to a UID stay together; lifetime users do not receive
    # more total weight merely because they provide more snapshots.
    fitting_weight = weights * strength_mask
    layer_strength = {}
    for name, translator in variants.items():
        joint = translator.with_time(features[:, None].expand(-1, 4, -1), phi)
        predicted = translator.rates(joint)
        weighted = fitting_weight[:, None, None, None] * counts[:, None, None, None].square()
        numerator = (weighted * predicted * rates).sum((0, 1, 3))
        denominator = (weighted * predicted.square()).sum((0, 1, 3)).clamp_min(1e-20)
        strength = (numerator / denominator).clamp(0, 1)
        before = predicted.clone()
        scale_layers(translator, strength)
        torch.testing.assert_close(translator.rates(joint), before * strength[None, None, :, None],
                                   atol=2e-7, rtol=2e-5)
        layer_strength[name] = strength.cpu().tolist()

    predictions = {"noop": reuse}
    for name, translator in variants.items():
        values = torch.zeros_like(exact)
        for batch in batches:
            source, _, candidates, delta = inputs(batch)
            values[batch] = score(current, source, candidates, delta,
                response_delta=translator.rates(features[batch]) * counts[batch, None, None],
                response_time_delta=translator.time_coefficients(features[batch]) * counts[batch, None, None, None])[0]
        predictions[name] = values
    losses, per_user = {}, {}
    for name, values in predictions.items():
        errors = (values - exact).square().mean(1)
        uids = sorted({s.uid for s in scenes if s.uid not in strength_uids})
        rows = [float(errors[[i for i, s in enumerate(scenes) if s.uid == uid]].mean()) for uid in uids]
        per_user[name] = dict(zip(map(str, uids), rows, strict=True))
        losses[name] = float(np.mean(rows))
    selected = min(losses, key=losses.get)  # deterministic: No-op, stable, source/time in a tie
    choice = copy.deepcopy(variants["stable"] if selected == "noop" else variants[selected])
    if selected == "noop":
        scale_layers(choice, features.new_zeros(anchor.layers))
    record = dict(selected=selected, layer_strength=layer_strength, validation_logit_mse=losses,
        per_user_validation_logit_mse=per_user,
        reference_drift=dict(native_to_exact_logit_rms=float((reuse-exact).square().mean().sqrt()),
            signed_probability_shift=float((exact.sigmoid()-reuse.sigmoid()).mean()),
            response_rms_per_layer=(rates*counts[:, None, None, None]).square().mean((0, 1, 3)).sqrt().cpu().tolist()),
        scope="pre-release label-free independent calibration users; actual nonlinear corrected forwards after response scaling")
    return choice, record


def main(cli):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    references = {"stable": "v6_refit64_01", "source_time": "v7_source_time64_01"}
    configs = {k: json.loads((ROOT / "results/design" / run / "configuration.json").read_text())
               for k, run in references.items()}
    fitted = configs["stable"]["calibration_uids"]
    assert fitted == configs["source_time"]["calibration_uids"]
    split = fixed_split()
    uids = split["calibration"][64:64+cli.users]
    assert len(uids) == cli.users and not set(uids) & set(fitted)
    args = SimpleNamespace(**configs["stable"])
    args.lifetime_users = min(16, cli.users // 4)
    # Both folds include actual lifetime scenes in a deterministic interleave.
    strength_uids = set(uids[::2])
    args.score_steps = 0
    args.targets = cli.targets
    config = dict(configs["stable"], run_id=cli.run_id, targets=cli.targets,
        calibration_uids=fitted+uids, development_uids=[], trajectory_uids=[],
        selection_uids=uids, strength_uids=sorted(strength_uids),
        design_validation_uids=[u for u in uids if u not in strength_uids],
        references=references, target_options={}, calibration_reference=None,
        selection_rule="fit six response strengths in [0,1] on one UID fold; choose No-op/stable/source-time by minimum teacher logit MSE on the other fold",
        selection_causality="all prefixes, candidates and teacher queries at or before each cutover; no feedback labels",
        estimate_seconds=[90, 300], expected_gpu_mib=18000, confirmation_read=False,
        admission_inputs=diagnostic_admissions(cli.targets),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "scripts/design/run.py", *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]})
    write_json(out / "configuration.json", config)
    start = time.perf_counter()
    ledger = defaultdict(float)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        history = timed(lambda: histories(uids, CUTOVER_DAYS[cli.targets-1]+1), ledger, "history_io")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, args, ledger)
        decisions = []
        for target in range(1, cli.targets+1):
            cutover = CUTOVER_DAYS[target-1]*DAY
            current = timed(lambda target=target: frozen_model(target, device), ledger, "model_io")
            variants = {}
            for name, run in references.items():
                translator = new_translator(SimpleNamespace(**configs[name]), target, device)
                saved = torch.load(ROOT / "results/design" / run / f"translator_v{target}.pt", map_location=device, weights_only=False)
                translator.load_state_dict(saved["state_dict"])
                translator.supported_producers = set(saved["supported_producers"])
                variants[name] = translator.eval()
            scenes = make_scenes(current, history, uids, states, early, cutover, ledger, previous, target, args)
            translator, decision = timed(lambda current=current, scenes=scenes, variants=variants:
                                         choose(current, scenes, variants, strength_uids, args.calibration_batch),
                                         ledger, "release_selection")
            del scenes, variants
            decision["target"] = target
            decisions.append(decision)
            config["target_options"][str(target)] = dict(temporal_source_rank=translator.temporal_source_rank,
                response_strength=translator.response_strength.cpu().tolist())
            torch.save(dict(state_dict=translator.state_dict(), supported_producers=sorted(translator.supported_producers)),
                       out / f"translator_v{target}.pt")
            write_json(out / "selection.json", decisions)
            print(json.dumps({k:v for k,v in decision.items() if k != "per_user_validation_logit_mse"}), flush=True)
            for state in states.values():
                state.release(target, translator)
            if target < cli.targets:
                early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY,
                                           ledger, 128, args.calibration_batch)
            previous = current
        selection_ledger = dict(ledger)
        # Candidate fitting is a method requirement. Count both preparations,
        # even though this run reuses frozen files instead of fitting again.
        preparation_keys = ("teacher_build", "teacher_query", "teacher_replay", "translator_fit",
            "calibration_source_backfill", "calibration_adjacent_source", "calibration_lifetime_source",
            "calibration_lifetime_replay", "calibration_lineage_replay", "calibration_query_selection")
        for run in references.values():
            measured = json.loads((ROOT / "results/design" / run / "summary.json").read_text())["ledger_seconds"]
            for key in preparation_keys:
                ledger[key] += measured.get(key, 0)
        write_json(out / "configuration.json", config)
        write_json(out / "summary.json", dict(status="development_complete", quality=[], mechanisms=[],
            configuration_sha256=hashlib.sha256((out / "configuration.json").read_bytes()).hexdigest(),
            elapsed_seconds=time.perf_counter()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
            ledger_seconds=dict(ledger), selection_execution_ledger_seconds=selection_ledger,
            selection=decisions, confirmation_read=False,
            limitation="selection includes both candidate fits; quality and population cost not yet validated"))
        print(json.dumps(dict(status="development_complete", elapsed_seconds=time.perf_counter()-start)), flush=True)
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())

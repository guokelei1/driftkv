#!/usr/bin/env python3
"""Compare frozen Translators on independent pre-release cache scenes."""

import argparse
import hashlib
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from design.data import DAY, ROOT, frozen_model, histories
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

from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def compare(current, scenes, stable, candidate, batch_size, candidate_score=None):
    states = [training_state(s, current, stable) for s in scenes]
    sources = [s.pack_source(stable.target) for s in states]
    features = torch.stack([stable.features(s) for s in sources])
    candidate_features = torch.stack([candidate.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    has_confidence = candidate.source_confidence
    strengths = candidate.confidence_strength(candidate_features).cpu().tolist() if has_confidence else [1.]*len(scenes)
    groups = defaultdict(list)
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
    rows = []
    for length, indices in groups.items():
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset+batch_size]
            source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                                torch.cat([states[i].cache.v for i in batch], 1), length)
            teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                                 torch.cat([scenes[i].teacher.v for i in batch], 1), length)
            candidates = torch.cat([scenes[i].candidates for i in batch])
            delta = torch.cat([scenes[i].query_delta for i in batch])
            exact = score(current, teacher, candidates, delta)[0]
            predictions = {"reuse": score(current, source, candidates, delta)[0]}
            variants = [("stable", stable), ("candidate", candidate)]
            if has_confidence:
                variants.append(("confidence_disabled", candidate))
            for name, mapper in variants:
                mapper.source_confidence = has_confidence and name == "candidate"
                score_fn = candidate_score if name == "candidate" and candidate_score is not None else score
                inputs = features[batch] if name == "stable" else candidate_features[batch]
                extra = {"source_features": inputs} if name == "candidate" and candidate_score is not None else {}
                if getattr(mapper,"query_affine",False):
                    intercept,query_slopes = mapper.query_view(inputs)
                    predictions[name] = score_fn(current, source, candidates, delta, **extra,
                        response_delta=intercept*counts[batch,None,None],
                        response_query_delta=query_slopes*counts[batch,None,None,None,None])[0]
                    continue
                predictions[name] = score_fn(current, source, candidates, delta, **extra,
                    response_delta=mapper.rates(inputs)*counts[batch, None, None],
                    response_time_delta=mapper.time_coefficients(inputs)*counts[batch, None, None, None])[0]
            candidate.source_confidence = has_confidence
            assert torch.isfinite(exact).all() and all(torch.isfinite(v).all() for v in predictions.values())
            errors = {name:(values-exact).square().mean(1).cpu().tolist() for name, values in predictions.items()}
            for index, i in enumerate(batch):
                rows.append(dict(uid=scenes[i].uid, scene=i, retained_events=length,
                    source_feature_writes=float(features[i, stable.base_inputs-1])*1024, strength=strengths[i],
                    **{name:values[index] for name, values in errors.items()}))
    frame = pd.DataFrame(rows)
    methods = ["reuse", "stable", "candidate"]+(["confidence_disabled"] if has_confidence else [])
    per_user = frame.groupby("uid")[methods].mean()
    return frame, dict(users=len(per_user), scenes=len(frame),
        uid_equal_logit_mse=per_user.mean().to_dict(),
        uid_fraction_improved_vs_stable=float((per_user.candidate < per_user.stable).mean()),
        uid_fraction_improved_by_shrink=(float((per_user.candidate < per_user.confidence_disabled).mean()) if has_confidence else None),
        strength_quantiles=frame.strength.quantile([0, .1, .5, .9, 1]).to_dict(),
        shrunk_scene_fraction=float((frame.strength < 1).mean()),
        scope="independent calibration users, actual continuous/adjacent/lifetime prefixes; teacher distance, not AUC")


def main(cli):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    references = {"stable":"v9_stratified_stable64_01", "candidate":cli.candidate_reference}
    configs = {name:json.loads((ROOT / "results/design" / run / "configuration.json").read_text())
               for name, run in references.items()}
    profile = json.loads((ROOT / "results/design/pre_release_auc128_01/configuration.json").read_text())
    uids = profile["calibration_uids"][:cli.users]
    assert len(uids) == cli.users and all(not set(uids) & set(c["calibration_uids"]) for c in configs.values())
    assert configs["stable"]["calibration_uids"] == configs["candidate"]["calibration_uids"]
    args = SimpleNamespace(**configs["stable"])
    args.run_id = cli.run_id
    args.second_moments = configs["candidate"].get("second_moments",False)
    args.lifetime_users = min(16, cli.users//4)
    scene_builder = make_scenes
    files = [Path(__file__), ROOT / "scripts/design/run.py", ROOT / "scripts/design/data.py",
             *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]
    if getattr(cli, "observed_queries", False):
        from design.fit_observed_queries import observed_scenes

        scene_builder = observed_scenes
        files.append(ROOT / "scripts/design/fit_observed_queries.py")
    write_json(out / "configuration.json", dict(vars(cli), calibration_uids=uids,
        references={name:dict(run=run,
            configuration_sha256=hashlib.sha256((ROOT / "results/design" / run / "configuration.json").read_bytes()).hexdigest(),
            translator_sha256={str(t):hashlib.sha256((ROOT / "results/design" / run / f"translator_v{t}.pt").read_bytes()).hexdigest()
                               for t in range(1, cli.targets+1)}) for name, run in references.items()},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        expected_seconds=[30, 120] if cli.users > 8 else [10, 40], expected_gpu_mib=18000,
        causality="pre-release historical scenes; no quality labels or development/confirmation outputs",
        confirmation_read=False))
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    start, ledger = time.perf_counter(), defaultdict(float)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        history = timed(lambda: histories(uids, CUTOVER_DAYS[cli.targets-1]+1), ledger, "history_io")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, args, ledger)
        frames, records = [], []
        for target in range(1, cli.targets+1):
            current = timed(lambda target=target: frozen_model(target, device), ledger, "model_io")
            translators = {}
            for name, run in references.items():
                mapper = new_translator(SimpleNamespace(**configs[name]), target, device)
                saved = torch.load(ROOT / "results/design" / run / f"translator_v{target}.pt", map_location=device, weights_only=False)
                mapper.load_state_dict(saved["state_dict"])
                mapper.supported_producers = set(saved["supported_producers"])
                translators[name] = mapper.eval()
            cutover = CUTOVER_DAYS[target-1]*DAY
            scenes = scene_builder(current, history, uids, states, early, cutover, ledger, previous, target, args)
            frame, record = timed(lambda current=current, scenes=scenes, translators=translators:
                compare(current, scenes, translators["stable"], translators["candidate"], args.calibration_batch),
                ledger, "translator_diagnostic")
            del scenes
            frames.append(frame.assign(target=target))
            records.append(dict(target=target, **record))
            pd.concat(frames, ignore_index=True).to_parquet(out / "calibration_scenes.parquet", index=False)
            print(json.dumps(records[-1]), flush=True)
            for state in states.values():
                state.release(target, translators["candidate"])
            if target < cli.targets:
                early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY,
                                           ledger, 128, args.calibration_batch)
            previous = current
        write_json(out / "summary.json", dict(status="diagnostic_complete", records=records,
            ledger_seconds=dict(ledger), elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20), confirmation_read=False))
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-reference", default="v10_source_confidence64_01")
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    parser.add_argument("--observed-queries", action="store_true",
                        help="Compare both frozen Translators on the same past observed CC query probes.")
    main(parser.parse_args())

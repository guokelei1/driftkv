#!/usr/bin/env python3
"""How much calibration error cannot be represented by a time-only response?"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
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

from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def dispersion(current, scenes, mapper, batch_size):
    states = [training_state(s, current, mapper) for s in scenes]
    sources = [s.pack_source(mapper.target) for s in states]
    features = torch.stack([mapper.features(s) for s in sources])
    counts = torch.stack([s.count.sum() for s in sources])
    multiplicity = Counter(s.uid for s in scenes)
    weights = features.new_tensor([1/multiplicity[s.uid] for s in scenes])
    totals = features.new_zeros(4, mapper.layers)
    diagnostics = []
    groups = defaultdict(list)
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
    for length, indices in groups.items():
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset+batch_size]
            source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                                torch.cat([states[i].cache.v for i in batch], 1), length)
            teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                                 torch.cat([scenes[i].teacher.v for i in batch], 1), length)
            candidates = torch.cat([scenes[i].candidates for i in batch])
            delta = torch.cat([scenes[i].query_delta for i in batch])
            response = mapper.rates(features[batch])
            temporal = mapper.time_coefficients(features[batch])
            phi = current.temporal_enc.features(delta)
            predicted = response[:, None]+torch.einsum("nqt,nltw->nqlw", phi, temporal)
            corrected, trace = score(current, source, candidates, delta, trace=True,
                response_delta=response*counts[batch, None, None],
                response_time_delta=temporal*counts[batch, None, None, None])
            exact_scores = score(current, teacher, candidates, delta)[0]
            reuse_scores = score(current, source, candidates, delta)[0]
            score_error = (corrected-exact_scores).square().mean(1).cpu().tolist()
            reuse_error = (reuse_scores-exact_scores).square().mean(1).cpu().tolist()
            for layer, (block, query, actual) in enumerate(zip(current.blocks, trace.queries, trace.history_heads, strict=True)):
                wanted = history_read(block.attn, query, teacher.k[layer], teacher.v[layer])
                change = ((wanted-actual)/counts[batch, None, None, None]).transpose(1, 2).flatten(2)
                change = change.reshape(len(batch), 4, -1, mapper.width)
                estimate = predicted[:, :, layer].reshape_as(change)
                average = change.mean(2, keepdim=True)
                weight = weights[batch]
                terms = (change.square(), (change-estimate).square(),
                         (change-average).square(), (average-estimate).square())
                totals[:, layer] += torch.stack([(value.mean((1, 2, 3))*weight).sum() for value in terms])
                predicted_heads = trace.corrections[layer]
                actual_norm = actual.square().sum(-1).sqrt().clamp_min(1e-20)
                next_heads = actual+predicted_heads
                quantities = dict(predicted_relative_norm=predicted_heads.square().sum(-1).sqrt()/actual_norm,
                    teacher_relative_norm=(wanted-actual).square().sum(-1).sqrt()/actual_norm,
                    corrected_to_native_norm=next_heads.square().sum(-1).sqrt()/actual_norm,
                    direction_cosine=(actual*next_heads).sum(-1)/(actual_norm*next_heads.square().sum(-1).sqrt().clamp_min(1e-20)))
                means = {k:v.flatten(1).mean(1).cpu().tolist() for k, v in quantities.items()}
                tails = {k:torch.quantile(v.flatten(1), .95, dim=1).cpu().tolist() for k, v in quantities.items()}
                for j, i in enumerate(batch):
                    diagnostics.append(dict(uid=scenes[i].uid, scene=i, layer=layer,
                        source_feature_writes=float(features[i, mapper.base_inputs-1])*1024,
                        retained_events=length, corrected_logit_mse=score_error[j], reuse_logit_mse=reuse_error[j],
                        **{k:values[j] for k, values in means.items()},
                        **{k+"_q95":values[j] for k, values in tails.items()}))
    # Orthogonal within-candidate and between-mean residual decomposition.
    torch.testing.assert_close(totals[1], totals[2]+totals[3], atol=1e-5, rtol=2e-5)
    signal, observed, within, between = totals.cpu().double()
    return dict(users=len(multiplicity), scenes=len(scenes), scene_diagnostics=diagnostics,
        response_energy=signal.tolist(), total_prediction_error=observed.tolist(),
        unrepresented_candidate_variance=within.tolist(), conditional_mean_error=between.tolist(),
        candidate_variance_fraction_of_error=(within/observed.clamp_min(1e-30)).tolist(),
        candidate_variance_fraction_of_signal=(within/signal.clamp_min(1e-30)).tolist(),
        scope="UID-equal calibration response rates at actual corrected queries; four fixed times, sixteen candidates/time; no quality claim")


def main(cli):
    out = ROOT / "results/design" / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    reference = ROOT / "results/design" / cli.reference
    config = json.loads((reference / "configuration.json").read_text())
    args = SimpleNamespace(**config)
    if cli.independent:
        profile = json.loads((ROOT / "results/design/pre_release_auc128_01/configuration.json").read_text())
        uids = profile["calibration_uids"][:cli.users]
        assert not set(uids) & set(config["calibration_uids"])
    else:
        uids = config["calibration_uids"][:cli.users]
    assert len(uids) == cli.users
    args.lifetime_users = min(args.lifetime_users, cli.users//4)
    write_json(out / "configuration.json", dict(vars(cli), calibration_uids=uids,
        reference_configuration_sha256=hashlib.sha256((reference / "configuration.json").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected_seconds=[30, 120] if cli.users > 8 else [10, 40], expected_gpu_mib=12000,
        confirmation_read=False))
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    start, ledger = time.perf_counter(), defaultdict(float)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        history = timed(lambda: histories(uids, CUTOVER_DAYS[cli.targets-1]+1), ledger, "history_io")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, args, ledger)
        rows = []
        for target in range(1, cli.targets+1):
            current = timed(lambda target=target: frozen_model(target, device), ledger, "model_io")
            mapper = new_translator(args, target, device)
            saved = torch.load(reference / f"translator_v{target}.pt", map_location=device, weights_only=False)
            mapper.load_state_dict(saved["state_dict"])
            mapper.supported_producers = set(saved["supported_producers"])
            cutover = CUTOVER_DAYS[target-1]*DAY
            scenes = make_scenes(current, history, uids, states, early, cutover, ledger, previous, target, args)
            record = timed(lambda current=current, scenes=scenes, mapper=mapper:
                           dispersion(current, scenes, mapper, args.calibration_batch), ledger, "response_diagnostic")
            del scenes
            pd.DataFrame(record.pop("scene_diagnostics")).assign(target=target).to_parquet(
                out / f"response_scenes_m{target}.parquet", index=False)
            rows.append(dict(target=target, **record))
            write_json(out / "dispersion.json", rows)
            print(json.dumps(rows[-1]), flush=True)
            for state in states.values():
                state.release(target, mapper)
            if target < cli.targets:
                early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY,
                                           ledger, 128, args.calibration_batch)
            previous = current
        write_json(out / "summary.json", dict(status="diagnostic_complete", rows=rows, ledger_seconds=dict(ledger),
            elapsed_seconds=time.perf_counter()-start, peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
            confirmation_read=False, passing_checks="orthogonal candidate/conditional-mean MSE decomposition"))
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference", default="v9_stratified_stable64_01")
    parser.add_argument("--users", type=int, default=64)
    parser.add_argument("--independent", action="store_true", help="Use the previously reserved independent128 calibration users")
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())

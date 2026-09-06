#!/usr/bin/env python3
"""Test a shared per-head correction using the already computed history read.

The source/time Translator remains frozen. Fit 36 extra slopes on its fitting
users, then evaluate the changed six-layer forward on separate calibration
users. This diagnostic does not install a new population reader.
"""

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from design import diagnose_query_dispersion as driver
from design import probe_source_confidence as probe
from design.data import ROOT
from design.run import timed, training_state, write_json

from hstu_kvcache.adaptation import reader
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def fit_slopes(current, scenes, mapper, batch_size):
    states = [training_state(scene, current, mapper) for scene in scenes]
    sources = [state.pack_source(mapper.target) for state in states]
    features = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([source.count.sum() for source in sources])
    multiplicity = Counter(scene.uid for scene in scenes)
    weights = features.new_tensor([1/multiplicity[scene.uid] for scene in scenes])
    active = mapper.active_layers(features)
    numerator = features.new_zeros(mapper.layers, current.cfg.num_heads, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.cache.seq_len].append(index)
    for length, indices in groups.items():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start+batch_size]
            source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                                torch.cat([states[i].cache.v for i in batch], 1), length)
            teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                                 torch.cat([scenes[i].teacher.v for i in batch], 1), length)
            _, trace = reader.score(current, source,
                torch.cat([scenes[i].candidates for i in batch]),
                torch.cat([scenes[i].query_delta for i in batch]), trace=True,
                response_delta=mapper.rates(features[batch])*counts[batch, None, None],
                response_time_delta=mapper.time_coefficients(features[batch])*counts[batch, None, None, None])
            for layer, (block, query, native, predicted) in enumerate(zip(
                    current.blocks, trace.queries, trace.history_heads, trace.corrections, strict=True)):
                wanted = reader.history_read(block.attn, query, teacher.k[layer], teacher.v[layer])
                x = native.double()/counts[batch, None, None, None]
                residual = (wanted-native-predicted).double()/counts[batch, None, None, None]
                weight = weights[batch]*active[batch, layer]
                numerator[layer] += ((x*residual).mean((2, 3))*weight[:, None]).sum(0)
                denominator[layer] += (x.square().mean((2, 3))*weight[:, None]).sum(0)
    slopes = numerator/denominator.clamp_min(1e-20)
    assert torch.isfinite(slopes).all()
    return slopes.float(), dict(numerator=numerator.cpu().tolist(), denominator=denominator.cpu().tolist(),
        fit="UID-equal per-token residual least squares, one scalar per layer/head; no model-ID branches",
        clearance="same dependency-based active-layer mask as the frozen Translator")


def main(cli):
    out = ROOT / "results/design" / cli.run_id
    checks, ledger = [], defaultdict(float)
    original_history = reader.history_read
    reference = "v9_stratified_stable64_01"
    if cli.coefficients:
        config = json.loads((ROOT / "results/design" / cli.coefficients / "configuration.json").read_text())
        assert config["reference"] == reference and not config["independent"]
    files = [Path(__file__), ROOT / "scripts/design/diagnose_query_dispersion.py",
             ROOT / "scripts/design/probe_source_confidence.py", ROOT / "scripts/design/run.py",
             ROOT / "scripts/design/data.py", *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]

    def compare(current, scenes, mapper, batch_size):
        if cli.coefficients:
            saved = ROOT / "results/design" / cli.coefficients / f"native_slopes_m{mapper.target}.json"
            record = json.loads(saved.read_text())
            slopes = mapper.center.new_tensor(record["slopes"])
            fitting = dict(reference=str(saved.relative_to(ROOT)), sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
        else:
            slopes, fitting = timed(lambda: fit_slopes(current, scenes, mapper, batch_size), ledger, "native_slope_fit")
        write_json(out / f"native_slopes_m{mapper.target}.json", dict(slopes=slopes.cpu().tolist(), **fitting))
        checked = False

        def score_variant(*args, source_features, **kwargs):
            nonlocal checked
            active = mapper.active_layers(source_features)
            layer_ids = {id(block.attn):layer for layer, block in enumerate(current.blocks)}
            enabled = False

            def history_read(attention, query, key, value, *, count=None):
                native = original_history(attention, query, key, value, count=count)
                layer = layer_ids[id(attention)]
                coefficient = slopes[layer][None, :, None, None]*active[:, layer, None, None, None]
                return native+native*coefficient if enabled else native

            expected = reader.score(*args, **kwargs)[0] if not checked else None
            reader.history_read = history_read
            try:
                if not checked:
                    actual = reader.score(*args, **kwargs)[0]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    checks.append(dict(target=mapper.target, disabled_max_abs=float((actual-expected).abs().max())))
                    checked = True
                enabled = True
                return reader.score(*args, **kwargs)
            finally:
                reader.history_read = original_history

        frame, result = probe.compare(current, scenes, mapper, mapper, batch_size, candidate_score=score_variant)
        result.update(scene_diagnostics=frame.to_dict(orient="records"),
            native_slopes=slopes.cpu().tolist(), fitting=fitting,
            scope=("independent calibration users" if cli.coefficients else "coefficient-fitting calibration users")
                  + "; actual continuous/adjacent/lifetime prefixes; teacher distance, not AUC",
            inference="native history plus frozen summary/time correction plus shared slope times native history; self unchanged")
        return result

    driver.dispersion = compare
    args = SimpleNamespace(run_id=cli.run_id, reference=reference, users=cli.users,
        independent=bool(cli.coefficients), targets=cli.targets, coefficients=cli.coefficients,
        override_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prospective_resources=dict(expected_seconds=[15, 60] if cli.users == 8 else [50, 180],
            expected_gpu_mib=14000, timing_scope="mechanism run may share a GPU with frozen development; not final cost"))
    try:
        driver.main(args)
    finally:
        reader.history_read = original_history
        if out.exists():
            with tarfile.open(out / "native_response_source.tar.gz", "w:gz") as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
    summary = json.loads((out / "summary.json").read_text())
    summary.update(reader_canary=checks, extra_ledger_seconds=dict(ledger),
                   passing_checks="disabled read wrapper bitwise equal to original frozen reader")
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(8, 64, 128), required=True)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    parser.add_argument("--coefficients", help="Read slopes fitted in this completed run; use independent calibration UIDs.")
    main(parser.parse_args())

#!/usr/bin/env python3
"""Separate per-state time-view representation from shared translation error.

Teacher-fitted views are diagnostic interventions only. Compare fitting the
four evaluation query groups with fitting 64 legal times on the same items;
both use the actual corrected queries over three passes and the same cache.
"""

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from design import diagnose_query_dispersion as driver
from design.data import ROOT
from design.run import training_state, write_json

from hstu_kvcache.adaptation import inference
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def diagnose(current, scenes, mapper, batch_size, passes=3):
    states = [training_state(scene, current, mapper) for scene in scenes]
    sources = [state.pack_source(mapper.target) for state in states]
    features = torch.stack([mapper.features(source) for source in sources])
    counts = torch.stack([source.count.sum() for source in sources])
    constant, temporal = mapper.rates(features), mapper.time_coefficients(features)
    active = mapper.active_layers(features)
    multiplicity = Counter(scene.uid for scene in scenes)
    weights = counts.new_tensor([1/multiplicity[scene.uid] for scene in scenes])
    response_totals = {name:{scope:counts.new_zeros(2, mapper.layers) for scope in ("fitting_times", "probe_times")}
                       for name in ("same_query_oracle", "broad_time_oracle")}
    groups = defaultdict(list)
    for index, state in enumerate(states):
        groups[state.cache.seq_len].append(index)
    rows, normal_errors = [], []
    for length, indices in groups.items():
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset+batch_size]
            source = HSTUKVCache(torch.cat([states[i].cache.k for i in batch], 1),
                                torch.cat([states[i].cache.v for i in batch], 1), length)
            teacher = HSTUKVCache(torch.cat([scenes[i].teacher.k for i in batch], 1),
                                 torch.cat([scenes[i].teacher.v for i in batch], 1), length)
            items = torch.cat([scenes[i].candidates for i in batch])
            delta = torch.cat([scenes[i].query_delta for i in batch])
            assert items.shape[1] == 64
            n = counts[batch]

            def evaluate(intercept, slopes, source=source, items=items, delta=delta, n=n):
                return score(current, source, items, delta,
                    response_delta=intercept*n[:, None, None],
                    response_time_delta=slopes*n[:, None, None, None])[0]

            exact = score(current, teacher, items, delta)[0]
            values = dict(reuse=score(current, source, items, delta)[0],
                          stable=evaluate(constant[batch], temporal[batch]))
            for name, times in (("same_query_oracle", 4), ("broad_time_oracle", 64)):
                if times == 4:
                    query_times = delta[:, ::16]
                else:
                    fraction = torch.linspace(0, 1, times, device=delta.device)
                    query_times = (delta.max(1).values.log()[:, None]*fraction).exp().round().clamp_min(1)
                    query_times = torch.minimum(query_times, delta.max(1).values[:, None])
                assert (query_times >= 1).all() and (query_times <= delta.max(1).values[:, None]).all()
                candidates = items[:, :16].repeat(1, times)
                query_delta = query_times.repeat_interleave(16, 1)
                phi = current.temporal_enc.features(query_times).double()
                centered = phi-phi.mean(1, keepdim=True)
                gram = centered.transpose(1, 2) @ centered
                regularizer = (gram.diagonal(dim1=-2, dim2=-1).sum(1)/32*1e-4).clamp_min(1e-8)
                system = gram+regularizer[:, None, None]*torch.eye(32, device=gram.device, dtype=gram.dtype)
                intercept, slopes = constant[batch].clone(), temporal[batch].clone()
                for _ in range(passes):
                    rates = inference.calibration_rates(current, source, teacher, candidates, query_delta, n,
                        intercept*n[:, None, None], slopes*n[:, None, None, None], times)
                    y = rates.flatten(2).double()
                    rhs = centered.transpose(1, 2) @ (y-y.mean(1, keepdim=True))
                    solution = torch.linalg.solve(system, rhs)
                    residual = system @ solution-rhs
                    normal_errors.append(float(residual.abs().max()/rhs.abs().max().clamp_min(1e-20)))
                    torch.testing.assert_close(system @ solution, rhs, atol=1e-7, rtol=1e-6)
                    intercept = (y.mean(1)-(phi.mean(1)[:, None] @ solution).squeeze(1)).float().reshape(-1, mapper.layers, mapper.width)
                    slopes = solution.float().reshape(-1, 32, mapper.layers, mapper.width).transpose(1, 2)
                    intercept *= active[batch, :, None]
                    slopes *= active[batch, :, None, None]
                values[name] = evaluate(intercept, slopes)
                for scope, ids, gaps, time_count in (("fitting_times", candidates, query_delta, times),
                                                     ("probe_times", items, delta, 4)):
                    actual_rates = inference.calibration_rates(current, source, teacher, ids, gaps, n,
                        intercept*n[:, None, None], slopes*n[:, None, None, None], time_count)
                    basis = current.temporal_enc.features(gaps[:, ::16])
                    prediction = intercept[:, None]+torch.einsum("bqt,bltw->bqlw", basis, slopes)
                    for index, quantity in enumerate(((actual_rates-prediction).square(), actual_rates.square())):
                        response_totals[name][scope][index] += (quantity.mean((1, 3))*weights[batch, None]).sum(0)
            assert all(torch.isfinite(value).all() for value in values.values())
            errors = {name:(value-exact).square().mean(1).cpu().tolist() for name, value in values.items()}
            for j, index in enumerate(batch):
                rows.append(dict(uid=scenes[index].uid, scene=index, retained_events=length,
                                 **{name:error[j] for name, error in errors.items()}))
    frame = pd.DataFrame(rows)
    methods = ["reuse", "stable", "same_query_oracle", "broad_time_oracle"]
    per_user = frame.groupby("uid")[methods].mean()
    return dict(users=len(per_user), scenes=len(scenes), scene_diagnostics=rows,
        corrected_query_passes=passes,
        uid_equal_logit_mse=per_user.mean().to_dict(),
        uid_fraction_improved_vs_stable={name:float((per_user[name] < per_user.stable).mean()) for name in methods[2:]},
        maximum_normal_equation_relative_residual=max(normal_errors),
        response_relative_mse={name:{scope:(totals[0]/totals[1].clamp_min(1e-20)).cpu().tolist()
                                    for scope, totals in scopes.items()} for name, scopes in response_totals.items()},
        scope="per-scene teacher intervention, not an executable method or formal upper bound; same-query oracle sees evaluation probes; broad-time oracle shares items and endpoints")


def main(cli):
    out = ROOT / "results/design" / cli.run_id
    files = [Path(__file__), ROOT / "scripts/design/diagnose_query_dispersion.py",
             ROOT / "scripts/design/run.py", ROOT / "scripts/design/data.py",
             *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]
    args = SimpleNamespace(run_id=cli.run_id, reference="v9_stratified_stable64_01", users=cli.users,
        independent=False, targets=cli.targets, passes=cli.passes,
        override_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prospective_resources=dict(expected_seconds=[15, 120] if cli.users == 8 else [100, 400],
            expected_gpu_mib=16000, query_budget=f"64 legal times x the same16 items, plus the four-probe oracle; {cli.passes} corrected-query passes",
            timing_scope="mechanism intervention may share GPU with development; not method cost"))
    driver.dispersion = lambda current, scenes, mapper, batch_size: diagnose(current, scenes, mapper, batch_size, cli.passes)
    try:
        driver.main(args)
    finally:
        if out.exists():
            with tarfile.open(out / "time_view_source.tar.gz", "w:gz") as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
    summary = json.loads((out / "summary.json").read_text())
    summary["passing_checks"] = "legal pre-release query times, finite outputs, and double-precision normal equations"
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(8, 64), required=True)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    parser.add_argument("--passes", type=int, choices=(3, 6), default=3)
    main(parser.parse_args())

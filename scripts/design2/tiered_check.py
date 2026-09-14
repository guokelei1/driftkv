"""Fixed32 spectral preparation and frozen-input replay for equivalent decisions."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from design.data import ROOT, DAY, frozen_model, histories
from design.diagnose_query_holdout import batch_cache
from design.run import initialize_calibration, replay_calibration, timed, write_json
from design2.common import FrozenC, TARGETS, batches, make_panel, original_args, source_scenes
from design2.run_detection import metadata
from design2.scale_calibration import assign_groups
from hstu_kvcache.design2.geometry import features, score_layer
from hstu_kvcache.design2.spectral import prepare_factor, quadratic_bounds, stack_specs
from insight_two.common import CUTOVER_DAYS

OUT = ROOT/"results/design2/tiered32_01"
CONFIG = ROOT/"configs/design2/tiered32_01.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    torch.set_num_threads(4)
    out = OUT/"spectrum"
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    for target in TARGETS:
        source = ROOT/f"results/design2/geometry_01/cholesky_m{target}.pt"
        layers = torch.load(source, map_location="cpu", weights_only=True)
        prepared = []
        for layer, factors in enumerate(layers):
            items = [prepare_factor(l, 32) for l in factors]
            rows.extend(dict(target=target, layer=layer, head=h, **item["metadata"])
                        for h, item in enumerate(items))
            prepared.append(stack_specs(items))
        torch.save(prepared, out/f"spectral_m{target}.pt")
        print(json.dumps(dict(target=target, elapsed_seconds=time.perf_counter()-started)), flush=True)
    p, matrices = 1281, len(rows)
    write_json(out/"summary.json", dict(status="complete", matrices=matrices, rank=32, rows=rows,
        elapsed_seconds=time.perf_counter()-started,
        verified_preparation_gemm_flops=matrices*6*p**3,
        preparation_scalar_allowance_flops=matrices*32*p*p,
        eigendecomposition_cost_model_flops=matrices*10*p**3,
        eigendecomposition_count_scope="10p^3 explicit planning charge, not measured LAPACK internal count; positive eigensolver work cannot be omitted in a success claim",
        projection_storage_bytes=sum(x.stat().st_size for x in out.glob("*.pt")),
        configuration_sha256=sha(CONFIG),
        geometry_sha256={str(t):sha(ROOT/f"results/design2/geometry_01/cholesky_m{t}.pt") for t in TARGETS},
        spectral_source_sha256=sha(ROOT/"src/hstu_kvcache/design2/spectral.py")))


def old_run(role, start):
    return "canary16_01" if role == "residual_calibration" and start == 0 else f"detection01_{role}_{start:04d}"


def evaluate(current, history, scenes, adapter, cutover, out, target, cli, ledger, fitted):
    p = timed(lambda: adapter.prepare(scenes), ledger, "source_and_view")
    meta = pd.DataFrame(metadata(scenes, p, target, cli.role, cutover))
    meta = assign_groups(meta, fitted["groups"])
    params = fitted["final_fit"]
    meta["b"] = meta.group.map(lambda g: params[g]["b"])
    meta["a"] = meta.group.map(lambda g: params[g]["a"])
    original = ROOT/"results/design2"/old_run(cli.role, cli.start)
    ref = pd.read_parquet(original/f"states_m{target}.parquet").set_index(["uid", "state_ordinal"])
    ref = ref.loc[list(zip(meta.uid, meta.state_ordinal))].reset_index()
    for key in ("uid", "state_ordinal", "target", "kind", "count", "active_layers"):
        np.testing.assert_array_equal(meta[key], ref[key])
    queries = pd.read_parquet(original/f"queries_m{target}.parquet",
        columns=["uid", "state_ordinal", "query_index", "item", "adapt"])
    queries = queries.set_index(["uid", "state_ordinal", "query_index"]).sort_index()
    device = next(current.parameters()).device
    factors = torch.load(ROOT/f"results/design2/geometry_01/cholesky_m{target}.pt", map_location=device, weights_only=True)
    spectra = torch.load(OUT/f"spectrum/spectral_m{target}.pt", map_location=device, weights_only=True)
    tau = fitted["thresholds"]["calibrated"]["0.8"]
    guard = 1e-10
    b = meta.b.to_numpy(); a = meta.a.to_numpy()
    direct = (b > tau+guard) | ((a == 0) & (np.abs(b-tau) > guard))
    decisions = b <= tau
    exact_decisions = b+a*ref.detection.to_numpy() <= tau
    stages = np.where(direct, "formula", "pending").astype(object)
    low = np.full(len(meta), np.nan); high = low.copy(); exact_new = low.copy()
    max_logit_delta, max_score_relative_delta = 0., 0.
    for indices in batches(scenes):
        pending = [i for i in indices if not direct[i]]
        if not pending:
            continue
        # Preserve original C batch, even if only some members require geometry.
        cache = batch_cache([scenes[i].state.cache for i in indices])
        panel = make_panel(history, scenes, indices, cutover, device)
        z, trace, obs, _ = timed(lambda: adapter.read(current, cache, panel, p, indices), ledger, "frozen_C_input_read")
        for j, i in enumerate(indices):
            expected = queries.loc[(scenes[i].uid, scenes[i].ordinal)].sort_index()
            np.testing.assert_array_equal(panel[0][j].cpu().numpy(), expected.item.to_numpy())
            delta = np.max(np.abs(z[j].cpu().numpy()-expected.adapt.to_numpy()))
            max_logit_delta = max(max_logit_delta, float(delta))
            np.testing.assert_allclose(z[j].cpu().numpy(), expected.adapt.to_numpy(), atol=2e-5, rtol=2e-5)
        local = [indices.index(i) for i in pending]
        lows, highs = [], []
        for layer in range(6):
            def coarse():
                f = features(p["latent"][pending], trace.queries[layer][local], obs[layer][local], adapter.parameters[layer])
                lo, hi = quadratic_bounds(f, spectra[layer])
                scale = (p["counts"][pending].double()*p["active"][pending, layer])[:, None, None]
                return (lo.sqrt()*scale).amax((1, 2)), (hi.sqrt()*scale).amax((1, 2))
            lo, hi = timed(coarse, ledger, "spectral_coarse")
            lows.append(lo); highs.append(hi)
        ul = torch.stack(lows).amax(0).cpu().numpy()
        uh = torch.stack(highs).amax(0).cpu().numpy()
        low[pending], high[pending] = ul, uh
        # Reference checks are offline diagnostics, never part of branching.
        retained_u = ref.detection.to_numpy()[pending]
        assert np.all(ul <= retained_u+2e-5*np.maximum(1, retained_u))
        assert np.all(uh >= retained_u-2e-5*np.maximum(1, retained_u))
        el, eh = b[pending]+a[pending]*ul, b[pending]+a[pending]*uh
        accept = np.isfinite(eh) & (eh <= tau-guard)
        reject = np.isfinite(el) & (el > tau+guard)
        assert not np.any(accept & reject)
        ambiguous = []
        for j, i in enumerate(pending):
            if accept[j] or reject[j]:
                stages[i] = "bound_accept" if accept[j] else "bound_reject"
                decisions[i] = accept[j]
            else:
                stages[i] = "exact_fallback"
                ambiguous.append(i)
        # Canary verifies exact scores for every coarse state; population only
        # solves ambiguous states. Extra canary validation solves are charged separately.
        verification = pending if cli.canary else ambiguous
        if verification:
            loc = [indices.index(i) for i in verification]
            scores = []
            for layer in range(6):
                def accurate():
                    return score_layer(p["latent"][verification], trace.queries[layer][loc], obs[layer][loc],
                        p["counts"][verification], p["active"][verification, layer], adapter.parameters[layer], factors[layer])[0]
                scores.append(timed(accurate, ledger, "accurate_and_canary_check"))
            actual = torch.stack(scores).amax(0).amax(-1).cpu().numpy()
            exact_new[verification] = actual
            expected = ref.detection.to_numpy()[verification]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            max_score_relative_delta = max(max_score_relative_delta, float(np.max(np.abs(actual-expected)/np.maximum(1, expected))))
            # Same-input bounds must enclose the actual accurate solve, not merely
            # pass the reference rehydration tolerance.
            assert np.all(low[verification] <= actual) and np.all(actual <= high[verification])
            for j, i in enumerate(verification):
                if stages[i] == "exact_fallback":
                    decisions[i] = b[i]+a[i]*actual[j] <= tau
    assert not np.any(stages == "pending")
    assert np.array_equal(decisions, exact_decisions), "Changed a frozen detector decision"
    meta = meta.assign(stage=stages, accept=decisions, reference_accept=exact_decisions,
        u_lower=low, u_upper=high, reference_u=ref.detection.to_numpy(), recomputed_u=exact_new)
    meta.to_parquet(out/f"states_m{target}.parquet", index=False)
    return dict(target=target, states=len(meta), stage_counts=meta.stage.value_counts().to_dict(),
        mismatches=int((decisions != exact_decisions).sum()), max_adapt_logit_delta=max_logit_delta,
        max_accurate_score_relative_delta=max_score_relative_delta,
        coarse_states=int((~direct).sum()), canary_extra_solve_states=int(sum(stages == "bound_accept")+sum(stages == "bound_reject")) if cli.canary else 0)


@torch.no_grad()
def replay(cli):
    out = OUT/"chunks"/f"{cli.role}_{cli.start:04d}"
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    device = torch.device(cli.device)
    torch.cuda.set_device(device)
    protocol = json.loads((ROOT/"configs/design2/detection_01.json").read_text())
    fitted = json.loads((ROOT/"configs/design2/scale_calibration_01_fitted.json").read_text())
    uids = protocol["groups"][cli.role][cli.start:cli.start+cli.users]
    lifetime = set(protocol["lifetime_uids"][cli.role])
    write_json(out/"configuration.json", dict(vars(cli), uids=uids, threshold=fitted["thresholds"]["calibrated"]["0.8"],
        protocol_sha256=sha(CONFIG), fitted_sha256=sha(ROOT/"configs/design2/scale_calibration_01_fitted.json"),
        source_sha256=sha(Path(__file__)), spectral_sha256=sha(ROOT/"src/hstu_kvcache/design2/spectral.py")))
    started = time.perf_counter(); ledger = defaultdict(float); results = []
    history = timed(lambda: histories(uids, CUTOVER_DAYS[-1]+1), ledger, "history_load")
    previous = frozen_model(0, device)
    states, early = initialize_calibration(previous, history, uids, CUTOVER_DAYS[0]*DAY, original_args(), ledger)
    for target in range(1, 6):
        current = frozen_model(target, device)
        cutover = CUTOVER_DAYS[target-1]*DAY
        if target in TARGETS:
            adapter = FrozenC(target, device)
            scenes = source_scenes(current, previous, history, uids, states, early, cutover, target, lifetime, ledger)
            results.append(evaluate(current, history, scenes, adapter, cutover, out, target, cli, ledger, fitted))
            print(json.dumps(dict(target=target, elapsed_seconds=time.perf_counter()-started, result=results[-1])), flush=True)
            del scenes, adapter
        for state in states.values():
            state.release(target, None)
        if target < 5:
            early = replay_calibration(current, history, states, uids, cutover, CUTOVER_DAYS[target]*DAY, ledger, 128, 16)
        previous = current
    write_json(out/"summary.json", dict(status="complete", users=len(uids), results=results,
        elapsed_seconds=time.perf_counter()-started, ledger_seconds=dict(ledger),
        peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,
        new_teacher_outputs=0, frozen_input_replay=True, C_refitted=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "replay"])
    p.add_argument("--role", default="residual_calibration")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--users", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--canary", action="store_true")
    args = p.parse_args()
    prepare() if args.mode == "prepare" else replay(args)

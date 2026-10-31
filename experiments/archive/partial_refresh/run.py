#!/usr/bin/env python3
"""Publication validation and continuous quality replay for chapter 4.4.

No foundation training, request labels in policy selection, or old-result writes.
All publication candidates share one actual retained old-code snapshot and one
full-current reference. Serving encodes only the chosen rows at the first read.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from hstu_kvcache.evaluation.release_context import END, RELEASES, load_models, setup_device, ledger_flops
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.methods.cachetide.compact.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.evaluation.cost import CostModel, eager_read
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256, write_json

CALIBRATION_PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"
QUALITY_PANEL = ROOT / "results/design_system_2026_10/phase2/representative4096/panel"
CALIBRATIONS = ROOT / "results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_lazy/calibration"
REFERENCE = ROOT / "results/design_system_2026_10/lightweight_read/fullitem8_local_kv32_joint16_support_lazy/development4096/scores.parquet"
RATIOS = (0., .125, .25, .5, .75, 1.)


def load_adapters(folder, device):
    adapters, records = {}, {}
    for version, _ in RELEASES:
        path = folder / version / "calibration.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["kind"] != LightweightFullItemKVAdapter.kind:
            raise ValueError("expected current full-item Key/Value adapter")
        adapter = LightweightFullItemKVAdapter(**payload["config"])
        adapter.load_state_dict(payload["state_dict"])
        adapters[version] = adapter.to(device).eval().requires_grad_(False)
        records[version] = dict(path=str(path), sha256=sha256(path), config=payload["config"])
    return adapters, records


def source_records():
    names = ("run.py", "session.py", "analyze.py")
    result = {name: sha256(Path(__file__).parent / name) for name in names
              if (Path(__file__).parent / name).exists()}
    dependencies = ("src/hstu_kvcache/adaptation/reader.py", "src/hstu_kvcache/models/hstu.py",
        "src/hstu_kvcache/models/attention.py", "src/hstu_kvcache/models/state_transition.py",
        "src/hstu_kvcache/evaluation/trajectory.py", "src/hstu_kvcache/methods/cachetide/lifecycle/lifecycle.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight.py", "src/hstu_kvcache/methods/cachetide/compact/lightweight_fullitem.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_fullitem_kv.py", "src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_dispatch.py",
        "src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_kernel.py", "src/hstu_kvcache/methods/cachetide/compact/lightweight_kv_runtime_kernel.py",
        "src/hstu_kvcache/evaluation/release_context.py", "src/hstu_kvcache/evaluation/panels.py", "src/hstu_kvcache/evaluation/history.py",
        "src/hstu_kvcache/evaluation/cost.py", "src/hstu_kvcache/methods/baselines/protocol/cost.py")
    result.update({name:sha256(ROOT/name) for name in dependencies})
    result.update({str(path.relative_to(ROOT)):sha256(path)
                   for path in (ROOT/"experiments/archive/partial_refresh/round2").glob("*.py")})
    return result


def new_output(args, settings):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "configuration.json", dict(
        **settings, gpu=args.gpu, seed=17, rank=8,
        source_code_sha256=source_records(), calibration_panel=str(args.calibration_panel),
        calibration_panel_sha256=sha256(args.calibration_panel/"binding.json"),
        quality_panel=str(args.quality_panel), calibrations=str(args.calibrations),
        quality_panel_sha256=sha256(args.quality_panel/"binding.json"),
        native_writes="unchanged", code_policy="actual prior code; previous encoder only; tail plus mandatory refresh",
        cost_scope="analytical FLOPs; no latency or throughput claim"))


def delta(session, at):
    previous = session.trajectory.last_timestamp
    return torch.tensor([0 if previous is None else int(at) - previous],
                        dtype=torch.float32, device=session.cache.k.device)


@torch.no_grad()
def full_state(session):
    return session.encode_reference()


def query_flops(model, adapter, length, queries=16):
    cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, "torch")
    return eager_read(cm, length, queries=queries) + adapter.estimate_flops(
        tokens=length, candidates=queries)["extra_history_read"]


@torch.no_grad()
def publication_scores(session, ids, at, state):
    return score(session.trajectory.model, session.cache, ids, delta(session, at),
                 history_read_fn=session.adapter.make_history_read(state))[0][0]


@torch.no_grad()
def fit_scales(models, adapters, histories, uids, known):
    """A predeclared fit-only correction-energy scale, not a learned policy."""
    result = {v: Counter() for v, _ in RELEASES}
    for number, uid in enumerate(uids):
        trajectory = NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1])
        for version, at in RELEASES:
            trajectory.switch(models[version], int(version[1:]), at)
            n = trajectory.cache.seq_len
            if not n:
                continue
            model, adapter = models[version], adapters[version]
            ids = torch.tensor([mixed_candidates(uid, trajectory.retained_events[1], known, 16, seed=17)],
                               device=trajectory.cache.k.device)
            item_ids = torch.tensor(trajectory.retained_events[1], device=ids.device)[None]
            state = adapter.encode_cache(trajectory.cache, model.lookup_item_embeddings(item_ids))
            gap = torch.tensor([at - trajectory.last_timestamp], device=ids.device, dtype=torch.float32)
            native = score(model, trajectory.cache, ids, gap)[0][0]
            full = score(model, trajectory.cache, ids, gap,
                         history_read_fn=adapter.make_history_read(state))[0][0]
            result[version]["squared_correction"] += float((full - native).square().sum())
            result[version]["queries"] += len(full)
            cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, "torch")
            result[version]["shared_forward_flops"] += (
                eager_read(cm, n, queries=16) + query_flops(model, adapter, n)
                + adapter.estimate_flops(tokens=n)["token_transform"] + 3 * len(full))
        if (number + 1) % 32 == 0:
            print(json.dumps(dict(stage="fit_scale", users=number + 1)), flush=True)
    return {v: dict(**c, correction_energy=c["squared_correction"] / max(c["queries"], 1))
            for v, c in result.items()}


@torch.no_grad()
def validate(args):
    from experiments.archive.partial_refresh.session import PartialRefreshSession
    began = time.perf_counter()
    panel, groups, requests = load_panel(args.calibration_panel)
    fit = list(groups["fit"][:args.fit_users])
    validation = list(groups["validation"][:args.validation_users])
    if set(fit) & set(validation):
        raise ValueError("fit and validation overlap")
    new_output(args, dict(stage="publication_validation", fit_uids=fit, validation_uids=validation,
                         ratios=list(args.ratios), tolerance=args.tolerance,storage_mode=args.storage_mode,
                         scale_reference=str(args.scale_reference) if args.scale_reference else None,
                         rule="minimum independently evaluated rho with E <= tolerance * fit correction energy",
                         queries_per_user=16, no_future_request_labels=True))
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device)
    adapters, weights = load_adapters(args.calibrations, device)
    histories = load_history(args.calibration_panel, requested_uids=fit + validation)
    if args.scale_reference:
        scale_record = json.loads(args.scale_reference.read_text())
        if scale_record["weights"] != weights or scale_record["fit_uids"] != fit:
            raise ValueError("reused fit energy needs identical fixed weights and fit users")
        scales = scale_record["fit_scales"]
        print(json.dumps(dict(stage="fit_scale_reused",users=len(fit),
                              source=str(args.scale_reference),deployment_forward_flops_charged=True)),flush=True)
    else:
        scales = fit_scales(models, adapters, histories, fit, panel["known_vocab_size"])
    sessions = {uid: PartialRefreshSession(NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1]),
                                           kernel="torch",storage_mode=args.storage_mode) for uid in validation}
    policies, releases = {}, {}
    diagnostics = []
    for index, (version, at) in enumerate(RELEASES):
        totals = {rho: Counter() for rho in args.ratios}
        resource = Counter()
        for uid, session in sessions.items():
            session.publish(models[version], int(version[1:]), at, adapters[version], rho=1.)
            n = session.cache.seq_len
            if not n:
                continue
            ids = torch.tensor([mixed_candidates(uid, session.trajectory.retained_events[1],
                               panel["known_vocab_size"], 16, seed=17)], device=device)
            reference = full_state(session)
            full = publication_scores(session, ids, at, reference)
            resource["full_reference_encoder_flops"] += adapters[version].estimate_flops(tokens=n)["token_transform"]
            resource["reference_support_mask_flops"] += reference.codes.numel() if args.storage_mode=="masked" else 0
            resource["full_reference_query_flops"] += query_flops(models[version], adapters[version], n)
            for rho in args.ratios:
                plan = session.refresh_plan(rho)
                candidate = session.candidate_state(rho, reference)
                actual = publication_scores(session, ids, at, candidate)
                if not torch.isfinite(actual).all():
                    raise RuntimeError("nonfinite publication scores")
                totals[rho]["squared_error"] += float((actual - full).square().sum())
                totals[rho]["queries"] += len(actual)
                totals[rho]["history_rows"] += n
                totals[rho]["refresh_rows"] += plan["selected"]
                totals[rho]["forced_missing_rows"] += plan["missing"]
                totals[rho]["forced_older_rows"] += plan["older"]
                resource["candidate_query_flops"] += query_flops(models[version], adapters[version], n)
                resource["score_difference_flops"] += 3 * len(full)
                if rho == 0.:
                    diagnostics.append(dict(uid=uid, version=version, history_length=n,
                        missing_rows=plan["missing"], older_rows=plan["older"],
                        relative_score_error=float((actual-full).square().mean()) / max(scales[version]["correction_energy"], 1e-30)))
        epsilon = args.tolerance * scales[version]["correction_energy"]
        curves = []
        for rho, c in totals.items():
            mse = c["squared_error"] / max(c["queries"], 1)
            curves.append(dict(rho=rho, **c, score_mse=mse, epsilon=epsilon,
                               relative_correction_energy_error=mse / max(scales[version]["correction_energy"], 1e-30),
                               pass_quality=mse <= epsilon,
                               actual_refresh_fraction=c["refresh_rows"] / max(c["history_rows"], 1)))
        selected = min(row["rho"] for row in curves if row["pass_quality"])
        policies[version] = selected
        until = RELEASES[index + 1][1] if index + 1 < len(RELEASES) else END
        for uid, session in sessions.items():
            session.rho = selected
            # Replay real request opportunities, ignoring their labels/items. Only
            # the first read materializes codes; native writes include idle time.
            times = sorted({int(r["query_timestamp"]) for r in requests[uid] if r["version"] == version})
            for timestamp in times:
                session.prepare_read(timestamp)
            session.advance(until)
        releases[version] = dict(selected_rho=selected, epsilon=epsilon, curves=curves,
                                 shared_components_flops=dict(resource),
                                 shared_forward_flops=sum(resource.values()) + scales[version]["shared_forward_flops"])
        print(json.dumps(dict(stage="publication_selected", version=version, rho=selected,
                              epsilon=epsilon, errors=[(x["rho"],x["relative_correction_energy_error"]) for x in curves])), flush=True)
    result = dict(status="complete", role="publication_validation_only", policy=policies,
                  storage_mode=args.storage_mode,
                  tolerance=args.tolerance, fit_scales=scales, by_release=releases,
                  fit_uids=fit, validation_uids=validation, weights=weights,
                  diagnostics=diagnostics,
                  shared_validation_flops=sum(x["shared_forward_flops"] for x in releases.values()),
                  validation_history_resource_flops=sum(sum(ledger_flops(s.trajectory.ledger, models["v2"], "triton").values())
                                                       for s in sessions.values()),
                  elapsed_seconds=time.perf_counter()-began,
                  source_code_sha256=json.loads((args.output/"configuration.json").read_text())["source_code_sha256"])
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status="complete",policy=policies,shared_validation_flops=result["shared_validation_flops"])), flush=True)


@torch.no_grad()
def evaluate(args):
    from experiments.archive.partial_refresh.session import PartialRefreshSession
    from experiments.archive.partial_refresh.analyze import summarize
    began = time.perf_counter()
    strategy = json.loads(args.strategy.read_text()) if args.strategy else None
    if strategy:
        from experiments.archive.partial_refresh.round2.adaptive_session import session_factory
        Session = session_factory(strategy)
    else:
        Session = PartialRefreshSession
    panel, groups, requests = load_panel(args.quality_panel)
    all_selected = list(groups["evaluation"][:args.users])
    selected = all_selected[args.start_user:][args.shard_index::args.shards]
    if args.policy is not None:
        policy_record = json.loads(args.policy.read_text())
        if args.storage_mode != policy_record.get("storage_mode","masked"):
            raise ValueError("publication policy and serving code storage differ")
        policy = policy_record["policy"]
        if (set(policy_record["fit_uids"]) | set(policy_record["validation_uids"])) & set(all_selected):
            raise ValueError("publication calibration overlaps quality users")
    else:
        policy_record = None
        policy = {v: args.rho for v, _ in RELEASES}
    new_output(args, dict(stage="continuous_development", evaluation_uids=selected,
                         all_evaluation_uids=all_selected, policy=policy,
                         policy_source=str(args.policy) if args.policy else None,
                         frozen_reference=None if args.compute_controls else str(args.reference or REFERENCE), users=args.users,
                         frozen_reference_sha256=None if args.compute_controls else sha256(args.reference or REFERENCE),
                         start_user=args.start_user,storage_mode=args.storage_mode,
                         shard_index=args.shard_index, shards=args.shards,
                         exploration_strategy=strategy,
                         exploration_strategy_source=str(args.strategy) if args.strategy else None,
                         exploration_strategy_sha256=sha256(args.strategy) if args.strategy else None,
                         static_item_table_sha256=sha256(Path(strategy["static_item_file"])) if strategy and strategy.get("static_item_file") else None,
                         no_policy_selection_on_evaluation=True))
    write_json(args.output / "users.json", selected)
    device = setup_device(args.gpu, args.threads)
    models = load_models(panel, device)
    adapters, weights = load_adapters(args.calibrations, device)
    if policy_record and weights != policy_record["weights"]:
        raise ValueError("policy and serving weights differ")
    histories = load_history(args.quality_panel, requested_uids=selected)
    if args.compute_controls:
        references = None
    else:
        reference_table = pq.read_table(args.reference or REFERENCE, filters=[("uid", "in", selected)])
        references = {row["request_id"]: row for row in reference_table.to_pylist()}
        if "full_logit" in reference_table.column_names:
            control_summary = json.loads((args.reference.parent/"summary.json").read_text())
            if control_summary["weights"] != weights:
                raise RuntimeError("retained round1 full-code controls use different adapter weights")
            for uid in selected:
                for request in requests[uid]:
                    old = references[request["request_id"]]
                    if any(old[k] != request[k] for k in ("uid","version","label","query_timestamp")):
                        raise RuntimeError("retained round1 control identities differ")
                    old["lightweight_logit"] = old["full_logit"]
                    # Item identity is supplied by the same frozen panel; all
                    # stored request IDs/UIDs/labels/timestamps matched above.
                    old["item_idx"] = request["item_idx"]
    rows, units = [], []
    max_errors = Counter()
    for number, uid in enumerate(selected):
        session = Session(NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1]), kernel=args.kernel,storage_mode=args.storage_mode)
        if args.compute_controls:
            from hstu_kvcache.methods.cachetide.lifecycle.lifecycle import LifecycleSession
            full_session = PartialRefreshSession(NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1]), kernel=args.kernel,storage_mode=args.storage_mode)
            rebuild_session = LifecycleSession(NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1]))
        costs = {}
        for index, (version, at) in enumerate(RELEASES):
            before = Counter(session.costs)
            model, adapter = models[version], adapters[version]
            session.publish(model, int(version[1:]), at, adapter, rho=policy[version])
            if args.compute_controls:
                full_session.publish(model,int(version[1:]),at,adapter,rho=1.)
                rebuild_session.publish(model,int(version[1:]),at,adapter,policy="rebuild_on_first_request")
            snapshots = defaultdict(list)
            for request in requests[uid]:
                if request["version"] == version:
                    snapshots[int(request["query_timestamp"])].append(request)
            native_query_work = 0
            for timestamp, pending in sorted(snapshots.items()):
                session.prepare_read(timestamp)
                for offset in range(0, len(pending), 16):
                    batch = pending[offset:offset+16]
                    ids = torch.tensor([[r["item_idx"] for r in batch]], device=device)
                    actual = session.score(timestamp, ids)[0]
                    if args.compute_controls:
                        full_values = full_session.score(timestamp,ids)[0].cpu().tolist()
                        reuse_values = score(model,session.cache,ids,delta(session,timestamp))[0][0].cpu().tolist()
                        rebuild_values = rebuild_session.score(timestamp,ids)[0].cpu().tolist()
                        if args.verify and number < 2:
                            if session.trajectory.ledger != full_session.trajectory.ledger:
                                raise RuntimeError("partial/full native event ledgers differ")
                            if not torch.equal(session.cache.k,full_session.cache.k) or not torch.equal(session.cache.v,full_session.cache.v):
                                raise RuntimeError("partial/full native caches differ")
                            max_errors["full_partial_native_equality_checked_snapshots"] += 1
                    if not torch.isfinite(actual).all():
                        raise RuntimeError("nonfinite serving scores")
                    if args.verify and number < 2:
                        wanted = score(model, session.cache, ids, delta(session,timestamp),
                                       history_read_fn=session.read_callback(ids,force_torch=True))[0][0]
                        error = float((actual-wanted).abs().max())
                        max_errors["kernel_vs_torch"] = max(max_errors["kernel_vs_torch"],error)
                        if not torch.allclose(actual,wanted,atol=2e-5,rtol=2e-5):
                            raise RuntimeError("fused partial reader differs from Torch")
                    backend = "triton" if session.cache.seq_len == model.cfg.max_seq_len else "torch"
                    cm = CostModel(model.cfg.hidden_size,len(model.blocks),model.cfg.num_heads,backend)
                    native_query_work += eager_read(cm,session.cache.seq_len,queries=len(batch))
                    for j,(request,value) in enumerate(zip(batch,actual.cpu().tolist(),strict=True)):
                        old = (dict(request,reuse_logit=reuse_values[j],first_request_rebuild_logit=rebuild_values[j],
                                    lightweight_logit=full_values[j]) if args.compute_controls else references[request["request_id"]])
                        if any(old[k] != request[k] for k in ("uid","version","item_idx","label","query_timestamp")):
                            raise RuntimeError("matched request identity differs")
                        if strategy is None and all(float(policy[v]) == 1. for v,_ in RELEASES):
                            max_errors["all_refresh_saved_reference"] = max(max_errors["all_refresh_saved_reference"],abs(value-old["lightweight_logit"]))
                        rows.append(dict(**{k:old[k] for k in ("request_id","uid","version","label","reuse_logit","first_request_rebuild_logit")},
                                         full_logit=old["lightweight_logit"],partial_logit=value,rho=session.rho,
                                         history_length=session.cache.seq_len,query_timestamp=timestamp))
            until = RELEASES[index+1][1] if index+1 < len(RELEASES) else END
            session.advance(until)
            if args.compute_controls:
                full_session.advance(until)
                rebuild_session.advance(until)
            costs[version] = dict(counters={k:int(v-before.get(k,0)) for k,v in session.costs.items() if v!=before.get(k,0)},
                                 native_query_flops=native_query_work)
        units.append(dict(uid=uid,releases=costs,memory=dict(session.memory)))
        if (number+1) % 32 == 0:
            print(json.dumps(dict(stage="quality",users=number+1,requests=len(rows))),flush=True)
    if max_errors["all_refresh_saved_reference"] > 2e-5:
        raise RuntimeError("all-refresh partial prototype does not reproduce saved current method")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table,args.output/"scores.parquet")
    quality = summarize(rows,selected,repetitions=args.bootstrap,releases=[v for v,_ in RELEASES])
    totals = Counter()
    for unit in units:
        for release in unit["releases"].values():
            totals.update(release["counters"])
    result = dict(status="complete",role="chapter4_4_continuous_development",users=len(selected),requests=len(rows),
                  policy=policy,storage_mode=args.storage_mode,exploration_strategy=strategy,
                  shared_selection_costs=getattr(Session,"shared_selection_costs",{}),
                  shared_encoder_inventory=Session.encoder_inventory() if hasattr(Session,"encoder_inventory") else None,
                  weights=weights,quality=quality,units=units,checks=dict(max_errors),
                  actual_service_counters=dict(totals),
                  shared_publication_validation_flops=policy_record["shared_validation_flops"] if policy_record else None,
                  cost_scope="executed quality-user shapes; fixed30k complete cost reported separately",
                  elapsed_seconds=time.perf_counter()-began,
                  controls="new matched GPU controls" if args.compute_controls else "immutable retained matched controls",
                  source_code_sha256=json.loads((args.output/"configuration.json").read_text())["source_code_sha256"])
    write_json(args.output/"summary.json",result)
    print(json.dumps(dict(status="complete",users=len(selected),requests=len(rows),quality=quality.get("overall"))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("validate","evaluate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--calibration-panel",type=Path,default=CALIBRATION_PANEL)
    parser.add_argument("--quality-panel",type=Path,default=QUALITY_PANEL)
    parser.add_argument("--calibrations",type=Path,default=CALIBRATIONS)
    parser.add_argument("--fit-users",type=int,default=256)
    parser.add_argument("--validation-users",type=int,default=16)
    parser.add_argument("--tolerance",type=float,default=.01)
    parser.add_argument("--ratios",type=float,nargs="+",default=list(RATIOS))
    parser.add_argument("--scale-reference",type=Path,help="Reuse identical fit-only full-code correction energy; charge its preparation")
    parser.add_argument("--users",type=int,default=512)
    parser.add_argument("--start-user",type=int,default=0,help="Extend an existing prefix run without repeating its users")
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--policy",type=Path)
    parser.add_argument("--strategy",type=Path,help="Fixed round2 causal length/selection/encoder alternative; exploratory development")
    parser.add_argument("--reference",type=Path,help="Retained matched control scores for the selected frozen panel")
    parser.add_argument("--rho",type=float,default=1.)
    parser.add_argument("--kernel",choices=("torch","vector"),default="vector")
    parser.add_argument("--storage-mode",choices=("masked","raw"),default="masked")
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--compute-controls",action="store_true",help="New-user matched full-code, Reuse and once-Rebuild controls")
    parser.add_argument("--bootstrap",type=int,default=500)
    args = parser.parse_args()
    (validate if args.stage=="validate" else evaluate)(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Causal per-release user selection and persistent once-per-release rebuilding.

Selection uses only publication-time histories. The nominal fraction limits
predicted resident rebuilding cost, not the later measured request-time cost.
This small runner checks policy endpoints on an existing development panel;
it is neither a population quality experiment nor a loaded service benchmark.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import score as native_score
from hstu_kvcache.evaluation.trajectory import NativeTrajectory


def plan_release(histories, at, nominal_fraction, estimate_flops, *,
                 lookback_seconds=14 * 86400, max_length=1024):
    """Greedily fit whole users ranked by past activity / predicted cost.

    ``histories`` maps uid to timestamp/item/behavior columns. Items, labels,
    future request occurrence, and post-publication history are never read.
    Ties are resolved by uid. Oversized users are skipped, allowing later
    affordable users to fit. At 100% every resident is selected.
    """
    fraction = float(nominal_fraction)
    if not 0 <= fraction <= 1:
        raise ValueError("nominal_fraction must lie in [0, 1]")
    profiles = []
    for uid, history in histories.items():
        times = history[0]
        stop = int(np.searchsorted(times, at, side="left"))
        recent = stop - int(np.searchsorted(times, at-lookback_seconds, side="left"))
        length = min(stop, max_length)
        cost = int(estimate_flops(length))
        profiles.append(dict(uid=int(uid), visible_rows=stop, history_length=length,
            recent_behavior_rows=recent, estimated_rebuild_flops=cost,
            priority=recent / max(cost, 1)))
    profiles.sort(key=lambda row: (-row["priority"], row["uid"]))
    denominator = sum(row["estimated_rebuild_flops"] for row in profiles)
    limit = int(fraction * denominator)
    selected, used = [], 0
    for row in profiles:
        cost = row["estimated_rebuild_flops"]
        if fraction == 1 or (fraction > 0 and cost > 0 and used+cost <= limit):
            selected.append(row["uid"])
            used += cost
    return dict(at=int(at), nominal_fraction=fraction, selected_uids=selected,
        estimated_resident_rebuild_flops=denominator, estimated_budget_flops=limit,
        selected_estimated_flops=used, unallocated_estimated_flops=limit-used,
        lookback_seconds=lookback_seconds, profiles=profiles,
        budget_scope="Publication-time resident estimate; no request-time hard cap",
        policy="descending recent_behavior_rows / estimated_rebuild_flops, uid tie")


class BudgetedRebuildSession:
    """Native state plus a pending first-recommendation rebuild bit.

    No correction state or inherited-support flags are maintained. Each
    instance owns its trajectory; rebuilding changes all subsequent writes.
    """
    def __init__(self, trajectory, uid, history_read_fn=None):
        self.trajectory, self.uid = trajectory, int(uid)
        self.history_read_fn = history_read_fn
        self.pending_rebuild = False
        self.costs, self.query_ledger = Counter(), Counter()
        self.events = []

    @property
    def cache(self):
        return self.trajectory.cache

    def advance(self, at):
        self.trajectory.advance(at)

    def publish(self, model, version, at, selected):
        self.trajectory.switch(model, version, at)
        self.pending_rebuild = bool(selected)
        self.costs["publication_events"] += 1

    def prepare_read(self, at):
        self.advance(at)
        if self.pending_rebuild:
            self.trajectory.switch(self.trajectory.model, self.trajectory.version, at, rebuild=True)
            self.pending_rebuild = False
            self.costs["rebuild_events"] += 1
            self.events.append(dict(uid=self.uid, at=int(at), version=self.trajectory.version,
                                    history_length=self.cache.seq_len))

    @torch.no_grad()
    def score(self, at, candidates):
        self.prepare_read(at)
        self.query_ledger[self.trajectory.version, self.cache.seq_len, candidates.shape[1]] += 1
        self.costs["candidate_scores"] += candidates.numel()
        return score_trajectory(self.trajectory, at, candidates, self.history_read_fn)


@torch.no_grad()
def score_trajectory(trajectory, at, candidates, history_read_fn=None):
    delta = 0 if trajectory.last_timestamp is None else int(at)-trajectory.last_timestamp
    query_delta = torch.tensor([delta], device=candidates.device, dtype=torch.float32)
    return native_score(trajectory.model, trajectory.cache, candidates, query_delta,
                        history_read_fn=history_read_fn)[0]


def native_history_read(index, attention, q, k, v):
    """Existing fused reader's no-code specialization, without adapter tables."""
    del index
    if torch.is_grad_enabled() or not q.is_cuda:
        raise ValueError("native fused read requires CUDA inference")
    if (q.shape[1], q.shape[-1], attention.activation, attention.block_variant) != (6, 32, "elu_plus1", "legacy") or attention.position_bias is not None:
        raise ValueError("native fused read is scoped to the existing Medium model")
    if k.shape[1] == 0:
        return torch.zeros_like(q)
    from hstu_kvcache.methods.cachetide.kernels.reader import launch_read
    return launch_read(q, k, v, k[:, :0, :48], q, q, q, None, (0, 0, 0, 0),
                       code_count=0, scale=float(attention.scale))


def records(counter):
    return [dict(action=action, version=f"v{version}", before=before, width=width, count=count)
            for (action, version, before, width), count in sorted(counter.items())]


def analyze_saved(output, relay_reference=None):
    """Describe frozen policy points and optionally join retained relay scores.

    This analysis never changes user selection or chooses a budget point.
    Relay is a quality-only reference until its complete same-panel cost is
    separately measured; no old population cost ratio is transplanted here.
    """
    from collections import defaultdict
    import pyarrow.parquet as pq
    from hstu_kvcache.evaluation.binary_metrics import _roc_auc
    from hstu_kvcache.common.artifacts import sha256, write_json

    summary = json.loads((output / "summary.json").read_text())
    rows = pq.read_table(output / "scores.parquet").to_pylist()
    identity = ("uid", "version", "label", "item_idx", "query_timestamp")
    originals = {(r["protocol"], r["request_id"]): r for r in rows if r["method"] == "reuse"}
    reference_record = None
    if relay_reference is not None:
        run_config = json.loads((output / "configuration.json").read_text())
        reference_config_path = relay_reference.parent / "configuration.json"
        reference_config = json.loads(reference_config_path.read_text())
        registration_path = ROOT / run_config["configuration"]["selection"]
        registration = json.loads(registration_path.read_text())
        freeze_path = Path(registration["frozen_candidate"]) / "freeze.json"
        freeze = json.loads(freeze_path.read_text())
        panel_path = ROOT / run_config["configuration"]["panel"] / "binding.json"
        panel = json.loads(panel_path.read_text())
        if not (run_config["registration_sha256"] == reference_config["registration_sha256"]
                == sha256(registration_path)):
            raise ValueError("relay and baseline engineering registrations differ")
        if not (run_config["panel_sha256"] == registration["panel_sha256"]
                == freeze["panel"]["sha256"] == sha256(panel_path)):
            raise ValueError("relay and baseline panel bindings differ")
        if sha256(freeze_path) != registration["freeze_sha256"]:
            raise ValueError("retained relay freeze differs from registration")
        release_bindings = {}
        for version, binding in freeze["release_bindings"].items():
            weight = reference_config["calibration_weights"][version]
            publication_path = freeze_path.parent / "publication" / f"{version}.pt"
            publication_hash = sha256(publication_path)
            if not (sha256(Path(weight["path"])) == weight["sha256"] == binding["calibration_sha256"]
                    and binding["model_sha256"] == panel["models"][version]["checkpoint_sha256"]
                    and publication_hash == reference_config["publications"][version]["old_artifact_sha256"]):
                raise ValueError("retained relay model, weight or publication binding differs")
            release_bindings[version] = dict(model_sha256=binding["model_sha256"],
                calibration_path=weight["path"], calibration_sha256=weight["sha256"],
                publication_path=str(publication_path), publication_sha256=publication_hash)
        saved = pq.read_table(relay_reference).to_pylist()
        matched = {(r["protocol"], r["request_id"]): r for r in saved
                   if (r["protocol"], r["request_id"]) in originals}
        if set(matched) != set(originals):
            raise ValueError("retained relay reference does not cover every pilot request")
        for key, row in matched.items():
            if any(row[name] != originals[key][name] for name in identity):
                raise ValueError("relay reference changed a request identity or label")
            rows.append(dict(protocol=row["protocol"], request_id=row["request_id"],
                method="retained_relay_all_optimized", score=row["all_optimized"],
                **{name: row[name] for name in identity}))
        reference_record = dict(path=str(relay_reference), sha256=sha256(relay_reference),
            matched_requests=len(matched), identity_fields=list(identity),
            configuration_path=str(reference_config_path), configuration_sha256=sha256(reference_config_path),
            registration_path=str(registration_path), registration_sha256=sha256(registration_path),
            panel_path=str(panel_path), panel_sha256=sha256(panel_path),
            freeze_path=str(freeze_path), freeze_sha256=sha256(freeze_path),
            release_bindings=release_bindings, releases=run_config["releases"],
            model_check="Baseline load_models verified actual foundation checkpoint hashes against this panel; retained relay configuration matches frozen weights and publication artifacts",
            cost_scope="Quality-only matched reference; complete same-panel relay cost not supplied")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["protocol"], row["method"], row["version"]].append(row)
    quality = {}
    versions = ("v3", "v4", "v5")
    for protocol in summary["protocols"]:
        method_records = {}
        methods = sorted({m for p, m, _ in grouped if p == protocol})
        for method in methods:
            releases = {}
            for version in versions:
                sample = grouped[protocol, method, version]
                labels = np.asarray([r["label"] for r in sample])
                scores = np.asarray([r["score"] for r in sample])
                releases[version] = dict(auc=_roc_auc(labels, scores) if sample else None,
                    candidate_scores=len(sample), positive_labels=int(labels.sum()),
                    users=len({r["uid"] for r in sample}))
            aucs = [releases[v]["auc"] for v in versions]
            mean_auc = float(np.mean(aucs)) if all(v is not None for v in aucs) else None
            ledger = summary["protocols"][protocol]["methods"].get(method)
            method_records[method] = dict(by_release=releases, equal_release_mean_auc=mean_auc,
                nominal_fraction=None if ledger is None else ledger["nominal_fraction"],
                actual_rebuild_fraction=None if ledger is None else ledger["actual_rebuild_fraction"],
                rebuilt_user_releases=None if ledger is None else ledger["rebuilt_user_releases"])
        base = method_records["reuse"]["equal_release_mean_auc"]
        for record in method_records.values():
            mean = record["equal_release_mean_auc"]
            record["mean_auc_difference_from_reuse"] = None if base is None or mean is None else mean-base
        quality[protocol] = method_records
    result = dict(status="complete_descriptive_analysis", selected_uids=summary["selected_uids"],
        protocols=quality, retained_relay_reference=reference_record,
        analysis_source_sha256=sha256(Path(__file__)), scores_sha256=sha256(output / "scores.parquet"),
        scope="Existing stratified engineering development panel; no population inference, blind confirmation or policy selection",
        metric="AUC per release and unweighted mean of three release AUCs; absolute differences, no recovery-percent headline")
    write_json(output / "quality_curve.json", result)
    lines = ["# 整户预算重建：开发面板结果", "",
        f"沿用已登记的 {len(summary['selected_uids'])} 位工程用户；所有预算点预先固定，未根据本轮质量选择用户或策略。",
        "名义预算以发布前驻留历史估价为分母；实际比例以同一 trace 的首次推荐 Rebuild 执行量为分母。", "",
        "| 协议 | 方法 | 名义预算 | 实际重建占比 | V3 AUC | V4 AUC | V5 AUC | 等权平均 AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    for protocol, methods in quality.items():
        for method, record in methods.items():
            nominal, actual = record["nominal_fraction"], record["actual_rebuild_fraction"]
            ntext = "—" if nominal is None else f"{nominal*100:.0f}%"
            atext = "—" if actual is None else f"{actual*100:.3f}%"
            values = [fmt(record["by_release"][v]["auc"]) for v in versions]
            lines.append(f"| {protocol} | {method} | {ntext} | {atext} | {' | '.join(values)} | {fmt(record['equal_release_mean_auc'])} |")
    lines.extend(["", "## 预算使用与检查", ""])
    for protocol, record in summary["protocols"].items():
        lines.append(f"- {protocol}：{record['recommendation_rounds']} 轮、{record['candidate_scores']} 个评分。")
        for method, ledger in record["methods"].items():
            idle = sum(len(r["selected_without_recommendation"]) for r in ledger["by_release"].values())
            unused = sum(r["unallocated_estimated_flops"] for r in ledger["by_release"].values())
            lines.append(f"  - {method}：实际重建 {ledger['rebuilt_user_releases']} 个用户版本；选中但无推荐 {idle} 个；整户粒度剩余估价额度 {unused} FLOPs。")
    lines.extend(["", "0%／100% 的最终 K/V、游标和原生操作账与独立控制一致；逐批评分通过 3e-5 绝对／相对容差检查。",
        "未分配估价额度与选中用户未请求分别记录，不能用名义预算代替实际费用。完整原生追加账、查询逻辑 FLOPs 和重建事件见 summary.json。",
        "本表描述小型分层开发面板，不据此选政策参数或判断总体优劣。计时含检查与重复控制，不作为加载服务性能。", ""])
    if reference_record is not None:
        lines.extend([f"已逐项匹配保留的 relay32 输出：{relay_reference}。该行只比较同请求质量；未将其他人口上的成本百分比移植为本面板成本。", ""])
    (output / "summary.md").write_text("\n".join(lines))
    return result


@torch.inference_mode()
def run(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from hstu_kvcache.evaluation.release_context import END, RELEASES, ledger_flops, load_models, setup_device
    from hstu_kvcache.evaluation.panels import load_history, load_panel
    from hstu_kvcache.methods.cachetide.lifecycle.lifecycle import LifecycleSession
    from hstu_kvcache.evaluation.cachetide_workload import candidate_rows
    from hstu_kvcache.evaluation.cost import CostModel, eager_read
    from experiments.archive.system_study.events import build_trace
    from hstu_kvcache.common.artifacts import sha256, write_json

    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(f"use a fresh attempt directory: {args.output}")
    registration_path = ROOT / config["selection"]
    registration = json.loads(registration_path.read_text())
    selection_mode = config.get("selection_mode", "first_per_stratum")
    if selection_mode == "first_per_stratum":
        chosen_by_stratum = {}
        for row in registration["profiles"]:
            chosen_by_stratum.setdefault(row["stratum"], int(row["uid"]))
        selected = list(chosen_by_stratum.values())
    elif selection_mode == "all_registered":
        selected = list(map(int, registration["selected_uids"]))
    else:
        raise ValueError("selection_mode must use the existing engineering registration")
    if len(selected) != config.get("expected_users", 4) or len(set(selected)) != len(selected):
        raise ValueError("registered user count differs from the configured development panel")
    panel_path = ROOT / config["panel"]
    if sha256(panel_path / "binding.json") != registration["panel_sha256"]:
        raise ValueError("engineering registration and panel differ")
    panel, groups, feedback = load_panel(panel_path)
    if not set(selected).issubset(groups["evaluation"]):
        raise ValueError("baseline users must retain development evaluation role")
    histories = load_history(panel_path, requested_uids=selected).rows
    args.output.mkdir(parents=True)
    write_json(args.output / "configuration.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, configuration=config, selected_uids=selected,
        selection_rule=selection_mode,
        source_sha256=sha256(Path(__file__)), panel_sha256=sha256(panel_path / "binding.json"),
        registration_sha256=sha256(registration_path), config_sha256=sha256(args.config),
        releases=list(RELEASES), end_time=END, evidence_role=config["purpose"]))
    device = setup_device(args.gpu, threads=config.get("threads", 2))
    models = load_models(panel, device, backend="triton")
    cost = CostModel(models["v2"].cfg.hidden_size, len(models["v2"].blocks), models["v2"].cfg.num_heads, "triton")
    dense_cost = CostModel(cost.hidden_size, cost.num_layers, cost.num_heads, "torch")
    cap = models["v2"].cfg.max_seq_len
    estimate = lambda n: (cost if n >= cap else dense_cost).full_cache(n)
    fractions = config["nominal_fractions"]
    methods = {f"budget{round(100*f)}": f for f in fractions}
    if methods.get("budget0") != 0 or methods.get("budget100") != 1:
        raise ValueError("smoke must include both endpoint controls")
    plans = {version: {name: plan_release(histories, at, fraction, estimate, max_length=cap)
                       for name, fraction in methods.items()} for version, at in RELEASES}
    write_json(args.output / "release_plans.json", plans)
    print(json.dumps(dict(status="models_ready", selected_uids=selected, methods=methods)), flush=True)
    all_scores, summaries = [], {}
    started = time.perf_counter()
    for protocol in config["protocols"]:
        requests = ({u: candidate_rows(u, histories[u], stride=16, candidates=16,
                    known_vocab_size=panel["known_vocab_size"]) for u in selected}
                    if protocol == "behavior_stride16_candidates16" else {u: feedback[u] for u in selected})
        if protocol not in ("behavior_stride16_candidates16", "original_feedback"):
            raise ValueError("unknown smoke protocol")
        trace = build_trace(histories, requests, RELEASES, END)
        make = lambda uid: NativeTrajectory(models["v2"], 2, histories[uid], RELEASES[0][1])
        sessions = {name: {u: BudgetedRebuildSession(make(u), u, native_history_read) for u in selected}
                    for name in methods}
        reuse = {u: make(u) for u in selected}
        full = {u: LifecycleSession(make(u)) for u in selected}
        endpoint_error = {"zero_vs_reuse": 0., "all_vs_rebuild": 0.}
        different20, recommendation_rounds = 0, 0
        first_read_lengths, query_shapes = {}, Counter()
        for event in trace:
            if event.kind == "behavior":
                continue
            version, at = int(event.version[1:]), event.logical_time
            if event.kind == "publication":
                for name, users in sessions.items():
                    chosen = set(plans[event.version][name]["selected_uids"])
                    for uid, session in users.items():
                        session.publish(models[event.version], version, at, uid in chosen)
                for uid in selected:
                    reuse[uid].switch(models[event.version], version, at)
                    full[uid].publish(models[event.version], version, at, None,
                                      policy="rebuild_on_first_request")
                print(json.dumps(dict(status="publication", protocol=protocol, version=event.version,
                    selected={name: plans[event.version][name]["selected_uids"] for name in methods})), flush=True)
            elif event.kind == "drain":
                for users in sessions.values():
                    for session in users.values():
                        session.advance(at)
                for uid in selected:
                    reuse[uid].advance(at)
                    full[uid].advance(at)
            elif event.kind == "recommendation":
                uid = event.uid
                recommendation_rounds += 1
                reuse[uid].advance(at)
                full[uid].prepare_read(at)
                first_read_lengths.setdefault((event.version, uid), reuse[uid].cache.seq_len)
                different_round = False
                query_batch = config.get("query_batch", 16)
                for start in range(0, len(event.requests), query_batch):
                    batch = event.requests[start:start+query_batch]
                    candidates = torch.tensor([[row["item_idx"] for row in batch]], device=device)
                    # Original Torch reader is an independent numerical oracle
                    # for the fused no-code read used by policy sessions.
                    values = {"reuse": score_trajectory(reuse[uid], at, candidates),
                              "rebuild": score_trajectory(full[uid].trajectory, at, candidates)}
                    values.update({name: users[uid].score(at, candidates) for name, users in sessions.items()})
                    for name, control, key in (("budget0", "reuse", "zero_vs_reuse"), ("budget100", "rebuild", "all_vs_rebuild")):
                        torch.testing.assert_close(values[name], values[control], rtol=3e-5, atol=3e-5)
                        endpoint_error[key] = max(endpoint_error[key], float((values[name]-values[control]).abs().max()))
                    different_round |= not torch.equal(values["budget20"], values["budget0"])
                    query_shapes[version, reuse[uid].cache.seq_len, len(batch)] += 1
                    for method, logits in values.items():
                        for row, value in zip(batch, logits[0].cpu().tolist(), strict=True):
                            all_scores.append(dict(protocol=protocol, method=method, score=value,
                                round_id=row.get("round_id"),
                                **{key: row[key] for key in ("request_id", "uid", "version", "label", "item_idx", "query_timestamp")}))
                different20 += int(different_round)
        for uid in selected:
            for endpoint, reference in ((sessions["budget0"][uid].trajectory, reuse[uid]),
                                         (sessions["budget100"][uid].trajectory, full[uid].trajectory)):
                if endpoint.cursor != reference.cursor or endpoint.ledger != reference.ledger:
                    raise AssertionError("endpoint cursor or native operation ledger differs")
                torch.testing.assert_close(endpoint.cache.k, reference.cache.k, rtol=0, atol=0)
                torch.testing.assert_close(endpoint.cache.v, reference.cache.v, rtol=0, atol=0)
        full_ledger = sum((session.trajectory.ledger for session in full.values()), Counter())
        full_flops = ledger_flops(full_ledger, models["v2"], "triton")
        denominator = sum(value for key, value in full_flops.items() if key.endswith("/rebuild"))
        query_flops = sum(count*eager_read(cost, length, queries=width)
                          for (_, length, width), count in query_shapes.items())
        method_rows = {}
        for name, users in sessions.items():
            ledger = sum((session.trajectory.ledger for session in users.values()), Counter())
            flops = ledger_flops(ledger, models["v2"], "triton")
            actual = sum(value for key, value in flops.items() if key.endswith("/rebuild"))
            rebuilds = [row for session in users.values() for row in session.events]
            for row in rebuilds:
                row["actual_rebuild_flops"] = estimate(row["history_length"])
            method_rows[name] = dict(nominal_fraction=methods[name], native_ledger=records(ledger),
                native_operation_flops=flops, common_logical_query_flops=query_flops,
                actual_rebuild_flops=actual, same_trace_actual_full_rebuild_flops=denominator,
                actual_rebuild_fraction=actual/denominator if denominator else None,
                rebuilt_user_releases=len(rebuilds), rebuild_events=rebuilds,
                by_release={v: dict(selected_uids=plans[v][name]["selected_uids"],
                    selected_without_recommendation=[u for u in plans[v][name]["selected_uids"] if (v, u) not in first_read_lengths],
                    unallocated_estimated_flops=plans[v][name]["unallocated_estimated_flops"],
                    selected_estimated_flops=plans[v][name]["selected_estimated_flops"],
                    actual_rebuild_flops=flops.get(f"{v}/rebuild", 0)) for v, _ in RELEASES},
                final_state={str(u): dict(cursor=s.trajectory.cursor, rows=s.cache.seq_len,
                    producers={str(k): int(v) for k, v in Counter(s.trajectory.producers).items()}) for u, s in users.items()})
        summaries[protocol] = dict(methods=method_rows, endpoint_max_absolute_logit_error=endpoint_error,
            endpoint_final_kv_and_ledgers_exact=True, budget20_rounds_differing_from_fused_budget0=different20,
            difference_scope="Same fused reader; output difference is not a quality-gain measure",
            recommendation_rounds=recommendation_rounds, query_batches=sum(query_shapes.values()),
            candidate_scores=sum(count*width for (_, _, width), count in query_shapes.items()),
            full_control_native_ledger=records(full_ledger),
            query_shapes=[dict(version=f"v{v}", history_length=n, candidates=q, count=count) for (v, n, q), count in sorted(query_shapes.items())])
        write_json(args.output / f"{protocol}.json", summaries[protocol])
        print(json.dumps(dict(status="protocol_complete", protocol=protocol, endpoint_error=endpoint_error,
                              actual_budget20_fraction=method_rows["budget20"]["actual_rebuild_fraction"])), flush=True)
        del sessions, reuse, full
    torch.cuda.synchronize(device)
    pq.write_table(pa.Table.from_pylist(all_scores), args.output / "scores.parquet")
    result = dict(status="complete", selected_uids=selected, protocols=summaries,
        elapsed_seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        checks=dict(publication_only_selection=True, independent_persistent_trajectories=True,
                    zero_equals_reuse=True, all_equals_first_recommendation_rebuild=True),
        scope=config["purpose"],
        arithmetic="Rebuild/append charge existing executed backend shapes; query FLOPs report common logical arithmetic",
        budget="Nominal fraction uses pre-publication resident estimate; actual fraction uses same-trace executed full rebuild; no runtime hard cap")
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status="complete", output=str(args.output), seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--relay-reference", type=Path)
    args = parser.parse_args()
    if args.analyze_only:
        result = analyze_saved(args.output, args.relay_reference)
        print(json.dumps(dict(status=result["status"], output=str(args.output))))
    else:
        run(args)

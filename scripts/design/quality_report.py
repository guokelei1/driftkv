#!/usr/bin/env python3
"""Paired development quality and cost accounting from retained raw results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hstu_kvcache.evaluation.binary_metrics import binary_metrics, sigmoid

ROOT = Path(__file__).resolve().parents[2]/"results/design"
NAMES = ("exact", "reuse", "learned")


def quality(rows):
    labels = rows.label.to_numpy()
    metrics = {name: binary_metrics(labels, rows[name].to_numpy()) for name in NAMES}
    auc = {name: metrics[name]["ROC_AUC"] for name in NAMES}
    full_gap = None if auc["exact"] is None else auc["exact"]-auc["reuse"]
    gain = None if auc["learned"] is None else auc["learned"]-auc["reuse"]
    return dict(requests=len(rows), users=int(rows.uid.nunique()), positives=int(labels.sum()),
        metrics=metrics, exact_minus_reuse_auc=full_gap, learned_minus_reuse_auc=gain,
        auc_gap_recovery=gain/full_gap if full_gap is not None and full_gap>1e-4 else None,
        ratio_reporting_rule="only if Exact-Reuse AUC > 0.0001; never a serving gate",
        probability_gap_to_exact={name:float(np.abs(
            1/(1+np.exp(-rows[name].to_numpy()))-1/(1+np.exp(-rows.exact.to_numpy()))).mean())
            for name in ("reuse","learned")})


def user_bootstrap(rows, repetitions):
    rng = np.random.default_rng(17)
    uids, inverse = np.unique(rows.uid.to_numpy(), return_inverse=True)
    labels = rows.label.to_numpy()
    ranked = {}
    for name in NAMES:
        # Rank the same FP64 probabilities used by binary_metrics. Counts at
        # tied scores give exactly the AUC of explicitly repeated UID groups.
        probabilities = sigmoid(rows[name].to_numpy())
        order = np.argsort(probabilities, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(probabilities[order]) != 0) + 1]
        ranked[name] = (inverse[order], labels[order], starts)
    samples = []
    for _ in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, len(uids), size=len(uids)), minlength=len(uids))
        auc = {}
        for name, (users, targets, starts) in ranked.items():
            weight = multiplicity[users]
            positive = np.add.reduceat(weight * targets, starts).astype(np.float64)
            negative = np.add.reduceat(weight * (1-targets), starts).astype(np.float64)
            denominator = positive.sum() * negative.sum()
            auc[name] = (float(np.dot(positive, np.cumsum(negative)-.5*negative)/denominator)
                         if denominator else None)
        if auc["exact"] is not None:
            samples.append([auc["exact"]-auc["reuse"],auc["learned"]-auc["reuse"],auc["learned"]-auc["exact"]])
    interval = np.quantile(samples,[.025,.975],axis=0).T.tolist() if samples else [None]*3
    return dict(repetitions=repetitions,valid_samples=len(samples),seed=17,
        implementation="fixed probability tie groups with sampled UID multiplicities; equivalent to explicit row repetition",
        scope="paired observed-feedback UID sampling conditional on this one backbone seed; not training-seed replication",
        auc_difference_95ci=dict(zip(("exact_minus_reuse","learned_minus_reuse","learned_minus_exact"),interval,strict=True)))


def main(run_id, repetitions):
    path = ROOT/run_id
    summary = json.loads((path/"summary.json").read_text())
    assert summary["status"] == "development_complete"
    config = json.loads((path/"configuration.json").read_text())
    raw_path = path/"quality_raw.parquet"
    raw = pd.read_parquet(raw_path)
    assert not raw.duplicated(["target","request_id"]).any()
    rows = []
    for target,group in raw.groupby("target",sort=True):
        row = dict(target=int(target),all_requests=quality(group),bootstrap=user_bootstrap(group,repetitions))
        row["cache_action"] = "native_no_op" if target in config.get("no_op_targets",[]) else "adaptation"
        if row["cache_action"] == "native_no_op":
            assert group.learned.equals(group.reuse) and not group.correction_active.any()
        row["release_scope"] = "E14_partial diagnostic, no serving admission" if target == 5 and config.get("v5_partial_tail") else "development diagnostic"
        row["coverage"] = dict(producer_supported_fraction=float(group.covered.mean()),
            correction_active_fraction=float(group.correction_active.mean()),
            no_old_state_requests=int((group.old_events==0).sum()))
        row["old_state_strata"] = {name:quality(subset) for name,subset in (
            ("none",group[group.old_events==0]),("1_to_256",group[group.old_events.between(1,256)]),
            ("257_to_768",group[group.old_events.between(257,768)]),
            ("769_to_1024",group[group.old_events.between(769,1024)])) if len(subset)}
        if config.get("write_mode") == "reuse" and "writes_since_release" in group:
            row["native_clearance_strata"] = {
                str(int(layers)):dict(quality(subset),
                    min_writes=int(subset.writes_since_release.min()),
                    max_writes=int(subset.writes_since_release.max()),
                    correction_active_fraction=float(subset.correction_active.mean()))
                for layers,subset in group.groupby("cleared_layers",sort=True)}
        rows.append(row)
    ref = config.get("calibration_reference")
    calibration = json.loads((ROOT/ref["run"]/"summary.json").read_text()) if ref else summary
    cl, ledger = calibration["ledger_seconds"], summary["ledger_seconds"]
    teacher_and_fit = sum(cl.get(key,0) for key in ("teacher_build","teacher_query","teacher_replay","translator_fit"))
    method_service = sum(ledger.get(key,0) for key in ("service_learned_append","service_learned_read","chain_release_translation"))
    exact_service = sum(ledger.get(key,0) for key in ("service_exact_append","service_exact_read","chain_exact_release"))
    report = dict(run=run_id,configuration_sha256=hashlib.sha256((path/"configuration.json").read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),confirmation_read=False,backbone_seed_repeats=1,
        quality=rows,elapsed_seconds=summary["elapsed_seconds"],
        adaptation_targets=config.get("adaptation_targets",list(range(1,config["targets"]+1))),
        no_op_targets=config.get("no_op_targets",[]),
        selected_development_users=len(config["development_uids"]),
        users_with_observed_feedback=int(raw.uid.nunique()),
        cost=dict(calibration_reference=ref["run"] if ref else run_id,
            original_teacher_and_fit_seconds=teacher_and_fit,
            original_release_selection_seconds=cl.get("release_selection",0),
            original_calibration_query_selection_seconds=cl.get("calibration_query_selection",0),
            original_calibration_lineage_seconds=cl.get("calibration_lineage_replay",0)+cl.get("calibration_lifetime_replay",0),
            original_calibration_source_build_seconds=sum(cl.get(key,0) for key in
                ("calibration_source_backfill","calibration_adjacent_source","calibration_lifetime_source")),
            measured_initial_summary_backfill_seconds=ledger.get("chain_initial_summary_backfill"),
            measured_inference_compilation_and_warmup_seconds=ledger.get("inference_compilation_and_warmup",0),
            measured_read_seconds={name:ledger.get("service_"+name+"_read",0) for name in NAMES},
            measured_append_seconds={name:ledger.get("service_"+name+"_append",0) for name in NAMES},
            measured_publication_seconds=dict(method=ledger.get("chain_release_translation",0),
                                              exact=ledger.get("chain_exact_release",0)),
            publication_backend="main runner uses eager Exact prefix reconstruction; compile_reads changes CC kernels only",
            measured_method_service_seconds=method_service,measured_exact_service_seconds=exact_service,
            service_ratio_excluding_calibration_and_initial_backfill=method_service/exact_service,
            scope="separate actual execution costs; no population extrapolation or 20-percent qualification"))
    out = ROOT/"analysis"
    out.mkdir(exist_ok=True)
    (out/f"{run_id}_quality.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    text = [f"# {run_id} 开发质量", "", "固定开发用户、单个backbone seed；未读取确认。",
        "UID bootstrap只描述本seed下有真实反馈用户的抽样不确定性，不构成训练seed重复。", "",
        "| 目标 | 请求/用户 | Exact AUC | Reuse AUC | 方法 AUC | Exact−Reuse | 方法恢复 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        q = row["all_requests"]
        values = [q['metrics'][name]['ROC_AUC'] for name in NAMES]+[q['exact_minus_reuse_auc'],q['auc_gap_recovery']]
        formatted = [f"{value:.6f}" if value is not None else "未定义/小gap" for value in values]
        text.append(f"| M{row['target']} | {q['requests']}/{q['users']} | "+" | ".join(formatted)+" |")
    if config.get("v5_partial_tail"):
        text += ["", "M5为原E14_partial诊断尾段，serving_admission仍为false。"]
    if config.get("no_op_targets"):
        names=", ".join(f"M{t}" for t in config["no_op_targets"])
        text += ["",f"{names}按用户指定范围执行native No-op，保留真实模型/缓存写入；该行仅为边界记录，不计入本轮适配优化目标。"]
    text += ["", "完整绝对指标、配对区间、全部旧状态分层和成本分项见同名JSON。",
        "所有请求计入主结果；No-op和未覆盖请求不移除。恢复比例仅在正AUC gap超过0.0001时显示，",
        "该显示规则不是上线或用户调度门槛。全局教师、拟合、校准谱系和初次backfill单列，",
        "服务比率不含这些准备成本，不能拿它声称完整方法已达到20%成本目标。", ""]
    (out/f"{run_id}_quality.md").write_text("\n".join(text))
    print("\n".join(text[:10+len(rows)]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--bootstrap",type=int,default=1000)
    args = parser.parse_args()
    main(args.run_id,args.bootstrap)

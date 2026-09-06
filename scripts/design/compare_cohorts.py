#!/usr/bin/env python3
"""Compare completed matched cohorts; keep every release and history stratum."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from design.data import DATASET, ROOT
from design.quality_report import quality, user_bootstrap
from design.run import write_json


def preparation(config):
    path = ROOT / "results/design" / config["calibration_reference"]["run"] / "summary.json"
    ledger = json.loads(path.read_text())["ledger_seconds"]
    return sum(ledger.get(k, 0) for k in ("teacher_build", "teacher_query", "teacher_replay", "translator_fit",
        "calibration_source_backfill", "calibration_adjacent_source", "calibration_lifetime_source",
        "calibration_lifetime_replay", "calibration_lineage_replay", "release_selection", "calibration_query_selection"))


def main(args):
    paths = {name:ROOT / "results/design" / run for name, run in (("control", args.control), ("adaptive", args.adaptive))}
    configs = {name:json.loads((path / "configuration.json").read_text()) for name, path in paths.items()}
    summaries = {name:json.loads((path / "summary.json").read_text()) for name, path in paths.items()}
    assert all(s["status"] == "development_complete" for s in summaries.values())
    assert configs["control"]["development_uids"] == configs["adaptive"]["development_uids"]
    keys = ["target", "uid", "timestamp", "request_id"]
    raw = {name:pd.read_parquet(path / "quality_raw.parquet").sort_values(keys).reset_index(drop=True)
           for name, path in paths.items()}
    control, adaptive = raw["control"], raw["adaptive"]
    metadata = keys + ["label", "old_events", "writes_since_release", "cleared_layers"]
    assert control[metadata].equals(adaptive[metadata])
    differences = {name:float((control[name]-adaptive[name]).abs().max()) for name in ("exact", "reuse")}
    assert max(differences.values()) <= 2e-5
    users = pq.read_table(DATASET.parent / "users.parquet", columns=["uid", "n_theta0"]).to_pandas()
    history_counts = users.set_index("uid").n_theta0
    strata = np.searchsorted([256, 1024, 4096], adaptive.uid.map(history_counts), side="right")
    rows = []
    for target in sorted(adaptive.target.unique()):
        mask = adaptive.target == target
        a, c = adaptive[mask], control[mask]
        # Reuse the checked paired-bootstrap implementation with explicit
        # aliases: its Exact-Reuse difference is adaptive-method minus control.
        paired = pd.DataFrame(dict(uid=a.uid, label=a.label, exact=a.learned, reuse=c.learned, learned=a.exact))
        interval = user_bootstrap(paired, 1000)["auc_difference_95ci"]["exact_minus_reuse"]
        q, baseline = quality(a), quality(c)
        auc = q["metrics"]["learned"]["ROC_AUC"]
        control_auc = baseline["metrics"]["learned"]["ROC_AUC"]
        rows.append(dict(target=int(target), adaptive=q, control=baseline,
            adaptive_action="native_no_op" if target in configs["adaptive"].get("no_op_targets",[]) else "adaptation",
            adaptive_minus_control_auc=auc-control_auc if auc is not None else None,
            paired_uid_bootstrap_95ci=interval,
            history_strata={name:dict(adaptive=quality(adaptive[mask & (strata == i)]),
                                      control=quality(control[mask & (strata == i)]))
                for i, name in enumerate(("1_to_255", "256_to_1023", "1024_to_4095", "4096_plus"))
                if (mask & (strata == i)).any()},
            scope="M5 E14_partial diagnostic, no serving admission" if target == 5 else "development, admitted Full-only edge"))
    report = dict(control=args.control, adaptive=args.adaptive, selected_users=len(configs["adaptive"]["development_uids"]),
        users_with_feedback=int(adaptive.uid.nunique()), requests=len(adaptive), quality=rows,
        maximum_baseline_logit_difference=differences, metadata_equal=True,
        raw_sha256={name:hashlib.sha256((path / "quality_raw.parquet").read_bytes()).hexdigest() for name, path in paths.items()},
        preparation_seconds={name:preparation(config) for name, config in configs.items()},
        elapsed_seconds={name:summary["elapsed_seconds"] for name, summary in summaries.items()},
        confirmation_read=False, training_seed_repeats=1,
        adaptation_targets=configs["adaptive"].get("adaptation_targets",[1,2,3,4,5]),
        no_op_targets=configs["adaptive"].get("no_op_targets",[]),
        scope="matched development sample, same five-release lineage; no cross-edge recovery average or population cost qualification")
    out = ROOT / "results/design/analysis" / f"{args.adaptive}_vs_{args.control}"
    write_json(out.with_suffix(".json"), report)
    text = [f"# {args.adaptive} 与固定对照", "",
        f"共同选择{report['selected_users']}人；{report['users_with_feedback']}人有真实反馈，共{len(adaptive)}条请求。",
        "每个用户完整经历五次更新；单一训练seed，独立确认未读。", "",
        f"| 模型 | 请求/反馈用户 | Exact AUC | Reuse AUC | 固定方案 AUC | {args.candidate_label} AUC | {args.candidate_label}−固定 95% CI |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        q = row["adaptive"]
        values = [q["metrics"][n]["ROC_AUC"] for n in ("exact", "reuse")]
        values += [row["control"]["metrics"]["learned"]["ROC_AUC"], q["metrics"]["learned"]["ROC_AUC"]]
        ci = row["paired_uid_bootstrap_95ci"]
        interval = f"[{ci[0]:.6f}, {ci[1]:.6f}]" if ci is not None else "未定义"
        text.append(f"| M{row['target']} | {q['requests']}/{q['users']} | "
                    + " | ".join(f"{v:.6f}" if v is not None else "未定义" for v in values)
                    + f" | {interval} |")
    if report["no_op_targets"]:
        names=", ".join(f"M{t}" for t in report["no_op_targets"])
        text += ["",f"{names}为用户指定的native No-op阶段，仅单列边界记录，不将它的差值算作适配方法收益。"]
    text += ["", "M5保留不完整E14诊断边界。区间为本seed下配对UID抽样，不增加模型训练重复数。",
        "各历史长度分层、PR-AUC/log-loss/Brier及逐段恢复保存在JSON；不以分块AUC均值代替全体AUC。",
        f"必要准备（不含模型/数据IO）：固定{report['preparation_seconds']['control']:.2f}s，"
        f"{args.candidate_label}{report['preparation_seconds']['adaptive']:.2f}s；这不是完整人口成本达标结论。", ""]
    out.with_suffix(".md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--adaptive", required=True)
    parser.add_argument("--candidate-label", default="自适应")
    main(parser.parse_args())

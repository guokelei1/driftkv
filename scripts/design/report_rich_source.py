#!/usr/bin/env python3
"""Paired assessment of the one bounded richer-source fallback."""

import argparse
import json
from pathlib import Path

import torch
from design.report_mechanism import interval, load_outputs

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    source = ROOT / "results/design" / args.run_id
    _, users, summary = load_outputs(source)
    _, control, _ = load_outputs(ROOT / "results/design/mechanism_aggregate_factorial192_01")
    rows = []
    for (target, group, panel), frame in users.groupby(["target", "group", "panel"]):
        wide = frame.pivot(index="uid", columns="method", values="logit_mse")
        old = control[(control.target == target) & (control.group == group) & (control.panel == panel)]
        old = old.pivot(index="uid", columns="method", values="logit_mse").reindex(wide.index)
        torch.testing.assert_close(torch.tensor(wide.reuse.to_numpy()), torch.tensor(old.reuse.to_numpy()), atol=0, rtol=0)
        rows.append(dict(target=int(target), group=group, panel=panel, means=wide.mean().to_dict(),
            medians=wide.median().to_dict(),
            functional_minus_same_capacity_mean=interval(wide.functional_response-wide.mean_response),
            functional_minus_reuse=interval(wide.functional_response-wide.reuse),
            functional_minus_small_functional=interval(wide.functional_response-old.functional_response),
            mean_capacity128_minus_capacity32=interval(wide.mean_response-old.mean_response),
            maximum_uid_error_fraction={m: float(wide[m].max()/max(wide[m].sum(), 1e-30)) for m in wide}))
    p = torch.load(source / "fixed_m0_probes.pt", weights_only=True).double()
    singular = torch.linalg.svdvals(p)
    probability = singular.square()/singular.square().sum(-1, keepdim=True)
    effective = (-(probability*probability.clamp_min(1e-30).log()).sum(-1)).exp()
    storage = json.loads((source / "cost_m5.json").read_text())
    state = torch.load(source / "source_projection_functional_m5.pt", weights_only=True)
    storage["m5_functional_projection_tensor_bytes"] = sum(v.numel()*v.element_size() for v in state.values())
    view = torch.load(source / "translator_functional_response_m5.pt", weights_only=True)
    storage["m5_shared_layer_tensor_bytes"] = sum(v.numel()*v.element_size() for layer in view for v in layer.values())
    report = dict(run=args.run_id, rows=rows, elapsed_seconds=summary["elapsed_seconds"],
        peak_allocated_mib=summary["peak_allocated_mib"], ledger_seconds=summary["ledger_seconds"], storage=storage,
        probe_geometry=dict(numeric_rank_median=float((singular > singular[..., :1]*1e-6).sum(-1).double().median()),
            uncentered_energy_rank_median=float(effective.median()),
            scope="fixed actual M0 query bank; not arbitrary/full-KV encoding or proof that all source information was accessible"),
        confirmation_read=False, checks="same retained baseline UID error records as aggregate factorial; no M2 target",
        scope="single richer source/encoder budget, matched mean-capacity control; paired1000 UID bootstrap, seed17; one backbone seed")
    out = ROOT / "results/design/analysis" / args.report_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 一次丰富源预算诊断", "", "32固定M0探针、PCA128、聚合响应监督，对照同容量均值输入。",
             "同64拟合/128已使用诊断UID，64配对query；M2不参与，M5 E14_partial，确认未读。", "",
             "| 目标 | Reuse | 均值/128 | 功能/128 | 功能−均值 [95% UID区间] |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        if row["group"] != "diagnostic_uid" or row["panel"] != "held64":
            continue
        m, effect = row["means"], row["functional_minus_same_capacity_mean"]
        lines.append(f"| M{row['target']} | {m['reuse']:.6g} | {m['mean_response']:.6g} | {m['functional_response']:.6g} | "
                     f"{effect['mean']:.5g} [{effect['ci95'][0]:.5g}, {effect['ci95'][1]:.5g}] |")
    lines += ["", "表为诊断UID留出query的UID等权logit MSE，不是AUC。拟合组、median、尾部、与小预算",
              "对照的区间及账本见JSON。全部用户保留；不因极端误差删用户。", "",
              f"功能辅助状态{storage['functional_auxiliary_state_bytes']} bytes/用户，不含原均值writer及query view；",
              f"M5源投影与共享层参数共{storage['m5_functional_projection_tensor_bytes']+storage['m5_shared_layer_tensor_bytes']} bytes。",
              "源扫描按诊断计费，未作人口I/O、在线增量维护或就绪等待验证。", "",
              "探针数增加不等于独立观测维度同比增加：此M0探针集合数值秩中位数31，但未中心化能量秩",
              f"中位数仅{float(effective.median()):.3f}。这是特定源编码对照，不能据负结果宣称完整KV不含所需信息。", ""]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="mechanism_rich_source192_01")
    parser.add_argument("--report-id", required=True)
    main(parser.parse_args())

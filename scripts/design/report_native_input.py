#!/usr/bin/env python3
"""Matched input prototype and conditional free16 diagnostic, with actual baselines."""

import argparse
import json

import numpy as np
import pandas as pd
from design.data import ROOT
from design.report_decoder_closure import interval


def main(cli):
    out = ROOT / "results/design/analysis" / cli.report_id
    out.mkdir(parents=True, exist_ok=False)
    prototype = ROOT / "results/design" / cli.prototype_run
    free16 = ROOT / "results/design" / cli.free16_run
    reference = ROOT / "results/design/mechanism_decoder_closure192_01"
    for folder in (prototype, free16, reference):
        assert json.loads((folder / "summary.json").read_text())["status"] == "complete"
    frames = []
    for t in (1, 3, 4, 5):
        current = pd.read_parquet(prototype / f"outputs_m{t}.parquet")
        frozen = pd.read_parquet(reference / f"outputs_m{t}.parquet")
        for method in ("reuse", "shared15"):
            pair = current[current.method == method].merge(frozen[(frozen.method == method) & (frozen.path == "actual")],
                on=["scene", "uid", "panel"], suffixes=("_new", "_old"), validate="one_to_one")
            assert len(pair) == len(current[current.method == method])
            np.testing.assert_allclose(pair.logit_mse_new, pair.logit_mse_old, atol=1e-7, rtol=3e-4)
        more = pd.read_parquet(free16 / f"outputs_m{t}.parquet")
        baseline = frozen[(frozen.path.isin(["actual", "free64"])) & (~frozen.method.isin(["reuse", "shared15"]))]
        oracle = pd.read_parquet(ROOT / f"results/design/mechanism_query_holdout192_01/outputs_m{t}.parquet")
        oracle = oracle[oracle.method == "oracle64"].copy()
        oracle["path"] = "oracle"
        oracle = oracle.merge(current[["scene", "uid", "state_group"]].drop_duplicates(), on=["scene", "uid"], validate="many_to_one")
        frames.extend([current, more, baseline, oracle])
    frame = pd.concat(frames)
    frame = frame[["target", "group", "state_group", "panel", "method", "path", "uid", "scene", "logit_mse"]]
    assert np.isfinite(frame.logit_mse).all(), "nonfinite rows must not be silently dropped by aggregation"
    assert not frame.duplicated(["target", "uid", "scene", "panel", "method", "path"]).any()
    uid = frame.groupby(["target", "group", "panel", "method", "path", "uid"], as_index=False).logit_mse.mean()
    rows, contrasts = [], []
    for keys, f in uid.groupby(["target", "group", "panel"]):
        reuse = f[(f.method == "reuse") & (f.path == "actual")].set_index("uid").logit_mse
        shared = f[(f.method == "shared15") & (f.path == "actual")].set_index("uid").logit_mse
        summary = f[(f.method == "summary_input") & (f.path == "actual")].set_index("uid").logit_mse
        for (method, path), candidate in f.groupby(["method", "path"]):
            value = candidate.set_index("uid").logit_mse.reindex(reuse.index)
            rows.append(dict(target=int(keys[0]), group=keys[1], panel=keys[2], method=method, path=path,
                mse=float(value.mean()), median=float(value.median()), reuse_ratio=float(value.mean()/reuse.mean()),
                better_than_reuse_fraction=float((value < reuse).mean()), max_uid_error_share=float(value.max()/value.sum()),
                difference_reuse_ci=interval(value-reuse), difference_shared15_ci=interval(value-shared)))
            if method == "native_input":
                difference = value-summary
                contrasts.append(dict(target=int(keys[0]), group=keys[1], panel=keys[2],
                    native_minus_summary=float(difference.mean()), interval=interval(difference),
                    native_better_uid_fraction=float((difference < 0).mean()), native_median=float(value.median()),
                    summary_median=float(summary.median())))
    strata = (frame.groupby(["target", "group", "state_group", "panel", "method", "path", "uid"]).logit_mse.mean()
              .groupby(["target", "group", "state_group", "panel", "method", "path"]).mean().reset_index().to_dict("records"))
    outlier = frame[(frame.target == 3) & (frame.uid == 988060)].groupby(["panel", "method", "path"], as_index=False).logit_mse.mean().to_dict("records")
    costs = {str(t): json.loads((prototype / f"cost_m{t}.json").read_text()) for t in (1, 3, 4, 5)}
    result = dict(rows=rows, matched_input_contrasts=contrasts, state_strata=strata, retained_uid988060=outlier,
        prototype_costs=costs, prototype_run=json.loads((prototype / "summary.json").read_text()),
        free16_run=json.loads((free16 / "summary.json").read_text()),
        scope="single matched input prototype; unchanged 64 fitting and reused128 diagnostic UIDs; no M2 target, M5 E14_partial, no confirmation or population run")
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    text = ["# 本次native读取信息：唯一匹配原型", "", "同面板logit MSE；场景平均后UID等权，非AUC恢复率。", "",
        "| 目标 | Reuse | 冻结方案15 | 摘要近似输入 | native输入 | native / Reuse | native优于Reuse的UID比例 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in (1, 3, 4, 5):
        r = {v["method"]: v for v in rows if v["target"] == t and v["group"] == "diagnostic_uid" and v["panel"] == "held64" and v["path"] == "actual"}
        text.append(f"| M{t} | {r['reuse']['mse']:.7g} | {r['shared15']['mse']:.7g} | {r['summary_input']['mse']:.7g} | {r['native_input']['mse']:.7g} | {r['native_input']['reuse_ratio']:.4g} | {r['native_input']['better_than_reuse_fraction']:.1%} |")
    text += ["", "| 目标 | native−摘要输入 配对差值 | 描述性95% UID区间 | native优于摘要的UID比例 |",
             "| --- | ---: | ---: | ---: |"]
    for v in contrasts:
        if v["group"] == "diagnostic_uid" and v["panel"] == "held64":
            text.append(f"| M{v['target']} | {v['native_minus_summary']:.7g} | [{v['interval'][0]:.7g}, {v['interval'][1]:.7g}] | {v['native_better_uid_fraction']:.1%} |")
    text += ["", "## 固定16查询自由编码", "", "这一部分使用教师，不是原型或低成本目标侧计算。原64拟合面板取0,4,...,60，包含4个不同item；留出仍为原64查询。", "",
        "| 目标 | Reuse | 冻结方案15 | free32/64 | free32/16 | free128/64 | free128/16 | 完整仿射64 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in (1, 3, 4, 5):
        r = {(v["method"], v["path"]): v["mse"] for v in rows if v["target"] == t and v["group"] == "diagnostic_uid" and v["panel"] == "held64"}
        cells = [r[key] for key in [("reuse", "actual"), ("shared15", "actual"), ("functional_response32", "free64"),
                 ("functional_response32", "free16"), ("functional_response128", "free64"), ("functional_response128", "free16"), ("oracle64", "oracle")]]
        text.append(f"| M{t} | "+" | ".join(f"{x:.7g}" for x in cells)+" |")
    text += ["", "## 边界与成本", "", "两个输入臂同参数、同ridge0.01、同聚合响应目标，固定mean PCA32；只改实际native响应与producer均值响应近似。W与U联合拟合，非36标量后置补丁。",
        "", "native臂不增加持久摘要或完整KV扫描；U增加221184个共享系数、每请求221184次乘加。对窗口1024，约为两次历史矩阵乘法的9.375%；历史仅16时则为其6倍。这不是端到端延迟或总服务成本结论。",
        "", "源PCA宽度不变；新增响应输入导致相对旧参照参数增加，只有native对摘要输入的匹配差异支持信息归因。两臂都保留count²权重、全部UID和确定性消退mask，无静默head gate。",
        "", "M2不拟合/诊断；M5 E14_partial。所有区间是已使用开发UID、单backbone seed17的描述性证据。未启动6000评价，确认未读；没有继续横向原型搜索。"]
    (out / "report.md").write_text("\n".join(text)+"\n")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-run", required=True)
    parser.add_argument("--free16-run", required=True)
    parser.add_argument("--report-id", required=True)
    main(parser.parse_args())

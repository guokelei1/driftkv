#!/usr/bin/env python3
"""Read frozen native raw: state bias, actual aggregate risk, and UID contributions."""

import json

import numpy as np
import pandas as pd
import torch
from design.data import DATASET, ROOT
from design.run import write_json

A = ROOT / "results/design/mechanism_native_input192_01"
META = ROOT / "results/design/mechanism_decoder_closure192_01"
OUT = ROOT / "results/design/analysis/native_residual_audit_01"


def main():
    OUT.mkdir(exist_ok=False)
    users = pd.read_parquet(DATASET.parent / "users.parquet", columns=["uid", "n_theta0"])
    rows, risks, objectives = [], [], []
    for target in (1, 3, 4, 5):
        meta = pd.read_parquet(META / f"scenes_m{target}.parquet")
        meta["state_ordinal"] = meta.groupby("uid").cumcount()
        meta = meta.merge(users, on="uid", validate="many_to_one")
        meta["cache_length_group"] = pd.cut(meta["count"], [0, 32, 256, 1023, 1024], labels=["1-32", "33-256", "257-1023", "1024"]).astype(str)
        mu = float(meta.loc[meta.group == "fitting_uid", "count"].pow(2).mean())
        for path in sorted(A.glob(f"raw_native_input_m{target}_batch*.pt")):
            raw = torch.load(path, map_location="cpu", weights_only=True)
            for panel in ("fit64", "held64"):
                exact = raw[panel]["exact"].double()
                for method, prediction in (("A", raw[panel]["prediction"]), ("reuse", raw["reuse_"+panel]), ("shared15", raw["shared15_"+panel])):
                    e = prediction.double()-exact
                    bias = e.mean(-1).square()
                    variance = (e-e.mean(-1, keepdim=True)).square().mean(-1)
                    torch.testing.assert_close(e.square().mean(-1), bias+variance, atol=1e-12, rtol=1e-12)
                    for j, scene in enumerate(raw["scene_indices"]):
                        record = meta.loc[meta.scene == scene].iloc[0].to_dict()
                        rows.append(dict(record, panel=panel, method=method, logit_mse=float(e[j].square().mean()),
                                         bias_squared=float(bias[j]), centered_mse=float(variance[j])))
        response = pd.read_parquet(A / f"responses_m{target}.parquet")
        response = response[response.method == "native_input"].merge(meta[["scene", "count", "state_ordinal"]], on="scene", validate="many_to_one")
        response["aggregate_risk_scaled"] = response.response_mse/mu
        response["rate_mse"] = response.response_mse/response["count"].pow(2)
        risks.append(response)
        params = torch.load(A / f"translator_native_input_m{target}.pt", map_location="cpu", weights_only=True)
        fits = json.loads((A / f"fits_m{target}.json").read_text())
        for layer, p in enumerate(params):
            fit = next(r for r in fits if r["method"] == "native_input" and r["layer"] == layer)
            ridge = .01*float(p["weights"].square().sum()+p["read_weights"].square().sum())/(6*32)
            masked = response[(response.group == "fitting_uid") & (response.panel == "fit64") & (response.layer == layer)].aggregate_risk_scaled.mean()
            objectives.append(dict(target=target, layer=layer, count_square_mean=mu,
                unmasked_fitting_data=fit["fitting_aggregate_objective"], masked_serving_data=float(masked), ridge=ridge,
                ridge_to_data=ridge/fit["fitting_aggregate_objective"]))
    frame = pd.DataFrame(rows)
    assert np.isfinite(frame[["logit_mse", "bias_squared", "centered_mse"]]).all().all()
    frame.to_parquet(OUT / "states.parquet", index=False)
    response = pd.concat(risks)
    response.to_parquet(OUT / "response_risk.parquet", index=False)
    uid = frame.groupby(["target", "group", "panel", "method", "uid"], as_index=False)[["logit_mse", "bias_squared", "centered_mse"]].mean()
    uid.to_parquet(OUT / "uids.parquet", index=False)
    actual = response[(response.panel == "fit64") & (response.group == "fitting_uid")].groupby(["target", "uid"]).aggregate_risk_scaled.sum().reset_index()
    actual["masked_aggregate_loss_share"] = actual.aggregate_risk_scaled/actual.groupby("target").aggregate_risk_scaled.transform("sum")
    actual.to_parquet(OUT / "training_uid_contributions.parquet", index=False)
    # Each stratum first averages states within UID; these conditional summaries do not add up to the overall mean.
    strata = []
    for field in ("cache_length_group", "state_group", "n_theta0", "release_age"):
        f = frame[(frame.panel == "held64") & (frame.method.isin(["A", "reuse"]))]
        if field == "n_theta0":
            f = f.copy()
            f[field] = pd.cut(f[field], [-1,255,1023,4095,np.inf], labels=["<256", "256-1023", "1024-4095", "4096+"]).astype(str)
        f = f.groupby(["target", "group", field, "method", "uid"], as_index=False, observed=True)[["logit_mse", "bias_squared", "centered_mse"]].mean()
        agg = f.groupby(["target", "group", field, "method"], as_index=False, observed=True).agg(logit_mse=("logit_mse","mean"), bias_squared=("bias_squared","mean"), centered_mse=("centered_mse","mean"), uids=("uid","nunique"))
        strata.extend(agg.rename(columns={field:"stratum"}).assign(field=field).to_dict("records"))
    write_json(OUT / "summary.json", dict(objectives=objectives, strata=strata,
        status="raw_audit_complete_producer_and_unmasked_uid_risk_pending_frozen_read", confirmation_read=False))
    lines = ["# Native原型残余与训练风险审查", "", "本报告读取冻结A（64×64）raw，无重新拟合。生产者字段和未mask的逐UID训练风险由下一次冻结读取补齐。", "",
        "目标逐head/output为 `(1/S) sum_s [(N_s²/mu_A)*(1/Q) sum_j ||rate_error||²] + 0.01||[W,U]||²`。表中再同时除以H×D。mu_A为原拟合场景mean(N²)；场景等权，场景多的UID在训练中权重更多；报告先场景平均再UID等权。",
        "", "目标仅除以N，没有输出通道标准化。PCA、query与native输入有冻结的标准化；截距也参与ridge。N²将率误差换成聚合响应误差，不是可以自动删除的错误权重。", "",
        "拟合使用未mask预测，服务安装确定性消退mask；二者数据项应分开。下表实际响应风险是服务mask之后的风险，尚不能替代未mask训练贡献。", "",
        "| M | 组 | A MSE | bias²占比 | centered MSE |", "| --- | --- | ---: | ---: | ---: |"]
    for (t,g), f in uid[(uid.panel == "held64") & (uid.method == "A")].groupby(["target","group"]):
        lines.append(f"| M{t} | {g} | {f.logit_mse.mean():.8g} | {f.bias_squared.sum()/f.logit_mse.sum():.2%} | {f.centered_mse.mean():.8g} |")
    lines += ["", "UID988060保留：", "", "| M | A logit MSE | 实际masked聚合损失份额（fit64） |", "| --- | ---: | ---: |"]
    for t in (1,3,4,5):
        err = uid[(uid.target == t)&(uid.uid==988060)&(uid.panel=="held64")&(uid.method=="A")].logit_mse.iloc[0]
        share = actual[(actual.target==t)&(actual.uid==988060)].masked_aggregate_loss_share.iloc[0]
        lines.append(f"| M{t} | {err:.8g} | {share:.4%} |")
    lines += ["", "全UID、逐状态、实际缓存N与初始n_theta0分别保存在states/uids/response_risk.parquet；分组见summary.json。这里只是MSE机制审查，不是AUC恢复，也不据此修改权重或删UID。"]
    (OUT / "report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

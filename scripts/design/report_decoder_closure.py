#!/usr/bin/env python3
"""UID-equal interpretation of frozen interventions; oracle is a diagnostic only."""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from design.data import ROOT


def interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(17)
    means = values[rng.integers(len(values), size=(1000, len(values)))].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def main(cli):
    run = ROOT / "results/design" / cli.run_id
    out = ROOT / "results/design/analysis" / cli.report_id
    out.mkdir(parents=True, exist_ok=False)
    outputs = pd.concat([pd.read_parquet(run / f"outputs_m{t}.parquet") for t in (1, 3, 4, 5)])
    meta = pd.concat([pd.read_parquet(run / f"scenes_m{t}.parquet") for t in (1, 3, 4, 5)])
    config = json.loads((run / "configuration.json").read_text())
    if len(config["fitting_uids"]) == 64 and len(config["diagnostic_uids"]) == 128:
        for t in (1, 3, 4, 5):
            old = pd.read_parquet(ROOT / f"results/design/mechanism_query_holdout192_01/outputs_m{t}.parquet")
            for method in ("reuse", "shared15"):
                x = outputs[(outputs.target == t) & (outputs.method == method) & (outputs.path == "actual")]
                pair = x.merge(old[old.method == method], on=["uid", "scene", "panel"], suffixes=("_new", "_old"), validate="one_to_one")
                assert len(pair) == len(x)
                np.testing.assert_allclose(pair.logit_mse_new, pair.logit_mse_old, atol=1e-7, rtol=3e-4)
            for name, previous in (("mean_coefficient32", "mean_coefficient"), ("functional_response32", "functional_response"),
                                   ("functional_response128", "functional_response")):
                folder = "mechanism_rich_source192_01" if name.endswith("128") else "mechanism_aggregate_factorial192_01"
                frozen = pd.read_parquet(ROOT / f"results/design/{folder}/outputs_m{t}.parquet")
                x = outputs[(outputs.target == t) & (outputs.method == name) & (outputs.path == "actual")]
                pair = x.merge(frozen[frozen.method == previous], on=["uid", "scene", "panel"], suffixes=("_new", "_old"), validate="one_to_one")
                np.testing.assert_allclose(pair.logit_mse_new, pair.logit_mse_old, atol=1e-7, rtol=3e-4)
            oracle = old[old.method == "oracle64"].copy()
            oracle = oracle.merge(meta[meta.target == t][["scene", "uid", "state_group", "count", "release_age"]], on=["scene", "uid"], validate="many_to_one")
            oracle["path"] = "oracle"
            outputs = pd.concat([outputs, oracle[outputs.columns]], ignore_index=True)
    assert np.isfinite(outputs.logit_mse).all(), "nonfinite rows must not be silently dropped by aggregation"
    assert not outputs.duplicated(["target", "uid", "scene", "panel", "method", "path"]).any()
    uid = outputs.groupby(["target", "group", "panel", "method", "path", "uid"], as_index=False).logit_mse.mean()
    tables, comparisons = [], []
    for keys, frame in uid.groupby(["target", "group", "panel", "method", "path"]):
        t, group, panel, method, path = keys
        tables.append(dict(target=int(t), group=group, panel=panel, method=method, path=path,
            mse=float(frame.logit_mse.mean()), median=float(frame.logit_mse.median()),
            max_uid=int(frame.loc[frame.logit_mse.idxmax(), "uid"]),
            max_uid_error_share=float(frame.logit_mse.max()/frame.logit_mse.sum())))
    for keys, frame in uid.groupby(["target", "group", "panel"]):
        ref = frame[(frame.method == "reuse") & (frame.path == "actual")].set_index("uid").logit_mse
        actual15 = frame[(frame.method == "shared15") & (frame.path == "actual")].set_index("uid").logit_mse
        for (method, path), candidate in frame.groupby(["method", "path"]):
            v = candidate.set_index("uid").logit_mse.reindex(ref.index)
            comparisons.append(dict(target=int(keys[0]), group=keys[1], panel=keys[2], method=method, path=path,
                reuse_ratio=float(v.mean()/ref.mean()), better_than_reuse_uid_fraction=float((v < ref).mean()),
                paired_difference_reuse_ci=interval(v-ref), paired_difference_shared15_ci=interval(v-actual15)))
    decomposition, rounding = [], []
    for t in (1, 3, 4, 5):
        target_meta = meta[meta.target == t].set_index("scene")
        for path in sorted(run.glob(f"raw_m{t}_batch*.pt")):
            raw = torch.load(path, map_location="cpu", weights_only=True)
            for method in ("mean_coefficient32", "functional_response32"):
                for context, direct in (("exact", raw[method]["held64"]["single_epsilon"]),
                                        (method, raw[method]["held64"]["epsilon"])):
                    cross = raw[f"cross_{context}_{method}"].double()
                    direct = direct.double()
                    rounding.append(dict(target=t, method=method, context=context,
                        difference_energy=float((cross-direct).square().sum()), direct_energy=float(direct.square().sum()),
                        cross_energy=float(cross.square().sum())))
            for method in ("mean_coefficient32", "functional_response32", "functional_response128", "shared15"):
                for panel in ("fit64", "held64"):
                    d = raw[method][panel]["d"].double()
                    gram = d @ d.transpose(-2, -1)/d.shape[-1]
                    direct = (raw[method][panel]["shared"].double()-raw["exact"][panel].double()).square().mean(-1)
                    torch.testing.assert_close(gram.sum((1, 2)), direct, atol=1e-12, rtol=1e-10)
                    for j, i in enumerate(raw["scene_indices"]):
                        record = target_meta.loc[i].to_dict()
                        decomposition.append(dict(**record, scene=i, method=method, panel=panel,
                            diagonal=float(gram[j].diagonal().sum()), cross=float(gram[j].sum()-gram[j].diagonal().sum()),
                            **{f"d{a+1}_d{b+1}": float(gram[j, a, b]) for a in range(6) for b in range(a, 6)}))
            del raw
    dframe = pd.DataFrame(decomposition)
    dframe.to_parquet(out / "decomposition.parquet", index=False)
    duid = dframe.groupby(["target", "group", "panel", "method", "uid"])[["diagonal", "cross"]].mean().reset_index()
    dsummary = duid.groupby(["target", "group", "panel", "method"])[["diagonal", "cross"]].mean().reset_index().to_dict("records")
    weights = []
    for t, frame in meta[meta.group == "fitting_uid"].groupby("target"):
        denominator = frame["count"].pow(2).sum()
        for uid_id, f in frame.groupby("uid"):
            weights.append(dict(target=int(t), uid=int(uid_id), scenes=len(f), scene_weight=len(f)/len(frame),
                aggregate_training_weight=float(f["count"].pow(2).sum()/denominator), report_weight=1/frame.uid.nunique()))
    stratified = (outputs.groupby(["target", "group", "state_group", "panel", "method", "path", "uid"]).logit_mse.mean()
                  .groupby(["target", "group", "state_group", "panel", "method", "path"]).mean().reset_index().to_dict("records"))
    responses = pd.concat([pd.read_parquet(run / f"responses_m{t}.parquet") for t in (1, 3, 4, 5)])
    local = (responses.groupby(["target", "group", "panel", "context", "method", "layer", "uid"]).response_mse.mean()
             .groupby(["target", "group", "panel", "context", "method", "layer"]).mean().reset_index().to_dict("records"))
    common = pd.concat([pd.read_parquet(run / f"common_query_m{t}.parquet") for t in (1, 3, 4, 5)])
    common_summary = (common.groupby(["target", "group", "context", "method", "layer", "uid"]).response_mse.mean()
                      .groupby(["target", "group", "context", "method", "layer"]).mean().reset_index().to_dict("records"))
    solvers = pd.concat([pd.read_parquet(run / f"solvers_m{t}.parquet") for t in (1, 3, 4, 5)])
    solver_summary = []
    for key, f in solvers[solvers.active].groupby(["target", "group", "method", "layer"]):
        solver_summary.append(dict(target=int(key[0]), group=key[1], method=key[2], layer=int(key[3]),
            minimum_rank=float(f["rank"].min()), median_effective_rank=float(f.effective_rank.median()),
            median_condition=float(f.condition.median()), maximum_condition=float(f.condition.max()),
            maximum_relative_normal_residual=float(f.relative_normal_residual.max()),
            median_latent_norm=float(f.latent_norm.median()), maximum_latent_norm=float(f.latent_norm.max())))
    # Preidentified retained outlier, not selected anew to exclude or tune it.
    outlier = outputs[(outputs.target == 3) & (outputs.uid == 988060)]
    outlier_summary = outlier.groupby(["panel", "method", "path"], as_index=False).logit_mse.mean().to_dict("records")
    probe_geometry = {}
    for folder in ("mechanism_aggregate_factorial192_01", "mechanism_rich_source192_01"):
        p = torch.load(ROOT / f"results/design/{folder}/fixed_m0_probes.pt", map_location="cpu", weights_only=True).double()
        record = {}
        for name, bank in (("uncentered", p), ("centered", p-p.mean(-2, keepdim=True))):
            singular = torch.linalg.svdvals(bank)
            probability = singular.square()/singular.square().sum(-1, keepdim=True)
            rank = (-(probability*probability.clamp_min(1e-30).log()).sum(-1)).exp()
            record[name] = dict(head_layer_entropy_ranks=rank.tolist(), median=float(rank.median()),
                                numeric_rank_median=float((singular > singular[..., :1]*1e-6).sum(-1).double().median()))
        probe_geometry[folder] = record
    source_variation = []
    for t in (1, 3, 4, 5):
        audit = json.loads((run / f"source_audit_m{t}.json").read_text())
        for method in ("mean_coefficient32", "functional_response32", "functional_response128"):
            f = audit[method]
            between, within = np.array(f["between_uid_energy_by_dimension"]), np.array(f["within_uid_energy_by_dimension"])
            source_variation.append(dict(target=t, method=method, source_retained=f["fitting_source_retained"],
                within_uid_latent_energy_fraction=float(within.sum()/(within+between).sum()),
                weaker96_within_fraction=float(within[:-32].sum()/(within+between)[:-32].sum()) if len(within)>32 else None,
                note="whitened latent coordinates, fitting-scene weighted; weaker96 vs strongest32 within PCA128, not nested across different probe banks"))
    result = dict(rows=tables, comparisons=comparisons, decomposition=dsummary, fitting_weights=weights,
        state_strata=stratified, local_response=local, common_query=common_summary, solvers=solver_summary,
        retained_m3_uid988060=outlier_summary, probe_geometry=probe_geometry, source_variation=source_variation,
        common_query_fp32_rounding=pd.DataFrame(rounding).groupby(["target", "method", "context"]).sum().reset_index().to_dict("records"),
        scope="scenario mean then UID equal; reused diagnostic UIDs, one backbone seed; descriptive paired intervals; no new quality run")
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    text = ["# 冻结decoder与闭环干预诊断", "", "同场景、同面板；先场景平均再UID等权。所有数值为logit MSE，非AUC恢复率。", "",
            "| 目标 | 路径 | 诊断held MSE | /Reuse | 中位数 | 优于Reuse的UID比例 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in tables:
        if row["group"] != "diagnostic_uid" or row["panel"] != "held64" or row["path"] not in ("actual", "free64", "oracle"):
            continue
        compare = next(c for c in comparisons if all(c[k] == row[k] for k in ("target", "group", "panel", "method", "path")))
        text.append(f"| M{row['target']} | {row['method']}/{row['path']} | {row['mse']:.8g} | {compare['reuse_ratio']:.5g} | {row['median']:.6g} | {compare['better_than_reuse_uid_fraction']:.1%} |")
    text += ["", "完整前缀/单层、拟合/留出、各UID权重和带符号交叉项见summary.json与decomposition.parquet。", "", "确认集未读取；M2仅native过渡；M5 E14_partial。自由编码使用场景教师，非部署方法。"]
    (out / "report.md").write_text("\n".join(text)+"\n")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-id", required=True)
    main(parser.parse_args())

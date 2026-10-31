#!/usr/bin/env python3
"""Post-hoc UID sensitivity check with a result-independent frozen split.

Membership is persisted before outcomes/weights are opened. Existing ridge
weights and selected lambdas are reused exactly: no retraining or selection.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments.ch3_motivation.code.analyze_candidate_residual_probe import (
    blocks, bootstrap_weights, centered, derangements, estimate, features, mean_stat, pair_geometry,
    ratio_stat, sha,
)

RULE_PREFIX = "insight-extensions/20261023/"


def group_for_uid(uid):
    return hashlib.sha256((RULE_PREFIX + str(int(uid))).encode()).digest()[0] % 2


def freeze_membership(run_root, output):
    """Only configurations/UIDs are read in this phase; no outcomes or weights."""
    configs = {edge: json.loads((run_root / f"candidate_edge{edge}/configuration.json").read_text())
               for edge in (2, 3)}
    uids = configs[2]["roles"]["main"]
    if len(uids) != 2000 or len(set(uids)) != 2000 or configs[3]["roles"]["main"] != uids:
        raise ValueError("both cases must share the original fixed2000 UID order")
    groups = {str(group):[uid for uid in uids if group_for_uid(uid) == group] for group in (0, 1)}
    if min(map(len, groups.values())) < 512:
        raise ValueError("a frozen group has fewer than512 users")
    membership = dict(format="candidate_uid_split_v1",
        rule="sha256('insight-extensions/20261023/'+str(uid)).digest()[0]%2",
        groups={group:dict(count=len(ids), uids=ids) for group,ids in groups.items()},
        source_configurations={str(edge):sha(run_root/f"candidate_edge{edge}/configuration.json") for edge in (2,3)},
        scope="Deterministic result-independent split of the already evaluated main2000; same UID belongs to same group across cases; not independent confirmation.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "membership.json"
    if path.exists():
        if json.loads(path.read_text()) != membership:
            raise ValueError("saved split membership differs; do not replace it")
    else:
        path.write_text(json.dumps(membership, indent=2, ensure_ascii=False)+"\n")
    return configs, membership


def collect_case(case_root, config):
    """Reuse saved physical-unit weights and fit-only scales without fitting."""
    original = json.loads((case_root/"analysis/summary.json").read_text())
    runner = json.loads((case_root/"summary.json").read_text())
    if not original["quality_use_permitted"] or original["main_users"] != 2000:
        raise ValueError("requires the completed original fixed2000 analysis")
    with np.load(case_root/"analysis/ridge_weights.npz", allow_pickle=False) as saved:
        scale = saved["broad16_feature_scale"]
        real_weight = saved["broad16_read_real_item_weight"]
        permuted_weight = saved["broad16_read_permuted_item_weight"]
    values = {name:[] for name in ("uid", "constant", "real_item", "permuted_item", "total")}
    geometry = {panel:{name:[] for name in ("numerator", "denominator")} for panel in ("broad16", "near16")}
    for data in blocks(case_root, "main", config, runner):
        uids = data["uid"]
        x,_ = features(data["features"][:,0], scale)
        target = centered(data["read_target"][:,0].reshape(len(x),16,-1))
        real = x @ real_weight
        permuted = x[np.arange(len(x))[:,None], derangements(uids)] @ permuted_weight
        values["uid"].extend(int(uid) for uid in uids)
        values["constant"].extend(np.mean(target**2,axis=(1,2)))
        values["real_item"].extend(np.mean((real-target)**2,axis=(1,2)))
        values["permuted_item"].extend(np.mean((permuted-target)**2,axis=(1,2)))
        values["total"].extend(data["read_total_mse"][:,0].mean((1,2)))
        for index,panel in enumerate(geometry):
            g = pair_geometry(data["teacher_scores"][:,index].astype(float), data["f_scores"][:,index].astype(float))
            geometry[panel]["numerator"].extend(g["teacher_valid_disagreement_fraction_all_pairs"])
            geometry[panel]["denominator"].extend(g["teacher_nonzero_pair_fraction"])
    values = {name:np.asarray(rows,dtype=np.uint64 if name=="uid" else np.float64) for name,rows in values.items()}
    geometry = {panel:{name:np.asarray(rows) for name,rows in d.items()} for panel,d in geometry.items()}
    reference = next(p for p in original["points"] if p["panel"]=="broad16" and p["task"]=="read")
    for method in ("constant", "real_item", "permuted_item"):
        if not np.isclose(values[method].mean(),reference["method_mse"][method]["mean"],rtol=1e-10,atol=1e-12):
            raise ValueError("saved-weight reconstruction differs from the original full2000 result")
    return values, geometry, dict(
        configuration_sha256=sha(case_root/"configuration.json"),
        original_analysis_sha256=sha(case_root/"analysis/summary.json"),
        saved_ridge_weights_sha256=sha(case_root/"analysis/ridge_weights.npz"),
        fixed_selected_lambdas=[p for p in original["fitting"] if p["panel"]=="broad16" and p["task"]=="read"],
        all2000_reconstruction_matches_original=True)


def analyze_case(values, geometry, *, replicates, seed):
    result = {}
    for group in (0,1):
        mask = np.array([group_for_uid(uid)==group for uid in values["uid"]])
        count = int(mask.sum())
        if count < 512:
            raise ValueError("a main subgroup is diagnostic only")
        v={name:rows[mask] for name,rows in values.items()}
        weights=bootstrap_weights(count,replicates,seed)
        point=dict(unique_users=count, quality_use_permitted=True,
            broad_read_mse={m:mean_stat(v[m],weights) for m in ("constant","real_item","permuted_item")},
            real_improvement_over_constant=mean_stat(v["constant"]-v["real_item"],weights),
            real_improvement_percent_of_centered=ratio_stat(v["constant"]-v["real_item"],v["constant"],weights),
            real_improvement_percent_of_total=ratio_stat(v["constant"]-v["real_item"],v["total"],weights),
            permutation_minus_real_mse=mean_stat(v["permuted_item"]-v["real_item"],weights),
            permutation_minus_real_percent_of_centered=ratio_stat(v["permuted_item"]-v["real_item"],v["constant"],weights))
        conditional, draws = {},{}
        for panel in ("broad16","near16"):
            n=geometry[panel]["numerator"][mask];d=geometry[panel]["denominator"][mask]
            point[panel+"_teacher_valid_disagreement_percent"]=ratio_stat(n,d,weights)
            point[panel+"_teacher_valid_pair_support"]=int(np.rint(d.sum()*120))
            conditional[panel]=n.mean()/d.mean()
            draws[panel]=(weights@n)/(weights@d)
        point["near_minus_broad_disagreement_pp"]=estimate(100*(conditional["near16"]-conditional["broad16"]),100*(draws["near16"]-draws["broad16"]))
        result[str(group)]=point
    return result


def run(run_root,output,replicates=500,seed=17):
    configs,membership=freeze_membership(run_root,output)  # MUST precede outcome/weight reads.
    cases={};provenance={}
    for edge in (2,3):
        values,geometry,provenance[str(edge)]=collect_case(run_root/f"candidate_edge{edge}",configs[edge])
        cases[str(edge)]=analyze_case(values,geometry,replicates=replicates,seed=seed)
        if any(point["unique_users"]!=membership["groups"][group]["count"] for group,point in cases[str(edge)].items()):
            raise ValueError("analyzed subgroup counts differ from frozen membership")
    result=dict(format="candidate_frozen_weight_uid_split_check_v1",status="complete",cases=cases,
        groups={g:x["count"] for g,x in membership["groups"].items()},membership_sha256=sha(output/"membership.json"),
        replicates=replicates,seed=seed,provenance=provenance,analysis_source_sha256=sha(Path(__file__)),
        limitations=["Post-hoc sensitivity within the same already evaluated main2000, not a new independent population or repeated training.",
            "Membership depends only on fixed UID hashing; same membership is used across cases and all groups are retained.",
            "Originalfit100/validation16 scales, ridge weights, projection, lambdas and permutation seed are reused exactly; no refitting or model selection.",
            "Intervals resample subgroup UIDs conditional on the fixed models/weights; training randomness and projection/permutation randomness are not tested.",
            "Teacher-order disagreement under selected synthetic panels is not real-feedback AUC or retrieval accuracy.",
            "The absolute candidate-centered improvement can be tiny even when its paired interval excludes zero."])
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    def stat(s):return f"{s['mean']:.5g} [{s['ci95'][0]:.5g}, {s['ci95'][1]:.5g}]"
    lines=["# 固定权重的UID人口敏感性核对", "", f"先固定哈希membership并持久化，再读取结果。group0={result['groups']['0']}人，group1={result['groups']['1']}人；同UID跨边同组。",
        "沿用原fit100/val16得到的全部尺度、ridge权重与lambda，不重新拟合；以下是配对UID区间。", "",
        "| edge | group | 人数 | real改善/centered read(%) | perm−real MSE | real改善/全部read(%) | near−broad次序不一致(pp) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for edge,groups in cases.items():
        for group,p in groups.items():
            lines.append("| "+" | ".join([edge,group,str(p["unique_users"]),stat(p["real_improvement_percent_of_centered"]),stat(p["permutation_minus_real_mse"]),stat(p["real_improvement_percent_of_total"]),stat(p["near_minus_broad_disagreement_pp"])])+" |")
    lines += ["", "全2000重算均值已与原分析的constant/real/permuted三个端点匹配。这里是原人口的敏感性，不是独立确认或训练随机性检验。teacher-valid次序指标将student ties计为不一致；teacher ties不进入分母。", "", *["- "+x for x in result["limitations"]]]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replicates",type=int,default=500);p.add_argument("--seed",type=int,default=17)
    args=p.parse_args()
    a=run(args.run_root.resolve(),args.output.resolve(),args.replicates,args.seed)
    print(json.dumps(dict(status=a["status"],groups=a["groups"])))

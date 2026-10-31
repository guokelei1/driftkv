#!/usr/bin/env python3
"""CPU ridge controls for fixed-panel candidate residual predictability.

Panel centering is a diagnostic operation. Teacher means define targets only;
they are never prediction inputs. Real/permuted item conditions have identical
capacity and each select regularization on the reserved validation users.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PANELS = ("broad16", "near16")
LAMBDAS = (.01, .1, 1., 10.)
CONDITIONS = ("constant", "real_item", "permuted_item")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def derangements(uids, seed=29):
    """Independent UID-keyed permutations, with no candidate left in place."""
    result = []
    for uid in uids:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(uid)]))
        order = rng.permutation(16)
        while np.any(order == np.arange(16)):
            order = rng.permutation(16)
        result.append(order)
    return np.stack(result)


def centered(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(1, keepdims=True)


def features(x, scale=None, *, permutation=None):
    x = centered(x)
    if scale is None:
        scale = np.maximum(np.sqrt(np.mean(x ** 2, axis=(0, 1))), 1e-8)
    x = x / scale
    if permutation is not None:
        x = x[np.arange(len(x))[:, None], permutation]
    return x, scale


def fit_ridge(x, y, validation_x, validation_y):
    """Mean squared objective +lambda*||W||²; no optimizer/early-stop confound."""
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    vx, vy = validation_x.reshape(-1, x.shape[-1]), validation_y.reshape(-1, y.shape[-1])
    gram, rhs = x.T @ x / len(x), x.T @ y / len(x)
    rows, best = [], None
    for regularization in LAMBDAS:
        weight = np.linalg.solve(gram + regularization * np.eye(x.shape[1]), rhs)
        loss = float(np.mean((vx @ weight - vy) ** 2))
        rows.append(dict(regularization=regularization, validation_normalized_mse=loss))
        if best is None or loss < best[0]:
            best = loss, regularization, weight
    return best[2], dict(selected_lambda=best[1], validation_grid=rows,
        coefficients=int(best[2].size), training_users=int(len(y) // 16))


def blocks(root, role, config, runner):
    root = Path(root)
    expected, consumed = config["roles"][role], 0
    for artifact in runner["artifacts"]:
        path = root / artifact["path"]
        if path.parent.name != role:
            continue
        manifest = root / artifact["manifest"]
        if sha(path) != artifact["sha256"] or sha(manifest) != artifact["manifest_sha256"]:
            raise ValueError("candidate block or manifest hash changed")
        m = json.loads(manifest.read_text())
        if (m["configuration_sha256"] != digest(config) or m["role"] != role
                or m["offset"] != consumed or m["npz_sha256"] != artifact["sha256"]):
            raise ValueError("candidate block configuration or role differs")
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        uids = [int(x) for x in data["uid"]]
        if uids != m["uids"] or uids != expected[consumed:consumed + len(uids)]:
            raise ValueError("fixed UID order or membership changed")
        if (data["features"].shape[1:] != (2, 16, 16)
                or data["logit_target"].shape[1:] != (2, 16)):
            raise ValueError("candidate panel feature/target shape differs")
        actual = data["teacher_scores"].astype(float) - data["f_scores"].astype(float)
        if not np.allclose(data["logit_target"], centered(actual.reshape(-1, 16)).reshape(actual.shape), rtol=1e-4, atol=2e-7):
            raise ValueError("centered logit target no longer matches actual candidates")
        consumed += len(uids)
        yield data
    if consumed != len(expected):
        raise ValueError("candidate role did not complete")


def load_role(root, role, config, runner):
    parts = list(blocks(root, role, config, runner))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def bootstrap_weights(users, replicates=500, seed=17):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(users, size=users), minlength=users) for _ in range(replicates)]).astype(np.uint16)


def estimate(value, draws):
    draws = np.asarray(draws)
    valid = np.isfinite(draws)
    return dict(mean=float(value) if np.isfinite(value) else None,
        ci95=np.percentile(draws[valid], [2.5, 97.5]).tolist() if valid.any() else None,
        invalid_bootstrap_draws=int((~valid).sum()))


def mean_stat(values, weights):
    return estimate(np.mean(values), weights @ values / weights.sum(1))


def ratio_stat(numerator, denominator, weights, *, percent=True):
    factor = 100. if percent else 1.
    n, d = np.mean(numerator), np.mean(denominator)
    with np.errstate(invalid="ignore", divide="ignore"):
        return estimate(factor * n / d if d > 0 else np.nan,
            factor * (weights @ numerator) / (weights @ denominator))


def pair_geometry(teacher, student):
    """Exact stored-score ties,120 unordered pairs per16-candidate UID panel.

    Conditional disagreement excludes teacher ties and counts student ties as
    disagreement. Strict opposite signs remain a separate, narrower measure.
    Fractions are initially over all120 pairs; conditional summaries divide
    UID-pooled numerator by teacher-valid support, not by all pairs.
    """
    i, j = np.triu_indices(16, 1)
    td, sd = teacher[:, i] - teacher[:, j], student[:, i] - student[:, j]
    valid = td != 0
    return dict(span=np.ptp(teacher, axis=1),
        strict_opposite_pair_fraction=np.mean(td * sd < 0, axis=1),
        teacher_nonzero_pair_fraction=np.mean(valid, axis=1),
        student_tie_pair_fraction=np.mean(sd == 0, axis=1),
        teacher_valid_disagreement_fraction_all_pairs=np.mean(valid & (np.sign(td) != np.sign(sd)), axis=1),
        teacher_valid_student_tie_fraction_all_pairs=np.mean(valid & (sd == 0), axis=1),
        median_absolute_teacher_pair_margin=np.median(np.abs(td), axis=1))


def analyze(root, *, replicates=500, seed=17):
    root = Path(root).resolve()
    config = json.loads((root / "configuration.json").read_text())
    runner = json.loads((root / "summary.json").read_text())
    if (config["format"] != "candidate_residual_probe_v1" or runner["status"] != "complete"
            or runner["feedback_labels_used"] or runner["quality_auc_computed"]):
        raise ValueError("requires the completed label-free candidate collector")
    if len(set(sum(config["roles"].values(), []))) != sum(map(len, config["roles"].values())):
        raise ValueError("candidate fitting/validation/main roles overlap")
    train, validation = (load_role(root, role, config, runner) for role in ("fit", "validation"))
    geometry = config["geometry"]
    layers, heads, dim = (geometry[k] for k in ("layers", "heads", "head_dim"))
    main_users = len(config["roles"]["main"])
    permitted = not config["diagnostic"] and runner["roles"] == dict(fit=100, validation=16, main=2000)
    models, fitting, model_arrays = {}, [], {}
    # Error arrays are tiny [UID,L,H]; main targets/predictions stream one block.
    errors = {(panel, task, condition): [] for panel in PANELS for task in ("read", "logit") for condition in CONDITIONS}
    totals = {(panel, task): [] for panel in PANELS for task in ("read", "logit")}
    geometry_names = ("span", "strict_opposite_pair_fraction", "teacher_nonzero_pair_fraction",
        "student_tie_pair_fraction", "teacher_valid_disagreement_fraction_all_pairs",
        "teacher_valid_student_tie_fraction_all_pairs", "median_absolute_teacher_pair_margin")
    panel_geometry = {panel: {name:[] for name in geometry_names} for panel in PANELS}
    for index, panel in enumerate(PANELS):
        x, scale = features(train["features"][:, index])
        vx, _ = features(validation["features"][:, index], scale)
        model_arrays[panel + "_feature_scale"] = scale
        read = centered(train["read_target"][:, index].reshape(len(x), 16, -1))
        read_scale = np.maximum(np.sqrt(np.mean(read.reshape(len(x), 16, layers, heads, dim) ** 2, axis=(0, 1, 4))), 1e-8)
        model_arrays[panel + "_read_scale"] = read_scale
        for task in ("read", "logit"):
            output_scale = np.repeat(read_scale.reshape(-1), dim) if task == "read" else np.ones(1)
            y = centered(train[task + "_target"][:, index].reshape(len(x), 16, -1)) / output_scale
            vy = centered(validation[task + "_target"][:, index].reshape(len(vx), 16, -1)) / output_scale
            for condition in CONDITIONS[1:]:
                tx, tvx = x, vx
                if condition == "permuted_item":
                    tx = x[np.arange(len(x))[:, None], derangements(train["uid"])]
                    tvx = vx[np.arange(len(vx))[:, None], derangements(validation["uid"])]
                weight, info = fit_ridge(tx, y, tvx, vy)
                weight *= output_scale  # predictions return to physical target units
                models[panel, task, condition] = (scale, weight)
                model_arrays[f"{panel}_{task}_{condition}_weight"] = weight
                fitting.append(dict(panel=panel, task=task, condition=condition, **info))
    for data in blocks(root, "main", config, runner):
        uids = data["uid"]
        permutation = derangements(uids)
        for index, panel in enumerate(PANELS):
            x, _ = features(data["features"][:, index], model_arrays[panel + "_feature_scale"])
            for task in ("read", "logit"):
                target = centered(data[task + "_target"][:, index].reshape(len(x), 16, -1))
                totals[panel, task].append(data[task + "_total_mse"][:, index])
                for condition in CONDITIONS:
                    prediction = np.zeros_like(target) if condition == "constant" else (
                        (x if condition == "real_item" else x[np.arange(len(x))[:, None], permutation]) @ models[panel, task, condition][1])
                    difference = prediction - target
                    error = (difference.reshape(len(x), 16, layers, heads, dim) ** 2).mean((1, 4)) if task == "read" else (difference ** 2).mean((1, 2))
                    errors[panel, task, condition].append(error)
            teacher, student = data["teacher_scores"][:, index].astype(float), data["f_scores"][:, index].astype(float)
            for name, values in pair_geometry(teacher, student).items():
                panel_geometry[panel][name].extend(values.tolist())
    errors = {key: np.concatenate(value) for key, value in errors.items()}
    totals = {key: np.concatenate(value) for key, value in totals.items()}
    weights = bootstrap_weights(main_users, replicates, seed)
    points, headwise = [], []
    for panel in PANELS:
        for task in ("read", "logit"):
            raw_errors = {c:errors[panel, task, c] for c in CONDITIONS}
            e = {c:value.mean((1, 2)) if task == "read" else value for c, value in raw_errors.items()}
            total = totals[panel, task].mean((1, 2)) if task == "read" else totals[panel, task]
            points.append(dict(panel=panel, task=task, unique_users=main_users,
                method_mse={c:mean_stat(e[c], weights) for c in CONDITIONS},
                candidate_centered_fraction_of_total_percent=ratio_stat(e["constant"], total, weights),
                real_improvement_over_constant=mean_stat(e["constant"]-e["real_item"], weights),
                real_improvement_percent_of_centered=ratio_stat(e["constant"]-e["real_item"], e["constant"], weights),
                real_improvement_percent_of_total=ratio_stat(e["constant"]-e["real_item"], total, weights),
                permutation_minus_real_mse=mean_stat(e["permuted_item"]-e["real_item"], weights),
                permutation_minus_real_percent_of_centered=ratio_stat(e["permuted_item"]-e["real_item"], e["constant"], weights)))
            if task == "read":
                scale2 = model_arrays[panel + "_read_scale"] ** 2
                points[-1]["head_balanced_method_mse"] = {c:mean_stat((raw_errors[c]/scale2).mean((1, 2)), weights) for c in CONDITIONS}
                for layer in range(layers):
                    for head in range(heads):
                        he = {c:raw_errors[c][:, layer, head] for c in CONDITIONS}
                        contrast = mean_stat(he["permuted_item"]-he["real_item"], weights)
                        headwise.append(dict(panel=panel, layer=layer, head=head,
                            constant_mse=float(he["constant"].mean()), real_item_mse=float(he["real_item"].mean()),
                            permuted_item_mse=float(he["permuted_item"].mean()),
                            permutation_minus_real=contrast["mean"], ci95_low=contrast["ci95"][0], ci95_high=contrast["ci95"][1],
                            fit_only_output_scale=float(np.sqrt(scale2[layer, head]))))
    geometry_summary = {p:{name:mean_stat(np.asarray(values), weights) for name,values in parts.items()} for p,parts in panel_geometry.items()}
    for panel in PANELS:
        raw = {name:np.asarray(values) for name,values in panel_geometry[panel].items()}
        denominator = raw["teacher_nonzero_pair_fraction"]
        geometry_summary[panel]["disagreement_given_teacher_nonzero"] = ratio_stat(
            raw["teacher_valid_disagreement_fraction_all_pairs"], denominator, weights, percent=False)
        geometry_summary[panel]["student_tie_given_teacher_nonzero"] = ratio_stat(
            raw["teacher_valid_student_tie_fraction_all_pairs"], denominator, weights, percent=False)
        geometry_summary[panel]["support"] = dict(unique_users=main_users, all_unordered_pairs=120*main_users,
            teacher_nonzero_pairs=int(np.rint(denominator.sum()*120)),
            users_without_teacher_nonzero_pairs=int(np.count_nonzero(denominator==0)))
    # Coarse dense arithmetic; teacher collection costs are recorded separately.
    fitting_flops = prediction_flops = 0
    for row in fitting:
        outputs = layers * heads * dim if row["task"] == "read" else 1
        fitting_flops += 2 * len(train["uid"]) * 16 * 16 * (16 + outputs)
        fitting_flops += len(LAMBDAS) * (int((2/3) * 16**3) + 2 * 16**2 * outputs
            + 2 * len(validation["uid"]) * 16 * outputs * 17)
        prediction_flops += 2 * main_users * 16 * 16 * outputs
    summary = dict(format="candidate_residual_ridge_analysis_v1", edge=config["edge"], main_users=main_users,
        quality_use_permitted=permitted, runner_diagnostic=config["diagnostic"], fitting=fitting,
        points=points, panel_geometry=geometry_summary, headwise_point_count=len(headwise),
        panel_geometry_definitions=dict(
            ties="Exact equality of storedfloat32 candidate scores after conversion tofloat64; no tolerance or near-tie threshold.",
            strict_opposite_pair_fraction="Among all120 unordered pairs/user: teacher_diff*student_diff<0; excludes either-side ties.",
            teacher_nonzero_pair_fraction="Among all120 pairs/user: teacher_diff!=0; denominator support for conditional statistics.",
            student_tie_pair_fraction="Among all120 pairs/user: student_diff==0, regardless of teacher tie.",
            disagreement_given_teacher_nonzero="Sum of teacher-valid pairs with sign(teacher_diff)!=sign(student_diff), divided by all teacher-valid pairs; student ties count as disagreement.",
            student_tie_given_teacher_nonzero="Teacher-valid pairs with student_diff==0 divided by all teacher-valid pairs.",
            uncertainty="Paired UID bootstrap; conditional ratios pool pair counts after each UID resample. All-teacher-tie draws are undefined and counted.",
            scope="Teacher-order consistency within selected panels, not observed-feedback AUC or actual retrieval accuracy."),
        ridge_fitting_and_validation_dense_flops_estimate=int(fitting_flops),
        main_linear_prediction_dense_flops_estimate=int(prediction_flops),
        arithmetic_scope="Coarse dense matrix/solve/prediction arithmetic; target centering/statistical bootstrap/I/O are analysis overhead, not a serving-method cost.",
        research_collection_costs=runner["research_collection_costs"],
        confidence=dict(method="paired UID-cluster bootstrap", replicates=replicates, seed=seed,
            scope="pointwise, conditional on these F weights, projection and derangement; not training-seed or multiple-head confirmation"),
        controls=dict(permutation_seed=29, permutation="independent deterministic UID-keyed derangement within16; native candidates and targets unchanged",
            features="item-only fixed16dim projection, per-panel candidate centering then fit-only feature RMS",
            target_scaling="read: fit-only RMS per layer/head; logit: original units; separately selected lambda by task/condition/panel",
            prediction="linear residual descriptor only; no read-path intervention or online Design evaluated"),
        provenance=dict(configuration_sha256=sha(root/"configuration.json"), runner_summary_sha256=sha(root/"summary.json"),
            analysis_source_sha256=sha(Path(__file__)), input_root=str(root)),
        limitations=["Teacher-only near-score panel selection changes the evaluated candidate distribution; no real task AUC or retrieval-quality claim.",
            "Targets and item features are panel-centered. This is a fixed-panel diagnostic, not a single-query serving gate.",
            "Candidate-centered signal may be tiny relative to total residual; both scales are retained.",
            "Real-vs-permutation evidence concerns item information for this simple descriptor, not gate necessity, optimality, or final Design effectiveness.",
            "Held-out UIDs do not imply every candidate item ID was absent from calibration.",
            "All heads are retained; per-head intervals are pointwise and not adjusted for multiple comparisons.",
            "Fewer than512 unique main users is diagnostic only; this planned report requires complete100fit/16validation/2000main."])
    return summary, headwise, model_arrays


def write_outputs(summary, heads, arrays, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    np.savez(output/"ridge_weights.npz", **arrays)
    with (output/"headwise.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(heads[0]));writer.writeheader();writer.writerows(heads)
    def stat(s):
        return "undefined" if s["mean"] is None else f"{s['mean']:.6g} [{s['ci95'][0]:.6g}, {s['ci95'][1]:.6g}]"
    lines=[f"# 候选残差描述：edge{summary['edge']}", "", f"固定{summary['main_users']}名main用户；{summary['confidence']['replicates']}次配对UID区间。",
        "这是离线描述，不注入读取路径，不计算真实AUC。近分panel只用teacher选择；broad16取同64池前8与后8，非旧Q16复刻。", "",
        "| panel | 目标 | 常数中心化MSE | 真实物品MSE | 置换物品MSE | 置换−真实 | 真实改进/中心化误差(%) | 真实改进/总误差(%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["points"]:
        fields=[row["panel"],row["task"],*[stat(row["method_mse"][c]) for c in CONDITIONS],stat(row["permutation_minus_real_mse"]),
            stat(row["real_improvement_percent_of_centered"]),stat(row["real_improvement_percent_of_total"])]
        lines.append("| "+" | ".join(fields)+" |")
    lines += ["", "真实与置换条件分别从头用相同闭式ridge和lambda网格拟合，按独立val16选lambda。所有归一化尺度仅来自fit100。",
        "read目标用F自己的q读取teacher KV，以免把下层query差异混入局部残差。read按历史长度归一化；主表保留物理单位，拟合时各层头用fit-only尺度平衡。",
        "常数对panel内去均值分量预测0。教师均值仅定义训练/评价目标，不作预测输入；特征panel均值来自已知候选。",
        "全部层头点见headwise.csv，不能挑选正头冒充普遍规律；负收益同样保留。", "",
        "## 候选次序几何的定义", "",
        "每用户16候选共有120个无序对。strict_opposite只计teacher差与student差严格异号；它排除了student ties。另报teacher差非零的支持率，以及在这些有效对中sign不相等的比例，后者把student ties计为不一致。student tie同时按全部对及teacher有效对两个分母报告。",
        "tie指保存float32分数的精确相等，没有人为epsilon。条件比例按UID重采样后汇总分子/有效对数量计算；没有teacher有效对的bootstrap draw记为undefined。支持数及全部定义见summary.json。它们是teacher次序一致性，不是真实AUC或检索准确率。", "",
        *["- "+x for x in summary["limitations"]]]
    if not summary["quality_use_permitted"]:lines.insert(2,"**诊断运行：不能用作人口机制结论。**")
    (output/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replicates",type=int,default=500);p.add_argument("--seed",type=int,default=17)
    args=p.parse_args()
    result, heads, models=analyze(args.input,replicates=args.replicates,seed=args.seed)
    write_outputs(result,heads,models,args.output)
    print(json.dumps(dict(status="complete",main_users=result["main_users"],quality_use_permitted=result["quality_use_permitted"])))

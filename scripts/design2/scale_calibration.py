"""One frozen, CPU-only target/length residual calibration experiment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from design2.audit_budget import curve as budget_curve, finite_metrics
from design2.report_detection import auc, mean, quantile, weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/design2/scale_calibration_01.json"
FROZEN = ROOT / "configs/design2/scale_calibration_01_fitted.json"
OUT = ROOT / "results/design2/analysis/scale_calibration_01"
METHODS = ("raw", "background", "calibrated")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(finite_metrics(obj), indent=2, allow_nan=False) + "\n")


def uid_folds(uids, seed, folds):
    ordered = np.random.default_rng(seed).permutation(np.sort(np.unique(uids)))
    return {int(uid): int(i % folds) for i, uid in enumerate(ordered)}


def make_groups(cal, cfg):
    """Residual-free common map, sufficient UID support in every training fold."""
    bounds = cfg["base_length_upper_bounds"]
    rows = []
    for target in cfg["targets"]:
        f = cal[cal.target == target]
        spans = [[1 if i == 0 else bounds[i-1] + 1, hi] for i, hi in enumerate(bounds)]

        def counts(span):
            g = f[f["count"].between(*span)]
            return [int(g[g.fold != k].uid.nunique()) for k in range(cfg["folds"])]

        while len(spans) > 1:
            sparse = next((i for i, span in enumerate(spans)
                           if min(counts(span)) < cfg["minimum_train_uids_per_group"]), None)
            if sparse is None:
                break
            left = sparse if sparse < len(spans)-1 else sparse-1
            spans[left:left+2] = [[spans[left][0], spans[left+1][1]]]
        for lo, hi in spans:
            support = counts([lo, hi])
            assert min(support) >= cfg["minimum_train_uids_per_group"]
            rows.append(dict(group=f"m{target}_n{lo}-{hi}", target=int(target), lower=lo, upper=hi,
                calibration_uids=int(f[f["count"].between(lo, hi)].uid.nunique()),
                training_fold_uids=support))
    return rows


def assign_groups(frame, groups):
    out = frame.copy()
    out["group"] = ""
    for g in groups:
        mask = (out.target == g["target"]) & out["count"].between(g["lower"], g["upper"])
        assert (out.loc[mask, "group"] == "").all()
        out.loc[mask, "group"] = g["group"]
    assert (out.group != "").all()
    return out


def fit_affine(u, e, w):
    """Nonnegative intercept/slope, with a coordinate change only for conditioning."""
    w = w / w.sum()
    scale = float(np.sqrt(np.sum(w * u*u)))
    x = u / scale if scale else u
    design = np.column_stack((np.ones(len(u)), x))
    beta, _ = nnls(design * np.sqrt(w[:, None]), e * np.sqrt(w))
    return dict(b=float(beta[0]), a=float(beta[1]/scale) if scale else 0.,
        background=float(np.sum(w*e)), conditioning_rms=scale,
        geometric_weighted_mse=float(np.sum(w*(e-design@beta)**2)),
        background_weighted_mse=float(np.sum(w*(e-np.sum(w*e))**2)))


def fit(frame):
    f = frame.assign(fit_weight=weights(frame))
    return {name: dict(**fit_affine(g.detection.to_numpy(), g.max_abs_error.to_numpy(),
                g.fit_weight.to_numpy()), users=int(g.uid.nunique()), states=len(g))
            for name, g in f.groupby("group", sort=True)}


def predict(frame, params):
    f = frame.copy()
    f["raw"] = f.detection
    f["base_component"] = f.group.map(lambda g: params[g]["b"])
    f["geometry_component"] = f.detection * f.group.map(lambda g: params[g]["a"])
    f["calibrated"] = f.base_component + f.geometry_component
    f["background"] = f.group.map(lambda g: params[g]["background"])
    return f


def describe(frame, score, thresholds, errors):
    w = weights(frame)
    s, e = frame[score].to_numpy(), frame.max_abs_error.to_numpy()
    records = []
    for q, threshold in thresholds.items():
        keep = s <= threshold
        records.append(dict(quantile=float(q), threshold=threshold, coverage=float(w[keep].sum()/w.sum()),
            states=int(keep.sum()), uids=int(frame.loc[keep, "uid"].nunique()),
            severe_rates=[mean((e[keep] >= t).astype(float), w[keep]) for t in errors],
            severe_states=[int((keep & (e >= t)).sum()) for t in errors],
            severe_uids=[int(frame.loc[keep & (e >= t), "uid"].nunique()) for t in errors],
            mean_error=mean(e[keep], w[keep]), max_error=float(e[keep].max()) if keep.any() else None))
    return dict(auroc=[auc(e >= t, s, w) for t in errors],
        mse_to_residual=float(np.sum(w*(s-e)**2)/w.sum()) if score != "raw" else None,
        acceptance=records)


def calibrate(cfg):
    started = time.perf_counter()
    assert not FROZEN.exists() and not OUT.exists(), "Preserve frozen experiment; no overwrite."
    OUT.mkdir(parents=True)
    source = ROOT / cfg["input"]
    cal = pd.read_parquet(source, filters=[("role", "==", cfg["calibration_role"])])
    cal = cal.sort_values(["uid", "target", "state_ordinal"]).reset_index(drop=True)
    old = json.loads((ROOT / "configs/design2/detection_01.json").read_text())
    assert set(cal.uid) == set(old["groups"]["residual_calibration"])
    folds = uid_folds(cal.uid, cfg["fold_seed"], cfg["folds"])
    cal["fold"] = cal.uid.map(folds)
    # Grouping receives no residual, score or failure indicator.
    groups = make_groups(cal[["uid", "target", "count", "fold"]], cfg)
    cal = assign_groups(cal, groups)
    parts, fold_fits = [], {}
    for k in range(cfg["folds"]):
        train, held = cal[cal.fold != k], cal[cal.fold == k]
        assert not set(train.uid) & set(held.uid)
        params = fit(train)
        fold_fits[str(k)] = params
        parts.append(predict(held, params))
    oof = pd.concat(parts).sort_index()
    assert len(oof) == len(cal) and not oof.index.duplicated().any()
    w = weights(oof)
    thresholds = {s: {str(q): quantile(oof[s].to_numpy(), w, q)
                      for q in cfg["acceptance_quantiles"]} for s in METHODS}
    params = fit(cal)
    final_cal = predict(cal, params)
    frozen = dict(status="fitted_only_on_original512_calibration_uids_before_new_dev_predictions",
        configuration_sha256=digest(CONFIG), input_sha256=digest(source), source_sha256=digest(Path(__file__)),
        groups=groups, uid_folds=folds, fold_fits=fold_fits, final_fit=params, thresholds=thresholds,
        calibration_users=int(cal.uid.nunique()), calibration_states=len(cal),
        development_rows_used=0, new_model_forward_calls=0, elapsed_seconds=time.perf_counter()-started)
    write(FROZEN, frozen)
    write(OUT / "configuration.json", dict(protocol=cfg, frozen=frozen))
    write(OUT / "calibration.json", dict(
        oof={s: describe(oof, s, thresholds[s], cfg["error_thresholds"]) for s in METHODS},
        final_in_sample={s: describe(final_cal, s, thresholds[s], cfg["error_thresholds"]) for s in METHODS},
        folds={str(k): {s: describe(g, s, thresholds[s], cfg["error_thresholds"]) for s in METHODS}
               for k, g in oof.groupby("fold")},
        note="OOF quantiles are frozen; final-in-sample is only a shift diagnostic, not held-out performance."))
    oof.to_parquet(OUT / "calibration_oof.parquet", index=False)
    print(json.dumps(dict(stage="calibration_frozen", groups=len(groups), users=512, states=len(cal),
        seconds=time.perf_counter()-started, development_rows_used=0)), flush=True)


def paired_bootstrap(dev, cfg):
    w, e = weights(dev), dev.max_abs_error.to_numpy()
    uu, index = np.unique(dev.uid.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    auc_draws, budget_draws = [], []
    scores = {s: dev[s].to_numpy() for s in METHODS}
    cost = dev.rebuild_flops.to_numpy()
    for _ in range(cfg["bootstrap_draws"]):
        m = np.bincount(rng.integers(len(uu), size=len(uu)), minlength=len(uu))[index]
        av = [[auc(e >= t, scores[s], w*m) for t in cfg["error_thresholds"]] for s in METHODS]
        auc_draws.append(av)
        curves = [budget_curve(dev, scores[s]/cost, multiplicity=m) for s in ("calibrated", "background")]
        budget_draws.append(np.array([r["recall"] for r in curves[0]]) - np.array([r["recall"] for r in curves[1]]))
    av = np.array(auc_draws, dtype=float)
    delta = av[:, 2] - av[:, 1]
    budgets = np.array(budget_draws)
    return dict(auroc_ci={s: np.nanquantile(av[:, i], [.025, .975], axis=0).T.tolist()
                         for i, s in enumerate(METHODS)},
        calibrated_minus_background_auroc_ci=np.nanquantile(delta, [.025, .975], axis=0).T.tolist(),
        valid_auroc_draws=np.isfinite(delta).sum(0).tolist(),
        calibrated_minus_background_budget_ci=np.nanquantile(budgets, [.025, .975], axis=0).transpose(1, 2, 0).tolist(),
        scope="Paired development UID resampling, frozen calibration map/coefficients/thresholds; no claim to include calibration uncertainty or training seed variability.")


def evaluate(cfg):
    started = time.perf_counter()
    assert not (OUT / "summary.json").exists(), "Preserve completed evidence."
    frozen = json.loads(FROZEN.read_text())
    source = ROOT / cfg["input"]
    assert digest(CONFIG) == frozen["configuration_sha256"]
    assert digest(source) == frozen["input_sha256"]
    assert digest(Path(__file__)) == frozen["source_sha256"]
    dev = pd.read_parquet(source, filters=[("role", "==", cfg["development_role"])])
    dev = dev.sort_values(["uid", "target", "state_ordinal"]).reset_index(drop=True)
    old = json.loads((ROOT / "configs/design2/detection_01.json").read_text())
    assert set(dev.uid) == set(old["groups"]["development"])
    assert not set(dev.uid) & set(map(int, frozen["uid_folds"]))
    dev = predict(assign_groups(dev, frozen["groups"]), frozen["final_fit"])
    errors, thresholds = cfg["error_thresholds"], frozen["thresholds"]
    results = {s: describe(dev, s, thresholds[s], errors) for s in METHODS}
    budgets = {s: budget_curve(dev, dev[s].to_numpy()/dev.rebuild_flops.to_numpy()) for s in METHODS}
    budgets["cost_only"] = budget_curve(dev, -dev.rebuild_flops.to_numpy())
    for s in ("version_age", "short_history", "source_norm"):
        budgets[s] = budget_curve(dev, dev[s].to_numpy()/dev.rebuild_flops.to_numpy())
    boot = paired_bootstrap(dev, cfg)
    w, e = weights(dev), dev.max_abs_error.to_numpy()
    overall = [mean((e >= t).astype(float), w) for t in errors]
    for s in METHODS:
        results[s]["auroc_uid95"] = boot["auroc_ci"][s]
    retain = next(v for v in results["calibrated"]["acceptance"] if v["quantile"] == .8)
    low = [v for v in results["calibrated"]["acceptance"] if v["quantile"] in (.1, .2, .5)]
    gates = dict(geometry_increment=boot["calibrated_minus_background_auroc_ci"][0][0] > 0,
        retain_coverage=retain["coverage"] >= .7,
        retain_risk=retain["severe_rates"][0] is not None and retain["severe_rates"][0] <= overall[0]/2,
        low_regions=all(v["states"] > 0 and v["severe_rates"][0] <= overall[0] for v in low))
    groups = []
    # Global UID/target weighting also governs how much coverage each group loses.
    for name, f in dev.assign(global_weight=w).groupby("group"):
        ww = f.global_weight.to_numpy()
        row = dict(group=name, global_weight_fraction=float(ww.sum()/w.sum()),
            mean_actual_residual=mean(f.max_abs_error.to_numpy(), ww),
            serious_states=int((f.max_abs_error >= .5).sum()),
            serious_uids=int(f.loc[f.max_abs_error >= .5, "uid"].nunique()),
            serious_rate=mean((f.max_abs_error >= .5).to_numpy().astype(float), ww),
            **frozen["final_fit"][name])
        # Fit users/states are explicitly separate from development support.
        row.update(fit_users=row.pop("users"), fit_states=row.pop("states"),
                   development_users=int(f.uid.nunique()), development_states=len(f))
        for s in METHODS:
            keep = f[s].to_numpy() <= thresholds[s]["0.8"]
            row[s+"_retained_fraction"] = float(ww[keep].sum()/ww.sum())
            row[s+"_lost_global_coverage"] = float(ww[~keep].sum()/w.sum())
            row[s+"_retained_severe_states"] = int((keep & (f.max_abs_error.to_numpy() >= .5)).sum())
            row[s+"_within_group_auroc"] = auc(f.max_abs_error.to_numpy() >= .5, f[s].to_numpy(), ww)
        groups.append(row)
    pd.DataFrame(groups).to_csv(OUT / "group_diagnostics.csv", index=False)
    for s in METHODS:
        for q in ("0.1", "0.8"):
            dev[f"{s}_keep{q}"] = dev[s] <= thresholds[s][q]
    dev[dev.max_abs_error >= .5].to_csv(OUT / "all_severe_development.csv", index=False)
    known = pd.read_parquet(source, filters=[("uid", "in", cfg["known_uid_audit"])])
    known = predict(assign_groups(known, frozen["groups"]), frozen["final_fit"])
    for s in METHODS:
        known[s+"_keep80"] = known[s] <= thresholds[s]["0.8"]
    known.to_csv(OUT / "known_failures.csv", index=False)
    dev.to_parquet(OUT / "development_predictions.parquet", index=False)
    audit = json.loads((ROOT / "results/design2/analysis/budget_audit_01/summary.json").read_text())
    costs = dict(audit["cost"], affine_score_flops_per_state=2,
        affine_total_flops=2*len(dev), calibration_seconds=frozen["elapsed_seconds"],
        background_geometry_check_flops=0, new_model_forward_calls=0,
        existing_teacher_scope="512 calibration UID residuals from retained Current-at-decision Exact panels; no new teacher computation. Original teacher/preparation cost records remain dependencies.")
    total = {s: [dict(**r, total_arithmetic_lower_bound_fraction=r["actual_rebuild_fraction"] +
                ((costs["check_all_panel_flops"]+costs["shared_geometry_preparation_leading_flops"]+
                  (2*len(dev) if s == "calibrated" else 0))/costs["exact_all_panel_flops"] if s in ("raw", "calibrated") else 0))
             for r in budgets[s]] for s in (*METHODS, "cost_only")}
    summary = dict(status="complete", users=int(dev.uid.nunique()), states=len(dev), queries=len(dev)*16,
        errors=errors, overall_severe_rates=overall, serious_states=[int((e >= t).sum()) for t in errors],
        serious_uids=[int(dev.loc[e >= t, "uid"].nunique()) for t in errors],
        methods=results, budgets=budgets, paired=boot, groups=groups, costs=costs,
        total_budget_curves=total, gates=gates, advance=all(gates.values()),
        strata={field: {str(value): {s: describe(f, s, thresholds[s], errors) for s in METHODS}
                       for value, f in dev.groupby(field)} for field in ("target", "kind")},
        calibration=json.loads((OUT / "calibration.json").read_text()),
        known_failures=known[known.max_abs_error >= .5].to_dict("records"),
        model_forward_calls=0, elapsed_seconds=time.perf_counter()-started,
        frozen_sha256=digest(FROZEN))
    write(OUT / "summary.json", summary)
    print(json.dumps(dict(stage="evaluation_complete", users=1024, states=len(dev), gates=gates,
        main_auroc={s: results[s]["auroc"][0] for s in METHODS},
        seconds=time.perf_counter()-started)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "evaluate"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    (calibrate if args.stage == "calibrate" else evaluate)(config)

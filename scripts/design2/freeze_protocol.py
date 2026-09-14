"""Freeze medium detection panels using only existing UID metadata."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from design.data import DATASET, ROOT, diagnostic_admissions
from design2.common import COHORT, CROOT, PROJECTION, TARGETS


def main():
    out = ROOT/"configs/design2/detection_01.json"
    if out.exists():
        raise FileExistsError(out)
    split_path = ROOT/"data/manifests/evokv_design_medium_6000_v1/split.json"
    split = json.loads(split_path.read_text())
    excluded = set(COHORT["original_fitting_uids"]+COHORT["additional_fitting_uids"])
    for path in sorted((ROOT/"results/design").glob("*/configuration.json")):
        config = json.loads(path.read_text())
        for key in ("calibration_uids","fitting_uids"):
            excluded.update(config.get(key, []))
    available = set(split["development"])-excluded
    forbidden = set(split["confirmation"]+split["reserved_legacy_confirmation"])
    assert not available & forbidden
    users = pq.read_table(DATASET.parent/"users.parquet",columns=["uid","n_theta0"]).to_pandas()
    users = users[users.uid.isin(available)].copy()
    users["stratum"] = np.searchsorted([256,1024,4096],users.n_theta0,side="right")
    groups = {}
    for role,total in (("residual_calibration",512),("development",1024)):
        sizes = users.groupby("stratum").size().reindex(range(4),fill_value=0).to_numpy()
        quota = sizes/sizes.sum()*total
        ns = np.floor(quota).astype(int)
        for j in np.argsort(-(quota-ns),kind="stable")[:total-ns.sum()]:
            ns[j] += 1
        chosen = []
        for j,n in enumerate(ns):
            candidates = users.loc[users.stratum==j,"uid"].astype(int).tolist()
            candidates.sort(key=lambda u:hashlib.sha256(f"design2-detection01:{role}:{u}".encode()).digest())
            chosen.extend(candidates[:n])
        # Interleave strata so every small resource probe also includes short users.
        by_stratum = [chosen[sum(ns[:j]):sum(ns[:j+1])] for j in range(4)]
        from itertools import zip_longest
        groups[role] = [u for row in zip_longest(*by_stratum) for u in row if u is not None]
        users = users[~users.uid.isin(chosen)]
    groups["historical_diagnostic"] = COHORT["diagnostic_uids"]
    groups["fitting_check"] = COHORT["original_fitting_uids"]+COHORT["additional_fitting_uids"]
    sets = [set(v) for v in groups.values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert not set.union(*sets)&forbidden
    weights = [*(CROOT.glob("translator_C_m*.pt")),*(PROJECTION.glob("source_projection_mean_m*.pt"))]
    record = dict(status="frozen_before_design2_outputs",groups=groups,
        selection="label-free initial-length proportional quotas and role/UID SHA256; already-opened development only",
        excluded_design1_fitting_uids=sorted(excluded),split_path=str(split_path.relative_to(ROOT)),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        frozen_weights_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in weights},
        targets=list(TARGETS),no_op_targets=[2],seed=17,confirmation_read=False,
        reference="Current rebuild of legal retained prefix at decision cutover; never release-rebuild-and-replay teacher",
        query_panel="16 distinct held-bank items; all queries at cutover, before same-timestamp writes",
        source_schedule="original source-only continuous/early/adjacent; six lifetime tails plus exact-source control for first quarter per role",
        fitting_lifetime_uids=COHORT["lifetime_fitting_uids"],
        lifetime_uids={role:uids[:len(uids)//4] for role,uids in groups.items()},
        score="max_active_layer_head N*sqrt(f^T H^-1 f); then max over 16 queries",
        primary_error="max absolute logit error over 16-query state panel",
        severe_absolute_logit_error=.5,secondary_error_thresholds=[.1,1.],
        acceptance_quantiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.],
        budget_fractions=[.05,.1,.2,.3,.5],
        controls={"version_age":"larger producer-weighted target-minus-producer age first",
                  "short_history":"smaller N first","source_norm":"larger frozen standardized source norm first"},
        budget_rule="descending score / rebuild FLOPs; UID-state counted once; compare uncosted ordering separately",
        aggregation="each UID equal total weight, each target equal, states equal within UID-target; cluster resample UID",
        primary_report="independent development; historical128 and fitting256 separate; all targets/state types retained",
        advance_rule="research signal only: development AUROC>=.70 with UID 95% CI lower>.50, retained80 severe rate<=half overall and actual coverage>=.70, top20% FLOPs covers>=40% severe states; inspect each target/type and false negatives",
        scope="detection diagnostic only; no C refit, sensitivity, scheduler, persistent rebuild or real task AUC claim",
        initial_resource_estimate=dict(seconds=1200,peak_gpu_mib=32000,
             basis="previous 384-UID A/B/C full diagnostic 302.55s; 1536 primary plus128 historical and256 fitting; frozen single C,16 queries; exact FP64 geometry adds work; refine with focused probe"),
        admissions=diagnostic_admissions(5))
    record["lifetime_uids"]["fitting_check"] = COHORT["lifetime_fitting_uids"]
    out.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(dict(path=str(out),group_sizes={k:len(v) for k,v in groups.items()})))


if __name__ == "__main__":
    main()

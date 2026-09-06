#!/usr/bin/env python3
"""Development-only response predictability and candidate-distribution probe.

The shared ridge maps below are diagnostics, not a new executable cache action.
They predict response means at the Reuse path's actual queries; they do not
establish closed-loop recommendation quality or continuous compatibility.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design.data import (  # noqa: E402
    calibration_candidates,
    diagnostic_admissions,
    fixed_split,
    frozen_model,
    histories,
)
from design.run import cache_at, response_targets, tensor, timed, write_json  # noqa: E402
from insight.candidate_shared_causal import signed_head_intervention  # noqa: E402
from insight_one_locality.common import candidate_panel  # noqa: E402
from insight_two.common import (  # noqa: E402
    CUTOVER_DAYS,
    DAY,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
)

from hstu_kvcache.adaptation.reader import score  # noqa: E402


def ridge_predict(x, y, dev):
    # Fixed shared ridge; rows are calibration UIDs, never per-user fitting.
    center, scale = x.mean(0), x.std(0).clamp_min(1e-3)
    x, dev = (x-center)/scale, (dev-center)/scale
    mean = y.mean(0)
    gram = x @ x.T
    ridge = (gram.trace()/len(x) * 0.01).clamp_min(1e-6)
    alpha = torch.linalg.solve(gram + ridge*torch.eye(len(x), device=x.device), y-mean)
    return mean + dev @ x.T @ alpha


def main(run_id, targets, fit_users):
    import hashlib
    from collections import defaultdict

    out = ROOT / "results/design" / run_id
    out.mkdir(parents=True, exist_ok=False)
    split = fixed_split()
    calibration, development = split["calibration"][:fit_users], split["development"][:128]
    cfg = dict(run=run_id, calibration=calibration, development=development,
        targets=targets, ridge_relative=0.01, confirmation_read=False, labels_read=False,
        candidate_protocols=["original calibration random16", "history-derived matched64", "frozen development64"],
        estimate_seconds=[30,90], estimate_basis="15s history + 6s loads + at most 640 user cache pairs/target at 10ms + paired readers/oracles",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_configuration_reference="results/design/v4_full_history512_01/configuration.json",
        admission_inputs=diagnostic_admissions(max(targets)),
        scope="development diagnostic; no model selection or serving-quality claim")
    write_json(out/"configuration.json", cfg)
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ledger = defaultdict(float)
    started = time.perf_counter()
    history = timed(lambda: histories(calibration+development, max(246,CUTOVER_DAYS[max(targets)-1]+1)), ledger, "history_io")
    _, panels, _ = load_frozen_inputs()
    summaries, raw = [], {}
    for target in targets:
        previous = frozen_model(target-1, device)
        current = frozen_model(target, device)
        cutover = CUTOVER_DAYS[target-1]*DAY
        cal_items = np.stack([history.prefix(uid,cutover,1024)[0] for uid in calibration])
        matched, _, _ = candidate_panel(cal_items)
        records, oracle_rows = defaultdict(list), []
        with torch.no_grad():
            for cohort, users in (("calibration",calibration),("development",development)):
                for index, uid in enumerate(users):
                    old, times = cache_at(previous,history,uid,cutover)
                    exact, _ = cache_at(current,history,uid,cutover)
                    feature = torch.stack((old.k[:,0].mean(1),old.v[:,0].mean(1)),1).flatten()
                    delta = tensor([cutover-int(times[-1])],device,floating=True)
                    bank = matched[index] if cohort=="calibration" else panels[target-1,index]
                    candidates = {"matched64":tensor(bank[None],device)}
                    if cohort=="calibration":
                        candidates["random16"] = tensor([calibration_candidates(history,uid,cutover)],device)
                    for protocol, ids in candidates.items():
                        reuse, trace = score(current,old,ids,delta,trace=True)
                        changes = response_targets(current,trace,old,exact,
                            torch.ones(old.seq_len,device=device,dtype=torch.bool))
                        means = torch.stack([value.mean(2).flatten() for value in changes])
                        queries = torch.stack([q.mean(2).flatten() for q in trace.queries])
                        residual_fraction = torch.stack([
                            (value-value.mean(2,keepdim=True)).square().sum()/value.square().sum().clamp_min(1e-20)
                            for value in changes])
                        records[f"{cohort}_{protocol}"].append(dict(uid=uid, source=feature.cpu(),
                            query=queries.flatten().cpu(), response=means.cpu(), residual_fraction=residual_fraction.cpu()))
                        if cohort=="development":
                            reference = score(current,exact,ids,delta)[0]
                            oracle = signed_head_intervention(current,exact,old,ids,delta,mode="shared_only")
                            oracle_rows.append(dict(uid=uid,**metrics_row(score_metrics(reference,reuse,oracle.scores))))
        dev = records["development_matched64"]
        dx = torch.stack([row["source"] for row in dev]).to(device)
        dq = torch.stack([row["query"] for row in dev]).to(device)
        dy = torch.stack([row["response"] for row in dev]).to(device)
        results = []
        for protocol in ("random16","matched64"):
            cal = records[f"calibration_{protocol}"]
            x = torch.stack([row["source"] for row in cal]).to(device)
            q = torch.stack([row["query"] for row in cal]).to(device)
            y = torch.stack([row["response"] for row in cal]).to(device)
            for size in sorted({min(128,len(cal)),len(cal)}):
                for inputs in ("source", "source_and_actual_reuse_query"):
                    features = x if inputs=="source" else torch.cat((x,q),1)
                    dev_features = dx if inputs=="source" else torch.cat((dx,dq),1)
                    predicted = ridge_predict(features[:size],y[:size].flatten(1),dev_features).reshape_as(dy)
                    relative = (predicted-dy).square().sum((0,2))/dy.square().sum((0,2)).clamp_min(1e-20)
                    results.append(dict(protocol=protocol,inputs=inputs,calibration_users=size,layer_relative_mse=relative.cpu().tolist(),
                                        mean_layer_relative_mse=float(relative.mean())))
        shared_recovery = np.mean([row["probability_gap_recovery"] for row in oracle_rows])
        summaries.append(dict(target=target,ridge=results,
            shared_response_oracle_mean_user_recovery=float(shared_recovery),
            dev_candidate_residual_energy_fraction_by_layer=torch.stack([r['residual_fraction'] for r in dev]).mean(0).tolist()))
        raw[str(target)] = dict(records=dict(records),oracle=oracle_rows)
        print(json.dumps(summaries[-1]),flush=True)
        torch.save(raw,out/"response_raw.pt")
    write_json(out/"summary.json",dict(status="diagnostic_complete", results=summaries,
        elapsed_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        ledger_seconds=dict(ledger),configuration_sha256=hashlib.sha256((out/"configuration.json").read_bytes()).hexdigest()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--targets",type=int,nargs="+",choices=(1,2,3,4,5),default=[1,2])
    parser.add_argument("--fit-users",type=int,default=128)
    cli=parser.parse_args()
    main(cli.run_id,cli.targets,cli.fit_users)

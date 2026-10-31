#!/usr/bin/env python3
"""V4-only development comparison for declared fit counts and retained fit64.

An original V3 session is replayed once and forked at V4 publication. Each V4
branch then owns its writes and codes. Only fixed registered32 / first128
development users are scored; original64 scores and native controls are reused.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import copy
from functools import partial
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from hstu_kvcache.evaluation.release_context import RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.cachetide.backend import InferenceAdapter, ImplementationConfig, configure_prefix
from hstu_kvcache.methods.cachetide.prepare import compile_item_tables, fixed_features_from_bank
from experiments.archive.query_relay.evaluate import IDENTITY, load_candidates
from hstu_kvcache.methods.cachetide.features import load_bank
from hstu_kvcache.methods.cachetide.precision import load_stable_prefix
from hstu_kvcache.methods.cachetide.session import RelaySession
from hstu_kvcache.evaluation.cachetide_workload import candidate_rows
from hstu_kvcache.common.artifacts import sha256, write_json

BASE = ROOT/"results/query_relay_2026_10/probes/A_itemgate6_rms_roll4"
REFERENCES = dict(behavior_stride16_candidates16=BASE/"population4096/merged_actual/scores.parquet",
                 original_feedback=BASE/"feedback4096_actual/merged/scores.parquet")


def fork_session(source):
    trajectory = copy.copy(source.trajectory)
    trajectory.cache = HSTUKVCache(source.cache.k.clone(), source.cache.v.clone(), source.cache.seq_len)
    trajectory.producers = source.trajectory.producers.copy()
    trajectory.ledger = source.trajectory.ledger.copy()
    trajectory.on_append = trajectory.append_context = None
    branch = RelaySession(trajectory)
    for name, value in source.__dict__.items():
        if name not in ("trajectory", "state"):
            # Adapters/publications are immutable and shared. copy.copy on the
            # inference proxy invokes its delegation before construction.
            setattr(branch, name, value if name in ("adapter", "prefix") else copy.copy(value))
    if source.state is not None:
        branch.state = LightweightValueState(source.state.codes.clone(), source.state.seq_len)
    return branch


def analyze(output, config):
    rows = pq.read_table(output/"scores.parquet").to_pylist()
    result = dict(scope="Single V4 edge with frozen original V3 source; fixed development panels, no fit-size choice or population/continuous-chain claim", panels={})
    for panel_name, users in (("registered32", config["evaluation_uids"]),
                              ("fixed_first128", config["trend_evaluation_uids"])):
        result["panels"][panel_name] = {}
        for protocol in config["protocols"]:
            group = [r for r in rows if r["uid"] in users and r["protocol"] == protocol]
            labels = np.array([r["label"] for r in group])
            methods = [f"fit{count}" for count in config["fit_sizes"]]+["fit64"]
            fields = ["reuse", "rebuild"]+methods
            auc = {field:float(roc_auc_score(labels,[r[field] for r in group])) for field in fields} if len(set(labels))>1 else {}
            denominator = auc.get("rebuild", 0)-auc.get("reuse", 0)
            result["panels"][panel_name][protocol] = dict(users=len(users), users_with_scores=len(set(r["uid"] for r in group)),
                candidate_scores=len(group), positive_labels=int(labels.sum()), auc=auc,
                auc_delta_to_fit64={k:auc[k]-auc["fit64"] for k in methods if k!="fit64"} if auc else {},
                recovery_percent={k:100*(auc[k]-auc["reuse"])/denominator for k in methods} if denominator>0 else None,
                rebuild_minus_reuse_auc=denominator,
                metric="Pooled within V4 AUC; positive-denominator recovery descriptive only. No averaging of per-release recovery.")
    write_json(output/"quality.json",result)
    return result


@torch.no_grad()
def run(args):
    start = time.perf_counter()
    config = json.loads(args.config.read_text())
    counts = config["fit_sizes"]
    if args.output.exists():
        raise FileExistsError("new evaluation attempt required")
    args.output.mkdir(parents=True)
    users = config["evaluation_union_uids"]
    panel_path = ROOT/config["evaluation_panel"]
    panel, roles, feedback = load_panel(panel_path)
    if not set(users)<=set(roles["evaluation"]) or set(users)&set(roles["fit"]+roles["validation"]):
        raise ValueError("development panel role mismatch")
    registration = json.loads((ROOT/"results/implementation_2026_10/registration.json").read_text())
    if config["evaluation_uids"] != registration["selected_uids"] or config["trend_evaluation_uids"] != roles["evaluation"][:128]:
        raise ValueError("fixed evaluation membership changed")
    references, bindings = {}, {}
    for protocol, path in REFERENCES.items():
        old = pq.read_table(path, filters=[("uid","in",users),("version","=","v4")]).to_pylist()
        references[protocol] = {r["request_id"]:r for r in old}
        old_config = json.loads(path.with_name("configuration.json").read_text())
        if old_config["panel_sha256"] != sha256(panel_path/"binding.json"):
            raise ValueError("retained reference panel differs")
        bindings[protocol] = dict(path=str(path),sha256=sha256(path),configuration_sha256=sha256(path.with_name("configuration.json")),rows=len(old))
    device = setup_device(0,2)
    models = load_models(panel,device,last_version=4)
    histories = load_history(panel_path,requested_uids=users)
    original, old_weights, _ = load_candidates(BASE/"calibration",device,["v3","v4"],users)
    for version in ("v3","v4"):
        if old_weights[version]["sha256"] != json.loads((BASE/"freeze.json").read_text())["release_bindings"][version]["calibration_sha256"]:
            raise ValueError("original fitted weights differ from frozen reference")
    bank = load_bank(original["v3"].feature_bank_path,original["v3"].feature_bank_sha256,device=device,dtype=torch.float32)
    feature = partial(fixed_features_from_bank,bank=bank)
    impl = ImplementationConfig(row_chunk=1024)
    p3, p3_meta = load_stable_prefix(BASE/"publication/v3.pt",original["v3"],device=device)
    initial_prefix = configure_prefix(p3,impl,feature_fn=feature)
    initial_reader = InferenceAdapter(original["v3"],impl)
    readers, prefixes, weights = {}, {}, {}
    for count in counts:
        adapters, record, _ = load_candidates(args.fits/f"fit{count}/calibration",device,["v4"],users)
        current, work = compile_item_tables(adapters["v4"],models["v4"],p3)
        readers[count] = InferenceAdapter(adapters["v4"],impl)
        prefixes[count] = configure_prefix(current,impl,feature_fn=feature)
        weights[str(count)] = record["v4"]
    # A small identity check uses the existing64 model; its quality scores for
    # all other users are reused, not rerun or refitted.
    p4, p4_meta = load_stable_prefix(BASE/"publication/v4.pt",original["v4"],device=device)
    control_prefix = configure_prefix(p4,impl,feature_fn=feature)
    control_reader = InferenceAdapter(original["v4"],impl)
    settings = dict(config=config,config_sha256=sha256(args.config),entry_sha256=sha256(Path(__file__)),
        weights=weights,original_weights=old_weights,references=bindings,
        semantics="V3 recommendation timestamps prepare codes without rescoring; read-only scores do not change states. Fork both own V4 trajectories at publication.",
        renderer="Existing validated InferenceAdapter and in-place conversion, row_chunk1024; no new fused conversion kernel",scope="Development fit-size probe, not throughput")
    write_json(args.output/"configuration.json",settings)
    rows, checks = [], []
    for protocol in config["protocols"]:
        requests = ({u:candidate_rows(u,histories.rows[u],stride=16,candidates=16,known_vocab_size=panel["known_vocab_size"]) for u in users}
                    if protocol.startswith("behavior") else feedback)
        for number, uid in enumerate(users):
            base = RelaySession(NativeTrajectory(models["v2"],2,histories.rows[uid],RELEASES[0][1]))
            base.publish(models["v3"],3,RELEASES[0][1],initial_reader,prefix=initial_prefix)
            for timestamp in sorted({int(r["query_timestamp"]) for r in requests[uid] if r["version"]=="v3"}):
                base.prepare_read(timestamp)
            base.advance(RELEASES[1][1])
            sessions = {count:fork_session(base) for count in counts}
            check64 = fork_session(base) if number<config.get("reference_check_users",4) else None
            for count,session in sessions.items():
                session.publish(models["v4"],4,RELEASES[1][1],readers[count],prefix=prefixes[count])
            if check64 is not None:
                check64.publish(models["v4"],4,RELEASES[1][1],control_reader,prefix=control_prefix)
            by_time=defaultdict(list)
            for request in requests[uid]:
                if request["version"]=="v4":by_time[int(request["query_timestamp"])].append(request)
            max_control_error = 0.
            for timestamp,pending in sorted(by_time.items()):
                # Keep original retained request order and query_batch16.
                for start_query in range(0,len(pending),16):
                    batch=pending[start_query:start_query+16]
                    candidate=torch.tensor([[r["item_idx"] for r in batch]],device=device)
                    values={count:session.score(timestamp,candidate).flatten().cpu().tolist() for count,session in sessions.items()}
                    control=None if check64 is None else check64.score(timestamp,candidate).flatten().cpu()
                    for j,request in enumerate(batch):
                        old=references[protocol][request["request_id"]]
                        if any(old[key]!=request[key] for key in IDENTITY):raise ValueError("retained candidate identity mismatch")
                        if control is not None:
                            err=abs(float(control[j])-old["partial_logit"])
                            if err>3e-5+3e-5*abs(old["partial_logit"]):raise ValueError("original64 replay does not match retained scores")
                            max_control_error=max(max_control_error,err)
                        rows.append(dict(protocol=protocol,**{k:request[k] for k in ("request_id",*IDENTITY)},
                            **{f"fit{count}":values[count][j] for count in counts},fit64=old["partial_logit"],
                            reuse=old["reuse_logit"],rebuild=old["first_request_rebuild_logit"]))
            if check64 is not None:checks.append(dict(protocol=protocol,uid=uid,max_abs=max_control_error))
            if (number+1)%8==0 or number+1==len(users):
                pq.write_table(pa.Table.from_pylist(rows),args.output/"scores.partial.parquet",compression="zstd")
                print(json.dumps(dict(protocol=protocol,users=number+1,total=len(users),scores=len(rows),elapsed_seconds=time.perf_counter()-start)),flush=True)
            del sessions, check64, base
    pq.write_table(pa.Table.from_pylist(rows),args.output/"scores.parquet",compression="zstd")
    (args.output/"scores.partial.parquet").unlink(missing_ok=True)
    quality=analyze(args.output,config)
    write_json(args.output/"summary.json",dict(status="complete",users=len(users),scores=len(rows),
        elapsed_seconds=time.perf_counter()-start,retained64_checks=checks,quality=quality,
        no_quality_selection=True,no_new_teacher=True,independent_confirmation=False))
    print(json.dumps(dict(status="complete",output=str(args.output))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/system_study_2026_10/fit_size_v4.json")
    p.add_argument("--fits",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())

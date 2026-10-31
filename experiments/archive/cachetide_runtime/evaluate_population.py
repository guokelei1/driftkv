#!/usr/bin/env python3
"""Recheck the complete, already revealed population with optimized inference.

The frozen model, code update, user roles, requests and native controls remain
unchanged. This is an engineering replay, not another independent quality test.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
from functools import partial
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from hstu_kvcache.evaluation.release_context import END, RELEASES, load_models, setup_device
from hstu_kvcache.evaluation.panels import load_history, load_panel
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.methods.cachetide.backend import ImplementationConfig, InferenceAdapter, configure_prefix
from hstu_kvcache.methods.cachetide.prepare import (
    compile_item_tables, fixed_features_from_bank, project_fixed_item_terms,
)
from experiments.archive.partial_refresh.analyze import summarize
from experiments.archive.query_relay.evaluate import IDENTITY, load_candidates, records
from hstu_kvcache.methods.cachetide.session import RelaySession
from hstu_kvcache.evaluation.cachetide_workload import candidate_rows
from hstu_kvcache.common.artifacts import sha256, write_json

PANEL = ROOT/"results/design_system_2026_10/phase2/representative4096/panel"
BASE = ROOT/"results/query_relay_2026_10/probes/A_itemgate6_rms_roll4"
VERSIONS = [version for version,_ in RELEASES]
ATOL, RTOL = 3e-5, 3e-5
SCORE_SCHEMA = pa.schema([(name,pa.string() if name in ("request_id","version") else pa.int64())
    for name in ("request_id",*IDENTITY)]+[(name,pa.float64()) for name in (
        "reuse_logit","full_logit","first_request_rebuild_logit","old_relay_logit","partial_logit")]
    +[(name,pa.int64()) for name in ("history_length","coded_rows")])


def workload(kind):
    return dict(kind=kind,stride=16 if kind=="behavior" else None,
        candidates=16 if kind=="behavior" else None,
        task="causal next-item with uniform sampled negatives" if kind=="behavior"
             else "retained feedback labels")


def reference_inputs(path, *, panel_hash, population, kind):
    """Require the whole retained panel, including reference users with no rows."""
    settings=json.loads((path.parent/"configuration.json").read_text())
    summary=json.loads((path.parent/"summary.json").read_text())
    if settings["panel_sha256"]!=panel_hash or settings["workload"]!=workload(kind):
        raise ValueError("retained scores belong to a different panel or workload")
    if len(summary["selected_uids"])!=len(population) or set(summary["selected_uids"])!=set(population):
        raise ValueError("reference must retain the complete frozen evaluation population, including idle users")
    units={int(unit["uid"]):unit for unit in summary["units"]}
    if len(units)!=len(population) or set(units)!=set(population):
        raise ValueError("reference per-user service units do not cover the frozen population")
    return settings,summary,units


def matched_reference(request, references):
    old=references.get(request["request_id"])
    if old is None or any(old[key]!=request[key] for key in IDENTITY):
        raise ValueError("candidate identity differs from retained reference: request/uid/version/label/item/time")
    return old


def paired_quality(table, selected, *, repetitions, seed=17):
    """Reuse summarize and the same UID draws for quality and direct old/new AUC."""
    quality=summarize(table,selected,repetitions=repetitions,seed=seed,
                      releases=VERSIONS,preservation_threshold=.8)
    quality["estimand"]="Equal-release mean AUC recovery against native first-request Rebuild."
    comparison=table.set_column(table.schema.get_field_index("full_logit"),
                                "full_logit",table["old_relay_logit"])
    comparison=comparison.drop(["first_request_rebuild_logit"])
    compared=summarize(comparison,selected,repetitions=repetitions,seed=seed,
                       releases=VERSIONS,preservation_threshold=.8)
    paired=dict(overall=compared["overall"]["paired"]["partial_minus_full_auc"],
        by_release={v:compared["by_release"][v]["paired"]["partial_minus_full_auc"] for v in VERSIONS},
        old_equal_release_mean_auc=compared["overall"]["equal_release_mean_auc"]["full"],
        optimized_equal_release_mean_auc=compared["overall"]["equal_release_mean_auc"]["partial"],
        bootstrap=compared["bootstrap"],
        scope="Paired implementation change on already revealed users, not independent confirmation")
    return quality,paired


def numeric_summary(table):
    if not table.num_rows:
        return dict(compared_scores=0,max_abs=0.,relative_l2=0.,absolute_tolerance=ATOL,relative_tolerance=RTOL)
    new=table["partial_logit"].to_numpy().astype(np.float64)
    old=table["old_relay_logit"].to_numpy().astype(np.float64)
    delta=new-old
    if not np.isfinite(new).all() or not np.isfinite(old).all():
        raise ValueError("retained or optimized logits are nonfinite")
    if not np.allclose(new,old,atol=ATOL,rtol=RTOL):
        raise ValueError(f"optimized scores exceed the frozen FP32 tolerance; max_abs={np.max(np.abs(delta))}")
    return dict(compared_scores=len(new),max_abs=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(old),1e-30)),
        absolute_tolerance=ATOL,relative_tolerance=RTOL)


@torch.no_grad()
def run(args):
    began=time.perf_counter()
    if args.shards<1 or not 0<=args.shard_index<args.shards or args.bootstrap<1:
        raise ValueError("invalid UID shard or bootstrap repetitions")
    panel,groups,feedback=load_panel(args.panel)
    population=list(groups["evaluation"])
    if len(population)!=len(set(population)):
        raise ValueError("frozen evaluation users are duplicated")
    selected=population[args.shard_index::args.shards]
    if not selected:
        raise ValueError("selected shard is empty")
    panel_hash=sha256(args.panel/"binding.json")
    _,reference_summary,reference_units=reference_inputs(args.reference,panel_hash=panel_hash,
        population=population,kind=args.workload)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    freeze=json.loads((BASE/"freeze.json").read_text())
    reference_table=pq.read_table(args.reference,filters=[("uid","in",selected)])
    if pc.count_distinct(reference_table["request_id"]).as_py()!=reference_table.num_rows:
        raise ValueError("retained candidate request IDs are duplicated")
    references={row["request_id"]:row for row in reference_table.to_pylist()}
    del reference_table
    device=setup_device(args.gpu,threads=2)
    models=load_models(panel,device)
    adapters,weights,_=load_candidates(BASE/"calibration",device,VERSIONS,population)
    for version in VERSIONS:
        sealed=freeze["release_bindings"][version]
        if weights[version]["sha256"]!=sealed["calibration_sha256"] \
                or panel["models"][version]["checkpoint_sha256"]!=sealed["model_sha256"] \
                or reference_summary["weights"][version]["sha256"]!=sealed["calibration_sha256"]:
            raise ValueError("model/calibration/reference differs from the frozen candidate")
    config=ImplementationConfig(reader_backend=args.reader_backend,pack_item_gates=True,row_chunk=128)
    wrapped={v:InferenceAdapter(adapters[v],config) for v in VERSIONS}
    bank,bank_work=project_fixed_item_terms(adapters["v3"],models["v3"].item_emb.weight,chunk_size=16384)
    feature_fn=partial(fixed_features_from_bank,bank=bank)
    prefixes,publications={},{}
    previous=None
    for version in VERSIONS:
        old_path=BASE/"publication"/f"{version}.pt"
        old_metadata=json.loads(old_path.with_suffix(".json").read_text())
        old_sha=sha256(old_path)
        if old_sha!=old_metadata["artifact_sha256"] \
                or old_metadata["bindings"]["calibration_sha256"]!=weights[version]["sha256"] \
                or old_metadata["bindings"]["current_model_sha256"]!=panel["models"][version]["checkpoint_sha256"]:
            raise ValueError("retained publication/model/calibration binding mismatch")
        generated,work=compile_item_tables(adapters[version],models[version],previous,chunk_size=16384)
        if work["arithmetic"]!=old_metadata["actual_compilation_components_flops"]:
            raise ValueError("optimized publication changed the accounted arithmetic")
        prefixes[version]=configure_prefix(generated,config,feature_fn=feature_fn)
        publications[version]=dict(generated_in_memory=True,generated_artifact_sha256=None,
            generation="New full-catalogue compile_item_tables from the new preceding prefix; not retained-D reuse",
            previous_generated_version=None if previous is None else f"v{int(version[1:])-1}",
            reference_publication_path=str(old_path),reference_publication_sha256=old_sha,
            reference_relation="Retained original publication binds comparison provenance only; this SHA is not the new D table's SHA",
            arithmetic=work,shape_D=list(generated.D.shape),shape_P=list(generated.P.shape),
            current_calibration_sha256=weights[version]["sha256"],
            current_model_sha256=panel["models"][version]["checkpoint_sha256"])
        previous=generated
    implementation_sources=[*sorted(Path(__file__).parent.glob("*.py")),
        ROOT/"src/hstu_kvcache/methods/cachetide/adapter.py",ROOT/"src/hstu_kvcache/methods/cachetide/session.py",
        ROOT/"src/hstu_kvcache/evaluation/cachetide_workload.py",ROOT/"src/hstu_kvcache/methods/cachetide/precision.py"]
    settings=dict(panel=str(args.panel.resolve()),panel_sha256=panel_hash,population=population,
        selected_uids=selected,shards=args.shards,shard_index=args.shard_index,versions=VERSIONS,
        workload=workload(args.workload),reference=str(args.reference.resolve()),reference_sha256=sha256(args.reference),
        reference_summary_sha256=sha256(args.reference.parent/"summary.json"),
        frozen_candidate=str(BASE),freeze_sha256=sha256(BASE/"freeze.json"),weights=weights,
        generated_publications=publications,
        generated_fixed_phi=dict(recomputed_in_memory=True,artifact_sha256=None,work=bank_work,
            origin_current_model_sha256=panel["models"]["v3"]["checkpoint_sha256"],
            origin_encoder_checkpoint_sha256=weights["v3"]["sha256"],
            reference_fixed_feature_bank=freeze["fixed_feature_bank"],
            reference_relation="Same fixed affine feature function, new concatenated GEMM; not the old bank artifact"),
        implementation={v:wrapped[v].describe() for v in VERSIONS},query_batch=16,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in implementation_sources},
        gpu=args.gpu,device=torch.cuda.get_device_name(device),torch=str(torch.__version__),
        inference_only=True,corrected_native_writes=True,no_parameter_or_workload_selection=True,
        scope="Complete frozen population engineering recheck; previously revealed users, not new independent confirmation",
        cost_scope="Quality-panel operation counts; fixed30k complete service budget remains separate")
    write_json(args.output/"configuration.json",settings)
    histories=load_history(args.panel,requested_uids=selected)
    requests=({uid:candidate_rows(uid,histories.rows[uid],stride=16,candidates=16,
        known_vocab_size=panel["known_vocab_size"]) for uid in selected}
        if args.workload=="behavior" else feedback)
    rows,units=[],[]
    counters,native_ledger,code_ledger=Counter(),Counter(),Counter()
    for number,uid in enumerate(selected):
        first_row=len(rows)
        session=RelaySession(NativeTrajectory(models["v2"],2,histories.rows[uid],RELEASES[0][1]))
        by_release={}
        for index,(version,at) in enumerate(RELEASES):
            births={str(k):v for k,v in Counter(int(p) for p in session.trajectory.producers).items()}
            before=Counter(session.costs)
            session.publish(models[version],int(version[1:]),at,wrapped[version],prefix=prefixes[version])
            by_time=defaultdict(list)
            for request in requests[uid]:
                if request["version"]==version:
                    matched_reference(request,references)
                    by_time[int(request["query_timestamp"])].append(request)
            first_births=None
            for timestamp,pending in sorted(by_time.items()):
                session.prepare_read(timestamp)
                if first_births is None:
                    first_births={str(k):v for k,v in Counter(int(p) for p in session.trajectory.producers).items()}
                for offset in range(0,len(pending),16):
                    batch=pending[offset:offset+16]
                    items=torch.tensor([[request["item_idx"] for request in batch]],device=device)
                    actual=session.score(timestamp,items)[0].cpu().tolist()
                    for request,value in zip(batch,actual,strict=True):
                        old=matched_reference(request,references)
                        if old["history_length"]!=session.cache.seq_len or old["coded_rows"]!=session.coded_rows:
                            raise ValueError("optimized service changed the retained history/code lengths")
                        rows.append(dict(**{key:request[key] for key in ("request_id",*IDENTITY)},
                            reuse_logit=old["reuse_logit"],full_logit=old["full_logit"],
                            first_request_rebuild_logit=old["first_request_rebuild_logit"],
                            old_relay_logit=old["partial_logit"],partial_logit=value,
                            history_length=session.cache.seq_len,coded_rows=session.coded_rows))
            session.advance(RELEASES[index+1][1] if index+1<len(RELEASES) else END)
            by_release[version]=dict(source_births_at_publish=births,
                source_births_at_first_request=first_births,requests=sum(map(len,by_time.values())),
                cost_counters=dict(session.costs-before),final_coded_rows=session.coded_rows,
                final_history_rows=session.cache.seq_len)
        if by_release!=reference_units[uid]["by_release"]:
            raise ValueError(f"literal per-release service counts differ from retained reference for UID {uid}")
        units.append(dict(uid=uid,by_release=by_release,memory=dict(session.memory),
                          literal_reference_counters_equal=True))
        counters.update(session.costs);native_ledger.update(session.trajectory.ledger)
        code_ledger.update(session.coded_read_ledger)
        if number==0:
            canary_table=pa.Table.from_pylist(rows[first_row:],schema=SCORE_SCHEMA)
            canary=dict(status="passed",uid=uid,all_three_releases=True,
                numeric=numeric_summary(canary_table),identity_and_literal_counters_equal=True,
                retained_in_shard=True,no_repeat_needed=True,
                elapsed_seconds=time.perf_counter()-began)
            write_json(args.output/"canary.json",canary)
            print(json.dumps(dict(status="canary_complete",**{key:value for key,value in canary.items()
                              if key!="status"})),flush=True)
        if (number+1)%16==0 or number+1==len(selected):
            write_json(args.output/"progress.json",dict(status="running",completed_users=number+1,
                total_users=len(selected),candidate_scores=len(rows),elapsed_seconds=time.perf_counter()-began))
            print(json.dumps(dict(status="quality",users=number+1,total_users=len(selected),
                requests=len(rows),elapsed_seconds=time.perf_counter()-began)),flush=True)
    table=pa.Table.from_pylist(rows,schema=SCORE_SCHEMA)
    if set(table["request_id"].to_pylist())!=set(references):
        raise ValueError("new requests do not exactly cover retained reference requests for this UID shard")
    if pc.count_distinct(table["request_id"]).as_py()!=table.num_rows:
        raise ValueError("optimized candidate request IDs are duplicated")
    numeric=numeric_summary(table)
    pq.write_table(table,args.output/"scores.parquet",compression="zstd")
    quality,paired=paired_quality(table,selected,repetitions=args.bootstrap) if rows else (None,None)
    summary=dict(status="complete",selected_uids=selected,weights=weights,quality=quality,
        optimized_minus_original_auc=paired,numeric=numeric,units=units,
        actual_service_counters=dict(counters),
        native_ledger=records(native_ledger,("action","version","before","width")),
        code_read_ledger=records(code_ledger,("kind","version","old_length","queries","coded_rows","native_rows")),
        implementation_counters={v:dict(wrapped[v].counters) for v in VERSIONS},
        generated_publications=publications,source_sha256=settings["source_sha256"],
        elapsed_seconds=time.perf_counter()-began,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        checks=dict(frozen_user_roles=True,matched_request_identities=True,matched_literal_service_counters=True,
            corrected_writes=True,new_current_rows_zero_code=True,all_selected_idle_users_retained=True),
        bootstrap_scope="Shard point diagnostics only; final inference uses a joint full-population UID bootstrap",
        scope=settings["scope"])
    write_json(args.output/"summary.json",summary)
    write_json(args.output/"units.json",units)
    print(json.dumps(dict(status="complete",users=len(selected),candidate_scores=table.num_rows,numeric=numeric)),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path,default=PANEL)
    p.add_argument("--workload",choices=("behavior","feedback"),default="behavior")
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--shards",type=int,default=1)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--reader-backend",choices=("triton","torch"),default="triton")
    p.add_argument("--bootstrap",type=int,default=1)
    return p


if __name__=="__main__":
    args=parser().parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/"failure.json",dict(type=type(error).__name__,message=str(error),
                                                     traceback=traceback.format_exc()))
        raise

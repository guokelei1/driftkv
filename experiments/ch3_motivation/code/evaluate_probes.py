#!/usr/bin/env python3
"""Fresh E5 timeline scores, paired anchors, and complete correction ledgers."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, quality_role, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.probe_replay import evaluate_probe
from experiments.ch3_motivation.code.probe_fast import evaluate_fast_probe, GraphProbePool
from experiments.ch3_motivation.code.probe_projected_fast import (
    evaluate_projected_probe, ProjectedGraphPool, projected_read_extra_flops,
    projected_row_preparation_flops,
)
from experiments.ch3_motivation.code.probe_projected_cost import (
    full_projected_preparation, deployment_projected_graph_setup, projected_padded_read_delta,
)
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_cost import (
    cached_preparation_flops, padded_read_extra_flops, scalar_behavior_flops, deployment_graph_setup,
)
from experiments.ch3_motivation.code.scenario_inputs import load_scenario_population, load_scenario_data
from experiments.ch3_motivation.code.probes import native_query_flops
from hstu_kvcache.evaluation.baselines import event_shapes
from hstu_kvcache.evaluation.fast_baselines import evaluate_checkpointed
from hstu_kvcache.evaluation.statistics import paired_user_bootstrap
from experiments.ch3_motivation.code.selective import repair_cost
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def cached_read_extra_flops(probe, history_length, inherited_rows, queries, *, kind="query"):
    """Replay's g/h arithmetic with immutable row features prepared once.

    This is deliberately separate from fitting's forward_flops: fitting may
    regenerate inputs on each sample, while ProbeTrajectory retains them. The
    last scalar corrected append can evict the final inherited row; its g MLP
    still executes before the empty inherited loop, although query scoring
    with no inherited rows takes the native bypass.
    """
    c, n, q = probe.cfg, int(inherited_rows), int(queries)
    if probe.stage == "f" or (n == 0 and kind == "query"):
        return 0
    lh, d = c.num_layers*c.num_heads, c.head_dim
    chunks = (n+c.row_chunk-1)//c.row_chunk

    def mlp(width):
        widths = [width]+[c.hidden_width]*c.depth+[2*d]
        return sum(2*i*o+o for i,o in zip(widths[:-1],widths[1:],strict=True))+2*c.depth*c.hidden_width

    # _output_scale is multiplied even when normalization is disabled.
    g = q*lh*((d if c.normalize else 0)+mlp(d)+2*d)
    # Per inherited block: query/Key shift; broadcast score add; weight sum;
    # Value multiplication; both additions in result + AV + shifted Value.
    # Dense reductions follow the existing N-operation convention.
    g += q*lh*(2*n+5*d*chunks)
    h = 0
    if probe.stage == "fgh" and n:
        # q normalization is repeated per row block. Row normalization is not:
        # prepare_read_features already generated and retained those inputs.
        h += q*lh*chunks*(d if c.normalize else 0)
        h += q*n*lh*(mlp(d+c.row_dim)+2*d+4*d+1)
        h += q*lh*d*chunks
    # Inherited and newly appended suffix reads are combined explicitly.
    suffix_add = q*lh*d if history_length > n else 0
    return int(g+h+suffix_add)


def correction_cost(probe, execution, calibration, cm, shapes, cfg, users, *, model):
    preparation=sum(row["calls"]*cached_preparation_flops(probe,row["history_length"])
        for row in execution["prepare_shapes"])
    reads=sum(row["calls"]*cached_read_extra_flops(probe,row["history_length"],
        row["inherited_rows"],row["queries"],kind=row["kind"])
        for row in execution["read_shapes"])
    padding_extra=0
    for row in execution.get("padded_read_shapes",[]):
        ordinary=cached_read_extra_flops(probe,row["physical_read_rows"],
            row["inherited_rows"],row["queries"],kind=row["kind"])
        actual=padded_read_extra_flops(probe,row["physical_read_rows"],row["queries"])
        padding_extra+=row["calls"]*(actual-ordinary)
    reads+=padding_extra
    cap=cfg["history_length"]
    # Behavior read records identify scalar corrected writes, including the
    # final one that evicts the last inherited row. inherited_rows==0 alone
    # cannot distinguish that write from the later native-band branch.
    corrected=Counter()
    dense_writer=0
    for row in execution["read_shapes"]:
        if row["kind"]=="behavior":
            corrected[row["history_length"],row["inherited_rows"]]+=row["calls"]
            dense_writer+=row["calls"]*scalar_behavior_flops(model,row["history_length"])
    native_band=0
    for row in execution["append_shapes"]:
        n,w=row["before_length"],row["width"]
        key=min(n,cap-1),row["inherited_rows"]
        dense_calls=min(row["calls"],corrected[key]) if w==1 else 0
        corrected[key]-=dense_calls
        cost=cm if n==cap else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")
        native_band+=(row["calls"]-dense_calls)*cost.band_append(n,w,window_size=cap)
    if any(corrected.values()):
        raise ValueError("corrected behavior reads and scalar append records do not match")
    native_actual=dense_writer+native_band
    native_baseline=sum(count*(cm if full else CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")).band_append(
        n,w,window_size=cap) for (n,w,full),count in shapes["appends"].items())
    extra_writer=native_actual-native_baseline
    denominator=repair_cost(dict(method="rebuild"),shapes,cm,cfg)[0]
    uses_padded=bool(execution.get("padded_execution",False))
    setup=(deployment_graph_setup(probe,execution,model,window=cap) if uses_padded else
        dict(shared_graph_setup_flops=0,required_graph_keys=[],scope="reference path has no graph startup"))
    startup=setup["shared_graph_setup_flops"]
    added=calibration+preparation+reads+extra_writer+startup
    normalized_reads=0
    for row in execution["initial_shapes"]:
        # Analytic release-source-length reference, 230 candidates/user.
        work=(padded_read_extra_flops(probe,row["history_length"],230)
            if uses_padded and row["history_length"]==cap and probe.stage!="f" else
            cached_read_extra_flops(probe,row["history_length"],row["history_length"],230))
        normalized_reads+=row["calls"]*work
    behavior_extra=sum(row["calls"]*cached_read_extra_flops(probe,row["history_length"],
        row["inherited_rows"],row["queries"],kind="behavior")
        for row in execution["read_shapes"] if row["kind"]=="behavior")
    for row in execution.get("padded_read_shapes",[]):
        if row["kind"]=="behavior":
            behavior_extra+=row["calls"]*(padded_read_extra_flops(probe,row["physical_read_rows"],row["queries"])
                -cached_read_extra_flops(probe,row["physical_read_rows"],row["inherited_rows"],row["queries"],kind="behavior"))
    # Every hypothetical 230-read user has a first consumer, including those
    # that are completely idle in the real event ledger.
    normalized_preparation=sum(row["calls"]*cached_preparation_flops(probe,row["history_length"])
        for row in execution["initial_shapes"])
    normalized_denominator=sum(row["calls"]*(cm if row["history_length"]==cap else
        CostModel(cm.hidden_size,cm.num_layers,cm.num_heads,"torch")).full_cache(row["history_length"])
        for row in execution["initial_shapes"])
    normalized_setup=setup
    if uses_padded:
        normalized_shapes=[row for row in execution["read_shapes"] if row["kind"]=="behavior"]
        batch=int(cfg.get("query_batch",16))
        for row in execution["initial_shapes"]:
            for offset in range(0,230,batch):
                normalized_shapes.append(dict(kind="query",history_length=row["history_length"],
                    inherited_rows=row["history_length"],queries=min(batch,230-offset),calls=row["calls"]))
        normalized_setup=deployment_graph_setup(probe,dict(read_shapes=normalized_shapes,
            append_shapes=execution["append_shapes"]),model,window=cap)
    normalized_startup=normalized_setup["shared_graph_setup_flops"]
    normalized_added=calibration+normalized_preparation+normalized_reads+behavior_extra+extra_writer+normalized_startup
    return dict(calibration_flops=int(calibration),release_preparation_flops=int(preparation),
        extra_read_flops=int(reads),padding_and_prefix_mask_extra_flops=int(padding_extra),
        extra_native_writer_flops=int(extra_writer),
        shared_graph_setup_flops=int(startup),graph_setup=setup,
        corrected_scalar_native_writer_flops=int(dense_writer),native_band_writer_flops=int(native_band),
        actual_native_writer_flops=int(native_actual),baseline_native_band_writer_flops=int(native_baseline),
        complete_added_flops=int(added),once_rebuild_flops=int(denominator),
        percent_of_once_rebuild=100*added/denominator,
        candidate_reads_230=dict(users=users,candidate_reads_per_user=230,
            release_preparation_flops=int(normalized_preparation),extra_candidate_read_flops=int(normalized_reads),
            actual_behavior_read_extra_flops=int(behavior_extra),extra_native_writer_flops=int(extra_writer),
            calibration_flops=int(calibration),complete_added_flops=int(normalized_added),
            shared_graph_setup_flops=int(normalized_startup),graph_setup=normalized_setup,
            once_rebuild_flops=int(normalized_denominator),percent_of_once_rebuild=100*normalized_added/normalized_denominator,
            scope="230 reads use release-source row lengths; actual behavior evolution retained separately; all users activate"),
        read_feature_scope="row features and normalization prepared once with f; no per-read row normalization",
        writer_scope="actual dense scalar corrected writes (behavior embedding, final norm, no score head) plus native-band branch; "
            "signed difference from the shared native-band baseline is reported separately",
        arithmetic_scope="forward FLOPs with multiply-add=2, reductions at declared dense lengths; fitting backward remains a separate 2x-forward estimate")


def apply_projected_costs(costs,probe,execution,histories,by_user,users,cfg,model):
    """Replace raw graph arithmetic; charge the new row table and all setup."""
    preparation=full_projected_preparation(probe,histories,by_user,users,cfg)
    reads=projected_padded_read_delta(probe,execution,padded_read_extra_flops)
    setup=deployment_projected_graph_setup(probe,execution,model,window=cfg["history_length"])
    old_total=costs["complete_added_flops"]
    old_normalized=costs["candidate_reads_230"]["complete_added_flops"]
    result=dict(costs,release_preparation_flops=costs["release_preparation_flops"]+preparation["extra_row_preparation_flops"],
        extra_read_flops=costs["extra_read_flops"]+reads["projected_minus_raw_padded_read_flops"],
        shared_graph_setup_flops=setup["shared_graph_setup_flops"],graph_setup=setup,
        extra_row_projection_preparation_flops=preparation["extra_row_preparation_flops"],
        row_projection=preparation,projected_physical_reads=reads,
        raw_graph_padding_and_prefix_mask_extra_flops=costs["padding_and_prefix_mask_extra_flops"],
        padding_and_prefix_mask_extra_flops=costs["padding_and_prefix_mask_extra_flops"]+reads["projected_minus_raw_padded_read_flops"])
    result["complete_added_flops"]=sum(result[field] for field in ("calibration_flops","release_preparation_flops",
        "extra_read_flops","extra_native_writer_flops","shared_graph_setup_flops"))
    result["percent_of_once_rebuild"]=100*result["complete_added_flops"]/result["once_rebuild_flops"]
    normalized=dict(costs["candidate_reads_230"])
    cap=cfg["history_length"]
    normalized["extra_candidate_read_flops"]=sum(row["calls"]*(projected_read_extra_flops(probe,cap,230)
        if row["history_length"]==cap else cached_read_extra_flops(probe,row["history_length"],row["history_length"],230))
        for row in execution["initial_shapes"])
    normalized["actual_behavior_read_extra_flops"]+=sum(row["calls"]*(
        projected_read_extra_flops(probe,row["physical_read_rows"],row["queries"])
        -padded_read_extra_flops(probe,row["physical_read_rows"],row["queries"]))
        for row in execution["padded_read_shapes"] if row["kind"]=="behavior")
    # The normalized reads occur at release-source lengths, so a later actual
    # query may no longer be what triggers a short user's first full graph.
    # Retain projection work required by behavior alone, then activate every
    # initially full user for the hypothetical 230 release-source reads.
    behavior_preparation=full_projected_preparation(probe,histories,{},users,cfg)
    histogram=Counter({row["inherited_rows"]:row["calls"] for row in behavior_preparation["preparation_shapes"]})
    if probe.h is not None:
        histogram[cap]=sum(row["calls"] for row in execution["initial_shapes"] if row["history_length"]==cap)
    extra=sum(count*projected_row_preparation_flops(probe,n) for n,count in histogram.items())
    normalized["release_preparation_flops"]+=extra
    normalized["extra_row_projection_preparation_flops"]=extra
    normalized_shapes=[row for row in execution["read_shapes"] if row["kind"]=="behavior"]
    batch=int(cfg.get("query_batch",16))
    for row in execution["initial_shapes"]:
        normalized_shapes.extend(dict(kind="query",history_length=row["history_length"],
            inherited_rows=row["history_length"],queries=min(batch,230-offset),calls=row["calls"])
            for offset in range(0,230,batch))
    norm_setup=deployment_projected_graph_setup(probe,dict(read_shapes=normalized_shapes,
        append_shapes=execution["append_shapes"]),model,window=cap)
    normalized.update(shared_graph_setup_flops=norm_setup["shared_graph_setup_flops"],graph_setup=norm_setup)
    normalized["complete_added_flops"]=sum(normalized[field] for field in ("calibration_flops","release_preparation_flops",
        "extra_candidate_read_flops","actual_behavior_read_extra_flops","extra_native_writer_flops","shared_graph_setup_flops"))
    normalized["percent_of_once_rebuild"]=100*normalized["complete_added_flops"]/normalized["once_rebuild_flops"]
    result["candidate_reads_230"]=normalized
    result["comparison_to_raw_graph"]=dict(complete_added_flops=old_total,normalized_230_complete_added_flops=old_normalized,
        scope="analytic execution comparison with identical weights and nonlinear h; exact short-window paths unchanged")
    result["read_feature_scope"]="raw normalized features retained; h first-affine row projection prepared once at first full graph; all nonlinear tail operations retained on every physical pair"
    return result


def merge_execution(parts):
    counts=Counter()
    shape_names={"initial_shapes":("history_length",),"prepare_shapes":("history_length",),
        "read_shapes":("kind","history_length","inherited_rows","queries"),
        "append_shapes":("before_length","width","inherited_rows"),
        "padded_read_shapes":("kind","physical_read_rows","inherited_rows","queries")}
    result={}
    for part in parts:
        counts.update(part["counts"])
    result["counts"]=dict(counts)
    for name,keys in shape_names.items():
        combined=Counter()
        for part in parts:
            for row in part.get(name,[]):
                combined[tuple(row[k] for k in keys)]+=row["calls"]
        result[name]=[dict(zip(keys,key),calls=count) for key,count in combined.items()]
    # A graph pool report is a cumulative snapshot, not a per-shard counter.
    # Preserve every saved snapshot on resume without summing setup/replays.
    result["block_graph_pool_snapshots"]=[dict(block=i,
        generation=part.get("graph_pool_generation"),graphs=part["graph_pool"])
        for i,part in enumerate(parts) if "graph_pool" in part]
    result["block_quality_shortcuts"]=[dict(block=i,**part["quality_shortcut"])
        for i,part in enumerate(parts) if "quality_shortcut" in part]
    result["block_projected_execution"]=[dict(block=i,**part["projected_execution"])
        for i,part in enumerate(parts) if "projected_execution" in part]
    result["graph_snapshot_scope"]="Cumulative within each generation; do not sum snapshots. Saved blocks retain original setup metadata on resume."
    return result


def run(args, *, shared_context=None):
    started=time.perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    population_file=getattr(args,"population_file",None)
    scenario_provenance=None
    if population_file is not None:
        binding,users,population,scenario_provenance=load_scenario_population(population_file,args.edge,
            role="quality",minimum_users=512,require_positive=False)
        histories,by_user=load_scenario_data(scenario_provenance)
        input_binding_sha256=scenario_provenance["input_binding_sha256"]
    else:
        population=json.loads((args.input_root/"population.json").read_text())
        users=(population["population_sensitivity"][str(args.population_seed)] if args.panel=="sensitivity" else population[args.panel])
        binding,histories,by_user=load_edge(args.input_root/f"edge_{args.edge}")
        for name in ("history","requests"):
            record=binding[name]
            if sha256(args.input_root/f"edge_{args.edge}"/record["path"])!=record["sha256"]:
                raise ValueError(f"frozen {name} pack changed")
        if binding["population_sha256"]!=sha256(args.input_root/"population.json"):
            raise ValueError("frozen population changed")
        input_binding_sha256=sha256(args.input_root/f"edge_{args.edge}/inputs.json")
    quality_role(users)
    cfg=binding["settings"]
    model_key=(args.edge,str(device),args.gpu)
    if shared_context is not None and shared_context.get("model_key")==model_key:
        parent,current,models=shared_context["models"]
    else:
        if shared_context is not None:shared_context.clear()
        parent,current,models=load_models(args.edge,device)
        if shared_context is not None:
            shared_context.update(model_key=model_key,models=(parent,current,models),anchors={})
    probe,fit=load_variant(args.checkpoint,device)
    if fit.get("quality_use_permitted") is False:
        raise ValueError("a bounded diagnostic checkpoint cannot produce evaluation quality")
    if fit["identity"]["edge"]!=args.edge or fit["identity"]["models"]!=models:
        raise ValueError("probe was fitted for a different edge or backbone")
    if set(users)&set(fit["identity"]["fit_uids"]+fit["identity"]["validation_users"]):
        raise ValueError("probe fitting/validation users overlap evaluation")
    if not set(fit["identity"]["fit_uids"])<=set(population["fit"]) or fit["identity"]["validation_users"]!=population["validation"]:
        raise ValueError("checkpoint does not preserve the frozen fitting/validation roles")
    if scenario_provenance is not None:
        if (scenario_provenance["anchor_models"]!=models or
            scenario_provenance["calibration_input_binding_sha256"]!=fit["identity"]["input_binding_sha256"]):
            raise ValueError("scenario anchors or calibration inputs differ from this checkpoint")
    elif fit["identity"]["input_binding_sha256"]!=input_binding_sha256:
        raise ValueError("checkpoint calibration inputs differ from the frozen evaluation pack")
    probe.eval()
    args.output.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),
        Path(__file__).with_name("probes.py"),Path(__file__).with_name("probe_replay.py"),
        Path(__file__).with_name("common.py"),ROOT/"src/hstu_kvcache/adaptation/reader.py")}
    for name in ("run_probes.py","run_probe_variants.py","probe_variants.py","selective.py","probe_fast.py","probe_ledger.py","probe_cost.py","scenario_inputs.py","probe_projected_fast.py","probe_projected_cost.py"):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(ROOT))]=sha256(path)
    for name in ("src/hstu_kvcache/evaluation/fast_baselines.py","src/hstu_kvcache/evaluation/baselines.py",
        "src/hstu_kvcache/evaluation/statistics.py","src/hstu_kvcache/evaluation/cost.py",
        "src/hstu_kvcache/methods/baselines/protocol/cost.py"):
        sources[name]=sha256(ROOT/name)
    identity=dict(edge=args.edge,panel=args.panel,users=users,models=models,backend=args.backend,
        checkpoint_sha256=sha256(args.checkpoint),checkpoint=str(args.checkpoint),
        sources=sources,input_binding_sha256=input_binding_sha256,scenario=scenario_provenance)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    config=args.output/"configuration.json"
    if config.exists() and json.loads(config.read_text())!=identity:
        raise ValueError("existing score attempt configuration changed")
    write_json(config,identity)
    anchor_folder=args.output/"anchors"
    anchor_folder.mkdir(exist_ok=True)
    policies=[dict(tag=name,method=name,calibration_flops=0) for name in ("reuse","rebuild")]
    anchor_key=hashlib.sha256(json.dumps(dict(edge=args.edge,models=models,users=users,
        input_binding_sha256=identity["input_binding_sha256"],sources=sources),sort_keys=True).encode()).hexdigest()
    cached_anchor=shared_context["anchors"].get(anchor_key) if shared_context is not None else None
    if cached_anchor is None:
        anchor_scores,anchor_execution=evaluate_checkpointed(parent,current,histories,by_user,users,
            policies,cfg,device,anchor_folder,identity=identity,
            progress=lambda value:print(json.dumps(dict(stage="anchors",**value)),flush=True))
        anchor_source=dict(key=anchor_key,path=str(anchor_folder.resolve()),recomputed_in_this_goal=True,
            scope="fresh matched anchors, shared only across this sequential process's identical edge/population/input/source")
        if shared_context is not None:
            shared_context["anchors"][anchor_key]=(anchor_scores,anchor_execution,anchor_source)
    else:
        anchor_scores,anchor_execution,anchor_source=cached_anchor
        write_json(anchor_folder/"shared_anchor_reference.json",anchor_source)
    blocks=args.output/"probe_blocks"
    blocks.mkdir(exist_ok=True)
    method_label=fit.get("method_label",probe.stage)
    scores=[]
    executions=[]
    graph_pool=(ProjectedGraphPool(current,probe,window=cfg["history_length"]) if args.backend=="projected" else
        GraphProbePool(current,probe,window=cfg["history_length"]) if args.backend=="graph" else None)
    graph_pool_generation=uuid.uuid4().hex if graph_pool is not None else None
    for offset in range(0,len(users),256):
        selected=users[offset:offset+256]
        path=blocks/f"part_{offset:06d}.parquet"
        marker=blocks/f"part_{offset:06d}.json"
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved["fingerprint"]!=fingerprint or saved["users"]!=selected or saved["sha256"]!=sha256(path):
                raise ValueError("saved probe score block changed")
            rows=pq.read_table(path).to_pylist()
            execution=saved["execution"]
        else:
            if graph_pool is not None:
                evaluate=evaluate_projected_probe if args.backend=="projected" else evaluate_fast_probe
                rows,execution=evaluate(parent,current,probe,histories,by_user,selected,cfg,
                    device,tag=method_label,pool=graph_pool,stop_after_last_query=True)
                execution["graph_pool_generation"]=graph_pool_generation
            else:
                rows,execution=evaluate_probe(parent,current,probe,histories,by_user,selected,cfg,device,tag=method_label)
            temporary=path.with_suffix(".parquet.writing")
            schema=pa.schema([("uid",pa.uint64()),("query_timestamp",pa.int64()),("raw_item_id",pa.uint64()),
                ("item_idx",pa.int64()),("label",pa.int64()),("method",pa.string()),("score",pa.float64())])
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),temporary)
            temporary.replace(path)
            write_json(marker,dict(fingerprint=fingerprint,users=selected,sha256=sha256(path),execution=execution))
        scores.extend(rows)
        executions.append(execution)
        progress=dict(status="probe_quality",edge=args.edge,stage=probe.stage,users=offset+len(selected),
            total=len(users),scores=len(scores),seconds=time.perf_counter()-started)
        write_json(args.output/"progress.json",progress)
        print(json.dumps(progress),flush=True)
    quality_execution=merge_execution(executions)
    execution=full_probe_ledger(histories,by_user,users,cfg,stage=probe.stage,padded=graph_pool is not None)
    execution["padded_execution"]=graph_pool is not None
    execution["shared_quality_graph_setup_snapshots"]=quality_execution["block_graph_pool_snapshots"]
    groups=defaultdict(list)
    for row in anchor_scores+scores:
        groups[row["method"]].append(row)
    for values in groups.values():
        values.sort(key=lambda row:(row["uid"],row["query_timestamp"],row["raw_item_id"]))
    confidence,ci=paired_user_bootstrap(groups,users,replicates=1000,seed=17021+args.edge)
    from sklearn.metrics import roc_auc_score
    aucs={tag:float(roc_auc_score([r["label"] for r in values],[r["score"] for r in values])) for tag,values in groups.items()}
    gap=aucs["rebuild"]-aucs["reuse"]
    cm=CostModel(current.cfg.hidden_size,len(current.blocks),current.cfg.num_heads,cfg["attention_backend"])
    shapes=event_shapes(histories,by_user,users,cfg)
    costs=correction_cost(probe,execution,fit["cumulative_calibration_flops_estimate"],cm,shapes,cfg,len(users),model=current)
    if args.backend=="projected":
        costs=apply_projected_costs(costs,probe,execution,histories,by_user,users,cfg,current)
    result=dict(status="complete",experiment="E5",edge=args.edge,stage=probe.stage,
        variant=fit.get("variant","head"),width=probe.cfg.hidden_width,depth=probe.cfg.depth,panel=args.panel,
        physical_gpu=args.gpu,hardware=torch.cuda.get_device_name(0),
        seed=fit["seed"],fit_users=fit["training_users"],quality_users=len(users),method_label=method_label,
        feedback_events=len(scores),users_with_feedback=len({r["uid"] for r in scores}),
        auc=aucs[method_label],reuse_auc=aucs["reuse"],rebuild_auc=aucs["rebuild"],
        recovery_percent=100*(aucs[method_label]-aucs["reuse"])/gap if gap else None,
        confidence=ci,intervals=confidence,costs=costs,execution=execution,quality_execution=quality_execution,
        anchor_source=anchor_source,
        identity=identity,evaluation_window=cfg.get("evaluation_window",{"name":"E14"}),
        seconds=time.perf_counter()-started)
    for path,digest in sources.items():
        if sha256(ROOT/path)!=digest:
            raise ValueError(f"numerical source changed during evaluation: {path}")
    write_json(args.output/"summary.json",result)
    with (args.output/"points.csv").open("w",newline="") as stream:
        fields=["edge","variant","method_label","stage","width","depth","panel","seed","fit_users","quality_users","auc","reuse_auc","rebuild_auc","recovery_percent",
            "calibration_flops","release_preparation_flops","extra_read_flops","extra_native_writer_flops","shared_graph_setup_flops","complete_added_flops","percent_of_once_rebuild",
            "normalized_230_complete_added_flops","normalized_230_percent_of_once_rebuild","normalized_230_candidate_read_flops","normalized_230_shared_graph_setup_flops"]
        row={**result,**costs}
        normalized=costs["candidate_reads_230"]
        row.update(normalized_230_complete_added_flops=normalized["complete_added_flops"],
            normalized_230_percent_of_once_rebuild=normalized["percent_of_once_rebuild"],
            normalized_230_candidate_read_flops=normalized["extra_candidate_read_flops"],
            normalized_230_shared_graph_setup_flops=normalized["shared_graph_setup_flops"])
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerow(row)
    print(json.dumps(dict(status="complete",output=str(args.output),auc=result["auc"],recovery_percent=result["recovery_percent"])),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--edge",type=int,choices=range(1,6),required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--panel",choices=["development","main","confirmation","sensitivity"],default="development")
    parser.add_argument("--population-seed",type=int,default=20261023)
    parser.add_argument("--population-file",type=Path,help="fresh three-anchor qualified scenario manifest")
    parser.add_argument("--gpu",type=int,default=1)
    parser.add_argument("--backend",choices=["graph","projected","reference"],default="graph")
    run(parser.parse_args())

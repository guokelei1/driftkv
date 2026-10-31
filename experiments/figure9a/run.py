#!/usr/bin/env python3
"""Prepare, benchmark, then explicitly launch the four-GPU Figure 9(a) study."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from experiments.figure9a.common import (
    HERE, ROOT, configuration, digest, edge_config, ledger, load_inputs, policies, Rolling, write_json,
)

def source_hashes():
    # Include the promoted implementations and their transitive shared helpers.
    sources = list((ROOT / "src/hstu_kvcache").rglob("*.py"))
    sources.extend(p for p in HERE.glob("*.py") if not p.name.startswith("test_"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}

def identity(cfg,edge,inputs):
    record=dict(config=cfg,edge=edge,input_fingerprint=inputs["fingerprint"],
        sources=source_hashes(),
        checkpoints={f"v{v}":digest(ROOT/cfg["checkpoints"]/f"v{v}/checkpoint_100.pt") for v in (edge-1,edge)})
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),record

def wait_children(children):
    """Fail together, including siblings waiting for the rank-0 fitting file."""
    while True:
        codes=[p.poll() for p in children]
        if any(c is not None and c!=0 for c in codes):
            for p in children:
                if p.poll() is None:
                    p.terminate()
            for p in children:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait()
            return [p.returncode for p in children]
        if all(c is not None for c in codes):
            return codes
        time.sleep(.5)

def summarize_arrays(labels,uids,scores,rows,population,replicates,seed):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    from hstu_kvcache.evaluation.statistics import weighted_auc
    assert scores.shape==(len(labels),len(rows)) and np.isfinite(scores).all()
    metrics=[float(roc_auc_score(labels,scores[:,i])) for i in range(len(rows))]
    results=[dict(**p,auc=metrics[i],auc_gain_over_reuse=metrics[i]-metrics[0],
        recovery_percent=100*(metrics[i]-metrics[0])/(metrics[1]-metrics[0])) for i,p in enumerate(rows)]
    confidence={}
    if replicates:
        pop=np.asarray(sorted(population),dtype=np.int64)
        index=np.searchsorted(pop,uids)
        assert np.array_equal(pop[index],uids)
        rng=np.random.default_rng(seed)
        weights=np.empty((replicates,len(pop)),dtype=np.uint16)
        for i in range(replicates):
            weights[i]=np.bincount(rng.integers(len(pop),size=len(pop)),minlength=len(pop))
        draws=[weighted_auc(scores[:,i],labels,index,weights) for i in range(len(rows))]
        gap=draws[1]-draws[0]
        valid=bool(np.all(np.isfinite(gap)&(gap>0)))
        ci=lambda a:np.percentile(a[np.isfinite(a)],[2.5,97.5]).tolist()
        for i,r in enumerate(results):
            r.update(auc_ci95=ci(draws[i]),auc_gain_over_reuse_ci95=ci(draws[i]-draws[0]),
                recovery_percent_ci95=ci(100*(draws[i]-draws[0])/gap) if valid else None)
        confidence=dict(replicates=replicates,seed=seed,reference_gap_ci95=ci(gap),
            scope="Paired nonparametric UID bootstrap conditional on fixed seed17 fits; all frozen UIDs, including those without feedback; not training-seed variance.")
    return results,confidence

def worker(args):
    import numpy as np
    import torch
    from experiments.figure9a.engine import Graphs, evaluate_wave, fit_maps, load_models, waves
    cfg=configuration()
    torch.set_num_threads(cfg["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(cfg["seed"])
    device=torch.device("cuda:0")
    rows=policies(cfg)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/"worker.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    edge=args.edge
    cfg=edge_config(cfg,edge)
    started=time.perf_counter()
    print(json.dumps(dict(stage="loading",edge=edge,gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))),flush=True)
    inputs=load_inputs(cfg,edge,args.rank,args.world,args.users if args.mode=="benchmark" else None,
        include_fit=args.rank==0 or args.mode=="benchmark")
    ident,binding=identity(cfg,edge,inputs)
    if args.mode=="formal":
        frozen=json.loads((output.parent.parent/"configuration.json").read_text())
        if frozen["sources"]!=binding["sources"] or frozen["config_sha256"]!=digest(HERE/"config.json"):
            raise RuntimeError("execution changed after the formal controller froze it")
        if any(frozen["checkpoints"][v]!=sha for v,sha in binding["checkpoints"].items()):
            raise RuntimeError("model changed after formal run configuration")
        if frozen["roles_sha256"]!=digest(ROOT/cfg["role_source"]):
            raise RuntimeError("frozen user roles changed")
        if (output/"binding.json").exists():
            if json.loads((output/"binding.json").read_text())["identity"]!=ident:
                raise RuntimeError("existing attempt identity differs; keep it and use a new attempt")
    if args.mode=="formal" and (output/"worker_summary.json").exists() and (output/"checkpoint.npz").exists():
        prior=json.loads((output/"binding.json").read_text())
        if prior["identity"]!=ident:
            raise RuntimeError("completed shard inputs changed")
        return
    write_json(output/"binding.json",dict(identity=ident,**binding,role_source=cfg["role_source"],rank=args.rank,world=args.world,
        input_load_seconds=inputs["load_seconds"],quality_users=len(inputs["quality"]),requests=inputs["requests"],
        machine=socket.gethostname(),physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))
    parent,current=load_models(cfg,edge,device)
    print(json.dumps(dict(stage="models_loaded",seconds=time.perf_counter()-started)),flush=True)
    fit_path=(output/"maps.pt" if args.mode=="benchmark" else output.parent/"rank0"/"maps.pt")
    map_binding=dict(checkpoints=binding["checkpoints"],dataset=digest(ROOT/cfg["dataset"]),
        roles=digest(ROOT/cfg["role_source"]),engine=digest(HERE/"engine.py"),common=digest(HERE/"common.py"),
        map_source=digest(HERE/"kv_map.py"),flop_source=digest(HERE/"kv_cost.py"))
    if fit_path.exists():
        saved=torch.load(fit_path,map_location=device,weights_only=True)
        if saved["config_sha256"]!=digest(HERE/"config.json"):
            raise RuntimeError("saved map configuration changed")
        if args.mode=="formal" and saved.get("binding")!=map_binding:
            raise RuntimeError("saved map model/data/code binding changed")
        maps,fit_costs,fit_seconds=saved["maps"],saved["fit_costs"],saved["fit_seconds"]
        fit_profiles=saved["fit_profiles"]
    elif args.rank==0 or args.mode=="benchmark":
        maps,fit_costs,fit_seconds,fit_profiles=fit_maps(parent,current,inputs["histories"],inputs["fit"],cfg,rows,device)
        temporary=fit_path.with_suffix(".tmp")
        torch.save(dict(config_sha256=digest(HERE/"config.json"),binding=map_binding,maps={k:tuple(t.cpu() for t in v) for k,v in maps.items()},
            fit_costs=fit_costs,fit_seconds=fit_seconds,fit_profiles=fit_profiles),temporary)
        os.replace(temporary,fit_path)
    else:
        while not fit_path.exists():
            time.sleep(1)
        saved=torch.load(fit_path,map_location=device,weights_only=True)
        if saved.get("binding")!=map_binding or saved["config_sha256"]!=digest(HERE/"config.json"):
            raise RuntimeError("rank0 map binding differs")
        maps,fit_costs,fit_seconds=saved["maps"],saved["fit_costs"],saved["fit_seconds"]
        fit_profiles=saved["fit_profiles"]
    print(json.dumps(dict(stage="maps_ready",fit_seconds=fit_seconds)),flush=True)
    import gc
    gc.collect();torch.cuda.empty_cache()
    if args.mode=="benchmark":
        # Check full event trajectories, including native appends and FIFO,
        # against independent scalar users and unmodified dense sparse replay.
        sample=[u for u in inputs["quality"] if inputs["by_user"].get(u)][:4]
        reference={}
        for uid in sample:
            out,_=evaluate_wave(parent,current,inputs,[uid],cfg,rows,maps,device,Graphs(False),reference=True)
            for u,y,s in out:
                reference[u]=(y,s)
        errors=np.zeros(len(rows))
        for validation_batch in (1,4):
            fast={}
            validation_graphs=Graphs(True)
            for group in waves(inputs,sample,validation_batch,cfg["release_day"]*86400):
                out,_=evaluate_wave(parent,current,inputs,group,cfg,rows,maps,device,validation_graphs)
                for u,y,s in out:
                    fast[u]=(y,s)
            for uid in sample:
                y,s=reference[uid]
                assert np.array_equal(y,fast[uid][0])
                errors=np.maximum(errors,np.max(np.abs(s-fast[uid][1]),axis=0))
            del validation_graphs,fast
        assert errors.max()<2e-4,dict(zip([p["tag"] for p in rows],errors.tolist()))
        # Resume real user trajectories after discarding all GPU trajectory
        # state. Only scalar evidence and the completed-wave cursor survive.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=output) as tmp:
            resumed=Rolling(tmp,ident,0)
            split=max(1,len(sample)//2)
            first_graphs=Graphs(True)
            for i,uid in enumerate(sample[:split]):
                data,_=evaluate_wave(parent,current,inputs,[uid],cfg,rows,maps,device,first_graphs)
                for u,y,s in data:
                    resumed.append(u,y,s)
                resumed.save(i+1,len(rows),True)
            del first_graphs,resumed
            resumed=Rolling(tmp,ident,0)
            assert resumed.cursor==split
            second_graphs=Graphs(True)
            for i in range(resumed.cursor,len(sample)):
                data,_=evaluate_wave(parent,current,inputs,[sample[i]],cfg,rows,maps,device,second_graphs)
                for u,y,s in data:
                    resumed.append(u,y,s)
                resumed.save(i+1,len(rows),True)
            ry,ru,rs=resumed.arrays(len(rows))
            for uid in sample:
                mask=ru==uid
                assert np.array_equal(ry[mask],reference[uid][0])
                assert np.max(np.abs(rs[mask]-reference[uid][1]))<2e-4
            assert len(list(Path(tmp).glob("*.npz")))==2
            del second_graphs,resumed
        write_json(output/"correctness.json",dict(status="passed",users=len(sample),max_score_error=float(errors.max()),
            errors_by_method=dict(zip([p["tag"] for p in rows],errors.tolist())),
            real_numeric_checkpoint_resume="passed; interrupted at user-wave boundary, reloaded scalars/cursor, rebuilt GPU states, matched uninterrupted dense reference for every method",
            scope="Every method, real current/old models, scalar dense-reference repair vs fused batched independent trajectories; native appends/FIFO and true old boundaries."))
        del reference
        import gc
        gc.collect();torch.cuda.empty_cache()
        cases=[]
        for batch in args.batches:
            graphs=Graphs(True)
            counts=Counter()
            begin=time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for group in waves(inputs,inputs["quality"],batch,cfg["release_day"]*86400):
                _,stats=evaluate_wave(parent,current,inputs,group,cfg,rows,maps,device,graphs)
                counts.update(stats)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin
            case=dict(user_batch=batch,seconds=elapsed,steady_seconds=elapsed-graphs.capture_seconds,
                counts=dict(counts),graphs=graphs.describe(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            cases.append(case)
            write_json(output/"benchmark.json",dict(status="running",edge=edge,cases=cases))
            print(json.dumps(dict(stage="batch_complete",**case)),flush=True)
            del graphs
            gc.collect();torch.cuda.empty_cache()
        # A compressed numeric checkpoint is sufficient for exact AUC and
        # resampling. Exercise rotation + reload without keeping probe scores.
        with TemporaryDirectory(dir=output) as tmp:
            store=Rolling(tmp,ident,0)
            store.append(7,[0,1],np.zeros((2,len(rows)),np.float32))
            store.save(1,len(rows),True);store.save(2,len(rows),True)
            restored=Rolling(tmp,ident)
            assert restored.cursor==2 and len(restored.arrays(len(rows))[0])==2
            assert len(list(Path(tmp).glob("*.npz")))==2
        write_json(output/"benchmark.json",dict(status="complete",edge=edge,cases=cases,fit_seconds=fit_seconds,
            config_sha256=digest(HERE/"config.json"),fit_profiles=fit_profiles,calibration_flops=fit_costs,
            input_load_seconds=inputs["load_seconds"],total_seconds=time.perf_counter()-started,
            checkpoint_rotation_resume="passed; two compressed scalar-only generations; no probe trajectories kept",
            model_device=torch.cuda.get_device_name(),torch_version=torch.__version__))
        return
    store=Rolling(output,ident,cfg["checkpoint_seconds"])
    schedule=list(waves(inputs,inputs["quality"],cfg["user_batch"],cfg["release_day"]*86400))
    graphs=Graphs(True)
    for index in range(store.cursor,len(schedule)):
        data,_=evaluate_wave(parent,current,inputs,schedule[index],cfg,rows,maps,device,graphs)
        for uid,y,s in data:
            store.append(uid,y,s)
        saved=store.save(index+1,len(rows))
        if saved or (index+1)%16==0:
            write_json(output/"progress.json",dict(stage="evaluation",edge=edge,waves_complete=index+1,
                waves_total=len(schedule),seconds=time.perf_counter()-started,last_checkpoint_wave=store.cursor if saved else None))
    store.save(len(schedule),len(rows),True)
    costs=ledger(inputs,cfg,rows,fit_costs)
    write_json(output/"worker_summary.json",dict(status="complete",edge=edge,rank=args.rank,quality_users=len(inputs["quality"]),
        requests=inputs["requests"],costs=costs,fit_profiles=fit_profiles,seconds=time.perf_counter()-started,graphs=graphs.describe(),
        checkpoint_save_seconds=store.save_seconds,checkpoint_saves=store.save_count))

def aggregate(folder,cfg,edge):
    import numpy as np
    rows=policies(cfg)
    analysis_lock=(folder/"analysis.lock").open("w")
    fcntl.flock(analysis_lock,fcntl.LOCK_EX)
    if (folder/"summary.json").exists():
        return json.loads((folder/"summary.json").read_text())
    all_y,all_u,all_s=[],[],[]
    parts=[]
    for rank in range(4):
        path=folder/f"rank{rank}"
        parts.append(json.loads((path/"worker_summary.json").read_text()))
        with np.load(path/"checkpoint.npz",allow_pickle=False) as a:
            all_y.append(a["labels"]);all_u.append(a["uids"]);all_s.append(a["scores"])
    y,u,s=np.concatenate(all_y),np.concatenate(all_u),np.concatenate(all_s)
    assert len(set(u.tolist()))<=75807 and sum(p["requests"] for p in parts)==len(y)
    assert len(y)==[345630,318504,323362,335971,347575][edge-1],"full frozen feedback coverage differs from prior protocol"
    roles=json.loads((ROOT/cfg["role_source"]).read_text())["roles"]
    started=time.perf_counter()
    results,confidence=summarize_arrays(y,u,s,rows,roles["quality"],cfg["bootstrap_replicates"],cfg["bootstrap_seed"])
    denominator=sum(p["costs"]["once_rebuild_flops"] for p in parts)
    for r in results:
        tag=r["tag"]
        repair=sum(p["costs"]["repair_flops"][tag] for p in parts)
        fitting=parts[0]["costs"]["calibration_flops"].get(tag,0)
        r.update(complete_added_flops=repair+fitting,calibration_flops=fitting,
            cost_percent=100*(repair+fitting)/denominator,
            old_boundary_bytes=sum(p["costs"]["old_boundary_bytes"][tag] for p in parts))
        if r["method"]=="kv_map":
            r["calibration_flop_breakdown"]=parts[0]["fit_profiles"][tag]["flop_breakdown"]
            r["deployment_mapping_flops"]=repair
    summary=dict(status="complete",edge=edge,window=[cfg["release_days"][edge-1],cfg["end_days"][edge-1]],
        window_name="E14_partial" if edge==5 else "E14",quality_users=len(roles["quality"]),requests=len(y),
        once_rebuild_flops=denominator,results=results,confidence=confidence,statistics_seconds=time.perf_counter()-started,
        protocol="Independent adjacent edge; old native initialization; own causal state within the edge; no cross-edge state or weights.")
    write_json(folder/"summary.json",summary)
    # All exact statistics are committed before disposing of temporary scalar
    # predictions. Core weights, bindings and numbers are retained.
    for rank in range(4):
        for name in ("checkpoint.npz","previous.npz"):
            (folder/f"rank{rank}"/name).unlink(missing_ok=True)
    return summary

def execute(args):
    """Controller runs inside tmux; one isolated GPU process per card."""
    cfg=configuration()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"controller.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode=="formal" and not args.authorize:
        raise RuntimeError("Formal launch requires the explicit --authorize flag after the user instructs launch")
    if args.mode=="formal":
        frozen=dict(config=cfg,config_sha256=digest(HERE/"config.json"),sources=source_hashes(),
            checkpoints={f"v{v}":digest(ROOT/cfg["checkpoints"]/f"v{v}/checkpoint_100.pt") for v in range(6)},
            roles_sha256=digest(ROOT/cfg["role_source"]),protocol="Independent adjacent edges; baseline quality only; authorized via --authorize")
        path=out/"configuration.json"
        if path.exists() and json.loads(path.read_text())!=frozen:
            raise RuntimeError("formal run configuration changed; preserve existing run and choose a new attempt")
        write_json(path,frozen)
    # A single rotated text log per worker, no unbounded per-step logs.
    def spawn(edge,rank,world,folder):
        folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-u",str(HERE/"run.py"),"worker","--mode",args.mode,
            "--edge",str(edge),"--rank",str(rank),"--world",str(world),"--output",str(folder)]
        if args.mode=="benchmark":
            command += ["--users",str(args.users),"--batches",*map(str,args.batches)]
        node=0 if rank<2 else 1
        command=["numactl",f"--cpunodebind={node}",f"--membind={node}",*command]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(cfg["gpus"][rank]),EVOKV_ATTENTION_BACKEND="triton",
            OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",PYTHONUNBUFFERED="1",
            PYTORCH_ALLOC_CONF="expandable_segments:True")
        with (folder/"worker.log").open("w") as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return child,dict(pid=child.pid,edge=edge,rank=rank,command=command,folder=str(folder))
    records=[]
    analyses=[]
    if args.mode=="benchmark":
        children=[]
        for rank,edge in enumerate([1,2,3,5]):
            p,r=spawn(edge,rank,4,out/f"gpu{rank}")
            children.append(p);records.append(r)
        write_json(out/"launch.json",dict(mode="benchmark",formal_started=False,workers=records))
        status=wait_children(children)
        if any(status):
            write_json(out/"status.json",dict(status="failed",exit_codes=status))
            return 1
        summaries=[json.loads((out/f"gpu{i}"/"benchmark.json").read_text()) for i in range(4)]
        write_json(out/"summary.json",dict(status="complete",formal_started=False,workers=summaries))
        return 0
    for edge in cfg["edges"]:
        if source_hashes()!=frozen["sources"] or digest(HERE/"config.json")!=frozen["config_sha256"]:
            raise RuntimeError("queued execution source changed; no later edge launched")
        folder=out/f"edge{edge}"
        if (folder/"summary.json").exists():
            continue
        children=[]
        for rank in range(4):
            p,r=spawn(edge,rank,4,folder/f"rank{rank}")
            children.append(p);records.append(r)
        write_json(out/"launch.json",dict(mode="formal",workers=records))
        status=wait_children(children)
        if any(status):
            write_json(out/"status.json",dict(status="failed",edge=edge,exit_codes=status))
            return 1
        command=[sys.executable,"-u",str(HERE/"run.py"),"analyze","--edge",str(edge),"--output",str(folder)]
        with (folder/"statistics.log").open("w") as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))
        analyses.append(process)
    if any(wait_children(analyses)):
        write_json(out/"status.json",dict(status="statistics_failed"))
        return 1
    edges=[json.loads((out/f"edge{e}"/"summary.json").read_text()) for e in cfg["edges"]]
    frontier=[]
    for i,p in enumerate(policies(cfg)):
        costs=[e["results"][i]["cost_percent"] for e in edges]
        auc=[e["results"][i]["auc"] for e in edges]
        recovery=[e["results"][i]["recovery_percent"] for e in edges]
        frontier.append(dict(**p,cost_pct_by_release=costs,auc_by_release=auc,recovery_pct_by_release=recovery,
            mean_cost_pct=sum(costs)/5,mean_recovery_pct=sum(recovery)/5,
            complete_E14_mean_recovery_pct=sum(recovery[:4])/4))
    write_json(out/"summary.json",dict(status="complete",edges=edges,frontier=frontier,
        aggregation="Equal-edge mean of per-edge recovery; all five reported, last E14_partial explicitly retained; complete-E14 mean is a separate sensitivity."))
    import csv
    with (out/"results.csv").open("w",newline="") as f:
        names=["edge","window","method","cost_percent","auc","recovery_percent"]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for e in edges:
            for r in e["results"]:
                writer.writerow(dict(edge=e["edge"],window=e["window_name"],method=r["tag"],
                    cost_percent=r["cost_percent"],auc=r["auc"],recovery_percent=r["recovery_percent"]))
    return 0

def launch(args):
    import shlex
    if args.mode=="formal" and not args.authorize:
        raise RuntimeError("Awaiting user instruction; explicit --authorize is required for formal launch")
    # Never reuse a live session or compete with an existing GPU task.
    gpu_state=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    if gpu_state.strip():
        raise RuntimeError("GPU compute processes already present; inspect allocation before launching")
    command=[sys.executable,"-u",str(HERE/"run.py"),"execute","--mode",args.mode,"--output",str(args.output.resolve()),
        "--users",str(args.users),"--batches",*map(str,args.batches)]
    if args.authorize:
        command.append("--authorize")
    args.output.mkdir(parents=True,exist_ok=True)
    shell="exec "+shlex.join(command)+" > "+shlex.quote(str(args.output.resolve()/"controller.log"))+" 2>&1"
    subprocess.run(["tmux","new-session","-d","-s",args.session,shell],check=True)
    print(json.dumps(dict(tmux_session=args.session,mode=args.mode,output=str(args.output.resolve()))))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("launch","execute","worker"):
        p=sub.add_parser(name)
        p.add_argument("--mode",choices=("benchmark","formal"),required=True)
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--users",type=int,default=256)
        p.add_argument("--batches",type=int,nargs="+",default=[1,4])
        p.add_argument("--authorize",action="store_true")
        if name=="launch":
            p.add_argument("--session",default="figure9a-benchmark")
        if name=="worker":
            p.add_argument("--edge",type=int,required=True)
            p.add_argument("--rank",type=int,default=0)
            p.add_argument("--world",type=int,default=4)
    p=sub.add_parser("analyze")
    p.add_argument("--edge",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="worker":
        worker(args)
    elif args.command=="launch":
        launch(args)
    elif args.command=="analyze":
        aggregate(args.output.resolve(),configuration(),args.edge)
    else:
        raise SystemExit(execute(args))

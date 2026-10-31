#!/usr/bin/env python3
"""Small CPU numerical audit and label-free inherited-row exposure analysis."""
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
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
import torch
from experiments.ch3_motivation.code.common import ROOT, restore_pack, write_json
from experiments.ch3_motivation.code.probes import bind_fit_scales, fit_probe, probe_loss
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from experiments.ch3_motivation.code.prepare_scenario_strata import DEFINITIONS
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.common.artifacts import sha256


def numerical_checks():
    torch.manual_seed(319)
    cfg = HSTUConfig(num_items=40, num_behaviors=2, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=8, input_dropout=0., attn_dropout=0.,
        block_variant="hstu_reference", activation="silu", causal_diagonal="inclusive", qk_scale=.7)
    old = HSTU(cfg).eval().requires_grad_(False)
    model = copy.deepcopy(old)
    with torch.no_grad():
        model.blocks[0].attn.k_proj.weight.add_(.03*torch.randn_like(model.blocks[0].attn.k_proj.weight))
        model.item_emb.weight[1:].add_(.01*torch.randn_like(model.item_emb.weight[1:]))
    ids = torch.tensor([[1,2,3,4,5]])
    behavior, delta = torch.ones_like(ids), torch.ones_like(ids, dtype=torch.float32)
    sample = dict(uid=1, item_ids=ids, parent_cache=old.compute_kv(ids,behavior,delta),
        teacher_cache=model.compute_kv(ids,behavior,delta), candidates=torch.tensor([[12,17,20]]),
        query_delta=torch.tensor([[1.,20.,600.]]))
    train, val, _ = bind_fit_scales(model, [sample], [dict(sample, uid=2)])
    probe = build_variant("full_item", model, stage="fgh", hidden_width=8, depth=1,
                          row_chunk=2, query_chunk=2)
    probe.fit_input_statistics(model, train)
    with torch.no_grad():
        for group in (probe.f,probe.g,probe.h):
            for layer in group:layer.weights[-1].normal_(0,.002)
    queries = []
    original_h = probe.interaction_correction
    def traced_h(layer, q, rows):
        if q.requires_grad:q.retain_grad()
        queries.append((layer,q))
        return original_h(layer,q,rows)
    probe.interaction_correction = traced_h
    value, parts = probe_loss(model,probe,train[0])
    value.backward()
    gradients = {name:[dict(first=float(layer.weights[0].grad.abs().sum()),
        output=float(layer.weights[-1].grad.abs().sum())) for layer in group]
        for name,group in (("f",probe.f),("g",probe.g),("h",probe.h))}
    upper_query_gradient = sum(float(q.grad.abs().sum()) for layer,q in queries if layer and q.grad is not None)
    assert all(row["first"]>0 and row["output"]>0 for group in gradients.values() for row in group)
    assert upper_query_gradient>0 and all(p.grad is None and not p.requires_grad for p in model.parameters())
    del probe.interaction_correction
    prediction, trace = probe.score(model,sample["parent_cache"],ids,sample["candidates"],sample["query_delta"],trace=True)
    n = sample["parent_cache"].seq_len
    manual = torch.stack([((actual/n-history_read(model.blocks[layer].attn,q.detach(),
        sample["teacher_cache"].k[layer],sample["teacher_cache"].v[layer])/n)/train[0]["read_scales"][layer]).square().mean()
        for layer,(q,actual) in enumerate(zip(trace.queries,trace.history_heads,strict=True))]).mean()
    torch.testing.assert_close(manual,parts["read_mse"])
    # Step0 remains a real selected state: zero LR executes gradient steps but
    # cannot improve validation, and restoring best must preserve every tensor.
    before = {name:tensor.clone() for name,tensor in probe.state_dict().items()}
    fit = fit_probe(model,probe,train,val,steps=2,learning_rate=0.,patience=0,validation_every=1)
    assert fit["best_step"] == 0 and fit["completed_steps"] == 2
    assert all(torch.equal(tensor,probe.state_dict()[name]) for name,tensor in before.items())
    # The loader initializes CPU MLPs before loading weights. Without an RNG
    # guard a partial-stage resume changes the next zero-output term's hidden
    # random features, even though the initial scores themselves stay equal.
    with tempfile.TemporaryDirectory(prefix="ch3-probe-rng-audit-") as directory:
        path = Path(directory)/"f.pt"
        torch.manual_seed(17)
        f = build_variant("full_item",model,stage="f",hidden_width=8,depth=1)
        save_variant(path,f,dict(diagnostic_only=True,quality_use_permitted=False))
        uninterrupted_fg = f.expanded("fg")
        uninterrupted_h = uninterrupted_fg.expanded("fgh")
        def resumed(guarded):
            torch.manual_seed(17)
            build_variant("full_item",model,stage="f",hidden_width=8,depth=1)
            if guarded:
                with torch.random.fork_rng(devices=[]):loaded,_=load_variant(path)
            else:loaded,_=load_variant(path)
            fg=loaded.expanded("fg")
            return fg,fg.expanded("fgh")
        legacy_fg,legacy_h = resumed(False)
        fixed_fg,fixed_h = resumed(True)
        rng_errors = dict(unprotected_g_hidden_max=float((legacy_fg.g[0].weights[0]-uninterrupted_fg.g[0].weights[0]).abs().max()),
            unprotected_h_hidden_max=float((legacy_h.h[0].weights[0]-uninterrupted_h.h[0].weights[0]).abs().max()),
            guarded_g_hidden_max=float((fixed_fg.g[0].weights[0]-uninterrupted_fg.g[0].weights[0]).abs().max()),
            guarded_h_hidden_max=float((fixed_h.h[0].weights[0]-uninterrupted_h.h[0].weights[0]).abs().max()))
        assert rng_errors["unprotected_g_hidden_max"]>0 and rng_errors["unprotected_h_hidden_max"]>0
        assert rng_errors["guarded_g_hidden_max"] == rng_errors["guarded_h_hidden_max"] == 0
    return dict(status="passed", probe_gradients=gradients, upper_query_gradient_l1=upper_query_gradient,
        frozen_backbone_gradients_absent=True, same_student_query_teacher_read_error=float((manual-parts["read_mse"]).abs()),
        step0_restore_exact=True, actual_executed_steps=2, resume_rng=rng_errors,
        interpretation="unprotected partial resume changes newly added hidden features; fresh runs are unaffected; guard restores uninterrupted initialization exactly",
        population_auc_computed=False)


def summarize_exposure(rows):
    if not rows:return dict(query_events=0,feedback_users=0)
    fraction=np.array([r["inherited_fraction_of_initial"] for r in rows])
    share=np.array([r["inherited_fraction_of_read"] for r in rows])
    remaining=np.array([r["inherited_rows"] for r in rows])
    users={r["uid"] for r in rows}
    user_values=defaultdict(list)
    for row in rows:user_values[row["uid"]].append(row["inherited_fraction_of_initial"])
    by_user=[np.mean(values) for values in user_values.values()]
    return dict(query_events=len(rows),feedback_users=len(users),
        initial_inherited_survival_median=float(np.median(fraction)),
        fraction_query_events_all_old_rows_gone=float(np.mean(remaining==0)),
        fraction_query_events_at_least_half_initial_old_rows=float(np.mean(fraction>=.5)),
        inherited_read_share_median=float(np.median(share)),
        median_user_mean_survival=float(np.median(by_user)),
        post_release_behaviors_before_query_median=float(np.median([r["post_release_behaviors"] for r in rows])))


def input_survival(input_root, feature_path):
    """Project only UID/time from requests; read neither label nor score."""
    input_root,feature_path=Path(input_root),Path(feature_path)
    population=json.loads((input_root/"population.json").read_text())
    users=population["development"]
    features=pq.read_table(feature_path,columns=["uid","edge","retained_history_rows",
        "pre14_behavior_events","pre14_feedback_events"]).to_pylist()
    features={(r["edge"],r["uid"]):r for r in features if r["uid"] in set(users)}
    output={}
    bindings={str(input_root/"population.json"):sha256(input_root/"population.json"),str(feature_path):sha256(feature_path)}
    for edge in range(1,6):
        folder=input_root/f"edge_{edge}"
        metadata=json.loads((folder/"inputs.json").read_text())
        cfg=metadata["settings"]
        at,end=cfg["release_day"]*86400,cfg["end_day"]*86400
        cap=cfg["history_length"]
        history_path=folder/metadata["history"]["path"]
        request_path=folder/metadata["requests"]["path"]
        assert sha256(history_path)==metadata["history"]["sha256"] and sha256(request_path)==metadata["requests"]["sha256"]
        bindings[str(folder/"inputs.json")]=sha256(folder/"inputs.json")
        bindings[str(history_path)]=metadata["history"]["sha256"]
        bindings[str(request_path)]=metadata["requests"]["sha256"]
        histories=restore_pack(history_path)
        requested=pq.read_table(request_path,columns=["uid","query_timestamp"]).to_pylist()
        membership={}
        for uid in users:
            f=features[edge,uid]
            membership[uid]=["all_development"]
            for name,definition in DEFINITIONS.items():
                if (f["retained_history_rows"]>=definition["retained_history_rows_at_least"] and
                    f["pre14_behavior_events"]>=definition["pre14_behavior_events_at_least"] and
                    f["pre14_feedback_events"]>=definition["pre14_feedback_events_at_least"]):membership[uid].append(name)
            if f["retained_history_rows"]>=512 and 32<=f["pre14_behavior_events"]<128:membership[uid].append("long_moderate32_127")
            if f["retained_history_rows"]>=512 and f["pre14_behavior_events"]>=1024:membership[uid].append("long_very_active1024")
        rows=[]
        for request in requested:
            uid,timestamp=int(request["uid"]),int(request["query_timestamp"])
            if uid not in membership:continue
            assert at<=timestamp<end
            times=histories.rows[uid][0]
            initial=min(cap,int(np.searchsorted(times,at,side="left")))
            appended=int(np.searchsorted(times,timestamp,side="left")-np.searchsorted(times,at,side="left"))
            remaining=min(initial,max(0,cap-appended))
            total=min(cap,initial+appended)
            rows.append(dict(uid=uid,post_release_behaviors=appended,inherited_rows=remaining,
                inherited_fraction_of_initial=remaining/initial if initial else 0.,
                inherited_fraction_of_read=remaining/total if total else 0., elapsed_days=(timestamp-at)/86400))
        cohorts={name:dict(reserved_users=sum(name in groups for groups in membership.values()),
            **summarize_exposure([r for r in rows if name in membership[r["uid"]]]))
            for name in sorted({name for values in membership.values() for name in values})}
        output[str(edge)]=dict(development_users=len(users),cohorts=cohorts,
            by_elapsed_window={label:summarize_exposure([r for r in rows if low<=r["elapsed_days"]<high])
                for label,low,high in (("D0_1",0,1),("D1_7",1,7),("D7_14",7,14))})
    return dict(edges=output,input_sha256=bindings,population="unchanged development512 only",
        projected_request_columns=["uid","query_timestamp"],labels_read=False,scores_read=False,
        inherited_rule="n0=min(cap,events<t_release); a=events in [release,t_query); m=min(n0,max(0,cap-a)); strict-before includes no same-timestamp behavior",
        definitions=DEFINITIONS,additional_input_only_bins=dict(long_moderate32_127="history>=512 and32<=pre14behavior<128",
            long_very_active1024="history>=512 andpre14behavior>=1024 (structural cache-cap threshold)"),
        scope="input exposure diagnosis, not AUC, cohort quality qualification or method-outcome-based population selection")


def fixed_panel_survival(input_root, natural_root):
    """Compare already frozen panels; membership never depends on survival."""
    input_root,natural_root=Path(input_root),Path(natural_root)
    population_path=input_root/"population.json"
    population=json.loads(population_path.read_text())
    output={};bindings={str(population_path):sha256(population_path)}
    for edge in range(1,6):
        output[str(edge)]={}
        for root,names in ((input_root,("development","main","confirmation")),
                           (natural_root,("long_feedback_active",))):
            folder=root/f"edge_{edge}";binding_path=folder/"inputs.json"
            metadata=json.loads(binding_path.read_text());cfg=metadata["settings"]
            bindings[str(binding_path)]=sha256(binding_path)
            history_path=folder/metadata["history"]["path"];request_path=folder/metadata["requests"]["path"]
            assert sha256(history_path)==metadata["history"]["sha256"] and sha256(request_path)==metadata["requests"]["sha256"]
            bindings[str(history_path)]=metadata["history"]["sha256"];bindings[str(request_path)]=metadata["requests"]["sha256"]
            histories=restore_pack(history_path)
            requests=pq.read_table(request_path,columns=["uid","query_timestamp"]).to_pylist()
            at,end,cap=cfg["release_day"]*86400,cfg["end_day"]*86400,cfg["history_length"]
            for name in names:
                users=metadata["candidate_panels"][name] if root==natural_root else population[name]
                selected=set(users);rows=[]
                assert len(selected)==(512 if name=="development" else 2000)
                for request in requests:
                    uid,timestamp=int(request["uid"]),int(request["query_timestamp"])
                    if uid not in selected:continue
                    assert at<=timestamp<end
                    times=histories.rows[uid][0];cut=int(np.searchsorted(times,at,side="left"))
                    initial=min(cap,cut);appended=int(np.searchsorted(times,timestamp,side="left")-cut)
                    remaining=min(initial,max(0,cap-appended));total=min(cap,initial+appended)
                    rows.append(dict(uid=uid,post_release_behaviors=appended,inherited_rows=remaining,
                        inherited_fraction_of_initial=remaining/initial if initial else 0.,
                        inherited_fraction_of_read=remaining/total if total else 0.,elapsed_days=(timestamp-at)/86400))
                output[str(edge)][name]=dict(reserved_users=len(users),**summarize_exposure(rows),
                    by_elapsed_window={label:summarize_exposure([r for r in rows if low<=r["elapsed_days"]<high])
                        for label,low,high in (("D0_1",0,1),("D1_7",1,7),("D7_14",7,14))})
    return dict(edges=output,input_sha256=bindings,projected_request_columns=["uid","query_timestamp"],
        labels_read=False,scores_read=False,new_membership_selected=False,
        long_feedback_active_rule="already frozen history>=512/pre14behavior>=32/pre14feedback>=4; no future survival selection",
        scope="input exposure only; no AUC or qualification claim; original E14 windows retained")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--features",type=Path,default=ROOT/"experiments/ch3_motivation/runs/scenario_qualification/strata_pre_release/attempt_1/pre_release_features.parquet")
    parser.add_argument("--natural-root",type=Path,default=ROOT/"experiments/ch3_motivation/runs/scenario_qualification/natural_inputs/attempt_1")
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=dict(numerical=numerical_checks(),input_survival=input_survival(args.input_root,args.features),
        fixed_panels=fixed_panel_survival(args.input_root,args.natural_root),
        source_sha256=sha256(Path(__file__)),gpu_used=False)
    write_json(args.output,result)
    print(json.dumps(dict(status="passed",gpu_used=False,output=str(args.output))))

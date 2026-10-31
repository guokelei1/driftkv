#!/usr/bin/env python3
"""CPU-only real fitting-input audit of h's q/row initialization contribution.

The nonzero mixed feature contrast is a random-feature sensitivity diagnostic;
the real new h output layer remains exactly zero. No weight is trained/saved.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.calibration_gaps import load_gap_manifest
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant
from hstu_kvcache.common.artifacts import sha256


@torch.no_grad()
def run(edges=(2,5),users=2):
    root=ROOT/"experiments/ch3_motivation/inputs_feedback_pool/fit64"
    population=json.loads((root/"population.json").read_text())
    uids=population["fit"][:users]
    result=[];sources={str(Path(__file__).resolve()):sha256(Path(__file__))}
    for edge in edges:
        folder=root/f"edge_{edge}"
        binding=json.loads((folder/"inputs.json").read_text())
        candidate=json.loads((folder/"calibration_feedback_candidates.json").read_text())
        history=folder/binding["history"]["path"]
        assert sha256(history)==binding["history"]["sha256"]
        parent,model,models=load_models(edge,torch.device("cpu"),backend="torch")
        samples=capture_samples(parent,model,restore_pack(history),uids,
            dict(binding["settings"],calibration_queries_per_user=16,seed=17),torch.device("cpu"))
        del parent
        checkpoint=ROOT/f"experiments/ch3_motivation/runs/e5/ladder_stability/attempt_1/full_item128_lr3e5_fit64/edge_{edge}/seed_17/fg.pt"
        probe,metadata=load_variant(checkpoint)
        assert metadata["identity"]["models"]==models
        # Recreate the uninterrupted fresh seed17 initialization sequence.
        torch.manual_seed(17)
        fresh=build_variant("full_item",model,stage="f",hidden_width=128,depth=1,
                            query_chunk=16,row_chunk=1024).expanded("fg").expanded("fgh")
        fresh.load_state_dict(probe.state_dict(),strict=False)
        manifest=ROOT/f"experiments/ch3_motivation/runs/e5/calibration_gap_views/attempt_1/edge_{edge}/gap_manifest.json"
        sources[str(checkpoint)]=sha256(checkpoint);sources[str(manifest)]=sha256(manifest)
        sources[str(history)]=binding["history"]["sha256"]
        sources.update(models)
        for rule in ("cutover","gap_mix50","gap_empirical100"):
            gaps,provenance=load_gap_manifest(manifest,root,edge,rule)
            records=[]
            for uid in uids:
                s=samples[uid]
                ids=torch.tensor([candidates_for(uid,candidate["recent_items"][str(uid)],
                    model.cfg.num_items-256,count=16,recent_fraction=.5,seed=17)])
                assert ids.tolist()==[provenance["candidate_ids"][str(uid)]]
                features=probe.prepare_read_features(model,s["parent_cache"],s["item_ids"])
                prepared=probe.prepare_cache(model,s["parent_cache"],s["item_ids"],read_features=features)
                _,trace=probe.score(model,s["parent_cache"],s["item_ids"],ids,gaps[uid],trace=True,
                                    prepared_cache=prepared,read_features=features)
                for layer,q in enumerate(trace.queries):
                    c=probe.cfg;rows=features[layer]
                    q=q/probe.query_scale[layer][None,:,None]
                    w=fresh.h[layer].weights[0]
                    zq=torch.einsum("bhqi,hiw->bhqw",q,w[:,:c.head_dim])
                    zr=torch.einsum("bhni,hiw->bhnw",rows,w[:,c.head_dim:])
                    eq,er=zq.square().mean(),zr.square().mean()
                    vq=(zq-zq.mean(2,keepdim=True)).square().mean()
                    vr=(zr-zr.mean(2,keepdim=True)).square().mean()
                    aq=((c.head_dim+c.row_dim)/(2*c.head_dim))**.5
                    ar=((c.head_dim+c.row_dim)/(2*c.row_dim))**.5
                    # Direct concat vs split on a small genuine query/row slice.
                    pair=torch.cat((q[:,:,:2,None].expand(-1,-1,-1,8,-1),
                        rows[:,:,:8][:,:,None].expand(-1,-1,2,-1,-1)),-1)
                    direct=torch.einsum("bhqni,hiw->bhqnw",pair,w)
                    split=zq[:,:,:2,None]+zr[:,:,:8][:,:,None]
                    maximum=float((direct-split).abs().max())
                    assert maximum<2e-5
                    def mixed(qpart,rpart):
                        left,right=qpart[:,:,0::2,None],qpart[:,:,1::2,None]
                        count=min(64,rpart.shape[2]);count-=count%2
                        a,b=rpart[:,:,:count:2][:,:,None],rpart[:,:,1:count:2][:,:,None]
                        return (F.silu(left+a)-F.silu(left+b)-F.silu(right+a)+F.silu(right+b)).square().mean().sqrt()
                    original_mixed=mixed(zq,zr);balanced_mixed=mixed(zq*aq,zr*ar)
                    records.append(dict(uid=uid,layer=layer,history_rows=s["history_count"],
                        q_projection_energy=float(eq),row_projection_energy=float(er),
                        q_energy_fraction=float(eq/(eq+er)),candidate_centered_q_energy_fraction=float(vq/(vq+vr)),
                        normalized_q_candidate_variance_fraction=float((q-q.mean(2,keepdim=True)).square().mean()/q.square().mean()),
                        balanced_q_energy_fraction=float(eq*aq**2/(eq*aq**2+er*ar**2)),
                        balanced_candidate_centered_q_energy_fraction=float(vq*aq**2/(vq*aq**2+vr*ar**2)),
                        mixed_hidden_feature_rms=float(original_mixed),balanced_mixed_hidden_feature_rms=float(balanced_mixed),
                        balanced_over_original_mixed_ratio=float(balanced_mixed/original_mixed),concat_split_max_error=maximum))
            fields=("q_energy_fraction","candidate_centered_q_energy_fraction","normalized_q_candidate_variance_fraction",
                "balanced_q_energy_fraction","balanced_candidate_centered_q_energy_fraction","balanced_over_original_mixed_ratio")
            result.append(dict(edge=edge,rule=rule,fit_uids=uids,layers=10,rows=records,
                equal_user_layer_means={key:sum(r[key] for r in records)/len(records) for key in fields}))
        del model,samples,probe,fresh
    return dict(status="passed",gpu_used=False,population_auc_computed=False,feedback_labels_read=False,
        measurements=result,input_sha256=sources,
        theory=dict(head=dict(q_dim=32,row_dim=96,q_expected_energy_fraction=32/128),
            full_item=dict(q_dim=32,row_dim=384,q_expected_energy_fraction=32/416),
            layerwise_kv=dict(q_dim=32,row_dim=960,q_expected_energy_fraction=32/992),
            balanced_initialization="scale Wq by sqrt((D+F)/(2D)), Wr by sqrt((D+F)/(2F)); same dense h shape, zero output and objective; not applied to any running weight"),
        interpretation="real pre-release q/row inputs; q/row first-affine contributions and candidate-centered variation; nonzero hidden feature contrasts do not imply trained h quality or a low-rank constraint")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges",type=int,nargs="+",default=[2,5])
    parser.add_argument("--users",type=int,default=2)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();value=run(args.edges,args.users);write_json(args.output,value)
    print(json.dumps(dict(status=value["status"],gpu_used=False,output=str(args.output))))

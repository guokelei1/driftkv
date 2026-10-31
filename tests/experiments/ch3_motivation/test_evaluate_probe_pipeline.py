"""CPU experiment pipeline with 512 synthetic residents and real typed probes.

Only two residents have feedback. Idle residents must still be in the full
deployment ledger. No files from this fixture are research measurements.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import evaluate_probes as driver
from experiments.ch3_motivation.code.probe_variants import build_variant, save_variant
from experiments.ch3_motivation.code.selective import evaluate_reference
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.common.artifacts import sha256


class CPUTorch:
    cuda=SimpleNamespace(set_device=lambda _:None,get_device_name=lambda _:"CPU synthetic fixture")

    def __getattr__(self,name):return getattr(torch,name)

    @staticmethod
    def device(_):return torch.device("cpu")


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
def test_typed_quality_pipeline_shared_anchors_resume_and_diagnostic_guard(tmp_path,monkeypatch,variant):
    threads=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    try:
        torch.manual_seed(193)
        config=HSTUConfig(num_items=40,num_behaviors=2,hidden_size=32,num_heads=1,
            num_layers=1,max_seq_len=4,input_dropout=0.,attn_dropout=0.)
        parent=HSTU(config).eval().requires_grad_(False)
        current=copy.deepcopy(parent)
        with torch.no_grad():current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.03)
        users=list(range(1000,1512))
        roles=dict(fit=[1,2],validation=[3],development=users,main=users,confirmation=users)
        edge=tmp_path/"inputs/edge_1";edge.mkdir(parents=True)
        (edge.parent/"population.json").write_text(json.dumps(roles))
        for name in ("history","requests"):(edge/f"{name}.pack").write_text("synthetic fixture")
        cfg=dict(edge=1,release_day=1,end_day=2,history_length=4,append_band_size=2,
            query_batch=16,repair_query_batch=16,attention_backend="torch")
        binding=dict(settings=cfg,population_sha256=sha256(edge.parent/"population.json"),
            **{name:dict(path=f"{name}.pack",sha256=sha256(edge/f"{name}.pack")) for name in ("history","requests")})
        (edge/"inputs.json").write_text(json.dumps(binding))
        prefix=(86400+np.arange(-4,0)*10,np.arange(1,5),np.zeros(4,dtype=np.int64))
        active=(86400+np.arange(-4,2)*10,np.arange(1,7),np.zeros(6,dtype=np.int64))
        histories=SimpleNamespace(rows={uid:(active if uid in users[:2] else prefix) for uid in users})
        by_user={uid:{86415:[dict(uid=uid,query_timestamp=86415,raw_item_id=21,item_idx=21,label=0)],
            86425:[dict(uid=uid,query_timestamp=86425,raw_item_id=22,item_idx=22,label=1)]} for uid in users[:2]}
        models={"fixture":"synthetic frozen backbones"}
        identity=dict(edge=1,models=models,fit_uids=roles["fit"],validation_users=roles["validation"],
            input_binding_sha256=sha256(edge/"inputs.json"))
        metadata=dict(identity=identity,variant=variant,seed=17,training_users=2,
            cumulative_calibration_flops_estimate=123)
        checkpoints={}
        for stage in ("f","fg"):
            probe=build_variant(variant,current,stage=stage,hidden_width=5)
            path=tmp_path/f"{stage}.pt";save_variant(path,probe,dict(metadata,method_label=stage))
            checkpoints[stage]=path
        calls=dict(models=0,anchors=0,probe=0)
        def load_models(*_):
            calls["models"]+=1;return parent,current,models
        def anchors(old,new,histories,by_user,uids,policies,cfg,device,output,**_):
            calls["anchors"]+=1
            return evaluate_reference(old,new,histories,by_user,uids,policies,cfg,device)
        actual_evaluate=driver.evaluate_probe
        def evaluate(*args,**kwargs):
            calls["probe"]+=1;return actual_evaluate(*args,**kwargs)
        monkeypatch.setattr(driver,"torch",CPUTorch())
        monkeypatch.setattr(driver,"load_edge",lambda _: (binding,histories,by_user))
        monkeypatch.setattr(driver,"load_models",load_models)
        monkeypatch.setattr(driver,"evaluate_checkpointed",anchors)
        monkeypatch.setattr(driver,"evaluate_probe",evaluate)
        context={}
        args=SimpleNamespace(input_root=edge.parent,edge=1,gpu=0,panel="development",population_seed=20261023,
            population_file=None,checkpoint=checkpoints["f"],output=tmp_path/"first",backend="reference")
        first=driver.run(args,shared_context=context)
        args.checkpoint=checkpoints["fg"];args.output=tmp_path/"second"
        second=driver.run(args,shared_context=context)
        assert calls==dict(models=1,anchors=1,probe=4)
        assert first["feedback_events"]==second["feedback_events"]==4
        assert first["quality_users"]==second["quality_users"]==512
        assert first["auc"]==second["auc"]==second["reuse_auc"]
        assert sum(row["calls"] for row in second["execution"]["initial_shapes"])==512
        assert second["users_with_feedback"]==2
        assert second["anchor_source"]==first["anchor_source"]
        reference=json.loads((args.output/"anchors/shared_anchor_reference.json").read_text())
        assert Path(reference["path"])==tmp_path/"first/anchors"
        hashes={p.name:sha256(p) for p in (args.output/"probe_blocks").glob("*.parquet")}
        resumed=driver.run(args,shared_context=context)
        assert calls==dict(models=1,anchors=1,probe=4)
        assert resumed["auc"]==second["auc"]
        assert hashes=={p.name:sha256(p) for p in (args.output/"probe_blocks").glob("*.parquet")}
        assert "normalized_230_percent_of_once_rebuild" in (args.output/"points.csv").read_text().splitlines()[0]
        path=tmp_path/"diagnostic.pt"
        save_variant(path,probe,dict(metadata,diagnostic_only=True,quality_use_permitted=False))
        args.checkpoint=path;args.output=tmp_path/"diagnostic_rejected"
        with pytest.raises(ValueError,match="diagnostic checkpoint"):
            driver.run(args,shared_context=context)
    finally:
        torch.set_num_threads(threads)

"""Bounded diagnostic checkpoints never masquerade as trained quality points."""
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.check_probe_variants import save_diagnostic_probe
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant
from hstu_kvcache.models import HSTU,HSTUConfig


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
@pytest.mark.parametrize("nonzero_h",[False,True])
def test_diagnostic_saved_fit_and_separate_pressure_preserve_identity(tmp_path,monkeypatch,variant,nonzero_h):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    torch.manual_seed(435)
    model=HSTU(HSTUConfig(num_items=16,num_behaviors=2,hidden_size=8,num_heads=2,
        num_layers=1,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    ids=torch.tensor([[1,2,3,4]])
    cache=model.compute_kv(ids,torch.ones_like(ids),torch.ones_like(ids,dtype=torch.float32))
    sample=dict(parent_cache=cache,item_ids=ids,candidates=torch.tensor([[5,6,7]]),query_delta=torch.ones(1))
    probe=build_variant(variant,model,stage="fgh",hidden_width=7,depth=1)
    probe.fit_input_statistics(model,[sample])
    if nonzero_h:
        with torch.no_grad():
            probe.h[0].weights[-1].normal_(std=.001)
    identity=dict(edge=1,models={"tiny":"fixture"},fit_uids=[11],validation_uids=[21])
    stages=[dict(stage=stage,bounded_training=dict(best_step=1 if nonzero_h else 0,completed_steps=2))
            for stage in ("f","fg","fgh")]
    result=save_diagnostic_probe(tmp_path,variant,probe,model,sample,identity,stages)
    loaded,metadata=load_variant(result["trained"]["path"])
    assert type(loaded) is type(probe)
    assert metadata["h_output"]["nonzero"] is nonzero_h
    assert metadata["best_step"]==(1 if nonzero_h else 0)
    assert metadata["diagnostic_only"] and not metadata["quality_use_permitted"]
    assert not metadata["quality_auc_computed"]
    assert metadata["identity"]["validation_users"]==[21]
    for name,value in probe.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name],value,atol=0,rtol=0)
    if nonzero_h:
        assert "nonzero_pressure" not in result
    else:
        pressure,pressure_metadata=load_variant(result["nonzero_pressure"]["path"])
        assert pressure_metadata["h_output"]["nonzero"]
        assert pressure_metadata["pressure_control"] and not pressure_metadata["trained_quality"]
        assert not pressure_metadata["quality_use_permitted"]
        for name,value in probe.state_dict().items():
            if not name.startswith("h."):
                torch.testing.assert_close(pressure.state_dict()[name],value,atol=0,rtol=0)

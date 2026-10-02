from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
from hstu_kvcache.read_correction_v4.token_read import score_token_corrected
from read_correction_v4.history_conditioned.fit_query_compensation import fit_query_compensation


def test_exact_constant_read_residual_and_frozen_token_maps():
    torch.manual_seed(53)
    model = HSTU(HSTUConfig(num_items=40,num_behaviors=3,hidden_size=8,num_heads=2,
        num_layers=2,max_seq_len=16,input_dropout=0.,temporal_num_freqs=2)).eval().requires_grad_(False)
    modules = [NonlinearTokenReadCorrection(2,4,nonlinear_width=4).eval() for _ in model.blocks]
    with torch.no_grad():
        for module in modules:
            # Nonlinear value map is nontrivial; mapped keys stay zero, making
            # an added teacher value offset exactly a constant per-token read rate.
            module.output.weight[8:].normal_(std=.03)
    before = [{name:value.clone() for name,value in module.state_dict().items()} for module in modules]
    rows={}
    with torch.no_grad():
        for uid in range(16):
            count=4+uid%3
            k=torch.zeros(2,1,count,8)
            v=torch.randn_like(k)*.2
            teacher_k=[];teacher_v=[]
            for layer,module in enumerate(modules):
                mk,mv=module.map_tokens(k[layer],v[layer],torch.tensor([count]))
                teacher_k.append(mk)
                teacher_v.append(mv+torch.tensor([.03,-.02,.01,.04,-.03,.02,.02,-.01]))
            rows[uid]={"parent":HSTUKVCache(k,v,count),
                "teacher":HSTUKVCache(torch.stack(teacher_k),torch.stack(teacher_v),count),
                "candidates":torch.tensor([1+uid,18+uid,34,35]),"query_delta":float(uid+1)}
    fitted,stats=fit_query_compensation(model,rows,list(range(12)),list(range(12,16)),modules,
        {"attention_backend":"torch","ridge":.01},"cpu","medium",batch_size=4)
    for layer,module in enumerate(modules):
        assert module.query_correction is None
        for name,value in module.state_dict().items():
            assert torch.equal(value,before[layer][name])
            assert torch.equal(value,fitted[layer].state_dict()[name])
        assert fitted[layer].get_config()["query_affine"] is True
        assert all(not p.requires_grad for p in fitted[layer].parameters())
        assert stats["layers"][layer]["diagnostics"]["validation"]["after_read_mse"] < 1e-12
    with torch.no_grad():
        for uid in range(12,16):
            row=rows[uid];counts=torch.tensor([row["parent"].seq_len])
            actual=score_token_corrected(model,row["parent"],row["candidates"][None],torch.tensor([row["query_delta"]]),fitted,counts)[0]
            teacher=score_token_corrected(model,row["teacher"],row["candidates"][None],torch.tensor([row["query_delta"]]),[None,None],counts)[0]
            torch.testing.assert_close(actual,teacher,rtol=2e-5,atol=2e-6)
    assert stats["selection"].startswith("none")

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.read_correction import score_corrected
from hstu_kvcache.read_correction.history_conditioned.core import HistoryCorrection
from hstu_kvcache.read_correction.history_conditioned.v2 import HistoryCorrectionV2
from read_correction_2026_09.v2.history_conditioned.fit import apply_context, fit, project_context


def model(seed, reference=False, diagonal="inclusive"):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(num_items=32, num_behaviors=3, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=16, input_dropout=0., temporal_num_freqs=2,
        block_variant="hstu_reference" if reference else "legacy",
        activation="silu" if reference else "elu_plus1", causal_diagonal=diagonal)).eval().requires_grad_(False)


@pytest.mark.parametrize("reference,diagonal", [(False,"inclusive"),(False,"exclusive"),(True,"inclusive"),(True,"exclusive")])
@torch.no_grad()
def test_cached_prefix_steps_match_original_reader(reference, diagonal):
    parent, current = model(3, reference, diagonal), model(7, reference, diagonal)
    items = torch.tensor([[1,2,3,4],[5,6,7,8]])
    cache = parent.compute_kv(items, torch.ones_like(items), torch.zeros_like(items).float())
    candidates, dt, counts = torch.tensor([[9,10,11],[12,13,14]]), torch.tensor([1.,2.]), torch.tensor([4,4])
    modules = [HistoryCorrectionV2(2,4,encoder_width=8) for _ in current.blocks]
    for module in modules:
        module.output.weight.normal_(std=.01)
    _, trace = score_corrected(current,cache,candidates,dt,modules,counts,trace=True)
    hidden = current.embed_query_tokens(candidates,dt)
    for layer, (block, module) in enumerate(zip(current.blocks,modules)):
        context = project_context(block,hidden,cache.k[layer],cache.v[layer])
        torch.testing.assert_close(context["query"],trace.queries[layer],rtol=2e-5,atol=2e-6)
        hidden = apply_context(block,context,cache.k[layer],cache.v[layer],counts,module)
        torch.testing.assert_close(hidden,trace.layer_outputs[layer],rtol=2e-5,atol=2e-6)


def test_zero_epoch_fit_preserves_base_and_records_disjoint_output_validation():
    parent,current = model(17),model(29)
    rows={}
    with torch.no_grad():
        for uid in range(4):
            items=torch.tensor([[1+uid,2+uid,3+uid,4+uid]])
            fields=(items,torch.ones_like(items),torch.zeros_like(items).float())
            rows[uid]={"parent":parent.compute_kv(*fields),"teacher":current.compute_kv(*fields),
                       "candidates":torch.tensor([12+uid,18+uid]),"query_delta":2.}
    bases=[HistoryCorrection(2,4,width=5) for _ in current.blocks]
    with torch.no_grad():
        for base in bases:
            base.history_weight.normal_(std=.001)
    config={"attention_backend":"torch","history_epochs":0,"seed":17,
            "history_encoder_width_fraction":.5,"history_attention_heads":2,
            "history_query_chunk":1,"history_learning_rate":.001,"weight_decay":0.}
    modules,stats=fit(current,rows,[0,1,2],bases,config,"cpu","medium",2,validation_uids=[3])
    assert stats["optimizer_steps"]==0
    assert stats["validation_uids"]==[3]
    assert stats["final_output_validation"]["base"]==stats["final_output_validation"]["v2"]
    assert stats["cost"]["history_train_flops_estimate"]==0
    assert all(record["selected_epoch"]==0 for record in stats["layers"])
    for base,new in zip(bases,modules):
        for name,value in base.state_dict().items():
            torch.testing.assert_close(value,new.base.state_dict()[name],rtol=0.,atol=0.)

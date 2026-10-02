import pytest
import torch

from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.read_correction_v4 import TokenReadCorrection, fit_affine_tokens, score_token_corrected


def model(seed=17):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(num_items=32, num_behaviors=3, hidden_size=8, num_heads=2,
        num_layers=2, max_seq_len=16, input_dropout=0., temporal_num_freqs=2)).eval()


@torch.no_grad()
def test_identity_mapping_matches_reuse_without_mutating_cache():
    parent, current = model(17), model(19)
    items = torch.tensor([[1,2,3,4],[5,6,7,8]])
    cache = parent.compute_kv(items, torch.ones_like(items), torch.zeros_like(items).float())
    original = (cache.k.clone(), cache.v.clone())
    candidates, delta, counts = torch.tensor([[9,10,11],[12,13,14]]), torch.tensor([1.,2.]), torch.tensor([4,4])
    modules = [TokenReadCorrection(2,4,query_affine=True) for _ in current.blocks]
    expected = current.score_cc_reuse(cache,candidates,delta)
    actual, trace = score_token_corrected(current,cache,candidates,delta,modules,counts,trace=True)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    assert torch.equal(cache.k,original[0]) and torch.equal(cache.v,original[1])
    for correction in trace.corrections:
        torch.testing.assert_close(correction,torch.zeros_like(correction),rtol=0.,atol=0.)


@torch.no_grad()
def test_known_cross_head_token_map_gives_same_query_teacher_read():
    torch.manual_seed(23)
    attention = model().blocks[0].attn.double()
    query = torch.randn(2,2,3,4,dtype=torch.float64)
    keys, values = torch.randn(2,5,8,dtype=torch.float64), torch.randn(2,5,8,dtype=torch.float64)
    joined = torch.cat((keys,values),-1)
    slope = torch.randn(16,16,dtype=torch.float64)*.1
    bias = torch.randn(16,dtype=torch.float64)*.05
    teacher = joined + joined@slope + bias
    module = TokenReadCorrection(2,4).double()
    module.map_weight.copy_(slope); module.map_bias.copy_(bias)
    actual = module.forward_new_read(query,keys,values,torch.tensor([5,5]),attention)
    teacher_k, teacher_v = teacher.split(8,-1)
    expected = history_read(attention,query,teacher_k,teacher_v)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


@torch.no_grad()
def test_padding_and_old_producer_prefix_are_respected():
    torch.manual_seed(29)
    module = TokenReadCorrection(2,4).double()
    module.map_weight.normal_(std=.05); module.map_bias.normal_(std=.1)
    attention = model().blocks[0].attn.double()
    query = torch.randn(3,2,2,4,dtype=torch.float64)
    keys, values = torch.randn(3,6,8,dtype=torch.float64), torch.randn(3,6,8,dtype=torch.float64)
    counts, old_counts = torch.tensor([6,4,0]), torch.tensor([2,0,0])
    mapped_k,mapped_v = module.map_tokens(keys,values,counts,old_counts)
    torch.testing.assert_close(mapped_k[0,2:],keys[0,2:],rtol=0.,atol=0.)
    torch.testing.assert_close(mapped_v[1],values[1],rtol=0.,atol=0.)
    actual = module.forward_new_read(query,keys,values,counts,attention,old_counts)
    expected = torch.zeros_like(actual)
    for user in range(2):
        n=int(counts[user])
        expected[user:user+1] = history_read(attention,query[user:user+1],mapped_k[user:user+1,:n],mapped_v[user:user+1,:n])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    # Explicit padding masks exclude otherwise nonzero K/V rows.
    padded_k,padded_v=keys.clone(),values.clone()
    padded_k[1,4:]=1e4; padded_v[1,4:]=-1e4
    torch.testing.assert_close(module.forward_new_read(query,padded_k,padded_v,counts,attention,old_counts),actual,rtol=0.,atol=0.)


@pytest.mark.parametrize("ridge", [0.,.03])
def test_closed_form_affine_fit_and_unpenalized_intercept(ridge):
    torch.manual_seed(31)
    x = torch.randn(80,6,dtype=torch.float64)*torch.tensor([.01,.1,1.,10.,100.,1000.]) + 3.
    target = x@torch.randn(6,6,dtype=x.dtype)*.01 + torch.randn(6,dtype=x.dtype)
    params,stats = fit_affine_tokens(x,target,ridge=ridge)
    normalized = (x-params["input_mean"])/params["input_scale"]
    prediction = normalized@params["map_weight"]+params["map_bias"]
    if ridge == 0:
        torch.testing.assert_close(prediction,target,rtol=1e-10,atol=1e-10)
    else:
        design = torch.cat((normalized,torch.ones(len(x),1,dtype=x.dtype)),dim=1)
        penalty = torch.diag(torch.tensor([ridge]*6+[0.],dtype=x.dtype))
        reference = torch.linalg.solve(design.T@design/len(x)+penalty,design.T@target/len(x))
        torch.testing.assert_close(prediction,design@reference,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(params["map_bias"],target.mean(0))
    assert stats["solve_dtype"] == "float64"

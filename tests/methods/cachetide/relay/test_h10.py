"""Explicit H10 storage, affine evolution and inference conversion checks."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]

from hstu_kvcache.methods.cachetide.structure import StructuredKVAdapter
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.cachetide.prepare import compile_item_tables
from hstu_kvcache.methods.cachetide.state import InplaceReleasePrefix
from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.features import build_bank, load_bank
from hstu_kvcache.methods.cachetide.precision import StableReleasePrefix


def source(layout="shared12_private6", device="cpu"):
    result = StructuredKVAdapter(2, 10, 32, layout=layout).to(device).eval()
    with torch.no_grad():
        for layer in result.layers:
            layer.input_weight.normal_(std=.02)
            layer.input_bias.normal_(std=.01)
            layer.decoder.normal_(std=.01)
            layer.key_decoder.normal_(std=.01)
    return result


@pytest.mark.parametrize("layout,width", [("shared12_private6",72), ("shared6_private3",36)])
@torch.no_grad()
def test_explicit_width_composition_and_zero_birth(layout, width):
    torch.manual_seed(21)
    origin = source(layout)
    adapter = UnifiedRelayAdapter.from_structured(origin, mode="A", read_shared=3, read_private=3)
    cache = HSTUKVCache(torch.randn(2,2,7,320), torch.randn(2,2,7,320), 7)
    catalog = torch.randn(19,320)
    ids = torch.randint(0,19,(2,7))
    zero = cache.k.new_zeros(2,2,7,width)
    state = zero
    prefix = None
    assert adapter.code_shape == (width,)
    assert origin.encode_cache(cache,catalog[ids]).codes.shape == zero.shape
    assert origin.code_width == adapter.code_width == width
    for _ in range(3):
        current = copy.deepcopy(adapter)
        for layer in current.layers:
            for rotation in layer.rotation:
                rotation.normal_(std=.01)
            layer.item_update_bias.normal_(std=.01)
        state = current.advance_codes(cache,state,catalog[ids])
        prefix = current.compile_release(prefix,catalog)
        stable = StableReleasePrefix.wrap(prefix)
        # Native source starts with zero correction; all following states
        # must agree between explicit recurrence and the composed prefix.
        from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
        anchors = LightweightValueState(zero.clone(),7)
        compiled = stable.materialize(cache,anchors,ids)
        torch.testing.assert_close(compiled.codes,state.codes,atol=3e-6,rtol=3e-5)
        assert compiled.storage_bytes() == 2*2*7*width*4
        restored = stable.anchorize(cache,compiled,ids)
        torch.testing.assert_close(restored.codes,zero,atol=3e-6,rtol=0)
        born = stable.zero_anchors(cache,ids)
        torch.testing.assert_close(stable.materialize(cache,born,ids).codes,zero,atol=3e-6,rtol=0)
        inplace = InplaceReleasePrefix.wrap(stable,row_chunk=3)
        inplace.materialize(cache,anchors,ids)
        torch.testing.assert_close(anchors.codes,compiled.codes,atol=3e-6,rtol=3e-5)
        inplace.anchorize(cache,anchors,ids)
        torch.testing.assert_close(anchors.codes,zero,atol=3e-6,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires allocated GPU")
@torch.no_grad()
def test_h10_fused_conversion_and_full_catalogue_compilers(tmp_path):
    from types import SimpleNamespace
    from experiments.archive.system_study.fused_conversion import FusedReleasePrefix
    torch.manual_seed(21)
    origin = source(device="cuda")
    catalog = torch.randn(19,320,device="cuda")
    path = tmp_path / "origin.pt"
    metadata = build_bank(origin,catalog,path,source_bindings={"test":"synthetic_h10"})
    bank = load_bank(path,metadata["sha256"],device="cuda",dtype=torch.float32)
    adapter = UnifiedRelayAdapter.from_structured(origin,mode="A",read_shared=3,read_private=3,
        feature_mode="origin_encoder",feature_bank_path=path,feature_bank_sha256=metadata["sha256"])
    model = SimpleNamespace(item_emb=SimpleNamespace(weight=catalog),
        cfg=SimpleNamespace(num_items=18),lookup_item_embeddings=lambda ids:catalog[ids])
    cache = HSTUKVCache(torch.randn(2,2,67,320,device="cuda"),torch.randn(2,2,67,320,device="cuda"),67)
    ids = torch.randint(0,19,(2,67),device="cuda")
    prefix = None
    from hstu_kvcache.methods.cachetide.publication import compile_full_catalogue
    for _ in range(2):
        for layer in adapter.layers:
            for rotation in layer.rotation:
                rotation.normal_(std=.01)
            layer.item_update_bias.normal_(std=.01)
        reference,_ = compile_full_catalogue(adapter,model,prefix,chunk_size=7)
        actual,_ = compile_item_tables(adapter,model,prefix,chunk_size=7)
        for key in ("P","Q","D"):
            torch.testing.assert_close(getattr(actual,key),getattr(reference,key),atol=1e-6,rtol=1e-5)
        assert actual.D.shape == (2,19,72,1)
        stable = StableReleasePrefix.wrap(actual)
        fused = FusedReleasePrefix.wrap(stable,bank=bank)
        anchors = stable.zero_anchors(cache,ids)
        expected = stable.materialize(cache,anchors,ids)
        fused.materialize(cache,anchors,ids)
        torch.testing.assert_close(anchors.codes,expected.codes,atol=4e-6,rtol=3e-5)
        fused.anchorize(cache,anchors,ids)
        wanted = stable.anchorize(cache,expected,ids)
        torch.testing.assert_close(anchors.codes,wanted.codes,atol=4e-6,rtol=3e-5)
        prefix = actual

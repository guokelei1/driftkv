"""Full artifact uses real item IDs and resumes cold states across releases."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from code_structure_2026_10.structure import StructuredKVAdapter
from hstu_kvcache.models import HSTUKVCache
from query_relay_2026_10.adapter import ReleasePrefix, UnifiedRelayAdapter
from query_relay_2026_10.fixed_features import build_bank
from query_relay_2026_10.publish import compile_full_catalogue, load_prefix, save_publication


class ToyModel:
    def __init__(self, embeddings):
        self.cfg = SimpleNamespace(num_items=len(embeddings) - 1)
        self.item_emb = SimpleNamespace(weight=embeddings)

    def lookup_item_embeddings(self, items):
        return self.item_emb.weight[items]


@pytest.mark.parametrize("feature_mode", ["native", "origin_encoder"])
def test_full_artifact_three_versions_subset_and_cold_materialization(tmp_path, feature_mode):
    torch.manual_seed(20261015)
    source = StructuredKVAdapter(2, 6, 4, layout="shared12_private6").double().eval()
    with torch.no_grad():
        for layer in source.layers:
            layer.input_weight.normal_(std=.08)
            layer.input_bias.normal_(std=.02)
    catalogue = torch.randn(17, 24, dtype=torch.float64)
    kwargs = {}
    if feature_mode == "origin_encoder":
        bank = build_bank(source, catalogue, tmp_path / "bank.pt",
                          source_bindings={"model": "toy_origin"}, chunk_size=4)
        kwargs = dict(feature_mode=feature_mode, feature_bank_path=bank["path"],
                      feature_bank_sha256=bank["sha256"])
    origin = UnifiedRelayAdapter.from_structured(source, mode="B", read_shared=3,
        read_private=3, gate_normalization="rms", **kwargs)
    cache = HSTUKVCache(torch.randn(2, 1, 5, 24, dtype=torch.float64),
                       torch.randn(2, 1, 5, 24, dtype=torch.float64), 5)
    items = torch.tensor([[0, 2, 7, 12, 16]])  # Includes first and final raw model ID.
    identity = ReleasePrefix.identity(origin, len(catalogue))
    anchors = identity.zero_anchors(cache, items)
    sequential = identity.materialize(cache, anchors, items)
    previous, previous_metadata = None, None
    subset_ids = torch.tensor([0, 2, 7, 12, 16])
    mapped_positions = torch.arange(5)[None]
    subset_prefix = None
    last_path = None
    birth_anchors, born_sequential = None, None
    for target in (3, 4, 5):
        actor = copy.deepcopy(origin)
        with torch.no_grad():
            for layer in actor.layers:
                for rotation in layer.rotation:
                    rotation.normal_(std=.05)
                for native in layer.native_update:
                    native.normal_(std=.1)
                layer.item_update_weight.normal_(std=.03)
                layer.item_update_bias.normal_(std=.02)
        model = ToyModel(catalogue + target * .03)
        if last_path is not None:
            previous, previous_metadata = load_prefix(last_path, actor)
            assert all(a is b for a, b in zip(previous.layers, actor.layers))
        full, work = compile_full_catalogue(actor, model, previous, chunk_size=3)
        reference = actor.compile_release(previous, model.item_emb.weight)
        for actual, expected in zip((full.P, full.Q, full.D),
                                   (reference.P, reference.Q, reference.D), strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        sequential = actor.advance_codes(cache, sequential, model.item_emb.weight[items], items)
        # These anchors were never prepared at V3/V4: direct latest preparation
        # must equal the explicit update chain, including all original IDs.
        direct = full.materialize(cache, anchors, items)
        torch.testing.assert_close(direct.codes, sequential.codes, atol=3e-12, rtol=3e-12)
        if target == 3:
            birth_anchors = full.zero_anchors(cache, items)
            born_sequential = full.materialize(cache, birth_anchors, items)
            torch.testing.assert_close(born_sequential.codes, torch.zeros_like(born_sequential.codes),
                                       atol=2e-12, rtol=0)
        else:
            born_sequential = actor.advance_codes(cache, born_sequential,
                model.item_emb.weight[items], items)
            born_direct = full.materialize(cache, birth_anchors, items)
            torch.testing.assert_close(born_direct.codes, born_sequential.codes,
                                       atol=3e-12, rtol=3e-12)
        subset_prefix = actor.compile_release(subset_prefix, model.item_emb.weight[subset_ids])
        torch.testing.assert_close(full.D[:, subset_ids], subset_prefix.D, atol=3e-12, rtol=3e-12)
        mapped = subset_prefix.materialize(cache, anchors, items, table_item_ids=mapped_positions)
        torch.testing.assert_close(mapped.codes, direct.codes, atol=3e-12, rtol=3e-12)
        path = tmp_path / f"v{target}.pt"
        metadata = save_publication(path, full, actor, model, target_version=target,
            work=work, bindings={"toy": "no GPU/model fitting"}, previous_metadata=previous_metadata)
        assert metadata["catalogue_table_rows"] == 17
        assert metadata["raw_model_item_id_range_inclusive"] == [0, 16]
        assert metadata["producer_had_full_device_table"]
        assert not metadata["producer_had_full_cuda_table"]
        assert metadata["centralized_compile_then_four_copies"]["additional_publication_transfer_bytes"] \
            == 3 * full.storage_bytes()
        if target == 3:
            assert work["history_composition_flops"] == 0
        else:
            assert work["history_composition_flops"] > 0
        with pytest.raises(FileExistsError):
            save_publication(path, full, actor, model, target_version=target,
                work=work, bindings={})
        previous, last_path = full, path
    bad = copy.deepcopy(actor)
    if feature_mode == "origin_encoder":
        bad.feature_bank_sha256 = "different"
    else:
        with torch.no_grad():
            bad.layers[0].native_basis_0.add_(.01)
    with pytest.raises(ValueError, match="fixed features"):
        load_prefix(last_path, bad)

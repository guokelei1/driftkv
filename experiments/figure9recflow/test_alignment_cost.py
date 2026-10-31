"""CPU checks for independent channel layouts, actual partial fits and scope."""
from copy import deepcopy
import unittest

import torch
import torch.nn.functional as F

from hstu_kvcache.models.kv_cache import HSTUKVCache
from experiments.figure9recflow.alignment import (MapWeights, apply_map, apply_maps,
                                                 fit_cache_pairs, fit_one)
from experiments.figure9recflow.cost import RecFlowCostModel, fitting_flops


def cache(layers=2, batch=3, n=7, heads=2, dim=3):
    return HSTUKVCache(torch.randn(layers, batch, n, heads * dim),
                      torch.randn(layers, batch, n, heads * dim), n)


def random_weights(layers=2, heads=2, dim=3, width=5):
    c = 2 * layers * heads
    return MapWeights(layers, heads, dim, (
        torch.randn(c, 1, dim), torch.rand(c, 1, dim) + .5,
        torch.randn(c, dim, width), torch.randn(c, 1, width),
        torch.randn(c, width, dim), torch.randn(c, 1, dim),
        torch.randn(c, 1, dim), torch.rand(c, 1, dim) + .5))


def independent_map_reference(source, weights):
    """Direct loops/Linear; no implementation channel reshapes or bmm reuse."""
    outputs = []
    for kind, array in enumerate((source.k, source.v)):
        layers = []
        for layer in range(weights.layers):
            heads = []
            for head in range(weights.heads):
                index = (kind * weights.layers + layer) * weights.heads + head
                mx, sx, w1, b1, w2, b2, my, sy = [t[index] for t in weights.tensors]
                x = array[layer, :, :, head * weights.dim:(head + 1) * weights.dim]
                z = F.linear(F.silu(F.linear((x - mx) / sx, w1.T, b1[0])), w2.T, b2[0])
                heads.append(x + (z * sy + my))
            layers.append(torch.cat(heads, -1))
        outputs.append(torch.stack(layers))
    return HSTUKVCache(*outputs, source.seq_len)


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_channel_layout_and_method_batch(self):
        torch.manual_seed(40)
        state = cache()
        original = state.k.clone()
        weights = [random_weights(width=5), random_weights(width=7), random_weights(width=5)]
        actual = apply_maps(state, weights)
        for batch, w in enumerate(weights):
            single = HSTUKVCache(state.k[:, batch:batch + 1], state.v[:, batch:batch + 1], state.seq_len)
            expected = independent_map_reference(single, w)
            torch.testing.assert_close(actual.k[:, batch:batch + 1], expected.k, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(actual.v[:, batch:batch + 1], expected.v, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(state.k, original, rtol=0, atol=0)
        common = apply_map(state, weights[0])
        expected = independent_map_reference(state, weights[0])
        torch.testing.assert_close(common.k, expected.k, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(common.v, expected.v, rtol=2e-6, atol=2e-6)

    def test_identity_empty_and_wrong_shape(self):
        w = random_weights()
        tensors = list(w.tensors)
        for index in (4, 5, 6):
            tensors[index].zero_()
        w = MapWeights(w.layers, w.heads, w.dim, tuple(tensors))
        state = cache()
        torch.testing.assert_close(apply_map(state, w).k, state.k, rtol=0, atol=0)
        empty = cache(n=0)
        self.assertEqual(apply_map(empty, w).k.shape, empty.k.shape)
        with self.assertRaises(ValueError):
            apply_map(cache(layers=3), w)

    def test_actual_fit_learns_detached_teacher_and_matches_old_recipe(self):
        # The official implementation is independently imported, not wrapped by
        # the new implementation. This catches initialization/order/norm drift.
        from hstu_kvcache.methods.kv_map.max import fit_one as original_fit
        from experiments.figure9recflow.alignment import DEFAULT_SETTINGS
        torch.manual_seed(2)
        x = torch.randn(4, 29, 3)
        delta = .4 * x + torch.tensor([.3, -.7, .2])
        settings = dict(DEFAULT_SETTINGS, epochs=8, batch_rows=11, learning_rate=.01)
        new, profile = fit_one(x.clone(), delta.clone(), 5, settings, 17)
        old, old_profile = original_fit(x.clone(), delta.clone(), 5, settings, 17)
        for actual, expected in zip(new, old):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(profile["steps"], 24)
        self.assertEqual(profile["epoch_batch_sizes"], [11, 11, 7])
        self.assertLess(profile["normalized_teacher_probe_mse_after"],
                        profile["normalized_teacher_probe_mse_before"])
        self.assertEqual(profile["parameters"], old_profile["parameters"])
        ledger = fitting_flops(29, 5, epochs=8, batch_rows=11, channels=4, dim=3)
        self.assertEqual(ledger["steps"], profile["steps"])
        self.assertEqual(ledger["parameters"], sum(t.numel() for t in new[2:6]))

    def test_capture_once_preserves_input_and_prefix_scope(self):
        pairs = []
        for n in (3, 0, 5):
            old = cache(layers=1, batch=1, n=n, heads=1, dim=3)
            new = HSTUKVCache(old.k + .3, old.v - .2, n)
            pairs.append((old, new))
        original = deepcopy(pairs)
        seen = []
        def generate():
            for i, pair in enumerate(pairs):
                seen.append(i)
                yield pair
            raise AssertionError("Must not fetch an extra teacher pair")
        policies = [dict(tag="small", fit_users=1, width=3), dict(tag="large", fit_users=3, width=4)]
        result = fit_cache_pairs(generate(), policies, num_heads=1,
            settings=dict(epochs=2, batch_rows=4), teacher_flops=[100, 150, 400])
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(result["capture"]["cumulative_rows"], [3, 3, 8])
        self.assertEqual(result["profiles"]["small"]["observations"], 3)
        self.assertEqual(result["profiles"]["large"]["observations"], 8)
        self.assertEqual(result["profiles"]["large"]["flop_breakdown"]["parts"]["teacher_source_preparation"], 400)
        for pair, expected in zip(pairs, original):
            torch.testing.assert_close(pair[0].k, expected[0].k, rtol=0, atol=0)
            torch.testing.assert_close(pair[1].v, expected[1].v, rtol=0, atol=0)


class CostTests(unittest.TestCase):
    def test_layer_endpoint_and_all_selected_equal_rebuild(self):
        cm = RecFlowCostModel(hidden_size=6, num_layers=3, num_heads=2, item_embedding_dim=7)
        self.assertEqual(cm.layer(11, 0, 2), cm.full_cache(11))
        self.assertEqual(cm.selected(11, 20), cm.full_cache(11))
        self.assertEqual(cm.tail(11, 20), cm.full_cache(11))
        self.assertEqual(cm.layer(11, 1, 1), cm.kv_only(11))
        self.assertEqual(cm.operation_cost("deviation", 11, selected_tokens=11)["selection_flops"], 0)
        self.assertGreater(cm.operation_cost("deviation", 11, selected_tokens=2)["selection_flops"], 0)

    def test_tiled_padding_metadata_and_missing_output_change_cost(self):
        dense = RecFlowCostModel(attention_backend="torch")
        tiled = RecFlowCostModel(attention_backend="triton")
        self.assertEqual(dense.attention_pairs(3, 3), 9)
        self.assertEqual(tiled.attention_pairs(3, 3), 32 * 64)
        self.assertEqual(tiled.attention_pairs(33, 33), 2 * 32 * 64)
        small_item = RecFlowCostModel(item_embedding_dim=64)
        self.assertEqual(dense.features(5) - small_item.features(5), 5 * 2 * (320 - 64) * 320)
        full_output = RecFlowCostModel(skip_last_output=False)
        self.assertGreater(full_output.full_cache(32), dense.full_cache(32))
        self.assertNotEqual(tiled.selected(1024, 64), tiled.tail(1024, 64))

    def test_fifo_repeated_writes_are_not_one_big_suffix(self):
        cm = RecFlowCostModel(attention_backend="triton")
        # Two scalar appends at cap each read 1023 retained entries. Their
        # feature setup is shared once, unlike two separate Python calls.
        expected = 2 * cm.append(1023, 1) - 3 * cm.temporal_frequencies
        self.assertEqual(cm.rolling_append(1024, 2), expected)
        self.assertNotEqual(cm.rolling_append(1024, 2), cm.append(1024, 2))
        self.assertGreater(cm.read(1024, 129, chunk_size=128), cm.read(1024, 129, chunk_size=256))


if __name__ == "__main__":
    unittest.main()

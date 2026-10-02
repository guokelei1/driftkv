from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from read_correction_2026_09.cost import correction_forward as original_cost
from read_correction_2026_09.v2.cost import base_config, correction_forward, history_encoder_forward


def test_heavy_cost_counts_quadratic_history_and_frozen_base():
    config = dict(heads=6, head_dim=32, encoder_width=96, attention_heads=4,
                  base_width=32, base_token_chunk=256, base_query_chunk=4)
    cost = lambda n: history_encoder_forward(config, n)
    # Full history attention makes equal increments in N increasingly costly.
    assert cost(1024) - cost(768) > cost(768) - cost(512) > cost(512) - cost(256)
    assert correction_forward(config, 1024) == (
        original_cost(base_config(config), 1024) + cost(1024) + 192)
    assert history_encoder_forward(config, 1024, batch=3) == 3 * cost(1024)
    # A multi-query call shares only input projections; attention repeats per query.
    shared = 2 * cost(1024) - history_encoder_forward(config, 1024, queries=2)
    assert 0 < shared < cost(1024) / 2
    assert correction_forward(config, 1024) > 20 * original_cost(base_config(config), 1024)


def test_query_cost_unchanged_and_chunks_do_not_remove_work():
    query = dict(heads=10, head_dim=32)
    assert correction_forward(query, 1024) == original_cost(query, 1024)
    heavy = dict(**query, encoder_width=160, attention_heads=4,
                 base_width=32, base_token_chunk=256, base_query_chunk=4, query_chunk=1)
    assert correction_forward(heavy, 1024, queries=16) == correction_forward(
        {**heavy, "query_chunk": 4}, 1024, queries=16)

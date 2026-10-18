"""Small CPU checks for cost-relevant causal and rolling-window boundaries."""
import numpy as np

from cost import simulate_user


def main():
    # A short initial cache grows before evicting its inherited prefix.
    events = (np.array([0, 1, 11, 12, 13]), np.arange(5))
    counts = simulate_user(events, [], [("v3", 10)], 20, cap=4, band_cap=2)["v3"]
    assert counts["writer_prefix_pairs"] == 6
    assert counts["writer_valid_prefix_pairs"] == 5
    assert counts["final_coded_rows"] == 1
    assert counts["anchor_to_current_rows"] == 2

    # Compare valid inherited-prefix pairs with the actual inclusive mask,
    # across short/full caches and prefix sizes. Masked work remains charged.
    for length in range(1, 5):
        for coded in range(length + 1):
            for width in (1, 2, 4):
                query_positions = length + np.arange(width)
                key_positions = np.arange(length + width)
                mask = ((key_positions[None] <= query_positions[:, None])
                        & (key_positions[None] > query_positions[:, None] - 4))
                valid = int(mask[:, :coded].sum())
                excluded = np.maximum(0, length + np.arange(1, width + 1) - 4)
                assert valid == int(np.maximum(0, coded - excluded).sum())

    # A recommendation at the same timestamp precedes that behavior write.
    events = (np.array([0, 1, 11]), np.arange(3))
    requests = [{"version": "v3", "query_timestamp": 11}]
    counts = simulate_user(events, requests, [("v3", 10)], 20,
                           cap=4, band_cap=2)["v3"]
    assert counts["first_consumption_query"] == 1
    assert counts["ranking_prefix_pairs"] == 2
    assert counts["writer_queries"] == 1
    print("cost causal/prefix boundary checks passed")


if __name__ == "__main__":
    main()

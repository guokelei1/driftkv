from hstu_kvcache.serving import ThreeStateCachePolicy


def test_three_state_decision():
    policy = ThreeStateCachePolicy(reuse_thr=0.05, recompute_thr=0.20)
    assert policy.decide(1, 0.01).action == "reuse"
    assert policy.decide(2, 0.10).action == "migrate"
    assert policy.decide(3, 0.30).action == "recompute"


def test_batch_decision_stats():
    policy = ThreeStateCachePolicy(reuse_thr=0.05, recompute_thr=0.20)
    decisions = policy.decide_batch({i: v for i, v in enumerate([0.01, 0.1, 0.3, 0.5, 0.02])})
    stats = policy.decision_stats(decisions)
    assert abs(stats["frac_reuse"] - 0.4) < 0.01
    assert abs(stats["frac_migrate"] - 0.2) < 0.01
    assert abs(stats["frac_recompute"] - 0.4) < 0.01

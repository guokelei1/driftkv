from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheDecision:
    user_id: int
    action: str  # "reuse" | "migrate" | "recompute"
    drift_estimate: float
    threshold_reuse: float
    threshold_recompute: float


class ThreeStateCachePolicy:
    """Three-state KV decision driven by a drift estimate (roadmap Insight 6).

    reuse      : drift < reuse_thr      -> serve with stale KV (cost ~0)
    migrate    : reuse_thr <= drift < r -> apply a cheap correction (cost < recompute)
    recompute  : drift >= recompute_thr -> full F(theta_new, x) (cost = 1 forward)

    The policy itself is trivial engineering; the research value is entirely in
    making ``drift_estimate`` both accurate and cheap (drift/ module).
    """

    def __init__(self, reuse_thr: float = 0.05, recompute_thr: float = 0.20) -> None:
        self.reuse_thr = reuse_thr
        self.recompute_thr = recompute_thr

    def decide(self, user_id: int, drift_estimate: float) -> CacheDecision:
        if drift_estimate < self.reuse_thr:
            action = "reuse"
        elif drift_estimate < self.recompute_thr:
            action = "migrate"
        else:
            action = "recompute"
        return CacheDecision(
            user_id=user_id,
            action=action,
            drift_estimate=float(drift_estimate),
            threshold_reuse=self.reuse_thr,
            threshold_recompute=self.recompute_thr,
        )

    def decide_batch(self, drift_estimates: dict[int, float]) -> dict[int, CacheDecision]:
        return {uid: self.decide(uid, d) for uid, d in drift_estimates.items()}

    def decision_stats(self, decisions: dict[int, CacheDecision]) -> dict[str, float]:
        actions = [d.action for d in decisions.values()]
        n = len(actions) + 1e-12
        return {
            "frac_reuse": actions.count("reuse") / n,
            "frac_migrate": actions.count("migrate") / n,
            "frac_recompute": actions.count("recompute") / n,
        }

"""Request-macro ranking statistics on the full fixed user population.

Each bootstrap draw resamples all chosen users, including those with no
requests or no positive requests, and preserves every request of a sampled
user. The six metric means remain request-weighted, not user-weighted. Methods
share the same draw. Zero-positive requests are retained in coverage, while
their ranking metrics remain undefined. No candidates or model scores are read.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from experiments.figure9recflow.common import METRIC_NAMES


def _interval(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) >= 2 else None


def _draw_means(sums, counts, samples, seed):
    """Bounded-memory UID draws; matrix multiplication only over positive UIDs."""
    active = np.flatnonzero(counts)
    output = np.full((samples, sums.shape[1]), np.nan, dtype=np.float64)
    if not len(active) or not samples:
        return output, samples
    rng = np.random.default_rng(seed)
    n_users = len(counts)
    active_sums = sums[active]
    active_counts = counts[active]
    # At 30k UIDs this is only ~15MB, instead of a 1000x30k draw matrix.
    batch_size = 64
    empty = 0
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        weights = np.empty((stop - start, len(active)), dtype=np.float64)
        for row in range(stop - start):
            chosen = rng.integers(0, n_users, size=n_users)
            weights[row] = np.bincount(chosen, minlength=n_users)[active]
        denominator = weights @ active_counts
        valid = denominator > 0
        empty += int((~valid).sum())
        if valid.any():
            output[start:stop][valid] = (weights[valid] @ active_sums) / denominator[valid, None]
    return output, empty


def summarize(records: Iterable[dict], uids, policies, bootstrap: int = 1000,
              seed: int = 17021) -> dict:
    """Summarize one edge's flat per-request records, without dropping any UID.

    records contain uid/rid/ts/candidates/positives/metrics[policy,metric].
    Undefined zero-positive metrics may be NumPy NaNs or JSON nulls. Population
    order is kept exactly as supplied. Inputs may be streamed from user shards.
    The output is JSON-safe: every undefined statistic is None, never NaN.
    """
    population = np.asarray(uids)
    if population.ndim != 1 or not len(population) or len(np.unique(population)) != len(population):
        raise ValueError("Expected the distinct, nonempty fixed quality/cost UID population")
    if not np.issubdtype(population.dtype, np.integer):
        raise ValueError("UID population must contain integers")
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("Bootstrap replicate count must be a nonnegative integer")
    policy_rows = [dict(p) for p in policies]
    tags = [p["tag"] for p in policy_rows]
    if len(set(tags)) != len(tags) or "reuse" not in tags or "rebuild" not in tags:
        raise ValueError("Policies need distinct tags and exactly one Reuse and Rebuild")
    reuse, rebuild = tags.index("reuse"), tags.index("rebuild")
    n_methods, n_metrics = len(tags), len(METRIC_NAMES)
    uid_index = {int(uid): i for i, uid in enumerate(population)}
    sums = np.zeros((len(population), n_methods * n_metrics), np.float64)
    counts = np.zeros(len(population), np.int64)
    all_counts = np.zeros(len(population), np.int64)
    seen = set()
    total_candidates = total_positives = n_requests = 0
    for record in records:
        uid, rid = int(record["uid"]), int(record["rid"])
        if uid not in uid_index:
            raise ValueError(f"Request UID is outside the fixed population: {uid}")
        if (uid, rid) in seen:
            raise ValueError(f"Duplicate request: UID {uid}, RID {rid}")
        seen.add((uid, rid))
        if not isinstance(record["ts"], (int, np.integer)):
            raise ValueError("Request timestamp must be an integer")
        candidates, positives = int(record["candidates"]), int(record["positives"])
        if candidates <= 0 or not 0 <= positives <= candidates:
            raise ValueError("Invalid complete-request candidate/positive counts")
        values = np.asarray(record["metrics"], dtype=np.float64)
        if values.shape != (n_methods, n_metrics):
            raise ValueError(f"Expected metrics[{n_methods},{n_metrics}] in METRIC_NAMES order")
        user = uid_index[uid]
        if positives:
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1 + 1e-8)):
                raise ValueError("Positive requests require finite ranking metrics in [0,1]")
            sums[user] += values.ravel()
            counts[user] += 1
        elif not np.isnan(values).all():
            raise ValueError("Zero-positive request metrics must all be undefined, not zero")
        all_counts[user] += 1
        n_requests += 1
        total_candidates += candidates
        total_positives += positives
    n_positive = int(counts.sum())
    point = sums.sum(0).reshape(n_methods, n_metrics) / n_positive if n_positive else None
    draws, empty_draws = _draw_means(sums, counts, int(bootstrap), int(seed))
    draws = draws.reshape(bootstrap, n_methods, n_metrics)
    defined_draws = bootstrap - empty_draws
    reference_gaps, results = {}, []
    for j, metric in enumerate(METRIC_NAMES):
        gap = float(point[rebuild, j] - point[reuse, j]) if point is not None else None
        gap_draws = draws[:, rebuild, j] - draws[:, reuse, j]
        ci = _interval(gap_draws)
        reference_gaps[metric] = dict(
            rebuild=float(point[rebuild, j]) if point is not None else None,
            reuse=float(point[reuse, j]) if point is not None else None,
            difference=gap, ci95=ci,
            ci_includes_zero=bool(ci[0] <= 0 <= ci[1]) if ci is not None else None,
            negative_point_gap=bool(gap < 0) if gap is not None else None,
            bootstrap_gap_nonpositive_draws=int((gap_draws <= 0).sum()),
            bootstrap_gap_zero_draws=int((gap_draws == 0).sum()))
    for i, policy in enumerate(policy_rows):
        metrics = {}
        for j, metric in enumerate(METRIC_NAMES):
            ref = reference_gaps[metric]
            gap, ci = ref["difference"], ref["ci95"]
            value = float(point[i, j]) if point is not None else None
            gain = float(point[i, j] - point[reuse, j]) if point is not None else None
            recovery = 100 * gain / gap if gap is not None and gap != 0 else None
            gain_draws = draws[:, i, j] - draws[:, reuse, j]
            gap_draws = draws[:, rebuild, j] - draws[:, reuse, j]
            ratio_ci = None
            if point is None:
                status = "undefined_no_positive_requests"
            elif not bootstrap:
                status = "not_estimated"
            elif gap == 0 or ci is None or ci[0] <= 0 <= ci[1]:
                status = "unstable_denominator"
            elif empty_draws:
                status = "unstable_empty_request_draws"
            else:
                # Do not silently remove zero-denominator replicates. A ratio
                # percentile interval is withheld if any such draw exists.
                if np.any(gap_draws == 0):
                    status = "unstable_zero_denominator_draws"
                else:
                    ratio_ci = _interval(100 * gain_draws / gap_draws)
                    status = "stable_positive_gap" if gap > 0 else "stable_negative_gap"
            metrics[metric] = dict(value=value, ci95=_interval(draws[:, i, j]),
                gain_over_reuse=gain, gain_over_reuse_ci95=_interval(gain_draws),
                recovery_percent=recovery, recovery_percent_ci95=ratio_ci,
                recovery_ci_status=status)
        results.append(dict(**policy, metrics=metrics))
    limitations = [
        "Paired UID intervals condition on these fixed checkpoints, fitted maps, selected users, and day; they do not measure training-seed or new-day variance.",
        "Ranking metrics are request-macro over observed-positive requests. Zero-positive requests and inactive users remain in coverage and the bootstrap population.",
        "Negative or above-100% recovery is retained. A negative reference gap does not describe recovery of lost quality.",
    ]
    if empty_draws:
        limitations.append("Some UID draws contain no positive requests: reported absolute/gain CIs condition on nonempty draws; recovery CIs are withheld.")
    return dict(status="complete", metric_names=list(METRIC_NAMES),
        primary_metric="ndcg@10", secondary_metric="recall@10", quality_defined=bool(n_positive),
        population_uids=population.astype(np.int64).tolist(), population_users=len(population),
        request_active_users=int((all_counts > 0).sum()),
        request_inactive_users=int((all_counts == 0).sum()),
        positive_request_users=int((counts > 0).sum()),
        requests=n_requests, positive_requests=n_positive, zero_positive_requests=n_requests - n_positive,
        candidates=total_candidates, positive_candidates=total_positives,
        bootstrap=dict(samples=int(bootstrap), seed=int(seed), unit="fixed_population_uid",
            includes_inactive_users=True, shared_method_draws=True, defined_draws=int(defined_draws),
            empty_request_draws=int(empty_draws), estimand="request_macro",
            absolute_ci_conditional_on_nonempty_draws=bool(empty_draws)),
        reference_gaps=reference_gaps, results=results, limitations=limitations)


def aggregate_edges(edge_summaries, *, expected_edges=None) -> dict:
    """Equal-edge mean of per-edge recovery, never a ratio of averaged metrics.

    Each summary must have a unique `edge` (or `edge_id`) added by the caller.
    Costs, if supplied as result-row `cost_percent`, are also equal-edge means.
    Undefined edge values make the corresponding full mean None; they are not
    silently dropped. This descriptive aggregate has no invented across-edge CI.
    """
    edges = list(edge_summaries)
    if not edges:
        raise ValueError("At least one completed edge summary is required")
    ids = [e.get("edge", e.get("edge_id")) for e in edges]
    if any(x is None for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Completed edge summaries need unique edge identifiers")
    if expected_edges is not None and ids != list(expected_edges):
        raise ValueError("Completed edges do not match the entire declared edge list/order")
    tags = [r["tag"] for r in edges[0]["results"]]
    for edge in edges:
        if edge.get("status") != "complete" or edge["metric_names"] != list(METRIC_NAMES):
            raise ValueError("Edge is incomplete or has a different metric order")
        if [r["tag"] for r in edge["results"]] != tags:
            raise ValueError("Policy order differs across edges")
    frontier = []
    for i, tag in enumerate(tags):
        first = edges[0]["results"][i]
        row = {key: value for key, value in first.items() if key not in {"metrics", "cost_percent"}}
        row["metrics"] = {}
        for name in METRIC_NAMES:
            measured = [e["results"][i]["metrics"][name] for e in edges]
            values = [x["recovery_percent"] for x in measured]
            defined = all(x is not None for x in values)
            row["metrics"][name] = dict(
                value_by_edge=[x["value"] for x in measured],
                gain_over_reuse_by_edge=[x["gain_over_reuse"] for x in measured],
                recovery_percent_by_edge=values,
                mean_recovery_percent=float(np.mean(values)) if defined else None,
                undefined_recovery_edges=[edge for edge, value in zip(ids, values) if value is None],
                recovery_ci_status_by_edge=[x["recovery_ci_status"] for x in measured],
                mean_recovery_ci95=None)
        costs = [e["results"][i].get("cost_percent") for e in edges]
        row["cost_percent_by_edge"] = costs
        row["mean_cost_percent"] = float(np.mean(costs)) if all(x is not None for x in costs) else None
        frontier.append(row)
    return dict(status="complete_descriptive_aggregate", edge_ids=ids, edge_count=len(edges),
        aggregation="Equal-edge arithmetic mean of individual edge recovery percentages; no edge filtered by sign or confidence interval; not a ratio of mean metrics.",
        metric_names=list(METRIC_NAMES), edges=edges, frontier=frontier,
        limitations=["Edges share users and weights. The descriptive mean is not an independent-edge statistical confirmation.",
                     "Per-edge uncertainty is retained; no across-edge confidence interval is inferred from individual intervals.",
                     "Any undefined edge recovery makes its full aggregate undefined; no complete-case edge selection is performed."])

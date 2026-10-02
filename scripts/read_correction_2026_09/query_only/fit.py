"""Query-only fitting entry, separated from history-conditioned exploration."""
from hstu_kvcache.read_correction import fit_query


def fit(query, target_rate, *, ridge):
    return fit_query(query, target_rate, ridge=ridge)

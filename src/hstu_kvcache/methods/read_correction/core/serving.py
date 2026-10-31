"""Connect correction modules to the frozen model's original query reader."""

from hstu_kvcache.adaptation.reader import score

from .history_conditioned import HistoryCorrection
from .query_only import QueryCorrection


def build_correction(kind, config, state_dict=None):
    classes = {"query_only": QueryCorrection, "history_conditioned": HistoryCorrection}
    module = classes[kind](**config)
    if state_dict is not None:
        example = next(iter(state_dict.values()))
        module.to(device=example.device, dtype=example.dtype)
        module.load_state_dict(state_dict)
    return module


def make_override(modules, cache, counts, *, corrections=None):
    """None layers are native reads, useful for sequential layer calibration."""
    def override(layer, query, native_history):
        module = modules[layer]
        if module is None:
            if corrections is not None:
                corrections.append(native_history.new_zeros(native_history.shape))
            return native_history
        delta = module(query, cache.k[layer], cache.v[layer], counts)
        if corrections is not None:
            corrections.append(delta)
        return native_history + delta
    return override


def score_corrected(model, cache, candidates, query_delta, modules, counts, *, trace=False):
    """Return (logits, ReadResult), without changing K/V or model parameters.

    Trace ``history_heads`` contains native history reads at this corrected
    branch's queries; ``queries`` can therefore form same-query teacher targets.
    """
    if len(modules) != len(model.blocks):
        raise ValueError("one module or None is required per model layer")
    corrections = [] if trace else None
    logits, result = score(model, cache, candidates, query_delta, trace=trace,
                           history_override=make_override(modules, cache, counts, corrections=corrections))
    if trace:
        result.corrections = tuple(corrections)
    return logits, result

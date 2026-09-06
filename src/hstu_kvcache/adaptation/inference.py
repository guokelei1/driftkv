"""Optional compiled execution of the unchanged six-layer temporal prototype."""

import torch

from hstu_kvcache.models import HSTUKVCache

from .reader import full_response_rates, score

_ready = None
_source = None
_prefix = None
_calibration = None


def enable():
    global _ready, _source
    # Five producer counts and fixed candidate buckets in this experiment.
    torch._dynamo.config.recompile_limit = 64
    _ready = torch.compile(ready_logits, mode="reduce-overhead", dynamic=True, fullgraph=True)
    _source = torch.compile(source_logits, mode="reduce-overhead", dynamic=True, fullgraph=True)


def enabled():
    return _ready is not None


def exact_tensors(model, items, behaviors, deltas, lengths=None):
    cache = model.compute_kv(items, behaviors, deltas, lengths=lengths)
    return cache.k, cache.v


def enable_prefixes():
    global _prefix
    torch._dynamo.config.recompile_limit = 64
    _prefix = torch.compile(exact_tensors, mode="reduce-overhead", dynamic=True, fullgraph=True)


@torch.no_grad()
def compute_prefix(model, items, behaviors, deltas):
    if _prefix is None:
        return model.compute_kv(items, behaviors, deltas)
    torch.compiler.cudagraph_mark_step_begin()
    values = _prefix(model, items, behaviors, deltas)
    # A teacher or source snapshot can outlive later prefix builds.
    k, v = (value.clone() for value in values)
    return HSTUKVCache(k, v, items.shape[1])


def rate_tensors(model, k, v, teacher_k, teacher_v, candidates, delta, counts, response, temporal, time_groups):
    cache = HSTUKVCache(k, v, k.shape[2])
    teacher = HSTUKVCache(teacher_k, teacher_v, teacher_k.shape[2])
    return full_response_rates(model, cache, teacher, candidates, delta, counts, response, temporal, time_groups)


def enable_calibration():
    global _calibration
    torch._dynamo.config.recompile_limit = 64
    _calibration = torch.compile(rate_tensors, mode="reduce-overhead", dynamic=True, fullgraph=True)


@torch.no_grad()
def calibration_rates(model, cache, teacher, candidates, delta, counts, response, temporal, time_groups):
    if _calibration is None:
        return full_response_rates(model, cache, teacher, candidates, delta, counts, response, temporal, time_groups)
    torch.compiler.cudagraph_mark_step_begin()
    return _calibration(model, cache.k, cache.v, teacher.k, teacher.v, candidates, delta, counts,
                        response, temporal, time_groups).clone()


def ready_logits(model, k, v, candidates, delta, response=None, temporal=None):
    return score(model, HSTUKVCache(k, v, k.shape[2]), candidates, delta,
                 response_delta=response, response_time_delta=temporal)[0]


def source_logits(model, k, v, candidates, delta, sums, metadata, translator):
    counts = metadata[:, 1]
    count = counts.sum()
    owners = torch.nn.functional.one_hot(metadata[:, 0].long(), translator.producer_count).float().T
    values = owners @ sums / count
    masses = owners @ counts / count
    values = translator.source_values(values, masses)
    features = torch.cat((values, masses[:, None]), 1).flatten()
    features = torch.cat((features, metadata[0, 2:3].clamp(max=6144) / 1024))
    response = translator.rates(features)[None] * count
    temporal = translator.time_coefficients(features)[None] * count
    return ready_logits(model, k, v, candidates, delta, response, temporal), response, temporal


def writer_inputs(writer, age, zero):
    segments = list(writer.segments.values())
    assert len(segments) <= 8  # Six frozen producers and a 1024-event window.
    values = [torch.cat((s["sums"][0].flatten(),s["squared_sums"][0].flatten()))
              if writer.second_moments else s["sums"][0].flatten() for s in segments]
    sums = torch.stack(values + [zero] * (8 - len(segments)))
    metadata = [[s["producer"], s["count"][0], 0] for s in segments] + [[0, 0, 0]] * (8 - len(segments))
    metadata[0][2] = age
    # Avoid waiting for earlier device work while staging a few CPU integers.
    return sums, torch.tensor(metadata, dtype=zero.dtype).to(zero.device, non_blocking=True)


def candidate_parts(candidates):
    """Bound graph shapes for independent, transient candidates of one user."""
    assert candidates.shape[0] == 1
    for ids in candidates.split(16, dim=1):
        count = ids.shape[1]
        padded = 1 << (count - 1).bit_length()
        if padded > count:
            ids = torch.cat((ids, ids[:, :1].expand(-1, padded - count)), dim=1)
        # NumPy's [None] can have a zero batch stride even when contiguous.
        # Own compact input also removes guards on the larger sliced base.
        yield count, ids.clone(memory_format=torch.contiguous_format)


@torch.no_grad()
def score_ready(model, cache, candidates, delta, response=None, temporal=None):
    if _ready is None:
        return ready_logits(model, cache.k, cache.v, candidates, delta, response, temporal)
    # Rolling eviction leaves views whose layer stride depends on the latest
    # chunk size. Canonical query inputs avoid a graph for each such stride.
    k, v = cache.k.contiguous(), cache.v.contiguous()
    outputs = []
    for count, ids in candidate_parts(candidates):
        torch.compiler.cudagraph_mark_step_begin()
        # Own graph output before the next call; padded queries are discarded.
        outputs.append(_ready(model, k, v, ids, delta, response, temporal).clone()[:, :count])
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


@torch.no_grad()
def score_writer(model, cache, candidates, delta, writer, age, translator):
    sums, metadata = writer_inputs(writer, age, translator.source_padding)
    k, v = cache.k.contiguous(), cache.v.contiguous()
    outputs = []
    for count, ids in candidate_parts(candidates):
        torch.compiler.cudagraph_mark_step_begin()
        if not outputs:
            values = _source(model, k, v, ids, delta, sums, metadata, translator)
            # Published views outlive this call and must own their storage.
            logits, response, temporal = (value.clone() for value in values)
        else:
            logits = _ready(model, k, v, ids, delta, response, temporal).clone()
        outputs.append(logits[:, :count])
    logits = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
    return logits, response, temporal

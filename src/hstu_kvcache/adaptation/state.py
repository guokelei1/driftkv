"""Synchronous in-memory cache lifetime, including mixed producers and eviction."""

import torch

from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band, retain_latest_cache

from . import inference
from .reader import read_embedded, score
from .summary import SummaryWriter
from .translator import RidgeTranslator


class AdaptationState:
    def __init__(self, cache, timestamps, producer, *, segment_size=64, slots=2, write_correction=True, history_scope="old",
                 second_moments=False):
        self.cache = cache.detach()
        self.writer = SummaryWriter.from_cache(cache, timestamps, producer,
                                               segment_size=segment_size, slots=slots, second_moments=second_moments)
        self.target = producer
        self.translator = None
        self.source = self.translated = None
        self.response_delta = None
        self.response_time_delta = None
        self.response_query_delta = None
        self.covered = True
        self.write_correction = write_correction
        self.view_dirty = False
        self.history_scope = history_scope
        self.release_ordinal = self.writer.next_ordinal
        self.counts = dict(releases=0, refreshes=0, translated_segments=0,
                           appends=0, evictions=0, old_evictions=0, reuse_reads=0, cleared_reads=0)

    @property
    def writes_since_release(self):
        return self.writer.next_ordinal - self.release_ordinal

    @property
    def clearance_complete(self):
        return (not self.write_correction and isinstance(self.translator, RidgeTranslator)
                and self.translator.layer_clearance
                and self.writes_since_release >= self.translator.layers * 1024)

    def pack_source(self, target=None):
        target = self.target if target is None else target
        source = self.writer.pack(target, include_current=self.history_scope == "all")
        if source is not None and target == self.target:
            source.release_age = self.writer.next_ordinal - self.release_ordinal
        return source

    def response_mask(self, target):
        if self.history_scope == "all":
            return torch.ones(self.cache.seq_len, dtype=torch.bool, device=self.cache.k.device)
        return self.writer.old_mask(target)

    @torch.no_grad()
    def refresh(self, dirty_segment=None):
        self.view_dirty = False
        self.response_delta = None
        self.response_time_delta = None
        self.response_query_delta = None
        previous = self.translated
        if self.translator is None or self.clearance_complete:
            self.source = self.translated = None
            self.covered = True
            return
        self.source = self.pack_source()
        self.translated = None
        if self.source is None:
            self.covered = True
            return
        producers = {self.writer.segments[sid]["producer"] for sid in self.source.segment_ids}
        self.covered = self.translator is not None and producers <= self.translator.supported_producers
        if self.covered:
            if self.translator.context == "local" and previous is not None and dirty_segment is not None:
                previous_rows = {sid:i for i,sid in enumerate(previous.segment_ids)}
                updated = []
                translated_segments = 0
                for index, sid in enumerate(self.source.segment_ids):
                    if sid == dirty_segment:
                        updated.append(self.translator(self.source.select([index])).payload[0])
                        translated_segments += 1
                    else:
                        updated.append(previous.payload[previous_rows[sid]])
                self.translated = self.source.with_payload(torch.stack(updated))
            else:
                self.translated = self.translator(self.source)
                translated_segments = len(self.source.segment_ids)
            self.counts["refreshes"] += 1
            self.counts["translated_segments"] += translated_segments

    @torch.no_grad()
    def release(self, target, translator):
        if translator is None and self.write_correction:
            raise ValueError("the No-op release experiment requires native writes")
        self.writer.close_segment()
        self.target, self.translator = target, translator
        self.release_ordinal = self.writer.next_ordinal
        self.counts["releases"] += 1
        self.refresh()

    @torch.no_grad()
    def score(self, model, candidates, query_delta, *, use_compiled=True):
        if self.translator is None:
            # A declared No-op release still writes this model's real K/V and
            # maintains its summaries for the next release.
            self.view_dirty = False
            self.counts["reuse_reads"] += 1
            if use_compiled:
                return inference.score_ready(model, self.cache, candidates, query_delta)
            return score(model, self.cache, candidates, query_delta)[0]
        if self.clearance_complete:
            # Keep the real writer for future releases. Its current-model
            # cache has cleared all initial dependencies, so no view is needed.
            self.source = self.translated = self.response_delta = self.response_time_delta = None
            self.response_query_delta = None
            self.view_dirty = False
            self.covered = True
            self.counts["cleared_reads"] += 1
            if use_compiled:
                return inference.score_ready(model, self.cache, candidates, query_delta)
            return score(model, self.cache, candidates, query_delta)[0]
        if self.view_dirty:
            if isinstance(self.translator, RidgeTranslator) and not self.write_correction:
                if use_compiled and inference.enabled() and not getattr(self.translator, "query_affine", False):
                    producers = {segment["producer"] for segment in self.writer.segments.values()}
                    self.covered = producers <= self.translator.supported_producers
                    if self.covered:
                        logits, self.response_delta, self.response_time_delta = inference.score_writer(
                            model, self.cache, candidates, query_delta, self.writer,
                            self.writer.next_ordinal-self.release_ordinal, self.translator)
                        self.source = self.translated = None
                        self.view_dirty = False
                        self.counts["refreshes"] += 1
                        self.counts["translated_segments"] += len(self.writer.segments)
                        return logits
                refresh_many([self], self.translator)
            else:
                self.refresh()
        if not self.covered:
            self.counts["reuse_reads"] += 1
        if self.response_query_delta is not None:
            return score(model, self.cache, candidates, query_delta, response_delta=self.response_delta,
                         response_query_delta=self.response_query_delta)[0]
        if use_compiled and self.source is None and self.translated is None:
            return inference.score_ready(model, self.cache, candidates, query_delta,
                                         self.response_delta, self.response_time_delta)
        return score(model, self.cache, candidates, query_delta, self.source, self.translated,
                     response_delta=self.response_delta,response_time_delta=self.response_time_delta,
                     response_query_delta=self.response_query_delta)[0]

    @property
    def correction_active(self):
        return (self.covered and not self.clearance_complete
                and (self.translated is not None or self.response_delta is not None))

    @torch.no_grad()
    def append(self, model, item, behavior, delta, timestamp):
        if self.cache.seq_len >= model.cfg.max_seq_len:
            dirty_segment = self.writer.events[0][0]
            evicted_producer = self.writer.evict_first(self.cache)
            self.cache = retain_latest_cache(self.cache, model.cfg.max_seq_len - 1)
            self.counts["evictions"] += 1
            if evicted_producer != self.target:
                self.counts["old_evictions"] += 1
                if self.write_correction:
                    self.refresh(dirty_segment)  # Refresh before the append's query.
                else:
                    self.view_dirty = True
        x = model.embed_inputs(item, behavior, delta)
        # Candidate queries and observed writes have distinct token embeddings.
        # The query-only variant applies no CC-calibrated correction to writes;
        # its original statistics are still updated, and the next read refreshes.
        source, translated = (self.source,self.translated) if self.write_correction else (None,None)
        result = read_embedded(model, self.cache, x, source, translated)
        new = result.new_kv.detach()
        self.writer.add(new, [timestamp], self.target)
        self.cache = HSTUKVCache(torch.cat((self.cache.k, new.k), 2),
                                 torch.cat((self.cache.v, new.v), 2), self.cache.seq_len + 1)
        self.counts["appends"] += 1
        self.view_dirty |= self.history_scope == "all"

    @torch.no_grad()
    def append_native_chunk(self, model, items, behaviors, deltas, timestamps):
        """Replay actual native writes up to the next request boundary."""
        if self.write_correction:
            raise ValueError("chunk replay requires native writes without learned correction")
        length = items.shape[1]
        if not 0 < length <= model.cfg.max_seq_len:
            raise ValueError("replay chunk must fit in the retained cache")
        result = append_with_rolling_band(model, self.cache, items, behaviors, deltas, model.cfg.max_seq_len)
        self.install_native_chunk(result, timestamps)

    @torch.no_grad()
    def install_native_chunk(self, result, timestamps):
        """Maintain actual source statistics after the shared native replay kernel."""
        length = len(timestamps)
        evictions = self.cache.seq_len + length - result.seq_len
        removed = self.writer.evict_prefix(self.cache, evictions)
        old_evictions = sum(count for producer, count in removed.items() if producer != self.target)
        new = HSTUKVCache(result.k[:, :, -length:], result.v[:, :, -length:], length)
        self.writer.add(new, timestamps, self.target)
        self.cache = result
        self.counts["appends"] += length
        self.counts["evictions"] += evictions
        self.counts["old_evictions"] += old_evictions
        self.view_dirty |= old_evictions > 0 or self.history_scope == "all"


@torch.no_grad()
def release_many(states, target, translator):
    """Close source segments and install views on an actual native-write batch."""
    if any(state.write_correction for state in states):
        raise ValueError("batched publication requires native writes")
    for state in states:
        state.writer.close_segment()
        state.target,state.translator=target,translator
        state.release_ordinal=state.writer.next_ordinal
        state.counts["releases"]+=1
    refresh_many(states, translator)


@torch.no_grad()
def refresh_many(states, translator):
    """Fuse the same writer statistics into ready views, preserving release age."""
    if translator is None:
        for state in states:
            state.refresh()
        return
    active=[]
    for state in states:
        if state.clearance_complete:
            state.source=state.translated=state.response_delta=state.response_time_delta=None
            state.response_query_delta=None
            state.view_dirty=False
            state.covered=True
        else:
            active.append(state)
    states=active
    if not states:
        return
    segments=[]
    for state in states:
        state.view_dirty=False
        state.source=state.translated=state.response_delta=None
        state.response_time_delta=None
        state.response_query_delta=None
        selected=[seg for seg in state.writer.segments.values()
                  if translator.history_scope == "all" or seg["producer"] != state.target]
        state.covered={seg["producer"] for seg in selected} <= translator.supported_producers
        segments.append(len(selected))
    features,counts=translator.writer_features([state.writer for state in states],
        [state.writer.next_ordinal-state.release_ordinal for state in states])
    query=None
    if getattr(translator,"query_affine",False):
        values,query=translator.query_view(features)
        values=values*counts[:,None,None]
        query=query*counts[:,None,None,None,None]
    else:
        values=translator.rates(features)*counts[:,None,None]
    temporal=translator.time_coefficients(features)
    if temporal is not None:
        temporal=temporal*counts[:,None,None,None]
    for index,state in enumerate(states):
        if state.covered and segments[index]:
            state.response_delta=values[index:index+1]
            if temporal is not None:
                state.response_time_delta=temporal[index:index+1]
            if query is not None:
                state.response_query_delta=query[index:index+1]
            state.counts["refreshes"]+=1
            state.counts["translated_segments"]+=segments[index]

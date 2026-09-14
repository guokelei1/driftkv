"""Minimal native-write state with a real rebuild anchor, separate from release age."""

from hstu_kvcache.adaptation.state import AdaptationState
from hstu_kvcache.adaptation.summary import SummaryWriter


class LifecycleState(AdaptationState):
    def __init__(self, cache, timestamps, producer):
        super().__init__(cache, timestamps, producer, segment_size=1024, slots=1,
                         write_correction=False, history_scope="all")
        self.native_anchor = producer
        self.last_rebuild = None
        self.rebuilds = 0
        self.revision = 0
        self.prepared = None
        self.prepared_revision = None

    @property
    def anchored_native(self):
        return self.native_anchor == self.target

    def invalidate(self):
        self.revision += 1
        self.prepared = None
        self.prepared_revision = None

    def release(self, target, translator=None):
        super().release(target, translator)
        if self.native_anchor != target:
            self.native_anchor = None
        self.invalidate()

    def install_native_chunk(self, result, timestamps):
        super().install_native_chunk(result, timestamps)
        self.invalidate()

    def rebuild(self, cache, timestamps, timestamp):
        """Install actual Current KV and summary; never reset the release age."""
        age = self.writes_since_release
        self.cache = cache.detach()
        self.writer = SummaryWriter.from_cache(cache, timestamps, self.target,
                                              segment_size=1024, slots=1)
        self.release_ordinal = self.writer.next_ordinal - age
        self.native_anchor = self.target
        self.last_rebuild = dict(target=self.target, timestamp=int(timestamp))
        self.rebuilds += 1
        self.refresh()
        self.invalidate()

"""Bounded fixed-shape CUDA graphs for the measured synchronous bulk executor.

Graphs cache computation, never user state. Every use stages that user's native
K/V, codes and operation inputs; graph-owned outputs must be consumed/copied
before another replay. Model/version identity is part of the key.
"""
from collections import Counter
import time
import torch

from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from hstu_kvcache.methods.cachetide.compact.lightweight import LightweightValueState
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.cachetide.read import _validate, _mask_geometry
from hstu_kvcache.methods.cachetide.adapter import operation_gate_query
from experiments.archive.system_study.fused_query_projection_kernel import launch_read
from experiments.archive.system_study.budgeted_rebuild import native_history_read


class DynamicCountAdapter:
    def __init__(self, adapter, count):
        self.adapter, self.count = adapter, count

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def make_history_read(self, state, *, valid_mask=None, code_count=None, gate_query=None):
        # HAS_CODE is fixed for a graph, but its positive prefix length is data.
        has_code = bool(code_count)
        gates = self.prepare_gates(gate_query) if has_code else None
        def read(index, attention, q, k, v):
            codes, _ = _validate(self.layers[index], attention, q, k, v,
                                 state.codes[index], state.seq_len if has_code else 0)
            mask, strides = _mask_geometry(valid_mask, *q.shape[:3], k.shape[1], q.device)
            if not has_code:
                return launch_read(q,k,v,codes,q,q,q,q,mask,strides,code_count=0,scale=float(attention.scale))
            gk, gv = gates[index]
            dk, dv = self.decoders[index]
            return launch_read(q,k,v,codes,gk,gv,dk,dv,mask,strides,
                code_count=state.seq_len,scale=float(attention.scale),count_tensor=self.count)
        return read


class GraphCompute:
    """Only the predefined high-frequency shapes; others retain eager execution."""
    def __init__(self, *, batch_sizes=(1,8), append_widths=(1,16,32), model_versions=None,
                 share_history_inputs=False):
        self.entries, self.counters = {}, Counter()
        self.batch_sizes, self.append_widths = tuple(batch_sizes), tuple(append_widths)
        self.model_versions = {} if model_versions is None else model_versions
        self.capture_seconds = 0.
        self.preparation_seconds = Counter()
        self.graph_allocation_bytes = Counter()
        self.share_history_inputs = share_history_inputs
        self.history_inputs = {}

    def _history_tensor(self, kind, source):
        if not self.share_history_inputs:
            return source.clone()
        # Match clone's staging layout without allocating a temporary GPU copy.
        # A sliced, non-dense user cache becomes contiguous; dense layouts retain
        # their stride. Only this persistent staging buffer is shared.
        shape = tuple(source.shape)
        stride = tuple(torch.empty_like(source, device='meta').stride())
        key = (kind, shape, stride, source.dtype, source.device)
        if key not in self.history_inputs:
            self.history_inputs[key] = torch.empty_strided(shape, stride,
                dtype=source.dtype, device=source.device)
        return self.history_inputs[key]

    def supports(self, operation, cache, ids):
        if cache.seq_len != 1024 or ids.shape[0] not in self.batch_sizes:
            return False
        width = ids.shape[1]
        return ((operation == 'score' and width == 16) or
                (operation == 'append' and width in self.append_widths) or
                (operation == 'rebuild' and width == 1024))

    @torch.no_grad()
    def run(self, operation, model, adapter, cache, state, ids, delta, *, behavior=None, coded_rows=0):
        key = (operation,id(model),id(adapter),ids.shape[0],ids.shape[1],bool(coded_rows))
        entry = self.entries.get(key)
        if entry is None:
            entry = self._capture(operation,model,adapter,cache,state,ids,delta,behavior,coded_rows)
            self.entries[key] = entry
            self.counters['captures'] += 1
        else:
            self._stage(entry,cache,state,ids,delta,behavior,coded_rows)
        entry['graph'].replay()
        self.counters[operation+'_replays'] += 1
        return entry['output']

    @staticmethod
    def _stage(entry,cache,state,ids,delta,behavior,coded_rows):
        if entry['operation'] != 'rebuild':
            entry['cache'].k.copy_(cache.k); entry['cache'].v.copy_(cache.v)
        if entry['state'] is not None:
            entry['state'].codes.copy_(state.codes)
        entry['ids'].copy_(ids); entry['delta'].copy_(delta)
        if behavior is not None:
            entry['behavior'].copy_(behavior)
        if entry['count'] is not None:
            entry['count'].fill_(coded_rows)

    def _capture(self, operation, model, adapter, cache, state, ids, delta, behavior, coded_rows):
        from experiments.archive.system_study.batched_session import BatchedSessions
        torch.cuda.synchronize(ids.device)
        preparation_start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated(ids.device)
        owner = 'native' if adapter is None else 'relay'
        entry = dict(operation=operation, initial_coded_rows=coded_rows,
            cache=None if operation == 'rebuild' else HSTUKVCache(
                self._history_tensor('k',cache.k),self._history_tensor('v',cache.v),cache.seq_len),
            state=None if state is None else LightweightValueState(self._history_tensor('code',state.codes),state.seq_len),
            ids=ids.clone(),delta=delta.clone(),behavior=None if behavior is None else behavior.clone(),
            count=None if adapter is None else torch.tensor(coded_rows,device=ids.device,dtype=torch.int32))
        if self.share_history_inputs:
            # New graphs also need current contents: unlike clone(), an input
            # bank can still contain the preceding graph's unrelated user state.
            self._stage(entry,cache,state,ids,delta,behavior,coded_rows)
        view = None if adapter is None else DynamicCountAdapter(adapter,entry['count'])
        engine = BatchedSessions({})
        def compute():
            if operation == 'rebuild':
                return model.compute_kv(entry['ids'],entry['behavior'],entry['delta'])
            if operation == 'score':
                reader = native_history_read if view is None else view.make_history_read(entry['state'],
                    code_count=coded_rows,gate_query=operation_gate_query(view,model,entry['ids']))
                return score(model,entry['cache'],entry['ids'],entry['delta'],history_read_fn=reader)[0]
            with engine._corrected_append(model,view,entry['cache'],entry['state'],entry['ids'],
                    ids.shape[1],coded_rows,1024):
                return append_with_rolling_band(model,entry['cache'],entry['ids'],entry['behavior'],entry['delta'],1024)
        stream = torch.cuda.Stream(device=ids.device)
        stream.wait_stream(torch.cuda.current_stream(ids.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                compute()
        torch.cuda.current_stream(ids.device).wait_stream(stream)
        torch.cuda.synchronize(ids.device)
        graph = torch.cuda.CUDAGraph()
        began = time.perf_counter()
        with torch.cuda.graph(graph,stream=stream):
            output = compute()
        torch.cuda.synchronize(ids.device)
        self.capture_seconds += time.perf_counter()-began
        if isinstance(output,HSTUKVCache):
            # B=1 must also clone during scatter: this memory belongs to a graph.
            output.transient_graph_output = True
        entry.update(graph=graph,output=output)
        self.preparation_seconds[owner] += time.perf_counter()-preparation_start
        self.graph_allocation_bytes[owner] += torch.cuda.memory_allocated(ids.device)-allocated_before
        return entry

    def describe(self):
        # Captured graph pools include scratch in addition to explicit inputs.
        inputs = [t for e in self.entries.values() for t in
            [e['ids'],e['delta'],*([e['behavior']] if e['behavior'] is not None else []),
             *([e['cache'].k,e['cache'].v] if e['cache'] is not None else []),
             *([e['state'].codes] if e['state'] is not None else [])]]
        staged_bytes = sum(t.numel()*t.element_size() for t in inputs)
        counts = [e['count'] for e in self.entries.values() if e['count'] is not None]
        unique_storages = {(str(t.device),t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                           for t in inputs+counts}
        return dict(entries=len(self.entries),capture_seconds=self.capture_seconds,
            share_history_inputs=self.share_history_inputs,
            history_input_pool_count=sum(k[0]=='k' for k in self.history_inputs),
            shared_history_tensor_buffers=len(self.history_inputs), graph_private_pool_count=len(self.entries),
            graph_batch_sizes=list(self.batch_sizes),graph_append_widths=list(self.append_widths),
            graph_shapes=[dict(operation=k[0],model_object_id=k[1],adapter_object_id=k[2],
                version=self.model_versions.get(k[1]),initial_coded_rows=self.entries[k]['initial_coded_rows'],
                owner='native' if self.entries[k]['count'] is None else 'relay',
                batch=k[3],width=k[4],coded=k[5],history_length=1024,
                initial_prepare_model_calls=4,
                initial_prepare_executed_model_calls=3,
                capture_record_only_calls=1) for k in self.entries],
            preparation_seconds_by_method=dict(self.preparation_seconds),
            allocated_delta_bytes_by_method=dict(self.graph_allocation_bytes),
            staged_input_bytes=staged_bytes,
            logical_staged_input_bytes=staged_bytes+sum(t.numel()*t.element_size() for t in counts),
            unique_persistent_input_bytes=sum(unique_storages.values()), counters=dict(self.counters),
            scope='Allocation deltas include retained staging and graph/private-pool live tensors, not complete driver/reserved GPU memory. Preparation includes staging allocation, three warm/JIT executions and one record-only capture; initial_prepare_model_calls counts Python calls, not device executions. capture_seconds is an inner subset. Warm service excludes preparation. No user state may alias graph outputs.')

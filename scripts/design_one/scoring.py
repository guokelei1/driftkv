"""Personalized views on the same causal replay used by the Q/H controls."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from hstu_kvcache.adaptation.reader import score as corrected_score
from hstu_kvcache.design_one import ProducerSummary, SharedReadAdapter
from hstu_kvcache.design_one.adapter import CompatibilityView, LayerView
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.training import collate_foundation_batch
from read_correction_2026_09.cost import CostModel
from read_correction_v4.evaluate_full import HISTOGRAMS
from read_correction_v5.common import sha256
from selective_recompute_2026_09.evaluate import all_snapshots, prefix_events


SHARED_KIND = 'design_one_shared_read_v1'
RESPONSE_KIND = 'design_one_nonlinear_response_v1'
KV_KIND = 'design_one_kv_view_v1'
CONTEXT_KV_KIND = 'design_one_context_kv_view_v1'
AFFINE_KV_KIND = 'design_one_affine_kv_view_v1'
ITEM_KV_KIND = 'design_one_item_kv_view_v1'
PRODUCER_SCALED_ITEM_KV_KIND = 'design_one_producer_scaled_item_kv_view_v1'
ITEM_RESPONSE_KIND = 'design_one_item_response_v1'
CONTEXT_KV_KINDS = (CONTEXT_KV_KIND, AFFINE_KV_KIND)
ITEM_KV_KINDS = (ITEM_KV_KIND, PRODUCER_SCALED_ITEM_KV_KIND, ITEM_RESPONSE_KIND)
KV_KINDS = (KV_KIND, *CONTEXT_KV_KINDS, *ITEM_KV_KINDS)
POLICY_METHODS = {SHARED_KIND: 'personalized_read', RESPONSE_KIND: 'nonlinear_response',
                  KV_KIND: 'kv_read_view', CONTEXT_KV_KIND: 'context_kv_read_view',
                  AFFINE_KV_KIND: 'affine_kv_read_view', ITEM_KV_KIND: 'item_kv_read_view',
                  PRODUCER_SCALED_ITEM_KV_KIND: 'producer_scaled_item_kv_read_view',
                  ITEM_RESPONSE_KIND: 'item_response_read_view'}
STATE_COSTS = ('summary_initial_flops', 'summary_update_flops', 'view_flops',
               'view_initial_flops', 'view_update_flops', 'context_update_flops')
STATE_MEMORY = ('persistent_extra_peak_bytes', 'native_kv_peak_bytes',
                'persistent_extra_to_native_ratio_peak', 'shared_projection_bytes')


def policy_kind(policy):
    return policy.get('kind', policy.get('variant', SHARED_KIND))


def compact_mode(policy):
    return policy.get('execution_mode', 'mapped_kv') == 'compact_hidden'


def load_policies(entries, device):
    policies = []
    for entry in entries:
        folder = Path(entry['calibration_dir'])
        calibration = json.loads((folder / 'calibration.json').read_text())
        saved = torch.load(folder / 'calibration.pt', map_location='cpu', weights_only=False)
        kind = saved['kind']
        if kind != entry['variant'] or kind not in POLICY_METHODS:
            raise ValueError('Design 1 adapter kind differs from its calibration binding')
        if kind == SHARED_KIND:
            adapter = SharedReadAdapter.from_state_dict(saved['adapter']).to(device)
        else:
            if kind == AFFINE_KV_KIND:
                from hstu_kvcache.design_one.affine import AffineKVReadViewAdapter
                cls = AffineKVReadViewAdapter
            elif kind == ITEM_KV_KIND:
                from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
                cls = ItemKVReadViewAdapter
            elif kind == PRODUCER_SCALED_ITEM_KV_KIND:
                from hstu_kvcache.design_one.producer_scaled_item import ProducerScaledItemKVAdapter
                cls = ProducerScaledItemKVAdapter
            elif kind == ITEM_RESPONSE_KIND:
                from hstu_kvcache.design_one.composite import FrozenItemResponseAdapter
                cls = FrozenItemResponseAdapter
            else:
                from hstu_kvcache.design_one.nonlinear import (
                    NonlinearResponseAdapter, KVReadViewAdapter, ContextKVReadViewAdapter)
                cls = {RESPONSE_KIND: NonlinearResponseAdapter, KV_KIND: KVReadViewAdapter,
                       CONTEXT_KV_KIND: ContextKVReadViewAdapter}[kind]
            adapter = cls.from_state_dict(saved['adapter']).to(device).eval().requires_grad_(False)
            if kind == CONTEXT_KV_KIND:
                width = adapter.heads * adapter.head_dim
                if any(layer.input.in_features != 2 * width + 2
                       or layer.input_center.numel() != 2 * width + 2
                       or layer.input_scale.numel() != 2 * width + 2
                       or layer.output.out_features != 2 * width
                       or layer.output_scale.numel() != 2 * width for layer in adapter.layers):
                    raise ValueError('context KV map normalization differs from its two-field input')
            if kind == AFFINE_KV_KIND:
                width = adapter.heads * adapter.head_dim
                inputs, outputs = 2 * width + 2, 2 * width
                if any(layer.input_center.shape != (inputs,)
                       or layer.input_scale.shape != (inputs,)
                       or layer.map_weight.shape != (inputs, outputs)
                       or layer.map_bias.shape != (outputs,) for layer in adapter.layers):
                    raise ValueError('affine KV map dimensions differ from its two-field input')
            if kind in CONTEXT_KV_KINDS:
                width = adapter.heads * adapter.head_dim
                if (calibration.get('adapter_config') != adapter.get_config()
                        or calibration.get('input_width') != 2 * width + 2
                        or calibration.get('context_features') != ['is_parent', 'old_fraction_at_write']):
                    raise ValueError('context KV metadata differs from the stored adapter inputs')
            if kind in ITEM_KV_KINDS:
                width = adapter.heads * adapter.head_dim
                if any(layer.input.in_features != 3 * width
                       or layer.input_center.numel() != 3 * width
                       or layer.input_scale.numel() != 3 * width
                       or layer.output.out_features != 2 * width
                       or layer.output_scale.numel() != 2 * width for layer in adapter.layers):
                    raise ValueError('item KV normalization differs from its current-embedding input')
                source = calibration.get('item_feature_source', {})
                if (calibration.get('adapter_config') != adapter.get_config()
                        or calibration.get('input_width') != 3 * width
                        or source.get('model') != 'Current V5' or source.get('parameter') != 'item_emb'
                        or source.get('checkpoint_sha256') != calibration['checkpoint_hashes']['v5']):
                    raise ValueError('item KV metadata differs from the stored adapter inputs')
            if kind == PRODUCER_SCALED_ITEM_KV_KIND:
                if (adapter.hidden_width != 64
                        or any(layer.native_scale.shape != torch.Size([])
                               or not bool(torch.isfinite(layer.native_scale)) for layer in adapter.layers)):
                    raise ValueError('producer-scaled item KV requires width64 and finite per-layer scalars')
                values = [float(layer.native_scale) for layer in adapter.layers]
                if calibration.get('producer_scale_refinement', {}).get('final_alpha') != values:
                    raise ValueError('producer-scale metadata differs from the stored layer scalars')
            if kind == ITEM_RESPONSE_KIND:
                width = adapter.heads * adapter.head_dim
                if (adapter.item.hidden_width != 64 or adapter.response.hidden_width != 256
                        or any(layer.input.in_features != 2 * width
                               or layer.input_center.numel() != 2 * width
                               or layer.input_scale.numel() != 2 * width
                               or layer.output.out_features != width
                               or layer.output_scale.numel() != width for layer in adapter.response.layers)):
                    raise ValueError('item-response dimensions differ from Item64 plus Response256')
                binding = calibration['initial_item_calibration']
                source = Path(binding['path'])
                if (sha256(source / 'calibration.json') != binding['metadata_sha256']
                        or sha256(source / 'calibration.pt') != binding['weights_sha256']):
                    raise ValueError('initial Item calibration differs from its composite binding')
                warm = torch.load(source / 'calibration.pt', map_location='cpu', weights_only=False)
                expected, actual = warm['adapter'], saved['adapter']['item']
                if (warm['kind'] != ITEM_KV_KIND or expected['config'] != actual['config']
                        or expected['state_dict'].keys() != actual['state_dict'].keys()
                        or any(not torch.equal(value, actual['state_dict'][key])
                               for key, value in expected['state_dict'].items())):
                    raise ValueError('composite Item network differs from its frozen checkpoint')
        mode = entry.get('execution_mode', 'mapped_kv')
        if mode not in ('mapped_kv', 'compact_hidden') or (mode == 'compact_hidden' and kind != ITEM_RESPONSE_KIND):
            raise ValueError('compact_hidden execution requires the Item+Response artifact')
        if mode == 'compact_hidden':
            from hstu_kvcache.design_one.compact import CompactItemResponseAdapter
            adapter = CompactItemResponseAdapter(adapter).eval().requires_grad_(False)
        policies.append({**entry, 'kind': kind, 'adapter': adapter,
                         'calibration': calibration})
    return policies


class ViewObserver:
    """Maintain read-only views alongside each cohort's actual native state.

    Event appends never receive a correction. The observer only reads entering
    and departing native K/V and installs a read override for candidate scoring.
    Repeated candidate owners select the same cached b/A without regeneration.
    """
    def __init__(self, policies, current_model=None):
        self.policies = policies
        self.summary_policies = [p for p in policies if policy_kind(p) == SHARED_KIND]
        self.kv_policies = [p for p in policies if policy_kind(p) in KV_KINDS and not compact_mode(p)]
        self.compact_policies = [p for p in policies if compact_mode(p)]
        self.context_policies = [p for p in policies if policy_kind(p) in CONTEXT_KV_KINDS]
        self.item_policies = [p for p in policies if policy_kind(p) in ITEM_KV_KINDS]
        self.needs_raw_events = bool(self.item_policies)
        self.current_model = current_model if self.needs_raw_events else None
        if self.needs_raw_events and current_model is None:
            raise ValueError('item KV views require the frozen Current model')
        if any(policy_kind(p) not in POLICY_METHODS for p in policies):
            raise ValueError('unknown Design 1 policy kind')
        if any(p.get('producer_scope', 'all') not in ('all', 'parent_only') for p in self.kv_policies):
            raise ValueError('producer_scope must be all or parent_only')
        if any(p.get('producer_scope', 'all') != 'all' for p in self.context_policies):
            raise ValueError('context KV policies require producer_scope=all')
        if any(p.get('producer_scope', 'all') != 'all' for p in self.item_policies):
            raise ValueError('item KV policies require producer_scope=all')
        self.costs = {p['tag']: Counter() for p in policies}
        self.item_lookup_bytes = Counter()
        self.memory = {p['tag']: Counter() for p in policies}
        self.maximum_state_bytes = 0

    def _item_features(self, raw_events):
        if not self.needs_raw_events:
            return None
        if raw_events is None:
            raise ValueError('item KV views require the actual native event item IDs')
        features = self.current_model.lookup_item_embeddings(raw_events[0])
        for policy in self.item_policies:
            self.item_lookup_bytes[policy['tag']] += features.numel() * features.element_size()
        return features

    def initialized(self, cache, raw_events=None):
        self.batch = cache.k.shape[1]
        self.layers, _, _, self.width = cache.k.shape
        self.seq_len = cache.seq_len
        self.device = cache.k.device
        self.dirty = set(range(self.batch))
        self.views, self.kv_views, self.compact_views = {}, {}, {}
        # Scalar size only: never retain a duplicate native cache reference.
        self.native_bytes_per_token = (self.layers * self.batch * self.width
                                      * (cache.k.element_size() + cache.v.element_size()))
        # Pure parent rows start at [is_parent, old_fraction_at_write]=[1,1].
        # Constant allocation/copies have zero arithmetic FLOPs; storage counts.
        self.write_context = (torch.ones((self.batch, cache.seq_len, 2), device=self.device)
                              if self.context_policies else None)
        self.summary = None
        if self.summary_policies:
            first = self.summary_policies[0]['adapter']
            self.summary = ProducerSummary.from_cache(cache, 4, producer_ids=first.producer_ids,
                                                      max_length=first.max_length)
        for policy in self.summary_policies:
            adapter = policy['adapter']
            layers = []
            for params, (response, _) in zip(adapter.parameters, adapter._response_maps, strict=True):
                _, heads, augmented, width = params['weights'].shape
                layers.append(LayerView(cache.k.new_zeros((self.batch, heads, width)),
                    cache.k.new_zeros((self.batch, heads, augmented - 1, width)), response))
            self.views[policy['tag']] = CompatibilityView(tuple(layers))
            self.costs[policy['tag']]['summary_initial_flops'] += self.summary.estimate_flops(
                'scan', length=cache.seq_len)
        item_features = self._item_features(raw_events)
        for policy in self.kv_policies:
            tag, adapter = policy['tag'], policy['adapter']
            kwargs = {'context': self.write_context} if policy_kind(policy) in CONTEXT_KV_KINDS else {}
            if policy_kind(policy) in ITEM_KV_KINDS:
                kwargs = {'item_features': item_features}
            self.kv_views[tag] = adapter.map_cache(cache, **kwargs)
            self.costs[tag]['view_initial_flops'] += adapter.estimate_flops(
                batch=self.batch, tokens=cache.seq_len)['token_transform']
        for policy in self.compact_policies:
            tag, adapter = policy['tag'], policy['adapter']
            self.compact_views[tag] = adapter.encode_cache(cache, item_features)
            self.costs[tag]['view_initial_flops'] += adapter.estimate_flops(
                batch=self.batch, tokens=cache.seq_len)['token_transform']
        self._record_state_bytes()

    def _record_state_bytes(self):
        size = 0 if self.summary is None else self.summary.storage_bytes()
        size += sum(view.storage_bytes() for view in self.views.values())
        size += sum((view.k.numel() * view.k.element_size()
                     + view.v.numel() * view.v.element_size()) for view in self.kv_views.values())
        size += sum(view.storage_bytes() for view in self.compact_views.values())
        if self.write_context is not None:
            size += self.write_context.numel() * self.write_context.element_size()
        self.maximum_state_bytes = max(self.maximum_state_bytes, size)
        native_bytes = self.seq_len * self.native_bytes_per_token
        for policy in self.policies:
            if policy_kind(policy) != ITEM_RESPONSE_KIND:
                continue
            tag = policy['tag']
            if compact_mode(policy):
                extra_bytes = self.compact_views[tag].storage_bytes()
                self.memory[tag]['shared_projection_bytes'] = policy['adapter'].projection_storage_bytes()
            else:
                view = self.kv_views[tag]
                extra_bytes = view.k.numel() * view.k.element_size() + view.v.numel() * view.v.element_size()
            values = dict(persistent_extra_peak_bytes=extra_bytes, native_kv_peak_bytes=native_bytes,
                          persistent_extra_to_native_ratio_peak=extra_bytes / native_bytes if native_bytes else 0.)
            for key, value in values.items():
                self.memory[tag][key] = max(self.memory[tag][key], value)

    def appended(self, indices, old, new, width, raw_events=None):
        removed = old.seq_len + width - new.seq_len
        appended = HSTUKVCache(new.k[:, :, -width:], new.v[:, :, -width:], width)
        updated_context = None
        if self.write_context is not None:
            from hstu_kvcache.design_one.nonlinear import append_context, append_context_flops
            cap = self.context_policies[0]['adapter'].max_length
            updated_context = append_context(self.write_context[indices], old.seq_len, width, cap)
            if self.write_context.shape[1] != new.seq_len:
                if len(indices) != self.batch:
                    raise RuntimeError('growing history must update its entire singleton cohort')
                self.write_context = updated_context
            else:
                self.write_context[indices] = updated_context
            for policy in self.context_policies:
                self.costs[policy['tag']]['context_update_flops'] += append_context_flops(
                    batch=len(indices), old_length=old.seq_len, width=width, max_length=cap)
        if self.summary is not None:
            selected = self.summary.select(indices)
            evict_cost = selected.estimate_flops('evict', length=removed)
            add_cost = selected.estimate_flops('add', length=width)
            selected.evict_prefix(old, removed)
            selected.add(appended, 5)
            if len(indices) == self.batch and selected.total_count != self.summary.total_count:
                # Short histories are singleton cohorts and grow until the cap.
                self.summary = selected
            else:
                self.summary.put(indices, selected)
            for policy in self.summary_policies:
                self.costs[policy['tag']]['summary_update_flops'] += evict_cost + add_cost
        item_features = self._item_features(raw_events)
        for policy in self.kv_policies:
            tag, adapter = policy['tag'], policy['adapter']
            view = self.kv_views[tag]
            # Initial rows are V4; every real append is native V5. The scope
            # changes only this read view, never the persistent native writes.
            transform = policy.get('producer_scope', 'all') == 'all'
            kwargs = ({'context': updated_context[:, -width:]}
                      if policy_kind(policy) in CONTEXT_KV_KINDS else {})
            if policy_kind(policy) in ITEM_KV_KINDS:
                kwargs = {'item_features': item_features}
            scale_cost = {}
            if policy_kind(policy) == PRODUCER_SCALED_ITEM_KV_KIND:
                # These are only the real newly written V5 rows. The mask is
                # temporary; retained mapped entries already contain their scale.
                kwargs['producer_mask'] = torch.ones((len(indices), width), dtype=torch.bool,
                                                     device=appended.k.device)
                scale_cost['current_rows'] = len(indices) * width
            entering = adapter.map_cache(appended, **kwargs) if transform else appended
            k = torch.cat((view.k[:, indices, removed:], entering.k), dim=2)
            v = torch.cat((view.v[:, indices, removed:], entering.v), dim=2)
            if view.seq_len != new.seq_len:
                if len(indices) != self.batch:
                    raise RuntimeError('growing history must update its entire singleton cohort')
                self.kv_views[tag] = HSTUKVCache(k, v, new.seq_len)
            else:
                view.k[:, indices], view.v[:, indices] = k, v
            if transform:
                self.costs[tag]['view_update_flops'] += adapter.estimate_flops(
                    batch=len(indices), tokens=width, **scale_cost)['token_transform']
        for policy in self.compact_policies:
            tag, adapter = policy['tag'], policy['adapter']
            view = self.compact_views[tag]
            entering = adapter.encode_cache(appended, item_features)
            hidden = torch.cat((view.hidden[:, indices, removed:], entering.hidden), dim=2)
            if view.seq_len != new.seq_len:
                if len(indices) != self.batch:
                    raise RuntimeError('growing history must update its entire singleton cohort')
                self.compact_views[tag] = type(view)(hidden, new.seq_len)
            else:
                view.hidden[:, indices] = hidden
            self.costs[tag]['view_update_flops'] += adapter.estimate_flops(
                batch=len(indices), tokens=width)['token_transform']
        self.seq_len = new.seq_len
        self.dirty.update(indices)
        self._record_state_bytes()

    def reading(self, owners):
        changed = sorted(set(owners).intersection(self.dirty))
        if changed and self.summary is not None:
            indices = torch.tensor(changed, dtype=torch.long, device=self.device)
            subset = self.summary.select(indices)
            counts = subset.batch_counts()
            features_by_mode = {}
            for policy in self.summary_policies:
                tag, adapter = policy['tag'], policy['adapter']
                if adapter.summary_mode not in features_by_mode:
                    features_by_mode[adapter.summary_mode] = subset.features(mode=adapter.summary_mode)
                features = features_by_mode[adapter.summary_mode]
                prepared = adapter.prepare_features(features, counts)
                self.views[tag].put(indices, prepared)
                estimate = adapter.estimate_flops(batch=len(changed))
                self.costs[tag]['view_flops'] += (estimate['summary_projection']
                    + estimate['view_generation'] + subset.estimate_flops('features'))
        self.dirty.difference_update(changed)
        indices = torch.tensor(owners, dtype=torch.long, device=self.device)
        context = {tag: view.select(indices) for tag, view in self.views.items()}
        for tag, view in self.kv_views.items():
            context[tag] = HSTUKVCache(view.k[:, indices], view.v[:, indices], view.seq_len)
        for tag, view in self.compact_views.items():
            context[tag] = view.select(indices)
        for policy in self.policies:
            if policy_kind(policy) == RESPONSE_KIND:
                counts = torch.full((len(owners),), self.seq_len, device=self.device)
                context[policy['tag']] = policy['adapter'].make_history_override(counts)
        return context


@torch.inference_mode()
def score_unit(current, parent, history, by_user, uids, cutover, policies, cfg, scale, device, *, verify):
    records = {p['tag']: [] for p in policies}
    stats = {name: Counter() for name in HISTOGRAMS}
    controls = {'requests': 0, 'reuse_max_abs_logit_error': 0.,
                'full_max_abs_logit_error': 0., 'identity_max_abs_logit_error': 0.}
    full = [u for u in uids if len(prefix_events(history.rows[u], cutover,
                                                cfg['history_length'])) == cfg['history_length']]
    full_set, size = set(full), cfg['cohort_sizes'][scale]
    cohorts = [full[i:i + size] for i in range(0, len(full), size)] + [[u] for u in uids if u not in full_set]
    observer = ViewObserver(policies, current_model=current)
    cost = CostModel.for_scale(scale, cfg['attention_backend'])
    for snap, _ in all_snapshots(cohorts, by_user, history, parent, current, cutover,
        cfg['query_batches'][scale], stats, cost, cfg['append_band_size'], observer=observer):
        cache, requests = snap.state.cache, snap.requests
        n = cache.seq_len
        if verify:
            native = current.observe_cc_reuse(cache, snap.candidates, snap.query_deltas)[0][:, 0]
            identity = corrected_score(current, cache, snap.candidates, snap.query_deltas)[0][:, 0]
            batch = collate_foundation_batch([{**r, 'weight': r.get('weight', 1.)} for r in requests],
                history, device=device, max_history=cfg['history_length'])
            actual_full = current.observe_cc_full(batch.item_ids, batch.behaviors, batch.time_deltas,
                batch.candidate_ids, batch.query_time_deltas, lengths=batch.lengths)[0][:, 0]
            for key, actual, expected in (
                ('reuse_max_abs_logit_error', native, native.new_tensor([r['reuse_logit'] for r in requests])),
                ('full_max_abs_logit_error', actual_full, native.new_tensor([r['full_logit'] for r in requests])),
                ('identity_max_abs_logit_error', identity, native)):
                controls[key] = max(controls[key], float((actual - expected).abs().max()))
            controls['requests'] += len(requests)
        for policy in policies:
            tag, adapter = policy['tag'], policy['adapter']
            override = snap.context[tag]
            extra_history_read_flops = response_flops = 0
            if policy_kind(policy) in KV_KINDS:
                override = (adapter.make_history_override(current, cache, override) if compact_mode(policy)
                            else adapter.make_history_override(current, override))
                # The common reader still computes the native history read.
                # Charge the complete second read; no subtraction or fusion.
                estimate = adapter.estimate_flops(tokens=n)
                read_flops = extra_history_read_flops = estimate['extra_history_read']
                if policy_kind(policy) == ITEM_RESPONSE_KIND:
                    response_flops = estimate['candidate_reads']
                    read_flops += response_flops
            else:
                read_flops = adapter.estimate_flops()['candidate_reads']
            logits = corrected_score(current, cache, snap.candidates, snap.query_deltas,
                                     history_override=override)[0][:, 0]
            values = logits.float().cpu().tolist()
            if not np.isfinite(values).all():
                raise RuntimeError('nonfinite Design 1 predictions')
            for row, value in zip(requests, values, strict=True):
                records[tag].append({key: row[key] for key in (
                    'request_id', 'uid', 'label', 'query_timestamp', 'full_logit', 'reuse_logit')}
                    | {'hstu_logit': value, 'correction_flops': read_flops, 'read_flops': read_flops,
                       **dict.fromkeys(STATE_COSTS, 0),
                       **dict.fromkeys(STATE_MEMORY, 0), 'execution_setup_flops': 0,
                       'extra_history_read_flops': extra_history_read_flops,
                       'response_flops': response_flops,
                       'item_lookup_bytes': 0,
                       'history_length': n, 'inherited_count': max(0, n - int(row['append_count_since_cutover'])),
                       'append_count_since_cutover': row['append_count_since_cutover']})
    if any(v > 2e-5 for k, v in controls.items() if k.endswith('error')):
        raise RuntimeError(f'Design 1 native controls differ: {controls}')
    for policy in policies:
        # State work belongs to the unit, not a particular request. Store its
        # complete ledger once so request-table sums include all lifecycle work.
        first = records[policy['tag']][0]
        first.update(observer.memory[policy['tag']])
        for key, value in observer.costs[policy['tag']].items():
            first[key] = int(value)
            first['correction_flops'] += int(value)
        # Lookup is existing-model table traffic, not arithmetic or a copied
        # adapter table. Item features are temporary locals, never view state.
        if policy_kind(policy) in ITEM_KV_KINDS:
            first['item_lookup_bytes'] = int(observer.item_lookup_bytes[policy['tag']])
    return records, {name: {str(k): int(v) for k, v in values.items()}
                     for name, values in stats.items()}, controls

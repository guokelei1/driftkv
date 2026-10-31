#!/usr/bin/env python3
"""Correct small-cap preparation denominators without changing saved scores.

The replay uses Triton at a full cache of *its configured cap*.  The original
cost summary instead used a literal1024 threshold.  This CPU-only derivation
keeps original summaries and all numerators except Rebuild materialization.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
from hstu_kvcache.methods.baselines.protocol.cost import CostModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corrected_reference(original, all_rebuild, feedback_rebuild):
    result = copy.deepcopy(original)
    result['all_resident_once_rebuild_flops'] = int(all_rebuild)
    result['feedback_resident_once_rebuild_flops'] = int(feedback_rebuild)
    result['f_feedback_preparation_percent_rebuild'] = 100*(
        result['f_calibration_flops']+result['feedback_resident_f_materialization_flops'])/feedback_rebuild
    result['f_all_resident_preparation_percent_rebuild'] = 100*(
        result['f_calibration_flops']+result['all_resident_f_materialization_flops'])/all_rebuild
    for method, point in result['persistent_method_preparation'].items():
        if method == 'rebuild':
            point['feedback_materialization_flops'] = int(feedback_rebuild)
            point['all_resident_materialization_flops'] = int(all_rebuild)
        point['feedback_preparation_percent_once_rebuild'] = 100*(
            point['shared_calibration_flops']+point['feedback_materialization_flops'])/feedback_rebuild
        point['all_resident_preparation_percent_once_rebuild'] = 100*(
            point['shared_calibration_flops']+point['all_resident_materialization_flops'])/all_rebuild
    return result


def derive(folder):
    folder = Path(folder)
    config_path, summary_path = folder/'configuration.json', folder/'summary.json'
    config, summary = json.loads(config_path.read_text()), json.loads(summary_path.read_text())
    if summary['status'] != 'complete':
        raise ValueError('wait for a complete original summary')
    population_path = Path(config['population'])
    if sha(population_path) != config['population_sha256']:
        raise ValueError('frozen population changed')
    population = json.loads(population_path.read_text())
    entry = population['edges'][str(config['edge'])]
    if entry['inputs']['binding']['sha256'] != config['input_binding_sha256']:
        raise ValueError('population refers to a different replay input binding')
    cap = int(config['history_cap'])
    if cap != int(config['settings']['history_length']):
        raise ValueError('configured history cap differs from execution settings')
    if config['settings']['attention_backend'] != 'triton':
        raise ValueError('this derivation is for the frozen Triton replay')
    if config['users'] != population['users'] and not config['diagnostic']:
        raise ValueError('formal execution changed frozen residents')
    selected, feedback = set(config['users']), set(config['feedback_users'])
    if not feedback <= selected:
        raise ValueError('feedback users outside selected residents')
    if summary['feedback_users'] != len(feedback):
        raise ValueError('incomplete feedback execution')
    history_path = Path(entry['inputs']['history']['path'])
    if sha(history_path) != entry['inputs']['history']['sha256']:
        raise ValueError('native event prefix changed')
    release = int(config['settings']['release_day'])*86400
    all_rebuild = feedback_rebuild = raw_all = raw_feedback = 0
    histogram, prefix_checks = {}, {}
    seen = set()
    with np.load(history_path, allow_pickle=False) as pack:
        offsets, times = pack['offsets'], pack['timestamps']
        for index, uid in enumerate(pack['uids']):
            uid = int(uid)
            if uid not in selected:
                continue
            timestamps = times[offsets[index]:offsets[index+1]]
            cut = int(np.searchsorted(timestamps, release, side='left'))
            metadata_count = int(entry['user_metadata'][str(uid)]['history_count'])
            if metadata_count != cut:
                raise ValueError('frozen prerelease history_count differs from native event prefix')
            n = min(cap, metadata_count)
            if n != len(timestamps[max(0,cut-cap):cut]):
                raise ValueError('capped metadata count differs from actual retained prefix')
            backend = 'triton' if n >= cap else 'torch'
            original_backend = 'triton' if n == 1024 else 'torch'
            cost = CostModel.for_scale('large', backend).full_cache(n)
            raw = CostModel.for_scale('large', original_backend).full_cache(n)
            all_rebuild += cost; raw_all += raw
            if uid in feedback:
                feedback_rebuild += cost; raw_feedback += raw
            key = f'{backend}/length{n}'
            histogram[key] = histogram.get(key,0)+1
            prefix_checks[str(uid)] = n
            seen.add(uid)
    if seen != selected:
        raise ValueError('native history pack omitted residents')
    original = summary['cost_reference']
    if (original['all_resident_once_rebuild_flops'] != raw_all
            or original['feedback_resident_once_rebuild_flops'] != raw_feedback):
        raise ValueError('original denominator differs from its declared literal1024 formula')
    corrected = corrected_reference(original,all_rebuild,feedback_rebuild)
    if cap == 1024 and corrected != original:
        raise ValueError('cap1024 must remain numerically unchanged')
    result = dict(format='real_quality_cost_reference_corrected_v1',
        edge=config['edge'], history_cap=cap, status='complete_cpu_derivation',
        source_configuration_sha256=sha(config_path), source_summary_sha256=sha(summary_path),
        population_sha256=sha(population_path), history_sha256=sha(history_path),
        derivation_source_sha256=sha(__file__),
        cost_model_source_sha256=sha(ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py'),
        corrected_cost_reference=corrected, original_cost_reference=original,
        corrected_all_resident_once_rebuild_flops=int(all_rebuild),
        corrected_feedback_resident_once_rebuild_flops=int(feedback_rebuild),
        all_resident_original_denominator_overestimate_percent=100*(raw_all/all_rebuild-1),
        feedback_original_denominator_overestimate_percent=100*(raw_feedback/feedback_rebuild-1),
        prefix_validation=dict(users=len(seen), feedback_users=len(feedback),
            min_metadata_history_count_cap_equals_actual_native_prefix=True,
            prefix_lengths_sha256=hashlib.sha256(json.dumps(prefix_checks,sort_keys=True).encode()).hexdigest(),
            actual_backend_length_histogram=histogram),
        correction_rule='Actual _short_prefix_backend uses Triton when retained_length>=history_cap; otherwise Torch. Original summary used retained_length==1024.',
        original_summary_modified=False, scores_modified=False, gpu_execution=False,
        cap1024_unchanged=(cap==1024),
        scope='Only once-Rebuild denominators and corresponding Rebuild materialization/preparation ratios corrected. F/simple-map numerators and calibration charges are unchanged. Simple-map fitting used1024 and has no small-cap backend error. Research oracle work is not deployment cost.')
    output=folder/'cost_reference_corrected.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (folder/'cost_reference_corrected.md').write_text(
        f"# Preparation cost correction: edge{config['edge']}, cap{cap}\n\n"
        'The native replay selects Triton at its configured full cap; the original cost summary used a literal1024 threshold. '
        'This CPU derivation preserves the original summary and scores. All persistent-method preparation ratios below must use the corrected attachment.\n\n'
        f"- Resident once-Rebuild FLOPs: {raw_all} → {all_rebuild}.\n"
        f"- Feedback-user once-Rebuild FLOPs: {raw_feedback} → {feedback_rebuild}.\n"
        f"- All {len(seen)} retained prefixes matched `min(frozen history_count,cap)`.\n"
        + ('- cap1024 costs are unchanged.\n' if cap==1024 else '- Small-cap original ratios used an inflated denominator and are superseded here.\n')
        + '- F and simple-map fitting/materialization numerators remain unchanged; the simple-map fit used1024.\n')
    print(json.dumps(dict(output=str(output),edge=config['edge'],history_cap=cap,
        cap1024_unchanged=cap==1024,all_resident_denominator=int(all_rebuild),
        feedback_denominator=int(feedback_rebuild))),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    arguments=parser.parse_args()
    for run_folder in arguments.input:
        derive(run_folder)

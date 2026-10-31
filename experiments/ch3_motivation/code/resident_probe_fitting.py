"""Optional input placement for one unchanged ``probes.fit_probe`` stage.

Only immutable fields consumed by ``sample_to_device`` are placed in advance.
In particular, the CPU ``parent``/``teacher`` aliases and ``raw`` are retained
as they are.  No correction, normalized row features, or teacher read at a
student query is cached here.  This module does not patch any fitting driver.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import torch

from experiments.ch3_motivation.code import probes
from hstu_kvcache.models import HSTUKVCache


# Keep this explicit policy aligned with the referenced core source SHA.  The
# original core mover remains in use inside fit_probe; same-device .to is free
# of an additional tensor allocation.
PLACED_FIELDS = (
    "parent_cache", "teacher_cache", "item_ids", "item_embeddings",
    "candidates", "query_delta", "teacher_scores", "read_scales", "score_scale",
)


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _storage(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes()), storage.nbytes()


def _place_samples(train, validation, device):
    """Shallow copies, reusing identical views of shared immutable storage."""
    memo = {}
    input_storages, resident_storages = {}, {}
    input_tensor_objects = set()
    tensor_references = cross_device_copies = cross_device_logical_bytes = 0

    def place_tensor(value):
        nonlocal tensor_references, cross_device_copies, cross_device_logical_bytes
        tensor_references += 1
        key, size = _storage(value)
        input_storages[key] = size
        input_tensor_objects.add(id(value))
        # Prepared loss scales are distinct Tensor wrappers of the same view
        # across users.  Identical views can share one placement; different
        # slices/strides still receive their ordinary Tensor.to result.
        identity = (key, value.dtype, tuple(value.shape), tuple(value.stride()), value.storage_offset())
        if identity not in memo:
            result = value.to(device)
            memo[identity] = result
            if value.device != result.device:
                cross_device_copies += 1
                cross_device_logical_bytes += value.numel()*value.element_size()
        result = memo[identity]
        key, size = _storage(result)
        resident_storages[key] = size
        return result

    def place_group(samples):
        result = []
        for sample in samples:
            placed = dict(sample)
            for field in PLACED_FIELDS:
                value = sample.get(field)
                if isinstance(value, torch.Tensor):
                    placed[field] = place_tensor(value)
                elif isinstance(value, HSTUKVCache):
                    placed[field] = HSTUKVCache(place_tensor(value.k), place_tensor(value.v), value.seq_len)
            result.append(placed)
        return result

    resident_train, resident_validation = place_group(train), place_group(validation)
    storage = dict(
        used_tensor_references=tensor_references,
        unique_input_tensor_objects=len(input_tensor_objects),
        unique_preplacement_views=len(memo),
        unique_input_storages=len(input_storages),
        input_storage_bytes=sum(input_storages.values()),
        unique_resident_storages=len(resident_storages),
        resident_storage_bytes=sum(resident_storages.values()),
        new_resident_storages=sum(key not in input_storages for key in resident_storages),
        new_resident_storage_bytes=sum(size for key, size in resident_storages.items() if key not in input_storages),
        cross_device_tensor_copies=cross_device_copies,
        cross_device_logical_bytes=cross_device_logical_bytes,
        byte_convention="preplacement only: unique backing storage bytes; copy payload is logical tensor bytes; unused aliases/core temporaries excluded",
    )
    # The lists own the placements.  Nothing survives into a later fit stage.
    memo.clear()
    return resident_train, resident_validation, storage


def fit_probe_resident(model, probe, train_samples, validation_samples, **fit_kwargs):
    """Run the original fitter with its immutable inputs resident for this call.

    The return value preserves every original fit_probe field and adds only
    ``execution_adapter={binding, metrics}``.  Binding is stable execution
    provenance; elapsed times and memory measurements are separate metrics.
    Arithmetic, optimizer, sampling, targets, and validation are all performed
    by the original fitter.  Host/device transfer is not counted as FLOPs.
    """
    train, validation = list(train_samples), list(validation_samples)
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    binding = dict(
        format="resident_probe_fitting_v1",
        helper_source=_source(__file__),
        reference_source=_source(probes.__file__),
        reference_function="experiments.ch3_motivation.code.probes.fit_probe",
        placed_fields=list(PLACED_FIELDS),
        placement_policy="one-stage shallow sample copies; identical storage-view memo; no computed-feature cache",
        core_arithmetic="unchanged original fit_probe operation_counts and arithmetic; placement adds no arithmetic charge",
        lifetime="resident sample lists released after this fit call; no cross-stage cache",
    )

    def memory():
        return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                    reserved_bytes=torch.cuda.memory_reserved(device))

    cuda_memory = None
    if is_cuda:
        torch.cuda.synchronize(device)
        cuda_memory = dict(scope="this process/device since adapter reset_peak_memory_stats; includes existing allocations",
                           before=memory())
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    resident_train, resident_validation, storage = _place_samples(train, validation, device)
    if is_cuda:
        torch.cuda.synchronize(device)
        cuda_memory["after_preplacement"] = memory()
    placed_at = time.perf_counter()
    try:
        report = probes.fit_probe(model, probe, resident_train, resident_validation, **fit_kwargs)
        if is_cuda:
            torch.cuda.synchronize(device)
            cuda_memory["after_fit"] = memory()
            cuda_memory["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            cuda_memory["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        finished = time.perf_counter()
    finally:
        resident_train.clear()
        resident_validation.clear()
    report["execution_adapter"] = dict(binding=binding, metrics=dict(
        model_device=str(device), training_samples=len(train), validation_samples=len(validation),
        preplacement_seconds=placed_at-started, core_fit_seconds=finished-placed_at,
        total_seconds=finished-started, storage=storage, cuda_memory=cuda_memory,
    ))
    return report

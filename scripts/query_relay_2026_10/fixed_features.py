"""One immutable origin-encoder bank, shared by every release in a process.

Only the item's fixed affine contribution is materialized. Native K/V enters
the frozen originating encoder when a code is prepared; no per-row feature is
retained. The bank is deliberately not an nn.Module buffer, so copying a small
release adapter cannot copy the catalogue-sized table.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from unified_reuse_2026_09.common import sha256

KIND = "query_relay_fixed_origin_encoder_v1"
_CPU_BANKS = {}
_DEVICE_BANKS = {}


@dataclass
class FrozenFeatureBank:
    table: torch.Tensor  # [layers,catalog,48], origin e @ origin item weights
    bias: torch.Tensor  # [layers,48], separate to keep the original add order
    native_weights: tuple[tuple[torch.Tensor, ...], ...]
    metadata: dict

    def storage_bytes(self):
        tensors = [self.table, self.bias]
        tensors.extend(weight for layer in self.native_weights for weight in layer)
        return sum(value.numel() * value.element_size() for value in tensors)


def encoder_digest(source):
    digest = hashlib.sha256()
    for layer in source.layers:
        for value in (layer.input_weight, layer.input_bias):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _description(path, bank, digest, *, created):
    return dict(path=str(path), sha256=digest, created=created,
                catalogue_items=bank.metadata["catalogue_items"],
                num_layers=bank.metadata["num_layers"], code_width=bank.metadata["code_width"],
                prepare_flops=bank.metadata["prepare_flops"] if created else 0,
                original_prepare_flops=bank.metadata["prepare_flops"],
                table_bytes=bank.table.numel() * bank.table.element_size(),
                storage_bytes=bank.storage_bytes(), file_bytes=path.stat().st_size,
                source_bindings=bank.metadata["source_bindings"])


@torch.no_grad()
def build_bank(source, origin_item_features, path, *, source_bindings, chunk_size=65536):
    """Build once from the origin's full catalogue, or verify and reuse it.

    Pass the original model's item embedding weight directly: its lookups are
    plain embedding lookups, so another full-catalogue embedding copy is not
    needed. Catalogue IDs are raw model IDs, including row zero.
    """
    path = Path(path).resolve()
    if not source_bindings:
        raise ValueError("the frozen item bank requires an origin model binding")
    reference = next(source.parameters())
    metadata = dict(kind=KIND, source_config=source.get_config(),
        encoder_sha256=encoder_digest(source), source_bindings=dict(source_bindings),
        num_layers=source.num_layers, catalogue_items=len(origin_item_features),
        item_width=source.heads * source.head_dim, code_width=source.code_width,
        prepare_flops=2 * source.num_layers * len(origin_item_features)
                      * source.heads * source.head_dim * source.code_width)
    if path.exists():
        digest = sha256(path)
        bank = load_bank(path, digest, device="cpu", dtype=reference.dtype)
        if bank.metadata != metadata:
            raise ValueError("existing feature bank belongs to a different origin")
        return _description(path, bank, digest, created=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = torch.empty(source.num_layers, len(origin_item_features), metadata["code_width"],
                        dtype=reference.dtype, device="cpu")
    biases, native = [], []
    for index, layer in enumerate(source.layers):
        weights, item_weight, bias = layer.effective_weights()
        native.append(tuple(weight.detach().cpu().clone() for weight in weights))
        biases.append(bias.detach().cpu().clone())
        for start in range(0, len(origin_item_features), chunk_size):
            stop = min(start + chunk_size, len(origin_item_features))
            features = origin_item_features[start:stop].to(item_weight)
            table[index, start:stop].copy_((features @ item_weight).cpu())
    bank = FrozenFeatureBank(table, torch.stack(biases), tuple(native), metadata)
    payload = dict(metadata=metadata, table=bank.table, bias=bank.bias,
                   native_weights=bank.native_weights)
    # Readers only see a complete artifact; experiments build this asset before
    # launching branches. There is no concurrent artifact-update protocol.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(path)
    _CPU_BANKS[(str(path), digest)] = bank
    return _description(path, bank, digest, created=True)


def load_bank(path, expected_sha256, *, device, dtype):
    """Hash-check once; share a single table per artifact, device and dtype."""
    path = Path(path).resolve()
    cpu_key = (str(path), expected_sha256)
    if cpu_key not in _CPU_BANKS:
        if sha256(path) != expected_sha256:
            raise ValueError("fixed feature bank SHA256 changed")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payload["metadata"]["kind"] != KIND:
            raise ValueError("not an origin encoder feature bank")
        _CPU_BANKS[cpu_key] = FrozenFeatureBank(payload["table"], payload["bias"],
            tuple(tuple(weights) for weights in payload["native_weights"]), payload["metadata"])
    device = torch.device(device)
    device_key = (cpu_key, str(device), dtype)
    if device_key not in _DEVICE_BANKS:
        source = _CPU_BANKS[cpu_key]
        _DEVICE_BANKS[device_key] = FrozenFeatureBank(source.table.to(device=device, dtype=dtype),
            source.bias.to(device=device, dtype=dtype),
            tuple(tuple(weight.to(device=device, dtype=dtype) for weight in weights)
                  for weights in source.native_weights), source.metadata)
    return _DEVICE_BANKS[device_key]

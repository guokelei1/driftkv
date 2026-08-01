from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

XP_THETA0_PROTOCOL = "evokv_xp_theta0_cooccurrence_training_development_v0"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(
            lambda: source.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class StructuredSemiOrthogonalExpansion:
    source_width: int
    target_width: int

    def __post_init__(self) -> None:
        if self.source_width < 1 or self.target_width <= self.source_width:
            raise ValueError("structured expansion dimensions are invalid")

    @property
    def full_repeats(self) -> int:
        return self.target_width // self.source_width

    @property
    def remainder(self) -> int:
        return self.target_width % self.source_width

    @property
    def nullspace_dimension(self) -> int:
        return self.target_width - self.source_width

    def occurrence_counts(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        counts = torch.full(
            (self.source_width,),
            self.full_repeats,
            dtype=dtype,
            device=device,
        )
        if self.remainder:
            counts[: self.remainder] += 1
        return counts

    def projection_weight(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        target = torch.device(device)
        counts = self.occurrence_counts(
            device=target,
            dtype=dtype,
        )
        scales = torch.rsqrt(counts)
        projection = torch.zeros(
            (self.source_width, self.target_width),
            dtype=dtype,
            device=target,
        )
        columns = torch.arange(
            self.source_width,
            dtype=torch.int64,
            device=target,
        )
        for repeat in range(self.full_repeats):
            sign = -1.0 if repeat % 2 else 1.0
            projection[
                columns,
                repeat * self.source_width + columns,
            ] = scales * sign
        if self.remainder:
            columns = columns[: self.remainder]
            sign = -1.0 if self.full_repeats % 2 else 1.0
            projection[
                columns,
                self.full_repeats * self.source_width + columns,
            ] = scales[: self.remainder] * sign
        return projection

    def expand_rows(
        self,
        source: torch.Tensor,
        *,
        row_chunk: int = 8192,
        global_row_start: int = 0,
        global_row_stride: int = 1,
        nullspace_norm_ratio: float = 0.05,
    ) -> torch.Tensor:
        if (
            source.ndim != 2
            or source.shape[1] != self.source_width
            or not source.is_floating_point()
            or row_chunk < 1
            or global_row_start < 0
            or global_row_stride < 1
            or nullspace_norm_ratio < 0
        ):
            raise ValueError("structured expansion source differs")
        counts = self.occurrence_counts(
            device=source.device,
            dtype=source.dtype,
        )
        scales = torch.rsqrt(counts)
        expanded = torch.empty(
            (source.shape[0], self.target_width),
            dtype=source.dtype,
            device=source.device,
        )
        if nullspace_norm_ratio > 0:
            support_indices, support_values = (
                self.nullspace_basis_support(
                    device=source.device,
                    dtype=source.dtype,
                )
            )
        for start in range(0, source.shape[0], row_chunk):
            stop = min(start + row_chunk, source.shape[0])
            block = source[start:stop]
            for repeat in range(self.full_repeats):
                sign = -1.0 if repeat % 2 else 1.0
                expanded[
                    start:stop,
                    repeat * self.source_width : (
                        (repeat + 1) * self.source_width
                    ),
                ] = block * scales * sign
            if self.remainder:
                sign = (
                    -1.0
                    if self.full_repeats % 2
                    else 1.0
                )
                expanded[
                    start:stop,
                    self.full_repeats * self.source_width :,
                ] = (
                    block[:, : self.remainder]
                    * scales[: self.remainder]
                    * sign
                )
            if nullspace_norm_ratio > 0:
                global_rows = (
                    torch.arange(
                        start,
                        stop,
                        dtype=torch.int64,
                        device=source.device,
                    )
                    * global_row_stride
                    + global_row_start
                )
                basis = self.selected_nullspace_basis(global_rows)
                indices = support_indices.index_select(0, basis)
                values = support_values.index_select(0, basis)
                magnitudes = (
                    torch.linalg.vector_norm(block, dim=1)
                    * nullspace_norm_ratio
                )
                signs = torch.where(
                    (
                        torch.div(
                            global_rows,
                            self.nullspace_dimension,
                            rounding_mode="floor",
                        )
                        + global_rows
                    ).remainder(2)
                    == 0,
                    1.0,
                    -1.0,
                ).to(source.dtype)
                expanded[start:stop].scatter_add_(
                    1,
                    indices,
                    magnitudes.unsqueeze(1)
                    * signs.unsqueeze(1)
                    * values,
                )
        return expanded

    def selected_nullspace_basis(
        self,
        global_row_ids: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.nullspace_dimension < 1
            or global_row_ids.ndim != 1
            or global_row_ids.dtype != torch.int64
            or (
                global_row_ids.numel()
                and bool(torch.any(global_row_ids < 0))
            )
        ):
            raise ValueError("nullspace row identity differs")
        return (
            global_row_ids * 104_729 + 8_191
        ).remainder(self.nullspace_dimension)

    def nullspace_basis_support(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maximum_count = self.full_repeats + (
            1 if self.remainder else 0
        )
        support_indices = []
        support_values = []
        for source_index in range(self.source_width):
            count = self.full_repeats + (
                1 if source_index < self.remainder else 0
            )
            coordinates = [
                repeat * self.source_width + source_index
                for repeat in range(self.full_repeats)
            ]
            if source_index < self.remainder:
                coordinates.append(
                    self.full_repeats * self.source_width
                    + source_index
                )
            signs = [
                -1.0 if repeat % 2 else 1.0
                for repeat in range(count)
            ]
            for basis_index in range(count - 1):
                denominator = (
                    (basis_index + 1) * (basis_index + 2)
                ) ** 0.5
                helmert = [
                    (
                        1.0 / denominator
                        if position <= basis_index
                        else (
                            -(basis_index + 1) / denominator
                            if position == basis_index + 1
                            else 0.0
                        )
                    )
                    for position in range(count)
                ]
                values = [
                    sign * value
                    for sign, value in zip(
                        signs,
                        helmert,
                        strict=True,
                    )
                ]
                support_indices.append(
                    coordinates
                    + [coordinates[0]] * (maximum_count - count)
                )
                support_values.append(
                    values + [0.0] * (maximum_count - count)
                )
        if len(support_indices) != self.nullspace_dimension:
            raise RuntimeError("nullspace basis dimension differs")
        return (
            torch.tensor(
                support_indices,
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                support_values,
                dtype=dtype,
                device=device,
            ),
        )

    def nullspace_basis(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        indices, values = self.nullspace_basis_support(
            device=device,
            dtype=dtype,
        )
        basis = torch.zeros(
            (self.nullspace_dimension, self.target_width),
            dtype=dtype,
            device=device,
        )
        basis.scatter_add_(1, indices, values)
        return basis

    def numeric_oracle(
        self,
        source: torch.Tensor,
        expanded: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        maximum_samples: int = 1024,
        global_row_start: int = 0,
        global_row_stride: int = 1,
        nullspace_norm_ratio: float = 0.05,
    ) -> dict[str, int | float | bool]:
        if (
            source.ndim != 2
            or expanded.shape
            != (source.shape[0], self.target_width)
            or projection_weight.shape
            != (self.source_width, self.target_width)
            or maximum_samples < 1
            or global_row_start < 0
            or global_row_stride < 1
            or nullspace_norm_ratio < 0
        ):
            raise ValueError("structured expansion oracle differs")
        count = min(source.shape[0], maximum_samples)
        if count == 0:
            raise ValueError("structured expansion oracle is empty")
        indices = torch.div(
            torch.arange(
                count,
                device=source.device,
                dtype=torch.int64,
            )
            * source.shape[0],
            count,
            rounding_mode="floor",
        )
        original = source.index_select(0, indices)
        sample_global_rows = (
            indices * global_row_stride + global_row_start
        )
        rowspace = self.expand_rows(
            original,
            row_chunk=count,
            global_row_start=0,
            global_row_stride=1,
            nullspace_norm_ratio=0.0,
        )
        sampled_expanded = expanded.index_select(0, indices)
        reconstructed = F.linear(
            sampled_expanded,
            projection_weight,
        )
        error = torch.abs(reconstructed - original)
        nullspace = sampled_expanded - rowspace
        nullspace_response = F.linear(
            nullspace,
            projection_weight,
        )
        global_rows = (
            torch.arange(
                source.shape[0],
                device=source.device,
                dtype=torch.int64,
            )
            * global_row_stride
            + global_row_start
        )
        selected = torch.unique(
            self.selected_nullspace_basis(global_rows)
        )
        source_energy = float(source.square().sum().item())
        nullspace_energy = (
            source_energy * nullspace_norm_ratio**2
        )
        row_norms = projection_weight.square().sum(dim=1)
        return {
            "sample_rows": count,
            "max_abs_error": float(error.max().item()),
            "mean_abs_error": float(error.mean().item()),
            "maximum_projection_row_norm_error": float(
                torch.max(torch.abs(row_norms - 1.0)).item()
            ),
            "nullspace_dimension": self.nullspace_dimension,
            "nullspace_norm_ratio": nullspace_norm_ratio,
            "source_energy": source_energy,
            "nullspace_energy": nullspace_energy,
            "nullspace_to_source_energy_ratio": (
                nullspace_norm_ratio**2
            ),
            "sampled_projected_nullspace_max_abs": float(
                torch.max(torch.abs(nullspace_response)).item()
            ),
            "local_selected_nullspace_basis_ids": [
                int(value) for value in selected.tolist()
            ],
            "sample_global_row_id_minimum": int(
                sample_global_rows.min().item()
            ),
            "sample_global_row_id_maximum": int(
                sample_global_rows.max().item()
            ),
            "all_target_coordinates_used": bool(
                torch.all(
                    torch.count_nonzero(
                        projection_weight,
                        dim=0,
                    )
                    == 1
                ).item()
            ),
        }

    def projection_nullspace_response(
        self,
        projection_weight: torch.Tensor,
        *,
        tolerance: float = 1e-8,
    ) -> dict[str, int | float]:
        if (
            projection_weight.shape
            != (self.source_width, self.target_width)
            or tolerance <= 0
        ):
            raise ValueError("projection nullspace response differs")
        basis = self.nullspace_basis(
            device=projection_weight.device,
            dtype=projection_weight.dtype,
        )
        response = F.linear(basis, projection_weight)
        norms = torch.linalg.vector_norm(response, dim=1)
        return {
            "nullspace_basis_directions": (
                self.nullspace_dimension
            ),
            "responding_basis_directions": int(
                torch.count_nonzero(norms > tolerance).item()
            ),
            "response_frobenius_norm": float(
                torch.linalg.vector_norm(response).item()
            ),
            "response_maximum_basis_norm": float(
                torch.max(norms).item()
            ),
            "response_mean_basis_norm": float(
                torch.mean(norms).item()
            ),
            "nonzero_tolerance": tolerance,
        }


@dataclass(frozen=True)
class XPBasePairCorpus:
    anchor_rows: np.ndarray
    positive_rows: np.ndarray
    occurrence_user_ids: np.ndarray
    negative_rows: np.ndarray
    semantic_rows: int
    isolated_rows: int
    file_sha256: str
    content_sha256: str
    metadata: dict[str, object]

    @property
    def pair_count(self) -> int:
        return len(self.anchor_rows)

    @property
    def pair_arrays_sha256(self) -> dict[str, str]:
        return {
            "anchor_rows": array_sha256(self.anchor_rows),
            "positive_rows": array_sha256(self.positive_rows),
            "occurrence_user_ids": array_sha256(
                self.occurrence_user_ids
            ),
            "negative_rows": array_sha256(self.negative_rows),
        }


def deterministic_cross_user_negatives(
    anchor_rows: np.ndarray,
    positive_rows: np.ndarray,
    occurrence_user_ids: np.ndarray,
    *,
    initial_stride: int = 1_000_003,
) -> np.ndarray:
    anchor = np.asarray(anchor_rows, dtype=np.int32)
    positive = np.asarray(positive_rows, dtype=np.int32)
    users = np.asarray(occurrence_user_ids, dtype=np.int64)
    if (
        anchor.ndim != 1
        or positive.shape != anchor.shape
        or users.shape != anchor.shape
        or len(anchor) < 2
        or initial_stride < 1
    ):
        raise ValueError("cross-user negative inputs differ")
    size = len(anchor)
    indices = np.arange(size, dtype=np.int64)
    negative = np.zeros(size, dtype=np.int32)
    unresolved = np.ones(size, dtype=bool)
    for attempt in range(32):
        stride = (
            initial_stride + attempt * 104_729
        ) % size
        if stride == 0:
            stride = attempt + 1
        candidates = (indices + stride) % size
        rows = positive[candidates]
        valid = (
            unresolved
            & (users[candidates] != users)
            & (rows != anchor)
            & (rows != positive)
        )
        negative[valid] = rows[valid]
        unresolved[valid] = False
        if not np.any(unresolved):
            break
    if np.any(unresolved):
        for index in np.flatnonzero(unresolved):
            for delta in range(1, size):
                candidate = (int(index) + delta) % size
                row = positive[candidate]
                if (
                    users[candidate] != users[index]
                    and row != anchor[index]
                    and row != positive[index]
                ):
                    negative[index] = row
                    unresolved[index] = False
                    break
    if np.any(unresolved):
        raise ValueError("cross-user negative corpus is degenerate")
    return negative


def load_xp_base_pair_corpus(
    path: str | Path,
    summary_path: str | Path,
    *,
    num_embeddings: int,
    expected_neighbor_rows: int | None = None,
    expected_isolated_rows: int | None = None,
    negative_stride: int = 1_000_003,
) -> XPBasePairCorpus:
    artifact = Path(path)
    summary = json.loads(Path(summary_path).read_text())
    artifact_hash = file_sha256(artifact)
    if (
        summary.get("protocol")
        != "evokv_qk_xp_base_row_coverage_development_v0"
        or summary.get("scientific_result") is not False
        or summary.get("artifact", {}).get("file_sha256")
        != artifact_hash
    ):
        raise ValueError("XP pair summary binding differs")
    with np.load(artifact, allow_pickle=False) as source:
        required = {
            "anchor_row",
            "positive_row",
            "occurrence_user_id",
            "has_same_user_neighbor",
            "metadata_json",
        }
        if not required.issubset(source.files):
            raise ValueError("XP pair artifact is incomplete")
        anchor = np.asarray(source["anchor_row"], dtype=np.int32)
        positive = np.asarray(source["positive_row"], dtype=np.int32)
        users = np.asarray(
            source["occurrence_user_id"],
            dtype=np.int64,
        )
        flags = np.asarray(
            source["has_same_user_neighbor"],
            dtype=np.uint8,
        )
        metadata = json.loads(str(source["metadata_json"].item()))
    semantic_rows = num_embeddings - 1
    if (
        anchor.shape != (semantic_rows,)
        or positive.shape != anchor.shape
        or users.shape != anchor.shape
        or flags.shape != anchor.shape
        or not np.array_equal(
            anchor,
            np.arange(1, num_embeddings, dtype=np.int32),
        )
        or np.any((flags != 0) & (flags != 1))
        or np.any(positive < 1)
        or np.any(positive >= num_embeddings)
        or metadata.get("content_sha256")
        != summary.get("content_sha256")
        or metadata.get("base_only_boundary", {}).get(
            "post_base_rows_used"
        )
        is not False
    ):
        raise ValueError("XP pair artifact semantics differ")
    mask = flags.astype(bool)
    pair_count = int(np.count_nonzero(mask))
    isolated_rows = semantic_rows - pair_count
    if (
        expected_neighbor_rows is not None
        and pair_count != expected_neighbor_rows
    ):
        raise ValueError("XP neighbor row count differs")
    if (
        expected_isolated_rows is not None
        and isolated_rows != expected_isolated_rows
    ):
        raise ValueError("XP isolated row count differs")
    anchor = anchor[mask].copy()
    positive = positive[mask].copy()
    users = users[mask].copy()
    negative = deterministic_cross_user_negatives(
        anchor,
        positive,
        users,
        initial_stride=negative_stride,
    )
    return XPBasePairCorpus(
        anchor_rows=anchor,
        positive_rows=positive,
        occurrence_user_ids=users,
        negative_rows=negative,
        semantic_rows=semantic_rows,
        isolated_rows=isolated_rows,
        file_sha256=artifact_hash,
        content_sha256=str(metadata["content_sha256"]),
        metadata=metadata,
    )


def projected_pairwise_contrastive_loss(
    item_vectors: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if (
        item_vectors.ndim != 3
        or item_vectors.shape[1] != 3
        or item_vectors.shape[2] < 1
        or temperature <= 0
    ):
        raise ValueError("projected contrastive batch differs")
    normalized = F.normalize(item_vectors, dim=-1)
    anchor = normalized[:, 0]
    positive = normalized[:, 1]
    negative = normalized[:, 2]
    positive_score = torch.sum(anchor * positive, dim=1)
    negative_score = torch.sum(anchor * negative, dim=1)
    return F.softplus(
        (negative_score - positive_score) / temperature
    ).mean()

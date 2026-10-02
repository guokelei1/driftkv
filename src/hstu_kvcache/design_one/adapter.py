"""Shared generator and response map for c = b(summary) + A(summary)q + Tr.

Fitting uses per-event response rates for numerical scaling. Preparation folds
that scaling and all input normalization into b, A and the shared cross-head T.
Only b and A are personalized; candidates always supply their actual q and r.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SummaryProjection:
    center: torch.Tensor
    scale: torch.Tensor
    projection: torch.Tensor
    diagnostics: dict = field(default_factory=dict)

    @classmethod
    @torch.no_grad()
    def fit(cls, features, rank=32):
        """Whiten source summaries on calibration users only, without teachers."""
        values = features.detach().double().cpu()
        center = values.mean(0)
        scale = values.std(0, correction=0).clamp_min(1e-3)
        normalized = (values - center) / scale
        # The user Gram matrix avoids an input-width squared decomposition.
        eigenvalues, vectors = torch.linalg.eigh(normalized @ normalized.T)
        positive = eigenvalues > eigenvalues[-1].clamp_min(1e-20) * 1e-8
        retained_rank = min(rank, int(positive.sum()))
        if retained_rank:
            retained = eigenvalues[-retained_rank:]
            projection = normalized.T @ vectors[:, -retained_rank:]
            projection *= ((max(1, len(values) - 1) ** .5) / retained)[None]
            energy = float(retained.sum() / eigenvalues.clamp_min(0).sum())
        else:
            projection = normalized.new_empty((values.shape[1], 0))
            energy = 0.0
        return cls(center, scale, projection, dict(
            input_dimension=values.shape[1], rank=retained_rank,
            fitting_scenes=len(values), source_variance_retained=energy,
        ))

    def encode_features(self, features):
        dtype, device = self.center.dtype, features.device
        latent = ((features.to(dtype) - self.center.to(device)) / self.scale.to(device)
                  @ self.projection.to(device))
        return torch.cat((latent.new_ones((len(latent), 1)), latent), dim=-1)

    def to(self, device, dtype=torch.float32):
        return SummaryProjection(*(value.to(device=device, dtype=dtype) for value in
                                   (self.center, self.scale, self.projection)),
                                 diagnostics=self.diagnostics)

    def state_dict(self):
        return dict(center=self.center.cpu(), scale=self.scale.cpu(),
                    projection=self.projection.cpu(), diagnostics=self.diagnostics)


@dataclass
class LayerView:
    offset: torch.Tensor           # [users, heads, output dimension]
    query_transform: torch.Tensor # [users, heads, input dimension, output dimension]
    response_transform: torch.Tensor # [heads, full response width, output dimension]

    def apply(self, query, native):
        delta = self.offset[:, :, None] + query @ self.query_transform
        if self.response_transform.shape[1]:
            response = native.transpose(1, 2).flatten(2)
            delta = delta + torch.einsum("bqa,had->bhqd", response, self.response_transform)
        return native + delta


@dataclass
class CompatibilityView:
    layers: tuple[LayerView, ...]
    revision: int | None = None

    def __call__(self, layer, query, native):
        # Prefix views support fitting the next layer at corrected lower reads.
        return self.layers[layer].apply(query, native) if layer < len(self.layers) else native

    def storage_bytes(self):
        # T belongs to the shared adapter, not the per-user compatibility view.
        return sum((layer.offset.numel() + layer.query_transform.numel())
                   * layer.offset.element_size() for layer in self.layers)

    def select(self, indices):
        """Select/repeat candidate owners, sharing release-level response maps."""
        return CompatibilityView(tuple(LayerView(
            layer.offset[indices], layer.query_transform[indices], layer.response_transform,
        ) for layer in self.layers), revision=self.revision)

    def put(self, indices, updated):
        """Replace only views of users whose summaries changed."""
        for layer, new in zip(self.layers, updated.layers, strict=True):
            layer.offset[indices] = new.offset
            layer.query_transform[indices] = new.query_transform


class SharedReadAdapter:
    def __init__(self, projection, parameters, *, producer_ids=(4, 5), target=5,
                 max_length=1024, summary_mode="producer_mean"):
        self.projection = projection
        self.parameters = list(parameters)
        self.producer_ids = tuple(producer_ids)
        self.target = target
        self.max_length = max_length
        self.summary_mode = summary_mode
        self._response_maps = [self._response_map(layer) for layer in self.parameters]

    @staticmethod
    def _response_map(parameters):
        # Folding the response scale leaves a shared full-width map, independent
        # of the user's retained length. Empty maps support q-only ablations.
        transform = parameters["read_weights"] / parameters["read_scale"][None, :, None]
        center = torch.einsum("a,had->hd", parameters["read_center"], transform)
        return transform, center

    def prepare(self, summary):
        if summary.producer_ids != self.producer_ids:
            raise ValueError("summary producer order must match the fitted generator")
        view = self.prepare_features(summary.features(mode=self.summary_mode), summary.batch_counts())
        view.revision = summary.revision
        return view

    def prepare_features(self, features, counts):
        latent = self.projection.encode_features(features)
        views = []
        for parameters, (response_map, response_center) in zip(
                self.parameters, self._response_maps, strict=True):
            weights = parameters["weights"]
            encoded = latent.to(device=weights.device, dtype=weights.dtype)
            coefficients = torch.einsum("br,rhjd->bhjd", encoded, weights)
            slope = coefficients[:, :, 1:] / parameters["query_scale"].transpose(-2, -1)
            offset = coefficients[:, :, 0] - (parameters["query_center"] @ slope).squeeze(-2)
            offset = offset - response_center[None]
            mass = counts.to(device=weights.device, dtype=weights.dtype)
            views.append(LayerView(offset * mass[:, None, None],
                                   slope * mass[:, None, None, None], response_map))
        return CompatibilityView(tuple(views))

    def to(self, device, dtype=torch.float32):
        parameters = [{key: value.to(device=device, dtype=dtype)
                       for key, value in layer.items()} for layer in self.parameters]
        return SharedReadAdapter(self.projection.to(device, dtype), parameters,
                                 producer_ids=self.producer_ids, target=self.target,
                                 max_length=self.max_length, summary_mode=self.summary_mode)

    def state_dict(self):
        return dict(projection=self.projection.state_dict(),
                    parameters=[{key: value.cpu() for key, value in layer.items()}
                                for layer in self.parameters],
                    producer_ids=self.producer_ids, target=self.target,
                    max_length=self.max_length, summary_mode=self.summary_mode)

    def estimate_flops(self, *, batch=1, candidates=1):
        """Analytic extra arithmetic; multiply-add counts as two FLOPs.

        Initial scans/changed-event sums and teacher/fitting work are accounted
        by the runner. These costs include every fitted prefix layer, with no
        assumed fusion or sparsity. They are not latency estimates.
        """
        features, rank = self.projection.projection.shape
        generation = reads = shared = 0
        for parameters in self.parameters:
            latent, heads, augmented, width = parameters["weights"].shape
            query_width = augmented - 1
            response_width = parameters["read_weights"].shape[1]
            generation += (2 * batch * latent * heads * augmented * width
                           + 4 * batch * heads * query_width * width
                           + 3 * batch * heads * width)
            reads += (2 * batch * candidates * heads * query_width * width
                      + 2 * batch * candidates * heads * width)
            if response_width:
                shared += 3 * heads * response_width * width
                reads += (2 * batch * candidates * heads * response_width * width
                          + batch * candidates * heads * width)
        return dict(summary_projection=2 * batch * features * (rank + 1),
                    view_generation=generation, candidate_reads=reads,
                    shared_response_preparation=shared)

    @classmethod
    def from_state_dict(cls, state):
        return cls(SummaryProjection(**state["projection"]), state["parameters"],
                   producer_ids=state["producer_ids"], target=state["target"],
                   max_length=state["max_length"],
                   summary_mode=state.get("summary_mode", "producer_mean"))

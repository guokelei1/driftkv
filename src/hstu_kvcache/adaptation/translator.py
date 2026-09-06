"""A shared residual segment Translator; one network per target release."""

from dataclasses import replace

import torch
from torch import nn

from .summary import Summary


class Translator(nn.Module):
    def __init__(self, *, layers=6, width=192, slots=2, hidden=64, target=1, context="cumulative"):
        super().__init__()
        self.target = target
        self.context = context
        self.supported_producers = set(range(target))
        self.register_buffer("scale", torch.ones(layers, 2, 1))
        dimension = slots * layers * 2 * width
        self.encoder = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU())
        self.mixer = nn.Sequential(nn.Linear(2 * hidden + 6 + 2 * slots, hidden), nn.SiLU())
        self.decoder = nn.Linear(hidden, dimension)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, source: Summary) -> Summary:
        normalized = source.payload / self.scale
        encoded = self.encoder(normalized.flatten(1))
        mass = source.count.sum(1, keepdim=True)
        if self.context == "local":
            context = torch.zeros_like(encoded)
        else:
            context = (encoded * mass).cumsum(0) / mass.cumsum(0).clamp_min(1)
        producer = nn.functional.one_hot(source.producer, 6).float()
        # Stable source coordinates only: current appends cannot invalidate these inputs.
        anchor = source.ordinal[:, :1] if self.context == "local" else source.ordinal[0, 0]
        ages = (source.ordinal - anchor).float() / 1024
        metadata = torch.cat((producer, source.count / 32, ages), dim=1)
        latent = self.mixer(torch.cat((encoded, context, metadata), dim=1))
        residual = self.decoder(latent).reshape_as(source.payload) * self.scale
        payload = source.payload + residual * (source.count > 0)[..., None, None, None]
        return source.with_payload(payload)


class FunctionalTranslator(nn.Module):
    """Shared source-to-functional-view estimator, not a per-user oracle fit.

For the frozen ELU+1 models, a zero-key slot reads its value independent of Q.
The source's actual K/V means remain estimator inputs. Its read-view retains
the constant (sum V) term; translation adds a learned change to that term.
Ordinary event K/V and producer identities are never overwritten.
"""

    def __init__(self, *, layers=6, width=192, hidden=64, target=1):
        super().__init__()
        self.target, self.context = target, "global"
        self.supported_producers = set(range(target))
        self.register_buffer("scale", torch.ones(layers, 2, 1))
        self.register_buffer("response_scale", torch.ones(layers, 1))
        self.encoder = nn.Sequential(nn.Linear(layers * 2 * width + 7, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.decoder = nn.Linear(hidden, layers * width)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, source: Summary) -> Summary:
        count = source.count.sum()
        mean = (source.payload * source.count[..., None, None, None]).sum((0, 1)) / count
        producers = nn.functional.one_hot(source.producer, 6).float()
        mixture = (producers * source.count.sum(1, keepdim=True)).sum(0) / count
        inputs = torch.cat(((mean / self.scale).flatten(), mixture, count.reshape(1) / 1024))
        delta = self.decoder(self.encoder(inputs)).reshape_as(mean[:, 0]) * self.response_scale
        payload = torch.stack((source.payload[..., 0, :], source.payload[..., 1, :] + delta / count), dim=-2)
        return replace(source, payload=payload, read_mode="constant_elu")


class RidgeTranslator(nn.Module):
    """Shared low-rank ridge map to per-old-event response-change rates.

Producer-conditioned first moments preserve which version wrote each input.
Calibration refits shared weights at the path's corrected queries; inference
only reads retained source statistics. Count scaling follows native additivity.
"""

    def __init__(self, *, layers=6, width=192, target=1, rank=32, history_scope="old", temporal=False,
                 layer_clearance=False, temporal_source_rank=0, response_strength=None,
                 source_confidence=False, second_moments=False, source_kernel=False):
        super().__init__()
        self.layers, self.width, self.target, self.rank = layers, width, target, rank
        self.context = "global"
        self.history_scope = history_scope
        self.temporal = temporal
        self.layer_clearance = layer_clearance
        # Six release-level scalars are saved in the experiment configuration.
        # Applying them after the affine map avoids cancellation changes from
        # refactoring its large decode/offset terms in FP32.
        self.register_buffer("response_strength", torch.ones(layers) if response_strength is None
                             else torch.tensor(response_strength, dtype=torch.float32), persistent=False)
        if layer_clearance and history_scope != "all":
            raise ValueError("layer clearance requires full-history native-write features")
        self.register_buffer("clearance_windows", torch.arange(1, layers + 1), persistent=False)
        self.time_dim = 32
        self.producer_count = target + int(history_scope == "all")
        self.supported_producers = set(range(self.producer_count))
        self.second_moments = second_moments
        self.summary_inputs = layers*2*width*(1+int(second_moments))
        inputs = self.producer_count * (self.summary_inputs + 1) + int(history_scope == "all")
        self.base_inputs = inputs
        self.source_kernel = source_kernel
        if source_kernel:
            # The current 64-user/16-lifetime experiment has at most 384 scenes.
            # Unused support rows have zero coefficients, including at identity.
            self.register_buffer("kernel_centers", torch.zeros(384, inputs))
            self.register_buffer("kernel_encode", torch.zeros(384, rank))
            self.register_buffer("kernel_bandwidth_squared", torch.ones(()))
        self.source_confidence = source_confidence
        if source_confidence:
            self.register_buffer("source_norm_limit", torch.tensor(float("inf")))
        self.temporal_source_rank = temporal_source_rank
        self.time_condition_dim = self.producer_count + temporal_source_rank
        if temporal_source_rank:
            if not temporal:
                raise ValueError("source-conditioned time features require the temporal reader")
            self.register_buffer("time_source_center", torch.zeros(inputs))
            self.register_buffer("time_source_scale", torch.ones(inputs))
            self.register_buffer("time_source_projection", torch.zeros(inputs, temporal_source_rank))
        self.register_buffer("source_padding", torch.zeros(self.summary_inputs), persistent=False)
        if temporal:
            inputs += self.time_condition_dim*self.time_dim
            self.register_buffer("time_weights",torch.zeros(self.time_condition_dim,self.time_dim,layers,width))
        self.register_buffer("center", torch.zeros(inputs))
        self.register_buffer("input_scale", torch.ones(inputs))
        self.register_buffer("encode", torch.zeros(inputs, rank))
        self.register_buffer("decode", torch.zeros(rank, layers * width))
        self.register_buffer("offset", torch.zeros(layers * width))

    def features(self, source):
        count = source.count.sum()
        values = []
        for producer in range(self.producer_count):
            mass = source.count * (source.producer == producer)[:, None]
            weighted = (source.payload * mass[..., None, None, None]).sum((0, 1)) / count
            fraction = mass.sum()/count
            inputs = weighted.flatten()
            if self.second_moments:
                second = (source.second_moment * mass[..., None, None, None]).sum((0, 1))/count
                inputs = self.source_values(torch.cat((inputs, second.flatten())), fraction)
            values.extend((inputs, fraction.reshape(1)))
        if self.history_scope == "all":
            values.append(count.new_tensor([min(source.release_age,6*1024)/1024]))
        return torch.cat(values)

    def source_values(self, values, fractions):
        """Producer-weighted mean and central variance from additive moments."""
        if not self.second_moments:
            return values
        mean, second = values.chunk(2, dim=-1)
        variance = (second-mean.square()/fractions[..., None].clamp_min(1e-20)).clamp_min(0)
        return torch.cat((mean, variance), dim=-1)

    def forward(self, source):
        features = self.features(source)
        rate = self.rates(features)
        # Adding a per-event rate to each retained slot produces count * rate
        # in the paired constant-basis response; no learned count extrapolation.
        payload = torch.stack((source.payload[..., 0, :], source.payload[..., 1, :] + rate),dim=-2)
        return replace(source,payload=payload,read_mode="constant_elu",
                       temporal_coefficients=self.time_coefficients(features))

    def rates(self, features):
        if self.temporal and features.shape[-1] == self.base_inputs:
            # Zero Fourier coordinates define the stored intercept. Actual
            # time weights are applied by the reader, not guessed at release.
            features = nn.functional.pad(features,(0,self.time_condition_dim*self.time_dim))
        normalized = (features-self.center)/self.input_scale
        values = (normalized @ self.encode @ self.decode + self.offset).reshape(*features.shape[:-1],self.layers,self.width)
        if self.source_kernel:
            latent = self.kernel_features(normalized[..., :self.base_inputs]) @ self.kernel_encode
            values = values+(latent @ self.decode).reshape_as(values)
        if self.layer_clearance:
            values = values * self.active_layers(features)[..., None]
        if self.source_confidence:
            values = values * self.confidence_strength(features)[..., None, None]
        return values * self.response_strength[:, None]

    def kernel_features(self, normalized_source):
        distance = (normalized_source.square().sum(-1, keepdim=True)
                    + self.kernel_centers.square().sum(-1)-2*normalized_source @ self.kernel_centers.T).clamp_min(0)
        return (-distance/self.kernel_bandwidth_squared).exp()

    def confidence_strength(self, features):
        """Shrink corrections outside the release's calibration source support.

        Only source coordinates enter this common rule; query time must not
        change the scale of an already installed constant/temporal view.
        """
        source = (features[..., :self.base_inputs]-self.center[:self.base_inputs])/self.input_scale[:self.base_inputs]
        norm = source.square().sum(-1).clamp_min(1e-20)
        return (self.source_norm_limit/norm).sqrt().clamp(max=1)

    def active_layers(self, features):
        # The stored age is actual native writes / 1024. K/V in layer j
        # depends on at most j preceding rolling attention windows.
        return features[..., self.base_inputs - 1, None] < self.clearance_windows

    def producer_masses(self, features):
        stride=self.summary_inputs+1
        return features[...,stride-1:self.producer_count*stride:stride]

    @torch.no_grad()
    def fit_time_source(self, features):
        """Whiten eight source directions using calibration inputs alone."""
        self.time_source_center.copy_(features.mean(0))
        self.time_source_scale.copy_(features.std(0).clamp_min(1e-3))
        x=(features-self.time_source_center)/self.time_source_scale
        eigenvalues,vectors=torch.linalg.eigh(x @ x.T)
        retained=eigenvalues[-self.temporal_source_rank:].clamp_min(1e-6)
        projection=x.T @ vectors[:,-self.temporal_source_rank:]
        projection=projection*((len(x)-1)**.5/retained)[None,:]
        self.time_source_projection.copy_(projection)
        return dict(rank=self.temporal_source_rank,calibration_scenes=len(x),
                    retained_input_variance=float(retained.sum()/eigenvalues.clamp_min(0).sum()),
                    scope="standardized calibration source PCA only; no responses or labels")

    def time_condition(self, features):
        masses=self.producer_masses(features)
        if not self.temporal_source_rank:
            return masses
        source=(features[...,:self.base_inputs]-self.time_source_center)/self.time_source_scale
        latent=source @ self.time_source_projection
        return torch.cat((masses,latent),dim=-1)

    def with_time(self, features, time_features):
        joint=self.time_condition(features)[...,None]*time_features[...,None,:]
        return torch.cat((features,joint.flatten(-2)),dim=-1)

    def time_coefficients(self, features):
        if not self.temporal:
            return None
        values = torch.einsum("...p,ptlw->...ltw",self.time_condition(features),self.time_weights)
        if self.layer_clearance:
            values = values * self.active_layers(features)[..., None, None]
        if self.source_confidence:
            values = values * self.confidence_strength(features)[..., None, None, None]
        return values * self.response_strength[:, None, None]

    def writer_features(self, writers, release_ages):
        """Batch the same producer first moments directly from one-slot sums.

        This avoids a separate mean/divide/translate launch for every user at
        release. Actual producer IDs and retained counts remain authoritative.
        """
        sums, masses, owners = [], [], []
        for user,writer in enumerate(writers):
            if writer.slots != 1:
                raise ValueError("ridge publication uses the prototype's one-slot writer")
            for segment in writer.segments.values():
                producer=segment["producer"]
                if self.history_scope == "old" and producer == self.target:
                    continue
                values = segment["sums"][0].flatten()
                if self.second_moments:
                    values = torch.cat((values, segment["squared_sums"][0].flatten()))
                sums.append(values)
                masses.append(segment["count"][0])
                owners.append(user*self.producer_count+producer)
        shape=(len(writers),self.producer_count)
        totals=self.center.new_zeros(*shape,self.summary_inputs)
        count=self.center.new_zeros(*shape)
        if sums:
            index=torch.tensor(owners,device=self.center.device)
            totals.flatten(0,1).index_add_(0,index,torch.stack(sums))
            count.flatten().index_add_(0,index,self.center.new_tensor(masses))
        total_count=count.sum(1)
        divisor=total_count.clamp_min(1)
        fractions = count/divisor[:, None]
        values = self.source_values(totals/divisor[:,None,None], fractions)
        features=torch.cat((values,fractions[...,None]),2).flatten(1)
        if self.history_scope == "all":
            age=self.center.new_tensor(release_ages).clamp(max=6*1024)/1024
            features=torch.cat((features,age[:,None]),1)
        return features,total_count

    @torch.no_grad()
    def fit(self, features, rates):
        self.center.copy_(features.mean(0))
        self.input_scale.copy_(features.std(0).clamp_min(1e-3))
        x = (features-self.center)/self.input_scale
        if self.source_kernel:
            return self.fit_kernel(x, features, rates)
        if self.source_confidence:
            self.source_norm_limit.copy_(torch.quantile(x[:, :self.base_inputs].square().sum(-1), .95))
        scale = rates.square().mean((0,2)).sqrt().clamp_min(1e-7).repeat_interleave(self.width)
        y = rates.flatten(1)/scale
        mean = y.mean(0)
        gram = x @ x.T
        regularizer = (gram.trace()/len(x)*0.01).clamp_min(1e-6)
        alpha = torch.linalg.solve(gram + regularizer*torch.eye(len(x),device=x.device),y-mean)
        fitted = gram @ alpha
        _, singular, vh = torch.linalg.svd(fitted,full_matrices=False)
        rank = min(self.rank,len(vh))
        basis = vh[:rank]
        self.encode.zero_()
        self.decode.zero_()
        self.encode[:,:rank] = x.T @ alpha @ basis.T
        self.decode[:rank] = basis * scale
        self.offset.copy_(mean*scale)
        if self.temporal:
            slopes=(self.encode[self.base_inputs:]/self.input_scale[self.base_inputs:,None]) @ self.decode
            self.time_weights.copy_(slopes.reshape_as(self.time_weights))
        stride = self.summary_inputs+1
        self.supported_producers = {p for p in range(self.producer_count)
                                   if bool((features[:,(p+1)*stride-1] > 0).any())}
        predicted = x @ self.encode @ self.decode + self.offset
        record = dict(effective_rank=rank,
            centered_prediction_energy_retained=float(singular[:rank].square().sum()/singular.square().sum().clamp_min(1e-20)),
            normalized_fit_mse=float(((predicted/scale-y).square()).mean()),
            supported_producers=sorted(self.supported_producers))
        if self.source_confidence:
            strength = self.confidence_strength(features)
            record["source_confidence"] = dict(norm_limit=float(self.source_norm_limit),
                mean_strength=float(strength.mean()), minimum_strength=float(strength.min()),
                shrunk_scene_fraction=float((strength < 1).float().mean()),
                normalized_fit_mse_after_shrink=float(((predicted*strength[:,None]/scale-y).square()).mean()))
        return record

    @torch.no_grad()
    def fit_kernel(self, x, features, rates):
        """Source RBF plus linear time kernel, with a free intercept.

        The source bandwidth is the median distinct calibration-state squared
        distance. Kernel and time blocks have unit self-energy scales; the
        regularization ratio and rank retain the existing fixed choices.
        """
        centers, inverse = torch.unique(x[:, :self.base_inputs], dim=0, return_inverse=True)
        assert 1 < len(centers) <= len(self.kernel_centers)
        distance = (centers.square().sum(-1)[:, None]+centers.square().sum(-1)[None, :]
                    -2*centers @ centers.T).clamp_min(0)
        upper = torch.triu_indices(len(centers), len(centers), offset=1, device=x.device)
        bandwidth = distance[upper[0], upper[1]].median().clamp_min(1e-6)
        self.kernel_bandwidth_squared.copy_(bandwidth)
        self.kernel_centers.zero_()
        self.kernel_centers[:len(centers)].copy_(centers)
        kernel = (-distance/bandwidth).exp()[inverse][:, inverse]
        column_mean = kernel.mean(0)
        gram = kernel-column_mean[None, :]-column_mean[:, None]+kernel.mean()
        time = x[:, self.base_inputs:]
        if self.temporal:
            gram = gram+time @ time.T/time.shape[-1]
        scale = rates.square().mean((0, 2)).sqrt().clamp_min(1e-7).repeat_interleave(self.width)
        y = rates.flatten(1)/scale
        regularizer = (gram.trace()/len(x)*0.01).clamp_min(1e-6)
        alpha = torch.linalg.solve(gram+regularizer*torch.eye(len(x), device=x.device, dtype=x.dtype), y-y.mean(0))
        fitted = gram @ alpha
        _, singular, vh = torch.linalg.svd(fitted, full_matrices=False)
        rank = min(self.rank, len(vh))
        basis = vh[:rank]
        projected = alpha @ basis.T
        self.encode.zero_()
        self.decode.zero_()
        self.kernel_encode.zero_()
        self.kernel_encode[:, :rank].index_add_(0, inverse, projected)
        self.decode[:rank] = basis*scale
        if self.temporal:
            self.encode[self.base_inputs:, :rank] = time.T @ projected/time.shape[-1]
            slopes = (self.encode[self.base_inputs:]/self.input_scale[self.base_inputs:, None]) @ self.decode
            self.time_weights.copy_(slopes.reshape_as(self.time_weights))
        self.offset.copy_(y.mean(0)*scale-column_mean @ projected @ self.decode[:rank])
        stride = self.summary_inputs+1
        self.supported_producers = {p for p in range(self.producer_count)
                                   if bool((features[:, (p+1)*stride-1] > 0).any())}
        predicted = ((x @ self.encode+self.kernel_features(x[:, :self.base_inputs]) @ self.kernel_encode)
                     @ self.decode+self.offset)
        return dict(effective_rank=rank,
            normalized_fit_mse=float((predicted/scale-y).square().mean()),
            centered_prediction_energy_retained=float(singular[:rank].square().sum()/singular.square().sum().clamp_min(1e-20)),
            supported_producers=sorted(self.supported_producers),
            source_kernel=dict(support_states=len(centers), bandwidth_squared=float(bandwidth),
                regularizer=float(regularizer), bandwidth_rule="median distinct-state squared distance, fitting inputs only",
                time_block="producer-conditioned linear Fourier block, divided by its dimension"))


class QueryTranslator(RidgeTranslator):
    """Source-conditioned affine views of each layer's actual head query.

    Fit lower layers first, then freeze them while calibrating each next layer.
    Query normalization is shared per release and folded into installed views.
    """

    def __init__(self, *, layers=6, width=192, heads=6, target=1, rank=32, layer_clearance=True):
        super().__init__(layers=layers, width=width, target=target, rank=rank,
                         history_scope="all", layer_clearance=layer_clearance)
        assert width % heads == 0
        self.query_affine = True
        self.heads, self.head_dim = heads, width//heads
        self.output_width = (self.head_dim+1)*width
        self.encode = torch.zeros(layers, self.base_inputs, rank)
        self.decode = torch.zeros(layers, rank, self.output_width)
        self.offset = torch.zeros(layers, self.output_width)
        self.register_buffer("query_center", torch.zeros(layers, heads, self.head_dim))
        self.register_buffer("query_scale", torch.ones(layers, heads, self.head_dim))

    def query_view(self, features):
        x = (features-self.center)/self.input_scale
        latent = torch.einsum("...i,lir->...lr", x, self.encode)
        values = torch.einsum("...lr,lro->...lo", latent, self.decode)+self.offset
        values = values.reshape(*features.shape[:-1], self.layers, self.head_dim+1, self.heads, self.head_dim)
        slopes = values[..., 1:, :, :].transpose(-3, -2)/self.query_scale[..., None]
        intercept = values[..., 0, :, :]-(self.query_center[..., None]*slopes).sum(-2)
        if self.layer_clearance:
            active = self.active_layers(features)
            intercept = intercept*active[..., None, None]
            slopes = slopes*active[..., None, None, None]
        intercept = intercept*self.response_strength[:, None, None]
        slopes = slopes*self.response_strength[:, None, None, None]
        return intercept.flatten(-2), slopes

    def rates(self, features):
        return self.query_view(features)[0]

    def forward(self, source):
        intercept, slopes = self.query_view(self.features(source))
        payload = torch.stack((source.payload[..., 0, :], source.payload[..., 1, :]+intercept), dim=-2)
        return replace(source, payload=payload, read_mode="constant_elu", query_coefficients=slopes)

    @torch.no_grad()
    def fit_layer(self, layer, features, coefficients):
        if layer == 0:
            self.center.copy_(features.mean(0))
            self.input_scale.copy_(features.std(0).clamp_min(1e-3))
            stride = self.summary_inputs+1
            self.supported_producers = {p for p in range(self.producer_count)
                                       if bool((features[:, (p+1)*stride-1] > 0).any())}
        x = (features-self.center)/self.input_scale
        y = coefficients.flatten(1)
        mean = y.mean(0)
        gram = x @ x.T
        regularizer = (gram.trace()/len(x)*.01).clamp_min(1e-6)
        alpha = torch.linalg.solve(gram+regularizer*torch.eye(len(x), device=x.device, dtype=x.dtype), y-mean)
        fitted = gram @ alpha
        _, singular, vh = torch.linalg.svd(fitted, full_matrices=False)
        rank = min(self.rank, len(vh))
        self.encode[layer].zero_()
        self.decode[layer].zero_()
        self.encode[layer, :, :rank] = x.T @ alpha @ vh[:rank].T
        self.decode[layer, :rank] = vh[:rank]
        self.offset[layer] = mean
        predicted = x @ self.encode[layer] @ self.decode[layer]+mean
        return dict(layer=layer, effective_rank=rank,
            relative_coefficient_fit_mse=float((predicted-y).square().mean()/y.square().mean().clamp_min(1e-20)),
            centered_prediction_energy_retained=float(singular[:rank].square().sum()/singular.square().sum().clamp_min(1e-20)),
            fitting="shared rank32 source map; calibrated lower layers frozen before collecting this layer's actual queries")

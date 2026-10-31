"""Pre-release query centering for the existing full-item MLP ladder.

Training uses (q-mean)/std. Before export, the first affine bias absorbs
-mean/std @ Wq, so the original reader executes q/std with no new online
operation or state. This changes input conditioning, not the read+score loss,
available information, MLP function family, or interaction-rank constraint.
"""
from collections import Counter
from contextlib import contextmanager
from types import MethodType

import torch

from experiments.ch3_motivation.code import probes
from experiments.ch3_motivation.code.probe_variants import FullItemProbeLadder
from experiments.ch3_motivation.code.resident_probe_fitting import fit_probe_resident
from experiments.ch3_motivation.code.probe_first_affine import (
    interaction as projected_interaction, fitting_ledger, counted_first_affine,
)

MODES = ("rms", "center_std")


@torch.no_grad()
def fit_statistics(probe, model, samples, mode):
    """Same row moments as ProbeLadder; query moments use fitting users only."""
    if mode not in MODES or type(probe) is not FullItemProbeLadder or not probe.cfg.normalize:
        raise ValueError("centering supports only the normalized full-item ladder")
    samples = list(samples)
    total = torch.zeros_like(probe.row_mean)
    squares = torch.zeros_like(total)
    query_total = torch.zeros_like(probe.query_scale)
    query_squares = torch.zeros_like(query_total)
    row_count = query_count = 0
    for sample in samples:
        sample = probes.sample_to_device(sample, probe.row_mean.device)
        source = sample["parent_cache"]
        items = probe._items(model, sample["item_ids"], sample.get("item_embeddings"))
        _, trace = probes.native_score(model, source, sample["candidates"], sample["query_delta"], trace=True)
        for layer, q in enumerate(trace.queries):
            rows = probe._raw_rows(layer, source, items)
            total[layer] += rows.sum((0, 2))
            squares[layer] += rows.square().sum((0, 2))
            query_total[layer] += q.sum((0, 2))
            query_squares[layer] += q.square().sum((0, 2))
        row_count += source.k.shape[1] * source.seq_len
        query_count += sample["candidates"].numel()
    if not row_count or not query_count:
        raise ValueError("query conditioning requires nonempty fitting histories/candidates")
    probe.row_mean.copy_(total / row_count)
    probe.row_scale.copy_((squares / row_count - probe.row_mean.square()).clamp_min(1e-8).sqrt())
    mean = query_total / query_count
    second = query_squares / query_count
    rms = second.clamp_min(1e-8).sqrt()
    std = (second - mean.square()).clamp_min(1e-8).sqrt()
    if not all(bool(torch.isfinite(value).all()) for value in (mean, rms, std)):
        raise FloatingPointError("nonfinite fitting-only query moments")
    probe.query_scale.copy_(rms if mode == "rms" else std)
    probe.statistics_initialized.fill_(True)
    c = probe.cfg
    lh = c.num_layers * c.num_heads
    # Explicit moment arithmetic, omitted by the legacy core's native-query
    # statistics charge. This includes BOTH legacy row moments and new means.
    moment_flops = (3 * row_count * lh * c.row_dim + 3 * query_count * lh * c.head_dim
                    + 6 * lh * c.row_dim + 8 * lh * c.head_dim)
    return dict(format="fit_query_centering_v1", mode=mode, source="fitting_users_only",
        users=len(samples), row_count=row_count, query_count=query_count,
        mean=mean.detach().cpu().tolist(), rms=rms.detach().cpu().tolist(),
        centered_std=std.detach().cpu().tolist(), variance_floor=1e-8,
        input_moment_flops=int(moment_flops), online_mean_subtraction=False,
        export="mean/std folded into first affine biases; original runtime and checkpoint shape")


@contextmanager
def centered_reads(probe, mean, counts, *, subtract_mean=True, factorized_h=False):
    """Temporary training interpretation; no methods survive checkpoint export."""
    names = ("query_correction", "interaction_correction")
    if any(name in probe.__dict__ for name in names):
        raise ValueError("nested query-conditioning method overrides are unsupported")

    def normalized(self, layer, q):
        if subtract_mean:
            phase = "training" if torch.is_grad_enabled() else "validation"
            counts[phase + "_query_subtractions"] += q.numel()
            q = q - mean[layer][None, :, None]
        return q / self.query_scale[layer][None, :, None]

    def query(self, layer, q):
        if self.g is None:
            return q.new_zeros(*q.shape[:-1], 2 * self.cfg.head_dim)
        value = self.g[layer](normalized(self, layer, q))
        return value * self._output_scale(layer, value.ndim)

    def interaction(self, layer, q, rows):
        if self.h is None:
            raise ValueError("interaction correction requires stage fgh")
        q = normalized(self, layer, q)
        if factorized_h:
            value = projected_interaction(self.h[layer], q, rows)
        else:
            nq, nr = q.shape[2], rows.shape[2]
            features = torch.cat((q[:, :, :, None].expand(-1, -1, -1, nr, -1),
                                  rows[:, :, None].expand(-1, -1, nq, -1, -1)), -1)
            value = self.h[layer](features)
        return value * self._output_scale(layer, value.ndim)

    probe.query_correction = MethodType(query, probe)
    probe.interaction_correction = MethodType(interaction, probe)
    try:
        yield
    finally:
        for name in names:
            delattr(probe, name)


@torch.no_grad()
def shift_biases(probe, mean, terms, *, fold):
    """Fold (-) or unfold (+) the same fixed mean through first affine maps."""
    flops = mean.numel()  # one fixed mean/scale division per coordinate
    normalized_mean = mean / probe.query_scale
    for term in terms:
        modules = getattr(probe, term)
        if modules is None:
            raise ValueError("requested absent conditioning term: " + term)
        for layer, module in enumerate(modules):
            weight = module.weights[0][:, :probe.cfg.head_dim]
            shift = torch.einsum("hd,hdo->ho", normalized_mean[layer], weight)
            module.biases[0].add_(shift, alpha=-1 if fold else 1)
            flops += 2 * weight.numel() + shift.numel()
    return int(flops)


def fit_query_conditioned(model, probe, train, validation, *, mode, statistics=None,
                          inherited_g=False, factorized_h=False, **kwargs):
    """Existing fitter/optimizer/loss, with explicitly counted conditioning."""
    if mode not in MODES or type(probe) is not FullItemProbeLadder or not probe.cfg.normalize:
        raise ValueError("unknown full-item query conditioning configuration")
    if "fit_input_statistics" in probe.__dict__:
        raise ValueError("unexpected existing input-statistics override")
    holder = dict(statistics=statistics, computed_here=False)
    counts = Counter()
    if bool(probe.statistics_initialized):
        if statistics is None or statistics["mode"] != mode:
            raise ValueError("initialized probe requires its own fitting-moment record")
        key = "rms" if mode == "rms" else "centered_std"
        wanted = torch.as_tensor(statistics[key], device=probe.query_scale.device, dtype=probe.query_scale.dtype)
        if not torch.equal(probe.query_scale, wanted):
            raise ValueError("existing query scales differ from frozen fitting moments")

    def initialize(self, current, samples):
        holder["statistics"] = fit_statistics(self, current, samples, mode)
        holder["computed_here"] = True

    probe.fit_input_statistics = MethodType(initialize, probe)
    transforms = 0
    backup = None
    try:
        needs_reads = probe.g is not None and (mode == "center_std" or (factorized_h and probe.h is not None))
        if needs_reads:
            if statistics is None or statistics["mode"] != mode:
                raise ValueError("nested terms require their own F fitting statistics")
            mean = torch.as_tensor(statistics["mean"], device=probe.query_scale.device, dtype=probe.query_scale.dtype)
            key = "centered_std" if mode == "center_std" else "rms"
            wanted = torch.as_tensor(statistics[key], device=mean.device, dtype=mean.dtype)
            if not torch.equal(probe.query_scale, wanted):
                raise ValueError("nested query scales differ from the frozen fitting statistics")
            if mode == "center_std" and inherited_g:
                backup = [module.biases[0].detach().clone() for module in probe.g]
                transforms += shift_biases(probe, mean, ("g",), fold=False)
            with fitting_ledger(probe, enabled=factorized_h):
                with centered_reads(probe, mean, counts, subtract_mean=mode == "center_std", factorized_h=factorized_h):
                    report = fit_probe_resident(model, probe, train, validation, **kwargs)
            if mode == "center_std":
                terms = ("g", "h") if probe.h is not None else ("g",)
                transforms += shift_biases(probe, mean, terms, fold=True)
            if mode == "center_std" and inherited_g and report["best_step"] == 0:
                # Exact checkpoint preservation when validation selects the
                # predecessor; avoid a gratuitous FP32 bias round trip.
                with torch.no_grad():
                    for module, value in zip(probe.g, backup, strict=True):
                        module.biases[0].copy_(value)
        else:
            report = fit_probe_resident(model, probe, train, validation, **kwargs)
    finally:
        delattr(probe, "fit_input_statistics")
    stats = holder["statistics"]
    if stats is None or stats["mode"] != mode:
        raise ValueError("fitting completed without its own input moments")
    additions = dict(query_conditioning_input_moment_flops=stats["input_moment_flops"] if holder["computed_here"] else 0,
        query_centering_training_forward_flops=counts["training_query_subtractions"],
        query_centering_validation_forward_flops=counts["validation_query_subtractions"],
        query_centering_backward_flops_estimate=2 * counts["training_query_subtractions"],
        query_centering_parameter_transform_flops=transforms)
    report["arithmetic"].update(additions)
    report["total_calibration_flops_estimate"] += sum(additions.values())
    report["query_conditioning_statistics"] = stats
    report["query_conditioning_execution"] = dict(mode=mode, counts=dict(counts), additions=additions,
        h_first_affine="differentiably_factorized" if factorized_h else "materialized_concatenation",
        inherited_g=inherited_g, original_read_score_objective=True,
        runtime="original MLP after bias folding; no new online operations or persistent query means")
    if factorized_h and probe.h is not None:
        report["query_conditioning_execution"]["first_affine_arithmetic"] = dict(
            training=counted_first_affine(probe.cfg, report["operation_counts"], "training"),
            validation=counted_first_affine(probe.cfg, report["operation_counts"], "validation"),
            already_applied_to_core_arithmetic=True,
            backward_convention="retain2x optimized forward; do not apply these deltas again")
    return report

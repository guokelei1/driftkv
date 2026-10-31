"""Layerwise K/V query conditioning; isolated from frozen full-item runs.

Reuse the same centering/export and fitter. Synchronize each head's output
scale with its actual K/V slice of the all-head row, as the native layerwise
statistics method does. No new loss, rank constraint or runtime operation.
"""
from collections import Counter
from types import MethodType
import torch
from experiments.ch3_motivation.code import probes
from experiments.ch3_motivation.code.probe_variants import LayerwiseKVProbeLadder
from experiments.ch3_motivation.code.resident_probe_fitting import fit_probe_resident
from experiments.ch3_motivation.code.probe_first_affine import fitting_ledger, counted_first_affine
from experiments.ch3_motivation.code.query_centering import MODES, centered_reads, shift_biases

@torch.no_grad()
def fit_statistics(probe, model, samples, mode):
    """Same row moments as ProbeLadder; query moments use fitting users only."""
    if mode not in MODES or type(probe) is not LayerwiseKVProbeLadder or not probe.cfg.normalize:
        raise ValueError("centering supports only the normalized layerwise K/V ladder")
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
    width = probe.cfg.num_heads * probe.cfg.head_dim
    for head in range(probe.cfg.num_heads):
        start = head * probe.cfg.head_dim
        end = start + probe.cfg.head_dim
        probe.kv_output_scale[:, head, :probe.cfg.head_dim].copy_(probe.row_scale[:, 0, start:end])
        probe.kv_output_scale[:, head, probe.cfg.head_dim:].copy_(probe.row_scale[:, 0, width+start:width+end])
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


def fit_query_conditioned(model, probe, train, validation, *, mode, statistics=None,
                          inherited_g=False, factorized_h=False, **kwargs):
    """Existing fitter/optimizer/loss, with explicitly counted conditioning."""
    if mode not in MODES or type(probe) is not LayerwiseKVProbeLadder or not probe.cfg.normalize:
        raise ValueError("unknown layerwise K/V query conditioning configuration")
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

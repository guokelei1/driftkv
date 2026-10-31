"""Execution-only h optimization for the frozen original calibration drivers."""
from pathlib import Path

from experiments.ch3_motivation.code import probe_first_affine as affine
from experiments.ch3_motivation.code.resident_probe_fitting import fit_probe_resident as _resident
from hstu_kvcache.common.artifacts import sha256


def binding():
    return dict(format="differentiable_first_affine_v1", algebra="W[q,x]+b=Wq*q+(Wx*x+b), same nonlinear MLP",
        source_sha256={str(Path(path).resolve()): sha256(Path(path)) for path in (__file__, affine.__file__)},
        cache="no computed features or trainable projections retained between optimizer steps",
        arithmetic="actual first-affine row/query GEMMs and broadcast addition; original2x-forward backward convention",
        input_statistics="charge the original executed row/query moment arithmetic, in addition to its existing native-query charge")


def fit_probe_factorized(model, probe, train, validation, **kwargs):
    train, validation = list(train), list(validation)
    with affine.factorized_reads(probe), affine.fitting_ledger(probe, enabled=True):
        result = _resident(model, probe, train, validation, **kwargs)
    counts, cfg = result["operation_counts"], probe.cfg
    moments = 0
    if counts.get("statistics_native_queries", 0):
        rows = sum(sample["parent_cache"].k.shape[1] * sample["parent_cache"].seq_len for sample in train)
        queries = sum(sample["candidates"].numel() for sample in train)
        if queries != counts["statistics_native_queries"]:
            raise ValueError("original input-statistics query counter changed")
        lh = cfg.num_layers * cfg.num_heads
        moments = int(3 * rows * lh * cfg.row_dim + 2 * queries * lh * cfg.head_dim
                      + 6 * lh * cfg.row_dim + 3 * lh * cfg.head_dim)
    result["arithmetic"]["input_statistics_moment_flops"] = moments
    result["total_calibration_flops_estimate"] += moments
    result["execution_optimizations"] = dict(binding=binding(), active_h=probe.h is not None,
        input_statistics_moment_flops=moments, original_training_objective=True,
        training_first_affine=affine.counted_first_affine(cfg, counts, "training") if probe.h is not None else None,
        validation_first_affine=affine.counted_first_affine(cfg, counts, "validation") if probe.h is not None else None,
        deltas_already_applied_to_core_arithmetic=True)
    return result

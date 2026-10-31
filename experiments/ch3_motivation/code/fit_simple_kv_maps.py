#!/usr/bin/env python3
"""Fit cheap constant/affine KV controls from native source/teacher rows only."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.simple_kv_maps import head_rows, maps_from_tensors
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import torch

LAMBDAS = (1e-4, .01, .1, 1.)


class RowMoments:
    """Sufficient statistics only: O(layer*head*feature²) resident storage."""
    def __init__(self, pairs, width, device):
        kwargs = dict(device=device, dtype=torch.float64)
        self.sx = torch.zeros(pairs, width, **kwargs)
        self.sy = torch.zeros_like(self.sx)
        self.yy = torch.zeros_like(self.sx)
        self.xx = torch.zeros(pairs, width, width, **kwargs)
        self.xy = torch.zeros_like(self.xx)
        self.rows = 0

    @torch.no_grad()
    def add(self, x, y):
        if x.shape != y.shape or x.shape[0] != self.sx.shape[0] or x.shape[2] != self.sx.shape[1]:
            raise ValueError("source/teacher row geometry differs")
        x, y = x.double(), y.double()
        self.sx.add_(x.sum(1)); self.sy.add_(y.sum(1))
        self.yy.add_(y.square().sum(1))
        self.xx.add_(x.transpose(1, 2).bmm(x))
        self.xy.add_(x.transpose(1, 2).bmm(y))
        self.rows += x.shape[1]

    def normalized(self):
        if self.rows <= 0:
            raise ValueError("cannot fit empty row moments")
        xmean, ymean = self.sx/self.rows, self.sy/self.rows
        covariance = self.xx/self.rows - xmean[:, :, None]*xmean[:, None, :]
        xscale = covariance.diagonal(dim1=1, dim2=2).clamp_min(1e-12).sqrt()
        gram = covariance / (xscale[:, :, None]*xscale[:, None, :])
        gram = .5*(gram+gram.transpose(1, 2))
        rhs = (self.xy/self.rows-xmean[:, :, None]*ymean[:, None, :])/xscale[:, :, None]
        return xmean, xscale, ymean, gram, rhs


@torch.no_grad()
def fit_from_moments(moment):
    mean, scale, target, gram, rhs = moment.normalized()
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None]
    candidates = {}
    for regularization in LAMBDAS:
        weight = torch.linalg.solve(gram+regularization*identity, rhs)
        candidates[regularization] = dict(shift=(target-mean).float(), xmean=mean.float(),
            xscale=scale.float(), ymean=target.float(), linear_weight=weight.float())
    return candidates


def prediction_mse(moment, tensors, kind):
    """Float64 moment MSE of frozen float32 affine coefficients (fit report)."""
    pairs, width = moment.sx.shape
    if kind == "source":
        affine = torch.eye(width, device=moment.xx.device, dtype=torch.float64)[None].expand(pairs, -1, -1)
        bias = torch.zeros_like(moment.sx)
    elif kind == "kv_shift":
        affine = torch.eye(width, device=moment.xx.device, dtype=torch.float64)[None].expand(pairs, -1, -1)
        bias = tensors["shift"].double()
    else:
        affine = tensors["linear_weight"].double()/tensors["xscale"].double()[:, :, None]
        bias = tensors["ymean"].double()-tensors["xmean"].double()[:, None].bmm(affine)[:, 0]
    # Output-coordinate SSE includes all intercept cross terms.
    xw = moment.xx.bmm(affine)
    predicted_sum = moment.sx[:, None].bmm(affine)[:, 0]
    sse = (affine*xw).sum(1) + 2*bias*predicted_sum + moment.rows*bias.square()
    sse += moment.yy - 2*(affine*moment.xy).sum(1) - 2*bias*moment.sy
    sse = sse.clamp_min(0.)
    dim = width//2
    return dict(k_mse=float(sse[:, :dim].sum()/(moment.rows*pairs*dim)),
        v_mse=float(sse[:, dim:].sum()/(moment.rows*pairs*dim)),
        pooled_mse=float(sse.sum()/(moment.rows*pairs*width)))


def capture_one(parent, current, histories, uid, cfg, device):
    # The canonical helper constructs unused candidate IDs, but no query is run.
    # Only its paired native source/teacher prefix tensors enter this fitter.
    return capture_samples(parent, current, histories, [uid], cfg, device)[uid]


def run(args):
    if args.gpu != 1:
        raise ValueError("these controls use physical GPU1 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    input_root = ROOT/"experiments/ch3_motivation/inputs_feedback_pool/fit256"
    population_path = input_root/"population.json"
    population = json.loads(population_path.read_text())
    fitting, validation = population["fit"][:100], population["validation"]
    assert len(fitting) == 100 and len(validation) == 16 and len(set(fitting+validation)) == 116
    directory = input_root/f"edge_{args.edge}"
    binding_path = directory/"inputs.json"
    binding = json.loads(binding_path.read_text())
    history_path = directory/binding["history"]["path"]
    if binding["population_sha256"] != sha256(population_path) or sha256(history_path) != binding["history"]["sha256"]:
        raise ValueError("fixed fitting role/history binding changed")
    reference, old = load_variant(args.reference_f, "cpu")
    old_identity = old["identity"]
    if (reference.stage != "f" or old_identity["edge"] != args.edge
            or old_identity["fit_uids"] != fitting or old_identity["validation_users"] != validation
            or old_identity["input_binding_sha256"] != sha256(binding_path)):
        raise ValueError("reference F does not share the fixed native source/history/roles")
    del reference
    parent, current, models = load_models(args.edge, device)
    if models != old_identity["models"]:
        raise ValueError("source/teacher backbones differ from frozen F")
    cfg = binding["settings"]
    sources = {str(path.resolve()): sha256(path) for path in [Path(__file__),
        Path(__file__).with_name("simple_kv_maps.py"), Path(__file__).with_name("common.py"),
        ROOT/"src/hstu_kvcache/evaluation/cost.py",
        ROOT/"src/hstu_kvcache/evaluation/yambda.py",
        ROOT/"src/hstu_kvcache/models/hstu.py", ROOT/"src/hstu_kvcache/models/attention.py"]}
    identity = dict(format="simple_native_kv_control_fit_v1", edge=args.edge, physical_gpu=1,
        models=models, history_cutoff=cfg["release_day"]*86400,
        input_binding_sha256=sha256(binding_path), history_path=str(history_path),
        history_sha256=sha256(history_path), population_sha256=sha256(population_path),
        fit_uids=fitting, validation_users=validation, reference_f=str(args.reference_f.resolve()),
        reference_f_sha256=sha256(args.reference_f), candidates_used=False, labels_used=False,
        objective="row-weighted raw native teacher KV MSE; differs from frozen F read+score alignment",
        source_state="native source model on the same retained causal prefix", regularizations=list(LAMBDAS),
        normalization="fit-row means/std only; source std floor1e-6, teacher centered without output scaling",
        fitting_precision="float64 streaming moments and batched solve; frozen map/runtime float32",
        numerical_sources=sources)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError("fitting identity changed; use another output")
    write_json(config_path, identity)
    checkpoint = args.output/"simple_kv_maps.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["metadata"]["identity"] != identity:
            raise ValueError("saved simple maps differ from requested fit")
        return saved["metadata"]
    histories = restore_pack(history_path)
    heads = current.cfg.num_heads
    pairs, width = len(current.blocks)*heads, 2*current.blocks[0].attn.head_dim
    moments = RowMoments(pairs, width, device)
    all_lengths, fit_rows, transfer_bytes = [], 0, 0
    for index, uid in enumerate(fitting):
        sample = capture_one(parent, current, histories, uid, cfg, device)
        source, target = sample["parent_cache"], sample["teacher_cache"]
        x, y = head_rows(source, heads).to(device), head_rows(target, heads).to(device)
        moments.add(x, y)
        all_lengths.append(source.seq_len); fit_rows += source.seq_len
        transfer_bytes += (source.k.numel()+source.v.numel()+target.k.numel()+target.v.numel())*source.k.element_size()*2
        del sample, source, target, x, y
        if (index+1)%20 == 0:
            print(json.dumps(dict(stage="native_row_moments", users=index+1, total_users=100)), flush=True)
    candidate_tensors = fit_from_moments(moments)
    candidate_maps = {regularization: maps_from_tensors(tensors, heads) for regularization,tensors in candidate_tensors.items()}
    error_names = ["source", "kv_shift"] + [f"linear_{r:g}" for r in LAMBDAS]
    errors = {name: torch.zeros(2, device=device, dtype=torch.float64) for name in error_names}
    validation_rows = 0
    for uid in validation:
        sample = capture_one(parent, current, histories, uid, cfg, device)
        source, target = sample["parent_cache"].to(device), sample["teacher_cache"].to(device)
        outputs = dict(source=source, kv_shift=candidate_maps[LAMBDAS[0]]["kv_shift"](source))
        outputs.update({f"linear_{r:g}": maps["kv_linear"](source) for r,maps in candidate_maps.items()})
        for name, cache in outputs.items():
            errors[name] += torch.stack(((cache.k.double()-target.k.double()).square().sum(),
                (cache.v.double()-target.v.double()).square().sum()))
        validation_rows += source.seq_len; all_lengths.append(source.seq_len)
        transfer_bytes += (source.k.numel()+source.v.numel()+target.k.numel()+target.v.numel())*source.k.element_size()*2
        del sample, source, target, outputs
    denominator = validation_rows*pairs*(width//2)
    validation_mse = {name:dict(k_mse=float(value[0]/denominator), v_mse=float(value[1]/denominator),
        pooled_mse=float(value.sum()/(2*denominator))) for name,value in errors.items()}
    selected = min(LAMBDAS, key=lambda r: validation_mse[f"linear_{r:g}"]["pooled_mse"])
    tensors = candidate_tensors[selected]
    cm = CostModel(current.cfg.hidden_size, len(current.blocks), heads, cfg["attention_backend"])
    prefills = sum(2*(cm if n == 1024 else CostModel(cm.hidden_size, cm.num_layers, cm.num_heads, "torch")).full_cache(n) for n in all_lengths)
    statistics = 4*fit_rows*pairs*width*width + 4*fit_rows*pairs*width
    normalization = 16*pairs*width*width + 16*pairs*width
    solves = len(LAMBDAS)*int((8/3)*pairs*width**3)
    val_predictions = validation_rows*pairs*(width + len(LAMBDAS)*(2*width*width+3*width))
    val_metrics = 3*validation_rows*pairs*width*len(error_names)
    costs = dict(source_and_teacher_prefill_flops=int(prefills), streaming_moments_flops=int(statistics),
        normalization_flops_estimate=int(normalization), regularized_solve_flops_estimate=int(solves),
        validation_map_application_flops=int(val_predictions), validation_error_flops=int(val_metrics),
        source_teacher_cpu_gpu_transfer_bytes=int(transfer_bytes),
        scope="Both native prefills for every100fit/16val user, streaming moments, four solves and actual validation predictions. Solve/normalization are dense arithmetic estimates; no candidate queries, labels, or deployment work.")
    total = sum(v for k,v in costs.items() if k.endswith("flops") or k.endswith("flops_estimate"))
    metadata = dict(identity=identity, selected_regularization=selected,
        fit_users=100, validation_users=16, fit_rows=fit_rows, validation_rows=validation_rows,
        validation_mse=validation_mse,
        fitting_mse={name:prediction_mse(moments,tensors,name) for name in ["source", "kv_shift", "kv_linear"]},
        parameters=dict(kv_shift=pairs*width, kv_linear_coefficients=pairs*(width*width+width),
            kv_linear_input_moment_buffers=2*pairs*width),
        calibration_cost=costs, total_calibration_flops_estimate=int(total),
        cost_scope="The single combined fitter's full preparation/statistics/selection charge is retained. Charging this full shared work to either point is conservative; no standalone lower cost is silently substituted.",
        seconds=time.perf_counter()-started, physical_gpu=1)
    if any(sha256(Path(path)) != value for path,value in sources.items()):
        raise ValueError("simple-map fitting sources changed during execution")
    payload = dict(format="simple_kv_maps_v1", num_heads=heads,
        tensors={name:value.cpu() for name,value in tensors.items()}, metadata=metadata)
    temporary = checkpoint.with_suffix(".pt.writing")
    torch.save(payload,temporary); temporary.replace(checkpoint)
    write_json(args.output/"summary.json", dict(metadata, status="complete", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint)))
    print(json.dumps(dict(status="complete", edge=args.edge, selected_regularization=selected,
        validation_mse=validation_mse, total_calibration_flops_estimate=int(total))), flush=True)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edge",type=int,choices=[2,3],required=True)
    p.add_argument("--reference-f",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=1)
    args = p.parse_args()
    try:
        run(args); write_json(args.output/"exit.json",dict(exit_code=0))
    except BaseException as error:
        write_json(args.output/"exit.json",dict(exit_code=1,error=str(error))); raise

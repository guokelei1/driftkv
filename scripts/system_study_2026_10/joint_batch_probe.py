#!/usr/bin/env python3
"""One unchanged joint optimizer batch: sequential vs same-length execution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch
from design_two.common import RELEASES, setup_device
from design_two.prepare import load_history, load_panel
from evaluate_yambda500m_foundation_raw import load_model
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache
from query_relay_2026_10.adapter import UnifiedRelayAdapter, operation_gate_query
from query_relay_2026_10.calibrate import attach_rollout_tail, prepare, rollout_score
from query_relay_2026_10.session import append_corrected_event
from unified_reuse_2026_09.common import sha256, write_json


def groups_for(rows, users, grouped):
    if not grouped:
        return [[uid] for uid in users]
    groups = defaultdict(list)
    for uid in users:
        row, tail = rows[uid], rows[uid]["rollout"]
        groups[row["cache"].seq_len, row["ids"].shape[1],
               tail["prefix_length"], tail["items"].numel()].append(uid)
    return list(groups.values())


def pack_rows(rows, users):
    if len(users) == 1:
        return rows[users[0]]
    picked = [rows[uid] for uid in users]
    row = dict(cache=HSTUKVCache(torch.cat([r["cache"].k for r in picked], 1),
        torch.cat([r["cache"].v for r in picked], 1), picked[0]["cache"].seq_len))
    for key in ("previous", "item", "item_ids", "ids", "delta", "wanted_logits"):
        row[key] = torch.cat([r[key] for r in picked], 1 if key == "previous" else 0)
    row["rollout"] = dict(prefix_length=picked[0]["rollout"]["prefix_length"],
        **{key:torch.stack([r["rollout"][key] for r in picked])
           for key in ("items", "behaviors", "deltas")})
    return row


def batched_rollout(model, adapter, row):
    """The identical four causal writes, using an independent user dimension."""
    tail, original = row["rollout"], row["cache"]
    prefix = tail["prefix_length"]
    cache = HSTUKVCache(original.k[:, :, :prefix], original.v[:, :, :prefix], prefix)
    state = adapter.advance_codes(cache, row["previous"][:, :, :prefix],
        row["item"][:, :prefix], item_ids=row["item_ids"][:, :prefix])
    for index in range(tail["items"].shape[1]):
        cache, state, _ = append_corrected_event(model, cache, state, adapter,
            tail["items"][:, index:index+1], tail["behaviors"][:, index:index+1],
            tail["deltas"][:, index:index+1], adapter.max_length, code_count=prefix)
    return score(model, cache, row["ids"], row["delta"],
        history_read_fn=adapter.make_history_read(state, code_count=prefix,
            gate_query=operation_gate_query(adapter, model, row["ids"])))[0]


def forward(model, adapter, row, grouped):
    state = adapter.advance_codes(row["cache"], row["previous"], row["item"],
                                  item_ids=row["item_ids"])
    actual = score(model, row["cache"], row["ids"], row["delta"],
        history_read_fn=adapter.make_history_read(state,
            gate_query=operation_gate_query(adapter, model, row["ids"])))[0]
    static = (actual - row["wanted_logits"]).square().mean()
    output = batched_rollout(model, adapter, row) if grouped else rollout_score(model, adapter, row)[0]
    rolled = (output - row["wanted_logits"]).square().mean()
    return .5*static + .5*rolled, static, rolled, actual, output


def execute(model, adapter, rows, users, *, grouped, backward):
    adapter.zero_grad(set_to_none=True)
    outputs = []
    with torch.set_grad_enabled(backward):
        for members in groups_for(rows, users, grouped):
            row = pack_rows(rows, members)
            values = forward(model, adapter, row, grouped=len(members) > 1)
            loss = values[0] * (len(members) / len(users))
            if backward:
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite joint score loss")
                loss.backward()
            outputs.append((members, [v.detach() for v in values]))
    return outputs


def flatten_result(outputs, users):
    values = torch.zeros(3, device=outputs[0][1][0].device)
    logits = {}
    for members, row in outputs:
        values += torch.stack(row[:3]) * (len(members) / len(users))
        for index, uid in enumerate(members):
            logits[uid] = torch.stack((row[3][index], row[4][index])).cpu()
    return values.cpu(), torch.stack([logits[uid] for uid in users])


def difference(reference, actual):
    reference, actual = reference.double().flatten(), actual.double().flatten()
    diff = actual-reference
    return dict(max_abs=float(diff.abs().max()), reference_l2=float(reference.norm()),
        error_l2=float(diff.norm()), relative_l2=float(diff.norm()/reference.norm().clamp_min(1e-30)))


def run(args):
    began = time.perf_counter()
    config = json.loads(args.config.read_text())
    metadata_path = ROOT/config["calibration_metadata"]
    metadata = json.loads(metadata_path.read_text())
    train = metadata["fit_uids"]
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config["seed"])).tolist()
    users = [train[index] for index in order[:config["batch_size"]]]
    if users != config["selected_uids"] or set(users)&set(metadata["validation_uids"]):
        raise ValueError("original first optimizer batch or fit roles changed")
    panel_path = ROOT/config["panel"]
    panel, _, _ = load_panel(panel_path)
    if sha256(panel_path/"binding.json") != metadata["panel_sha256"]:
        raise ValueError("frozen calibration panel differs")
    source = ROOT/metadata["source_pack"]
    checkpoint = metadata_path.with_suffix(".pt")
    record = panel["models"][config["target"]]
    model_path = ROOT/record["checkpoint"]
    for path, digest in ((source, metadata["source_pack_sha256"]),
                         (checkpoint, metadata["weights_sha256"]),
                         (model_path, record["checkpoint_sha256"])):
        if sha256(path) != digest:
            raise ValueError(f"frozen input differs: {path}")
    settings = dict(config=config, config_sha256=sha256(args.config), pid=__import__("os").getpid(),
        source_path=str(source), source_sha256=metadata["source_pack_sha256"],
        adapter_sha256=metadata["weights_sha256"], model_sha256=record["checkpoint_sha256"],
        scope="One original V4 optimizer batch at retained finalV4 parameters; no optimizer step, new fit, quality claim or complete-S estimate",
        sources={str(p.relative_to(ROOT)):sha256(p) for p in
            (Path(__file__), ROOT/"scripts/query_relay_2026_10/calibrate.py",
             ROOT/"scripts/query_relay_2026_10/session.py", ROOT/"scripts/query_relay_2026_10/adapter.py")})
    write_json(args.output/"configuration.json", settings)
    print(json.dumps(dict(status="loading_fixed_inputs", selected_uids=users)), flush=True)
    device = setup_device(args.gpu, 2)
    model, payload = load_model(model_path, device)
    del payload
    model.eval().requires_grad_(False)
    for block in model.blocks:
        block.attn.backend = "triton"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Constructor preserves the original mode's trainable/frozen parameter set.
    adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval()
    adapter.load_state_dict(payload["state_dict"])
    baseline = {n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()}
    del payload
    pack = torch.load(source, map_location="cpu", weights_only=False)
    captured = {uid:pack["rows"][uid] for uid in users}
    histories = load_history(panel_path, requested_uids=users)
    tail_records = attach_rollout_tail(captured, histories, dict(RELEASES)[config["target"]],
        metadata["rollout_steps"], adapter.max_length, pack["records"])
    del pack, histories
    rows = prepare(model, captured, adapter, device)
    del captured
    groups = groups_for(rows, users, True)
    write_json(args.output/"prepared.json",dict(groups=groups, tail_records=tail_records,
        preparation_seconds=time.perf_counter()-began,
        trainable_parameter_names=[n for n,p in adapter.named_parameters() if p.requires_grad]))
    for grouped in (False, True):
        for _ in range(config["warmup_repeats"]):
            execute(model,adapter,rows,users,grouped=grouped,backward=True)
    torch.cuda.synchronize(device)
    checks, references = {}, {}
    for grouped in (False, True):
        values, logits = flatten_result(execute(model,adapter,rows,users,
            grouped=grouped,backward=True),users)
        gradients = {n:None if p.grad is None else p.grad.detach().cpu().clone()
                     for n,p in adapter.named_parameters() if p.requires_grad}
        if not grouped:
            references = dict(values=values,logits=logits,gradients=gradients)
        else:
            per_parameter = {}
            for n,ref in references["gradients"].items():
                got = gradients[n]
                if (ref is None) != (got is None):
                    raise AssertionError(("gradient presence changed",n))
                per_parameter[n] = None if ref is None else difference(ref,got)
            ref = torch.cat([x.flatten() for x in references["gradients"].values() if x is not None])
            got = torch.cat([x.flatten() for x in gradients.values() if x is not None])
            checks = dict(serial_losses=references["values"].tolist(), grouped_losses=values.tolist(),
                loss_order=["half_static_half_rollout","static","rollout"],
                losses=difference(references["values"],values), logits=difference(references["logits"],logits),
                all_gradients=difference(ref,got), per_parameter=per_parameter)
            checks["within_declared_tolerance"] = bool(
                torch.allclose(references["values"],values,atol=config["atol"],rtol=config["rtol"]) and
                torch.allclose(references["logits"],logits,atol=config["atol"],rtol=config["rtol"]) and
                torch.allclose(ref,got,atol=config["gradient_atol"],rtol=config["gradient_rtol"]))
    write_json(args.output/"equivalence.json",checks)
    if not checks["within_declared_tolerance"]:
        raise AssertionError("batched loss/logits/gradient exceed declared tolerance")
    timings = []
    for backward in (True, False):
        for repeat in range(config["timing_repeats"]):
            order = (False,True) if repeat % 2 == 0 else (True,False)
            for grouped in order:
                adapter.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                begin = time.perf_counter()
                result = execute(model,adapter,rows,users,grouped=grouped,backward=backward)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter()-begin
                timings.append(dict(mode="grouped" if grouped else "serial",
                    operation="forward_backward" if backward else "validation_forward",
                    repeat=repeat, wall_seconds=elapsed,
                    peak_increment_bytes=torch.cuda.max_memory_allocated(device)-base,
                    absolute_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
                del result
                print(json.dumps(timings[-1]),flush=True)
    unchanged = all(torch.equal(baseline[n],v.detach().cpu()) for n,v in adapter.state_dict().items())
    if not unchanged or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("probe changed frozen parameters or populated backbone gradients")
    aggregates = {}
    for operation in ("forward_backward","validation_forward"):
        aggregates[operation] = {}
        for mode in ("serial","grouped"):
            picked = [r for r in timings if r["operation"]==operation and r["mode"]==mode]
            values = [r["wall_seconds"] for r in picked]
            aggregates[operation][mode] = dict(median_seconds=statistics.median(values),
                min_seconds=min(values),max_seconds=max(values),
                max_peak_increment_bytes=max(r["peak_increment_bytes"] for r in picked))
        aggregates[operation]["serial_over_grouped_median"] = (
            aggregates[operation]["serial"]["median_seconds"] /
            aggregates[operation]["grouped"]["median_seconds"])
    summary = dict(status="complete",selected_uids=users,groups=groups,equivalent=True,
        unchanged_parameters=unchanged,backbone_gradients_absent=True,optimizer_steps=0,
        timings=timings,aggregates=aggregates,total_probe_seconds=time.perf_counter()-began,
        scope=settings["scope"],
        timing_scope="Warm interleaved complete forward/backward (or no-grad same objective); includes same-N packing, finite checks and zero_grad; excludes loading, initial prepare, diagnostics, optimizer step, checkpoint I/O and complete calibration.",
        memory_scope="Grouped mode retains eight users' differentiable rollout activations together; serial releases each graph after its backward. Peak includes temporary concatenations.")
    write_json(args.output/"summary.json",summary)
    lines=["# V4 单个既定 optimizer 批的合批验证","",settings["scope"],"",
        "8人全部N=1024；同原batch8边界，逐人loss/8反传对比合批均值反传；四步续写保持时间顺序。权重未更新，backbone无梯度。原始loss/logits及逐参数梯度误差见equivalence.json。","",
        "|阶段|逐人中位秒|合批中位秒|逐人/合批|","|---|---:|---:|---:|"]
    for op,a in aggregates.items():
        lines.append(f"|{op}|{a['serial']['median_seconds']:.6f}|{a['grouped']['median_seconds']:.6f}|{a['serial_over_grouped_median']:.3f}|")
    lines += ["",summary["timing_scope"],"",summary["memory_scope"],"",
        "这是同一小批执行方式的可行性验证；不能直接将速度倍数用于完整S、全部层校准或source捕获，也未重新训练发布权重。"]
    (args.output/"summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status="complete",aggregates=aggregates)),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.stdout=sys.stderr=(args.output/"run.log").open("a",buffering=1)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc()))
        raise

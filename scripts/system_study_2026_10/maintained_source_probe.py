#!/usr/bin/env python3
"""Price a V4 calibration snapshot when the deployment already maintained KV.

The retained source pack supplies only the existing native KV/current codes.
Teachers and current item features are generated again. Two idle users must
first materialize their unconsumed V3 update; that work is explicitly charged.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import torch
from design_two.common import RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.design_two.replay import NativeTrajectory
from hstu_kvcache.models import HSTUKVCache
from query_relay_2026_10.capture import load_prior_candidates, mixed_candidates
from query_relay_2026_10.calibration_cost import OperationCounts, update_forward_flops
from query_relay_2026_10.fixed_features import load_bank
from query_relay_2026_10.session import zero_state
from read_correction_2026_09.cost import CostModel
from unified_reuse_2026_09.common import sha256, write_json


def nbytes(*tensors):
    return sum(x.numel()*x.element_size() for x in tensors)


@torch.no_grad()
def run(args):
    began = time.perf_counter()
    config = json.loads(args.config.read_text())
    phases, device = [], None

    @contextmanager
    def phase(name, *, uid=None, scope="", payload_bytes=0):
        if device is not None:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        yield
        if device is not None:
            torch.cuda.synchronize(device)
        phases.append(dict(name=name, uid=uid, wall_seconds=time.perf_counter()-start,
                           scope=scope, payload_bytes=int(payload_bytes)))
        if uid is None:
            write_json(args.output / "phases.json", phases)
            print(json.dumps(dict(status="phase", **phases[-1])), flush=True)

    with phase("fixture_input_load", scope="Retained pack emulates existing source state. Full file load, including unused old teachers, is retained as a separate conservative fixture cost; it is not teacher generation."):
        metadata_path = ROOT/config["calibration_metadata"]
        metadata = json.loads(metadata_path.read_text())
        panel_path = ROOT/config["panel"]
        panel, roles, _ = load_panel(panel_path)
        uids = metadata["fit_uids"] + metadata["validation_uids"]
        if (metadata["target"] != "v4" or metadata["fit_uids"] != roles["fit"][:64]
            or metadata["validation_uids"] != roles["validation"][:16]
            or set(uids) & set(roles["evaluation"])):
            raise ValueError("this probe is fixed to original V4 64fit+16validation")
        source_path = ROOT/metadata["source_pack"]
        if sha256(source_path) != metadata["source_pack_sha256"]:
            raise ValueError("source fixture binding differs")
        pack = torch.load(source_path, map_location="cpu", weights_only=False)
        histories = load_history(panel_path, requested_uids=uids)
        cutover, previous_cutover = dict(RELEASES)["v4"], dict(RELEASES)["v3"]
        records = {row["uid"]:row for row in pack["records"]}
        retained, idle = {}, []
        for uid in uids:
            events = histories.rows[uid]
            stop = int(np.searchsorted(events[0],cutover,side="left"))
            previous = int(np.searchsorted(events[0],previous_cutover,side="left"))
            raw = pack["rows"][uid]
            length = raw["parent"].seq_len
            kept = tuple(np.asarray(x[stop-length:stop]) for x in events)
            if length != min(stop,1024) or int(kept[0][-1]) != records[uid]["last_history_timestamp"]:
                raise ValueError(f"retained causal prefix mismatch: {uid}")
            if not torch.equal(raw["item_ids"], torch.as_tensor(kept[1],dtype=torch.long)[None]):
                raise ValueError(f"retained source IDs mismatch: {uid}")
            if stop == previous:
                idle.append(uid)
                if records[uid]["source_code_counters"].get("zero_code_appended_rows",0):
                    raise ValueError("idle provenance contradicts events")
            retained[uid] = kept
        if idle != config["idle_uids"]:
            raise ValueError(f"idle selection changed: {idle}")
        settings = dict(config=config, pid=os.getpid(), fit_uids=metadata["fit_uids"],
            validation_uids=metadata["validation_uids"], idle_uids=idle,
            source_pack=str(source_path),source_pack_sha256=metadata["source_pack_sha256"],
            predecessor_sha256=metadata["origin_sha256"],
            sources={str(p.relative_to(ROOT)):sha256(p) for p in (
                Path(__file__),ROOT/"scripts/query_relay_2026_10/capture.py",
                ROOT/"src/hstu_kvcache/design_two/replay.py")},
            source_scope="Snapshot at old-release drain before anchorization. Native source plus current codes already maintained for78; two idle code updates are forced by calibration and recomputed. No source history replay.",
            fixture_teacher_scope="Old teachers are read only as later numerical references; none is emitted as a newly generated teacher.")
        write_json(args.output/"configuration.json",settings)
    if args.prepare_only:
        write_json(args.output/"summary.json",dict(status="cpu_prepared_only",settings=settings,
            history_lengths=dict(Counter(len(retained[u][0]) for u in uids)),phases=phases,
            actual_teacher_generation_executed=False, gpu_used=False))
        print(json.dumps(dict(status="cpu_prepared_only",idle=idle,users=len(uids))),flush=True)
        return

    with phase("device_context"):
        device = setup_device(args.gpu,config["threads"])
    with phase("foundation_model_load",scope="Original loader V2/V3/V4. Kept separately: a pipeline sharing already loaded V4 calibration and V3 service models must not count this twice."):
        models = load_models(panel,device,4)
        current, prior_model = models["v4"],models["v3"]
    with phase("prior_adapter_and_bank_load",scope="Existing V3 adapter/immutable bank load and H2D for two idle updates. No new shared-table generation; prior publication preparation belongs to the existing release and is not repeated."):
        prior, bindings = load_prior_candidates(ROOT/config["calibration_root"],["v3"],device)
        adapter = prior["v3"]
        if bindings["v3"]["sha256"] != metadata["origin_sha256"]:
            raise ValueError("prior adapter differs")
        load_bank(adapter.feature_bank_path,adapter.feature_bank_sha256,
                  device=device,dtype=next(adapter.parameters()).dtype)
    costs = OperationCounts()
    teachers = {backend:CostModel(current.cfg.hidden_size,len(current.blocks),
        current.cfg.num_heads,backend) for backend in ("torch","triton")}
    rows, checks, row_ledger = {}, {}, []
    for uid in uids:
        raw, tail = pack["rows"][uid],retained[uid]
        input_bytes = nbytes(raw["parent"].k,raw["parent"].v)
        if uid not in idle:
            input_bytes += nbytes(raw["previous_codes"])
        with phase("source_h2d",uid=uid,payload_bytes=input_bytes,
                   scope="Actual CPU-resident maintained source to GPU transfer; included, not assumed free."):
            parent = raw["parent"].to(device)
            previous = (zero_state(parent,adapter) if uid in idle else
                LightweightValueState(raw["previous_codes"].to(device),parent.seq_len))
        with phase("raw_ids_h2d",uid=uid,payload_bytes=len(tail[1])*8):
            items = torch.as_tensor(tail[1],device=device,dtype=torch.long)[None]
        if uid in idle:
            with phase("idle_current_code_materialize",uid=uid,
                       scope="Discard stored post-capture code and recompute actual pending V3 update from zero; no KV rebuild."):
                old_features = prior_model.lookup_item_embeddings(items)
                previous = adapter.advance_codes(parent,previous,old_features,item_ids=items)
            costs.add("required_idle_current_code_flops",update_forward_flops(adapter,rows=parent.seq_len))
            costs.calls["required_idle_item_lookup_bytes"] += nbytes(old_features)
            costs.calls["required_idle_current_code_rows"] += parent.seq_len
            del old_features
        with phase("source_snapshot_clone",uid=uid,
                   payload_bytes=nbytes(parent.k,parent.v,previous.codes)):
            snapshot = HSTUKVCache(parent.k.detach().clone(),parent.v.detach().clone(),parent.seq_len)
            previous_snapshot = previous.codes.detach().clone()
        with phase("fresh_teacher_compute",uid=uid,payload_bytes=20*len(tail[0]),
                   scope="Exact original NativeTrajectory prefill on current V4 and raw retained gaps; includes event input H2D."):
            # _prefill only consumes retained_events/max_length. Supplying the
            # prefix avoids the NativeTrajectory constructor's source rebuild.
            view = SimpleNamespace(retained_events=tail,max_length=current.cfg.max_seq_len)
            teacher = NativeTrajectory._prefill(view,current)
        teacher_flops = teachers["triton" if teacher.seq_len==current.cfg.max_seq_len else "torch"].full_cache(teacher.seq_len)
        costs.add("teacher_cache_capture_flops",teacher_flops)
        with phase("current_item_features",uid=uid):
            features = current.lookup_item_embeddings(items).detach()
        bytes_out = nbytes(snapshot.k,snapshot.v,previous_snapshot,teacher.k,teacher.v,items,features)
        with phase("snapshot_teacher_features_d2h",uid=uid,payload_bytes=bytes_out):
            row = dict(parent=snapshot.to("cpu"),teacher=teacher.to("cpu"),
                previous_codes=previous_snapshot.cpu(),item_ids=items.cpu(),
                item_features=features.cpu(),
                candidates=torch.tensor(mixed_candidates(uid,tail[1],panel["known_vocab_size"],16,seed=17)),
                query_delta=float(cutover-int(tail[0][-1])))
        rows[uid] = row
        with phase("numerical_reference_check",uid=uid,scope="Validation only, excluded from capture timing totals."):
            differences = {name:float((got-ref).abs().max()) for name,got,ref in (
                ("teacher_k",row["teacher"].k,raw["teacher"].k),
                ("teacher_v",row["teacher"].v,raw["teacher"].v),
                ("previous_codes",row["previous_codes"],raw["previous_codes"]),
                ("item_features",row["item_features"],raw["item_features"]))}
            if not torch.equal(row["parent"].k,raw["parent"].k) or not torch.equal(row["parent"].v,raw["parent"].v):
                raise AssertionError("native source changed")
            for got,ref in ((row["teacher"].k,raw["teacher"].k),(row["teacher"].v,raw["teacher"].v),
                            (row["previous_codes"],raw["previous_codes"]),(row["item_features"],raw["item_features"])):
                torch.testing.assert_close(got,ref,atol=config["atol"],rtol=config["rtol"])
            if not torch.equal(row["candidates"],raw["candidates"]) or row["query_delta"]!=raw["query_delta"]:
                raise AssertionError("query input differs")
            checks[uid]=differences
        row_ledger.append(dict(uid=uid,history_length=parent.seq_len,
            idle_materialize=uid in idle,teacher_flops=teacher_flops,
            idle_code_flops=update_forward_flops(adapter,rows=parent.seq_len) if uid in idle else 0))
        del parent,previous,snapshot,previous_snapshot,teacher,items,features,row
        if len(rows)%16==0:
            print(json.dumps(dict(status="snapshot",users=len(rows),total=len(uids))),flush=True)
    with phase("pack_save_and_hash",scope="Save freshly generated teachers and actual snapshots. No fsync or distributed network guarantee."):
        output_pack = dict(rows=rows,records=pack["records"],settings=settings,
            prior_checkpoints=bindings,required_snapshot_operations=costs.report(origin={}),
            benchmark_scope="Preparation timing artifact only; not a replacement calibration candidate or replay pack.")
        torch.save(output_pack,args.output/"source.pt")
        output_hash = sha256(args.output/"source.pt")
    times, transfers = Counter(),Counter()
    for item in phases:
        times[item["name"]]+=item["wall_seconds"]
        transfers[item["name"]]+=item["payload_bytes"]
    setup={"fixture_input_load","device_context","foundation_model_load","prior_adapter_and_bank_load"}
    generation=sum(t for n,t in times.items() if n not in setup|{"numerical_reference_check"})
    summary=dict(status="complete",phases=phases,phase_seconds=dict(times),phase_payload_bytes=dict(transfers),
        retained_source_generation_seconds=generation,
        conservative_standalone_seconds=generation+sum(times[n] for n in setup),
        required_snapshot_operations=costs.report(origin={}),row_ledger=row_ledger,
        numerical_checks=checks,source_pack_sha256=output_hash,source_pack_bytes=(args.output/"source.pt").stat().st_size,
        whole_experiment_seconds=time.perf_counter()-began,settings=settings,
        scope="One V4 source preparation from already maintained CPU source state. All80 Fresh teacher caches regenerated. Two idle V3 code updates included; previous service trajectory excluded because already executed by the declared deployment. No inference about V5 or training.",
        composition_boundary="Keep conservative replay timing separately. New source generation may replace only its matching V4 source component. Fixture/model/bank loads remain explicit: combine with calibration only after deciding which inputs are shared; never silently delete teacher computation or double-charge model loading.")
    write_json(args.output/"summary.json",summary)
    print(json.dumps(dict(status="complete",capture_seconds=generation,
        standalone_seconds=summary["conservative_standalone_seconds"],idle=idle)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.stdout=sys.stderr=(args.output/"run.log").open("a",buffering=1)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc()))
        raise

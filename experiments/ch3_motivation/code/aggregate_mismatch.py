#!/usr/bin/env python3
"""Collect the five fresh E4 squared-mismatch measurements on CPU."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.ch3_motivation.code.analysis import MismatchAccumulator
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.run_mismatch import _restore_block
from experiments.ch3_motivation.code.run_mismatch_variants import aggregate_records
from hstu_kvcache.common.artifacts import sha256


def merge_exact_csv(paths,output):
    """Copy the original numeric strings and line endings without rounding."""
    header=None
    count=0
    with Path(output).open("wb") as stream:
        for path in paths:
            data=Path(path).read_bytes()
            lines=data.splitlines(keepends=True)
            if header is None:
                header=lines[0]
                stream.write(header)
            elif lines[0]!=header:
                raise ValueError("E4 point CSV headers differ")
            stream.write(b"".join(lines[1:]))
            count+=len(list(csv.DictReader(io.StringIO(data.decode()))))
    return count


def median_ci(values,*,replicates=1000,seed=20261023):
    values=np.asarray(values,dtype=np.float64)
    values=values[np.isfinite(values)]
    if len(values)<512:
        raise ValueError("sub-512 concentration groups are diagnostic only")
    rng=np.random.default_rng(seed)
    draws=[]
    for offset in range(0,replicates,64):
        index=rng.integers(len(values),size=(min(64,replicates-offset),len(values)))
        draws.extend(np.median(values[index],axis=1))
    return dict(users=len(values),median=float(np.median(values)),
        median_uid_ci95=np.percentile(draws,[2.5,97.5]).tolist(),replicates=replicates,seed=seed,
        scope="UID percentile bootstrap of a within-user mismatch concentration median; not quality contribution")


def aggregate(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    base=ROOT/"experiments/ch3_motivation/runs/e4"
    population_path=ROOT/"experiments/ch3_motivation/inputs/population.json"
    population=json.loads(population_path.read_text())
    audited={}
    def ref(path,expected=None):
        path=Path(path).resolve()
        key=str(path)
        if key not in audited:
            audited[key]=dict(path=key,sha256=sha256(path),bytes=path.stat().st_size)
        if expected is not None and audited[key]["sha256"]!=expected:
            raise ValueError("E4 file hash differs: "+key)
        return audited[key]
    def sources(identity):
        for path,digest in identity["source_sha256"].items():
            ref(ROOT/path,digest)
        for path,digest in identity.get("checkpoints",identity.get("models",{})).items():
            ref(ROOT/path,digest)
    edges=[]
    confidence=[]
    point_paths=[]
    for edge in range(1,6):
        folder=base/f"edge_{edge}/attempt_1"
        config_path=folder/"configuration.json"
        config=json.loads(config_path.read_text())
        # Match the original CPU reduction partition for bitwise summary checks.
        torch.set_num_threads(config["settings"]["torch_threads"])
        identity=config["identity"]
        token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        saved=json.loads((folder/"summary.json").read_text())
        if (saved["status"]!="complete" or saved["users"]!=5000 or not saved["paper_result_eligible"]
                or saved["identity_hash"]!=token or identity["users"]!=population["mismatch"]):
            raise ValueError("E4 main population/completion identity differs")
        refs=dict(configuration=ref(config_path,saved["configuration_sha256"]),summary=ref(folder/"summary.json"),
            population=ref(population_path,identity["population_sha256"]),points=ref(folder/"points.csv"),
            numerical_check=ref(folder/"numerical_performance.json"))
        inputs=ROOT/f"experiments/ch3_motivation/inputs/edge_{edge}"
        refs["input_binding"]=ref(inputs/"inputs.json",identity["input_binding_sha256"])
        binding=json.loads((inputs/"inputs.json").read_text())
        for name in ("history","requests"):
            refs[name]=ref(inputs/binding[name]["path"],binding[name]["sha256"])
        if config["settings"]!=binding["settings"] or identity["edge"]!=edge:
            raise ValueError("E4 configuration differs from its edge input binding")
        numerical=json.loads((folder/"numerical_performance.json").read_text())
        if numerical!=saved["performance"] or not numerical["passed"] or numerical["identity_hash"]!=token:
            raise ValueError("E4 scalar/batched numerical check differs")
        sources(identity)
        accumulator=MismatchAccumulator()
        blocks=[]
        expected=[f"part_{offset:06d}" for offset in range(0,5000,256)]
        if [p.stem for p in sorted((folder/"blocks").glob("part_*.json"))]!=expected:
            raise ValueError("E4 primary block marker coverage differs")
        for offset,stem in zip(range(0,5000,256),expected,strict=True):
            part=_restore_block(folder/"blocks",offset,identity["users"][offset:offset+256],token)
            if part is None:
                raise ValueError("missing E4 primary block")
            accumulator.rows.extend(part.rows)
            accumulator._row_values.extend(part._row_values)
            blocks.append({suffix:ref(folder/"blocks"/(stem+suffix)) for suffix in (".json",".parquet",".npz")})
        if len({row["uid"] for row in accumulator.rows})!=5000 or accumulator.summary()!=saved["results"]:
            raise ValueError("E4 primary summary differs from its raw blocks")
        for metric in ("absolute","relative"):
            stats=median_ci([row[metric+"_top10_row_share"] for row in accumulator.rows],seed=20261023+edge)
            confidence.append(dict(edge=edge,metric=metric+"_squared_top10_row_share",**stats))
        edges.append(dict(edge=edge,users=5000,source_window=saved["source_window"],results=saved["results"],
            references=refs,identity_hash=token,block_files=blocks,blocks=20,raw_summary_recomputed_exactly=True))
        point_paths.append(folder/"points.csv")
        del accumulator
    rows=merge_exact_csv(point_paths,output/"points.csv")

    sensitivities=[]
    for name,script,field in (("sensitivity_existing_5000","analyze_mismatch_sensitivity.py","input_hashes"),
                               ("squared_vs_frobenius_5000","analyze_mismatch_norms.py","source_hashes")):
        folder=base/name/"attempt_1"
        value=json.loads((folder/"summary.json").read_text())
        if value["status"]!="complete" or len(value["edges"])!=5:
            raise ValueError("E4 descriptive sensitivity is incomplete")
        ref(ROOT/"experiments/ch3_motivation/code"/script,value["source_sha256"])
        for edge in value["edges"]:
            sources_for_edge=edge[field]
            source=base/f"edge_{edge['edge']}/attempt_1"
            ref(source/"configuration.json",sources_for_edge["configuration"])
            ref(source/"summary.json",sources_for_edge.get("primary_summary",sources_for_edge.get("e4_summary")))
            if "histories" in sources_for_edge:
                ref(ROOT/f"experiments/ch3_motivation/inputs/edge_{edge['edge']}/histories.npz",sources_for_edge["histories"])
            for path,digest in sources_for_edge["blocks"].items():
                ref(ROOT/path,digest)
        sensitivities.append(dict(name=name,summary=ref(folder/"summary.json"),points=ref(folder/"points.csv"),
            role="descriptive analysis of the original squared-mismatch measurements; norm sensitivity never replaces the primary"))

    variants=[]
    for edge in range(1,6):
        folder=ROOT/f"experiments/ch3_motivation/runs/e4_variants/edge_{edge}/attempt_1"
        identity=json.loads((folder/"configuration.json").read_text())
        token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        saved=json.loads((folder/"summary.json").read_text())
        if (saved["status"]!="complete" or saved["users"]!=512 or saved["identity"]!=token
                or identity["users"]!=population["development"] or identity["windows"]!=[256,512,1024]
                or set(identity["users"])&set(identity["fit_users"]) or not identity["attention_divisor_unchanged"]):
            raise ValueError("E4 window/RMS sensitivity identity differs")
        sources(identity)
        ref(ROOT/f"experiments/ch3_motivation/inputs/edge_{edge}/inputs.json",identity["input_binding_sha256"])
        scales=json.loads((folder/"fit_scales.json").read_text())
        if scales["identity"]!=token or scales["users"]!=identity["fit_users"]:
            raise ValueError("E4 RMS normalization used another fitting population")
        blocks=[]
        for window in identity["windows"]:
            records=[]
            directory=folder/f"window_{window}/blocks"
            for offset in (0,256):
                marker_path=directory/f"part_{offset:06d}.json"
                marker=json.loads(marker_path.read_text())
                if marker["identity"]!=token or marker["uids"]!=identity["users"][offset:offset+256]:
                    raise ValueError("E4 window sensitivity block UID identity differs")
                parquet=ref(directory/marker["parquet"],marker["sha256"])
                part=pq.read_table(parquet["path"]).to_pylist()
                if [row["uid"] for row in part]!=marker["uids"] or any(row["window"]!=window for row in part):
                    raise ValueError("E4 window sensitivity row coverage differs")
                records.extend(part)
                blocks.append(dict(marker=ref(marker_path),parquet=parquet))
            expected=dict(all=aggregate_records(records),
                history_under_256=aggregate_records([r for r in records if r["history_rows"]<256]),
                history_256_to_1023=aggregate_records([r for r in records if 256<=r["history_rows"]<1024]),
                history_1024=aggregate_records([r for r in records if r["history_rows"]==1024]))
            if expected!=saved["results"][str(window)]:
                raise ValueError("E4 window/RMS summary differs from its blocks")
        variants.append(dict(edge=edge,users=512,windows=identity["windows"],
            summary=ref(folder/"summary.json"),configuration=ref(folder/"configuration.json"),
            points=ref(folder/"points.csv"),fit_scales=ref(folder/"fit_scales.json"),block_files=blocks,
            role="separate fixed development population; unchanged 1024-window model; normalization fitted only on 64 reserved users"))

    result=dict(status="complete",experiment="E4 fresh K/V mismatch concentration",primary_users_per_edge=5000,
        primary_edges=edges,primary_points=dict(file=ref(output/"points.csv"),rows=rows,numeric_strings_preserved=True),
        primary_definition="sum of squared K/V differences over layers and heads per row; relative divides each layer/row by rebuilt K/V squared norm, floor 1e-12",
        top10_median_uid_confidence=confidence,descriptive_sensitivities=sensitivities,window_rms_sensitivities=variants,
        audit=dict(primary_blocks=100,variant_blocks=30,all_configurations_inputs_sources_models_block_hashes_and_uids_passed=True,
            all_raw_summaries_recomputed_exactly=True,files=list(audited.values())),
        limitations=["Snapshot mismatch concentration is not Layer quality contribution, AUC importance, or an attribution of update benefit.",
            "Squared deviation remains primary; Frobenius/norm sensitivity is a distinct metric and is not substituted to match an expected concentration.",
            "5000-user main and 512-user window/RMS sensitivity populations differ; window comparisons hold the latter users fixed.",
            "Each edge is independently reconstructed immediately before release; E14_partial is retained as an edge label."])
    write_json(output/"summary.json",result)
    write_json(output/"top10_median_uid_confidence.json",confidence)
    (output/"README.md").write_text("# E4 final data aggregation\n\n"
        "`points.csv` joins the five original 5,000-user tables without rewriting their numeric strings. "
        "`summary.json` retains all median/IQR and pooled definitions, exact raw-block reconstruction checks, and file hashes. "
        "`top10_median_uid_confidence.json` adds 1,000 UID-bootstrap median intervals for the primary absolute/relative squared concentration.\n\n"
        "The summary separately links the fixed 512-user, three-window, layer/head RMS sensitivity and the existing 5,000-user descriptive and squared-versus-Frobenius analyses. "
        "Squared mismatch remains the primary metric. These distributions are not quality contributions and do not establish a selective-recomputation AUC curve.\n")
    print(json.dumps(dict(output=str(output),point_rows=rows,primary_blocks=100,variant_blocks=30,audited_files=len(audited))),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/ch3_motivation/runs/e4/final_aggregate/attempt_1")
    args=parser.parse_args()
    aggregate(args.output)

#!/usr/bin/env python3
"""Retain every pre-release configuration and compare matching validation tasks."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def summarize(paths,output):
    rows=[]
    groups=defaultdict(list)
    for path in sorted(paths):
        result=json.loads(path.read_text())
        identity=result.get("identity",{})
        if "best_step" not in result or not identity:
            continue
        chosen=next(value for value in result["history"] if value["step"]==result["best_step"])
        first=result["history"][0]
        name=identity.get("variant","head")
        config=result["config"]
        settings={key:identity.get(key) for key in ("width","depth","learning_rate","read_weight",
            "score_weight","fit_uids","candidate_source","recent_fraction")}
        settings.update(variant=name,include_item=config["include_item"],normalize=config["normalize"],
            method_label=result.get("method_label",result["stage"]))
        # Candidate targets and loss scales differ between candidate rules.
        # Do not rank their absolute validation objectives as one common task.
        task=(identity.get("candidate_source","legacy_listen_exclusive_K"),identity.get("recent_fraction",.5),
            len(identity["fit_uids"]),identity.get("queries",16),identity.get("population_sha256"))
        row=dict(edge=identity["edge"],variant=name,method_label=settings["method_label"],
            width=config["hidden_width"],depth=config["depth"],seed=result["seed"],
            fit_users=result["training_users"],include_item=config["include_item"],normalize=config["normalize"],
            learning_rate=identity.get("learning_rate"),read_weight=identity.get("read_weight"),
            candidate_source=task[0],recent_fraction=task[1],best_step=result["best_step"],
            completed_steps=result["completed_steps"],
            validation_read_mse=chosen["validation_read_mse"],validation_score_mse=chosen["validation_score_mse"],
            common_validation_objective=chosen["validation_read_mse"]+chosen["validation_score_mse"],
            initial_common_validation_objective=first["validation_read_mse"]+first["validation_score_mse"],
            checkpoint_sha256=result.get("checkpoint_sha256"),summary=str(path),summary_sha256=sha256(path))
        row["validation_ratio_to_zero"]=row["common_validation_objective"]/row["initial_common_validation_objective"]
        rows.append(row)
        # UIDs are the same roles across edges, excluded from the displayed key.
        settings.pop("fit_uids")
        groups[(task,json.dumps(settings,sort_keys=True))].append(row)
    aggregates=[]
    for (task,key),values in groups.items():
        by_edge=defaultdict(list)
        for row in values:by_edge[row["edge"]].append(row)
        # Average repeated seeds inside edge before equal-edge aggregation.
        mean=lambda field:sum(sum(r[field] for r in edge)/len(edge) for edge in by_edge.values())/len(by_edge)
        aggregates.append(dict(task=list(task),settings=json.loads(key),edges=sorted(by_edge),
            complete_five_edges=set(by_edge)==set(range(1,6)),runs=len(values),
            equal_edge_mean_validation_objective=mean("common_validation_objective"),
            equal_edge_mean_validation_ratio_to_zero=mean("validation_ratio_to_zero"),
            per_edge={str(edge):values for edge,values in sorted(by_edge.items())}))
    output.mkdir(parents=True,exist_ok=True)
    if rows:
        with (output/"points.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(scope="pre-release validation only; no evaluation AUC read",runs=len(rows),
        groups=aggregates,selection_rule="Compare complete five-edge configurations within a matching task; equal edge means; retain all outcomes and seeds. Candidate-composition changes are separate sensitivity tasks.")
    write_json(output/"summary.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pattern",default="**/f.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.root.glob(args.pattern),args.output)
    print(json.dumps(dict(runs=result["runs"],groups=len(result["groups"]))))

#!/usr/bin/env python3
"""Separate calibration views with observed fitting-feedback quota filled.

History/request bytes, population roles, and their bindings stay identical.
Only the explicitly hashed calibration candidate file changes. The existing
recent_items field is an ordered own+fit-pool candidate sequence in this view,
not a claim that every item is observed for that individual user.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys

from experiments.ch3_motivation.code.common import ROOT, write_json
from experiments.ch3_motivation.code.feedback_candidate_pool import (
    fitting_feedback_catalog, pooled_candidates,
)
from hstu_kvcache.common.artifacts import sha256


def prepare(input_root,output_root,fit_users):
    original=json.loads((input_root/"population.json").read_text())
    fit=original["fit"][:fit_users]
    validation=original["validation"]
    output_root.mkdir(parents=True,exist_ok=True)
    population_path=output_root/"population.json"
    population_path.write_bytes((input_root/"population.json").read_bytes())
    outputs=[]
    for edge in range(1,6):
        source=input_root/f"edge_{edge}";folder=output_root/f"edge_{edge}"
        folder.mkdir(exist_ok=True)
        binding=json.loads((source/"inputs.json").read_text())
        if binding["population_sha256"]!=sha256(population_path):raise ValueError("role source changed")
        for name in ("history","requests"):
            path=source/binding[name]["path"]
            if sha256(path)!=binding[name]["sha256"]:raise ValueError("original raw inputs changed")
            destination=folder/binding[name]["path"]
            if not destination.exists():destination.symlink_to(path.resolve())
            if sha256(destination)!=binding[name]["sha256"]:raise ValueError("new view raw data differs")
        (folder/"inputs.json").write_bytes((source/"inputs.json").read_bytes())
        candidate_path=source/"calibration_feedback_candidates.json"
        candidates=json.loads(candidate_path.read_text())
        if candidates["population_sha256"]!=sha256(population_path):raise ValueError("candidate roles differ")
        catalog=fitting_feedback_catalog(candidates["recent_items"],fit,candidates["known_items"])
        ordered={};coverage={}
        for uid in fit+validation:
            ordered[str(uid)],coverage[str(uid)]=pooled_candidates(uid,candidates["recent_items"][str(uid)],
                catalog,candidates["known_items"],count=16,recent_fraction=1.,seed=17)
        value=dict(candidates,recent_items=ordered,source_candidate_path=str(candidate_path.resolve()),
            source_candidate_sha256=sha256(candidate_path),fitting_pool_uids=fit,validation_uids=validation,
            fitting_feedback_catalog=catalog,
            fitting_feedback_catalog_sha256=hashlib.sha256(json.dumps(catalog).encode()).hexdigest(),
            ordered_candidate_composition=coverage,seed=17,
            selection="own pre-release recent feedback first, then other FITTING users' pre-release observed items; labels never used; validation/evaluation users never contribute to shared pool",
            candidate_sequence_semantics="recent_items is an ordered own+fitting-pool candidate sequence in this view, not exclusively individual own-feedback history",
            uniform_fill="the fit driver takes the requested prefix quota, then adds unique uniform known-item candidates",
            source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),Path(__file__).with_name("feedback_candidate_pool.py"))})
        output=folder/"calibration_feedback_candidates.json"
        if output.exists() and json.loads(output.read_text())!=value:raise ValueError("existing pooled candidate view changed")
        write_json(output,value)
        outputs.append(dict(edge=edge,candidate_path=str(output.resolve()),candidate_sha256=sha256(output),
            original_input_binding_sha256=sha256(source/"inputs.json"),new_input_binding_sha256=sha256(folder/"inputs.json"),
            fit_users=fit_users,feedback_pool_items=len(catalog),
            all_users_have_16_observed_candidates=all(x["actual_feedback_quota"]==16 for x in coverage.values())))
    write_json(output_root/"preparation_summary.json",dict(role="causal calibration-input view only; no quality result",edges=outputs))
    return outputs


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--fit-users",type=int,choices=[64,128,256],required=True)
    args=parser.parse_args()
    outputs=prepare(args.input_root,args.output_root,args.fit_users)
    print(json.dumps(dict(edges=len(outputs),fit_users=args.fit_users,all_quotas_filled=all(x["all_users_have_16_observed_candidates"] for x in outputs))))

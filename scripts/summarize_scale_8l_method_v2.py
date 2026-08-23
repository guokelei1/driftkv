#!/usr/bin/env python3
"""Replace only the invalid v1 quality summary with user-weighted v2."""

import json
from pathlib import Path
import train_p7_theta0 as p7

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/"results/scale_8l_v1/method_summary_v1.json"
QUALITY=ROOT/"results/scale_8l_v1/policy_quality_adjudication_v2.json"
OUTPUT=ROOT/"results/scale_8l_v1/method_summary_v2.json"

def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    value=json.loads(V1.read_text()); quality=json.loads(QUALITY.read_text())
    value["status"]="scale_8l_frozen_method_full_replay_complete_user_weighted_v2"
    value["quality"]=quality["cells"]
    value["inputs"]["quality_sha256"]=p7.sha256_file(QUALITY)
    value["supersedes"]="results/scale_8l_v1/method_summary_v1.json"
    value["v1_raw_scores_or_assignments_invalidated"]=False
    OUTPUT.write_text(json.dumps(value,indent=2)+"\n"); print(json.dumps(value,indent=2))

if __name__=="__main__": main()

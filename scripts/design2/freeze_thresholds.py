"""Fix thresholds from residual-calibration users before inspecting development."""

import hashlib
import json

import pandas as pd
from design.data import ROOT
from design2.report_detection import SCORES, quantile, weights


def main():
    protocol_path=ROOT/"configs/design2/detection_01.json"
    protocol=json.loads(protocol_path.read_text())
    runs=["canary16_01"]+[f"detection01_residual_calibration_{s:04d}" for s in (16,128,256,384)]
    inputs=[ROOT/"results/design2"/r/f"states_m{t}.parquet" for r in runs for t in protocol["targets"]]
    cal=pd.concat([pd.read_parquet(p) for p in inputs],ignore_index=True)
    assert set(cal.uid)==set(protocol["groups"]["residual_calibration"])
    assert set(cal.role)=={"residual_calibration"}
    assert not cal.duplicated(["uid","target","state_ordinal"]).any()
    cal["short_history"]=1/cal["count"]
    w=weights(cal)
    values={s:{str(q):quantile(cal[s].to_numpy(),w,q) for q in protocol["acceptance_quantiles"]} for s in SCORES}
    out=ROOT/"configs/design2/thresholds_01.json"
    if out.exists():
        raise FileExistsError(out)
    record=dict(status="fixed_from_calibration_before_development_analysis",thresholds=values,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        calibration_inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        calibration_users=int(cal.uid.nunique()),calibration_states=len(cal),development_outputs_read=False)
    out.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(dict(path=str(out),users=record["calibration_users"],states=len(cal))))


if __name__=="__main__":
    main()

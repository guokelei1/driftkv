from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def test_m1_contract_is_frozen_and_blind_safe():
    value=yaml.safe_load((ROOT/"configs/contracts/scale_8l_m1_seed17_v1.yaml").read_text())
    assert value["status"]=="frozen_before_M1_theta0_training"
    assert value["scope"]["model"]=="M1_shared_N_R_F"
    assert value["scope"]["seed"]==17
    assert value["scope"]["qualification_or_theta3"]=="prohibited"

def test_m1_queue_uses_all_four_gpus_and_stops_at_hs():
    text=(ROOT/"scripts/run_scale_8l_m1_seed17.py").read_text()
    assert '"--nproc_per_node=4"' in text
    assert '"m1"' in text
    assert "train_theta3" not in text.lower()
    assert '"theta3_access":False' in text
    assert "eval_scale_8l_m1_hs_raw.py" in text

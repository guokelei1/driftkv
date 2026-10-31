#!/bin/bash
set -e
cd /home/gkl/work/evokv
export PYTHONPATH=src:.
exec /home/gkl/miniconda3/bin/python -m experiments.recflow_continuous_chain.code.run_train_pair --run /home/gkl/work/evokv/experiments/recflow_continuous_chain/training_runs/m1_d29_listwise03 > /home/gkl/work/evokv/experiments/recflow_continuous_chain/training_runs/m1_d29_listwise03/driver.log 2>&1

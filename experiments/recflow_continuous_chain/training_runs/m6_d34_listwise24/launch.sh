#!/bin/bash
set -e
cd /home/gkl/work/evokv
export PYTHONPATH=/home/gkl/work/evokv/src:/home/gkl/work/evokv
exec /home/gkl/miniconda3/bin/python -m experiments.recflow_continuous_chain.code.run_train_pair --run /home/gkl/work/evokv/experiments/recflow_continuous_chain/training_runs/m6_d34_listwise24 > /home/gkl/work/evokv/experiments/recflow_continuous_chain/training_runs/m6_d34_listwise24/driver.log 2>&1

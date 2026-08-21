# P8 monitoring handoff (completed)

P8 已于 2026-08-21 完成，所有 R0/R1/R2 artifact 已封存。此文件仅保留当时的运行手册；
当前执行入口是 [P9 计划](p9_plan.md)，不得用本文件重新启动 P8 或自动启动 tomography/controller。

P8 is frozen to the following execution order:

```text
R0 blocking control (passed)
  -> R1 edge1: 6 train, 6 raw, seal, H/S adjudication
  -> R1 edge2: 6 train, 6 raw, seal, H/S adjudication
  -> R2 edge1: 6 train, 6 raw, seal, H/S adjudication
  -> stop for human review
```

Use the conservative driver from the repository root:

```bash
PYTHONPATH=src python scripts/run_p8_pipeline.py --status
PYTHONPATH=src python scripts/run_p8_pipeline.py --next-command --device cuda:0
PYTHONPATH=src python scripts/run_p8_pipeline.py --run-next --device cuda:0
PYTHONPATH=src python scripts/run_p8_pipeline.py --next-wave-commands
PYTHONPATH=src python scripts/run_p8_pipeline.py --launch-wave \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

`--run-next` runs at most one missing job. `--launch-wave` starts at most one
same-phase job per listed device and writes detached-process output under
`results/p8/monitor_logs/`. Both refuse to launch while another P8 training or
raw-evaluation process is active. Parallel GPU launches therefore remain an
explicit action.

The driver enforces these boundaries:

- all seeds 17/37/71 are retained;
- no seed is selected by H, S, or controller results;
- raw scores are sealed before metrics are computed;
- a rejected release or missing parent adjudication blocks automatic progress;
- R2 follows completion of the two-edge R1 chain;
- completion stops after the H/S matrix; tomography and controller work remain
  unauthorized.

Do not delete or overwrite an output directory to rerun a job. If a process
fails after creating a partial directory, stop and request a human audit of that
exact path.

The execution-level regression suite is:

```bash
PYTHONPATH=src pytest -q \
  tests/test_p8_release_contract.py \
  tests/test_p8_execution_handoff.py
```

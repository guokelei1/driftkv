# Read-correction exploration — September 2026

This directory now retains shared fitting, reading, data and cost utilities.
The v1/v2/v3 and Max-diagnostic runners and the independent v4 queue are retired.
Current execution uses `scripts/read_correction_v5/`,
`scripts/read_correction_motivation/` and the v4 calibration/evaluation modules.
Current Motivation selects nonlinear Q-v5 `cross_phi_joint8` C128/256/512
(45 points) plus nonlinear H-v4 `map_all` C64/128/256 (45 points); see
[the final results](../../results/read_correction_2026_09/motivation_final/README.md).

The descriptions below record the original v1 methods. That authorized
four-GPU exploration completed in September 2026 on the existing 3,000-user
panels. Calibration UIDs remain separate.

Two independently stored methods share only the model, history replay,
aggregation and cost primitives:

- `query_only/`: a shared query-affine read correction.
- `history_conditioned/`: explicit additional query-dependent processing of
  the user's full inherited K/V history inside correction.

The latter examines recovery potential with extra work. This experiment does
not implement Design summaries, incremental state maintenance or compression.
Weak edges and negative points remain; there is no quality admission threshold.

See the [design record](../../docs/design/read_correction_probe_2026_09.md) and
[prospective candidate settings](../../configs/read_correction_2026_09/plan.json).
Method outputs live separately under
`results/read_correction_2026_09/<method>/development/<revision>/<scale>/<edge>/`.
This directory reuses completed four-baseline inputs without modifying them.

The former `run.py`, `launch.sh` and `run_in_tmux.sh` entry points are retired.
Historical source hashes and results remain unchanged. Old queues cannot
resume against those hashes in the cleaned source tree; saved results remain
available to the current figure generators.

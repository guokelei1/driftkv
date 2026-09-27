# Max V2 startup observation

Read-only snapshot at 2026-09-19 22:50:23 +08:00. See `summary.json` for exact
source snapshot hashes, interval timing, recovery saves, cache counts and selected
compiled-source hashes. No GPU work, active execution-source change or current
formal evaluation-quality read was performed.

At this snapshot the Max training log reached step 10,250. The shared Triton cache
contained 3,060 newly written matching HSTU kernels since launch: 1,020 forward,
1,020 query-backward and 1,020 key/value-backward variants. Every source checked
matches H10/BF16/D32; 955 forward variants use one query with varying context
lengths and 65 use complete sequences. Actual M/N/QSTART and strides are constexpr
in the implementation. The early-block cache counts and decreasing step times
support shape-compilation warmup as a material early-training cost.

Cache writes are not an exact compilation-time profile or process ownership
record. Approximately 30.5 minutes between launch and the first new kernel artifact
remain unassigned startup. V1 already took 35.36 minutes from launch to step 250;
V2 took 40.24 minutes. Neither whole duration should be called compilation.

The matched 12-step, four-rank/global80 comparison is Torch 0.983963 seconds/step,
cold auto 3.959460, cached auto 0.782497: cached step time falls 20.47%, throughput
rises 25.75%. This scope includes batch collation, training computation, optimizer
and communication, and excludes initialization and checkpoint saves. Later formal
interval timings use later batches within the V2 window, not a controlled backend
comparison; the older V1 run also uses a different training window. Cache files
and the active log may change after this snapshot.

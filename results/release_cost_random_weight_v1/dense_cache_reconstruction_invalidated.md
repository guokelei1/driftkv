# Invalidated dense-cache reconstruction timings

The first measurement revision timed Reuse through
`HSTU.forward_with_cache`, which concatenated old and new K/V at every layer
to reconstruct a dense full cache. That adds a prefix-sized K/V copy which is
not part of the intended append-only serving operation. These values are kept
for audit only and must not be compared with the append-only results in
`report.md`.

| Configuration | Reuse card-hours | Recompute card-hours | Ratio |
| --- | ---: | ---: | ---: |
| 4L/context512/H128, 2 heads | 0.532 | 0.658 | 1.24x |
| 6L/context1K/H256 | 0.738 | 5.548 | 7.52x |
| 8L/context2K/H512 | 1.672 | 50.126 | 29.99x |
| 16L/context4K/H512 | 5.784 | 368.530 | 63.71x |

The 4L point also used two heads and was not identical to the formal Small
architecture, which uses four. Both issues are corrected by the append-only
revision.

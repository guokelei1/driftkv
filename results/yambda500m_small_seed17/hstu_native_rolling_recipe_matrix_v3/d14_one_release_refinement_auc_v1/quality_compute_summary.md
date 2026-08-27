# One-release quality and theoretical compute

The headline compute uses a conservative ideal causal-attention FLOP count. It assumes a 512-position full state and includes per-user CAST plus compact Current PATCH and SCALE. Reuse state conversion is 0 FLOPs; state I/O and common serving work are outside this boundary.

| Edge | Recompute AUC | Reuse AUC | Our AUC | Our - Reuse (pp) | Our gain retained | Reuse harm recovered | Recompute compute | Reuse compute | Our compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 -> v1 | 0.681769 | 0.678709 | 0.681432 | +0.272304 | +97.2% | +89.0% | 100.0% | 0.0%* | 48.0% |
| v1 -> v2 | 0.670493 | 0.668496 | 0.671611 | +0.311450 | +117.9% | +156.0% | 100.0% | 0.0%* | 48.0% |
| v2 -> v3 | 0.615001 | 0.613514 | 0.614609 | +0.109449 | +87.3% | +73.6% | 100.0% | 0.0%* | 48.0% |
| v3 -> v4 | 0.620994 | 0.619054 | 0.619960 | +0.090555 | -123.2% | +46.7% | 100.0% | 0.0%* | 48.0% |
| v4 -> v5 | 0.551858 | 0.551221 | 0.548563 | -0.265765 | -49.4% | -417.0% | 100.0% | 0.0%* | 48.0% |

Conservative causal FLOPs per full user: Recompute 0.625 GFLOPs; Our 0.301 GFLOPs (48.0% of Recompute, 52.0% lower).

The current dense PyTorch attention graph gives Recompute 0.893 GFLOPs and Our 0.305 GFLOPs (34.1% of Recompute). This is a secondary implementation-graph count, not a runtime claim.

`*` Reuse has zero release-time neural recomputation in this table, not zero state read, network, storage, or serving cost. The v3 -> v4 retained ratio has a near-zero Full-only gain denominator; v4 -> v5 is a genuine negative edge.

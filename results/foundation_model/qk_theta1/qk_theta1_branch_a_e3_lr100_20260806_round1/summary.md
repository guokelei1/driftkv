# QK theta1 Branch A e3 Summary

- Target gap found: `True`
- Training targets: `247,638`
- Optimizer-updated embedding rows: `357,036`
- Qualification/final consumed: `False/False`

## Update-local full-catalog rolling-next-item, all participants

| Metric | Reuse | Recompute | Relative gap % | CI positive |
|---|---:|---:|---:|---:|
| cross_entropy | 11.186713 | 11.182629 | 0.0365027 | True |
| ndcg_at_10 | 0.0053461224 | 0.0055328907 | 3.49353 | True |
| mrr | 0.005720009 | 0.005841971 | 2.1322 | True |
| hit_rate_at_10 | 0.011199277 | 0.011538268 | 3.02691 | True |

## Predeclared gates

- Full-catalog alignment: `aligned_protocol_found`
- Candidate protocol sweep: `no_admitted_protocol`
- Candidate selection remains manual after this complete development round.

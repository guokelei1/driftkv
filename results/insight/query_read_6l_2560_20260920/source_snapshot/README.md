# Retained probability-recovery table

These exact source and table copies preserve the 256-calibration / 2560-evaluation
query-only versus query-plus-response presentation before the paper switches to
real-feedback AUC. All original results remain in `../diagnostic/`.

The original renderer remains at `figures/src/shared_read_insight.py`. From the
repository root, regenerate the old table without replacing the current paper
table with:

```sh
python figures/src/shared_read_insight.py --output /tmp/insight_query_probability.tex
```

`shared_read_insight.py` here is an unchanged archival source copy; its relative
repository-root calculation assumes the original `figures/src/` location.

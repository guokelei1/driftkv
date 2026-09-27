# Retained cache-summary table snapshot

The renderer and LaTeX table were copied byte-for-byte before the user redirected
the Insight question to sharing and user-specific read information. All original
diagnostic outputs remain in `../diagnostic/`; this experiment is not a selection
candidate for the new `user_information_6l_20260920` diagnostic.

- `shared_read_insight.py` SHA256: `b10e8cdc9613a97fb96f4c50e5d76f14fdc5632e1d431819c4ae8d8a65431ff6`
- `insight_shared_read.tex` SHA256: `b1fdaa876851898caefa0971ca1aba4d1c253f8cfc2e91625f620a97d3dfc174`

From the repository root, the saved renderer reproduces the table values with:

```bash
python results/insight/shared_read_6l_20260920/source_snapshot/shared_read_insight.py \
  --result results/insight/shared_read_6l_20260920/diagnostic \
  --output /tmp/insight_shared_read_original.tex
```

Relocating the unchanged renderer shortens only its source-path comments. For
byte-identical comments as well, import the saved module, set `ROOT = Path.cwd()`,
and call `render_table(*read_results(Path("results/insight/shared_read_6l_20260920/diagnostic")))`.
This byte-identical comparison was checked against the retained table.

"""hstu_kvcache: low-cost drift estimation for HSTU KV cache under streaming updates.

Package layout mirrors docs/08_core_insights_and_roadmap.md:
  models/    - HSTU architecture (pointwise attention, first-class KV output)
  data/      - KuaiRand streaming-trace + ML1m generative-rec loaders
  streaming/ - streaming training -> theta checkpoints -> dtheta -> oracle recompute
  drift/     - drift estimation (naive per-user JVP, Fisher-spectrum, cross-user low-rank)
  serving/   - three-state cache decision (reuse/migrate/recompute)
"""

__version__ = "0.0.1"

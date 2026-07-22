"""HSTU K/V cache migration across streaming model versions.

Package layout mirrors docs/08_core_insights_and_roadmap.md:
  models/    - HSTU architecture (pointwise attention, first-class KV output)
  data/      - KuaiRand streaming-trace + ML1m generative-rec loaders
  streaming/ - next-item streaming training and model-version checkpoints
  migration/ - structure-aware state capture and K/V migration operators
"""

__version__ = "0.0.1"

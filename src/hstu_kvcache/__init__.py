"""HSTU K/V cache migration across streaming model versions.

Package layout mirrors the active first-stage route in
docs/01_foundation_model_training_plan.md; historical system design is archived under
docs/archive/:
  models/    - HSTU architecture (pointwise attention, first-class KV output)
  data/      - KuaiRand streaming-trace + ML1m generative-rec loaders
  streaming/ - next-item streaming training and model-version checkpoints
  migration/ - structure-aware state capture and K/V migration operators
"""

__version__ = "0.0.1"

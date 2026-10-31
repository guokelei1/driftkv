"""Execution provenance for attention backends, separate from model architecture."""

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def attention_backend_info() -> dict:
    """Record the requested runtime policy and its code without loading CUDA.

    The policy is not a claim that every attention call uses Triton: dispatch
    still depends on tensor shapes, dtype, dropout, masks and compilation.
    """
    try:
        triton_version = version("triton")
    except PackageNotFoundError:
        triton_version = None
    directory = Path(__file__).resolve().parent
    sources = ("attention.py", "triton_attention.py", "backend_info.py")
    return {
        "policy": os.environ.get("EVOKV_ATTENTION_BACKEND", "auto"),
        "policy_scope": "Requested backend; per-call dispatch may retain PyTorch.",
        "torch_version": str(torch.__version__),
        "triton_version": triton_version,
        "cuda_version": torch.version.cuda,
        "source_sha256": {
            f"src/hstu_kvcache/models/{name}": (
                hashlib.sha256((directory / name).read_bytes()).hexdigest()
                if (directory / name).is_file() else None
            )
            for name in sources
        },
    }

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

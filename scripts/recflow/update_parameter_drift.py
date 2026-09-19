#!/usr/bin/env python3
"""CPU-only parameter drift from the same A3 to the predeclared daily B1 models."""

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hstu_kvcache.models.hstu import HSTUConfig  # noqa: E402
from hstu_kvcache.recflow.model import RecFlowGenerator  # noqa: E402

DEVELOPMENT = ROOT / "results/recflow/development"
REFERENCE = DEVELOPMENT / "window_6l_full_seed17/A/A_epoch3/checkpoint.pt"
CHECKPOINTS = [DEVELOPMENT / "window_6l_daily_seed17/B_day19/B_epoch1/checkpoint.pt"] + [
    DEVELOPMENT / f"window_6l_daily_lr_seed17/{setting}/B/B_epoch1/checkpoint.pt"
    for setting in ("lr1e4", "lr3e5")
]
GROUPS = {
    "item_embedding": "backbone.item_emb.weight: entire frozen initial table, including PAD/OOV. Its tied leaf-output use is counted once.",
    "other_backbone": "All other named parameters below backbone, including any unchanged/unused query and readout parameters.",
    "generation_other": "Non-backbone parameters: REC token, category embeddings/heads and leaf bias; excludes the shared item table.",
}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load(path):
    # Trusted repository-produced files. mmap does not materialize AdamW arrays;
    # discard their references immediately and never move any tensor to CUDA.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return payload["model"], payload["configuration"], payload["phase"], payload["epoch"]


def parameter_layout(config, state):
    # Build only metadata-sized parameter objects, using the actual architecture
    # and its named_parameters() deduplication rather than state_dict heuristics.
    with torch.device("meta"):
        model = RecFlowGenerator(HSTUConfig(**config["model"]), state["item_paths"],
                                 history_categories=config.get("history_categories", False))
    unique = dict(model.named_parameters())
    aliases = dict(model.named_parameters(remove_duplicate=False))
    layout = {}
    for name, parameter in unique.items():
        group = ("item_embedding" if name == "backbone.item_emb.weight" else
                 "other_backbone" if name.startswith("backbone.") else "generation_other")
        layout[name] = dict(group=group, shape=list(parameter.shape), numel=parameter.numel())
    excluded = sorted(set(state) - set(aliases))
    duplicates = sorted(set(aliases) - set(unique))
    return layout, excluded, duplicates


def measure(reference, current, layout, chunk_elements):
    totals = {name: dict(reference_squared_l2=0.0, delta_squared_l2=0.0,
                         max_abs_delta=0.0, numel=0, parameter_tensors=0)
              for name in (*GROUPS, "overall")}
    for name, spec in layout.items():
        old, new = reference[name], current[name]
        if list(old.shape) != spec["shape"] or new.shape != old.shape:
            raise ValueError(f"Parameter shape changed: {name}")
        old, new = old.reshape(-1), new.reshape(-1)
        squared_reference = squared_delta = maximum = 0.0
        for start in range(0, old.numel(), chunk_elements):
            baseline = old[start:start + chunk_elements].to(torch.float64)
            delta = new[start:start + chunk_elements].to(torch.float64) - baseline
            squared_reference += baseline.square().sum().item()
            squared_delta += delta.square().sum().item()
            maximum = max(maximum, delta.abs().max().item())
        for group in (spec["group"], "overall"):
            total = totals[group]
            total["reference_squared_l2"] += squared_reference
            total["delta_squared_l2"] += squared_delta
            total["max_abs_delta"] = max(total["max_abs_delta"], maximum)
            total["numel"] += old.numel()
            total["parameter_tensors"] += 1
    for total in totals.values():
        before, difference = total.pop("reference_squared_l2"), total.pop("delta_squared_l2")
        if not math.isfinite(before + difference + total["max_abs_delta"]):
            raise ValueError("Nonfinite parameter norm or drift")
        total.update(reference_l2=math.sqrt(before), delta_l2=math.sqrt(difference),
                     relative_l2=math.sqrt(difference / before) if before else None)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--output", type=Path,
                        default=DEVELOPMENT / "window_6l_daily_lr_seed17/parameter_drift")
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or min(args.chunk_elements, args.threads) < 1:
        parser.error("Use a fresh output directory and positive chunk/thread counts.")
    paths = [args.reference.resolve(), *(path.resolve() for path in args.checkpoints)]
    for path in paths:
        if not path.is_file():
            parser.error(f"Checkpoint not ready: {path}")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    reference_sha = digest(paths[0])
    reference, config, phase, epoch = load(paths[0])
    if (phase, epoch) != ("A", 3):
        raise ValueError("This diagnostic starts from the declared A3 checkpoint")
    layout, excluded, duplicates = parameter_layout(config, reference)
    rows = []
    for path in paths[1:]:
        checkpoint_sha = digest(path)
        current, saved, phase, epoch = load(path)
        for key in ("model", "catalog_sha256", "cohort_sha256", "history_categories", "seed"):
            if saved[key] != config[key]:
                raise ValueError(f"Reference/current {key} differs")
        if ((phase, epoch) != ("B", 1) or saved["training_window_days"] != [19, 19]
                or saved["source_checkpoint_sha256"] != reference_sha):
            raise ValueError("Comparison must be daily B1 continued from this exact A3")
        measured = measure(reference, current, layout, args.chunk_elements)
        rows.append(dict(checkpoint=str(path), sha256=checkpoint_sha, bytes=path.stat().st_size,
            learning_rate=saved["learning_rate"], training_days=saved["training_window_days"],
            phase=phase, epoch=epoch, source_checkpoint_sha256=saved["source_checkpoint_sha256"],
            optimizer_state_steps_before=saved["optimizer_state_steps_before"], groups=measured))
        print(json.dumps(dict(checkpoint=str(path), learning_rate=saved["learning_rate"],
                              groups=measured), allow_nan=False), flush=True)
        del current
        gc.collect()
    report = dict(scope="CPU parameter drift only; not KV drift, cache compatibility or quality evidence.",
        training_seed=config["seed"],
        reference=dict(checkpoint=str(paths[0]), sha256=reference_sha, bytes=paths[0].stat().st_size,
                       learning_rate=config["learning_rate"], model=config["model"],
                       catalog_sha256=config["catalog_sha256"], cohort_sha256=config["cohort_sha256"]),
        formula="relative_l2 = sqrt(sum((current-A3)^2) / sum(A3^2)); max_abs_delta = max(abs(current-A3)). Zero reference norm yields null relative_l2.",
        accumulation="FP64 subtraction and squared sums in bounded chunks; group totals sum squared norms, not per-tensor relative norms.",
        grouping=GROUPS, parameter_layout=layout, excluded_non_parameter_state_keys=excluded,
        duplicate_parameter_aliases_counted_once=duplicates, comparisons=rows,
        chunk_elements=args.chunk_elements, threads=args.threads, torch_version=str(torch.__version__),
        seconds=time.monotonic() - started, source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))})
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), comparisons=len(rows), seconds=report["seconds"])))


if __name__ == "__main__":
    main()

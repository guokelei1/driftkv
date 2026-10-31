#!/usr/bin/env python3
"""Frozen feedback checkpoints: observed-positive retrieval against 1000 distractors."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.recflow.feedback import FeedbackData, grouped_scores
from hstu_kvcache.recflow.metrics import random_expected_metrics
from experiments.archive.baseline_studies.recflow_feedback_update_analysis import digest


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save_rows(path, **rows):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez(stream, **rows)
    os.replace(temporary, path)


def make_panel(data, known, day, limit, include_all=False):
    eligible = data.indices(2, day, day)
    selected, positives = [], []
    for index in eligible:
        req = data.requests[index]
        events = data.source_events[req["event_start"]:req["event_stop"]]
        values = np.unique(events["raw_item_id"][(events["behavior"] & 1) != 0])
        if len(values) or include_all:
            selected.append(index)
            positives.append(values)
    # Deterministic subset, independent of predictions; all-OOV positive requests stay.
    rng = np.random.default_rng(20261022 + day)
    chosen = np.sort(rng.choice(len(selected), size=min(limit, len(selected)), replace=False))
    indices = np.asarray(selected)[chosen]
    positives = [positives[i] for i in chosen]
    candidates = np.zeros((len(indices), 1024), dtype=np.int64)
    known_counts = np.zeros(len(indices), dtype=np.int32)
    positive_counts = np.asarray([len(p) for p in positives], dtype=np.int32)
    for row, (index, positive) in enumerate(zip(indices, positives, strict=True)):
        local = np.searchsorted(known, positive)
        inside = local < len(known)
        inside[inside] &= known[local[inside]] == positive[inside]
        local = local[inside] + 1
        known_counts[row] = len(local)
        key = f"recflow-feedback-retrieval:17:{int(index)}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        random = np.random.default_rng(seed)
        distractors = random.choice(len(known), 1100, replace=False) + 1
        distractors = distractors[~np.isin(distractors, local)][:1000]
        candidates[row, :len(local)] = local
        candidates[row, len(local):len(local) + 1000] = distractors
    return dict(indices=indices, uid=data.requests["uid"][indices], candidates=candidates,
        positive_count=positive_counts, known_positive_count=known_counts,
        eligible_requests=len(selected), eligible_positive_requests=sum(len(p) > 0 for p in positives))


def metric_values(ranks, panel):
    values, random = {}, {}
    known, total = panel["known_positive_count"], panel["positive_count"]
    for cutoff in [20, 50, 100]:
        hit = (ranks > 0) & (ranks <= cutoff)
        values[f"Recall@{cutoff}"] = np.divide(hit.sum(axis=1), total,
            out=np.zeros(len(total)), where=total > 0)
        discount = np.r_[0., np.cumsum(1 / np.log2(np.arange(cutoff) + 2))]
        gain = np.where(hit, 1 / np.log2(np.maximum(ranks, 1) + 1), 0.).sum(axis=1)
        ideal = discount[np.minimum(total, cutoff)]
        values[f"NDCG@{cutoff}"] = np.divide(gain, ideal,
            out=np.zeros_like(gain), where=ideal > 0)
        expected = [random_expected_metrics(int(m), int(k), int(1000 + k), [cutoff])
            for m, k in zip(total, known, strict=True)]
        random[f"Recall@{cutoff}"] = np.nan_to_num(np.asarray([e[f"recall@{cutoff}"] for e in expected]))
        random[f"NDCG@{cutoff}"] = np.nan_to_num(np.asarray([e[f"ndcg@{cutoff}"] for e in expected]))
    first = np.where(ranks > 0, ranks, 65535).min(axis=1)
    values["MRR"] = np.where(first < 65535, 1 / first, 0.)
    mrr_random = {}
    for k in np.unique(known):
        n = 1000 + int(k)
        survival, expected = 1., 0.
        if k:
            for r in range(1, n - int(k) + 2):
                expected += survival * k / (n - r + 1) / r
                survival *= (n - r + 1 - k) / (n - r + 1)
        mrr_random[int(k)] = expected
    random["MRR"] = np.asarray([mrr_random[int(k)] for k in known])
    return values, random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--edges", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--include-all-requests", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.15)
    device = torch.device("cuda:0")
    data_root = Path("data/processed/recflow_feedback_large_v1")
    root = Path("experiments/foundation_recflow/runs/chain_2gpu/attempt_1/large/training")
    data = FeedbackData(data_root, device)
    known = np.load(data_root / "known_ids.npy", mmap_mode="r")
    source_paths = [Path(__file__), Path("src/hstu_kvcache/recflow/feedback.py"),
        Path("src/hstu_kvcache/models/hstu.py"), Path("src/hstu_kvcache/models/block.py"),
        Path("src/hstu_kvcache/models/attention.py"), Path("src/hstu_kvcache/recflow/metrics.py")]
    configuration = dict(status="exploratory_readonly_inference", scale="large",
        requested_positive_requests_per_day=args.requests, edges=args.edges, seed=17,
        data_manifest_sha256=digest(data_root / "manifest.json"),
        source_sha256={str(p): digest(p) for p in source_paths},
        batch=args.batch, device=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        include_all_requests=args.include_all_requests,
        per_process_memory_fraction=.15, precision="FP32",
        protocol="Frozen final-role positive requests; retain all-OOV misses in denominators. "
        "All distinct known observed effective-view positives plus exactly 1000 uniformly "
        "sampled unique known-catalog distractors, excluding observed positives. "
        "Distractors have unobserved relevance, not verified negative feedback. "
        "Candidate scoring uses the frozen scalar feedback head, not the old generative head. "
        "Histories exclude the entire same-timestamp block. With include_all_requests, "
        "all-negative requests score zero and stay in the population. Zero-known-positive "
        "requests need no candidate inference because every retrieval metric is exactly zero; "
        "this is an offline evaluation shortcut, not deployable serving work. "
        "This diagnostic does not replace feedback AUC. No model/data/config changes or training.")
    write_json(args.output / "configuration.json", configuration)
    edges = []
    for edge in args.edges:
        day, names = 20 + edge, [f"v{edge}", f"v{edge + 1}"]
        panel = make_panel(data, known, day, args.requests, args.include_all_requests)
        folder = args.output / f"{names[0]}_{names[1]}_d{day}"
        folder.mkdir(exist_ok=True)
        panel_path = folder / "panel.npz"
        save_rows(panel_path, **{k: v for k, v in panel.items() if isinstance(v, np.ndarray)})
        binding = dict(panel_sha256=digest(panel_path), configuration=configuration)
        results, version_values = {}, {}
        for version in names:
            checkpoint_path = root / version / "checkpoint.pt"
            marker = json.loads(checkpoint_path.with_suffix(".json").read_text())
            identity = dict(**binding, checkpoint_sha256=marker["sha256"])
            checkpoint = torch.load(checkpoint_path, mmap=True, map_location="cpu", weights_only=False)
            with torch.device(device):
                model = HSTU(HSTUConfig(**checkpoint["config"])).eval()
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            ranks = np.zeros((len(panel["indices"]), int(panel["known_positive_count"].max())), dtype=np.uint16)
            phase_started = time.time()
            with torch.inference_mode():
                for start in range(0, len(ranks), 128):
                    target = folder / f"{version}_block_{start:06}.npz"
                    metadata = target.with_suffix(".json")
                    stop = min(start + 128, len(ranks))
                    if target.exists() and metadata.exists():
                        previous = json.loads(metadata.read_text())
                        assert previous["binding"] == identity and previous["sha256"] == digest(target)
                        with np.load(target) as rows:
                            ranks[start:stop] = rows["positive_ranks"]
                        continue
                    needed = np.flatnonzero(panel["known_positive_count"][start:stop] > 0) + start
                    for offset in range(0, len(needed), args.batch):
                        take = needed[offset:offset + args.batch]
                        indices = panel["indices"][take]
                        width = next(n for n in [64, 128, 256, 512, 1024] if n >= data.lengths[indices].max())
                        batch = data.batch(indices, len(indices), width)
                        candidates = torch.as_tensor(panel["candidates"][take], device=device)
                        scores = grouped_scores(model, *batch[:4], candidates, batch[5]).cpu().numpy()
                        assert np.isfinite(scores).all()
                        for j, row in enumerate(take):
                            count = int(panel["known_positive_count"][row])
                            size = count + 1000
                            order = np.lexsort((panel["candidates"][row, :size], -scores[j, :size]))
                            inverse = np.empty(size, dtype=int)
                            inverse[order] = np.arange(1, size + 1)
                            ranks[row, :count] = inverse[:count]
                    save_rows(target, positive_ranks=ranks[start:stop])
                    write_json(metadata, dict(binding=identity, sha256=digest(target)))
                    print(json.dumps(dict(edge=edge, version=version, completed=stop,
                        requests=len(ranks), elapsed=time.time() - phase_started)), flush=True)
            values, random = metric_values(ranks, panel)
            version_values[version] = values
            results[version] = dict(metrics={k: float(v.mean()) for k, v in values.items()},
                random={k: float(v.mean()) for k, v in random.items()},
                checkpoint_sha256=marker["sha256"], seconds=time.time() - phase_started)
            del model
            torch.cuda.empty_cache()
        result = dict(parent=names[0], current=names[1], day=day,
            requests=len(panel["indices"]), eligible_requests=panel["eligible_requests"],
            eligible_positive_requests=panel["eligible_positive_requests"],
            users=len(np.unique(panel["uid"])),
            all_negative_requests=int((panel["positive_count"] == 0).sum()),
            scored_requests=int((panel["known_positive_count"] > 0).sum()),
            all_oov_positive_requests=int(((panel["known_positive_count"] == 0) & (panel["positive_count"] > 0)).sum()),
            results=results, relative_improvement_percent={key:
                float(100 * (results[names[1]]["metrics"][key] / value - 1)) if value else None
                for key, value in results[names[0]]["metrics"].items()})
        write_json(folder / "summary.json", result)
        edges.append(result)
        print(json.dumps(result), flush=True)
    result = dict(status="complete_exploratory_sampled_retrieval", configuration=configuration,
        edges=edges, elapsed_seconds=time.time() - started,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
    write_json(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

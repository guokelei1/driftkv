#!/usr/bin/env python3
"""RecFlow-Max frozen-checkpoint retrieval; reuses the Large protocol helpers."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.recflow.feedback import FeedbackData, grouped_scores
from recflow_feedback_update_analysis import digest
from recflow_feedback_sampled_retrieval import make_panel, metric_values, save_rows, write_json


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
    torch.cuda.set_per_process_memory_fraction(.25)
    device = torch.device("cuda:0")
    data_root = Path("data/processed/recflow_feedback_max_v1")
    root = Path("experiments/foundation_recflow/runs/chain_2gpu/attempt_1/max/training")
    data = FeedbackData(data_root, device)
    assert data.manifest["known_videos"] == 3_500_000
    known = np.load(data_root / "known_ids.npy", mmap_mode="r")
    source_paths = [Path(__file__), Path(__file__).with_name("recflow_feedback_sampled_retrieval.py"), Path("src/hstu_kvcache/recflow/feedback.py"),
        Path("src/hstu_kvcache/models/hstu.py"), Path("src/hstu_kvcache/models/block.py"),
        Path("src/hstu_kvcache/models/attention.py"), Path("src/hstu_kvcache/recflow/metrics.py")]
    configuration = dict(status="exploratory_readonly_inference", scale="max",
        requested_positive_requests_per_day=args.requests, edges=args.edges, seed=17,
        data_manifest_sha256=digest(data_root / "manifest.json"),
        source_sha256={str(p): digest(p) for p in source_paths},
        batch=args.batch, device=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        include_all_requests=args.include_all_requests,
        per_process_memory_fraction=.25, precision="FP32",
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
            assert marker["binding"]["scale"] == "max"
            identity = dict(**binding, checkpoint_sha256=marker["sha256"])
            checkpoint = torch.load(checkpoint_path, mmap=True, map_location="cpu", weights_only=False)
            assert checkpoint["config"]["num_layers"] == 16
            assert checkpoint["config"]["num_items"] == 3_565_536
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

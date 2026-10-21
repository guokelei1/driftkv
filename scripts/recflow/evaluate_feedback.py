#!/usr/bin/env python3
"""Paired adjacent versions on every eligible next-day final-role exposure."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.recflow.feedback import FeedbackData, grouped_scores
from hstu_kvcache.evaluation.binary_metrics import sigmoid, stable_log_loss
from hstu_kvcache.training.recovery import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recflow.train_feedback import atomic_json, barrier, read_run_config, setup, shape_for
from scripts.experiments_2026_10.statistics import weighted_auc


def metrics(rows):
    labels, logits = rows["label"], rows["logit"]
    probability = sigmoid(logits)
    result = dict(targets=len(labels), users=len(np.unique(rows["uid"])), positives=int(labels.sum()),
        log_loss=float(stable_log_loss(logits, labels).mean()), Brier=float(brier_score_loss(labels, probability)))
    if len(np.unique(labels)) == 2:
        result.update(ROC_AUC=float(roc_auc_score(labels, logits)), effective_view_PR_AUC=float(average_precision_score(labels, logits)),
            ineffective_view_PR_AUC=float(average_precision_score(1 - labels, -logits)))
    else:
        result.update(ROC_AUC=None, effective_view_PR_AUC=None, ineffective_view_PR_AUC=None)
    user_aucs = []
    for uid in np.unique(rows["uid"]):
        mask = rows["uid"] == uid
        if len(np.unique(labels[mask])) == 2:
            user_aucs.append(roc_auc_score(labels[mask], logits[mask]))
    result["user_macro_AUC"] = float(np.mean(user_aucs)) if user_aucs else None
    result["user_macro_AUC_eligible_users"] = len(user_aucs)
    _, request_index = np.unique(rows["request_index"], return_inverse=True)
    order = np.lexsort((rows["event_index"], -logits, request_index))
    sorted_request = request_index[order]
    starts = np.flatnonzero(np.r_[True, sorted_request[1:] != sorted_request[:-1]])
    ranks = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    positives = np.bincount(request_index, weights=labels).astype(int)
    counts = np.bincount(request_index)
    mixed = (positives > 0) & (positives < counts)
    for cutoff in [1, 3, 5]:
        discounts = 1 / np.log2(np.arange(cutoff) + 2)
        ideal = np.r_[0, discounts.cumsum()][np.minimum(positives, cutoff)]
        gain = labels[order] / np.log2(ranks + 2) * (ranks < cutoff)
        dcg = np.bincount(sorted_request, weights=gain, minlength=len(positives))
        ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
        result[f"within_exposure_request_NDCG@{cutoff}"] = float(ndcg.mean())
        result[f"mixed_label_request_NDCG@{cutoff}"] = float(ndcg[mixed].mean()) if mixed.any() else None
    result.update(requests=len(counts), mixed_label_requests=int(mixed.sum()),
        ndcg_scope="rank only actual exposed slate, all-negative slate NDCG=0; differs from historical uniform1000 NDCG")
    return result


def aggregate(output, data, names=("v0", "v1"), day=20, scale="large"):
    parent, current = names
    versions = {}
    for version in names:
        pieces = [dict(np.load(path)) for path in sorted((output / "scores" / version).glob("rank_*.npz"))]
        rows = {name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]}
        order = np.argsort(rows["event_index"])
        versions[version] = {key: value[order] for key, value in rows.items()}
    for key in ["event_index", "uid", "request_index", "label", "known"]:
        assert np.array_equal(versions[parent][key], versions[current][key]), key
    assert len(np.unique(versions[parent]["event_index"])) == len(versions[parent]["event_index"])
    expected = data.indices(2, day, day)
    assert np.array_equal(versions[parent]["request_index"], np.repeat(expected, data.counts[expected]))
    results = {}
    for version, rows in versions.items():
        results[version] = {"all": metrics(rows)}
        for known, tag in [(True, "known"), (False, "hashed_oov")]:
            mask = rows["known"] == known
            results[version][tag] = metrics({key: value[mask] for key, value in rows.items()})
    delta = {key: results[current]["all"][key] - value for key, value in results[parent]["all"].items()
        if isinstance(value, float) and results[current]["all"].get(key) is not None}
    # Paired user bootstrap includes all 3210 reserved users, including inactive users.
    population = np.sort(data.population["uids"][data.population["role"] == 2])
    uid_index = np.searchsorted(population, versions[parent]["uid"])
    rng = np.random.default_rng(20261021)
    draws = np.array([np.bincount(rng.integers(len(population), size=len(population)), minlength=len(population)) for _ in range(1000)], dtype=np.uint16)
    auces = {v: weighted_auc(r["logit"], r["label"], uid_index, draws) for v, r in versions.items()}
    auc_ci = np.percentile(auces[current] - auces[parent], [2.5, 97.5]).tolist()
    summary = dict(status="complete_adjacent_version_comparison", scale=scale, parent=parent, current=current, day=day,
        scope="all eligible initial final-role next-day exposure targets, same causal histories, no cache reuse experiment",
        eligible_test_users=len(population), metrics=results,
        paired_uid_auc_gain_ci95=auc_ci, bootstrap_repetitions=1000,
        selection_note="Report all metrics and regressions. Metric choice informed by this comparison is exploratory, not independent confirmation.")
    summary[f"{current}_minus_{parent}"] = delta
    atomic_json(output / "summary.json", summary)
    lines = [f"# RecFlow-{scale.title()} {parent} / {current} D{day}", "", "All eligible final-role exposure targets.", "",
        f"| Metric | {parent} | {current} | {current} minus {parent} |", "|---|---:|---:|---:|"]
    for key in delta:
        lines.append(f"| {key} | {results[parent]['all'][key]:.6f} | {results[current]['all'][key]:.6f} | {delta[key]:+.6f} |")
    lines.extend(["", f"Paired UID 95% interval for AUC gain: {auc_ci}.", "",
        "Higher AUC / PR-AUC / NDCG is better; lower LogLoss / Brier is better. Exposure-slate NDCG differs from the old uniform1000 task.",
        "Metric selection based on this result remains exploratory; retain every negative outcome."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", choices=["large", "max"])
    parser.add_argument("--edge", choices=[f"v{i}_v{i + 1}" for i in range(5)], default="v0_v1")
    args = parser.parse_args()
    config = read_run_config(args.config, args.scale)
    names = tuple(args.edge.split("_"))
    design = json.loads(Path(config["design"]).read_text())
    day = next(v["evaluate_day"] for v in design["training"]["versions"] if v["name"] == names[1])
    rank, world, device = setup()
    assert world == len(config["physical_gpus"])
    data = FeedbackData(config["data"], device)
    indices = data.indices(2, day, day)[rank::world]
    batch_size = config.get("evaluation_batch", 48)
    source_paths = [__file__, "src/hstu_kvcache/recflow/feedback.py", "src/hstu_kvcache/models/hstu.py",
        "src/hstu_kvcache/models/block.py", "src/hstu_kvcache/models/attention.py", "src/hstu_kvcache/models/embeddings.py",
        "src/hstu_kvcache/models/rmsnorm.py", "src/hstu_kvcache/models/triton_attention.py"]
    source_hashes = {path: digest(path) for path in source_paths}
    for version in names:
        path = args.training / version / "checkpoint.pt"
        checkpoint_marker = json.loads(path.with_suffix(".json").read_text())
        target = args.output / "scores" / version / f"rank_{rank}.npz"
        binding = dict(checkpoint_sha256=checkpoint_marker["sha256"], data_sha256=digest(Path(config["data"]) / "manifest.json"),
            world_size=world, role=2, day=day, scale=config["scale"], torch_version=str(torch.__version__), source_sha256=source_hashes)
        reused = False
        if target.with_suffix(".json").exists():
            previous = json.loads(target.with_suffix(".json").read_text())
            if previous["binding"] == binding and digest(target) == previous["sha256"]:
                reused = True
            else:
                raise ValueError("Existing evaluation score binding differs")
        if rank == 0:
            assert digest(path) == checkpoint_marker["sha256"]
        barrier(world)
        if reused:
            barrier(world)
            continue
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        with torch.device(device):
            model = HSTU(HSTUConfig(**checkpoint["config"])).eval()
        model.load_state_dict(checkpoint["model"])
        rows = {key: [] for key in ["event_index", "uid", "request_index", "label", "logit", "known"]}
        with torch.inference_mode():
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset:offset + batch_size]
                width, candidates = shape_for(data, selected)
                batch = data.batch(selected, batch_size, width, candidates)
                logits = grouped_scores(model, *batch[:6])
                mask = batch[8].cpu().numpy()
                values = logits.cpu().numpy()[mask]
                assert np.isfinite(values).all()
                event_index = np.concatenate([np.arange(data.requests["event_start"][i], data.requests["event_stop"][i]) for i in selected])
                metadata = dict(event_index=event_index, uid=data.source_events["uid"][event_index],
                    request_index=np.repeat(selected, data.counts[selected]), label=data.source_events["behavior"][event_index] & 1,
                    logit=values, known=data.mapped_items[event_index] <= data.manifest["known_videos"])
                for key, value in metadata.items():
                    rows[key].append(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez(stream, **{key: np.concatenate(value) for key, value in rows.items()})
        os.replace(temporary, target)
        atomic_json(target.with_suffix(".json"), dict(binding=binding, sha256=digest(target)))
        del model, checkpoint
        torch.cuda.empty_cache()
        barrier(world)
    barrier(world)
    if rank == 0:
        aggregate(args.output, data, names, day, config["scale"])
    barrier(world)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

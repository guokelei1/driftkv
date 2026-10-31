#!/usr/bin/env python3
"""Paired UID uncertainty for retained sampled-retrieval ranks, CPU only."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.archive.baseline_studies.recflow_feedback_update_analysis import digest
from experiments.archive.baseline_studies.recflow_feedback_sampled_retrieval import metric_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load("data/processed/recflow_v1/population.npz") as rows:
        population = np.sort(rows["uids"][rows["role"] == 2])
    rng = np.random.default_rng(20261021)
    draws = np.asarray([np.bincount(rng.integers(len(population), size=len(population)),
        minlength=len(population)) for _ in range(1000)], dtype=float)
    sources = [(root, json.loads((root / "summary.json").read_text())) for root in args.root]
    entries = [(root, source, edge) for root, source in sources for edge in source["edges"]]
    entries.sort(key=lambda item: item[2]["day"])
    assert len({edge["day"] for _, _, edge in entries}) == len(entries)
    requests = np.load("data/processed/recflow_v1/requests.npy", mmap_mode="r")
    edges, samples, inputs = [], {}, {}
    for root, source, edge in entries:
        old, new = edge["parent"], edge["current"]
        folder = root / f"{old}_{new}_d{edge['day']}"
        with np.load(folder / "panel.npz") as rows:
            panel = dict(rows)
        panel_sha256 = digest(folder / "panel.npz")
        uid = np.searchsorted(population, panel["uid"])
        assert np.array_equal(population[uid], panel["uid"])
        if source["configuration"].get("include_all_requests"):
            expected = np.flatnonzero((requests["role"] == 2) &
                (requests["day"] == edge["day"]) & (requests["history_stop"] > requests["history_start"]))
            assert np.array_equal(expected, panel["indices"])
        counts = np.bincount(uid, minlength=len(population))
        denominator = draws @ counts
        values = {}
        for version in [old, new]:
            parts = []
            for path in sorted(folder.glob(f"{version}_block_*.npz")):
                inputs[str(path)] = digest(path)
                marker = json.loads(path.with_suffix(".json").read_text())
                assert inputs[str(path)] == marker["sha256"]
                assert marker["binding"]["panel_sha256"] == panel_sha256
                assert marker["binding"]["configuration"] == source["configuration"]
                assert marker["binding"]["checkpoint_sha256"] == edge["results"][version]["checkpoint_sha256"]
                with np.load(path) as rows:
                    parts.append(rows["positive_ranks"])
            ranks = np.concatenate(parts)
            assert len(ranks) == len(uid)
            # int64 makes NumPy's logarithmic discounts float64 rather than
            # uint16 -> float32. Inference ranks and original summaries stay frozen.
            values[version], random = metric_values(ranks.astype(np.int64), panel)
        metrics = {}
        for key in values[old]:
            a, b = values[old][key], values[new][key]
            pooled = [np.bincount(uid, weights=v, minlength=len(population))
                for v in [a, b, random[key]]]
            ba, bb, br = [(draws @ v) / denominator for v in pooled]
            metrics[key] = dict(parent=float(a.mean()), current=float(b.mean()),
                random=float(random[key].mean()),
                relative_improvement_percent=float(100 * (b.mean() / a.mean() - 1)),
                current_over_random_multiple=float(b.mean() / random[key].mean()),
                paired_uid_difference_ci95=np.percentile(bb - ba, [2.5, 97.5]).tolist(),
                paired_uid_relative_improvement_percent_ci95=np.percentile(100 * (bb / ba - 1), [2.5, 97.5]).tolist(),
                current_minus_random_ci95=np.percentile(bb - br, [2.5, 97.5]).tolist())
            samples.setdefault(key, []).append((ba, bb))
        edges.append(dict(parent=old, current=new, day=edge["day"], requests=len(uid),
            active_users=len(np.unique(panel["uid"])),
            all_negative_requests=int((panel["positive_count"] == 0).sum()),
            all_oov_positive_requests=int(((panel["positive_count"] > 0) &
                (panel["known_positive_count"] == 0)).sum()),
            metrics=metrics))
    mean = {}
    for key, versions in samples.items():
        point = [e["metrics"][key]["relative_improvement_percent"] for e in edges]
        bootstrap = np.mean([100 * (b / a - 1) for a, b in versions], axis=0)
        mean[key] = dict(positive_edges=sum(v > 0 for v in point),
            difference=float(np.mean([e["metrics"][key]["current"] - e["metrics"][key]["parent"] for e in edges])),
            paired_uid_difference_ci95=np.percentile(np.mean([b - a for a, b in versions], axis=0), [2.5, 97.5]).tolist(),
            mean_relative_improvement_percent=float(np.mean(point)),
            paired_uid_mean_relative_improvement_percent_ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist())
    output = dict(status="complete_exploratory_paired_uid_analysis", edges=edges,
        equal_release_mean=mean, eligible_population=len(population), bootstrap_repetitions=1000,
        input_sha256=inputs, source_sha256=digest(Path(__file__)),
        metric_source_sha256=digest(Path(__file__).with_name("recflow_feedback_sampled_retrieval.py")),
        input_summaries_sha256={str(root / "summary.json"): digest(root / "summary.json") for root, _ in sources},
        include_all_requests=all(s["configuration"].get("include_all_requests", False) for _, s in sources),
        analysis_precision="Float64 rank discounts and aggregation; original inference ranks and summaries are unchanged.",
        scope="Conditional on these frozen models and deterministic request panels. "
        "1000 shared paired UID bootstrap draws include inactive users. "
        "All pointwise intervals are unadjusted for exploratory metric selection. "
        "Random comparisons use exact expected rank metrics for each identical candidate pool. "
        "This observed-positive retrieval diagnostic does not establish negative-feedback "
        "discrimination for unexposed distractors or full-catalog retrieval quality.")
    folder = args.output or args.root[0] / "paired_uid_analysis"
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(edges=edges, equal_release_mean=mean), indent=2))


if __name__ == "__main__":
    main()

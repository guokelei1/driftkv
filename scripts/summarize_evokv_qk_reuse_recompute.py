from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _format_number(value: object, precision: str) -> str:
    if value is None:
        return "N/A"
    return format(float(value), precision)


def _row(
    metric: str,
    better: str,
    reuse: float,
    recompute: float,
    gap: dict[str, object],
) -> dict[str, object]:
    interval = gap["record_cluster_bootstrap_95"]
    return {
        "metric": metric,
        "better": better,
        "reuse": reuse,
        "recompute": recompute,
        "oriented_gap": gap["absolute"],
        "gap_direction": gap["direction"],
        "relative_percent": gap["relative_percent"],
        "ci95_lower": interval["lower"],
        "ci95_upper": interval["upper"],
        "positive_direction_with_ci": gap["positive_direction_with_ci"],
    }


def build_rows(tuning: dict[str, object]) -> list[dict[str, object]]:
    reuse = tuning["reuse"]
    recompute = tuning["recompute"]
    gaps = tuning["gaps"]
    rows = [
        _row(
            "full_catalog_cross_entropy",
            "lower",
            reuse["cross_entropy"],
            recompute["cross_entropy"],
            gaps["cross_entropy"],
        )
    ]
    perplexity_gap = gaps["perplexity"]
    perplexity_interval = perplexity_gap[
        "record_cluster_bootstrap_95_penalty_percent"
    ]
    rows.append(
        {
            "metric": "perplexity",
            "better": "lower",
            "reuse": reuse["perplexity"],
            "recompute": recompute["perplexity"],
            "oriented_gap": reuse["perplexity"]
            - recompute["perplexity"],
            "gap_direction": "reuse_minus_recompute",
            "relative_percent": perplexity_gap["penalty_percent"],
            "ci95_lower": perplexity_interval["lower"],
            "ci95_upper": perplexity_interval["upper"],
            "positive_direction_with_ci": perplexity_interval["lower"] > 0,
        }
    )
    for metric in (
        "ndcg_at_10",
        "mrr",
        "hit_rate_at_10",
        "hit_rate_at_50",
        "hit_rate_at_200",
    ):
        rows.append(
            _row(
                metric,
                "higher",
                reuse[metric],
                recompute[metric],
                gaps[metric],
            )
        )
    return rows


def main() -> None:
    config = json.loads(parse_args().config.read_text())
    result_path = Path(config["outputs"]["result"])
    result = json.loads(result_path.read_text())
    if (
        result.get("status") != "complete_tuning_measurement"
        or result.get("quality", {}).get("qualification_consumed") is not False
        or result.get("checkpoint", {}).get("provisional_retained") is not True
    ):
        raise ValueError("QK reuse/recompute tuning result differs")
    tuning = result["quality"]["tuning"]
    rows = build_rows(tuning)
    payload = {
        "protocol": "evokv_qk_reuse_recompute_metric_table_v1",
        "status": "complete_tuning_measurement",
        "dataset": result["dataset"],
        "edge": {
            "source_version": result["edge"]["source_version"],
            "target_version": result["edge"]["target_version"],
        },
        "role": tuning["role"],
        "candidate_set": tuning["candidate_set"],
        "num_prediction_items": tuning["num_prediction_items"],
        "records": tuning["records"],
        "positive_targets": tuning["positive_targets"],
        "rows": rows,
        "decision": "manual_after_tuning",
    }
    output_json = Path(config["outputs"]["metric_table_json"])
    output_csv = Path(config["outputs"]["metric_table_csv"])
    output_markdown = Path(config["outputs"]["metric_table_markdown"])
    _atomic_text(
        output_json,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    fields = list(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(output_csv, buffer.getvalue())
    lines = [
        "| Metric | Better | Reuse | Recompute | Oriented gap | Relative % | 95% CI | CI positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['better']} | "
            f"{_format_number(row['reuse'], '.8g')} | "
            f"{_format_number(row['recompute'], '.8g')} | "
            f"{_format_number(row['oriented_gap'], '.8g')} | "
            f"{_format_number(row['relative_percent'], '.6g')} | "
            f"[{_format_number(row['ci95_lower'], '.8g')}, "
            f"{_format_number(row['ci95_upper'], '.8g')}] | "
            f"{row['positive_direction_with_ci']} |"
        )
    _atomic_text(output_markdown, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E6 paired UID-cluster analysis of committed formal scores, on CPU only.

Resample the union of the actual UID cohorts. A UID has one multiplicity in
every edge/seed/method in a family; each edge's sampled size may therefore vary.
Average edge AUC differences before dividing. Main, seed17 K/V sensitivity and
the separate two-edge512 protocol comparison never share an aggregate.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.ch3_motivation.code import plan_lowrank_formal as planner
from experiments.ch3_motivation.code.analyze_probe_quality import ShardLoader, _interval
from experiments.ch3_motivation.code.analyze_probe_quality_multiedge import _ratio
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.fg_trajectory_control import (
    TRAINED_FG_METHOD as FG, H_DISABLED_DIAGNOSTIC as STRUCTURAL,
)
from experiments.ch3_motivation.code.run_lowrank_release import sources as release_sources
from experiments.ch3_motivation.code.run_lowrank_timestamp_guarded import sources as timestamp_sources
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

EDGES, SEEDS, RANKS = planner.EDGES, planner.SEEDS, planner.RANKS
VARIANTS = planner.ALL_VARIANTS
CONTROLS = (FG, "full", STRUCTURAL)
KEY_FIELDS = ("uid", "query_timestamp", "raw_item_id", "item_idx", "label", "request_instance_ordinal")
KEY_DTYPE = np.dtype(list(zip(KEY_FIELDS, ("u8", "i8", "u8", "i8", "i1", "i4"), strict=True)))


def sources():
    paths = [Path(__file__), Path(__file__).with_name("analyze_probe_quality.py"),
        Path(__file__).with_name("analyze_probe_quality_multiedge.py"),
        Path(__file__).with_name("probe_cost_export.py"),
        Path(__file__).with_name("calibration_cost_correction.py"),
        ROOT/"src/hstu_kvcache/evaluation/statistics.py"]
    return planner.merge_frozen(planner.sources(), {str(path): sha256(path) for path in paths})


def event_keys(table, *, assign_ordinals=False):
    """The two actual E6 formats; timestamp's absent ordinal means zero."""
    keys = np.zeros(len(table), dtype=KEY_DTYPE)
    for name in KEY_FIELDS:
        if name in table.column_names:
            keys[name] = table[name].to_numpy(zero_copy_only=False)
        elif name != "request_instance_ordinal":
            raise ValueError(f"E6 event column absent: {name}")
    if np.any((table["label"].to_numpy() != 0) & (table["label"].to_numpy() != 1)):
        raise ValueError("E6 feedback labels must be binary")
    if assign_ordinals:
        # Request files have no ordinal. Distinguish any repeated event instances
        # with the same original key, matching the release writer's convention.
        order = np.argsort(keys, kind="stable", order=KEY_FIELDS[:-1])
        ordered = keys[order]
        group = np.zeros(len(keys), dtype=bool)
        if len(keys):
            group[0] = True
            group[1:] = (ordered["uid"][1:] != ordered["uid"][:-1]) | (
                ordered["query_timestamp"][1:] != ordered["query_timestamp"][:-1]) | (
                ordered["raw_item_id"][1:] != ordered["raw_item_id"][:-1])
            starts = np.maximum.accumulate(np.where(group, np.arange(len(keys)), 0))
            keys["request_instance_ordinal"][order] = np.arange(len(keys))-starts
    return keys


def _methods(variants):
    return [*CONTROLS, *(f"{variant}_r{rank}" for variant in variants for rank in RANKS)]


def _job_key(job):
    family = "main" if job["id"].startswith("main_") else "protocol"
    return family, job["protocol"], job["edge"], job["seed"]


def audit_plan(plan):
    if plan["format"] != "e6_formal_handoff_v1":
        raise ValueError("E6 analysis requires the actual formal handoff plan")
    expected = {("main", "release", e, s) for e in EDGES for s in SEEDS} | {
        ("protocol", protocol, e, 17) for protocol in ("release", "timestamp") for e in (2, 5)}
    if (len(plan["jobs"]) != 19 or len({job["id"] for job in plan["jobs"]}) != 19 or
            {_job_key(job) for job in plan["jobs"]} != expected):
        raise ValueError("E6 analysis needs all19 predeclared jobs; no outcome-based edge/seed selection")
    if len(plan["weight_bindings"]) != 15:
        raise ValueError("E6 requires the actual15 independent FG/H pairs")
    for job in plan["jobs"]:
        family, _, edge, seed = _job_key(job)
        binding = plan["weight_bindings"][job["weight_binding_id"]]
        n = 2000 if family == "main" else 512
        variants = list(VARIANTS) if family == "main" and seed == 17 else ["both_norm"]
        if (binding["edge"] != edge or binding["seed"] != seed or binding["users"] != 2000 or
                len(binding["uids"]) != 2000 or len(set(binding["uids"])) != 2000 or
                planner.uid_hash(binding["uids"]) != binding["uids_sha256"] or job["users"] != n or
                planner.uid_hash(binding["uids"][:n]) != job["uids_sha256"] or
                job["variants"] != variants or job["ranks"] != list(RANKS) or not job["quality_auc_permitted"] or
                job["full_h_checkpoint"] != binding["full_h_checkpoint"] or
                job["trained_fg_checkpoint"] != binding["trained_fg_checkpoint"] or
                job["population_file"] != binding["population_file"]):
            raise ValueError("E6 job roles/weights/ranks differ from its frozen binding")


def load_job(job, binding, reader, request_cache):
    """Read one job columnwise and retain only aligned score vectors and keys."""
    folder = Path(job["output"])
    summary = reader.json(folder/"summary.json")
    config = reader.json(folder/"configuration.json")
    identity_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    users = binding["uids"][:job["users"]]
    protocol = job["protocol"]
    if (summary["status"] != "complete" or summary.get("quality_auc_computed") is not True or
            summary["users"] != len(users) or summary["identity_hash"] != identity_hash or
            config["users"] != users or config["edge"] != job["edge"] or
            config["models"] != binding["models"] or config["ranks"] != list(RANKS) or config["variants"] != job["variants"] or
            config["population"] != binding["population_provenance"] or
            config.get("full_h_checkpoint_sha256", config.get("checkpoint_sha256")) != binding["full_h_checkpoint_sha256"] or
            config["trained_fg_control"] != binding["paired_control"] or
            config.get("full_h_fitting", config.get("fitting")) != binding["fitting"] or
            config.get("candidates", config.get("candidate_size")) != 230):
        raise ValueError("E6 result/configuration differs from the actual population/FG/H/ranks")
    wanted_protocol = "joint_e14_request_instance_release_v1" if protocol == "release" else "e6_trained_fg_own_state_v2"
    if config.get("protocol", config.get("protocol_version")) != wanted_protocol:
        raise ValueError("E6 protocol cannot be pooled under another label")
    recorded_sources = config.get("sources", config.get("source_sha256", {}))
    if recorded_sources != (release_sources() if protocol == "release" else timestamp_sources()):
        raise ValueError("E6 result lacks the actual complete numerical dependency identities")
    expected_role = "main_2000_feedback" if _job_key(job)[0] == "main" else "additional_fixed_subset_protocol_sensitivity"
    if config["role"] != expected_role or summary["role"] != expected_role:
        raise ValueError("E6 quality role differs from the planned2000/512 scope")
    decomp, execution = config["decomposition"], config["execution"]
    if (any(decomp.get(key) != value for key, value in (("method", "randomized"), ("oversample", 16),
            ("power_iterations", 2), ("batch_heads", 25))) or execution["candidate_chunk"] != 230 or
            execution["row_chunk"] != 1024 or execution.get("mode", execution.get("interaction_and_writer")) != "projected" or
            not execution.get("batch_rank_scores", config.get("batch_rank_scores", False))):
        raise ValueError("E6 quality decomposition/execution differs from the actual formal scope")
    if not np.isfinite(summary["full_rank_max_error"]) or not 0 <= summary["full_rank_max_error"] <= 1e-5:
        raise ValueError("E6 full-rank numerical equality failed")
    for name, digest in recorded_sources.items():
        path = str((ROOT/name).resolve())
        if plan_digest := reader.hashes.get(path):
            if plan_digest != digest:
                raise ValueError("E6 result sources differ from the frozen plan")
        reader.verify(path, digest)
    for field in ("full_h_checkpoint", "trained_fg_checkpoint", "population_file"):
        digest_field = "population_sha256" if field == "population_file" else field+"_sha256"
        reader.verify(binding[field], binding[digest_field])
    provenance = binding["population_provenance"]
    source = provenance["requests_path"]
    cache_key = source, tuple(users)
    reader.verify(source, provenance["requests_sha256"])
    if cache_key not in request_cache:
        table = pq.read_table(source, filters=[("uid", "in", users)], columns=list(KEY_FIELDS[:-1]))
        expected = np.sort(event_keys(table, assign_ordinals=True), order=KEY_FIELDS)
        if set(expected["uid"].tolist()) != set(users) or len(np.unique(expected)) != len(expected):
            raise ValueError("E6 expected events do not cover each frozen feedback UID exactly")
        request_cache[cache_key] = expected
    expected = request_cache[cache_key]
    width = config["execution"]["checkpoint_users"]
    if width != 4:
        raise ValueError("formal E6 uses the planned four-user committed blocks")
    marker_paths = [folder/"blocks"/f"part_{offset:06d}.json" for offset in range(0, len(users), width)]
    if set(marker_paths) != set((folder/"blocks").glob("part_*.json")):
        raise ValueError("E6 committed blocks are missing, extra, or from another UID scope")
    tables = []
    for offset, path in zip(range(0, len(users), width), marker_paths, strict=True):
        marker = reader.json(path)
        selected = users[offset:offset+width]
        if marker["identity"] != identity_hash or marker["report"]["uids"] != selected:
            raise ValueError("E6 committed block identity/contiguous UID coverage differs")
        parquet = path.parent/marker["parquet"]
        if parquet != path.with_suffix(".parquet"):
            raise ValueError("E6 committed score path differs from its offset")
        reader.verify(parquet, marker["sha256"])
        table = pq.read_table(parquet)
        if set(int(uid) for uid in table["uid"].to_numpy()) != set(selected):
            raise ValueError("E6 score block does not cover exactly its selected feedback users")
        tables.append(table)
    table = pa.concat_tables(tables)
    methods = table["method"].to_numpy(zero_copy_only=False)
    names = _methods(job["variants"])
    if set(methods) != set(names):
        raise ValueError("E6 committed method/rank coverage differs")
    keys, raw_scores = event_keys(table), table["score"].to_numpy(zero_copy_only=False)
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError("nonfinite E6 score")
    scores = {}
    for method in names:
        select = np.flatnonzero(methods == method)
        select = select[np.argsort(keys[select], order=KEY_FIELDS, kind="stable")]
        if not np.array_equal(keys[select], expected):
            raise ValueError("E6 FG/fullH/rank events, labels, ordinals or multiplicity differ")
        scores[method] = np.asarray(raw_scores[select], dtype=np.float64)
    points = {row["method"]: row["auc"] for row in summary["points"]}
    if len(points) != len(summary["points"]) or set(points) != set(names):
        raise ValueError("E6 summary method coverage differs from committed scores")
    if any(not np.isfinite(value) for value in points.values()):
        raise ValueError("nonfinite E6 summary AUC")
    return dict(id=job["id"], family=_job_key(job)[0], protocol=protocol, edge=job["edge"], seed=job["seed"],
        users=users, keys=expected, scores=scores, expected_auc=points,
        evaluation_window=binding["qualification"].get("evaluation_window"),
        qualification=binding["qualification"], source_summary=str((folder/"summary.json").resolve()),
        score_rows=len(table), blocks=len(marker_paths))


def union_weights(items, replicates, seed):
    population = np.asarray(sorted({uid for item in items for uid in item["users"]}), dtype=np.uint64)
    rng = np.random.default_rng(seed)
    weights = np.empty((replicates+1, len(population)), dtype=np.uint16)
    weights[0] = 1
    for draw in range(1, replicates+1):
        weights[draw] = np.bincount(rng.integers(len(population), size=len(population)), minlength=len(population))
    return population, weights


def _metric(values, fg, full):
    gain, gap = values-fg, full-fg
    ratio = _ratio(gain[0], gain[1:], gap[0], gap[1:])
    return dict(auc=float(values[0]), auc_ci95=_interval(values[1:]),
        auc_gain_over_trained_fg=float(gain[0]), auc_gain_over_trained_fg_ci95=_interval(gain[1:]),
        auc_minus_full_h=float(values[0]-full[0]), auc_minus_full_h_ci95=_interval(values[1:]-full[1:]),
        full_h_minus_trained_fg=float(gap[0]), full_h_minus_trained_fg_ci95=_interval(gap[1:]),
        interaction_gain_retention_percent=ratio["point"] if ratio["positive_reference_gap_point"] else None,
        signed_ratio_percent=ratio["point"], interaction_gain_retention_ci95=ratio["conditional_uid_ci95"],
        positive_h_gain_point=ratio["positive_reference_gap_point"],
        nonpositive_or_invalid_h_gain_replicates=ratio["nonpositive_or_invalid_gap_replicates"],
        finite_auc_replicates=int(np.isfinite(values[1:]).sum()))


def _family(items, *, edges, seeds, variants, population, weights, cache):
    indexed = {(item["edge"], item["seed"]): item for item in items}
    if len(indexed) != len(items):
        raise ValueError("duplicate E6 family edge/seed cell")
    if set(indexed)-{(e,s) for e in edges for s in seeds}:
        raise ValueError("E6 family contains an unplanned edge/seed cell")
    missing = [dict(edge=e, seed=s) for e in edges for s in seeds if (e, s) not in indexed]
    result = dict(status="pending" if missing else "complete", expected_edges=list(edges), expected_seeds=list(seeds),
        missing=missing, results_count=0, per_seed=[], per_edge=[])
    if missing:
        return result, {}
    names = _methods(variants)
    values = {}
    population_key = hashlib.sha256(population.tobytes()+weights.tobytes()).hexdigest()
    for edge in edges:
        for seed in seeds:
            item = indexed[edge, seed]
            keys = item["keys"]
            reference = indexed[edge, seeds[0]]
            if item["users"] != reference["users"] or not np.array_equal(keys, reference["keys"]):
                raise ValueError("E6 same-edge training seeds must share identical UID/event populations")
            if (len(set(item["users"])) != len(item["users"]) or
                    set(keys["uid"].tolist()) != set(item["users"]) or len(np.unique(keys)) != len(keys)):
                raise ValueError("E6 numerical input event/UID coverage differs")
            index = np.searchsorted(population, keys["uid"])
            if np.any(index >= len(population)) or not np.array_equal(population[index], keys["uid"]):
                raise ValueError("E6 score UID absent from union population")
            for method in names:
                key = item["id"], method, population_key
                if key not in cache:
                    scores = item["scores"][method]
                    if len(scores) != len(keys) or not np.all(np.isfinite(scores)):
                        raise ValueError("E6 numerical scores invalid or misaligned")
                    cache[key] = weighted_auc(scores, keys["label"], index, weights)
                    if not np.isfinite(cache[key][0]):
                        raise ValueError("E6 AUC point requires both observed labels")
                    if method in item.get("expected_auc", {}) and abs(cache[key][0]-item["expected_auc"][method]) > 1e-12:
                        raise ValueError("E6 summary AUC disagrees with verified committed scores")
                values[edge, seed, method] = cache[key]
            fg, full = values[edge, seed, FG], values[edge, seed, "full"]
            result["per_edge"].append(dict(edge=edge, seed=seed, feedback_events=len(keys),
                users=len(item["users"]), uids_sha256=planner.uid_hash(item["users"]),
                evaluation_window=item.get("evaluation_window"), qualification=item.get("qualification"),
                methods={method: _metric(values[edge, seed, method], fg, full) for method in names}))
    means = {}
    for seed in seeds:
        measured = {method: np.mean([values[e, seed, method] for e in edges], axis=0) for method in names}
        means[seed] = measured
        result["per_seed"].append(dict(seed=seed, methods={method: _metric(value, measured[FG], measured["full"])
            for method, value in measured.items()}))
    result.update(results_count=len(names)*len(seeds), union_users=len(population),
        seed_point_ranges={method: dict(auc=[min(means[s][method][0] for s in seeds), max(means[s][method][0] for s in seeds)],
            auc_gain_over_trained_fg=[min(means[s][method][0]-means[s][FG][0] for s in seeds),
                max(means[s][method][0]-means[s][FG][0] for s in seeds)]) for method in names},
        aggregation="equal-edge mean AUC differences, then ratio; never average per-edge ratios",
        bootstrap_scope="conditional on fixed fitted weights; seed point ranges are descriptive, not a training-seed confidence interval")
    return result, means


def aggregate_loaded(loaded, *, replicates=1000, seed=20261031):
    """Numerical core for loader-audited artifacts; tests use known synthetic scores."""
    if replicates < 1:
        raise ValueError("positive UID bootstrap replicate count required")
    if len({item["id"] for item in loaded}) != len(loaded):
        raise ValueError("duplicate E6 artifact id")
    main = [item for item in loaded if item["family"] == "main"]
    protocol = [item for item in loaded if item["family"] == "protocol"]
    if len(main)+len(protocol) != len(loaded):
        raise ValueError("unknown E6 family; no implicit population pooling")
    if any(item["protocol"] != "release" for item in main) or any(
            item["protocol"] not in ("release", "timestamp") for item in protocol):
        raise ValueError("unknown E6 protocol within a planned family")
    same_edge = {}
    for item in main:
        reference = same_edge.setdefault(item["edge"], item)
        if item["users"] != reference["users"] or not np.array_equal(item["keys"], reference["keys"]):
            raise ValueError("E6 same-edge training seeds must share identical UID/event populations")
    families, cache = {}, {}
    if main:
        population, weights = union_weights(main, replicates, seed)
        families["main_both_norm"], _ = _family(main, edges=EDGES, seeds=SEEDS, variants=("both_norm",),
            population=population, weights=weights, cache=cache)
        families["seed17_kv_sensitivity"], _ = _family([item for item in main if item["seed"] == 17],
            edges=EDGES, seeds=(17,), variants=VARIANTS, population=population, weights=weights, cache=cache)
    else:
        families.update({name: dict(status="pending", results_count=0, per_seed=[], per_edge=[])
            for name in ("main_both_norm", "seed17_kv_sensitivity")})
    compare = dict(status="pending", results_count=0, protocols={}, contrasts=[])
    if protocol:
        population, weights = union_weights(protocol, replicates, seed+1)
        means = {}
        for name in ("release", "timestamp"):
            compare["protocols"][name], means[name] = _family([item for item in protocol if item["protocol"] == name],
                edges=(2, 5), seeds=(17,), variants=("both_norm",), population=population, weights=weights, cache=cache)
        compare["results_count"] = sum(value["results_count"] for value in compare["protocols"].values())
        if all(value["status"] == "complete" for value in compare["protocols"].values()):
            by_key = {(item["protocol"], item["edge"]): item for item in protocol}
            for edge in (2, 5):
                if not np.array_equal(by_key["release", edge]["keys"], by_key["timestamp", edge]["keys"]):
                    raise ValueError("protocol comparison requires exactly paired feedback instances")
            for method in _methods(("both_norm",)):
                a, b = means["release"][17], means["timestamp"][17]
                delta = a[method]-b[method]
                gains = (a[method]-a[FG])-(b[method]-b[FG])
                compare["contrasts"].append(dict(method=method, auc_release_minus_timestamp=float(delta[0]),
                    auc_release_minus_timestamp_ci95=_interval(delta[1:]),
                    own_fg_gain_release_minus_timestamp=float(gains[0]), own_fg_gain_release_minus_timestamp_ci95=_interval(gains[1:])))
            compare.update(status="complete", results_count=sum(value["results_count"] for value in compare["protocols"].values()),
                expected_edges=[2, 5], expected_seeds=[17], users_per_edge=512,
                interpretation="two-edge512 protocol sensitivity; separate subspaces and denominators, no pooling with main2000")
    families["protocol512_sensitivity"] = compare
    return dict(status="complete" if all(f["status"] == "complete" for f in families.values()) else "pending",
        families=families, results_count=sum(f["results_count"] for f in families.values()),
        confidence=dict(replicates=replicates, seed=seed, protocol_seed=seed+1,
            sampling="sample the family UID union with replacement, size equal to union size; map each UID multiplicity into every edge/method/seed",
            edge_sample_size="random after restriction to an edge; not exactly2000/512 draws within each edge",
            main_and_kv_share_draws=True, protocol_family_separate=True,
            ratio_rule="report positive-gap retention only for positive aggregate H gain; signed points retained as diagnostic; no bounded ratio CI if any aggregate draw is nonpositive/invalid"))


def analyze(plan_path, output, *, replicates=1000, seed=20261031):
    plan_path, output = Path(plan_path).resolve(), Path(output)
    reader, code = ShardLoader(), sources()
    result = dict(status="pending", results_count=0, gpu_used=False, plan=str(plan_path), sources_sha256=code,
        missing=[], families={})
    if not plan_path.exists():
        result["missing"] = [str(plan_path)]
    else:
        plan = reader.json(plan_path)
        result["plan_sha256"] = reader.verify(plan_path)
        if plan["status"] != "prepared_for_root_review":
            result["upstream_status"] = plan["status"]
        else:
            audit_plan(plan)
            frozen = planner.merge_frozen(plan["frozen_sha256"])
            for path, digest in frozen.items():
                reader.verify(path, digest)
            loaded, request_cache = [], {}
            for job in plan["jobs"]:
                folder = Path(job["output"])
                absent = [str(folder/name) for name in ("summary.json", "configuration.json") if not (folder/name).is_file()]
                if absent:
                    result["missing"].extend(absent)
                else:
                    binding = plan["weight_bindings"][job["weight_binding_id"]]
                    loaded.append(load_job(job, binding, reader, request_cache))
            result.update(aggregate_loaded(loaded, replicates=replicates, seed=seed),
                audited_jobs=len(loaded), audited_blocks=sum(item["blocks"] for item in loaded),
                committed_score_rows=sum(item["score_rows"] for item in loaded),
                inference_scope=plan["inference_scope"], population_policy=plan["population_policy"])
    if sources() != code:
        raise ValueError("E6 analysis sources changed during aggregation")
    result["inputs_sha256"] = reader.hashes
    write_json(output/"summary.json", result)
    rows = []
    for name, family in result["families"].items():
        groups = family.get("protocols", {"release": family})
        for protocol, group in groups.items():
            for scope, field in (("equal_edge_mean", "per_seed"), ("edge", "per_edge")):
                for entry in group.get(field, []):
                    for method, metrics in entry["methods"].items():
                        row = dict(family=name, scope=scope, protocol=protocol, edge=entry.get("edge"), seed=entry["seed"], method=method)
                        for key, value in metrics.items():
                            if key.endswith("ci95"):
                                row[key+"_low"], row[key+"_high"] = value or [None, None]
                            else:
                                row[key] = value
                        rows.append(row)
    if rows:
        with (output/"points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"README.md").write_text("E6 paired multiedge analysis\n\nStatus: "+result["status"]+". Results: "+str(result["results_count"])+". No GPU used.\n\n"
        "Main2000 is reported by training seed across five edges; seed17 K/V and the two-edge512 protocol sensitivity stay separate. "
        "UID-cluster draws sample each family's union, so an edge's resampled size can vary. "
        "AUC differences are averaged across edges before any ratio. Nonpositive/zero H gains and partial edge5 remain; signed ratios are diagnostic only. "
        "Intervals condition on the fitted weights and do not measure training-seed variability. "
        "Quality snapshots are hindsight interventions; complete deployment costs belong to the independent E6 CPU sidecar.\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    result = analyze(args.plan, args.output, replicates=args.replicates, seed=args.seed)
    print(json.dumps(dict(status=result["status"], results=result["results_count"], gpu_used=False)))

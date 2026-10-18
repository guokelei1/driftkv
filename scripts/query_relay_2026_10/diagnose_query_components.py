#!/usr/bin/env python3
"""Causal V3 calibration-query composition and time-gap diagnostics.

No training or recommendation-quality evaluation. Candidate sets and all gap
statistics use only the fixed fit/validation histories strictly before release.
Alternative vectors are diagnostic inputs, not changes to the serving reader.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import numpy as np
import torch
import torch.nn.functional as F
from design_two.common import RELEASES, load_models, setup_device
from design_two.prepare import load_history, load_panel
from read_correction_2026_09.v2.calibrate import mixed_candidates
from unified_reuse_2026_09.common import sha256, write_json

PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"


def pre_cut_gaps(times, cut):
    """Strict previous-time gaps match query-before-all-same-time behavior."""
    past = np.asarray(times)[:np.searchsorted(times, cut, side="left")]
    if len(past) < 2:
        raise ValueError("gap diagnostic needs two retained pre-release behaviors")
    previous = np.searchsorted(past, past, side="left") - 1
    selected = previous >= 0
    strict = past[selected] - past[previous[selected]]
    return dict(cut_gap=int(cut - past[-1]), adjacent=np.diff(past), strict=strict,
                representative=float(np.median(strict)), retained_events=len(past))


def normalize(values):
    return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)


def first_query_from_raw(model, raw):
    token = model.input_dropout(model.in_proj(raw))
    return model.blocks[0].attn._project(model.blocks[0].norm(token))[0]


def query_components(model, candidates, delta):
    """Exact native summands before the shared projection and first norm."""
    shape = candidates.shape
    dtype = model.item_emb.weight.dtype
    gap = torch.as_tensor(delta, device=candidates.device, dtype=dtype).reshape(1, 1).expand(shape)
    types = torch.full_like(candidates, model.cfg.query_type_id)
    actions = torch.full_like(candidates, model.cfg.query_action_id)
    return dict(item=model.lookup_item_embeddings(candidates),
                query_type=model.query_encoder.type_embedding(types),
                query_action=model.query_encoder.action_embedding(actions),
                time=model.temporal_enc(gap))


def vector_metrics(values):
    # Canonical [batch=1,heads,candidates,width]; embedding uses heads=1.
    data = values.detach().double().cpu()[0]
    std = data.var(1, unbiased=False).mean().sqrt()
    total = data.square().mean().sqrt()
    directions = F.normalize(data, dim=-1, eps=1e-12)
    pair = directions @ directions.transpose(-2, -1)
    count = data.shape[1]
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    mean_direction = F.normalize(data.mean(1, keepdim=True), dim=-1, eps=1e-12)
    return dict(rms=float(total), candidate_std_rms=float(std),
                candidate_std_over_rms=float(std / total) if total else None,
                mean_pair_cosine=float(pair[:, off_diagonal].mean()),
                mean_cosine_to_set_mean=float((directions * mean_direction).sum(-1).mean()))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    probabilities = [0, .1, .25, .5, .75, .9, .99, 1]
    return dict(count=len(values), quantiles_seconds={str(p): float(x) for p, x in
        zip(probabilities, np.quantile(values, probabilities), strict=True)})


@torch.no_grad()
def run(args):
    metadata = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if metadata["target"] != "v3":
        raise ValueError("this first-query probe is fixed to the V3 release")
    roles = {name: list(metadata[name + "_uids"]) for name in ("fit", "validation")}
    binding, groups, _ = load_panel(args.panel)
    for name in roles:
        if not set(roles[name]) <= set(groups[name]):
            raise ValueError("query diagnostic changed calibration roles")
    uids = roles["fit"] + roles["validation"]
    args.output.mkdir(parents=True, exist_ok=False)
    device = setup_device(args.gpu)
    model = load_models(binding, device, last_version=3)["v3"]
    histories = load_history(args.panel, requested_uids=uids)
    cut = dict(RELEASES)["v3"]
    units, saved, totals, gaps = [], {}, defaultdict(list), defaultdict(list)
    for uid in uids:
        times, items, _ = histories.rows[uid]
        before = np.searchsorted(times, cut, side="left")
        history = items[:before][-model.cfg.max_seq_len:]
        ids = torch.tensor([mixed_candidates(uid, history, binding["known_vocab_size"], 16, seed=17)], device=device)
        data = pre_cut_gaps(times, cut)
        gaps["cutover_minus_last_behavior"].append(data["cut_gap"])
        gaps["adjacent_pre_cut_behavior"].extend(data["adjacent"].tolist())
        gaps["strict_previous_time_pre_cut_behavior"].extend(data["strict"].tolist())
        gaps["per_uid_median_strict_gap"].append(data["representative"])
        components = query_components(model, ids, data["cut_gap"])
        raw = sum(components.values())
        embedded = model.embed_query_tokens(ids, torch.tensor([data["cut_gap"]], device=device))
        torch.testing.assert_close(embedded, model.input_dropout(model.in_proj(raw)), rtol=2e-6, atol=2e-6)
        variants = {}
        for name, delta in (("cut_gap", data["cut_gap"]), ("zero_gap", 0.),
                            ("pre_cut_median_gap", data["representative"])):
            variant_components = query_components(model, ids, delta)
            q0 = first_query_from_raw(model, sum(variant_components.values()))
            pad_components = query_components(model, torch.zeros_like(ids), delta)
            pad_q0 = first_query_from_raw(model, sum(pad_components.values()))
            variants[name] = q0
            variants[name + "_candidate_minus_pad_q0"] = q0 - pad_q0
        variants["item_only_q0"] = first_query_from_raw(model, components["item"])
        variants["item_plus_time_q0"] = first_query_from_raw(model, components["item"] + components["time"])
        variants["current_item_embedding"] = components["item"][:, None]
        values = {name: dict(raw=vector_metrics(vector), rms_gate_input=vector_metrics(normalize(vector)))
                  for name, vector in variants.items()}
        components_stats = {name: dict(raw_rms=float(value.square().mean().sqrt()),
            projected_rms=float(model.in_proj(value).square().mean().sqrt()))
            for name, value in components.items()}
        for name, vector in variants.items():
            totals[name].append(normalize(vector).double().cpu()[0].mean(1))
        unit = dict(uid=uid, role="fit" if uid in roles["fit"] else "validation",
                    cut_gap=data["cut_gap"], representative_pre_cut_gap=data["representative"],
                    retained_pre_cut_behaviors=data["retained_events"], components=components_stats,
                    variants=values, candidate_ids=ids[0].cpu().tolist())
        units.append(unit)
        saved[uid] = dict(candidates=ids.cpu(), components={name: value.cpu() for name, value in components.items()},
                         variants={name: value.cpu() for name, value in variants.items()})
    overall = {}
    for name, means in totals.items():
        matrix = torch.stack(means)
        overall[name] = dict(
            mean_candidate_std_rms=float(np.mean([unit["variants"][name]["rms_gate_input"]["candidate_std_rms"] for unit in units])),
            between_set_mean_std_rms=float(matrix.var(0, unbiased=False).mean().sqrt()),
            mean_pair_cosine=float(np.mean([unit["variants"][name]["rms_gate_input"]["mean_pair_cosine"] for unit in units])),
            mean_raw_rms=float(np.mean([unit["variants"][name]["raw"]["rms"] for unit in units])))
    result = dict(status="complete", target="v3", roles=roles, units=units,
        gap_statistics={name: quantiles(value) for name, value in gaps.items()},
        components_mean={name: {metric: float(np.mean([unit["components"][name][metric] for unit in units]))
                               for metric in ("raw_rms", "projected_rms")} for name in components},
        variants=overall, model=binding["models"]["v3"],
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        panel_sha256=sha256(args.panel / "binding.json"), source_sha256=sha256(Path(__file__)),
        query_ids=dict(type=model.cfg.query_type_id, action=model.cfg.query_action_id),
        scope="Fixed fit/validation users, retained pre-V3 history only, label-free numeric diagnosis; no serving change or quality assertion",
        gap_scope="All retained pre-cut behaviors; strict previous time matches query-before-same-timestamp events; no evaluation/confirmation or post-release gap statistics",
        embedding_gate_scope="Full-width Current item vector only; not a proposed head slicing or trained gate implementation")
    torch.save(saved, args.output / "queries.pt")
    write_json(args.output / "summary.json", result)
    print(json.dumps(dict(status="complete", components=result["components_mean"],
                          gap_statistics=result["gap_statistics"], variants=overall)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--panel", type=Path, default=PANEL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=3)
    run(p.parse_args())

#!/usr/bin/env python3
"""Conditional fixed-16 observation check; run only after the free64 decision."""

import argparse
from collections import defaultdict

import pandas as pd
import torch
from design.diagnose_decoder_closure import free_code, load_views, main
from design.diagnose_query_holdout import batch_cache, panels, view_score
from design.diagnose_summary_objective import install_layer
from design.run import training_state

from hstu_kvcache.adaptation.reader import history_read, score


def experiment(current, scenes, mapper, history, cutover, fit_uids, uid_count, batch_size, out, target, canary):
    """Frozen functional-response32/128, 16 fit indices0,4,...,60; original held64.

    Only sixteen teacher responses enter each layer's solve. All ordinary fit64
    queries may be evaluated to obtain the matching lower-branch coordinates.
    Four unique fitting items remain in this predefined subset; no reselection.
    Teacher cache construction is still full-history computation.
    """
    states = [training_state(scene, current, mapper) for scene in scenes]
    mapper.diagnostic_attention_scale = current.blocks[0].attn.scale
    base, counts, active, views, layers = load_views(states, scenes, mapper, target, fit_uids, out)
    groups, panel_list, rows, solver_rows = defaultdict(list), [], [], []
    stride = 2 if target == 1 else 3
    for i, state in enumerate(states):
        groups[state.cache.seq_len].append(i)
        panel_list.append(panels(history, scenes[i].uid, cutover, float(scenes[i].query_delta.max()), base.device)[0])
    batches = [indices[j:j+batch_size] for indices in groups.values() for j in range(0, len(indices), batch_size)]
    for batch, indices in enumerate(batches):
        source, teacher = batch_cache([states[i].cache for i in indices]), batch_cache([scenes[i].teacher for i in indices])
        n = counts[indices]
        panel = {name: tuple(torch.cat([panel_list[i][name][j] for i in indices]) for j in range(2)) for name in ("fit64", "held64")}
        exact = {name: score(current, teacher, *p)[0] for name, p in panel.items()}
        raw = dict(scene_indices=indices, exact={k: v.cpu() for k, v in exact.items()})
        for method in ("functional_response32", "functional_response128"):
            b, a = (torch.zeros_like(v[indices]) for v in views[method])
            for l, layer in enumerate(layers[method]):
                _, trace = view_score(current, source, panel["fit64"], b, a, n, True)
                q = trace.queries[l][:, :, ::4]
                wanted = (history_read(current.blocks[l].attn, q, teacher.k[l], teacher.v[l])-
                          history_read(current.blocks[l].attn, q, source.k[l], source.v[l]))/n[:, None, None, None]
                latent, stats = free_code(q, wanted, layer)
                b[:, l], a[:, l] = install_layer(latent, layer["weights"], layer["query_center"], layer["query_scale"], active[indices, l])
                for j, i in enumerate(indices):
                    solver_rows.append(dict(target=target, uid=scenes[i].uid, scene=i, method=method, layer=l,
                        **{key: float(value[j]) for key, value in stats.items()}))
            for name, p in panel.items():
                z, trace = view_score(current, source, p, b, a, n, True)
                raw[f"{method}_{name}"] = z.cpu()
                error = (z.double()-exact[name].double()).square().mean(-1)
                for j, i in enumerate(indices):
                    rows.append(dict(target=target, uid=scenes[i].uid, scene=i,
                        group="fitting_uid" if scenes[i].uid in fit_uids else "diagnostic_uid",
                        state_group="continuous_cutover" if i < uid_count*stride and i % stride == 0 else "auxiliary",
                        method=method, path="free16", panel=name, logit_mse=float(error[j])))
        torch.save(raw, out / f"free16_raw_m{target}_batch{batch:03d}.pt")
    pd.DataFrame(rows).to_parquet(out / f"outputs_m{target}.parquet", index=False)
    pd.DataFrame(solver_rows).to_parquet(out / f"solvers_m{target}.parquet", index=False)
    print(f"M{target} free16 complete: {len(scenes)} scenes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-users", type=int, default=64)
    parser.add_argument("--diagnostic-users", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--estimate-seconds", type=int, required=True)
    main(parser.parse_args(), experiment)

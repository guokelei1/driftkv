"""Same-length execution of the original joint objective and optimizer batches.

No backbone, loss, user order, optimizer step, or epoch-selection rule changes.
Only independent users within each original optimizer batch execute together.
"""
from collections import Counter, defaultdict
import copy

import torch

from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.cachetide.adapter import operation_gate_query
from hstu_kvcache.methods.cachetide.calibration import _update_trainable, emit, rollout_score
from hstu_kvcache.methods.cachetide.session import append_corrected_event


def groups_for(rows, users, grouped=True, use_rollout=True):
    if not grouped:
        return [[uid] for uid in users]
    groups = defaultdict(list)
    for uid in users:
        row = rows[uid]
        tail = row.get("rollout") if use_rollout else None
        geometry = (row["cache"].seq_len, row["ids"].shape[1],
                    tail["prefix_length"] if tail else 0,
                    tail["items"].numel() if tail else 0)
        groups[geometry].append(uid)
    return list(groups.values())


def pack_rows(rows, users, use_rollout=True):
    if len(users) == 1:
        return rows[users[0]]
    picked = [rows[uid] for uid in users]
    row = dict(cache=HSTUKVCache(torch.cat([r["cache"].k for r in picked], 1),
        torch.cat([r["cache"].v for r in picked], 1), picked[0]["cache"].seq_len))
    for key in ("previous", "item", "item_ids", "ids", "delta", "wanted_logits"):
        if key == "item_ids" and picked[0].get(key) is None:
            row[key] = None
        else:
            row[key] = torch.cat([r[key] for r in picked], 1 if key == "previous" else 0)
    if use_rollout:
        row["rollout"] = dict(prefix_length=picked[0]["rollout"]["prefix_length"],
            **{key:torch.stack([r["rollout"][key] for r in picked])
               for key in ("items", "behaviors", "deltas")})
    return row


def batched_rollout(model, adapter, row):
    tail, original = row["rollout"], row["cache"]
    prefix = tail["prefix_length"]
    cache = HSTUKVCache(original.k[:, :, :prefix], original.v[:, :, :prefix], prefix)
    state = adapter.advance_codes(cache, row["previous"][:, :, :prefix],
        row["item"][:, :prefix],
        item_ids=None if row.get("item_ids") is None else row["item_ids"][:, :prefix])
    for index in range(tail["items"].shape[1]):
        cache, state, _ = append_corrected_event(model, cache, state, adapter,
            tail["items"][:, index:index+1], tail["behaviors"][:, index:index+1],
            tail["deltas"][:, index:index+1], adapter.max_length, code_count=prefix)
    return score(model, cache, row["ids"], row["delta"],
        history_read_fn=adapter.make_history_read(state, code_count=prefix,
            gate_query=operation_gate_query(adapter, model, row["ids"])))[0]


def forward(model, adapter, row, grouped=False, use_rollout=True):
    state = adapter.advance_codes(row["cache"], row["previous"], row["item"],
                                  item_ids=row.get("item_ids"))
    actual = score(model, row["cache"], row["ids"], row["delta"],
        history_read_fn=adapter.make_history_read(state,
            gate_query=operation_gate_query(adapter, model, row["ids"])))[0]
    static = (actual - row["wanted_logits"]).square().mean()
    rolled = output = None
    if use_rollout and row["rollout"]["items"].numel():
        output = batched_rollout(model, adapter, row) if grouped else rollout_score(model, adapter, row)[0]
        rolled = (output - row["wanted_logits"]).square().mean()
    combined = static if rolled is None else .5*static + .5*rolled
    return combined, static, rolled, actual, output


def _account_uid(adapter, row, costs, operations, *, backward, use_rollout):
    """Keep the original logical per-user ledger; batching introduces no padding."""
    n, queries = row["cache"].seq_len, row["ids"].numel()
    costs["joint_forward_token_queries"] += n*queries
    costs["joint_forward_calls"] += 1
    if operations is not None:
        operations.record_gate_item_input(adapter, items=queries, stage="joint_static_query")
        operations.record_update(adapter, length=n,
            backward=backward and _update_trainable(adapter.layers[0]), stage="joint_static_encoder")
        operations.record_query(adapter, length=n, queries=queries,
                                backward=backward, stage="joint_static_query")
        operations.record_score_loss(outputs=queries, backward=backward, stage="joint_static")
    if use_rollout and row["rollout"]["items"].numel():
        tail = row["rollout"]
        prefix, steps = tail["prefix_length"], tail["items"].numel()
        if operations is not None:
            operations.record_update(adapter, length=prefix,
                backward=backward and _update_trainable(adapter.layers[0]), stage="joint_rollout_encoder")
            for index in range(steps):
                operations.record_event_write(adapter, old_length=prefix+index, width=1,
                    coded_rows=prefix, backward=backward, stage="joint_rollout_write")
                operations.record_gate_item_input(adapter, items=1, stage="joint_rollout_write")
            operations.record_gate_item_input(adapter, items=queries, stage="joint_rollout_query")
            operations.record_query(adapter, length=n, queries=queries, coded_rows=prefix,
                                    backward=backward, stage="joint_rollout_query")
            operations.record_score_loss(outputs=queries, backward=backward, stage="joint_rollout")
            operations.add("joint_two_view_combination_flops", 3)
        costs["joint_rollout_behavior_queries"] += steps
        costs["joint_rollout_forward_calls"] += 1
        costs["joint_rollout_forward_token_queries"] += n*queries


def execute(model, adapter, rows, users, *, grouped=True, backward=False,
            use_rollout=True, costs=None, operations=None, execution_counts=None):
    """One original optimizer batch; returns detached per-group objective views.

    Backward accumulates each group's loss weighted by its original user count.
    The caller owns zero_grad, one clipping operation and one optimizer step.
    """
    outputs = []
    for members in groups_for(rows, users, grouped, use_rollout):
        row = pack_rows(rows, members, use_rollout)
        values = forward(model, adapter, row, grouped=len(members)>1, use_rollout=use_rollout)
        if costs is not None:
            for uid in members:
                _account_uid(adapter,rows[uid],costs,operations,backward=backward,use_rollout=use_rollout)
        if backward:
            loss = values[0]*(len(members)/len(users))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite joint score loss")
            loss.backward()
            if costs is not None:
                costs["joint_backward_calls"] += len(members)
        if execution_counts is not None:
            stage = "train" if backward else "validation"
            execution_counts[stage+"_group_forward_calls"] += 1
            execution_counts[stage+"_logical_users"] += len(members)
            execution_counts[stage+"_group_backward_calls"] += int(backward)
            execution_counts[stage+"_group_rollout_writes"] += (
                rows[members[0]]["rollout"]["items"].numel()
                if use_rollout and values[2] is not None else 0)
        outputs.append((members,[None if v is None else v.detach() for v in values]))
    return outputs


def fit_joint_batched(model, adapter, rows, train, validation, args, costs, operations=None):
    """Drop-in alternative to calibrate.fit_joint; returns its diagnostics + counts."""
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters,lr=args.learning_rate*.25)
    generator = torch.Generator().manual_seed(args.seed)
    use_rollout = getattr(args,"rollout_steps",0)>0
    execution_counts = Counter()

    @torch.no_grad()
    def validate():
        outputs = []
        # Validation has no optimizer ordering constraint, but keeps the same
        # <=batch_size memory ceiling and exact equal-user aggregation.
        for start in range(0,len(validation),args.batch_size):
            outputs.extend(execute(model,adapter,rows,validation[start:start+args.batch_size],
                backward=False,use_rollout=use_rollout,costs=costs,operations=operations,
                execution_counts=execution_counts))
        combined = sum(v[0]*len(u) for u,v in outputs)/len(validation)
        static = sum(v[1]*len(u) for u,v in outputs)/len(validation)
        rolled = [(u,v[2]) for u,v in outputs if v[2] is not None]
        return dict(validation_teacher_logit_mse=float(combined),
            validation_static_teacher_logit_mse=float(static),
            validation_rollout_teacher_logit_mse=(
                None if not rolled else float(sum(v*len(u) for u,v in rolled)/sum(len(u) for u,_ in rolled))))

    initial = validate()
    best, best_epoch = initial["validation_teacher_logit_mse"],0
    state = copy.deepcopy(adapter.state_dict())
    history = [dict(epoch=0,**initial)]
    first_gradients = None
    for epoch in range(1,args.joint_epochs+1):
        order = torch.randperm(len(train),generator=generator).tolist()
        for start in range(0,len(order),args.batch_size):
            users = [train[i] for i in order[start:start+args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            execute(model,adapter,rows,users,backward=True,use_rollout=use_rollout,
                costs=costs,operations=operations,execution_counts=execution_counts)
            if first_gradients is None:
                first_gradients = {n:float(p.grad.norm()) if p.grad is not None else None
                                   for n,p in adapter.named_parameters() if p.requires_grad}
                if operations is not None:
                    operations.record_gradient_diagnostic(
                        parameters=sum(p.numel() for p in parameters),tensors=len(parameters))
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            optimizer.step()
            execution_counts["optimizer_steps"] += 1
            if operations is not None:
                operations.record_optimizer(parameters=sum(p.numel() for p in parameters),stage="joint")
        metrics = validate()
        history.append(dict(epoch=epoch,**metrics))
        if metrics["validation_teacher_logit_mse"]<best:
            best,best_epoch,state=metrics["validation_teacher_logit_mse"],epoch,copy.deepcopy(adapter.state_dict())
        emit(dict(status="joint_epoch",epoch=epoch,**metrics,selected_epoch=best_epoch))
    adapter.load_state_dict(state)
    return dict(selected_epoch=best_epoch,history=history,first_gradients=first_gradients,
        objective_views="half_static_half_rollout" if use_rollout else "static_only",
        execution_group_calls=dict(execution_counts),
        execution_scope="Same original optimizer subsets and per-user logical operation ledger; same-N groups introduce no padding. Floating-point reductions need not be bitwise identical.")

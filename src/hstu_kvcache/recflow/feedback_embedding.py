"""Direct dense-buffer lookup gradients; full AdamW updates remain unchanged.

This removes an entire temporary table gradient. It does not skip zero-gradient
rows in AdamW: their usual momentum and weight decay still apply.
"""
import torch
import torch.nn.functional as F


class _Lookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, ids, gradient):
        ctx.save_for_backward(ids)
        ctx.gradient = gradient
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, output_grad):
        (ids,) = ctx.saved_tensors
        if output_grad.is_cuda:
            from .feedback_embedding_cuda import scatter_add
            scatter_add(ids, output_grad, ctx.gradient)
        else:
            ctx.gradient.index_add_(0, ids.flatten(), output_grad.reshape(-1, output_grad.shape[-1]))
        return None, None, None


def lookup(weight, ids):
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    return _Lookup.apply(weight, ids, weight.grad)

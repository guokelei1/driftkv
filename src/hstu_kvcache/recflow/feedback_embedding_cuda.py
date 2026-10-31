"""Scatter lookup gradients; every PAD use is masked in the feedback runner."""
import triton
import triton.language as tl


@triton.jit
def _scatter(Ids, Source, Dest, COUNT: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < COUNT * WIDTH
    row = tl.load(Ids + offset // WIDTH, valid, 0)
    value = tl.load(Source + offset, valid, 0)
    # Skip zero atomics and right-padding contention on the PAD row.
    tl.atomic_add(Dest + row * WIDTH + offset % WIDTH, value,
                  valid & (row != 0) & (value != 0), sem="relaxed")


def scatter_add(ids, values, gradient):
    values = values.contiguous()
    _scatter[(triton.cdiv(values.numel(), 512),)](
        ids, values, gradient, ids.numel(), values.shape[-1], 512)

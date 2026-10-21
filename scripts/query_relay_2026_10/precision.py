"""Optional anchor numerics; the frozen reader/updater prototype is unchanged.

A product of orthogonal matrices is orthogonal in exact arithmetic. Its saved
FP32 product is only approximately orthogonal, however, so its transpose can
be a noticeably inaccurate inverse when multiplied by a large Q/D offset.
This extension inverts the actual saved small blocks once per publication.
It still stores one FP32 code/anchor buffer per native row. Large offsets can
still cause FP32 cancellation; this is not an unlimited-precision guarantee.
"""
from dataclasses import dataclass, field

import torch

from query_relay_2026_10.adapter import ReleasePrefix, _flat, _state


@dataclass
class StableReleasePrefix(ReleasePrefix):
    """Alias existing P/Q/D; retain only their small block inverses additionally."""

    inverse_blocks: tuple = field(init=False, repr=False)

    def __post_init__(self):
        inverses=[]
        for index, layer in enumerate(self.layers):
            blocks=[]
            for branch in layer.branches:
                section=slice(branch.code_start,branch.code_start+branch.code_width)
                blocks.append(torch.linalg.inv(self.P[index,section,section].double())
                              .to(self.P.dtype))
            inverses.append(tuple(blocks))
        self.inverse_blocks=tuple(inverses)

    @classmethod
    def wrap(cls, prefix):
        if isinstance(prefix,cls):
            return prefix
        return cls(prefix.P,prefix.Q,prefix.D,prefix.layers,prefix.release)

    def anchorize(self,cache,current,item_ids,*,table_item_ids=None):
        rhs=_flat(current,self.layers[0].code_shape).to(self.P)-self._offset(cache,item_ids,table_item_ids)
        rows=[]
        for index,(layer,blocks) in enumerate(zip(self.layers,self.inverse_blocks,strict=True)):
            pieces=[]
            for branch,inverse in zip(layer.branches,blocks,strict=True):
                section=slice(branch.code_start,branch.code_start+branch.code_width)
                pieces.append(rhs[index,...,section]@inverse.T)
            rows.append(torch.cat(pieces,-1))
        return _state(torch.stack(rows).to(cache.k.dtype),code_shape=self.layers[0].code_shape)

    def inverse_cost(self):
        """LU inverse estimate; copies/casts excluded, FP64 compute disclosed."""
        widths=[branch.code_width for layer in self.layers for branch in layer.branches]
        tensors=[block for blocks in self.inverse_blocks for block in blocks]
        return dict(global_inverse_bytes=sum(x.numel()*x.element_size() for x in tensors),
            preparation_flops_estimate=sum((8*n**3+2)//3 for n in widths),
            preparation_arithmetic='FP64 small-block matrix inverse, stored in prefix dtype',
            estimate_convention='LU factorization plus inverse triangular solves, approximately 8/3 n^3 per block',
            extra_per_row_flops=0,extra_per_row_persistent_bytes=0,
            note='Row block matrix multiply has the original geometry. Existing P/Q/D are referenced without copying.')

    def storage_bytes(self):
        return super().storage_bytes()+self.inverse_cost()['global_inverse_bytes']


def stable_anchorize(prefix,cache,current,item_ids,*,table_item_ids=None):
    """Convenience call; retain the wrapped prefix to amortize inverse setup."""
    return StableReleasePrefix.wrap(prefix).anchorize(
        cache,current,item_ids,table_item_ids=table_item_ids)


def load_stable_prefix(path,adapter,*,device=None,expected_sha256=None):
    """Read an existing complete publication without modifying its artifact."""
    from query_relay_2026_10.publish import load_prefix
    prefix,metadata=load_prefix(path,adapter,device=device,expected_sha256=expected_sha256)
    stable=StableReleasePrefix.wrap(prefix)
    # The original publication metadata/identity continues to describe P/Q/D;
    # this separate field identifies the new, not previously evaluated path.
    metadata={**metadata,'anchor_numerics_extension':stable.inverse_cost()}
    return stable,metadata

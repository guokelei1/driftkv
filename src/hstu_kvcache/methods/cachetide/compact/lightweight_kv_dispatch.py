"""Use the retained fast full-history KV kernel only at exactly 1024 rows.

Every shorter history uses the bounded runtime-length kernel. This restores
the measured full-prefix fast path without compiling each observed length.
Both implementations and their original evidence remain unchanged.
"""
from .lightweight_kv_kernel import fused_kv_read as full_length_read
from .lightweight_kv_runtime_kernel import fused_kv_read as variable_length_read


def fused_kv_read(q,k,v,codes,key_decoder,value_decoder,*,scale=1.):
    reader=full_length_read if k.shape[1]==1024 else variable_length_read
    return reader(q,k,v,codes,key_decoder,value_decoder,scale=scale)


def make_history_read(adapter,state):
    def read(index,attention,q,k,v):
        if (attention.training or attention.position_bias is not None
                or attention.block_variant!='legacy' or attention.cfg.activation!='elu_plus1'):
            raise ValueError('KV dispatch implements frozen legacy ELU+1 history reads')
        if state.seq_len!=k.shape[1]:
            raise ValueError('KV codes and history rows differ')
        return fused_kv_read(q,k,v,state.codes[index],adapter.layers[index].key_decoder,
                             adapter.layers[index].decoder,scale=attention.scale)
    return read

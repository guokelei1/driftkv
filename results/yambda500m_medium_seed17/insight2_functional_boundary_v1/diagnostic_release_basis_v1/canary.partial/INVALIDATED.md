# Invalid partial canary

The run stopped on the first edge before producing score or summary artifacts.
`HSTUConfig.head_dim` is `None` for the Medium payload because the effective
head dimension is inferred from hidden size and head count; the diagnostic
incorrectly used that optional configuration field when rebuilding an S4
correction. No scientific metric can be read from this directory. The fixed
run reads `current.blocks[0].attn.head_dim` and writes to sibling `canary_v2/`.

# Adjudication invalidation

The raw canary remains valid. This first analysis incorrectly treated the
legacy helper stage name `final_readout` as a scalar readout. The tensor is in
fact the 192-dimensional final normalized query representation *before* the
scalar CC head, so it is an eligible S7 functional-boundary candidate.

The corrected, non-overwriting adjudication is in `../analysis_v2/`. No raw
score, energy, correctness record, contract, or threshold changed.


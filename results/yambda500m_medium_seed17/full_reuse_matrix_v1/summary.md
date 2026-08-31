# Medium D7/D14: Old Full / New Full / adjacent one-hop Reuse

Training/Full-only used GPU2/3 world size 2. Completed Reuse artifacts are preserved; remaining D14 Reuse uses GPU0/1/2/3 world size 4 after its raw-only runtime canary.

Status: **medium_full_then_reuse_matrix_complete** (32/32 Full-only cells).

| Recipe | Edge | E | Requests | Old AUC | New AUC | Reuse AUC | AUC recovery | Old loss | New loss | Reuse loss | Loss recovery | Reuse status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D7 | v0 → v1 | 3 | 30,413 | 0.635196 | 0.658151 | — | — | 0.316800 | 0.313301 | — | — | locked: full_only_quality_gate_failed |
| D7 | v0 → v1 | 7 | 72,053 | 0.627385 | 0.655353 | — | — | 0.327693 | 0.323798 | — | — | locked: full_only_quality_gate_failed |
| D7 | v1 → v2 | 3 | 30,064 | 0.663533 | 0.665362 | — | — | 0.284651 | 0.283644 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v1 → v2 | 7 | 68,627 | 0.668484 | 0.668890 | — | — | 0.298250 | 0.297610 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v2 → v3 | 3 | 29,987 | 0.667730 | 0.674250 | — | — | 0.288726 | 0.289289 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v2 → v3 | 7 | 68,541 | 0.657082 | 0.662167 | — | — | 0.301346 | 0.301206 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v3 → v4 | 3 | 27,510 | 0.650862 | 0.653316 | — | — | 0.356443 | 0.357058 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v3 → v4 | 7 | 64,727 | 0.657687 | 0.658919 | — | — | 0.334379 | 0.334432 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v4 → v5 | 3 | 27,000 | 0.620565 | 0.629351 | — | — | 0.325796 | 0.323035 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v4 → v5 | 7 | 66,320 | 0.633430 | 0.642333 | — | — | 0.323611 | 0.321558 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v5 → v6 | 3 | 26,926 | 0.633693 | 0.629761 | — | — | 0.367119 | 0.365002 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v5 → v6 | 7 | 65,254 | 0.643635 | 0.644721 | — | — | 0.333783 | 0.332552 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v6 → v7 | 3 | 26,643 | 0.664206 | 0.689405 | — | — | 0.323385 | 0.320363 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v6 → v7 | 7 | 66,529 | 0.672822 | 0.697065 | — | — | 0.347874 | 0.347325 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v7 → v8 | 3 | 29,242 | 0.667058 | 0.663552 | — | — | 0.303034 | 0.302467 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v7 → v8 | 7 | 67,842 | 0.666902 | 0.666709 | — | — | 0.326107 | 0.323788 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v8 → v9 | 3 | 27,306 | 0.642731 | 0.646303 | — | — | 0.326868 | 0.325976 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v8 → v9 | 7 | 65,995 | 0.643761 | 0.646921 | — | — | 0.335946 | 0.334690 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v9 → v10 | 3 | 31,388 | 0.664513 | 0.672394 | — | — | 0.276310 | 0.273320 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D7 | v9 → v10 | 7 | 72,074 | 0.674436 | 0.689808 | — | — | 0.291638 | 0.287747 | — | — | locked: parent_not_in_accepted_diagnostic_lineage |
| D14 | v0 → v1 | 3 | 30,064 | 0.644006 | 0.664409 | 0.656679 | +62.1% | 0.289193 | 0.284340 | 0.285562 | +74.8% | unlocked |
| D14 | v0 → v1 | 7 | 68,627 | 0.641208 | 0.665338 | 0.657544 | +67.7% | 0.302909 | 0.298241 | 0.299358 | +76.1% | unlocked |
| D14 | v0 → v1 | 14 | 137,168 | 0.635498 | 0.659177 | 0.653302 | +75.2% | 0.304501 | 0.300172 | 0.301090 | +78.8% | unlocked |
| D14 | v1 → v2 | 3 | 27,510 | 0.648291 | 0.647101 | 0.641617 | — | 0.359129 | 0.357385 | 0.359417 | -16.5% | unlocked |
| D14 | v1 → v2 | 7 | 64,727 | 0.651277 | 0.652690 | 0.649608 | -118.0% | 0.336681 | 0.335256 | 0.336244 | +30.7% | unlocked |
| D14 | v1 → v2 | 14 | 131,047 | 0.639586 | 0.640693 | 0.639055 | -48.0% | 0.330523 | 0.329579 | 0.330148 | +39.8% | unlocked |
| D14 | v2 → v3 | 3 | 26,926 | 0.602385 | 0.615553 | 0.602494 | +0.8% | 0.371866 | 0.366942 | 0.369469 | +48.7% | unlocked |
| D14 | v2 → v3 | 7 | 65,254 | 0.621931 | 0.634682 | 0.625313 | +26.5% | 0.336891 | 0.334226 | 0.336077 | +30.6% | unlocked |
| D14 | v2 → v3 | 14 | 131,783 | 0.637739 | 0.651257 | 0.641689 | +29.2% | 0.344300 | 0.341612 | 0.343246 | +39.2% | unlocked |
| D14 | v3 → v4 | 3 | 29,242 | 0.646094 | 0.655240 | 0.646075 | -0.2% | 0.305439 | 0.303832 | 0.305999 | -34.8% | unlocked |
| D14 | v3 → v4 | 7 | 67,842 | 0.647103 | 0.657131 | 0.647781 | +6.8% | 0.326616 | 0.325300 | 0.326582 | +2.6% | unlocked |
| D14 | v3 → v4 | 14 | 133,837 | 0.633708 | 0.649226 | 0.636473 | +17.8% | 0.332783 | 0.330657 | 0.332452 | +15.6% | unlocked |

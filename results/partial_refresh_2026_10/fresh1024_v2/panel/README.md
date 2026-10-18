# Frozen fresh-user panel

1024 users from the fixed 30k Medium population; 6809 requests across V2→V3→V4→V5. Selection excludes the prior calibration, reservation, development, Motivation and formal confirmation users named in selection.json. Users were selected before reading labels, without activity or request filtering.

users.json records all selected users and original fit/validation roles; histories.npz contains evaluation histories only. requests.parquet preserves original candidates, labels and request identities. binding.json freezes source identities, model metadata and timestamp rules. No previous scores or weights are copied.

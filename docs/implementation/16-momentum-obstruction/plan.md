# 16 — EA Test 3 momentum obstruction

## Goal

Use the published May-2010 terrain and hydrograph to test whether a transient
retains enough momentum to cross the obstruction between two depressions.

## Dataset and harness

The native two-metre DEM covers a 300 m × 100 m modelled area with 150 × 50
cells. The published 35-second hydrograph peaks at 65.5 m³/s and injects
1310 m³. Manning is 0.01 and the run ends at 900 s.

Run `inundation/bench/momentum_obstruction.py` through
`just benchmark-obstruction`. The default is `gpu_resident_batch`; other
retained Vulkan implementations are selectable with `--backends`.

## Gates

The release must pond at least 0.02 m beyond the obstruction while an
equal-volume still-water control leaves that point below 0.005 m. Both ponds
must finish below the crest, depth must remain finite and non-negative, relative
volume drift must remain below 5e-5, and repeated runs must agree within 2e-3.

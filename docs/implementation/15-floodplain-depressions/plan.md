# 15 — EA Test 2 floodplain depressions

## Goal

Exercise disconnected water bodies, wetting and drying, and low-momentum
inundation extent using the published May-2010 Test 2 dataset.

## Dataset and model

The 1201 × 1201 two-metre DEM is cell-averaged to a 100 × 100 finite-volume
mesh over the 2000 m square. The published hydrograph enters along the western
boundary through five equal volume sources. It injects 97,200 m³ over roughly
85 minutes; the simulation settles for 48 hours with Manning 0.03.

## Harness

Run `deceris/inundation/bench/floodplain_depressions.py` through
`just benchmark-depressions`. The default is `gpu_resident_batch`; other
retained Vulkan implementations are selectable with `--backends`.

## Gates

Relative mass error must remain at most 2e-4, depth finite and non-negative,
pond surfaces no more than 0.03 m above their terrain-derived sills, filled
basin capacity no greater than injected volume, at least 90% of water in ponded
storage, the eastern gauge column below 0.05 m, and repeated runs within 1e-4
of peak depth.

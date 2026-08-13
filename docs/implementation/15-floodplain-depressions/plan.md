# 15 — EA Test 2 floodplain depressions

## Goal

Exercise disconnected water bodies, wetting and drying, and low-momentum
inundation extent with a generated 4 × 4 egg-box floodplain.

## Harness

Run `deceris/inundation/bench/floodplain_depressions.py` through
`just benchmark-depressions`. The default is `gpu_resident_batch`; other
retained Vulkan implementations are selectable with `--backends`.

The 2000 m square uses 100 × 100 quads, 16 depressions, a northeast rise,
Manning 0.03, an 85-minute triangular inflow, and a 3600 s settling tail.

## Gates

Relative mass error must remain at most 5e-3, depth finite and non-negative, at
least three depressions ponded, points 15 and 16 below 1 mm, plateau wet
fraction at most 0.15, and repeated runs within 5e-4 of peak depth.

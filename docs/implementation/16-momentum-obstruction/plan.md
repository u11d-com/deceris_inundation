# 16 — EA Test 3 momentum over an obstruction

## Goal

Exercise inertia by releasing a still block from an elevated shelf, sending the
resulting bore through a valley and over a sill into a far bowl.

## Adaptation

The published open-outlet layout does not match this solver's reflective
boundaries. The closed adaptation sizes the release so static filling remains
below the sill. A paired still-water control distinguishes momentum-driven
run-up from numerical well-balance leakage.

## Harness

Run `deceris/inundation/bench/momentum_obstruction.py` through
`just benchmark-obstruction`. The default is `gpu_resident_batch`; other
retained Vulkan implementations are selectable with `--backends`.

## Gates

Relative volume drift must remain at most 5e-5, depth finite and non-negative,
the valley pond at least 0.05 m, both pond surfaces at least 0.10 m below the
crest, far-bowl depth at least 0.02 m after release, and far-bowl depth at most
0.005 m in the still-water control.

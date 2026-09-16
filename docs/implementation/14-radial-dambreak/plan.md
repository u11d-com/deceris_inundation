# 14 — Tier 1 radially symmetric dam-break benchmark

## Goal

Exercise 2D flux directionality and grid-orientation isotropy with a circular
dam-break whose bore crosses a Cartesian mesh.

## Reference and harness

A 2000-cell axisymmetric finite-volume solve supplies the radial reference.
Run `inundation/bench/radial_dambreak.py` through
`just benchmark-radial-dambreak`. The default is `gpu_resident_batch`; other
retained Vulkan implementations are selectable with `--backends`.

## Default setup

The 40 m square uses 200 × 200 quads, a 2.5 m radius column with 2.5 m inner
and 0.5 m outer depth, and a 1.5 s horizon.

## Gates

Depth L1 relative error must remain at most 0.10, front-radius error at most
0.05, angular-sector radius spread at most 0.08, relative volume drift at most
1e-5, and depth must remain finite and non-negative. Repeated runs must agree
within 1e-5 of peak depth.

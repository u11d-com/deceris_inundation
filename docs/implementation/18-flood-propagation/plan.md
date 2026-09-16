# 18 — EA Test 4 flood propagation

## Goal

Measure flood-front arrival, depth, and velocity over the published flat
frictional floodplain.

## Dataset and reference

The 1000 m × 2000 m plain uses 80,000 five-metre cells, Manning 0.05, a dry
start, and the published five-hour hydrograph injecting 285,000 m³. Reflection
across the inflow wall gives an axisymmetric reference before the front reaches
other boundaries. A fine radial solve supplies arrivals, depths, and speeds.

## Harness

Run `inundation/bench/flood_propagation.py` through
`just benchmark-propagation`. The default is `gpu_resident_batch`; other
retained Vulkan implementations are selectable with `--backends`.

## Gates

Relative volume drift must remain below 1e-3, front arrival error below 0.15,
depth L1 below 0.10, speed L1 below 0.25, off-axis excess below 0.02 m, and
depth finite and non-negative. Repeated runs must agree within 1e-4.

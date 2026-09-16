# Roadmap

## Current scope

Maintain the Vulkan/Kompute solver, its application workflow, mesh processing,
and reproducible analytical benchmarks.

## Current evidence

All retained analytical and EA gates pass with the MoltenVK reference. Grid
convergence shows expected profile order and confirms Test 4 front bias shrinks
with resolution. Its mass drift changes sign with mesh and stopping time; the
positivity clamp contributes exactly zero.

## Priorities

1. Build a float64 state reference for the measured mass-drift cases.
2. Use the reference to distinguish float32 arithmetic from discretization.
3. Optimize only after a measured Vulkan bottleneck identifies a useful target.

## Invariants

- Repeated runs remain deterministic within each benchmark's stated tolerance.
- Kernel restructuring preserves per-cell CSR gather order.
- New execution strategies match existing physical invariants before benchmark
  results are accepted.
- Numerical-scheme changes require a new baseline rather than relaxed checks.

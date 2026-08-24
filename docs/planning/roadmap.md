# Roadmap

## Current scope

Maintain the Vulkan/Kompute solver, its application workflow, mesh processing,
and reproducible analytical benchmarks.

## Current evidence

All retained analytical and EA gates pass with the MoltenVK reference. Grid
convergence shows expected profile order and confirms Test 4 front bias shrinks
with resolution. Its mass drift is a difference of opposing terms rather than a
single bounded error.

## Priorities

1. Instrument volume added by the positivity clamp and infer the remaining sink
   by difference at existing convergence levels.
2. Use the isolated sink scaling to decide whether a float64 state comparison
   is warranted.
3. Optimize only after a measured Vulkan bottleneck identifies a useful target.

## Invariants

- Repeated runs remain deterministic within each benchmark's stated tolerance.
- Kernel restructuring preserves per-cell CSR gather order.
- New execution strategies match existing physical invariants before benchmark
  results are accepted.
- Numerical-scheme changes require a new baseline rather than relaxed checks.

# Roadmap

## Current scope

Maintain the Vulkan/Kompute solver, its application workflow, mesh processing,
and reproducible analytical benchmarks.

## Current evidence

The Stoker, Ritter, radial dam-break, and floodplain-depression gates pass with
the MoltenVK reference. The momentum-obstruction case remains blocked by its
still-water control, which exposes the known hydrostatic well-balance gap.

## Priorities

1. Implement and validate the hydrostatic interface correction against the
   lake-at-rest and momentum-obstruction controls.
2. Preserve momentum warm starts and measure mass drift in float64 reporting.
3. Optimize only after a measured Vulkan bottleneck identifies a useful target.

## Invariants

- Repeated runs remain deterministic within each benchmark's stated tolerance.
- Kernel restructuring preserves per-cell CSR gather order.
- New execution strategies match existing physical invariants before benchmark
  results are accepted.
- Numerical-scheme changes require a new baseline rather than relaxed checks.

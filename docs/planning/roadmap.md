# Roadmap

## Current scope

Maintain the Vulkan/Kompute solver, its application workflow, mesh processing,
and reproducible analytical benchmarks.

## Current evidence

The Stoker, Ritter, radial dam-break, floodplain-depression, and corrected
lake-at-rest gates pass with the MoltenVK reference. The hydrostatic interface
correction removes the still-water obstruction leak.

## Priorities

1. Retune the momentum-obstruction release so genuine momentum overtops the sill
   without weakening its passing still-water control.
2. Preserve momentum warm starts and measure mass drift in float64 reporting.
3. Optimize only after a measured Vulkan bottleneck identifies a useful target.

## Invariants

- Repeated runs remain deterministic within each benchmark's stated tolerance.
- Kernel restructuring preserves per-cell CSR gather order.
- New execution strategies match existing physical invariants before benchmark
  results are accepted.
- Numerical-scheme changes require a new baseline rather than relaxed checks.

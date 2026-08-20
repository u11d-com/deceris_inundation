# Roadmap

## Current scope

Maintain the Vulkan/Kompute solver, its application workflow, mesh processing,
and reproducible analytical benchmarks.

## Current evidence

The Stoker, Ritter, radial dam-break, floodplain-depression, flood-propagation,
and corrected lake-at-rest gates pass with the MoltenVK reference. The
hydrostatic interface correction removes the still-water obstruction leak.

## Priorities

1. Run a grid-convergence study for the flood-propagation front bias and mass
   drift, including float64 reporting.
2. Retune the momentum-obstruction release so genuine momentum overtops the sill
   without weakening its passing still-water control.
3. Optimize only after a measured Vulkan bottleneck identifies a useful target.

## Invariants

- Repeated runs remain deterministic within each benchmark's stated tolerance.
- Kernel restructuring preserves per-cell CSR gather order.
- New execution strategies match existing physical invariants before benchmark
  results are accepted.
- Numerical-scheme changes require a new baseline rather than relaxed checks.

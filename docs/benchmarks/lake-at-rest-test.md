# Lake-at-Rest Vulkan Test

## Purpose

Measure the Vulkan solver's still-water behavior over non-flat bathymetry and
protect the synchronization fix with a reproducible numerical baseline.

## Scenario

Build a 128 by 128 quad grid on the unit square, apply a smooth Gaussian bed,
and Hilbert-reorder the resulting geometry. Run two water-surface variants:
fully wet at elevation 1.0 m and partially dry at elevation 0.5 m. Run each
with Manning values 1e-4 and 0.035.

Use gravity 9.81, dry tolerance 1e-4, CFL 0.45, workgroup size 256, maximum
timestep 0.05 s, and a 600 s horizon. No source terms are active.

## Metrics

Record maximum absolute momentum, maximum wet-cell velocity, relative volume
drift, finite-state checks, and final-state hashes for repeated runs. The
`fixed_dt_batch_barrier` implementation is the reference; cross-check the same
matrix with `gpu_resident_batch`.

See `lake-at-rest-baseline-results.md` for observed values and the barrier-race
root cause.

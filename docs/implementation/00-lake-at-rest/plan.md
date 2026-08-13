# Lake-at-Rest Vulkan Baseline Plan

## Goal

Measure still-water behavior over non-flat bathymetry and protect the Vulkan
synchronization fix with a reproducible numerical baseline.

Build a 128 by 128 quad grid, apply a smooth Gaussian bed, and run fully wet and
partially dry surfaces with Manning values 1e-4 and 0.035. Use gravity 9.81,
dry tolerance 1e-4, CFL 0.45, workgroup size 256, maximum timestep 0.05 s, and
a 600 s horizon.

Record maximum absolute momentum, maximum wet-cell velocity, relative volume
drift, finite-state checks, and repeated-run hashes. Use
`fixed_dt_batch_barrier` as the reference and cross-check
`gpu_resident_batch`.

# Lake-at-Rest Vulkan Baseline Plan

<<<<<<< HEAD
## Goal
=======
Referenced from `../../architecture/validation-plan.md` §1 (Tier 1 validation). This file
is a self-contained work order: it can be executed today, before any CUDA
code exists, and does not require external data, the 5M-cell mesh, or GPU
cluster access.
>>>>>>> 2f36b6f (fix: Docs rework 2)

Measure still-water behavior over non-flat bathymetry and protect the Vulkan
synchronization fix with a reproducible numerical baseline.

## Scenario

Build a 128 by 128 quad grid on the unit square, apply a smooth Gaussian bed,
and Hilbert-reorder the geometry. Run fully wet and partially dry surfaces with
Manning values 1e-4 and 0.035.

Use gravity 9.81, dry tolerance 1e-4, CFL 0.45, workgroup size 256, maximum
timestep 0.05 s, and a 600 s horizon. No source terms are active.

## Evidence

Record maximum absolute momentum, maximum wet-cell velocity, relative volume
drift, finite-state checks, and repeated-run hashes. Use
`fixed_dt_batch_barrier` as the reference and cross-check
`gpu_resident_batch`.

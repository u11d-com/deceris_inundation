# 14 — Tier 1 radially symmetric dam-break benchmark

## Goal

Close the last remaining Tier 1 analytical case in
`../../architecture/validation-plan.md` §1 / Go/No-Go criterion #1: the
**radially symmetric (circular) dam-break**. With effort 13 (1D
Stoker/Ritter) and `00-lake-at-rest` (C-property) landed, this is the
third and final closed-form correctness case in the checklist.

Why it is a distinct test, not a rerun of the 1D case: the outward bore
and inward rarefaction spread _radially_, so a circular front sweeps
across a Cartesian mesh. This exercises 2D flux directionality and
grid-orientation isotropy — a circular front reveals mesh-imprint /
anisotropy bugs (front faster along axes than diagonals, or worse, flow
that only propagates along grid lines) that a channel-aligned 1D shock
cannot.

## Reference (no closed form)

The circular dam-break has no closed-form solution. The reference is a
fine-grid 1D **axisymmetric** finite-volume solve
(`bench/common.solve_radial_dambreak`) of

$$
\partial_t(r h) + \partial_r(r h u) = 0, \qquad
\partial_t(r h u) + \partial_r\!\big(r(h u^2 + \tfrac{1}{2} g h^2)\big) = \tfrac{1}{2} g h^2 .
$$

First-order HLL flux, SSP-RK2 (Heun) time stepping, `n_cells = 2000` over
`[0, r_max]`. The `r`-weighting makes the inner (`r = 0`) face flux vanish
by construction (radial symmetry) and conserves annular mass to round-off.
At `r_max = L/2` (the inscribed-circle radius) the wave is still far away,
so the outer boundary is inert. Pure-NumPy unit tests
(`tests/unit/test_bench_radial_dambreak.py`) cover positivity, annular
mass conservation, the undisturbed outer layer, the inner rarefaction,
front-position bounds, grid convergence, and determinism.

The coarse 2D solver's depth is azimuthally binned onto the same radii
(only bins whose full annulus lies inside the square domain, i.e.
`r < L/2`) and gated against the reference profile.

## Harness

`deceris/inundation/bench/radial_dambreak.py`, run via
`just benchmark-radial-dambreak`. Backends are configurable
(`--backends`). Default is `gpu_resident_batch`, correct on both
macOS/MoltenVK and CUDA/Linux; on the GPU node add
`fixed_dt_batch_barrier` (production reference) and
`cuda_graphs`/`cuda_streams`.

## Setup (defaults, all overridable)

| Parameter             | Value                                                         |
| --------------------- | ------------------------------------------------------------- |
| Domain                | 40 m × 40 m, flat frictionless bed (`n = 1e-4`, solver floor) |
| Grid                  | 200 × 200 quads (dx = dy = 0.2 m)                             |
| Dam center / radius   | (20 m, 20 m) / 2.5 m                                          |
| Inner depth           | 2.5 m                                                         |
| Outer depth           | 0.5 m (wet bed → outward bore)                                |
| End time              | 1.5 s (float32-exact stop; bore stays well clear of walls)    |
| dt_max / cfl_interval | 0.05 s / 5                                                    |
| Reference cells       | 2000 over [0, 20 m]                                           |

## Gates (per validation-plan.md §1: documented, not implicit)

| Gate                                        | Tolerance                                        | Justification                                                                                                                                                                                                                                                               |
| ------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Depth L1 relative error vs radial reference | ≤ 0.10                                           | First-order FV resolving a _circular_ bore on a Cartesian grid carries the 1D shock error plus front staircasing; the circular dam-break literature puts first-order L1 in ~3–8 % (Toro, _Shock-Capturing Methods for Free-Surface Shallow Flows_, circular dam-break test) |
| Bore-front radius error                     | ≤ 0.05 of reference radius                       | First-order fronts lag slightly                                                                                                                                                                                                                                             |
| Front-radius isotropy                       | ≤ 0.08 relative spread across 12 angular sectors | Core 2D directional-correctness check: on a Cartesian grid the circular front is marginally faster along the axes than the diagonals (4-fold imprint)                                                                                                                       |
| Volume drift (zero-source run)              | ≤ 1e-5 relative                                  | float32 accumulation error only                                                                                                                                                                                                                                             |
| Negative depths                             | none                                             | Solver clamps; gate catches clamp regressions                                                                                                                                                                                                                               |
| Run-to-run reproducibility                  | ≤ 1e-5 relative to peak depth (`--repeats ≥ 2`)  | GPU `atomicAdd` flux scatter is not bit-reproducible; healthy runs reproduce to ~1e-6                                                                                                                                                                                       |

## Performance measurement

Per backend: `--warmup` runs excluded, median wall seconds over
`--repeats`, steps/s and cell-steps/s. Timed runs default to a single
final snapshot. `--gif` renders a **separate untimed** run into a radial
depth-profile animation (numerical vs reference); `--gif-2d` renders a
top-down 2D depth heatmap from the same untimed run (directly shows the
circular spreading / any grid imprint). Frame count is snapped to the
nearest power of two so `t_end / frames` is float32-exact
(`gpu_resident_batch`'s device-side time advance stalls otherwise).

## Artifacts

`--output-dir` (default `.tmp/radial-dambreak-bench/`): generated square
mesh, `summary.json` (gates + reference front radius + correctness/perf
metrics), optional `circular-<backend>.gif` (radial profile) and
`circular-<backend>-2d.gif` (top-down heatmap).

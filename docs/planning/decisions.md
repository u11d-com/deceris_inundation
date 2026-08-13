# Decisions

Settled engineering choices and their rationale. Open work belongs in
`roadmap.md`; measurements belong under `../implementation/`.

## Vulkan reference implementation

`fixed_dt_batch_barrier` is the validated barrier reference. The older
`fixed_dt_batch` path omitted a compute barrier between dependent dispatches,
causing a read-after-write race, rapid divergence, and nondeterminism.
`gpu_resident_batch` independently confirms the corrected results.

## Mesh representation

Flux and update stages retain CSR adjacency with packed signed edge
orientation. Fixed per-cell gather order supports deterministic accumulation;
int32 CSR offsets fail explicitly before overflow.

## Simulated-time precision

Long-horizon simulated time must use float64 or integer representation. Float32
resolution becomes too coarse for ordinary timesteps at production horizons.

## Momentum-obstruction benchmark

The published open-outlet case does not match reflective boundaries. The
retained elevated-release case pairs every release with a still-water control;
control leakage is never credited as momentum.

## Hydrostatic interface correction

The Audusse source correction is part of both Vulkan GLSL flux variants. Each
cell uses its own actual and reconstructed depth along its own outward normal.
This reduces lake-at-rest velocity from roughly 0.3–0.5 m/s to below 1e-4 m/s
and removes the obstruction control leak. See
`../implementation/17-audusse-well-balance-correction/results.md`.

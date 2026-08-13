# Decisions

Settled engineering choices and their rationale. Open work belongs in
`roadmap.md`; measurements belong under `../implementation/`.

## Vulkan reference implementation

`fixed_dt_batch_barrier` is the validated reference. The older
`fixed_dt_batch` path omitted a compute barrier between dependent dispatches,
causing a read-after-write race, rapid divergence, and nondeterminism.
`gpu_resident_batch` independently confirms the corrected results.

The lake-at-rest matrix and rationale are recorded in
`../implementation/00-lake-at-rest/results.md`.

## Mesh representation

Flux and update stages retain CSR adjacency with packed signed edge
orientation. Fixed per-cell gather order supports deterministic accumulation;
int32 CSR offsets fail explicitly before overflow.

## Simulated-time precision

Long-horizon simulated time must use float64 or integer representation. Float32
resolution becomes too coarse for ordinary timesteps at production horizons.

## Momentum-obstruction benchmark

The published open-outlet case does not match this solver's reflective boundary
conditions. The retained case uses an elevated still-water release, a valley,
and a sill so reaching the far bowl requires momentum-driven run-up rather than
static filling. A paired still-water control must leave the far bowl dry; a
control leak is reported as a well-balance failure, not credited as momentum.

See `../implementation/16-momentum-obstruction/`.

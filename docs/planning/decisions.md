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

## Momentum state

Initial depth and both momentum components are preserved across workflow
preparation and phase resets. Final momentum downloads are part of the retained
warm-start and velocity-scoring contract.

## Hydrostatic interface correction

The Audusse source correction is part of both Vulkan GLSL flux variants. Each
cell uses its own actual and reconstructed depth along its own outward normal.
This reduces lake-at-rest velocity from roughly 0.3–0.5 m/s to below 1e-4 m/s.

## Momentum-obstruction benchmark

The published Test 3 dataset is accepted only with a paired equal-volume
still-water control. After the hydrostatic correction, the release ponds
0.047 m beyond the obstruction while the control remains exactly dry. This
makes the far pond a momentum signature rather than numerical leakage.

## Float32 mass floor

Flood-propagation drift worsened under timestep and grid refinement and then
saturated. This is consistent with small `dt * dh` increments being absorbed
by float32 state, but remains a hypothesis until a float64 state comparison.
The 1e-3 mass gate applies only to the measured Test 4 configuration.

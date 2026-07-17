# SWE Solver Architecture

The solver advances the shallow-water equations on an unstructured mesh with
HLLC fluxes and SSP-RK2 integration. NumPy loads and preprocesses geometry;
Vulkan/Kompute executes the timestep kernels.

## Pipeline

1. Load mesh cells and build edge plus CSR adjacency.
2. Optionally Hilbert-reorder cells and cache the geometry.
3. Compile GLSL compute shaders to SPIR-V.
4. Dispatch flux, update, source, and CFL stages through a selected Vulkan
   synchronization strategy.
5. Download snapshots at time- or step-based cadence.

`fixed_dt_batch_barrier` is the validated barrier-based reference.
`gpu_resident_batch` keeps control state on the device to reduce host
synchronization.

## Synchronization

Batched dispatches require compute barriers between a stage's writes and the
next stage's reads. The barrier implementation fixed the race present in the
unbarriered fixed-step batch and agrees with the independently barriered
GPU-resident implementation on the lake-at-rest matrix.

## Mesh and precision constraints

Geometry retains Hilbert ordering, content-keyed caching, dense and CSR
adjacency, and packed signed edge orientation. CSR offsets are int32 and reject
meshes whose slot count would overflow. Initial and warm-start states preserve
depth and both momentum components.

See `../implementation/00-lake-at-rest/` for retained evidence.

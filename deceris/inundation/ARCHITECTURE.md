# SWE Solver Architecture

The solver advances the shallow-water equations on an unstructured mesh with
HLLC fluxes and SSP-RK2 integration. NumPy loads and preprocesses geometry;
Vulkan/Kompute runs the timestep kernels.

## Pipeline

1. Load mesh cells and build edge plus CSR adjacency.
2. Optionally Hilbert-reorder cells and cache the resulting geometry.
3. Compile GLSL compute shaders to SPIR-V.
4. Advance flux, update, source, and CFL stages through a selected Vulkan
   synchronization strategy.
5. Download snapshots at time- or step-based cadence.

The `fixed_dt_batch_barrier` implementation is the numerical reference.
`gpu_resident_batch` reduces host synchronization for benchmark runs. Geometry
and solver state use float32; accumulated reporting values use wider host-side
arithmetic where required.

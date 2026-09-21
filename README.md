# deceris-inundation

GPU-accelerated shallow-water flood modelling on polygonal meshes.
NumPy preprocesses the mesh; Kompute dispatches the Vulkan compute shaders.

## How it works

`SWEWorkflow` has two stages: preparation and simulation.

1. **Prepare the mesh.** Read polygon cells and bed elevations from GeoPackage,
   Shapefile, Parquet/GeoParquet, or OBJ input.
2. **Build the geometry.** NumPy computes cell and edge geometry plus signed
   adjacency. The result can be Hilbert-reordered and cached for reuse.
3. **Prepare the GPU.** Compile the GLSL kernels to SPIR-V, allocate Vulkan
   buffers, and upload the initial state and mesh data.
4. **Run the simulation.** For each timestep, Vulkan computes HLLC edge fluxes,
   applies source terms, updates depth and momentum, and periodically estimates
   a stable CFL timestep.
5. **Return results.** The workflow records depth snapshots at the configured
   output interval and can optionally capture momentum snapshots.

Call `prepare()` once before `run()`. `run()` advances one or more
`SimulationPhase` objects and returns a `WorkflowResult`.

`fixed_dt_batch_barrier` is the validated reference implementation.
`gpu_resident_batch` reduces host synchronization for GPU-resident runs. The
top-level `inundation` package exports `SWEWorkflow`, `WorkflowConfig`,
`SimulationPhase`, `PointSource`, and `WorkflowResult`.

## Supported runtime

The supported development environment is Docker Compose. The image contains
Python 3.12, all optional Python dependencies, Vulkan development tools, and
the Kompute binding. Source code is bind-mounted into the running container.

For GPU execution, the host must provide a working Vulkan driver and ICD. The
production path uses an NVIDIA node and Apptainer's `--nv` passthrough. The
macOS host path is useful for the patched Kompute development setup, but is
not the production GPU environment.

## Development setup

Build the image once, start the persistent container, then run checks:

```sh
just image-build
just up
just check
```

Stop the container when finished:

```sh
just down
```

`just up` assumes the `deceris-inundation:dev` image already exists. Rebuild
it after changing `Dockerfile`, `pyproject.toml`, `uv.lock`,
`inundation/vulkan/patches/`, or
`inundation/scripts/build-kp-linux.sh`. Source changes under the
bind-mounted repository do not require an image rebuild.

## macOS development

Docker Desktop on macOS is suitable for repository checks and packaging, but
it does not provide the host Apple GPU as a Vulkan device inside this Linux
container. Do not use the default Docker-backed `just benchmark-*` recipes for
GPU execution on macOS.

Use the native host environment for Vulkan work through MoltenVK:

```sh
brew install uv cmake molten-vk vulkan-headers vulkan-loader vulkan-tools glslang
uv sync --all-extras
inundation/scripts/build-kp-macos.sh
export VK_ICD_FILENAMES="$(brew --prefix)/etc/vulkan/icd.d/MoltenVK_icd.json"
```

If `patch` is unavailable, install the Xcode command-line tools with
`xcode-select --install`. The Kompute build script validates a real
`kp.Manager()` round trip against MoltenVK and installs the patched binding in
`.venv`.

Run checks natively by bypassing the Docker prefix:

```sh
IN_CONTAINER=1 just lint
IN_CONTAINER=1 just typecheck
IN_CONTAINER=1 just test
```

Run GPU benchmarks either through the same `IN_CONTAINER=1` switch:

```sh
IN_CONTAINER=1 just benchmark-lake --backend gpu_resident_batch
```

or invoke the host interpreter directly:

```sh
.venv/bin/python -m inundation.bench.lake_at_rest \
  --backend gpu_resident_batch
```

The macOS/MoltenVK path is for development and correctness checks. Linux
NVIDIA nodes remain the production performance environment; the Apptainer
recipe and `--nv` passthrough do not apply to macOS.

## Running the solver

The workflow accepts `.gpkg`, `.shp`, `.parquet`, `.geoparquet`, and `.obj`
meshes. Polygon features need at least three vertices. GIS formats require a
`z_mean` column supplying per-cell bed elevation; OBJ files require a Z value
on every vertex, which is averaged per cell. Meshes without elevation data
are rejected.

Minimal application usage:

```python
from inundation import PointSource, SimulationPhase, SWEWorkflow, WorkflowConfig

config = WorkflowConfig(
    mesh_source="mesh.gpkg",
    manning_n=0.035,
    output_interval_s=300.0,
    dt_max=0.05,
    cfl_interval=1,
)
workflow = SWEWorkflow(config)
workflow.prepare()  # load mesh, compile shaders, allocate Vulkan resources
result = workflow.run([
    SimulationPhase(
        duration_s=3600.0,
        sources=[PointSource(10.0, (100.0, 200.0), 2.0)],
    ),
])
print(result.h_final, result.volume_final_m3)
```

`prepare()` must run before `run()`. Set `geometry_cache_dir` in
`WorkflowConfig` to reuse the processed mesh between runs. See
`inundation/workflow.py` for the complete configuration surface.

## Benchmarks

On Linux, or when the Docker-backed development environment is available,
benchmark recipes run inside the development container. On macOS, use the
native workflow above instead.

```sh
just benchmark-lake --help
just benchmark-radial-dambreak --help
just benchmark-depressions --help
just benchmark-obstruction --help
just benchmark-propagation --help
just benchmark-report --help
```

The lake-at-rest benchmark is a quick correctness starting point:

```sh
just benchmark-lake --backend fixed_dt_batch_barrier --flat-bed --hash
```

Detailed benchmark contracts and flags are documented beside their harnesses:

- [lake-at-rest](inundation/bench/lake_at_rest.md)
- [analytical dam-break](inundation/bench/dambreak.md)
- [radial dam-break](inundation/bench/radial_dambreak.md)
- [floodplain depressions](inundation/bench/floodplain_depressions.md)
- [momentum obstruction](inundation/bench/momentum_obstruction.md)
- [flood propagation](inundation/bench/flood_propagation.md)

## Production Apptainer image

Build the SIF from the already-built Docker image and run a benchmark with
NVIDIA Vulkan passthrough:

```sh
just image-build
just apptainer-build
just apptainer-run -- python -m inundation.bench.lake_at_rest --help
```

The repository is bind-mounted at `/workspace`; code changes do not require a
new image. Rebuild the base image when its dependencies or Kompute build inputs
change.

## Project layout

- `inundation/` — solver package and workflow API
- `inundation/vulkan/` — Vulkan/Kompute implementations and shaders
- `inundation/mesh/` — mesh loading, geometry, and caching
- `inundation/bench/` — benchmark harnesses and reports
- `tests/` — unit tests
- `docs/architecture/overview.md` — solver architecture and invariants

Optional host-only dependencies can be installed with:

```sh
uv sync --all-extras
```

Linux host execution also needs the Kompute binding built by
`inundation/scripts/build-kp-linux.sh`; the Docker image performs that
step automatically.

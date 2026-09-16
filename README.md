# deceris-inundation

GPU-accelerated shallow-water flood modelling on polygonal meshes.
NumPy preprocesses the mesh; Kompute dispatches the Vulkan compute shaders.

## How it works

The runtime pipeline is:

1. Load mesh cells and optional bed elevations.
2. Build edge and CSR adjacency, then optionally Hilbert-reorder and cache it.
3. Compile the GLSL compute shaders to SPIR-V.
4. Run HLLC flux, source, update, and CFL stages through a Vulkan solver.
5. Download depth and momentum snapshots at the configured cadence.

`fixed_dt_batch_barrier` is the validated reference implementation. The
`gpu_resident_batch` implementation reduces host synchronization. The public
Python API is `SWEWorkflow`, `WorkflowConfig`, `SimulationPhase`, and
`PointSource` from `deceris.inundation`.

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
it after changing `Dockerfile`, `pyproject.toml`, `uv.lock`, or the Kompute
build inputs under `deceris/inundation/vulkan/` and
`deceris/inundation/scripts/`.

## Running the solver

The workflow accepts `.gpkg`, `.shp`, `.parquet`, `.geoparquet`, and `.obj`
meshes. Polygon features need at least three vertices. For GIS formats, an
optional `z_mean` column supplies per-cell bed elevation; OBJ Z coordinates
are ignored. Without `z_mean`, the bed is initialized at zero.

Minimal application usage:

```python
from deceris.inundation import PointSource, SimulationPhase, SWEWorkflow, WorkflowConfig

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
`deceris/inundation/workflow.py` for the complete configuration surface.

## Benchmarks

All benchmark recipes run inside the development container:

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

- [lake-at-rest](deceris/inundation/bench/lake_at_rest.md)
- [analytical dam-break](deceris/inundation/bench/dambreak.md)
- [radial dam-break](deceris/inundation/bench/radial_dambreak.md)
- [floodplain depressions](deceris/inundation/bench/floodplain_depressions.md)
- [momentum obstruction](deceris/inundation/bench/momentum_obstruction.md)
- [flood propagation](deceris/inundation/bench/flood_propagation.md)

## Production Apptainer image

Build the SIF from the already-built Docker image and run a benchmark with
NVIDIA Vulkan passthrough:

```sh
just image-build
just apptainer-build
just apptainer-run -- python -m deceris.inundation.bench.lake_at_rest --help
```

The repository is bind-mounted at `/workspace`; code changes do not require a
new image. Rebuild the base image when its dependencies or Kompute build inputs
change.

## Project layout

- `deceris/inundation/` — solver package and workflow API
- `deceris/inundation/vulkan/` — Vulkan/Kompute implementations and shaders
- `deceris/inundation/mesh/` — mesh loading, geometry, and caching
- `deceris/inundation/bench/` — benchmark harnesses and reports
- `tests/` — unit tests
- `docs/architecture/overview.md` — solver architecture and invariants

Optional host-only dependencies can be installed with:

```sh
uv sync --all-extras
```

Host execution also needs the Kompute binding built by
`deceris/inundation/scripts/build-kp-linux.sh`; the Docker image performs that
step automatically.

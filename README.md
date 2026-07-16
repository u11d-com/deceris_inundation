# deceris-inundation

GPU-accelerated shallow-water flood modelling with a Vulkan/Kompute solver.
NumPy handles mesh preprocessing and shared benchmark analysis.

## Development

Docker is the supported development environment:

```sh
just image-build
just up
just check
just down
```

## Optional dependencies

```sh
uv sync --extra mesh-parquet --extra mesh-gis --extra viz
```

The Kompute binding is built by `deceris/inundation/build-kp-linux.sh` in the
development image.

## Benchmarks

```sh
just benchmark-lake -- --help
```

The solver package lives under `deceris/inundation/`; unit tests live under
`tests/unit/`. Production execution uses `inundation.def` and the
`apptainer-run` recipe with Vulkan driver passthrough.

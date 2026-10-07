# Mesh Geometry Cache Artifacts — Results

**Status: done** (2026-10-07). Plan: [`plan.md`](plan.md).

## What changed

- `inundation.mesh.build_geometry_cache(mesh_source, out_path, *, ...)` runs the
  same load → `build_geometry` → `hilbert_reorder` chain as `prepare()` and
  writes a self-contained artifact. No GPU needed.
- `inundation.mesh.load_geometry_cache_artifact(path)` returns
  `(MeshGeometry, perm, GeometryCacheMeta)` and raises on a missing file or a
  foreign cache version.
- `WorkflowConfig.geometry_cache_source` + `mesh_source: str | None`; the
  artifact wins over `geometry_cache_dir` and `mesh_source`.

## Reconciliation with upstream

Rebased onto upstream's required-bed-elevation work (`zb_from_file` is now
keyword-only and mandatory, OBJ meshes must carry Z, and `MeshGeometry` gained
`cell_vertices`/`cell_vertex_ptr`/`cell_bbox` for point-source containment):

- `load_mesh_file` returns five values with `zb` no longer optional —
  `(verts, faces, face_offsets, zb, manning_n)`; OBJ bed = mean of face vertex Z.
- Those new array fields join `_GEOMETRY_ARRAY_FIELDS`, so the cache format
  version is the union of both changes: **4**.
- `build_geometry` callers pass `zb_from_file` always, plus
  `manning_n_from_file` when the mesh carries the column.

## Evidence

Smoke run in the dev container, after which the source mesh files were deleted
before `prepare()`:

```text
cache version: 4 | artifact: 0.92 MB
meta: GeometryCacheMeta(version=4, mesh_source='.../mesh.parquet', use_hilbert_reorder=True)
matches direct build: True
fields: 4000 8130 | manning_n: None
roughness aligned: True
mesh files deleted: True True
[prepare] geometry cache artifact loaded (.../mesh.geometry.npz; built from .../mesh.parquet, hilbert=True)
ERROR: vkEnumeratePhysicalDevices: Invalid instance
```

- 4000-cell / 8130-edge mesh: the artifact reloads to exactly the arrays a
  direct `build_geometry()` + `hilbert_reorder()` produces — `zb`, `area`,
  `perm`, and upstream's `cell_vertices`/`cell_vertex_ptr`/`cell_bbox`.
- Mesh-sourced Manning roughness survives the round-trip in solver order
  (`manning_n == source_n[perm]`); a mesh without the column reloads as `None`.
- The cache-only path needs no mesh file: both source meshes were deleted, and
  `prepare()` loaded the geometry from the artifact and proceeded to shader
  compilation and Vulkan device creation.
- `just check` green: ruff, `ruff format --check`, pyright strict (0 errors),
  pymarkdown, pytest — 155 passed, including
  `tests/unit/test_workflow_cache_only.py` (artifact-only `prepare()`,
  precedence over `mesh_source`, stale-version rejection) and
  `tests/unit/test_geometry_cache.py` (artifact round-trip, version and
  missing-file rejection, real mesh file → artifact with roughness).

## Limit

A full end-to-end `run()` from a cache-only config was **not** exercised here:
the dev container has no Vulkan ICD (`/dev/dri` and `/usr/share/vulkan/icd.d`
are both absent), so device creation aborts. The cache-only path is verified up
to that boundary, with CPU-only solver stubs, in
`tests/unit/test_workflow_cache_only.py`.

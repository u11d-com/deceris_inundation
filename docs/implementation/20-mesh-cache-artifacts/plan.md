# Mesh Geometry Cache Artifacts — Plan

**Status: done.** Extends the geometry-cache layer rather than the solver
core; the solver-facing `prepare()`/`run()` contract is unchanged.

## Goal

Let a solver job run from a prebuilt geometry artifact alone, with no access to
the original mesh file.

Motivation: production pipelines prepare meshes in an earlier ("silver") stage
and run the solver in a later job. A cache that can travel on its own removes
the requirement that every solver job also download the source mesh.

## Problem

`geometry_cache_dir` was a cache *behind* the mesh, not an *input instead of*
it:

- `WorkflowConfig.mesh_source: str` was required, so a cache-only config was
  impossible to express.
- The cache key is a blake2b over the mesh file's bytes, so a hit still had to
  open — and therefore possess — the mesh file.

## Design

Two layers in `inundation.mesh.cache`:

| Layer | Produces | Consumes | On failure |
|---|---|---|---|
| Key-addressed directory cache | `save_geometry_cache(dir, key, ...)` | `load_geometry_cache(dir, key)` | miss → rebuild from the mesh |
| Explicit artifact file | `build_geometry_cache(mesh, out)` | `load_geometry_cache_artifact(path)` | raise — the artifact *is* the geometry source |

An artifact is one `.npz` holding the same geometry and `perm` arrays as the
directory cache, plus a JSON provenance blob (`GeometryCacheMeta`: cache
version, mesh source, reorder mode). `WorkflowConfig.geometry_cache_source`
points at it; when set it takes precedence over `geometry_cache_dir` and
`mesh_source` (which may be `None`), and its reorder mode must match
`WorkflowConfig.use_hilbert_reorder` or the load is rejected.

Resolution order in `prepare()`: artifact → directory cache → mesh file.

## Tasks

1. Public `build_geometry_cache()` / `load_geometry_cache_artifact()` in
   `inundation.mesh`, exported through the subpackage `__all__`.
2. Cache-version validation inside the artifact loader, so an artifact from a
   different release fails loudly instead of running the wrong geometry.
3. `WorkflowConfig`: `mesh_source: str | None`, new `geometry_cache_source`;
   `__post_init__` rejects the neither-set case.
4. Shared `_mesh_vertices_for_plot()` for the benchmark plotting helpers, which
   re-read the mesh for raw vertices and report a clear error when there is no
   mesh file to re-read (cache-only run).
5. Tests: artifact round-trip, `manning_n is None` encoding, missing-file and
   stale-version rejection, cache-only `prepare()` with mesh reads forbidden.

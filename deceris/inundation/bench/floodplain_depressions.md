# Floodplain-depressions benchmark (EA Test 2)

Overland-flow correctness benchmark — UK Environment Agency "Benchmarking of
2D Hydraulic Modelling Packages" **Test 2, filling of floodplain
depressions**, run from the published May-2010 dataset. Harness:
[floodplain_depressions.py](floodplain_depressions.py). Effort:
[docs/implementation/15-floodplain-depressions/](../../../docs/implementation/15-floodplain-depressions/).

## What it validates

A dry 2000 m × 2000 m floodplain — a "flattened egg box" of 16 ~0.5 m
depressions on a plane falling ~2 m along the NW→SE diagonal — is flooded by
a slow inflow hydrograph entering at the high (NW) corner. Unlike the
dam-break cases, this exercises the parts of the solver a channel-aligned
shock never touches: disconnected water bodies, wetting/drying of dry
floodplain, and low-momentum inundation extent, judged on the _final_ ponded
distribution.

The settled endpoint has no closed form and no defensible routed prediction
(see [results.md](../../../docs/implementation/15-floodplain-depressions/results.md)
for the fill-spill cascade that was tried and rejected). What the terrain
_does_ fix is a set of **bounds**: `bench/common.depression_basin` walks
downhill from each output point to its basin floor and raises a level until
the pool finds a lower outlet, yielding that depression's **sill elevation**
and **storage capacity**. Water at rest cannot stand above its sill, and the
depressions that fill cannot together hold more than was injected. The gates
are those bounds plus conservation.

## Dataset

`benchmark_assets/Test2 dataset 2010` (tracked in the repo; seven required EA Test 2/3/4 input files are committed,
`--dataset-dir` to override):

| File                     | Use                                                |
| ------------------------ | -------------------------------------------------- |
| `test2DEM.asc`           | 1201 × 1201 georeferenced 2 m raster (200 m apron) |
| `Test2_BC.csv`           | inflow hydrograph, minutes vs cumecs               |
| `Test2output.csv`        | 16 output points, one per depression centre        |
| `Test2ActiveArea_region` | modelled area — x, y ∈ [0, 2000] m                 |
| `Test2BC_polyline`       | inflow line — western boundary, y ∈ [1900, 2000] m |

The 2 m raster is cell-averaged onto the model resolution
(`bench/common.block_average_grid`), which is the finite-volume-consistent
way to coarsen a published DEM. Mesh coordinates coincide with the DEM's
georeference, so the published output points are used as-is.

## Setup

| Parameter            | Default                    | Meaning                                      |
| -------------------- | -------------------------- | -------------------------------------------- |
| Domain               | 2000 m × 2000 m            | closed (reflective) walls                    |
| Grid                 | 100 × 100                  | `--nx` / `--ny` (dx = 20 m, spec resolution) |
| Manning `n`          | 0.03                       | spec value, uniform                          |
| Inflow               | peak 20 m³/s, ~85 min base | published hydrograph, 97 200 m³              |
| Initial condition    | dry bed                    | spec                                         |
| `t_end`              | 172 800 s (48 h)           | `--t-end` (spec: settle to final state)      |
| `dt_max` / `dt_init` | 5.0 s / 1e-2 s             | timestep bounds                              |
| `cfl_interval`       | 10                         | steps per CFL recompute                      |

Output-point numbering is the EA convention `p = col·4 + row + 1` (columns
west→east, rows south→north): `p1` is the SW depression, `p16` the NE one.

The published inflow is a boundary condition; this solver has no open
boundaries, so it is injected as five equal-split volume sources on the first
cell column along the published inflow line. Volume sources carry no momentum
vector — conservative for a case judged on the settled distribution. The
hydrograph is applied as 60 s piecewise-constant phases sampled at midpoints,
which integrates the published piecewise-linear curve exactly (every
breakpoint is a whole minute) and keeps the default backend on its
GPU-resident path.

## Gates

| Gate                           | Threshold | Notes                                               |
| ------------------------------ | --------- | --------------------------------------------------- |
| `GATE_VOLUME_DRIFT_REL`        | 2e-4      | final volume vs injected (closed, dry start)        |
| positivity + finite            | min h ≥ 0 | wetting/drying stability                            |
| `GATE_ABOVE_SILL_M`            | 0.03 m    | no pond stands above the sill it would spill over   |
| `GATE_FULL_BASIN_CAPACITY_REL` | 1.0       | filled basins hold no more than was injected        |
| `GATE_PONDED_STORAGE_FRAC`     | 0.90      | volume has drained off the ridges into storage      |
| `GATE_FAR_COLUMN_DRY_M`        | 0.05 m    | east column (p13–p16) is out of reach of the volume |
| `GATE_DETERMINISM_REL`         | 1e-4      | run-to-run reproducibility                          |

`POND_LEVEL_M` (0.05 m) is the wet threshold — below it is residual film, not
inundation. Mass balance is the anchor gate: a closed domain started dry
retains every injected cubic metre. The sill allowance covers the sub-cell
discretisation of the pool boundary (one 20 m cell spans ~0.026 m of the local
bed gradient). Determinism is tolerance-based, not bit-exact (see the
`atomicAdd` note in [dambreak.md](dambreak.md)); despite being the longest run
in the suite it holds ~1.8e-6, because the settled ponds damp the drift out
rather than accumulating it.

## Backends

Same policy as the dam-break cases: default is `gpu_resident_batch` on
MoltenVK. The host-CFL-readback `fixed_dt_batch_barrier` does not develop the
wave under MoltenVK, so it is opt-in via `--backends` where supported.
The inflow is applied as a piecewise-constant per-phase `source_rate` (not a
`source_fn`) so the default backend stays on its fully GPU-resident path.

## Running

```sh
# In the dev container:
just benchmark-depressions --repeats 3 --warmup 1 --gif

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m deceris.inundation.bench.floodplain_depressions --repeats 3 --warmup 1 --gif

# Explicit backend matrix:
just benchmark-depressions --backends gpu_resident_batch,fixed_dt_batch_barrier
```

`--repeats >= 2` enables the determinism check; `--warmup` runs are excluded
from timing. The animation comes from a single _untimed_ extra run, so perf
numbers stay clean. Frames are resampled onto uniform simulated time before
rendering: the solver emits a snapshot at every phase boundary, so the 91
one-minute hydrograph phases would otherwise crowd the first 3% of the run
into most of the animation and make playback lurch when the inflow stops.

### Key flags

- `--backends` — CSV of solver impls (see above).
- `--dataset-dir` — published dataset location.
- `--nx` / `--ny` — grid resolution.
- `--t-end` / `--dt-max` / `--cfl-interval` — run length and timestep controls.
- `--output-interval-s` — snapshot cadence for timed runs (default: one final
  snapshot).
- `--gif` — top-down 2D depth-heatmap animation.
- `--gif-backend` / `--gif-frames` — animation controls.

## Artifacts

Written under `.tmp/floodplain-depressions-bench/`:

- `eggbox-dem2010-<nx>x<ny>.parquet` — DEM-sampled mesh.
- `summary.json` — gates, per-point depths and basin sill depths, correctness
  - perf metrics.
- `eggbox-<backend>-2d.gif` — top-down depth heatmap of the flooding.

## Measured result

PASS on MoltenVK / `gpu_resident_batch` — mass balance 9.4e-5, ponds at most
0.007 m above their sills, filled storage 0.835 of injected, 99.9% of the
volume in depression storage, far column dry, reproducibility 1.8e-6. See
[results.md](../../../docs/implementation/15-floodplain-depressions/results.md).

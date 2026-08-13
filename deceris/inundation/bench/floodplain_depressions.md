# Floodplain-depressions benchmark (EA Test 2)

Overland-flow correctness benchmark — UK Environment Agency "Benchmarking of
2D Hydraulic Modelling Packages" **Test 2, filling of floodplain
depressions**. Harness:
[floodplain_depressions.py](floodplain_depressions.py). Effort:
[docs/implementation/15-floodplain-depressions/](../../../docs/implementation/15-floodplain-depressions/).

## What it validates

A dry 2000 m × 2000 m floodplain with a "flattened egg box" topography — a
flat plateau carved with a 4 × 4 grid of 16 shallow circular depressions — is
flooded by a slow inflow hydrograph at the top-left corner. Unlike the
dam-break cases, this exercises the parts of the solver a channel-aligned
shock never touches: disconnected water bodies, wetting/drying of dry
floodplain, and low-momentum inundation extent, judged on the _final_ ponded
distribution.

No closed form exists (EA Test 2 is a model-_intercomparison_ benchmark) and
there is no external DEM (repo convention is synthetic meshes), so the egg-box
bed is generated analytically (`bench/common.eggbox_bed`) with raised-cosine
bowls plus a gentle NE rise that makes the top-right depressions the highest
ground. Gates are therefore invariant/qualitative, not an error norm.

## Setup

| Parameter            | Default                          | Meaning                       |
| -------------------- | -------------------------------- | ----------------------------- |
| Domain               | 2000 m × 2000 m                  | closed (reflective) walls     |
| Grid                 | 100 × 100                        | `--nx` / `--ny` (dx = 20 m)   |
| Depressions          | 4 × 4 = 16, R = 100 m, depth 0.5 m | raised-cosine bowls           |
| NE rise              | 1.0 m                            | keeps points 15 & 16 dry      |
| Manning `n`          | 0.03                             | floodplain roughness          |
| Inflow               | peak 20 m³/s, base 85 min        | `--inflow-peak` / `--hydro-base` |
| Settle tail          | 3600 s                           | `--settle`                    |
| `t_end`              | 8700 s (float32-exact)           | base + settle                 |
| `dt_max` / `dt_init` | 5.0 s / 1e-2 s                   | timestep bounds               |
| `cfl_interval`       | 10                               | steps per CFL recompute       |

Output-point numbering is the EA convention `p = col·4 + row + 1` (columns
west→east, rows south→north); `p15`/`p16` are the far-NE depressions.

## Gates

| Gate                     | Threshold      | Notes                                          |
| ------------------------ | -------------- | ---------------------------------------------- |
| `GATE_MASS_BALANCE_REL`  | 5e-3           | final volume vs injected (closed, dry start)   |
| positivity + finite      | min depth ≥ 0  | wetting/drying stability                       |
| `GATE_MIN_PONDED`        | 3 of 16        | disconnected ponded bodies (`POND_LEVEL_M` 0.05) |
| points 15 & 16 dry       | < 1e-3 m       | far-NE high ground never floods                |
| `GATE_PLATEAU_WET_FRAC`  | 0.15           | ridges stay dry (`PLATEAU_WET_M` 0.05)         |
| `GATE_DETERMINISM_REL`   | 5e-4           | run-to-run reproducibility                     |

Mass balance is the anchor gate: a closed domain started dry retains every
injected cubic metre. Determinism is tolerance-based, not bit-exact (see the
`atomicAdd` note in [dambreak.md](dambreak.md)); the bound is looser than the
dam-break cases' 1e-5 because this run is ~20× longer (~9k steps), so the
non-associative device-side reductions accumulate more drift (~1e-4).

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
numbers stay clean.

### Key flags

- `--backends` — CSV of solver impls (see above).
- `--nx` / `--ny` — grid resolution.
- `--dt-max` / `--cfl-interval` — timestep controls.
- `--inflow-peak` / `--hydro-base` / `--settle` — hydrograph overrides.
- `--output-interval-s` — snapshot cadence for timed runs (default: one final
  snapshot).
- `--gif` — top-down 2D depth-heatmap animation.
- `--gif-backend` / `--gif-frames` — animation controls.

## Artifacts

Written under `.tmp/floodplain-depressions-bench/`:

- `eggbox-<nx>x<ny>.parquet` — generated egg-box mesh.
- `summary.json` — gates, per-point levels, correctness + perf metrics.
- `eggbox-<backend>-2d.gif` — top-down depth heatmap of the flooding.

## Measured result (MoltenVK, `gpu_resident_batch`)

PASS — mass balance = 1.8e-6, min h = 0.0, ponded = 5/16, points 15 & 16
dry, plateau wet = 0.104, determinism ✓. See
[results.md](../../../docs/implementation/15-floodplain-depressions/results.md).

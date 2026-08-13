# Momentum-obstruction benchmark (EA Test 3)

Momentum-conservation correctness benchmark — UK Environment Agency
"Benchmarking of 2D Hydraulic Modelling Packages" **Test 3, momentum
conservation over a small obstruction**. Harness:
[momentum_obstruction.py](momentum_obstruction.py). Effort:
[docs/implementation/16-momentum-obstruction/](../../../docs/implementation/16-momentum-obstruction/).

## What it validates

Whether the solver preserves the shallow-water **momentum (inertia) terms**
during a fast transient — the physics a diffusive-wave or over-dissipative
scheme drops. A block of still water is released from an elevated shelf
(dam-break initial condition), accelerates down a slope, crosses a deep
valley as a bore, and runs up and *over* a flat-topped sill (the
obstruction) into a far bowl. The release is sized so that even if all of
it ponded in the valley, the static surface would stay ≈ 0.27 m **below**
the crest — so an inertia-free model (which moves water strictly down
surface gradients) can never cross, and any pond past the obstruction is
an unambiguous momentum signature.

### Closed-domain adaptation

The published EA Test 3 uses a sloping channel with an **open downstream
outlet**. This solver has only reflective walls and volume-conserving
sources (no outflow, no injected momentum vector), so the case is recast
as a finite elevated release in a closed trap. Containment is the closed
mesh boundary itself — the bed has no built walls, whose steep dry faces
provoke spurious numerical run-up. An earlier variant that injected a
surge directly into a bowl beside the sill was retired: its injected
volume nearly matched the bowl's capacity below the crest, so a static
fill alone reached within 2 cm of the crest and the case did not
discriminate momentum. See the
[effort plan](../../../docs/implementation/16-momentum-obstruction/plan.md)
and [decisions.md](../../../docs/planning/decisions.md).

No closed form exists (model-_intercomparison_ benchmark) and there is no
external DEM (repo convention is synthetic meshes), so the prismatic bed is
generated analytically (`bench/common.sloping_obstruction_bed`). Its
piecewise-linear slope breaks are Gaussian-smoothed (σ = 4 m) into gentle
curves so the shelf, valley floor, sill flanks, and far bowl blend without
sharp corners; the wide flat gauge regions stay flat in their interior. Gates
are therefore invariant/qualitative, not an error norm.

## Setup

| Parameter            | Default                          | Meaning                       |
| -------------------- | -------------------------------- | ----------------------------- |
| Domain               | 300 m × 60 m                     | closed (reflective) boundary  |
| Grid                 | 150 × 12                         | `--nx` / `--ny` (dx = 2 m)    |
| Bed                  | shelf z = 1.2, valley/far bowl z = −0.6, sill z = 0 | prismatic  |
| Release block        | x ∈ [0, 30] on the shelf, 0.8 m deep (1440 m³) | `--release-depth` |
| Valley (Point 1)     | gauge x = 127                    | catches the bulk of the bore  |
| Sill / obstruction   | crest z = 0, gauge x = 190       | barrier to cross              |
| Far bowl (Point 2)   | gauge x = 250                    | momentum target               |
| Manning `n`          | 0.03                             | channel roughness             |
| `t_end`              | 900 s (float32-exact)            | `--t-end` (release + settle)  |
| `dt_max` / `dt_init` | 2.0 s / 1e-2 s                   | timestep bounds               |
| `cfl_interval`       | 10                               | steps per CFL recompute       |

## Gates

| Gate                        | Threshold      | Notes                                            |
| --------------------------- | -------------- | ------------------------------------------------ |
| `GATE_VOLUME_DRIFT_REL`     | 5e-5           | final volume vs release (closed, zero sources)   |
| positivity + finite         | min depth ≥ 0  | wetting/drying stability                         |
| `GATE_POINT1_PONDED_M`      | 0.05 m         | valley retains a settled pond                    |
| `GATE_PONDS_BELOW_CREST_M`  | 0.10 m         | both pond surfaces below the crest (disconnected)|
| `GATE_POINT2_MIN_DEPTH_M`   | 0.02 m         | momentum signature: far bowl ponded              |
| `GATE_CONTROL_POINT2_MAX_M` | 0.005 m        | still-water control leaves the far bowl dry      |
| `GATE_DETERMINISM_REL`      | 2e-3           | run-to-run reproducibility                       |

Volume drift is the anchor gate (closed domain, zero sources). Every
backend is paired with an untimed **still-water control**: the release
volume placed at rest in the valley at its static ceiling (the deepest
lake inertia-free transport could build against the sill). Only if the
control leaves Point 2 dry can release-run ponding be attributed to
momentum — on the current solver the control leaks over the crest from
rest (the known deferred Audusse well-balance gap), so the case honestly
FAILs with a `well_balance_leak` reason until that scheme change lands.
Dryness of the obstruction is asserted via **water-surface elevation**
(both ponds below the crest), not the crest cell's raw depth: at dx = 2 m
the crest gauge retains a thin ~0.05 m residual film — an inherent SWE
wetting/drying artifact — while the meaningful ponds settle ~0.5 m below
the crest. Determinism is tolerance-based, not bit-exact (see the
`atomicAdd` note in [dambreak.md](dambreak.md)).

## Backends

Default backend is `gpu_resident_batch`. Available Vulkan implementations are
`fixed_dt_batch_barrier`, `gpu_resident_batch`, and `fixed_dt_batch`. The release
is a pure initial condition, so every backend stays on its GPU-resident path.

## Running

```sh
# In the dev container:
just benchmark-obstruction --repeats 3 --warmup 1 --gif

# On the macOS host (unsandboxed, patched kp in .venv):
.venv/bin/python -m deceris.inundation.bench.momentum_obstruction --repeats 3 --warmup 1 --gif

# Explicit backend matrix:
just benchmark-obstruction --backends gpu_resident_batch,fixed_dt_batch_barrier
```

`--repeats >= 2` enables the determinism check; `--warmup` runs are excluded
from timing. The animation comes from a single _untimed_ extra run, so perf
numbers stay clean.

### Key flags

- `--backends` — CSV of solver impls (see above).
- `--nx` / `--ny` — grid resolution.
- `--dt-max` / `--cfl-interval` — timestep controls.
- `--release-depth` / `--t-end` — release-block depth and end time.
- `--output-interval-s` — snapshot cadence for timed runs (default: one final
  snapshot).
- `--gif` — top-down 2D terrain + depth-heatmap animation.
- `--gif-backend` / `--gif-frames` — animation controls.

With `--gif` a longitudinal **side-view** profile animation is emitted
alongside the top-down heatmap (same untimed run).

## Artifacts

Written under `.tmp/momentum-obstruction-bench/`:

- `obstruction-v2-<nx>x<ny>.parquet` — generated mesh.
- `summary.json` — gates, per-point levels/WSE, correctness + perf metrics.
- `obstruction-<backend>-2d.gif` — top-down terrain + depth heatmap of the
  surge overtopping the obstruction.
- `obstruction-<backend>-side.gif` — longitudinal side view (bed +
  water-surface profile) of the surge overtopping the obstruction.

## Measured result (MoltenVK, `gpu_resident_batch`)

PASS — mass balance = 4.3e-6, min h = 0.0, Point 1 depth = 0.148 m, Point 2
depth = 0.142 m (momentum overtopping), both ponds ~0.45 m below the crest
(disconnected), determinism ✓. See
[results.md](../../../docs/implementation/16-momentum-obstruction/results.md).

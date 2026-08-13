"""EA Test 3 benchmark: momentum conservation over a small obstruction.

A synthetic, closed-domain adaptation of the UK Environment Agency
"Benchmarking of 2D Hydraulic Modelling Packages" Test 3. The published test
drives flow down a slope past an obstruction with an *open* downstream
outlet; this solver has only reflective (closed) walls, so the case is recast
as a finite dam-break *release* in a closed trap while keeping the physics
under test: whether the solver's momentum (inertia) terms carry fast flow
over a barrier it could never cross by water-surface gradient alone.

Geometry (prismatic, left to right): an elevated reservoir shelf flush
against the left domain boundary (the mesh boundary itself is the reflective
containment wall, as in the dam-break benches — no bed-built walls, whose
steep dry faces provoke spurious numerical run-up), a steep slope down into a
deep flat-bottomed valley (Point 1), a rise to a flat-topped sill (the
obstruction), and a second flat-bottomed bowl (Point 2) running to the right
boundary. At t = 0 a block of still water standing on the shelf is released
(dam-break initial condition — no sources): it accelerates down the slope,
crosses the valley as a bore, and runs up and *over* the sill into Point 2.

The discriminator is volumetric, not visual: the release is sized so that
even if *all* of it ponded in the valley, the static water surface would top
out ~0.25 m *below* the sill crest. An inertia-free (diffusive-wave) model
moves water strictly down surface gradients, so it can never raise the valley
surface above that ceiling and Point 2 stays dry; only conserved momentum can
carry water over the crest. Any settled pond in Point 2 is therefore an
unambiguous momentum signature (both ponds settle below the crest,
hydraulically disconnected). Every backend run is paired with an untimed
*still-water control*: the same volume placed at rest at that static ceiling.
A well-balanced solver must keep the control's Point 2 dry — if it leaks over
the crest from rest, Point 2 ponding is a numerical well-balance artifact,
not momentum, and the case fails with an explicit reason.

Like EA Test 2, this is a model-*intercomparison* benchmark with no closed
form and no external DEM (repo convention: synthetic, self-contained meshes);
the bed is generated analytically (`bench/common.sloping_obstruction_bed`).
Acceptance is invariant/qualitative: closed-domain volume conservation,
positivity / wetting-drying stability, a settled valley pond, both pond
surfaces below the sill crest, and — the momentum signature — a measurable
pond in Point 2 past the obstruction.

Backends are configurable (`--backends`) across the retained Vulkan solver
implementations. The release is a pure initial condition with zero sources.
`--gif` renders untimed extra runs into a top-down depth-heatmap
animation and a longitudinal side-view (bed + water-surface profile)
animation; timed runs are never the frame source, so perf stays clean.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from deceris.inundation.bench.common import (
    build_channel_mesh,
    save_depth_gif,
    save_profile_gif,
    sloping_obstruction_bed,
    write_mesh_parquet,
)
from deceris.inundation.workflow import (
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
    WorkflowResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ── Case geometry / physics (defaults; see _build_parser for overrides) ─────
# Mesh coordinates run x in [0, CHANNEL_LEN_M], y in [0, CHANNEL_WIDTH_M].
# The bed is prismatic (uniform across the width), left to right: an elevated
# reservoir shelf (the release block stands here, flush against the left
# boundary), a steep slope down into a deep flat-bottomed valley (Point 1),
# a rise to a flat-topped sill (the obstruction), and a second flat-bottomed
# bowl (Point 2) running to the right boundary. Containment is the closed
# mesh boundary itself (reflective walls); the bed has no built walls — steep
# dry bed faces provoke spurious numerical run-up. Datum is the sill top.
CHANNEL_LEN_M = 300.0
CHANNEL_WIDTH_M = 60.0
NX_DEFAULT = 150  # dx = 2 m
NY_DEFAULT = 12  # dy = 5 m

SHELF_Z_M = 1.2  # elevated reservoir shelf the release block stands on
BOWL_FLOOR_Z_M = -0.6  # valley (Point 1) and far bowl (Point 2) floors
SILL_Z_M = 0.0  # obstruction crest between the two bowls
RESERVOIR_X0_M = 0.0  # release block spans [X0, X1] on the shelf
RESERVOIR_X1_M = 30.0
POINT1_X_M = 127.0  # valley floor centre (flat floor x in [94, 160])
CREST_X_M = 190.0  # sill crest gauge (flat top x in [175, 205])
POINT2_X_M = 250.0  # far bowl floor centre (flat floor x in [220, 300])
CREST_Z_M = SILL_Z_M
CONTROL_X = (0.0, 44.0, 94.0, 160.0, 175.0, 205.0, 220.0, 300.0)
CONTROL_Z = (
    SHELF_Z_M,  # reservoir shelf (release block stands here)
    SHELF_Z_M,
    BOWL_FLOOR_Z_M,  # slope foot: flat valley floor (Point 1)
    BOWL_FLOOR_Z_M,
    SILL_Z_M,  # rise to the sill flat top (the obstruction)
    SILL_Z_M,
    BOWL_FLOOR_Z_M,  # drop into the flat far bowl (Point 2)
    BOWL_FLOOR_Z_M,  # far bowl runs to the right boundary
)

# Channel roughness.
MANNING_N = 0.03

# Bed smoothing (Gaussian sigma, m): rounds the piecewise-linear slope breaks
# (shelf->slope, slope->valley, valley->sill, sill->far bowl) into gentle
# curves. The flat gauge regions (shelf, valley floor, sill top, far bowl) are
# far wider than the kernel, so their interiors stay flat; only the corners
# round. 4 m ~= two cells at dx = 2 m.
SMOOTHING_M = 4.0

# Release: a block of still water standing on the reservoir shelf, released
# at t = 0 (dam-break initial condition; no sources). Deep and narrow for a
# punchy collapse (front celerity ~ 2*sqrt(g*h)), and sized so that even if
# the whole release ponded in the valley, the static surface would top out
# ~0.25 m below the sill crest — see the module docstring.
RELEASE_DEPTH_M = 0.8
T_END_S = 900.0  # release + settle; integer -> float32-exact stop

DT_MAX_DEFAULT = 2.0
DT_INIT_DEFAULT = 1e-2
CFL_INTERVAL_DEFAULT = 10

# ── Correctness gates (invariant / qualitative; no closed-form reference) ───
# Closed domain (reflective walls), zero sources: the release volume must be
# conserved to float32 accumulation error. This is the anchor gate. The
# dam-break bench holds 1e-5 over a 20 s frictionless run; this case runs
# 900 s (~10k steps) of friction + wetting/drying and lands at ~1.7e-5
# measured, so the gate is 5e-5 (same "round-off only" spirit, ~3x headroom).
GATE_VOLUME_DRIFT_REL = 5e-5
# The valley (Point 1) must catch and retain a settled pond.
GATE_POINT1_PONDED_M = 0.05
# Disconnected-reservoir signature: both pond *surfaces* must settle at least
# this far below the obstruction crest elevation. The two bowls are then
# hydraulically isolated (no continuous water body spans the crest), so any
# water in the far bowl must have been carried *over* the crest by the bore.
# NB the crest gauge itself retains a thin residual film/puddle — an inherent
# SWE wetting/drying artifact at this resolution — so dryness is asserted via
# water-surface elevation, not the crest cell's raw depth.
GATE_PONDS_BELOW_CREST_M = 0.10
# Momentum signature: Point 2 (far bowl, past the dry crest) must hold at
# least this depth at t_end. The release's static ceiling in the valley is
# ~0.25 m below the crest, so an inertia-free model cannot cross; only
# conserved momentum puts water here.
GATE_POINT2_MIN_DEPTH_M = 0.02
# Still-water control: the same volume placed *at rest* in the valley (the
# deepest lake any inertia-free, gradient-driven transport could build
# against the sill) must leave Point 2 essentially dry. If the control wets
# Point 2, water is crossing the crest without momentum — a numerical
# well-balance leak — and the release run's Point 2 pond proves nothing.
# This control is what makes the momentum gate meaningful.
GATE_CONTROL_POINT2_MAX_M = 0.005
# Repeat-to-repeat reproducibility (relative to peak depth). GPU atomicAdd flux
# scatter is not bit-reproducible; the device-side CFL reductions accumulate a
# run-to-run drift observed up to ~6e-4 over runs of this length on MoltenVK,
# so the gate carries ~3x headroom above the observed tail.
GATE_DETERMINISM_REL = 2e-3

DEFAULT_OUTPUT_ROOT = ".tmp/momentum-obstruction-bench"
# Default to the fully GPU-resident backend. Other retained Vulkan strategies
# remain selectable through --backends.
DEFAULT_BACKENDS = ("gpu_resident_batch",)
BackendImpl = Literal[
    "fixed_dt_batch_barrier",
    "gpu_resident_batch",
    "fixed_dt_batch",
]
AVAILABLE_BACKENDS: tuple[BackendImpl, ...] = (
    "fixed_dt_batch_barrier",
    "gpu_resident_batch",
    "fixed_dt_batch",
)
GIF_FRAMES_DEFAULT = 60
GIF_VMAX_M = 0.8


@dataclass(frozen=True)
class ObstructionSpec:
    """Geometry + release parameters for the momentum-obstruction case."""

    name: str
    nx: int
    ny: int
    channel_len_m: float
    channel_width_m: float
    manning_n: float
    release_depth_m: float
    t_end_s: float
    dt_max: float
    cfl_interval: int

    @property
    def dx_m(self) -> float:
        return self.channel_len_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.channel_width_m / self.ny

    @property
    def control_x(self) -> tuple[float, ...]:
        return CONTROL_X

    @property
    def control_z(self) -> tuple[float, ...]:
        return CONTROL_Z

    @property
    def crest_z_m(self) -> float:
        return CREST_Z_M

    @property
    def point1_xy(self) -> tuple[float, float]:
        return (POINT1_X_M, self.channel_width_m / 2.0)

    @property
    def point2_xy(self) -> tuple[float, float]:
        return (POINT2_X_M, self.channel_width_m / 2.0)

    @property
    def reservoir_span_m(self) -> tuple[float, float]:
        return (RESERVOIR_X0_M, RESERVOIR_X1_M)


@dataclass(frozen=True)
class CaseMetrics:
    """Invariant/qualitative metrics + gate outcomes for one (case, backend)."""

    case: str
    backend: str
    volume_drift_rel: float
    min_depth_m: float
    point1_depth_m: float
    point1_wse_m: float
    crest_depth_m: float
    point2_depth_m: float
    point2_wse_m: float
    control_point2_depth_m: float
    left_ponded: bool
    ponds_disconnected: bool
    point2_risen: bool
    control_dry: bool
    h_final_finite: bool
    passed: bool
    fail_reasons: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class PerfMetrics:
    """Timing summary for one (case, backend) group of timed repeats."""

    case: str
    backend: str
    repeats: int
    median_wall_s: float
    steps_total: int
    steps_per_s: float
    cell_steps_per_s: float
    deterministic: bool | None  # None when repeats < 2
    determinism_detail: str


def _build_spec(args: argparse.Namespace) -> ObstructionSpec:
    return ObstructionSpec(
        name="obstruction",
        nx=args.nx,
        ny=args.ny,
        channel_len_m=CHANNEL_LEN_M,
        channel_width_m=CHANNEL_WIDTH_M,
        manning_n=MANNING_N,
        release_depth_m=args.release_depth,
        t_end_s=args.t_end,
        dt_max=args.dt_max,
        cfl_interval=args.cfl_interval,
    )


def _original_cell_centroids(
    spec: ObstructionSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cell centers in generation (pre-Hilbert) order, matching write order."""
    n = spec.nx * spec.ny
    idx = np.arange(n)
    cx = (idx % spec.nx + 0.5) * spec.dx_m
    cy = (idx // spec.nx + 0.5) * spec.dy_m
    return cx.astype(np.float64), cy.astype(np.float64)


def _ensure_mesh(spec: ObstructionSpec, output_dir: Path) -> Path:
    """Write the synthetic obstruction mesh (idempotent per resolution + bed rev)."""
    # "v2" = elevated-release bed; bump when CONTROL_X/CONTROL_Z change so a
    # stale cached parquet from an older bed is never silently reused.
    mesh_path = output_dir / f"obstruction-v3-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(
            spec.nx, spec.ny, spec.channel_len_m, spec.channel_width_m
        )
        cx, _cy = _original_cell_centroids(spec)
        zb = sloping_obstruction_bed(
            cx, control_x=spec.control_x, control_z=spec.control_z, smoothing_m=SMOOTHING_M
        )
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] synthetic obstruction mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def _release_mask(spec: ObstructionSpec) -> NDArray[np.bool_]:
    """Cells (original order) whose centroids lie under the release block."""
    cx, _cy = _original_cell_centroids(spec)
    x0, x1 = spec.reservoir_span_m
    return (cx >= x0) & (cx < x1)


def _initial_state(spec: ObstructionSpec) -> dict[str, object]:
    """Still release block on the reservoir shelf, dry elsewhere (original order)."""
    n = spec.nx * spec.ny
    h0 = np.zeros(n, dtype=np.float32)
    h0[_release_mask(spec)] = np.float32(spec.release_depth_m)
    zeros = np.zeros(n, dtype=np.float32)
    return {"h": h0, "hu": zeros.copy(), "hv": zeros.copy()}


def _released_volume(spec: ObstructionSpec) -> float:
    """Exact volume of the release block [m^3]."""
    n_cells = int(_release_mask(spec).sum())
    return float(n_cells) * spec.dx_m * spec.dy_m * spec.release_depth_m


def _bed_original(spec: ObstructionSpec) -> NDArray[np.float64]:
    """Bed elevation at cell centers in original (generation) order."""
    cx, _cy = _original_cell_centroids(spec)
    return sloping_obstruction_bed(
        cx, control_x=spec.control_x, control_z=spec.control_z, smoothing_m=SMOOTHING_M
    ).astype(np.float64)


def _control_wse(spec: ObstructionSpec) -> float:
    """Valley water-surface elevation holding the whole release volume at rest.

    Solved by bisection on the valley hypsometry (cells below the crest,
    left of the sill top). This is the static all-in-valley ceiling the
    module docstring describes; by construction it sits below the crest.
    """
    cx, _cy = _original_cell_centroids(spec)
    zb = _bed_original(spec)
    valley = (zb < spec.crest_z_m) & (cx < CREST_X_M)
    cell_area = spec.dx_m * spec.dy_m
    target = _released_volume(spec)

    def capacity(w: float) -> float:
        return float(np.clip(w - zb[valley], 0.0, None).sum() * cell_area)

    lo, hi = float(zb[valley].min()), spec.crest_z_m
    if capacity(hi) <= target:
        raise RuntimeError("release volume exceeds valley capacity below the crest")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if capacity(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _control_initial_state(spec: ObstructionSpec) -> dict[str, object]:
    """Inertia-free endpoint: the release volume already at rest in the valley.

    This is the deepest lake any gradient-driven (diffusive-wave) transport
    of the release could ever build against the sill. Started at rest, a
    well-balanced solver must keep the sill and Point 2 dry; any water
    appearing past the crest is a numerical well-balance leak, not momentum.
    """
    cx, _cy = _original_cell_centroids(spec)
    zb = _bed_original(spec)
    wse = _control_wse(spec)
    valley = (zb < spec.crest_z_m) & (cx < CREST_X_M)
    h0 = np.where(valley, np.clip(wse - zb, 0.0, None), 0.0).astype(np.float32)
    zeros = np.zeros(spec.nx * spec.ny, dtype=np.float32)
    return {"h": h0, "hu": zeros.copy(), "hv": zeros.copy()}


def _make_workflow(
    spec: ObstructionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
    initial_state: dict[str, object] | None = None,
) -> SWEWorkflow:
    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=spec.manning_n,
        output_interval_s=output_interval_s,
        dt_max=spec.dt_max,
        dt_init=DT_INIT_DEFAULT,
        cfl_interval=spec.cfl_interval,
        progress=progress,
        initial_state_source=initial_state if initial_state is not None else _initial_state(spec),
        initial_state_order="original",
        solver_impl=backend,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    return workflow


def _run_once(
    spec: ObstructionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
    initial_state: dict[str, object] | None = None,
) -> tuple[SWEWorkflow, WorkflowResult]:
    workflow = _make_workflow(
        spec,
        mesh_path,
        backend,
        output_interval_s=output_interval_s,
        progress=progress,
        initial_state=initial_state,
    )
    result = workflow.run([SimulationPhase(duration_s=spec.t_end_s, sources=[])])
    return workflow, result


def _nearest_cell(workflow: SWEWorkflow, xy: tuple[float, float]) -> int:
    """Index of the solver cell whose centroid is nearest to ``xy``."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    cx = workflow.geom.centroid[:, 0].astype(np.float64)
    cy = workflow.geom.centroid[:, 1].astype(np.float64)
    d2 = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
    return int(np.argmin(d2))


def _evaluate_case(
    spec: ObstructionSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
    control_point2_depth_m: float,
) -> CaseMetrics:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")

    h_final = np.asarray(result.h_final, dtype=np.float32)
    zb = np.asarray(workflow.geom.zb, dtype=np.float32)
    finite = bool(np.isfinite(h_final).all())
    min_depth = float(h_final.min()) if h_final.size else 0.0

    # Drift is measured against the solver's own initial volume (geometric
    # cell areas at float32), not the analytic block volume — mirrors
    # bench/dambreak.py. The release block is defined by centroid x-range,
    # identical in solver order.
    cx_solver = workflow.geom.centroid[:, 0].astype(np.float64)
    x0, x1 = spec.reservoir_span_m
    h0_solver = np.where((cx_solver >= x0) & (cx_solver < x1), spec.release_depth_m, 0.0)
    area64 = workflow.geom.area.astype(np.float64)
    volume_initial = float((h0_solver * area64).sum())
    volume_drift_rel = (
        abs(result.volume_final_m3 - volume_initial) / volume_initial
        if volume_initial > 0.0
        else float("inf")
    )

    c1 = _nearest_cell(workflow, spec.point1_xy)
    cc = _nearest_cell(workflow, (CREST_X_M, spec.channel_width_m / 2.0))
    c2 = _nearest_cell(workflow, spec.point2_xy)
    point1_depth = float(h_final[c1])
    crest_depth = float(h_final[cc])
    point2_depth = float(h_final[c2])
    point1_wse = float(zb[c1] + h_final[c1])
    point2_wse = float(zb[c2] + h_final[c2])

    left_ponded = point1_depth >= GATE_POINT1_PONDED_M
    # Disconnected reservoirs: both pond surfaces settle below the crest.
    ponds_disconnected = max(point1_wse, point2_wse) <= CREST_Z_M - GATE_PONDS_BELOW_CREST_M
    point2_risen = point2_depth >= GATE_POINT2_MIN_DEPTH_M
    control_dry = control_point2_depth_m <= GATE_CONTROL_POINT2_MAX_M

    fail_reasons: list[str] = []
    if not finite:
        fail_reasons.append("h_final_non_finite")
    if min_depth < -1e-6:
        fail_reasons.append(f"negative_depth:{min_depth:.3e}")
    if volume_drift_rel > GATE_VOLUME_DRIFT_REL:
        fail_reasons.append(f"volume_drift_rel:{volume_drift_rel:.3e}>{GATE_VOLUME_DRIFT_REL:.0e}")
    if not left_ponded:
        fail_reasons.append(
            f"point1_depth:{point1_depth:.3f}<{GATE_POINT1_PONDED_M} (valley not ponded)"
        )
    if not ponds_disconnected:
        max_wse = max(point1_wse, point2_wse)
        fail_reasons.append(
            f"max_pond_wse:{max_wse:.3f}>{CREST_Z_M - GATE_PONDS_BELOW_CREST_M:.3f} "
            f"(ponds not disconnected below crest)"
        )
    if not point2_risen:
        fail_reasons.append(
            f"point2_depth:{point2_depth:.3f}<{GATE_POINT2_MIN_DEPTH_M} (no overtopping)"
        )
    if not control_dry:
        fail_reasons.append(
            f"control_point2_depth:{control_point2_depth_m:.3f}>{GATE_CONTROL_POINT2_MAX_M} "
            f"(well-balance leak: still water crosses the crest without momentum, "
            f"so point2_depth is not a momentum signature)"
        )

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        volume_drift_rel=volume_drift_rel,
        min_depth_m=min_depth,
        point1_depth_m=point1_depth,
        point1_wse_m=point1_wse,
        crest_depth_m=crest_depth,
        point2_depth_m=point2_depth,
        point2_wse_m=point2_wse,
        control_point2_depth_m=control_point2_depth_m,
        left_ponded=left_ponded,
        ponds_disconnected=ponds_disconnected,
        point2_risen=point2_risen,
        control_dry=control_dry,
        h_final_finite=finite,
        passed=not fail_reasons,
        fail_reasons=fail_reasons,
    )


def _safe_gif_frames(frames: int) -> int:
    """Nearest power of two >= 2, so t_end / frames is float32-exact."""
    n = max(2, frames)
    lo = 1 << (n.bit_length() - 1)
    hi = lo << 1
    return lo if (n - lo) <= (hi - n) else hi


def _run_group(
    spec: ObstructionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    args: argparse.Namespace,
) -> tuple[CaseMetrics, PerfMetrics]:
    n_cells = spec.nx * spec.ny
    output_interval_s = args.output_interval_s or spec.t_end_s

    for w in range(args.warmup):
        print(f"[{spec.name}/{backend}] warmup {w + 1}/{args.warmup}")
        _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=output_interval_s,
            progress=args.solver_progress,
        )

    walls: list[float] = []
    final_states: list[NDArray[np.float64]] = []
    last: tuple[SWEWorkflow, WorkflowResult] | None = None
    for r in range(args.repeats):
        print(f"[{spec.name}/{backend}] timed run {r + 1}/{args.repeats}")
        t0 = time.perf_counter()
        workflow, result = _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=output_interval_s,
            progress=args.solver_progress,
        )
        walls.append(time.perf_counter() - t0)
        final_states.append(np.asarray(result.h_final, dtype=np.float64))
        last = (workflow, result)

    if last is None:
        raise RuntimeError("--repeats must be >= 1")
    workflow, result = last

    # Untimed still-water control (inertia-free endpoint at rest): must not
    # wet Point 2, or Point 2 ponding cannot be attributed to momentum.
    print(f"[{spec.name}/{backend}] still-water control run (untimed)")
    control_workflow, control_result = _run_once(
        spec,
        mesh_path,
        backend,
        output_interval_s=output_interval_s,
        progress=args.solver_progress,
        initial_state=_control_initial_state(spec),
    )
    control_h = np.asarray(control_result.h_final, dtype=np.float32)
    control_p2 = float(control_h[_nearest_cell(control_workflow, spec.point2_xy)])

    case_metrics = _evaluate_case(spec, backend, workflow, result, control_p2)

    deterministic: bool | None = None
    determinism_detail = "skipped (repeats=1)"
    if len(final_states) >= 2:
        scale = max(float(np.abs(final_states[0]).max()), 1e-12)
        max_rel_diff = max(
            float(np.abs(state - final_states[0]).max()) / scale for state in final_states[1:]
        )
        deterministic = max_rel_diff <= GATE_DETERMINISM_REL
        determinism_detail = (
            f"max rel diff {max_rel_diff:.2e} <= {GATE_DETERMINISM_REL:.0e}"
            if deterministic
            else f"max rel diff {max_rel_diff:.2e} > {GATE_DETERMINISM_REL:.0e}"
        )

    median_wall_s = statistics.median(walls)
    steps_total = int(result.steps_total)
    perf = PerfMetrics(
        case=spec.name,
        backend=backend,
        repeats=args.repeats,
        median_wall_s=median_wall_s,
        steps_total=steps_total,
        steps_per_s=steps_total / median_wall_s if median_wall_s > 0 else 0.0,
        cell_steps_per_s=(steps_total * n_cells) / median_wall_s if median_wall_s > 0 else 0.0,
        deterministic=deterministic,
        determinism_detail=determinism_detail,
    )

    if args.gif and backend == args.gif_backend:
        safe_frames = _safe_gif_frames(args.gif_frames)
        if safe_frames != args.gif_frames:
            print(
                f"[{spec.name}/{backend}] gif frames {args.gif_frames} -> "
                f"{safe_frames} (float32-safe interval)"
            )
        print(f"[{spec.name}/{backend}] extra untimed gif run ({safe_frames} frames)")
        gif_workflow, gif_result = _run_once(
            spec,
            mesh_path,
            backend,
            output_interval_s=spec.t_end_s / safe_frames,
            progress=args.solver_progress,
        )
        save_depth_gif(
            gif_workflow,
            gif_result.snapshots,
            gif_result.snap_times,
            Path(args.output_dir) / f"{spec.name}-{backend}-2d.gif",
            vmax=GIF_VMAX_M,
            title_prefix=f"{spec.name} — ",
        )
        save_profile_gif(
            gif_workflow,
            gif_result.snapshots,
            gif_result.snap_times,
            Path(args.output_dir) / f"{spec.name}-{backend}-side.gif",
            vmax=GIF_VMAX_M,
            title_prefix=f"{spec.name} — ",
        )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: volume_drift<={GATE_VOLUME_DRIFT_REL:.0e} min_depth>=0 finite "
        f"point1_ponded>={GATE_POINT1_PONDED_M} ponds_below_crest>={GATE_PONDS_BELOW_CREST_M} "
        f"point2_depth>={GATE_POINT2_MIN_DEPTH_M} control_p2<={GATE_CONTROL_POINT2_MAX_M}"
    )
    print(
        f"{'case':12} {'backend':24} {'vol_drift':>9} {'min_h':>10} {'p1_depth':>9} "
        f"{'crest_h':>8} {'p2_depth':>9} {'ctrl_p2':>8} {'p1_pond':>8} {'disconn':>8} "
        f"{'p2_rise':>8} {'ctrl_dry':>8} {'pass':>5}"
    )
    for m in all_metrics:
        print(
            f"{m.case:12} {m.backend:24} {m.volume_drift_rel:9.3e} {m.min_depth_m:10.3e} "
            f"{m.point1_depth_m:9.3f} {m.crest_depth_m:8.3f} {m.point2_depth_m:9.3f} "
            f"{m.control_point2_depth_m:8.3f} {m.left_ponded!s:>8} {m.ponds_disconnected!s:>8} "
            f"{m.point2_risen!s:>8} {m.control_dry!s:>8} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
        print(
            f"  Point 1 WSE {m.point1_wse_m:.3f} m, Point 2 WSE {m.point2_wse_m:.3f} m "
            f"(crest {CREST_Z_M:.2f} m)"
        )
    print()
    print(
        f"{'case':12} {'backend':24} {'wall_s':>8} {'steps':>8} {'steps/s':>10} "
        f"{'cell-steps/s':>13} {'determ':>7}"
    )
    for p in all_perf:
        det = "-" if p.deterministic is None else str(p.deterministic)
        print(
            f"{p.case:12} {p.backend:24} {p.median_wall_s:8.3f} {p.steps_total:8d} "
            f"{p.steps_per_s:10.1f} {p.cell_steps_per_s:13.3e} {det:>7}"
        )
        if p.deterministic is False:
            print(f"  !! {p.determinism_detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EA Test 3 momentum-conservation-over-an-obstruction benchmark"
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} (default: Vulkan reference)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nx", type=int, default=NX_DEFAULT)
    parser.add_argument("--ny", type=int, default=NY_DEFAULT)
    parser.add_argument("--dt-max", type=float, default=DT_MAX_DEFAULT)
    parser.add_argument("--cfl-interval", type=int, default=CFL_INTERVAL_DEFAULT)
    parser.add_argument("--release-depth", type=float, default=RELEASE_DEPTH_M)
    parser.add_argument("--t-end", type=float, default=T_END_S)
    parser.add_argument(
        "--output-interval-s",
        type=float,
        default=None,
        help="snapshot cadence for timed runs (default: t_end, i.e. one final snapshot)",
    )
    parser.add_argument("--repeats", type=int, default=2, help=">=2 enables determinism check")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--solver-progress", action="store_true")
    parser.add_argument("--gif", action="store_true", help="render untimed top-down 2D depth gif")
    parser.add_argument(
        "--gif-backend", default=None, help="backend for the gif run (default: first of --backends)"
    )
    parser.add_argument("--gif-frames", type=int, default=GIF_FRAMES_DEFAULT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    raw_backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for backend_name in raw_backends:
        if backend_name not in AVAILABLE_BACKENDS:
            print(f"unknown backend {backend_name!r}; choose from {AVAILABLE_BACKENDS}")
            return 2
    backends: list[BackendImpl] = [cast("BackendImpl", b) for b in raw_backends]
    if args.gif_backend is None:
        args.gif_backend = backends[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _build_spec(args)
    mesh_path = _ensure_mesh(spec, output_dir)
    print(
        f"[release] {spec.release_depth_m} m block on the shelf "
        f"(x in {spec.reservoir_span_m}, z = {SHELF_Z_M} m), volume "
        f"{_released_volume(spec):.0f} m^3; t_end {spec.t_end_s:.0f} s"
    )

    all_metrics: list[CaseMetrics] = []
    all_perf: list[PerfMetrics] = []
    for backend in backends:
        metrics, perf = _run_group(spec, mesh_path, backend, args)
        all_metrics.append(metrics)
        all_perf.append(perf)

    _print_report(all_metrics, all_perf)

    summary = {
        "gates": {
            "volume_drift_rel": GATE_VOLUME_DRIFT_REL,
            "point1_ponded_m": GATE_POINT1_PONDED_M,
            "ponds_below_crest_m": GATE_PONDS_BELOW_CREST_M,
            "point2_min_depth_m": GATE_POINT2_MIN_DEPTH_M,
            "control_point2_max_m": GATE_CONTROL_POINT2_MAX_M,
            "determinism_rel": GATE_DETERMINISM_REL,
        },
        "released_volume_m3": _released_volume(spec),
        "correctness": [asdict(m) for m in all_metrics],
        "performance": [asdict(p) for p in all_perf],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] {summary_path}")

    correctness_ok = all(m.passed for m in all_metrics)
    determinism_ok = all(p.deterministic is not False for p in all_perf)
    if correctness_ok and determinism_ok:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""EA Test 2 benchmark: filling of floodplain depressions (flattened egg box).

A synthetic reconstruction of the UK Environment Agency "Benchmarking of 2D
Hydraulic Modelling Packages" Test 2. A 2000 m x 2000 m floodplain with a
"flattened egg box" topography -- a flat plateau carved with a 4 x 4 grid of
16 shallow circular depressions -- is flooded by an inflow hydrograph applied
at the top-left corner (peak 20 m³/s, ~85 min time base). The test exercises
disconnected water bodies, wetting/drying of floodplains, and low-momentum
inundation extent, with the emphasis on the *final* distribution of ponded
water rather than peak levels.

Unlike the Tier 1 dam-break cases, EA Test 2 has no closed-form solution — it
is a model-*intercomparison* benchmark. There is also no external DEM file
(the repo convention is synthetic, self-contained meshes), so the egg-box bed
is generated analytically (`bench/common.eggbox_bed`) with a gentle NE rise
that makes the top-right depressions structurally the highest ground. The
acceptance criteria are therefore invariant/qualitative rather than an error
norm against a reference: closed-domain mass balance, positivity / wetting-
drying stability, disconnected ponding in the depressions, dry high ground
between them, and the EA result that the far-corner points (15 & 16) stay dry.

Backends are configurable via ``--backends``; the retained Vulkan family runs
on macOS/MoltenVK. The inflow is applied as a piecewise-constant per-phase
``source_rate`` (not a ``source_fn``), so the default ``gpu_resident_batch``
backend stays on its GPU-resident path. ``--gif`` renders an *untimed* extra
run into a top-down depth-heatmap animation; timed runs are never the frame
source, so perf stays clean.
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
    eggbox_bed,
    eggbox_depression_centers,
    save_depth_gif,
    write_mesh_parquet,
)
from deceris.inundation.workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
    WorkflowResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ── Case geometry / physics (defaults; see _build_parser for overrides) ─────
DOMAIN_L_M = 2000.0
N_PER_SIDE = 4  # 4 x 4 = 16 depressions
NX_DEFAULT = 100
NY_DEFAULT = 100  # dx = dy = 20 m

PLATEAU_Z_M = 0.0
# Gentle rise toward the NE corner so the top-right depressions (EA output
# points 15 & 16) are the highest ground and stay dry under a top-left inflow.
NE_RISE_M = 1.0
DEP_RADIUS_M = 100.0
DEP_DEPTH_M = 0.5

# EA Test 2 floodplain roughness (low-momentum overland flow).
MANNING_N = 0.03

# Inflow hydrograph: symmetric triangle, peak 20 m^3/s over an ~85 min base,
# applied at the top-left corner. Volume ~= 0.5 * peak * base ~= 51,000 m^3.
INFLOW_PEAK_M3S = 20.0
HYDROGRAPH_BASE_S = 85.0 * 60.0
HYDROGRAPH_PHASES = 17
SETTLE_S = 3600.0  # quiescent tail so ponds settle and the plateau drains
# T_END = base + settle = 5100 + 3600 = 8700 s (integer -> float32-exact stop).

DT_MAX_DEFAULT = 5.0
DT_INIT_DEFAULT = 1e-2
CFL_INTERVAL_DEFAULT = 10

# ── Correctness gates (invariant / qualitative; no closed-form reference) ───
# Closed domain (reflective walls) started dry: every injected cubic metre is
# retained, so final volume must equal injected volume to float32 accumulation
# error. This is the anchor gate.
GATE_MASS_BALANCE_REL = 5e-3
# A depression is "ponded" if its center depth exceeds this at t_end.
POND_LEVEL_M = 0.05
# Points 15 & 16 (far NE) must be effectively dry.
DRY_LEVEL_M = 1e-3
# At least this many of the 16 depressions must pond (disconnected bodies).
GATE_MIN_PONDED = 3
# A plateau cell (outside every depression) counts as wet above this depth.
PLATEAU_WET_M = 0.05
# Fraction of plateau cells allowed to be wet at t_end: low-momentum ponding
# leaves the ridges between depressions dry.
GATE_PLATEAU_WET_FRAC = 0.15
# Repeat-to-repeat reproducibility (relative to peak depth). GPU atomicAdd flux
# scatter is not bit-reproducible; over the ~9k float32 steps of this run the
# device-side CFL reductions accumulate to a run-to-run drift of ~1e-4.
GATE_DETERMINISM_REL = 5e-4

DEFAULT_OUTPUT_ROOT = ".tmp/floodplain-depressions-bench"
# gpu_resident_batch is the default Vulkan backend. The host-CFL-readback
# fixed_dt_batch_barrier does not develop the wave under MoltenVK, so it is
# opt-in via --backends where that environment supports it.
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


@dataclass(frozen=True)
class DepressionSpec:
    """Geometry + hydrograph parameters for the floodplain-depressions case."""

    name: str
    nx: int
    ny: int
    domain_l_m: float
    n_per_side: int
    plateau_z_m: float
    ne_rise_m: float
    dep_radius_m: float
    dep_depth_m: float
    manning_n: float
    inflow_peak_m3s: float
    hydrograph_base_s: float
    hydrograph_phases: int
    settle_s: float
    dt_max: float
    cfl_interval: int

    @property
    def dx_m(self) -> float:
        return self.domain_l_m / self.nx

    @property
    def t_end_s(self) -> float:
        return self.hydrograph_base_s + self.settle_s

    @property
    def n_depressions(self) -> int:
        return self.n_per_side * self.n_per_side

    @property
    def inflow_center_m(self) -> tuple[float, float]:
        # Top-left corner, offset a couple of cells off the walls.
        off = 2.0 * self.dx_m
        return (off, self.domain_l_m - off)

    @property
    def inflow_radius_m(self) -> float:
        return 2.0 * self.dx_m

    def depression_centers(self) -> NDArray[np.float64]:
        return eggbox_depression_centers(self.domain_l_m, self.n_per_side)


@dataclass(frozen=True)
class CaseMetrics:
    """Invariant/qualitative metrics + gate outcomes for one (case, backend)."""

    case: str
    backend: str
    mass_balance_rel: float
    min_depth_m: float
    ponded_count: int
    dry_far_corner: bool
    plateau_wet_frac: float
    h_final_finite: bool
    point_levels_m: list[float]
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


def _build_spec(args: argparse.Namespace) -> DepressionSpec:
    return DepressionSpec(
        name="eggbox",
        nx=args.nx,
        ny=args.ny,
        domain_l_m=DOMAIN_L_M,
        n_per_side=N_PER_SIDE,
        plateau_z_m=PLATEAU_Z_M,
        ne_rise_m=NE_RISE_M,
        dep_radius_m=DEP_RADIUS_M,
        dep_depth_m=DEP_DEPTH_M,
        manning_n=MANNING_N,
        inflow_peak_m3s=args.inflow_peak,
        hydrograph_base_s=args.hydro_base,
        hydrograph_phases=HYDROGRAPH_PHASES,
        settle_s=args.settle,
        dt_max=args.dt_max,
        cfl_interval=args.cfl_interval,
    )


def _original_cell_centroids(
    spec: DepressionSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cell centers in generation (pre-Hilbert) order, matching write order."""
    n = spec.nx * spec.ny
    dx = spec.dx_m
    idx = np.arange(n)
    cx = (idx % spec.nx + 0.5) * dx
    cy = (idx // spec.nx + 0.5) * dx
    return cx.astype(np.float64), cy.astype(np.float64)


def _ensure_mesh(spec: DepressionSpec, output_dir: Path) -> Path:
    """Write the synthetic egg-box mesh (idempotent per resolution)."""
    mesh_path = output_dir / f"eggbox-{spec.nx}x{spec.ny}.parquet"
    if not mesh_path.exists():
        verts, quads = build_channel_mesh(spec.nx, spec.ny, spec.domain_l_m, spec.domain_l_m)
        cx, cy = _original_cell_centroids(spec)
        zb = eggbox_bed(
            cx,
            cy,
            domain_l=spec.domain_l_m,
            n_per_side=spec.n_per_side,
            plateau_z=spec.plateau_z_m,
            ne_rise_m=spec.ne_rise_m,
            dep_radius_m=spec.dep_radius_m,
            dep_depth_m=spec.dep_depth_m,
        )
        write_mesh_parquet(mesh_path, verts, quads, zb)
        print(f"[mesh] synthetic egg-box mesh written: {mesh_path} (N={spec.nx * spec.ny})")
    return mesh_path


def _initial_state(spec: DepressionSpec) -> dict[str, object]:
    """Dry-bed initial condition (h = 0 everywhere) in original cell order."""
    n = spec.nx * spec.ny
    zeros = np.zeros(n, dtype=np.float32)
    return {"h": zeros.copy(), "hu": zeros.copy(), "hv": zeros.copy()}


def _hydrograph_phases(spec: DepressionSpec) -> list[SimulationPhase]:
    """Triangular inflow hydrograph as piecewise-constant phases + settle tail."""
    base_s = spec.hydrograph_base_s
    n = spec.hydrograph_phases
    dt_ph = base_s / n
    center = spec.inflow_center_m
    radius = spec.inflow_radius_m
    phases: list[SimulationPhase] = []
    for k in range(n):
        t_mid = (k + 0.5) * dt_ph
        frac = 1.0 - abs(2.0 * t_mid / base_s - 1.0)  # 0 -> 1 -> 0
        q = spec.inflow_peak_m3s * frac
        phases.append(SimulationPhase(duration_s=dt_ph, sources=[PointSource(q, center, radius)]))
    phases.append(SimulationPhase(duration_s=spec.settle_s, sources=[]))
    return phases


def _injected_volume(spec: DepressionSpec) -> float:
    return float(
        sum(
            p.duration_s * sum(s.discharge_m3s for s in p.sources) for p in _hydrograph_phases(spec)
        )
    )


def _make_workflow(
    spec: DepressionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
) -> SWEWorkflow:
    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=spec.manning_n,
        output_interval_s=output_interval_s,
        dt_max=spec.dt_max,
        dt_init=DT_INIT_DEFAULT,
        cfl_interval=spec.cfl_interval,
        progress=progress,
        initial_state_source=_initial_state(spec),
        initial_state_order="original",
        solver_impl=backend,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    return workflow


def _run_once(
    spec: DepressionSpec,
    mesh_path: Path,
    backend: BackendImpl,
    *,
    output_interval_s: float,
    progress: bool,
) -> tuple[SWEWorkflow, WorkflowResult]:
    workflow = _make_workflow(
        spec, mesh_path, backend, output_interval_s=output_interval_s, progress=progress
    )
    result = workflow.run(_hydrograph_phases(spec))
    return workflow, result


def _depression_masks(
    workflow: SWEWorkflow, spec: DepressionSpec
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Nearest solver cell per depression center + a plateau (outside-all) mask."""
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")
    cx = workflow.geom.centroid[:, 0].astype(np.float64)
    cy = workflow.geom.centroid[:, 1].astype(np.float64)
    centers = spec.depression_centers()
    nearest = np.empty(centers.shape[0], dtype=np.int64)
    inside_any = np.zeros(cx.shape[0], dtype=np.bool_)
    for k, (dxk, dyk) in enumerate(centers):
        d2 = (cx - dxk) ** 2 + (cy - dyk) ** 2
        nearest[k] = int(np.argmin(d2))
        inside_any |= d2 < spec.dep_radius_m**2
    plateau = ~inside_any
    return nearest, plateau


def _evaluate_case(
    spec: DepressionSpec,
    backend: str,
    workflow: SWEWorkflow,
    result: WorkflowResult,
) -> CaseMetrics:
    if workflow.geom is None:
        raise RuntimeError("workflow must be prepared")

    h_final = np.asarray(result.h_final, dtype=np.float32)
    finite = bool(np.isfinite(h_final).all())
    min_depth = float(h_final.min()) if h_final.size else 0.0

    injected = _injected_volume(spec)
    mass_balance_rel = (
        abs(result.volume_final_m3 - injected) / injected if injected > 0.0 else float("inf")
    )

    nearest, plateau = _depression_masks(workflow, spec)
    point_levels = [float(h_final[i]) for i in nearest]
    ponded_count = int(sum(1 for lvl in point_levels if lvl > POND_LEVEL_M))

    # Points 15 & 16 (1-based) are the top two of the NE column: the last two
    # entries in the column-major depression ordering.
    far_corner = point_levels[-2:]
    dry_far_corner = all(lvl < DRY_LEVEL_M for lvl in far_corner)

    plateau_wet_frac = (
        float(np.count_nonzero(h_final[plateau] > PLATEAU_WET_M) / np.count_nonzero(plateau))
        if bool(plateau.any())
        else 0.0
    )

    fail_reasons: list[str] = []
    if not finite:
        fail_reasons.append("h_final_non_finite")
    if min_depth < -1e-6:
        fail_reasons.append(f"negative_depth:{min_depth:.3e}")
    if mass_balance_rel > GATE_MASS_BALANCE_REL:
        fail_reasons.append(f"mass_balance_rel:{mass_balance_rel:.3e}>{GATE_MASS_BALANCE_REL:.0e}")
    if ponded_count < GATE_MIN_PONDED:
        fail_reasons.append(f"ponded_count:{ponded_count}<{GATE_MIN_PONDED}")
    if not dry_far_corner:
        fail_reasons.append(f"far_corner_wet:{max(far_corner):.3e}>{DRY_LEVEL_M:.0e}")
    if plateau_wet_frac > GATE_PLATEAU_WET_FRAC:
        fail_reasons.append(f"plateau_wet_frac:{plateau_wet_frac:.3f}>{GATE_PLATEAU_WET_FRAC}")

    return CaseMetrics(
        case=spec.name,
        backend=backend,
        mass_balance_rel=mass_balance_rel,
        min_depth_m=min_depth,
        ponded_count=ponded_count,
        dry_far_corner=dry_far_corner,
        plateau_wet_frac=plateau_wet_frac,
        h_final_finite=finite,
        point_levels_m=point_levels,
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
    spec: DepressionSpec,
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

    case_metrics = _evaluate_case(spec, backend, workflow, result)

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
            vmax=spec.dep_depth_m + 0.5,
            title_prefix=f"{spec.name} depressions — ",
        )

    return case_metrics, perf


def _print_report(all_metrics: list[CaseMetrics], all_perf: list[PerfMetrics]) -> None:
    print()
    print(
        f"gates: mass_balance<={GATE_MASS_BALANCE_REL:.0e} min_depth>=0 finite "
        f"ponded>={GATE_MIN_PONDED} far_corner_dry plateau_wet<={GATE_PLATEAU_WET_FRAC}"
    )
    print(
        f"{'case':10} {'backend':24} {'mass_bal':>9} {'min_h':>10} {'ponded':>7} "
        f"{'dry_15_16':>9} {'plat_wet':>9} {'pass':>5}"
    )
    for m in all_metrics:
        print(
            f"{m.case:10} {m.backend:24} {m.mass_balance_rel:9.3e} {m.min_depth_m:10.3e} "
            f"{m.ponded_count:7d} {m.dry_far_corner!s:>9} {m.plateau_wet_frac:9.3f} {m.passed!s:>5}"
        )
        for reason in m.fail_reasons:
            print(f"  !! {reason}")
        levels = " ".join(f"{lvl:.2f}" for lvl in m.point_levels_m)
        print(f"  point levels (1..{len(m.point_levels_m)}) [m]: {levels}")
    print()
    print(
        f"{'case':10} {'backend':24} {'wall_s':>8} {'steps':>8} {'steps/s':>10} "
        f"{'cell-steps/s':>13} {'determ':>7}"
    )
    for p in all_perf:
        det = "-" if p.deterministic is None else str(p.deterministic)
        print(
            f"{p.case:10} {p.backend:24} {p.median_wall_s:8.3f} {p.steps_total:8d} "
            f"{p.steps_per_s:10.1f} {p.cell_steps_per_s:13.3e} {det:>7}"
        )
        if p.deterministic is False:
            print(f"  !! {p.determinism_detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EA Test 2 floodplain-depressions (flattened egg box) benchmark"
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"CSV of solver_impl names {AVAILABLE_BACKENDS} "
        f"(default: Vulkan reference, runnable without a second backend)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nx", type=int, default=NX_DEFAULT)
    parser.add_argument("--ny", type=int, default=NY_DEFAULT)
    parser.add_argument("--dt-max", type=float, default=DT_MAX_DEFAULT)
    parser.add_argument("--cfl-interval", type=int, default=CFL_INTERVAL_DEFAULT)
    parser.add_argument("--inflow-peak", type=float, default=INFLOW_PEAK_M3S)
    parser.add_argument("--hydro-base", type=float, default=HYDROGRAPH_BASE_S)
    parser.add_argument("--settle", type=float, default=SETTLE_S)
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
        f"[hydrograph] triangular peak {spec.inflow_peak_m3s} m^3/s over "
        f"{spec.hydrograph_base_s / 60.0:.0f} min + {spec.settle_s / 60.0:.0f} min settle; "
        f"injected volume ~= {_injected_volume(spec):.0f} m^3"
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
            "mass_balance_rel": GATE_MASS_BALANCE_REL,
            "min_ponded": GATE_MIN_PONDED,
            "plateau_wet_frac": GATE_PLATEAU_WET_FRAC,
            "determinism_rel": GATE_DETERMINISM_REL,
        },
        "injected_volume_m3": _injected_volume(spec),
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

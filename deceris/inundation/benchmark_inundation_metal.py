"""Benchmark inundation solver variants on Metal with safety invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from deceris.inundation.solver_workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_MESH_PATH = (
    "api/seed/tool/.state/tasks/flood_mesh/downloads/powiat-klodzki-mesh-280k.parquet"
)
DEFAULT_OUTPUT_ROOT = ".tmp/inundation-metal-bench"
AVAILABLE_SOLVER_IMPLS = (
    "batched_submit",
    "async_sync_window",
    "fixed_dt_batch",
    "gpu_resident_batch",
)
DEFAULT_SOLVER_IMPLS = ("batched_submit", "fixed_dt_batch")
ComparedSolver = Literal[
    "batched_submit", "async_sync_window", "fixed_dt_batch", "gpu_resident_batch"
]


@dataclass(frozen=True)
class RunFingerprint:
    """Stable fingerprint + invariants for one simulation run."""

    run_index: int
    solver_impl: str
    snap_count: int
    snap_times: list[float]
    snap_times_hash: str
    snapshot_hashes: list[str]
    h_final_hash: str
    volume_final_bits: int
    volume_injected_bits: int
    max_depth: float
    wet_cells_final: int
    wall_seconds: float
    invariant_ok: bool
    invariant_reason: str


def _save_depth_png(
    workflow: SWEWorkflow, h_final: NDArray[np.float32], t_final_s: float, out_path: Path
) -> None:
    """Save a water-depth PNG (see new/workflow.py for the plotting recipe this mirrors).

    Requires matplotlib — install via ``uv pip install -e ".[viz]"`` or
    ``pip install -r requirements.txt`` (both now include it). Imported lazily
    so benchmark runs without --plot-dir never need matplotlib installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot-dir. Install with: "
            'uv pip install -e ".[viz]" (or pip install -r requirements.txt)'
        ) from exc

    if workflow.geom is None or workflow.perm is None:
        raise RuntimeError("Workflow must be prepared before plotting depth")

    verts = workflow.verts
    faces_flat = workflow.faces_flat
    face_offsets = workflow.face_offsets
    if verts is None or faces_flat is None or face_offsets is None:
        # Geometry-cache hit path (solver_workflow.py's prepare()) skips
        # load_mesh_file entirely, so verts/faces_flat/face_offsets are never
        # populated on the workflow — the cached .npz only stores geom+perm,
        # not raw mesh vertices. Re-read the mesh file just for plotting.
        from deceris.inundation.swe_mesh import load_mesh_file

        sys.stdout.write(f"[plot] re-loading mesh for plotting: {workflow.config.mesh_source}\n")
        sys.stdout.flush()
        verts, faces_flat, face_offsets, _zb_from_file = load_mesh_file(workflow.config.mesh_source)

    perm = workflow.perm
    n_cells = workflow.geom.area.shape[0]

    polygons_reordered = []
    for new_i in range(n_cells):
        old_i = int(perm[new_i])
        idx = faces_flat[face_offsets[old_i] : face_offsets[old_i + 1]]
        polygons_reordered.append(verts[idx])

    water_cmap_offset = 0.28
    water_base = plt.cm.Blues(np.linspace(water_cmap_offset, 1.0, 256))
    water_cmap = LinearSegmentedColormap.from_list("BluesOffset", water_base)

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    pc = PolyCollection(
        polygons_reordered,
        array=h_final,
        cmap=water_cmap,
        edgecolors="face",
        linewidths=0.1,
        norm=Normalize(vmin=0.0, vmax=4.0),
    )
    ax.add_collection(pc)
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    fig.colorbar(pc, ax=ax, label="h [m]")
    ax.set_aspect("equal")
    ax.set_title(f"Water depth at t={t_final_s:.1f}s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    sys.stdout.write(f"[plot] water-depth PNG saved: {out_path}\n")
    sys.stdout.flush()


def _hash_array(arr: NDArray[np.float32]) -> str:
    a = np.ascontiguousarray(arr)
    data = a.view(np.uint8).tobytes()
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def _float64_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def _build_pipeline_like_phases(
    workflow: SWEWorkflow, phase_duration_s: float
) -> list[SimulationPhase]:
    """Return deterministic two-phase schedule resembling production threat runs."""
    if workflow.geom is None:
        raise RuntimeError("Workflow must be prepared before building benchmark phases")

    valid = workflow.geom.area > workflow.config.area_tol
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        raise RuntimeError("Benchmark mesh must contain at least two valid cells")
    sorted_indices = valid_indices[np.argsort(workflow.geom.centroid[valid_indices, 0])]
    source_a = int(sorted_indices[sorted_indices.size // 3])
    source_b = int(sorted_indices[(sorted_indices.size * 2) // 3])
    radius_a = max(1.0, float(np.sqrt(workflow.geom.area[source_a] / np.pi)))
    radius_b = max(1.0, float(np.sqrt(workflow.geom.area[source_b] / np.pi)))
    center_a = (
        float(workflow.geom.centroid[source_a, 0]),
        float(workflow.geom.centroid[source_a, 1]),
    )
    center_b = (
        float(workflow.geom.centroid[source_b, 0]),
        float(workflow.geom.centroid[source_b, 1]),
    )

    return [
        SimulationPhase(
            duration_s=phase_duration_s,
            sources=[
                PointSource(300.0, center_a, radius_a),
                PointSource(306.0, center_b, radius_b),
            ],
        ),
        SimulationPhase(
            duration_s=phase_duration_s,
            sources=[
                PointSource(320.0, center_a, radius_a),
                PointSource(314.5, center_b, radius_b),
            ],
        ),
    ]


def _validate_run_invariants(
    snap_times: NDArray[np.float32],
    snapshots: list[NDArray[np.bool_]],
    h_final: NDArray[np.float32],
    volume_final: float,
    volume_injected: float,
    expected_final_time_s: float,
) -> tuple[bool, str]:
    """Validate production-like correctness invariants for one run."""
    if len(snapshots) < 10:
        return False, f"too_few_snapshots:{len(snapshots)}"
    if not np.all(np.diff(snap_times) >= 0.0):
        return False, "non_monotonic_snap_times"
    if abs(snap_times[-1] - expected_final_time_s) > 5.0:
        return False, f"unexpected_final_time:{snap_times[-1]:.3f}"
    max_depth = float(np.max(h_final))
    if max_depth <= 0.05:
        return False, f"max_depth_too_small:{max_depth:.6f}"
    wet_cells_final = int(np.count_nonzero(h_final > 1e-6))
    if wet_cells_final < 100:
        return False, f"too_few_wet_cells:{wet_cells_final}"
    if volume_injected <= 0.0:
        return False, f"invalid_volume_injected:{volume_injected:.3f}"
    vol_ratio = volume_final / volume_injected
    if vol_ratio < 0.01:
        return False, f"volume_ratio_too_small:{vol_ratio:.6f}"
    if not np.isfinite(vol_ratio):
        return False, "volume_ratio_non_finite"
    return True, "ok"


def _run_once(
    mesh_path: Path,
    solver_impl: Literal[
        "baseline", "batched_submit", "async_sync_window", "fixed_dt_batch", "gpu_resident_batch"
    ],
    run_index: int,
    solver_progress: bool,
    phase_duration_s: float,
    output_interval_s: float,
    dt_max: float,
    cfl_interval: int,
    plot_path: Path | None = None,
) -> RunFingerprint:
    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=0.035,
        output_interval_s=output_interval_s,
        dt_max=dt_max,
        cfl_interval=cfl_interval,
        progress=solver_progress,
        solver_impl=solver_impl,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    result = workflow.run(_build_pipeline_like_phases(workflow, phase_duration_s))

    snap_times = np.asarray(result.snap_times, dtype=np.float64)
    snapshots = [np.asarray(s, dtype=np.float32) for s in result.snapshots]
    h_final = np.asarray(result.h_final, dtype=np.float32)

    if plot_path is not None:
        _save_depth_png(workflow, h_final, float(snap_times[-1]), plot_path)

    invariant_ok, invariant_reason = _validate_run_invariants(
        snap_times=snap_times,
        snapshots=snapshots,
        h_final=h_final,
        volume_final=float(result.volume_final_m3),
        volume_injected=float(result.volume_injected_m3),
        expected_final_time_s=phase_duration_s * 2.0,
    )

    return RunFingerprint(
        run_index=run_index,
        solver_impl=solver_impl,
        snap_count=len(snapshots),
        snap_times=[float(t) for t in snap_times],
        snap_times_hash=_hash_array(snap_times),
        snapshot_hashes=[_hash_array(snapshot) for snapshot in snapshots],
        h_final_hash=_hash_array(h_final),
        volume_final_bits=_float64_bits(result.volume_final_m3),
        volume_injected_bits=_float64_bits(result.volume_injected_m3),
        max_depth=float(np.max(h_final)),
        wet_cells_final=int(np.count_nonzero(h_final > 1e-6)),
        wall_seconds=float(result.wall_seconds),
        invariant_ok=invariant_ok,
        invariant_reason=invariant_reason,
    )


def _compare(a: RunFingerprint, b: RunFingerprint) -> tuple[bool, str]:
    if a.snap_count != b.snap_count:
        return False, f"snap_count mismatch: {a.snap_count} != {b.snap_count}"
    if a.snap_times_hash != b.snap_times_hash:
        return False, "snap_times_hash mismatch"
    if a.snapshot_hashes != b.snapshot_hashes:
        return False, "snapshot_hashes mismatch"
    if a.h_final_hash != b.h_final_hash:
        return False, "h_final_hash mismatch"
    if a.volume_final_bits != b.volume_final_bits:
        return False, "volume_final_bits mismatch"
    if a.volume_injected_bits != b.volume_injected_bits:
        return False, "volume_injected_bits mismatch"
    return True, "ok"


def _run_group(
    mesh_path: Path,
    solver_impl: Literal[
        "baseline", "batched_submit", "async_sync_window", "fixed_dt_batch", "gpu_resident_batch"
    ],
    repeats: int,
    warmup: int,
    output_dir: Path,
    solver_progress: bool,
    phase_duration_s: float,
    output_interval_s: float,
    dt_max: float,
    cfl_interval: int,
    plot_dir: Path | None = None,
) -> tuple[list[RunFingerprint], float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = warmup + repeats
    completed = 0
    wall_samples: list[float] = []

    for warmup_index in range(warmup):
        sys.stdout.write(
            f"[{solver_impl}] warmup {warmup_index + 1}/{warmup} "
            f"(overall {completed + 1}/{total_runs}) start\n"
        )
        sys.stdout.flush()
        warmup_fp = _run_once(
            mesh_path,
            solver_impl,
            run_index=-1 - warmup_index,
            solver_progress=solver_progress,
            phase_duration_s=phase_duration_s,
            output_interval_s=output_interval_s,
            dt_max=dt_max,
            cfl_interval=cfl_interval,
        )
        wall_samples.append(warmup_fp.wall_seconds)
        completed += 1
        avg_wall = statistics.fmean(wall_samples)
        eta = max(avg_wall * (total_runs - completed), 0.0)
        sys.stdout.write(
            f"[{solver_impl}] warmup {warmup_index + 1}/{warmup} done "
            f"({warmup_fp.wall_seconds:.2f}s, eta~{eta:.1f}s, "
            f"inv={warmup_fp.invariant_reason})\n"
        )
        sys.stdout.flush()

    runs: list[RunFingerprint] = []
    for run_index in range(repeats):
        sys.stdout.write(
            f"[{solver_impl}] measured {run_index + 1}/{repeats} "
            f"(overall {completed + 1}/{total_runs}) start\n"
        )
        sys.stdout.flush()
        fingerprint = _run_once(
            mesh_path,
            solver_impl,
            run_index=run_index,
            solver_progress=solver_progress,
            phase_duration_s=phase_duration_s,
            output_interval_s=output_interval_s,
            dt_max=dt_max,
            cfl_interval=cfl_interval,
            plot_path=(plot_dir / f"water_depth_{solver_impl}_run{run_index}.png")
            if plot_dir is not None
            else None,
        )
        runs.append(fingerprint)
        wall_samples.append(fingerprint.wall_seconds)
        completed += 1
        avg_wall = statistics.fmean(wall_samples)
        eta = max(avg_wall * (total_runs - completed), 0.0)
        sys.stdout.write(
            f"[{solver_impl}] measured {run_index + 1}/{repeats} done "
            f"({fingerprint.wall_seconds:.2f}s, eta~{eta:.1f}s, "
            f"inv={fingerprint.invariant_reason})\n"
        )
        sys.stdout.flush()
        (output_dir / f"run_{run_index}.json").write_text(
            json.dumps(asdict(fingerprint), indent=2),
            encoding="utf-8",
        )

    median_wall = statistics.median(fp.wall_seconds for fp in runs)
    return runs, median_wall


def _assert_deterministic(runs: list[RunFingerprint]) -> tuple[bool, str]:
    reference = runs[0]
    for index, run in enumerate(runs[1:], start=1):
        ok, reason = _compare(reference, run)
        if not ok:
            return False, f"run0 vs run{index}: {reason}"
    return True, "ok"


def _all_invariants_ok(runs: list[RunFingerprint]) -> tuple[bool, str]:
    for run in runs:
        if not run.invariant_ok:
            return False, f"run{run.run_index}:{run.invariant_reason}"
    return True, "ok"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default=DEFAULT_MESH_PATH)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--solver-progress", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--require-bitwise-parity", action="store_true")
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=AVAILABLE_SOLVER_IMPLS,
        default=list(DEFAULT_SOLVER_IMPLS),
    )
    parser.add_argument("--phase-duration-s", type=float, default=300.0)
    parser.add_argument("--output-interval-s", type=float, default=60.0)
    parser.add_argument("--dt-max", type=float, default=0.5)
    parser.add_argument("--cfl-interval", type=int, default=10)
    parser.add_argument(
        "--plot-dir",
        default=None,
        help=(
            "If set, save a water-depth PNG per solver impl (first measured run) "
            'to this directory. Requires matplotlib: uv pip install -e ".[viz]"'
        ),
    )
    return parser


def _validate_runtime_deps() -> None:
    """Fail fast with actionable message when GPU benchmark deps are missing."""
    missing: list[str] = []
    for module_name in ("kp", "hilbertcurve"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing benchmark dependencies: {missing_csv}. "
            "Use GPU worker environment (.venv-gpu). Run: just gpu-worker::setup"
        )


def main() -> int:
    """Run benchmark and enforce parity + invariants under pipeline-like settings."""
    _validate_runtime_deps()
    args = _build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.phase_duration_s <= 0.0:
        raise ValueError("--phase-duration-s must be > 0")
    if args.output_interval_s <= 0.0:
        raise ValueError("--output-interval-s must be > 0")
    if args.dt_max <= 0.0:
        raise ValueError("--dt-max must be > 0")
    if args.cfl_interval < 1:
        raise ValueError("--cfl-interval must be >= 1")

    repo_root = Path(__file__).resolve().parents[4]
    mesh_path = Path(args.mesh)
    if not mesh_path.is_absolute():
        mesh_path = repo_root / mesh_path
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    plot_dir: Path | None = None
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir)
        if not plot_dir.is_absolute():
            plot_dir = repo_root / plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

    solver_runs: dict[str, list[RunFingerprint]] = {}
    solver_medians: dict[str, float] = {}
    solver_det: dict[str, tuple[bool, str]] = {}
    solver_inv: dict[str, tuple[bool, str]] = {}

    for solver_impl in args.solvers:
        runs, median = _run_group(
            mesh_path,
            solver_impl,
            args.repeats,
            args.warmup,
            output_root / solver_impl,
            args.solver_progress,
            args.phase_duration_s,
            args.output_interval_s,
            args.dt_max,
            args.cfl_interval,
            plot_dir,
        )
        solver_runs[solver_impl] = runs
        solver_medians[solver_impl] = median
        solver_det[solver_impl] = _assert_deterministic(runs)
        solver_inv[solver_impl] = _all_invariants_ok(runs)

    batched_submit_runs = solver_runs.get("batched_submit")
    batched_submit_median = solver_medians.get("batched_submit")
    parity: dict[str, tuple[bool, str]] = {}
    if batched_submit_runs is not None:
        for solver_impl, runs in solver_runs.items():
            if solver_impl == "batched_submit":
                continue
            parity[f"batched_submit_vs_{solver_impl}"] = _compare(batched_submit_runs[0], runs[0])

    baseline_det_ok = None
    baseline_det_reason = None
    baseline_inv_ok = None
    baseline_inv_reason = None
    baseline_median = None
    batched_submit_vs_baseline_parity_ok = None
    batched_submit_vs_baseline_parity_reason = None
    async_sync_window_vs_baseline_parity_ok = None
    async_sync_window_vs_baseline_parity_reason = None
    fixed_dt_batch_vs_baseline_parity_ok = None
    fixed_dt_batch_vs_baseline_parity_reason = None
    gpu_resident_batch_vs_baseline_parity_ok = None
    gpu_resident_batch_vs_baseline_parity_reason = None

    if args.include_baseline:
        baseline_runs, baseline_median = _run_group(
            mesh_path,
            "baseline",
            args.repeats,
            args.warmup,
            output_root / "baseline",
            args.solver_progress,
            args.phase_duration_s,
            args.output_interval_s,
            args.dt_max,
            args.cfl_interval,
            plot_dir,
        )
        baseline_det_ok, baseline_det_reason = _assert_deterministic(baseline_runs)
        baseline_inv_ok, baseline_inv_reason = _all_invariants_ok(baseline_runs)
        if "batched_submit" in solver_runs:
            batched_submit_vs_baseline_parity_ok, batched_submit_vs_baseline_parity_reason = (
                _compare(
                    baseline_runs[0],
                    solver_runs["batched_submit"][0],
                )
            )
        if "async_sync_window" in solver_runs:
            async_sync_window_vs_baseline_parity_ok, async_sync_window_vs_baseline_parity_reason = (
                _compare(
                    baseline_runs[0],
                    solver_runs["async_sync_window"][0],
                )
            )
        if "fixed_dt_batch" in solver_runs:
            fixed_dt_batch_vs_baseline_parity_ok, fixed_dt_batch_vs_baseline_parity_reason = (
                _compare(
                    baseline_runs[0],
                    solver_runs["fixed_dt_batch"][0],
                )
            )
        if "gpu_resident_batch" in solver_runs:
            (
                gpu_resident_batch_vs_baseline_parity_ok,
                gpu_resident_batch_vs_baseline_parity_reason,
            ) = _compare(
                baseline_runs[0],
                solver_runs["gpu_resident_batch"][0],
            )

    summary = {
        "mesh": str(mesh_path),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "solver_progress": args.solver_progress,
        "solvers": args.solvers,
        "require_bitwise_parity": args.require_bitwise_parity,
        "scenario": {
            "duration_total_s": args.phase_duration_s * 2.0,
            "phase_count": 2,
            "phase_duration_s": args.phase_duration_s,
            "dt_max": args.dt_max,
            "cfl_interval": args.cfl_interval,
            "output_interval_s": args.output_interval_s,
        },
        "solver_deterministic": {key: value[0] for key, value in sorted(solver_det.items())},
        "solver_deterministic_reason": {key: value[1] for key, value in sorted(solver_det.items())},
        "solver_invariants_ok": {key: value[0] for key, value in sorted(solver_inv.items())},
        "solver_invariants_reason": {key: value[1] for key, value in sorted(solver_inv.items())},
        "batched_submit_parity": {key: value[0] for key, value in sorted(parity.items())},
        "batched_submit_parity_reason": {key: value[1] for key, value in sorted(parity.items())},
        "median_wall_seconds": solver_medians,
        "speedup_vs_batched_submit": {
            key: (
                batched_submit_median / value
                if batched_submit_median is not None and value > 0
                else None
            )
            for key, value in sorted(solver_medians.items())
            if key != "batched_submit"
        },
        "include_baseline": args.include_baseline,
        "baseline_deterministic": baseline_det_ok,
        "baseline_deterministic_reason": baseline_det_reason,
        "baseline_invariants_ok": baseline_inv_ok,
        "baseline_invariants_reason": baseline_inv_reason,
        "baseline_median_wall_seconds": baseline_median,
        "batched_submit_vs_baseline_bitwise_parity": batched_submit_vs_baseline_parity_ok,
        "batched_submit_vs_baseline_bitwise_parity_reason": (
            batched_submit_vs_baseline_parity_reason
        ),
        "async_sync_window_vs_baseline_bitwise_parity": async_sync_window_vs_baseline_parity_ok,
        "async_sync_window_vs_baseline_bitwise_parity_reason": (
            async_sync_window_vs_baseline_parity_reason
        ),
        "fixed_dt_batch_vs_baseline_bitwise_parity": fixed_dt_batch_vs_baseline_parity_ok,
        "fixed_dt_batch_vs_baseline_bitwise_parity_reason": (
            fixed_dt_batch_vs_baseline_parity_reason
        ),
        "gpu_resident_batch_vs_baseline_bitwise_parity": gpu_resident_batch_vs_baseline_parity_ok,
        "gpu_resident_batch_vs_baseline_bitwise_parity_reason": (
            gpu_resident_batch_vs_baseline_parity_reason
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(f"Summary written to {summary_path}\n")

    has_fail = any(not value[0] for value in solver_det.values())
    has_fail = has_fail or any(not value[0] for value in solver_inv.values())
    if args.require_bitwise_parity:
        has_fail = has_fail or any(not value[0] for value in parity.values())
    if args.include_baseline:
        has_fail = has_fail or (baseline_det_ok is not True)
        has_fail = has_fail or (baseline_inv_ok is not True)
        if args.require_bitwise_parity and "batched_submit" in solver_runs:
            has_fail = has_fail or (batched_submit_vs_baseline_parity_ok is not True)
        if args.require_bitwise_parity and "async_sync_window" in solver_runs:
            has_fail = has_fail or (async_sync_window_vs_baseline_parity_ok is not True)
        if args.require_bitwise_parity and "fixed_dt_batch" in solver_runs:
            has_fail = has_fail or (fixed_dt_batch_vs_baseline_parity_ok is not True)
        if args.require_bitwise_parity and "gpu_resident_batch" in solver_runs:
            has_fail = has_fail or (gpu_resident_batch_vs_baseline_parity_ok is not True)

    if has_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared data, plotting, fingerprint, and invariant helpers for benchmarks."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .solver_workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowResult,
)
from .swe_mesh import load_mesh_file
from .swe_tuning import (
    FINAL_TIME_TOLERANCE_S,
    MIN_DEPTH_THRESHOLD_M,
    MIN_REPEATS_FOR_DETERMINISM,
    MIN_SNAPSHOTS,
    MIN_VALID_CELLS,
    MIN_VOLUME_RATIO,
    MIN_WET_CELLS,
    WET_CELL_THRESHOLD,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


# Deterministic two-phase threat-run schedule.
_PHASE1_FLOW_RATES: tuple[float, float] = (300.0, 306.0)
_PHASE2_FLOW_RATES: tuple[float, float] = (320.0, 314.5)


@dataclass(frozen=True)
class RunFingerprint:
    """Stable fingerprint + invariants for one simulation run.

    ``steps_total`` is populated for both backends — ``WorkflowResult.steps_total``
    is sourced from ``getattr(self.solver, "steps_total", 0)`` and the Vulkan
    ``SWESolver`` base class maintains the counter (see ``swe_gpu.py``).
    """

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
    steps_total: int
    invariant_ok: bool
    invariant_reason: str


def hash_array(arr: NDArray[np.floating[Any]]) -> str:
    a = np.ascontiguousarray(arr)
    data = a.view(np.uint8).tobytes()
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def float64_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def validate_run_invariants(
    snap_times: NDArray[np.floating[Any]],
    snapshots: Sequence[NDArray[np.floating[Any]]],
    h_final: NDArray[np.floating[Any]],
    volume_final: float,
    volume_injected: float,
    expected_final_time_s: float,
    *,
    min_snapshots: int = MIN_SNAPSHOTS,
) -> tuple[bool, str]:
    """Validate production-like correctness invariants for one run.

    The ``min_snapshots`` kwarg lets shorter warm-start half-runs use a lower
    bar without weakening the full-duration determinism/parity checks.
    """
    if len(snapshots) < min_snapshots:
        return False, f"too_few_snapshots:{len(snapshots)}"
    if not np.all(np.diff(snap_times) >= 0.0):
        return False, "non_monotonic_snap_times"
    if abs(snap_times[-1] - expected_final_time_s) > FINAL_TIME_TOLERANCE_S:
        return False, f"unexpected_final_time:{snap_times[-1]:.3f}"
    if not np.isfinite(h_final).all():
        return False, "h_final_non_finite"
    max_depth = float(np.max(h_final))
    if max_depth <= MIN_DEPTH_THRESHOLD_M:
        return False, f"max_depth_too_small:{max_depth:.6f}"
    wet_cells_final = int(np.count_nonzero(h_final > WET_CELL_THRESHOLD))
    if wet_cells_final < MIN_WET_CELLS:
        return False, f"too_few_wet_cells:{wet_cells_final}"
    if volume_injected <= 0.0:
        return False, f"invalid_volume_injected:{volume_injected:.3f}"
    vol_ratio = volume_final / volume_injected
    if not np.isfinite(vol_ratio):
        return False, "volume_ratio_non_finite"
    if vol_ratio < MIN_VOLUME_RATIO:
        return False, f"volume_ratio_too_small:{vol_ratio:.6f}"
    return True, "ok"


def build_fingerprint(
    run_index: int,
    solver_impl: str,
    result: WorkflowResult,
    expected_final_time_s: float,
    *,
    min_snapshots: int = MIN_SNAPSHOTS,
) -> RunFingerprint:
    """Build a RunFingerprint from a completed WorkflowResult."""
    snap_times = np.asarray(result.snap_times, dtype=np.float64)
    snapshots = [np.asarray(s, dtype=np.float32) for s in result.snapshots]
    h_final = np.asarray(result.h_final, dtype=np.float32)
    invariant_ok, invariant_reason = validate_run_invariants(
        snap_times=snap_times,
        snapshots=snapshots,
        h_final=h_final,
        volume_final=float(result.volume_final_m3),
        volume_injected=float(result.volume_injected_m3),
        expected_final_time_s=expected_final_time_s,
        min_snapshots=min_snapshots,
    )
    return RunFingerprint(
        run_index=run_index,
        solver_impl=solver_impl,
        snap_count=len(snapshots),
        snap_times=[float(t) for t in snap_times],
        snap_times_hash=hash_array(snap_times),
        snapshot_hashes=[hash_array(s) for s in snapshots],
        h_final_hash=hash_array(h_final),
        volume_final_bits=float64_bits(result.volume_final_m3),
        volume_injected_bits=float64_bits(result.volume_injected_m3),
        max_depth=float(np.max(h_final)) if h_final.size else 0.0,
        wet_cells_final=int(np.count_nonzero(h_final > WET_CELL_THRESHOLD)),
        wall_seconds=float(result.wall_seconds),
        steps_total=int(result.steps_total),
        invariant_ok=invariant_ok,
        invariant_reason=invariant_reason,
    )


def save_depth_png(
    workflow: SWEWorkflow,
    h_final: NDArray[np.float32],
    t_final_s: float,
    out_path: Path,
) -> None:
    """Save a water-depth PNG. Requires matplotlib (--viz extra)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot[/-dir]. Install with: "
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
        sys.stdout.write(f"[plot] re-loading mesh for plotting: {workflow.config.mesh_source}\n")
        sys.stdout.flush()
        verts, faces_flat, face_offsets, _zb_from_file = load_mesh_file(workflow.config.mesh_source)

    perm = workflow.perm
    n_cells = workflow.geom.area.shape[0]

    polygons_reordered: list[NDArray[np.float32]] = []
    for new_i in range(n_cells):
        old_i = int(perm[new_i])
        idx = faces_flat[face_offsets[old_i] : face_offsets[old_i + 1]]
        polygons_reordered.append(verts[idx])

    water_cmap_offset = 0.28
    water_base: NDArray[np.floating[Any]] = cast(
        "NDArray[np.floating[Any]]",
        cast("Any", plt.cm.Blues)(np.linspace(water_cmap_offset, 1.0, 256)),
    )
    water_cmap = LinearSegmentedColormap.from_list("BluesOffset", water_base)

    _plt: Any = plt
    fig_ax: tuple[Figure, Axes] = cast("tuple[Figure, Axes]", _plt.subplots(1, 1, figsize=(13, 5)))
    fig, ax = fig_ax
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
    cast("Any", fig).colorbar(pc, ax=ax, label="h [m]")
    ax.set_aspect("equal")
    cast("Any", ax).set_title(f"Water depth at t={t_final_s:.1f}s")
    cast("Any", ax).set_xlabel("x")
    cast("Any", ax).set_ylabel("y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    cast("Any", fig).savefig(out_path, dpi=300)
    plt.close(fig)
    sys.stdout.write(f"[plot] water-depth PNG saved: {out_path}\n")
    sys.stdout.flush()


def build_pipeline_like_phases(
    workflow: SWEWorkflow,
    phase_duration_s: float,
) -> list[SimulationPhase]:
    """Return a deterministic two-phase schedule resembling production threat runs.

    Centers and radii derive from the mesh geometry.
    """
    if workflow.geom is None:
        raise RuntimeError("Workflow must be prepared before building benchmark phases")
    valid = workflow.geom.area > workflow.config.area_tol
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < MIN_VALID_CELLS:
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

    def _phase(flow_rates: tuple[float, float]) -> SimulationPhase:
        return SimulationPhase(
            duration_s=phase_duration_s,
            sources=[
                PointSource(flow_rates[0], center_a, radius_a),
                PointSource(flow_rates[1], center_b, radius_b),
            ],
        )

    return [_phase(_PHASE1_FLOW_RATES), _phase(_PHASE2_FLOW_RATES)]


def compare_fingerprints_exact(a: RunFingerprint, b: RunFingerprint) -> tuple[bool, str]:
    """Bit-exact comparison — only valid for repeats of the *same* solver_impl."""
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


def assert_deterministic(runs: list[RunFingerprint]) -> tuple[bool | None, str]:
    if len(runs) < MIN_REPEATS_FOR_DETERMINISM:
        return None, "skipped (repeats=1)"
    reference = runs[0]
    for index, run in enumerate(runs[1:], start=1):
        ok, reason = compare_fingerprints_exact(reference, run)
        if not ok:
            return False, f"run0 vs run{index}: {reason}"
    return True, "ok"


def all_invariants_ok(runs: list[RunFingerprint]) -> tuple[bool, str]:
    for run in runs:
        if not run.invariant_ok:
            return False, f"run{run.run_index}:{run.invariant_reason}"
    return True, "ok"


def write_fingerprint_json(fingerprint: RunFingerprint, out_path: Path) -> None:
    out_path.write_text(json.dumps(asdict(fingerprint), indent=2), encoding="utf-8")


def median_wall(runs: list[RunFingerprint]) -> float:
    return statistics.median(fp.wall_seconds for fp in runs)

"""Application-facing workflow wrapper for the mesh-based SWE GPU solver.

This module turns the notebook-style sequence into a reusable API that can be
called from an existing Python application.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .swe_geometry import MeshGeometry, build_geometry, hilbert_reorder
from .swe_geometry_cache import geometry_cache_key, load_geometry_cache, save_geometry_cache
from .swe_mesh import load_mesh_file
from .swe_shaders import SOURCE_GLSL, compile_all, compile_glsl
from .swe_tuning import (
    MIN_NDIM_NPY_INPUT,
    NPY_COLS_HUV,
    NPY_COLS_WITH_MANNING,
    REQUIRED_CLI_ARGS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .swe_gpu_async_sync_window import SWESolverAsyncSyncWindow
    from .swe_gpu_baseline import SWESolverBaseline
    from .swe_gpu_batched_submit import SWESolverBatchedSubmit
    from .swe_gpu_device_cfl import SWESolverDeviceCfl
    from .swe_gpu_fixed_dt_batch import SWESolverFixedDtBatch
    from .swe_gpu_gpu_resident_batch import SWESolverGpuResidentBatch


# Metadata keys that ``_load_initial_state`` ignores rather than rejecting.
INITIAL_STATE_METADATA_KEYS = frozenset(
    {"saved_order", "mesh_source", "t_end_s", "dt_max", "cfl_interval"}
)


@dataclass(frozen=True)
class PointSource:
    """Volumetric source over a circular area.

    discharge_m3s: total discharge assigned to this source [m^3/s].
    center_xy:     source center in mesh coordinates.
    radius_m:      source radius in mesh units.
    """

    discharge_m3s: float
    center_xy: tuple[float, float]
    radius_m: float


@dataclass(frozen=True)
class SimulationPhase:
    """One phase in a multi-phase hydrograph schedule."""

    duration_s: float
    sources: Sequence[PointSource]


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime configuration for SWEWorkflow."""

    mesh_source: str
    manning_n: float
    output_interval_s: float
    dt_max: float
    cfl_interval: int
    gravity: float = 9.81
    dry_tol: float = 1e-4
    cfl: float = 0.45
    workgroup_size: int = 256
    dt_init: float = 1e-2
    use_hilbert_reorder: bool = True
    progress: bool = True
    area_tol: float = 1e-6
    initial_state_source: dict[str, object] | str | os.PathLike[str] | None = None
    initial_state_order: str = "solver"
    initial_state_id_col: str = "cell_id"
    geometry_cache_dir: str | os.PathLike[str] | None = None
    output_every_steps: int | None = None
    solver_impl: Literal[
        "baseline",
        "batched_submit",
        "device_cfl",
        "async_sync_window",
        "fixed_dt_batch",
        "fixed_dt_batch_barrier",
        "gpu_resident_batch",
    ] = "baseline"

    def __post_init__(self) -> None:
        """Validate that required solver parameters are within sensible ranges."""
        if self.manning_n <= 0:
            raise ValueError(f"manning_n must be positive, got {self.manning_n}")
        if self.output_interval_s <= 0:
            raise ValueError(f"output_interval_s must be positive, got {self.output_interval_s}")
        if self.dt_max <= 0:
            raise ValueError(f"dt_max must be positive, got {self.dt_max}")
        if self.cfl_interval <= 0:
            raise ValueError(f"cfl_interval must be positive, got {self.cfl_interval}")
        if self.steps_per_graph is not None and self.steps_per_graph <= 0:
            raise ValueError(f"steps_per_graph must be positive, got {self.steps_per_graph}")
        if self.output_every_steps is not None and self.output_every_steps <= 0:
            raise ValueError(f"output_every_steps must be positive, got {self.output_every_steps}")
        if self.solver_impl not in {
            "baseline",
            "batched_submit",
            "device_cfl",
            "async_sync_window",
            "fixed_dt_batch",
            "fixed_dt_batch_barrier",
            "gpu_resident_batch",
        }:
            raise ValueError(f"unsupported solver_impl: {self.solver_impl}")


@dataclass
class WorkflowResult:
    """Primary outputs needed by application code."""

    snapshots: list[NDArray[np.float32]]
    snap_times: list[float]
    h_final: NDArray[np.float32]
    volume_final_m3: float
    volume_injected_m3: float
    wall_seconds: float
    steps_total: int
    hu_final: NDArray[np.float32] | None = None
    hv_final: NDArray[np.float32] | None = None


def select_cells_within_radius(
    centroid: NDArray[np.float32],
    center_xy: tuple[float, float],
    radius_m: float,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Return a boolean mask of cells within a circle."""
    center = np.asarray(center_xy, dtype=np.float32)
    dist = np.hypot(centroid[:, 0] - center[0], centroid[:, 1] - center[1])
    mask = dist <= float(radius_m)
    if valid_mask is not None:
        mask &= valid_mask
    return mask


def _as_1d_float32(name: str, values: object, n_expected: int) -> NDArray[np.float32]:
    """Convert an input vector to finite float32 and validate expected length."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != n_expected:
        raise ValueError(f"{name}: expected {n_expected} values, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: contains non-finite values")
    return arr


def _load_initial_state(
    source: dict[str, object] | str | os.PathLike[str] | Path | None,
    n_cells: int,
    zb_local: NDArray[np.float32],
    *,
    order: str,
    id_col: str,
    perm: NDArray[np.int32],
    dry_tol: float,
) -> (
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]
    | None
):
    """Load optional per-cell initial state (h, hu, hv, optional n_mann)."""
    if source is None:
        return None

    if isinstance(source, dict):
        state: dict[str, NDArray[Any]] = {k: np.asarray(v) for k, v in source.items()}
    else:
        path = Path(os.fspath(source))
        if not path.exists():
            raise FileNotFoundError(f"Initial state file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path) as npz:
                state = {k: npz[k] for k in npz.files}
        elif suffix == ".npy":
            arr = np.load(path)
            if arr.ndim != MIN_NDIM_NPY_INPUT or arr.shape[1] not in (
                NPY_COLS_HUV,
                NPY_COLS_WITH_MANNING,
            ):
                raise ValueError(
                    "For .npy input, expected shape (N,3)=[h,hu,hv] or (N,4)=[h,hu,hv,n_mann]"
                )
            state = {
                "h": arr[:, 0],
                "hu": arr[:, 1],
                "hv": arr[:, 2],
            }
            if arr.shape[1] == NPY_COLS_WITH_MANNING:
                state["n_mann"] = arr[:, 3]
        elif suffix in (".csv", ".parquet"):
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError(
                    "pandas is required to load .csv/.parquet initial state files"
                ) from exc

            if suffix == ".csv":
                df: Any = pd.read_csv(path)
            else:
                _pd: Any = pd
                df = _pd.read_parquet(path)
            state: dict[str, NDArray[Any]] = {c: df[c].to_numpy() for c in df.columns}
        else:
            raise ValueError(
                f"Unsupported initial state format: {suffix}. Use .npz, .npy, .csv, or .parquet"
            )

    # Keep only recognized per-cell state fields; ignore metadata entries.
    valid_state_keys = {"h", "wse", "hu", "hv", "u", "v", "n_mann", id_col}
    state = {k: v for k, v in state.items() if k not in INITIAL_STATE_METADATA_KEYS}
    invalid_state_keys = sorted(k for k in state if k not in valid_state_keys)
    if invalid_state_keys:
        raise ValueError(
            "initial state contains unsupported fields: "
            f"{invalid_state_keys}. "
            f"Supported fields: {sorted(valid_state_keys)}"
        )

    # Optional explicit row->cell mapping via integer cell_id.
    if id_col in state:
        raw_cell_id = np.asarray(state[id_col]).reshape(-1)
        try:
            cell_id_f = raw_cell_id.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{id_col}: values must be numeric integers") from exc

        if not np.isfinite(cell_id_f).all():
            raise ValueError(f"{id_col}: values must be finite numeric integers")
        if not np.equal(cell_id_f, np.floor(cell_id_f)).all():
            raise ValueError(f"{id_col}: values must be integers")

        cell_id = cell_id_f.astype(np.int64)
        if cell_id.size != n_cells:
            raise ValueError(f"{id_col}: expected {n_cells} rows, got {cell_id.size}")
        if np.unique(cell_id).size != n_cells:
            raise ValueError(f"{id_col}: values must be unique")
        if cell_id.min() < 0 or cell_id.max() >= n_cells:
            raise ValueError(f"{id_col}: values must be in [0, {n_cells - 1}]")

        aligned: dict[str, NDArray[np.float32]] = {}
        for key, values in state.items():
            if key == id_col:
                continue
            src = np.asarray(values).reshape(-1)
            if src.size != n_cells:
                raise ValueError(f"{key}: expected {n_cells} rows, got {src.size}")
            dst = np.empty(n_cells, dtype=src.dtype)
            dst[cell_id] = src
            aligned[key] = dst
        state = aligned

    if "h" in state:
        h = _as_1d_float32("h", state["h"], n_cells)
    elif "wse" in state:
        wse = _as_1d_float32("wse", state["wse"], n_cells)
        h = wse - zb_local.astype(np.float32)
    else:
        raise ValueError("Initial state must provide 'h' or 'wse'")

    if "hu" in state and "hv" in state:
        hu = _as_1d_float32("hu", state["hu"], n_cells)
        hv = _as_1d_float32("hv", state["hv"], n_cells)
    elif "u" in state and "v" in state:
        u = _as_1d_float32("u", state["u"], n_cells)
        v = _as_1d_float32("v", state["v"], n_cells)
        hu = h * u
        hv = h * v
    else:
        raise ValueError("Initial state must provide (hu,hv) or (u,v)")

    n_mann = _as_1d_float32("n_mann", state["n_mann"], n_cells) if "n_mann" in state else None

    if order not in ("solver", "original"):
        raise ValueError("initial_state_order must be 'solver' or 'original'")

    if order == "original":
        h = h[perm]
        hu = hu[perm]
        hv = hv[perm]
        if n_mann is not None:
            n_mann = n_mann[perm]

    h = np.maximum(h, 0.0).astype(np.float32)
    wet = h > np.float32(dry_tol)
    hu = np.where(wet, hu, 0.0).astype(np.float32)
    hv = np.where(wet, hv, 0.0).astype(np.float32)

    if n_mann is not None:
        n_mann = np.maximum(n_mann.astype(np.float32), np.float32(1e-4))

    return h, hu, hv, n_mann


def _build_phase_source_rate(
    geom: MeshGeometry,
    phase: SimulationPhase,
    valid_cell: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Build per-cell source array [m^3/s] for one phase."""
    src = np.zeros(geom.N, dtype=np.float32)
    for source in phase.sources:
        mask = select_cells_within_radius(
            geom.centroid,
            source.center_xy,
            source.radius_m,
            valid_mask=valid_cell,
        )
        if not mask.any():
            continue
        cell_area = geom.area[mask]
        area_sum = float(cell_area.sum())
        if area_sum <= 0.0:
            continue
        src[mask] += np.float32(source.discharge_m3s) * (cell_area / area_sum)
    return src


def _compile_solver_shaders() -> dict[str, bytes]:
    """Compile all required shader programs, including source kernel."""
    spv = compile_all()
    spv["source"] = compile_glsl(SOURCE_GLSL, "source_kernel")
    return spv


class SWEWorkflow:
    """Reusable workflow facade for integration into existing applications.

    Typical usage:
    1) Create instance once at application startup.
    2) Call prepare() to load mesh, compile shaders, allocate GPU resources.
    3) Call run(phases) per simulation request.
    """

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        self.geom: MeshGeometry | None = None
        self.perm: NDArray[np.int32] | None = None
        self.verts: NDArray[np.float32] | None = None
        self.faces_flat: NDArray[np.int32] | None = None
        self.face_offsets: NDArray[np.int32] | None = None
        self.zb_from_file: NDArray[np.float32] | None = None
        self.solver: (
            SWESolverBaseline
            | SWESolverBatchedSubmit
            | SWESolverDeviceCfl
            | SWESolverAsyncSyncWindow
            | SWESolverFixedDtBatch
            | SWESolverGpuResidentBatch
            | None
        ) = None

    def prepare(self) -> None:
        """Prepare geometry, compile shaders, and allocate solver resources."""
        progress = self.config.progress

        def _log(msg: str) -> None:
            if progress:
                sys.stdout.write(f"[prepare] {msg}\n")
                sys.stdout.flush()

        geom: MeshGeometry | None = None
        perm: NDArray[np.int32] | None = None
        cache_dir = self.config.geometry_cache_dir
        cache_key: str = ""
        if cache_dir is not None:
            cache_key = geometry_cache_key(
                self.config.mesh_source, use_hilbert_reorder=self.config.use_hilbert_reorder
            )
            cached = load_geometry_cache(cache_dir, cache_key)
            if cached is not None:
                geom, perm = cached
                _log(f"geometry cache hit ({cache_dir}/geometry_{cache_key}.npz)")
            else:
                _log(f"geometry cache miss ({cache_dir}, key={cache_key[:12]}...) — building")

        verts: NDArray[np.float32] | None = None
        faces_flat: NDArray[np.int32] | None = None
        face_offsets: NDArray[np.int32] | None = None
        zb_from_file: NDArray[np.float32] | None = None

        if geom is None:
            # Cache miss (or caching disabled) — the only path that needs the
            # mesh file itself; a cache hit skips both the file read and the
            # Python-loop-heavy build_geometry()/hilbert_reorder() entirely.
            _t_stage = time.perf_counter()
            verts, faces_flat, face_offsets, zb_from_file = load_mesh_file(self.config.mesh_source)
            _log(
                f"mesh file loaded in {time.perf_counter() - _t_stage:.1f}s "
                f"({self.config.mesh_source})"
            )
            geom = build_geometry(
                verts,
                faces_flat,
                face_offsets=face_offsets,
                zb_from_file=zb_from_file,
                progress=progress,
            )

            perm = np.arange(geom.N, dtype=np.int32)
            if self.config.use_hilbert_reorder:
                geom, perm = hilbert_reorder(geom, verbose=progress)

            if cache_dir is not None:
                _t_stage = time.perf_counter()
                cache_path = save_geometry_cache(cache_dir, cache_key, geom, perm)
                _log(
                    f"geometry cache saved in {time.perf_counter() - _t_stage:.1f}s ({cache_path})"
                )

        if perm is None:
            raise RuntimeError("perm was not set alongside geom — internal prepare() bug")

        spv = _compile_solver_shaders()

        h0 = np.zeros(geom.N, dtype=np.float32)
        hu0 = np.zeros(geom.N, dtype=np.float32)
        hv0 = np.zeros(geom.N, dtype=np.float32)
        n0 = np.full(geom.N, self.config.manning_n, dtype=np.float32)

        loaded_state = _load_initial_state(
            self.config.initial_state_source,
            geom.N,
            geom.zb,
            order=self.config.initial_state_order,
            id_col=self.config.initial_state_id_col,
            perm=perm,
            dry_tol=self.config.dry_tol,
        )
        if loaded_state is not None:
            h0, hu0, hv0, n_loaded = loaded_state
            if n_loaded is not None:
                n0 = n_loaded

        from .swe_gpu_async_sync_window import SWESolverAsyncSyncWindow
        from .swe_gpu_baseline import SWESolverBaseline
        from .swe_gpu_batched_submit import SWESolverBatchedSubmit
        from .swe_gpu_device_cfl import SWESolverDeviceCfl
        from .swe_gpu_fixed_dt_batch import SWESolverFixedDtBatch
        from .swe_gpu_fixed_dt_batch_barrier import SWESolverFixedDtBatchBarrier
        from .swe_gpu_gpu_resident_batch import SWESolverGpuResidentBatch

        solver_cls_by_impl = {
            "baseline": SWESolverBaseline,
            "batched_submit": SWESolverBatchedSubmit,
            "device_cfl": SWESolverDeviceCfl,
            "async_sync_window": SWESolverAsyncSyncWindow,
            "fixed_dt_batch": SWESolverFixedDtBatch,
            "fixed_dt_batch_barrier": SWESolverFixedDtBatchBarrier,
            "gpu_resident_batch": SWESolverGpuResidentBatch,
        }
        solver_cls = solver_cls_by_impl[self.config.solver_impl]
        solver = solver_cls(
            geom,
            spv,
            h0,
            hu0,
            hv0,
            n0,
            g=self.config.gravity,
            dry_tol=self.config.dry_tol,
            cfl=self.config.cfl,
            workgroup_size=self.config.workgroup_size,
        )

        self.geom = geom
        self.perm = perm
        self.verts = verts
        self.faces_flat = faces_flat
        self.face_offsets = face_offsets
        self.zb_from_file = zb_from_file
        self.solver = solver

    def run(self, phases: Sequence[SimulationPhase]) -> WorkflowResult:
        """Execute one or more discharge phases and return simulation outputs."""
        if self.geom is None:
            raise RuntimeError("Workflow not prepared. Call prepare() first.")
        if self.solver is None:
            raise RuntimeError("Workflow not prepared. Call prepare() first.")

        if not phases:
            raise ValueError("At least one phase is required.")

        area = self.geom.area
        valid_cell = area > self.config.area_tol

        snapshots: list[NDArray[np.float32]] = []
        snap_times: list[float] = []
        t_current = 0.0

        t_wall_start = time.perf_counter()

        for phase_index, phase in enumerate(phases):
            if phase.duration_s <= 0.0:
                raise ValueError(f"Phase {phase_index} has non-positive duration.")

            t_end = t_current + phase.duration_s
            src = _build_phase_source_rate(self.geom, phase, valid_cell)

            solver_kwargs: dict[str, object] = {}
            if self.config.solver_impl == "fixed_dt_batch_barrier":
                solver_kwargs["output_every_steps"] = self.config.output_every_steps

            phase_snapshots, phase_times = self.solver.run(
                t_end=t_end,
                output_interval_s=self.config.output_interval_s,
                dt_max=self.config.dt_max,
                dt_init=self.config.dt_init,
                progress=self.config.progress,
                source_rate=src,
                resume=phase_index > 0,
                t_start=t_current,
                cfl_interval=self.config.cfl_interval,
                **solver_kwargs,
            )

            snapshots.extend(phase_snapshots)
            snap_times.extend(phase_times)
            t_current = t_end

        wall_seconds = time.perf_counter() - t_wall_start
        h_final = self.solver.download_h()
        volume_final_m3 = float((h_final * area).sum())
        volume_injected_m3 = float(
            sum(
                phase.duration_s * sum(source.discharge_m3s for source in phase.sources)
                for phase in phases
            )
        )

        download_hu = getattr(self.solver, "download_hu", None)
        download_hv = getattr(self.solver, "download_hv", None)
        hu_final: NDArray[np.float32] | None = (
            cast("NDArray[np.float32]", download_hu()) if callable(download_hu) else None
        )
        hv_final: NDArray[np.float32] | None = (
            cast("NDArray[np.float32]", download_hv()) if callable(download_hv) else None
        )

        return WorkflowResult(
            snapshots=snapshots,
            snap_times=snap_times,
            h_final=h_final,
            volume_final_m3=volume_final_m3,
            volume_injected_m3=volume_injected_m3,
            wall_seconds=wall_seconds,
            steps_total=int(getattr(self.solver, "steps_total", 0)),
            hu_final=hu_final,
            hv_final=hv_final,
        )



def build_example_phases() -> list[SimulationPhase]:
    """Return a ready-to-run phase schedule equivalent to the notebook sample."""
    return [
        SimulationPhase(
            duration_s=10000.0,
            sources=[
                PointSource(5.0, (349304.72, 266737.27), 1.0),
                PointSource(5.0, (350264.247, 266736.603), 1.0),
            ],
        ),
        SimulationPhase(
            duration_s=5400.0,
            sources=[
                PointSource(100.0, (349304.72, 266737.27), 1.0),
                PointSource(20.0, (350264.247, 266736.603), 1.0),
            ],
        ),
        SimulationPhase(
            duration_s=5400.0,
            sources=[
                PointSource(250.0, (349304.72, 266737.27), 1.0),
                PointSource(20.0, (350264.247, 266736.603), 1.0),
            ],
        ),
    ]


def run_example(mesh_source: str) -> WorkflowResult:
    """Small convenience entrypoint for smoke-testing integration."""
    config = WorkflowConfig(
        mesh_source=mesh_source,
        manning_n=0.035,
        output_interval_s=300.0,
        dt_max=0.05,
        cfl_interval=1,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    phases = build_example_phases()
    return workflow.run(phases)


if __name__ == "__main__":
    # Example CLI-style invocation:
    #   python solver_workflow.py ../../mesh/mesh_triangles_z2.shp
    import sys

    if len(sys.argv) != REQUIRED_CLI_ARGS:
        raise SystemExit("Usage: python solver_workflow.py <mesh_source>")

    result = run_example(sys.argv[1])

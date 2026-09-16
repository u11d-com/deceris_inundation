"""Cross-module numerical + structural tuning constants and small helpers.

Pure-data module: imports only stdlib at runtime. Shared constants prevent
drift between geometry, workflow, Vulkan solver, and benchmarks.
"""

from __future__ import annotations

import math

# ── Geometry (swe_geometry.py) ────────────────────────────────────────────────
POLYGON_AREA_EPSILON = 1e-30
POLYGON_CENTROID_EPSILON = 1e-20
EDGE_LENGTH_EPSILON = 1e-30
FACE_NDIM = 2

# ── Mesh geometry ─────────────────────────────────────────────────────────────
MIN_POLYGON_VERTICES = 3

# ── Physical and numerical solver defaults ────────────────────────────────────
GRAVITY_G = 9.81
DRY_TOL_DEFAULT = 1e-4
CFL_DEFAULT = 0.45
REGIME_DRY = 2

# ── Vulkan solver ─────────────────────────────────────────────────────────────
CFL_EPSILON_MIN = 1e-10
CFL_SANITY_MAX = 1e10
SIMULATION_TIME_EPSILON = 1e-12


# ── Workflow (workflow.py) ────────────────────────────────────────────────────
MIN_NDIM_NPY_INPUT = 2
NPY_COLS_HUV = 3
NPY_COLS_WITH_MANNING = 4
REQUIRED_CLI_ARGS = 2

# ── Benchmark validation ──────────────────────────────────────────────────────
MIN_VALID_CELLS = 2
MIN_SNAPSHOTS = 10
FINAL_TIME_TOLERANCE_S = 5.0
MIN_DEPTH_THRESHOLD_M = 0.05
WET_CELL_THRESHOLD = 1e-6
MIN_WET_CELLS = 100
MIN_VOLUME_RATIO = 0.01
MIN_REPEATS_FOR_DETERMINISM = 2


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_workgroups(
    n: int, e: int, work_group_size: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (edge_workgroup, cell_workgroup) for Vulkan dispatch."""
    return (
        (math.ceil(e / work_group_size), 1, 1),
        (math.ceil(n / work_group_size), 1, 1),
    )


def compute_eta_seconds(wall_elapsed: float, t_sim: float, t_offset: float, t_end: float) -> float:
    """ETA at current progress; returns inf when t_sim hasn't advanced past t_offset."""
    if t_sim - t_offset <= SIMULATION_TIME_EPSILON:
        return float("inf")
    return (wall_elapsed / max(t_sim - t_offset, SIMULATION_TIME_EPSILON)) * max(t_end - t_sim, 0.0)

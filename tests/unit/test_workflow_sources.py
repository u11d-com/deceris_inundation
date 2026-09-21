"""Tests for geometry-aware point-source cell selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from inundation.mesh.geometry import build_geometry
from inundation.workflow import (
    PointSource,
    SimulationPhase,
    _build_phase_source_rate,
    select_cells_within_radius,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Deliberate white-box test of workflow._build_phase_source_rate.
# pyright: reportPrivateUsage=false


def _two_cell_mesh() -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    verts = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int32)
    return verts, faces


def test_tiny_radius_selects_containing_cell() -> None:
    verts, faces = _two_cell_mesh()
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32))

    mask = select_cells_within_radius(
        geom.centroid,
        (0.1, 0.1),
        0.01,
        cell_vertices=geom.cell_vertices,
        cell_vertex_ptr=geom.cell_vertex_ptr,
        cell_bbox=geom.cell_bbox,
    )

    np.testing.assert_array_equal(mask, [True, False])


def test_tiny_radius_inside_cell_does_not_expand_selection() -> None:
    verts, faces = _two_cell_mesh()
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32))

    mask = select_cells_within_radius(
        geom.centroid,
        (0.99, 0.5),
        0.02,
        cell_vertices=geom.cell_vertices,
        cell_vertex_ptr=geom.cell_vertex_ptr,
        cell_bbox=geom.cell_bbox,
    )

    np.testing.assert_array_equal(mask, [True, False])


def test_tiny_radius_source_preserves_discharge() -> None:
    verts, faces = _two_cell_mesh()
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32))
    phase = SimulationPhase(
        duration_s=1.0,
        sources=[PointSource(12.0, (0.1, 0.1), 0.01)],
    )

    source_rate = _build_phase_source_rate(geom, phase, geom.area > 0.0)

    np.testing.assert_allclose(source_rate, [12.0, 0.0])
    assert float(source_rate.sum()) == 12.0

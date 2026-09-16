"""Tests for packed signed CSR adjacency.

``cell_edge_idx_signed`` packs ``(edge_idx << 1) | is_right`` so gather
operations can determine incident-edge orientation without another lookup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from inundation.mesh.geometry import MeshGeometry, build_geometry, hilbert_reorder

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _make_triangulated_grid_mesh(
    nx: int = 4, ny: int = 4
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Build a small nx*ny vertex grid, triangulated into two triangles per cell."""
    xs, ys = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    verts = np.stack([xs.ravel(), ys.ravel()], axis=1)

    def vid(i: int, j: int) -> int:
        return j * nx + i

    faces: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, np.array(faces, dtype=np.int32)


def _assert_signed_csr_consistent(geom: MeshGeometry, label: str) -> None:
    """Assert the signed CSR packing is self-consistent and covers every edge exactly once."""
    total_slots = int(geom.cell_edge_ptr[geom.N])
    seen_left = np.zeros(geom.E, dtype=bool)
    seen_right = np.zeros(geom.E, dtype=bool)

    for ci in range(geom.N):
        s, e = int(geom.cell_edge_ptr[ci]), int(geom.cell_edge_ptr[ci + 1])
        for slot in range(s, e):
            eid = int(geom.cell_edge_idx[slot])
            is_right = int(geom.cell_edge_side[slot])
            signed = int(geom.cell_edge_idx_signed[slot])

            assert signed == (eid << 1) | is_right, f"{label}: pack mismatch at slot {slot}"
            assert is_right in (0, 1), f"{label}: side bit not 0/1 at slot {slot}"

            if is_right == 0:
                assert geom.edge_cellL[eid] == ci, (
                    f"{label}: cell {ci} claims L-ownership of edge {eid} "
                    f"but edge_cellL={geom.edge_cellL[eid]}"
                )
                assert not seen_left[eid], f"{label}: edge {eid} has duplicate L slot"
                seen_left[eid] = True
            else:
                assert geom.edge_cellR[eid] == ci, (
                    f"{label}: cell {ci} claims R-ownership of edge {eid} "
                    f"but edge_cellR={geom.edge_cellR[eid]}"
                )
                assert not seen_right[eid], f"{label}: edge {eid} has duplicate R slot"
                seen_right[eid] = True

    # Every edge must have exactly one L-owner slot (its left cell always exists).
    assert seen_left.all(), f"{label}: not every edge has an L-owner CSR slot"

    # Every interior edge must have exactly one R-owner slot; boundary edges (cellR == -1)
    # must have none.
    interior = geom.edge_cellR != -1
    assert seen_right[interior].all(), f"{label}: not every interior edge has an R-owner slot"
    assert not seen_right[~interior].any(), f"{label}: boundary edge unexpectedly has R slot"

    assert total_slots == int(seen_left.sum() + seen_right.sum())


def test_signed_csr_consistent_after_build_geometry() -> None:
    verts, faces = _make_triangulated_grid_mesh()
    geom = build_geometry(verts, faces)
    _assert_signed_csr_consistent(geom, "build_geometry")


def test_signed_csr_consistent_after_hilbert_reorder() -> None:
    verts, faces = _make_triangulated_grid_mesh()
    geom = build_geometry(verts, faces)
    reordered, _perm = hilbert_reorder(geom, verbose=False)
    _assert_signed_csr_consistent(reordered, "hilbert_reorder")


def test_signed_csr_shapes_match_cell_edge_idx() -> None:
    verts, faces = _make_triangulated_grid_mesh()
    geom = build_geometry(verts, faces)
    assert geom.cell_edge_side.shape == geom.cell_edge_idx.shape
    assert geom.cell_edge_idx_signed.shape == geom.cell_edge_idx.shape
    assert geom.cell_edge_side.dtype == np.int32
    assert geom.cell_edge_idx_signed.dtype == np.int32

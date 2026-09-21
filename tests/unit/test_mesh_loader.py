"""Tests for required mesh bed elevations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.mesh.loader import load_mesh_file

if TYPE_CHECKING:
    from pathlib import Path


def test_obj_requires_vertex_elevations(tmp_path: Path) -> None:
    path = tmp_path / "missing-z.obj"
    path.write_text("v 0 0\nv 1 0\nv 0 1\nf 1 2 3\n")

    with pytest.raises(ValueError, match="required Z elevation"):
        load_mesh_file(str(path))


def test_obj_vertex_elevations_become_cell_elevations(tmp_path: Path) -> None:
    path = tmp_path / "with-z.obj"
    path.write_text("v 0 0 1\nv 1 0 2\nv 0 1 3\nf 1 2 3\n")

    _verts, _faces, _offsets, zb = load_mesh_file(str(path))

    np.testing.assert_array_equal(zb, np.array([2.0], dtype=np.float32))

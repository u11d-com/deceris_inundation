"""Tests for mesh geometry cache round-trip (ported from swe_geometry_cache._self_check)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from inundation.mesh.cache import (
    _GEOMETRY_ARRAY_FIELDS,
    _GEOMETRY_SCALAR_FIELDS,
    load_geometry_cache,
    save_geometry_cache,
)
from inundation.mesh.geometry import build_geometry, hilbert_reorder

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _small_quad_mesh() -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """2x2 grid of unit quads -> 4 cells, shared vertices."""
    verts = np.array(
        [[x, y] for y in range(3) for x in range(3)],
        dtype=np.float32,
    )

    def quad(ix: int, iy: int) -> list[int]:
        return [
            iy * 3 + ix,
            iy * 3 + ix + 1,
            (iy + 1) * 3 + ix + 1,
            (iy + 1) * 3 + ix,
        ]

    faces = np.array([quad(0, 0), quad(1, 0), quad(0, 1), quad(1, 1)], dtype=np.int32)
    return verts, faces


def test_geometry_cache_roundtrip() -> None:
    """Cache a geometry, reload it, and diff all arrays/scalars."""
    verts, faces = _small_quad_mesh()
    geom = build_geometry(verts, faces)
    geom, perm = hilbert_reorder(geom, verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "geom-cache"
        cache_key = "test-key"
        save_geometry_cache(cache_dir, cache_key, geom, perm)
        loaded = load_geometry_cache(cache_dir, cache_key)
        assert loaded is not None, "cache load returned None right after save"
        loaded_geom, loaded_perm = loaded

        for name in _GEOMETRY_ARRAY_FIELDS:
            np.testing.assert_array_equal(
                getattr(loaded_geom, name), getattr(geom, name), err_msg=f"field mismatch: {name}"
            )
        for name in _GEOMETRY_SCALAR_FIELDS:
            assert getattr(loaded_geom, name) == getattr(geom, name), f"scalar mismatch: {name}"
        np.testing.assert_array_equal(loaded_perm, perm)


def test_geometry_cache_key_mismatch_misses() -> None:
    """A cache-key mismatch must return None."""
    verts, faces = _small_quad_mesh()
    geom = build_geometry(verts, faces)
    geom, perm = hilbert_reorder(geom, verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "geom-cache"
        save_geometry_cache(cache_dir, "correct-key", geom, perm)
        assert load_geometry_cache(cache_dir, "wrong-key") is None

"""Disk cache for preprocessed ``MeshGeometry``.

``build_geometry`` and ``hilbert_reorder`` are Python-loop heavy. The cache
avoids rebuilding identical geometry for repeated benchmark runs.

Cache keys include mesh bytes, the cache format version, and reorder mode.
Run ``python -m inundation.mesh.cache`` for the standalone self-check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry import MeshGeometry

if TYPE_CHECKING:
    import os

    from numpy.typing import NDArray

# Bump whenever build_geometry()/hilbert_reorder() output arrays change
# shape, dtype, or semantics, so old on-disk caches are rejected instead of
# silently served with stale/incompatible contents.
GEOMETRY_CACHE_VERSION = 3

# Field order matters only for readability; np.savez uses keyword storage.
_GEOMETRY_ARRAY_FIELDS = (
    "centroid",
    "area",
    "zb",
    "cell_vertices",
    "cell_vertex_ptr",
    "cell_bbox",
    "edge_len",
    "edge_nx",
    "edge_ny",
    "edge_cellL",
    "edge_cellR",
    "edge_sideL",
    "edge_sideR",
    "cell_edges",
    "cell_neighbors",
    "cell_edge_ptr",
    "cell_edge_idx",
    "cell_nbr_idx",
    "cell_edge_side",
    "cell_edge_idx_signed",
)
_GEOMETRY_SCALAR_FIELDS = ("N", "E", "V", "max_degree")


def geometry_cache_key(mesh_path: str | Path, *, use_hilbert_reorder: bool) -> str:
    """Return a stable cache key derived from mesh file content and configuration."""
    path = Path(mesh_path)
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(f"v{GEOMETRY_CACHE_VERSION}|hilbert={use_hilbert_reorder}|".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def geometry_cache_path(cache_dir: str | os.PathLike[str], cache_key: str) -> Path:
    """Return the on-disk path for a given cache key under ``cache_dir``."""
    return Path(cache_dir) / f"geometry_{cache_key}.npz"


def save_geometry_cache(
    cache_dir: str | os.PathLike[str],
    cache_key: str,
    geom: MeshGeometry,
    perm: NDArray[np.int32],
) -> Path:
    """Persist ``geom`` + ``perm`` to ``<cache_dir>/geometry_<cache_key>.npz``."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = geometry_cache_path(out_dir, cache_key)

    arrays = {name: getattr(geom, name) for name in _GEOMETRY_ARRAY_FIELDS}
    scalars = {name: np.int64(getattr(geom, name)) for name in _GEOMETRY_SCALAR_FIELDS}
    # Write to a temp file then rename so interruption cannot leave a truncated
    # cache. Passing an open handle avoids np.savez appending another suffix.
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, perm=perm, **arrays, **scalars)
    tmp_path.replace(path)
    return path


def load_geometry_cache(
    cache_dir: str | os.PathLike[str], cache_key: str
) -> tuple[MeshGeometry, NDArray[np.int32]] | None:
    """Load a previously cached ``(MeshGeometry, perm)`` pair, or ``None`` on any miss.

    Any structural problem (missing file, missing field, unreadable archive)
    is treated as a cache miss — always falls back to rebuilding from scratch
    rather than raising, since the cache is a pure performance optimization.
    """
    path = geometry_cache_path(cache_dir, cache_key)
    if not path.exists():
        return None

    try:
        with np.load(path) as npz:
            missing = [
                name
                for name in (*_GEOMETRY_ARRAY_FIELDS, *_GEOMETRY_SCALAR_FIELDS, "perm")
                if name not in npz
            ]
            if missing:
                return None
            arrays: dict[str, NDArray[Any]] = {name: npz[name] for name in _GEOMETRY_ARRAY_FIELDS}
            scalars: dict[str, int] = {name: int(npz[name]) for name in _GEOMETRY_SCALAR_FIELDS}
            perm = npz["perm"]
    except (OSError, ValueError, KeyError, EOFError):
        return None

    geom = MeshGeometry(
        **arrays,
        N=scalars["N"],
        E=scalars["E"],
        V=scalars["V"],
        max_degree=scalars["max_degree"],
    )
    return geom, perm


def _self_check() -> None:
    """Build a tiny synthetic quad mesh, cache it, reload it, and diff arrays."""
    import tempfile

    from .geometry import build_geometry, hilbert_reorder

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 2x2 grid of unit quads -> 4 cells, shared vertices. Built directly
        # from arrays (not via a mesh file / load_mesh_file) since this is a
        # synthetic geometry-cache round-trip check, not a mesh-loader check.
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
        zb = np.zeros(len(faces), dtype=np.float32)

        geom = build_geometry(verts, faces, zb_from_file=zb)
        geom, perm = hilbert_reorder(geom, verbose=False)

        cache_dir = tmp_dir / "geom-cache"
        cache_key = "self-check-key"
        save_geometry_cache(cache_dir, cache_key, geom, perm)
        loaded = load_geometry_cache(cache_dir, cache_key)
        if loaded is None:
            raise AssertionError("cache load returned None right after save")
        loaded_geom, loaded_perm = loaded

        for name in _GEOMETRY_ARRAY_FIELDS:
            np.testing.assert_array_equal(
                getattr(loaded_geom, name), getattr(geom, name), err_msg=f"field mismatch: {name}"
            )
        for name in _GEOMETRY_SCALAR_FIELDS:
            if getattr(loaded_geom, name) != getattr(geom, name):
                raise AssertionError(f"scalar mismatch: {name}")
        np.testing.assert_array_equal(loaded_perm, perm)

        # A cache-key mismatch (e.g. different mesh content or version) must miss.
        if load_geometry_cache(cache_dir, "wrong-key") is not None:
            raise AssertionError("cache lookup with a mismatched key unexpectedly hit")

        print("geometry cache self-check: OK")


if __name__ == "__main__":
    _self_check()

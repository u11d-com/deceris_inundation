"""Tests for mesh geometry cache round-trip (ported from swe_geometry_cache._self_check)."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from inundation.mesh.cache import (
    _GEOMETRY_ARRAY_FIELDS,
    _GEOMETRY_SCALAR_FIELDS,
    GEOMETRY_CACHE_VERSION,
    GeometryCacheMeta,
    _write_geometry_npz,
    build_geometry_cache,
    load_geometry_cache,
    load_geometry_cache_artifact,
    save_geometry_cache,
)
from inundation.mesh.geometry import build_geometry, hilbert_reorder

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry


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


def test_bed_elevations_are_required() -> None:
    """Geometry construction rejects meshes without bed elevations."""
    verts, faces = _small_quad_mesh()
    with pytest.raises(TypeError, match="zb_from_file"):
        build_geometry(verts, faces)  # pyright: ignore[reportCallIssue]


def test_geometry_defaults_are_explicitly_flat() -> None:
    """Explicit zero elevations produce a flat bed."""
    verts, faces = _small_quad_mesh()
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32))
    assert np.all(geom.zb == 0.0)


def _meta(*, version: int = GEOMETRY_CACHE_VERSION, reordered: bool = True) -> GeometryCacheMeta:
    return GeometryCacheMeta(
        version=version, mesh_source="synthetic", use_hilbert_reorder=reordered
    )


def test_geometry_cache_roundtrip() -> None:
    """Cache a geometry, including optional mesh roughness, and reload it."""
    verts, faces = _small_quad_mesh()
    manning_n = np.asarray([0.02, 0.025, 0.03, 0.035], dtype=np.float32)
    geom = build_geometry(
        verts,
        faces,
        zb_from_file=np.zeros(len(faces), dtype=np.float32),
        manning_n_from_file=manning_n,
    )
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
    geom = build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32))
    geom, perm = hilbert_reorder(geom, verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "geom-cache"
        save_geometry_cache(cache_dir, "correct-key", geom, perm)
        assert load_geometry_cache(cache_dir, "wrong-key") is None


def test_geometry_cache_artifact_roundtrip(tmp_path: Path) -> None:
    """An explicit artifact round-trips geometry, perm and provenance."""
    verts, faces = _small_quad_mesh()
    manning_n = np.asarray([0.02, 0.025, 0.03, 0.035], dtype=np.float32)
    geom = build_geometry(
        verts,
        faces,
        zb_from_file=np.zeros(len(faces), dtype=np.float32),
        manning_n_from_file=manning_n,
    )
    geom, perm = hilbert_reorder(geom, verbose=False)

    artifact = tmp_path / "nested" / "geometry.npz"  # parent dirs are created on write
    _write_geometry_npz(artifact, geom, perm, _meta())
    loaded_geom, loaded_perm, loaded_meta = load_geometry_cache_artifact(artifact)

    assert loaded_meta == _meta()
    for name in _GEOMETRY_ARRAY_FIELDS:
        np.testing.assert_array_equal(
            getattr(loaded_geom, name), getattr(geom, name), err_msg=f"field mismatch: {name}"
        )
    np.testing.assert_array_equal(loaded_perm, perm)


def test_geometry_cache_artifact_roundtrips_missing_roughness(tmp_path: Path) -> None:
    """A mesh without a roughness column reloads with manning_n None, not empty."""
    verts, faces = _small_quad_mesh()
    geom, perm = hilbert_reorder(
        build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32)),
        verbose=False,
    )

    artifact = tmp_path / "geometry.npz"
    _write_geometry_npz(artifact, geom, perm, _meta(reordered=False))
    loaded_geom, _, _ = load_geometry_cache_artifact(artifact)

    assert loaded_geom.manning_n is None


def test_geometry_cache_artifact_missing_file_raises(tmp_path: Path) -> None:
    """A missing artifact is not a rebuild trigger — it raises."""
    with pytest.raises(FileNotFoundError):
        load_geometry_cache_artifact(tmp_path / "absent.npz")


def test_geometry_cache_artifact_stale_version_rejected(tmp_path: Path) -> None:
    """An artifact from another cache version cannot be silently consumed."""
    verts, faces = _small_quad_mesh()
    geom, perm = hilbert_reorder(
        build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32)),
        verbose=False,
    )

    stale = tmp_path / "stale.npz"
    _write_geometry_npz(stale, geom, perm, _meta(version=GEOMETRY_CACHE_VERSION + 1))
    with pytest.raises(ValueError, match="cache version"):
        load_geometry_cache_artifact(stale)


def test_build_geometry_cache_from_mesh_file(tmp_path: Path) -> None:
    """build_geometry_cache() round-trips a mesh file, roughness included."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shapely = pytest.importorskip("shapely")

    polygons = [
        shapely.Polygon([(2, 0), (3, 0), (2, 1)]),
        shapely.Polygon([(0, 0), (1, 0), (0, 1)]),
    ]
    source_roughness = np.asarray([0.018, 0.052], dtype=np.float32)
    source_bed = np.asarray([10.0, 20.0], dtype=np.float32)
    mesh_path = tmp_path / "mesh.parquet"
    pq.write_table(
        pa.table(
            {
                "geometry": pa.array(shapely.to_wkb(polygons)),
                "z_mean": pa.array(source_bed),
                "manning_n": pa.array(source_roughness),
            }
        ),
        mesh_path,
    )

    reordered = build_geometry_cache(mesh_path, tmp_path / "reordered.npz")
    geom, perm, meta = load_geometry_cache_artifact(reordered)
    assert meta.mesh_source == str(mesh_path)
    assert meta.use_hilbert_reorder is True
    assert geom.N == 2
    assert geom.manning_n is not None
    np.testing.assert_array_equal(geom.manning_n, source_roughness[perm])
    np.testing.assert_array_equal(geom.zb, source_bed[perm])

    plain = build_geometry_cache(mesh_path, tmp_path / "plain.npz", use_hilbert_reorder=False)
    plain_geom, plain_perm, plain_meta = load_geometry_cache_artifact(plain)
    assert plain_meta.use_hilbert_reorder is False
    assert plain_geom.manning_n is not None
    np.testing.assert_array_equal(plain_perm, np.arange(2, dtype=np.int32))
    np.testing.assert_array_equal(plain_geom.manning_n, source_roughness)


def _reordered_tiny_geometry() -> tuple[MeshGeometry, NDArray[np.int32]]:
    verts, faces = _small_quad_mesh()
    return hilbert_reorder(
        build_geometry(verts, faces, zb_from_file=np.zeros(len(faces), dtype=np.float32)),
        verbose=False,
    )


def test_directory_cache_treats_truncated_file_as_miss(tmp_path: Path) -> None:
    """A torn directory-cache file must be a miss, not a crash (rebuild path)."""
    geom, perm = _reordered_tiny_geometry()
    cache_dir = tmp_path / "geom-cache"
    path = save_geometry_cache(cache_dir, "test-key", geom, perm)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    assert load_geometry_cache(cache_dir, "test-key") is None


def test_artifact_truncated_archive_raises_value_error(tmp_path: Path) -> None:
    """A truncated artifact raises ValueError, not zipfile.BadZipFile."""
    geom, perm = _reordered_tiny_geometry()
    artifact = tmp_path / "geometry.npz"
    _write_geometry_npz(artifact, geom, perm, _meta())
    artifact.write_bytes(artifact.read_bytes()[: artifact.stat().st_size // 2])

    with pytest.raises(ValueError, match="unreadable"):
        load_geometry_cache_artifact(artifact)


def test_artifact_missing_fields_raises(tmp_path: Path) -> None:
    """An artifact missing required arrays is rejected, not partially read."""
    artifact = tmp_path / "incomplete.npz"
    np.savez(artifact, perm=np.arange(2, dtype=np.int32))

    with pytest.raises(ValueError, match="missing fields"):
        load_geometry_cache_artifact(artifact)


def test_artifact_bad_metadata_raises(tmp_path: Path) -> None:
    """An undecodable provenance blob is rejected."""
    geom, perm = _reordered_tiny_geometry()
    good = tmp_path / "good.npz"
    _write_geometry_npz(good, geom, perm, _meta())

    payload = dict(np.load(good))
    payload["meta"] = np.asarray("not json{")
    bad = tmp_path / "bad-meta.npz"
    np.savez(bad, **payload)

    with pytest.raises(ValueError, match="invalid geometry cache metadata"):
        load_geometry_cache_artifact(bad)

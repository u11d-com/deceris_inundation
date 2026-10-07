"""CPU-only tests for mesh-sourced Manning roughness."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ._fixtures import CapturingSolver, forbid_mesh_reads, stub_solver_backends

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from inundation.mesh.geometry import MeshGeometry


def test_geoparquet_roughness_stays_aligned_after_hilbert_reorder(tmp_path: Path) -> None:
    """GeoParquet roughness follows its source cell into solver order."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shapely = pytest.importorskip("shapely")
    pytest.importorskip("hilbertcurve")

    from inundation.mesh.geometry import build_geometry, hilbert_reorder
    from inundation.mesh.loader import load_mesh_file

    polygons = [
        shapely.Polygon([(2, 0), (3, 0), (2, 1)]),
        shapely.Polygon([(0, 0), (1, 0), (0, 1)]),
    ]
    source_roughness = np.asarray([0.018, 0.052], dtype=np.float32)
    mesh_path = tmp_path / "mesh.parquet"
    pq.write_table(
        pa.table(
            {
                "geometry": pa.array(shapely.to_wkb(polygons)),
                "z_mean": pa.array([10.0, 20.0]),
                "manning_n": pa.array(source_roughness),
            }
        ),
        mesh_path,
    )

    verts, faces, face_offsets, zb, manning_n = load_mesh_file(str(mesh_path))

    np.testing.assert_array_equal(zb, np.asarray([10.0, 20.0], dtype=np.float32))
    assert manning_n is not None
    np.testing.assert_array_equal(manning_n, source_roughness)
    reordered, perm = hilbert_reorder(
        build_geometry(
            verts,
            faces,
            face_offsets=face_offsets,
            zb_from_file=zb,
            manning_n_from_file=manning_n,
        ),
        verbose=False,
    )
    assert reordered.manning_n is not None
    np.testing.assert_array_equal(reordered.manning_n, source_roughness[perm])


def test_workflow_cache_roughness_and_global_fallback_without_mesh_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache-hit preparation uses cached mesh n, else global n; state n wins."""
    from inundation import workflow as workflow_module
    from inundation.mesh.cache import geometry_cache_key, save_geometry_cache
    from inundation.mesh.geometry import build_geometry

    stub_solver_backends(monkeypatch)
    forbid_mesh_reads(monkeypatch)

    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("# cache key source only\n")
    cache_dir = tmp_path / "geometry-cache"
    cache_key = geometry_cache_key(str(mesh_path), use_hilbert_reorder=False)
    verts = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    flat_zb = np.zeros(len(faces), dtype=np.float32)

    def prepare(
        geom: MeshGeometry,
        *,
        initial_state_source: dict[str, object] | None = None,
    ) -> NDArray[np.float32]:
        save_geometry_cache(cache_dir, cache_key, geom, np.arange(geom.N, dtype=np.int32))
        workflow = workflow_module.SWEWorkflow(
            workflow_module.WorkflowConfig(
                mesh_source=str(mesh_path),
                manning_n=0.06,
                output_interval_s=1.0,
                dt_max=0.1,
                cfl_interval=1,
                progress=False,
                use_hilbert_reorder=False,
                geometry_cache_dir=cache_dir,
                initial_state_source=initial_state_source,
            )
        )
        workflow.prepare()
        return cast("CapturingSolver", workflow.solver).n0

    mesh_n = np.asarray([0.02, 0.04], dtype=np.float32)
    np.testing.assert_array_equal(
        prepare(build_geometry(verts, faces, zb_from_file=flat_zb, manning_n_from_file=mesh_n)),
        mesh_n,
    )
    np.testing.assert_array_equal(
        prepare(build_geometry(verts, faces, zb_from_file=flat_zb)),
        np.asarray([0.06, 0.06], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        prepare(
            build_geometry(verts, faces, zb_from_file=flat_zb, manning_n_from_file=mesh_n),
            initial_state_source={
                "h": [0.0, 0.0],
                "hu": [0.0, 0.0],
                "hv": [0.0, 0.0],
                "n_mann": [0.03, 0.05],
            },
        ),
        np.asarray([0.03, 0.05], dtype=np.float32),
    )


def test_parquet_without_roughness_column_loads_none(tmp_path: Path) -> None:
    """A mesh without a manning_n column yields None, not zeros."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shapely = pytest.importorskip("shapely")

    from inundation.mesh.loader import load_mesh_file

    mesh_path = tmp_path / "mesh.parquet"
    pq.write_table(
        pa.table(
            {
                "geometry": pa.array(shapely.to_wkb([shapely.Polygon([(0, 0), (1, 0), (0, 1)])])),
                "z_mean": pa.array([10.0]),
            }
        ),
        mesh_path,
    )

    _verts, _faces, _offsets, _zb, manning_n = load_mesh_file(str(mesh_path))
    assert manning_n is None


@pytest.mark.parametrize("bad_value", [0.0, -0.05, float("nan")])
def test_loader_rejects_non_positive_or_non_finite_roughness(
    tmp_path: Path, bad_value: float
) -> None:
    """Mesh roughness must be finite and positive — rejected, never clamped."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shapely = pytest.importorskip("shapely")

    from inundation.mesh.loader import load_mesh_file

    mesh_path = tmp_path / "mesh.parquet"
    pq.write_table(
        pa.table(
            {
                "geometry": pa.array(shapely.to_wkb([shapely.Polygon([(0, 0), (1, 0), (0, 1)])])),
                "z_mean": pa.array([10.0]),
                "manning_n": pa.array([bad_value]),
            }
        ),
        mesh_path,
    )

    with pytest.raises(ValueError, match="finite positive"):
        load_mesh_file(str(mesh_path))


def test_build_geometry_rejects_wrong_length_roughness() -> None:
    """Per-cell roughness must have exactly one value per cell."""
    from inundation.mesh.geometry import build_geometry

    verts = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="expected 2 values"):
        build_geometry(
            verts,
            faces,
            zb_from_file=np.zeros(len(faces), dtype=np.float32),
            manning_n_from_file=np.asarray([0.03], dtype=np.float32),
        )


def test_gpkg_roughness_survives_loading(tmp_path: Path) -> None:
    """The GeoPackage branch reads and validates its manning_n column."""
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")

    from inundation.mesh.loader import load_mesh_file

    source_roughness = np.asarray([0.018, 0.052], dtype=np.float32)
    source_bed = np.asarray([10.0, 20.0], dtype=np.float32)
    gdf = gpd.GeoDataFrame(
        {
            "z_mean": source_bed,
            "manning_n": source_roughness,
            "geometry": [
                shapely.Polygon([(2, 0), (3, 0), (2, 1)]),
                shapely.Polygon([(0, 0), (1, 0), (0, 1)]),
            ],
        }
    )
    mesh_path = tmp_path / "mesh.gpkg"
    gdf.to_file(mesh_path, driver="GPKG")

    _verts, _faces, _offsets, zb, manning_n = load_mesh_file(str(mesh_path))
    assert manning_n is not None
    np.testing.assert_array_equal(manning_n, source_roughness)
    np.testing.assert_array_equal(zb, source_bed)

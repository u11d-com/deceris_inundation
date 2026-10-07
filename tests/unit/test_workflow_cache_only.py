# pyright: reportPrivateUsage=false
"""Cache-only preparation: geometry from a prebuilt artifact, no mesh file."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from inundation.mesh.cache import (
    GEOMETRY_CACHE_VERSION,
    _write_geometry_npz,
    build_geometry_cache,
    load_geometry_cache_artifact,
)
from inundation.workflow import SWEWorkflow, WorkflowConfig

from ._fixtures import CapturingSolver, forbid_mesh_reads, stub_solver_backends

if TYPE_CHECKING:
    from pathlib import Path

_QUAD_OBJ = """\
v 0 0 1
v 1 0 2
v 2 0 3
v 0 1 4
v 1 1 5
v 2 1 6
f 1 2 5 4
f 2 3 6 5
"""


def _write_quad_obj(path: Path) -> None:
    """Write a dependency-free 2-cell quad mesh (bed from vertex Z, no manning_n)."""
    path.write_text(_QUAD_OBJ, encoding="ascii")


def _config(mesh_source: str | None, artifact: Path) -> WorkflowConfig:
    return WorkflowConfig(
        mesh_source=mesh_source,
        geometry_cache_source=artifact,
        manning_n=0.05,
        output_interval_s=1.0,
        dt_max=0.1,
        cfl_interval=1,
        progress=False,
    )


def test_config_requires_mesh_source_or_cache_artifact() -> None:
    """A config with neither input is rejected before any GPU work."""
    with pytest.raises(ValueError, match="mesh_source is required"):
        WorkflowConfig(
            mesh_source=None,
            manning_n=0.05,
            output_interval_s=1.0,
            dt_max=0.1,
            cfl_interval=1,
        )


def test_prepare_uses_cache_artifact_without_mesh_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare() builds its geometry from the artifact alone."""
    mesh_path = tmp_path / "mesh.obj"
    _write_quad_obj(mesh_path)
    artifact = build_geometry_cache(mesh_path, tmp_path / "mesh.geometry.npz")

    stub_solver_backends(monkeypatch)
    forbid_mesh_reads(monkeypatch)

    workflow = SWEWorkflow(_config(mesh_source=None, artifact=artifact))
    workflow.prepare()

    solver = cast("CapturingSolver", workflow.solver)
    assert workflow.geom is not None
    assert workflow.perm is not None
    expected_geom, expected_perm, meta = load_geometry_cache_artifact(artifact)
    assert meta.mesh_source == str(mesh_path)
    np.testing.assert_array_equal(workflow.geom.centroid, expected_geom.centroid)
    np.testing.assert_array_equal(workflow.geom.zb, expected_geom.zb)
    np.testing.assert_array_equal(workflow.perm, expected_perm)
    assert solver.geom.N == expected_geom.N == 2
    # prepare() forwards the configured solver keywords to the backend.
    assert set(solver.kwargs) == {"g", "dry_tol", "cfl", "workgroup_size"}
    # An OBJ mesh carries no roughness column, so the config value applies.
    np.testing.assert_array_equal(solver.n0, np.float32(0.05))
    # Raw mesh arrays stay unpopulated: the loader was never called.
    assert workflow.verts is None
    assert workflow.faces_flat is None


def test_cache_artifact_takes_precedence_over_mesh_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With both inputs set the artifact wins and the mesh path is never used."""
    mesh_path = tmp_path / "mesh.obj"
    _write_quad_obj(mesh_path)
    artifact = build_geometry_cache(mesh_path, tmp_path / "mesh.geometry.npz")

    stub_solver_backends(monkeypatch)
    forbid_mesh_reads(monkeypatch)

    # A path that does not exist: hashing or loading it would fail loudly.
    config = _config(mesh_source=str(tmp_path / "absent.obj"), artifact=artifact)
    workflow = SWEWorkflow(config)
    workflow.prepare()

    assert workflow.geom is not None
    assert workflow.perm is not None
    np.testing.assert_array_equal(workflow.perm, load_geometry_cache_artifact(artifact)[1])


def test_prepare_rejects_artifact_from_another_cache_version(tmp_path: Path) -> None:
    """A stale artifact must fail loudly instead of running the wrong geometry."""
    mesh_path = tmp_path / "mesh.obj"
    _write_quad_obj(mesh_path)
    artifact = build_geometry_cache(mesh_path, tmp_path / "mesh.geometry.npz")
    geom, perm, meta = load_geometry_cache_artifact(artifact)
    _write_geometry_npz(artifact, geom, perm, replace(meta, version=GEOMETRY_CACHE_VERSION + 1))

    with pytest.raises(ValueError, match="cache version"):
        SWEWorkflow(_config(mesh_source=None, artifact=artifact)).prepare()


def test_prepare_rejects_artifact_with_mismatched_reorder_mode(tmp_path: Path) -> None:
    """An artifact built in a different cell order is rejected, not silently used."""
    mesh_path = tmp_path / "mesh.obj"
    _write_quad_obj(mesh_path)
    artifact = build_geometry_cache(mesh_path, tmp_path / "plain.npz", use_hilbert_reorder=False)

    # _config keeps the default use_hilbert_reorder=True, so the modes disagree.
    with pytest.raises(ValueError, match="use_hilbert_reorder"):
        SWEWorkflow(_config(mesh_source=None, artifact=artifact)).prepare()


def test_cache_only_workflow_cannot_be_plotted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plotting re-reads raw mesh vertices, which a cache-only workflow lacks."""
    from inundation.bench.common import _mesh_vertices_for_plot

    mesh_path = tmp_path / "mesh.obj"
    _write_quad_obj(mesh_path)
    artifact = build_geometry_cache(mesh_path, tmp_path / "mesh.geometry.npz")

    stub_solver_backends(monkeypatch)
    workflow = SWEWorkflow(_config(mesh_source=None, artifact=artifact))
    workflow.prepare()

    with pytest.raises(RuntimeError, match="geometry cache artifact"):
        _mesh_vertices_for_plot(workflow)

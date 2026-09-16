"""Mesh package export contract relied on by downstream consumers."""

from __future__ import annotations

from inundation.mesh import hilbert_permutation
from inundation.mesh.geometry import hilbert_permutation as direct


def test_mesh_package_exports_hilbert_permutation() -> None:
    """The monorepo GPU worker imports hilbert_permutation from inundation.mesh."""
    assert hilbert_permutation is direct

"""Mesh I/O, geometry preprocessing, and geometry cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import (
        GeometryCacheMeta,
        build_geometry_cache,
        geometry_cache_key,
        load_geometry_cache,
        load_geometry_cache_artifact,
        save_geometry_cache,
    )
    from .geometry import MeshGeometry, build_geometry, hilbert_permutation, hilbert_reorder
    from .loader import load_mesh_file

__all__ = [
    "GeometryCacheMeta",
    "MeshGeometry",
    "build_geometry",
    "build_geometry_cache",
    "geometry_cache_key",
    "hilbert_permutation",
    "hilbert_reorder",
    "load_geometry_cache",
    "load_geometry_cache_artifact",
    "load_mesh_file",
    "save_geometry_cache",
]

_MESH_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _MESH_EXPORTS:
        from .cache import (  # pyright: ignore[reportMissingModuleSource]
            GeometryCacheMeta,
            build_geometry_cache,
            geometry_cache_key,
            load_geometry_cache,
            load_geometry_cache_artifact,
            save_geometry_cache,
        )
        from .geometry import (  # pyright: ignore[reportMissingModuleSource]
            MeshGeometry,
            build_geometry,
            hilbert_permutation,
            hilbert_reorder,
        )
        from .loader import load_mesh_file  # pyright: ignore[reportMissingModuleSource]

        return {
            "GeometryCacheMeta": GeometryCacheMeta,
            "build_geometry_cache": build_geometry_cache,
            "geometry_cache_key": geometry_cache_key,
            "load_geometry_cache": load_geometry_cache,
            "load_geometry_cache_artifact": load_geometry_cache_artifact,
            "save_geometry_cache": save_geometry_cache,
            "MeshGeometry": MeshGeometry,
            "build_geometry": build_geometry,
            "hilbert_permutation": hilbert_permutation,
            "hilbert_reorder": hilbert_reorder,
            "load_mesh_file": load_mesh_file,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Mesh I/O helpers for the mesh-based SWE GPU solver.

Supports loading meshes (triangles, quads, or mixed polygons) from
GeoPackage (.gpkg), Shapefile (.shp), GeoParquet (.parquet / .geoparquet),
and Wavefront OBJ (.obj) formats.

Usage
-----
    from swe_mesh import load_mesh_file

    verts, faces, face_offsets, zb_from_file = load_mesh_file("path/to/mesh.gpkg")
"""

import numpy as np


def load_mesh_file(path: str):
    """Load a polygonal mesh from a .gpkg, .shp, .parquet, .geoparquet, or .obj file.

    Accepts triangles, quads, and arbitrary convex polygons.

    .gpkg / .shp / .parquet / .geoparquet :
                   each feature must be a simple polygon (convex recommended).
                   .parquet / .geoparquet use pyarrow + shapely only (no geopandas).
                   .gpkg / .shp require geopandas  (pip install geopandas).
                   The optional ``z_mean`` column is used as bed elevation.
    .obj         : Wavefront OBJ; 'v x y [z]' and 'f i j k [l ...]' tokens.
                   Z coordinate is ignored.

    Returns
    -------
    verts        : (V, 2) float32   vertex (x, y) coordinates
    faces        : (sum_of_degrees,) int32  flat vertex indices for all faces
    face_offsets : (N+1,) int32  face i has verts faces[face_offsets[i]:face_offsets[i+1]]
    zb_from_file : (N,) float32 or None  per-cell bed elevation attribute

    """
    import os as _os

    ext = _os.path.splitext(path)[1].lower()

    if ext in (".parquet", ".geoparquet"):
        # pyarrow + shapely only — no geopandas needed for the parquet path.
        # Both real GeoParquet (written with 'geo' metadata) and plain parquet
        # (e.g. produced by a DuckDB Query clip node) store geometry as a WKB
        # binary column, so reading it with pyarrow and decoding via
        # shapely.from_wkb works uniformly for both.
        import pyarrow.parquet as pq
        from shapely import from_wkb

        table = pq.read_table(path)
        geometry_wkb = table.column("geometry").to_numpy(zero_copy_only=False)
        geoms = from_wkb(geometry_wkb)
        not_null = np.array([g is not None for g in geoms], dtype=bool)

        coord_map: dict = {}
        all_coords: list = []
        faces_list: list = []
        for geom in geoms:
            if geom is None:
                continue
            coords = list(geom.exterior.coords)[:-1]  # drop closing duplicate
            if len(coords) < 3:
                raise ValueError(f"Geometry has {len(coords)} vertices — need at least 3.")
            face_idx = []
            for xy in coords:
                key = (float(xy[0]), float(xy[1]))
                if key not in coord_map:
                    coord_map[key] = len(all_coords)
                    all_coords.append(key)
                face_idx.append(coord_map[key])
            faces_list.append(face_idx)

        if not faces_list:
            raise ValueError(
                f"Mesh file '{path}' contains no valid polygon geometries "
                f"({table.num_rows} rows, {int(not_null.sum())} non-null geometries)."
            )

        zb_vals = (
            table.column("z_mean").to_numpy(zero_copy_only=False)[not_null].astype(np.float32)
            if "z_mean" in table.column_names
            else None
        )

        verts = np.array(all_coords, dtype=np.float32)
        faces_flat = np.concatenate([np.array(f, dtype=np.int32) for f in faces_list])
        offsets = np.zeros(len(faces_list) + 1, dtype=np.int32)
        for i, f in enumerate(faces_list):
            offsets[i + 1] = offsets[i] + len(f)

        return verts, faces_flat, offsets, zb_vals

    if ext in (".gpkg", ".shp"):
        import geopandas as gpd

        gdf = gpd.read_file(path)
        coord_map = {}
        all_coords = []
        faces_list = []
        for geom in gdf.geometry:
            if geom is None:
                continue
            coords = list(geom.exterior.coords)[:-1]  # drop closing duplicate
            if len(coords) < 3:
                raise ValueError(f"Geometry has {len(coords)} vertices — need at least 3.")
            face_idx = []
            for xy in coords:
                key = (float(xy[0]), float(xy[1]))
                if key not in coord_map:
                    coord_map[key] = len(all_coords)
                    all_coords.append(key)
                face_idx.append(coord_map[key])
            faces_list.append(face_idx)

        if not faces_list:
            n_rows = len(gdf)
            n_valid = gdf.geometry.notna().sum()
            raise ValueError(
                f"Mesh file '{path}' contains no valid polygon geometries "
                f"({n_rows} rows, {n_valid} non-null geometries)."
            )

        zb_vals = (
            gdf.loc[gdf.geometry.notna(), "z_mean"].to_numpy(dtype=np.float32)
            if "z_mean" in gdf.columns
            else None
        )

        verts = np.array(all_coords, dtype=np.float32)
        faces_flat = np.concatenate([np.array(f, dtype=np.int32) for f in faces_list])
        offsets = np.zeros(len(faces_list) + 1, dtype=np.int32)
        for i, f in enumerate(faces_list):
            offsets[i + 1] = offsets[i] + len(f)

        return verts, faces_flat, offsets, zb_vals

    if ext == ".obj":
        vert_list: list = []
        faces_list: list = []
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts or parts[0].startswith("#"):
                    continue
                if parts[0] == "v":
                    vert_list.append([float(parts[1]), float(parts[2])])
                elif parts[0] == "f":
                    idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    if len(idx) < 3:
                        raise ValueError(f"OBJ face has {len(idx)} vertices — need at least 3.")
                    faces_list.append(idx)

        if not faces_list:
            raise ValueError(f"OBJ file '{path}' contains no face ('f') records.")

        verts = np.array(vert_list, dtype=np.float32)
        faces_flat = np.concatenate([np.array(f, dtype=np.int32) for f in faces_list])
        offsets = np.zeros(len(faces_list) + 1, dtype=np.int32)
        for i, f in enumerate(faces_list):
            offsets[i + 1] = offsets[i] + len(f)

        return verts, faces_flat, offsets, None

    raise ValueError(
        f"Unsupported format: '{ext}'. Supported: .gpkg, .shp, .parquet, .geoparquet, .obj"
    )

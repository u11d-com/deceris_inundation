"""Tests for the floodplain-depressions benchmark (EA Test 2, 2010 dataset).

Pure-NumPy validation of the published-dataset loaders (DEM raster, inflow
hydrograph, output points), the cell-averaged bed, and the inertia-free
fill-spill reference the harness gates against — no GPU, no solver run. The
solver-facing gates live in bench/floodplain_depressions.py (manual,
GPU-required).
"""

# pyright: reportPrivateUsage=false
# The harness's dataset helpers are module-private (leading underscore);
# exercising them here is intentional, so silence the private-usage warning.

from __future__ import annotations

import numpy as np
import pytest

from deceris.inundation.bench.common import (
    AsciiGrid,
    block_average_grid,
    depression_basin,
    load_ascii_grid,
    resample_snapshots_uniform,
)
from deceris.inundation.bench.floodplain_depressions import (
    DEFAULT_DATASET_DIR,
    DOMAIN_L_M,
    DepressionSpec,
    _bed_grid,
    _hydrograph_phases,
    _injected_volume,
    _load_gauges,
    _load_hydrograph,
)

# Published hydrograph: 0 -> 20 m^3/s over 5 min, 76 min plateau, 5 min recession.
PUBLISHED_VOLUME_M3 = 0.5 * 20.0 * 300.0 + 20.0 * 76.0 * 60.0 + 0.5 * 20.0 * 300.0


def _spec(**over: object) -> DepressionSpec:
    base: dict[str, object] = {
        "name": "eggbox",
        "nx": 100,
        "ny": 100,
        "domain_l_m": DOMAIN_L_M,
        "manning_n": 0.03,
        "t_end_s": 48.0 * 3600.0,
        "dt_max": 5.0,
        "cfl_interval": 10,
        "dataset_dir": DEFAULT_DATASET_DIR,
    }
    base.update(over)
    return DepressionSpec(**base)  # pyright: ignore[reportArgumentType]


class TestDemRaster:
    def test_header_matches_published_raster(self) -> None:
        grid = load_ascii_grid(_spec().dem_path)
        assert grid.z.shape == (1201, 1201)
        assert grid.cellsize == 2.0
        # 2000 m modelled area plus a ~200 m apron on every side.
        assert grid.x[0] == pytest.approx(-199.0)
        assert grid.y[0] == pytest.approx(-199.0)
        assert grid.x[-1] == pytest.approx(2201.0)
        assert grid.y[-1] == pytest.approx(2201.0)

    def test_axes_ascend_and_no_nodata(self) -> None:
        grid = load_ascii_grid(_spec().dem_path)
        assert np.all(np.diff(grid.x) > 0.0)
        assert np.all(np.diff(grid.y) > 0.0)
        assert bool(np.isfinite(grid.z).all())

    def test_rows_are_flipped_to_y_ascending(self) -> None:
        # Within the modelled area the plane falls north -> south at 1:1500, so
        # a northern row must sit above a southern one once the file's
        # north-first row order has been flipped. (The 200 m apron outside the
        # modelled area does not continue the plane, so sample inside it.)
        grid = load_ascii_grid(_spec().dem_path)
        north = grid.z[int(np.argmin(np.abs(grid.y - 1900.0)))]
        south = grid.z[int(np.argmin(np.abs(grid.y - 100.0)))]
        assert float(north.mean()) > float(south.mean())


class TestBedGrid:
    def test_shape_and_ordering(self) -> None:
        spec = _spec()
        zb = _bed_grid(spec)
        assert zb.shape == (spec.ny, spec.nx)
        assert bool(np.isfinite(zb).all())

    def test_northwest_is_the_high_corner(self) -> None:
        # Inflow enters at the NW corner, the highest ground; the SE corner is
        # the low end of the ~2 m diagonal drop.
        zb = _bed_grid(_spec())
        assert float(zb[-1, 0]) > float(zb[0, -1])

    def test_cell_average_stays_within_the_raster_range(self) -> None:
        spec = _spec()
        grid = load_ascii_grid(spec.dem_path)
        zb = _bed_grid(spec)
        assert float(zb.min()) >= float(np.nanmin(grid.z))
        assert float(zb.max()) <= float(np.nanmax(grid.z))

    def test_block_average_of_a_linear_ramp_is_the_midpoint(self) -> None:
        # 4 fine cells per coarse cell over a linear field: the coarse value is
        # the mean of the four, i.e. the coarse cell's centre value.
        spec = _spec()
        grid = load_ascii_grid(spec.dem_path)
        ramp = np.broadcast_to(grid.x, grid.z.shape).copy()
        averaged = block_average_grid(
            AsciiGrid(x=grid.x, y=grid.y, z=ramp, cellsize=grid.cellsize),
            x_edges=np.linspace(0.0, DOMAIN_L_M, 101),
            y_edges=np.linspace(0.0, DOMAIN_L_M, 101),
        )
        centres = (np.arange(100, dtype=np.float64) + 0.5) * 20.0
        assert np.allclose(averaged[0], centres)

    def test_mesh_outside_raster_coverage_is_rejected(self) -> None:
        spec = _spec()
        grid = load_ascii_grid(spec.dem_path)
        with pytest.raises(ValueError, match="beyond the raster coverage"):
            block_average_grid(
                grid,
                x_edges=np.linspace(0.0, 10_000.0, 3),
                y_edges=np.linspace(0.0, 10_000.0, 3),
            )


class TestGauges:
    def test_sixteen_published_points(self) -> None:
        gauges = _load_gauges(_spec().gauges_path)
        assert gauges.shape == (16, 2)

    def test_column_major_ordering_from_the_southwest(self) -> None:
        # EA numbering: p = col*4 + row + 1, columns west->east, rows south->north.
        gauges = _load_gauges(_spec().gauges_path)
        assert np.allclose(gauges[0], (250.0, 250.0))  # p1  (SW)
        assert np.allclose(gauges[3], (250.0, 1750.0))  # p4  (NW, top of west column)
        assert np.allclose(gauges[12], (1750.0, 250.0))  # p13 (SE)
        assert np.allclose(gauges[15], (1750.0, 1750.0))  # p16 (NE)

    def test_points_sit_in_depressions(self) -> None:
        # Every output point is at the centre of a depression: its cell is at
        # the floor of the bowl (the sampled minimum can land a cell away, so
        # allow a centimetre) and well below the surrounding high ground.
        spec = _spec()
        zb = _bed_grid(spec)
        for gx, gy in _load_gauges(spec.gauges_path):
            i, j = int(gx // spec.dx_m), int(gy // spec.dy_m)
            assert float(zb[j, i]) <= float(zb[j - 1 : j + 2, i - 1 : i + 2].min()) + 0.01
            assert float(zb[j - 5 : j + 6, i - 5 : i + 6].max()) - float(zb[j, i]) > 0.3


class TestHydrograph:
    def test_times_are_converted_from_minutes(self) -> None:
        t, q = _load_hydrograph(_spec().bc_path)
        assert t[0] == 0.0
        assert t[-1] == pytest.approx(2880.0 * 60.0)
        assert float(q.max()) == pytest.approx(20.0)

    def test_injected_volume_matches_the_published_hydrograph(self) -> None:
        # Midpoint sampling is volume-exact: every breakpoint is a whole minute.
        assert _injected_volume(_spec()) == pytest.approx(PUBLISHED_VOLUME_M3)

    def test_phase_durations_sum_to_t_end(self) -> None:
        spec = _spec()
        assert sum(p.duration_s for p in _hydrograph_phases(spec)) == pytest.approx(spec.t_end_s)

    def test_settle_tail_has_no_sources(self) -> None:
        phases = _hydrograph_phases(_spec())
        assert list(phases[-1].sources) == []
        # 91 min of inflow at 60 s per phase, then the settle tail.
        assert len(phases) == 91 + 1

    def test_sources_split_the_discharge_along_the_inflow_line(self) -> None:
        spec = _spec()
        peak = max(_hydrograph_phases(spec), key=lambda p: sum(s.discharge_m3s for s in p.sources))
        assert sum(s.discharge_m3s for s in peak.sources) == pytest.approx(20.0)
        ys = sorted(s.center_xy[1] for s in peak.sources)
        assert ys[0] >= 1900.0
        assert ys[-1] <= 2000.0


class TestDepressionBasins:
    def test_every_output_point_sits_in_a_closed_basin(self) -> None:
        spec = _spec()
        zb = _bed_grid(spec)
        for gx, gy in _load_gauges(spec.gauges_path):
            cell = (int(gy // spec.dy_m), int(gx // spec.dx_m))
            sill, pool, capacity = depression_basin(zb, cell, cell_area=spec.cell_area_m2)
            assert sill > float(zb[cell])
            assert pool.size > 1
            assert capacity > 0.0

    def test_basin_storage_exceeds_the_injected_volume(self) -> None:
        # The floodplain can hold more than the hydrograph delivers, so the
        # far column cannot fill through — which is what the harness gates.
        spec = _spec()
        zb = _bed_grid(spec)
        capacity = sum(
            depression_basin(
                zb, (int(gy // spec.dy_m), int(gx // spec.dx_m)), cell_area=spec.cell_area_m2
            )[2]
            for gx, gy in _load_gauges(spec.gauges_path)
        )
        assert capacity > 2.0 * _injected_volume(spec)

    def test_sill_is_the_lowest_saddle(self) -> None:
        # A basin between two ridges spills over the lower one.
        zb = np.array([[5.0, 1.0, 3.0, 0.0, 4.0]], dtype=np.float64)
        sill, pool, capacity = depression_basin(zb, (0, 1), cell_area=1.0)
        assert sill == pytest.approx(3.0)
        assert sorted(pool.tolist()) == [1, 2]
        assert capacity == pytest.approx(2.0)

    def test_walks_downhill_to_the_basin_floor_first(self) -> None:
        # Seeding on the flank must find the same basin as seeding on the floor.
        zb = np.array([[5.0, 1.0, 2.0, 3.0, 4.0, 0.5, 6.0]], dtype=np.float64)
        assert depression_basin(zb, (0, 3), cell_area=1.0)[0] == pytest.approx(
            depression_basin(zb, (0, 1), cell_area=1.0)[0]
        )

    def test_rejects_a_cell_outside_the_bed(self) -> None:
        zb = np.zeros((2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="outside"):
            depression_basin(zb, (5, 5), cell_area=1.0)


class TestFrameResampling:
    def test_evenly_spaces_frames_bunched_by_phase_boundaries(self) -> None:
        # Mirrors the harness schedule: many short phases each force a
        # snapshot, then the settle tail reports on the output interval.
        times = [0.0, 1.0, 2.0, 3.0, 100.0, 200.0]
        snaps = [np.full(2, t, dtype=np.float32) for t in times]
        frames, frame_times = resample_snapshots_uniform(snaps, times, 5)
        assert frame_times == pytest.approx([0.0, 50.0, 100.0, 150.0, 200.0])
        assert [float(f[0]) for f in frames] == [0.0, 3.0, 100.0, 100.0, 200.0]

    def test_endpoints_are_preserved(self) -> None:
        times = [0.0, 7.0, 9.0]
        snaps = [np.zeros(1, dtype=np.float32) for _ in times]
        _frames, frame_times = resample_snapshots_uniform(snaps, times, 4)
        assert frame_times[0] == pytest.approx(0.0)
        assert frame_times[-1] == pytest.approx(9.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            resample_snapshots_uniform([np.zeros(1, dtype=np.float32)], [0.0, 1.0], 2)

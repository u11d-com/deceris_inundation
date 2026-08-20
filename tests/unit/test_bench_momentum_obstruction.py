"""Tests for the momentum-obstruction benchmark (EA Test 3, published dataset).

Pure-NumPy validation of the dataset loaders (DEM raster + inflow
hydrograph), the phase schedule, and the still-water control state — no GPU,
no solver run. The solver-facing invariant/qualitative gates live in
bench/momentum_obstruction.py (manual, GPU-required).
"""

# pyright: reportPrivateUsage=false
# The harness's helpers are module-private (leading underscore); exercising
# them here is intentional, so silence the private-usage warning.

from __future__ import annotations

import numpy as np

from deceris.inundation.bench.momentum_obstruction import (
    CREST_X_M,
    DEFAULT_DATASET_DIR,
    INFLOW_CENTERS_Y_M,
    INFLOW_PHASE_S,
    INFLOW_RADIUS_M,
    INFLOW_X_M,
    POINT1_X_M,
    POINT2_X_M,
    ObstructionSpec,
    _bed_original,
    _control_initial_state,
    _control_wse,
    _crest_z,
    _hydrograph_phases,
    _inflow_sources,
    _initial_state,
    _injected_volume,
    _load_dem_profile,
    _load_hydrograph,
    _original_cell_centroids,
)


def _spec(**over: object) -> ObstructionSpec:
    base: dict[str, object] = {
        "name": "obstruction",
        "nx": 150,
        "ny": 50,
        "channel_len_m": 300.0,
        "channel_width_m": 100.0,
        "manning_n": 0.01,
        "t_end_s": 900.0,
        "dt_max": 2.0,
        "cfl_interval": 10,
        "dataset_dir": DEFAULT_DATASET_DIR,
    }
    base.update(over)
    return ObstructionSpec(**base)  # pyright: ignore[reportArgumentType]


class TestDemProfile:
    def test_parses_and_is_finite(self) -> None:
        x, z = _load_dem_profile(_spec().dem_path)
        assert x.shape == z.shape == (186,)
        assert np.all(np.isfinite(z))
        assert np.all(np.diff(x) == 2.0)  # native 2 m raster

    def test_covers_modelled_area(self) -> None:
        # The raster apron extends beyond the spec's [0, 300] modelled area.
        x, _z = _load_dem_profile(_spec().dem_path)
        assert x[0] < 0.0 < 300.0 < x[-1]

    def test_published_landmarks(self) -> None:
        # Troughs at the gauges, obstruction crest ~10.0 between them, and
        # the 1:200 approach slope falling left to right.
        x, z = _load_dem_profile(_spec().dem_path)
        assert abs(float(np.interp(POINT1_X_M, x, z)) - 9.75) < 0.01
        assert abs(float(np.interp(POINT2_X_M, x, z)) - 9.75) < 0.01
        between = (x > POINT1_X_M) & (x < POINT2_X_M)
        assert abs(float(z[between].max()) - 10.0) < 0.01
        slope = (x >= 0.0) & (x <= 100.0)
        assert np.all(np.diff(z[slope]) < 0.0)

    def test_bed_original_matches_dem(self) -> None:
        spec = _spec()
        zb = _bed_original(spec)
        cx, _cy = _original_cell_centroids(spec)
        x, z = _load_dem_profile(spec.dem_path)
        assert zb.shape == (spec.nx * spec.ny,)
        assert np.allclose(zb, np.interp(cx, x, z))

    def test_crest_z(self) -> None:
        assert abs(_crest_z(_spec()) - 10.0) < 0.01


class TestHydrograph:
    def test_parses_published_curve(self) -> None:
        t, q = _load_hydrograph(_spec().bc_path)
        assert t[0] == 0.0
        assert t[-1] == 900.0
        assert float(q.max()) == 65.5
        assert q[-1] == 0.0

    def test_total_volume_is_exact(self) -> None:
        # Midpoint sampling of the piecewise-linear curve on 1 s phases is
        # volume-exact: matches the trapezoidal integral of the CSV.
        spec = _spec()
        t, q = _load_hydrograph(spec.bc_path)
        expected = float((0.5 * (q[1:] + q[:-1]) * np.diff(t)).sum())
        assert abs(_injected_volume(spec) - expected) < 1e-9
        assert abs(expected - 1310.0) < 1e-9

    def test_phase_schedule_spans_t_end(self) -> None:
        spec = _spec()
        phases = _hydrograph_phases(spec)
        assert abs(sum(p.duration_s for p in phases) - spec.t_end_s) < 1e-9
        # Active phases are 1 s; the settle tail carries no sources.
        assert list(phases[-1].sources) == []
        assert all(p.duration_s == INFLOW_PHASE_S for p in phases[:-1])

    def test_sources_hug_the_upstream_boundary(self) -> None:
        sources = _inflow_sources(65.5)
        assert len(sources) == len(INFLOW_CENTERS_Y_M)
        assert abs(sum(s.discharge_m3s for s in sources) - 65.5) < 1e-12
        for s in sources:
            assert s.center_xy[0] == INFLOW_X_M
            assert s.radius_m == INFLOW_RADIUS_M
            # Footprint stays on the upstream slope, far from Point 1.
            assert s.center_xy[0] + s.radius_m < POINT1_X_M / 2.0

    def test_rejects_short_t_end(self) -> None:
        try:
            _hydrograph_phases(_spec(t_end_s=10.0))
        except ValueError:
            return
        raise AssertionError("expected ValueError for t_end shorter than the hydrograph")


class TestInitialState:
    def test_dry_bed(self) -> None:
        spec = _spec()
        state = _initial_state(spec)
        for key in ("h", "hu", "hv"):
            arr = np.asarray(state[key], dtype=np.float32)
            assert arr.shape == (spec.nx * spec.ny,)
            assert not arr.any()


class TestStillWaterControl:
    def test_dataset_fills_depression_to_the_brim(self) -> None:
        # The published design: inflow volume ~= Point 1 capacity below the
        # crest, so the control surface tops out at the crest itself.
        spec = _spec()
        assert abs(_control_wse(spec) - _crest_z(spec)) < 0.01

    def test_control_volume_capped_at_capacity(self) -> None:
        spec = _spec()
        state = _control_initial_state(spec)
        h = np.asarray(state["h"], dtype=np.float64)
        volume = float(h.sum()) * spec.dx_m * spec.dy_m
        injected = _injected_volume(spec)
        # Never exceeds the injected volume; close to it (brim-full design).
        assert volume <= injected + 1e-6
        assert volume > 0.9 * injected

    def test_control_is_still_level_point1_pond(self) -> None:
        spec = _spec()
        state = _control_initial_state(spec)
        h = np.asarray(state["h"], dtype=np.float64)
        hu = np.asarray(state["hu"], dtype=np.float32)
        hv = np.asarray(state["hv"], dtype=np.float32)
        assert not hu.any()
        assert not hv.any()
        cx, _cy = _original_cell_centroids(spec)
        zb = _bed_original(spec)
        wet = h > 0.0
        # Water only in the first depression (left of the crest, below it).
        assert np.all(cx[wet] < CREST_X_M)
        assert np.all(zb[wet] < _crest_z(spec))
        # Level surface: h + zb is constant across wet cells.
        wse = h[wet] + zb[wet]
        assert float(wse.max() - wse.min()) < 1e-6

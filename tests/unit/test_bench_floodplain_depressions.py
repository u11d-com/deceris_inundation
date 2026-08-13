"""Tests for the flattened egg-box floodplain-depressions benchmark (EA Test 2).

Pure-NumPy validation of the synthetic egg-box bed builder and the inflow
hydrograph — no GPU, no solver run. The solver-facing invariant/qualitative
gates live in bench/floodplain_depressions.py (manual, GPU-required).
"""

# pyright: reportPrivateUsage=false
# The harness's hydrograph helpers are module-private (leading underscore);
# exercising them here is intentional, so silence the private-usage warning.

from __future__ import annotations

import numpy as np

from deceris.inundation.bench.common import eggbox_bed, eggbox_depression_centers
from deceris.inundation.bench.floodplain_depressions import (
    DepressionSpec,
    _hydrograph_phases,
    _injected_volume,
)

DOMAIN_L = 2000.0
N_PER_SIDE = 4
NE_RISE = 1.0
DEP_RADIUS = 100.0
DEP_DEPTH = 0.5


def _spec(**over: object) -> DepressionSpec:
    base: dict[str, object] = {
        "name": "eggbox",
        "nx": 100,
        "ny": 100,
        "domain_l_m": DOMAIN_L,
        "n_per_side": N_PER_SIDE,
        "plateau_z_m": 0.0,
        "ne_rise_m": NE_RISE,
        "dep_radius_m": DEP_RADIUS,
        "dep_depth_m": DEP_DEPTH,
        "manning_n": 0.03,
        "inflow_peak_m3s": 20.0,
        "hydrograph_base_s": 85.0 * 60.0,
        "hydrograph_phases": 17,
        "settle_s": 3600.0,
        "dt_max": 5.0,
        "cfl_interval": 10,
    }
    base.update(over)
    return DepressionSpec(**base)  # pyright: ignore[reportArgumentType]


class TestDepressionCenters:
    def test_count_and_shape(self) -> None:
        centers = eggbox_depression_centers(DOMAIN_L, N_PER_SIDE)
        assert centers.shape == (16, 2)

    def test_corner_ordering(self) -> None:
        # Column-major from bottom-left: p1 = SW, p13 = SE, p16 = NE.
        centers = eggbox_depression_centers(DOMAIN_L, N_PER_SIDE)
        assert np.allclose(centers[0], (250.0, 250.0))  # p1
        assert np.allclose(centers[3], (250.0, 1750.0))  # p4 (top of west column)
        assert np.allclose(centers[12], (1750.0, 250.0))  # p13 (bottom of east column)
        assert np.allclose(centers[15], (1750.0, 1750.0))  # p16 (NE corner)

    def test_far_corner_points_are_northeast(self) -> None:
        # Gate uses the last two entries as points 15 & 16 (top of NE column).
        centers = eggbox_depression_centers(DOMAIN_L, N_PER_SIDE)
        assert np.allclose(centers[14], (1750.0, 1250.0))  # p15
        assert np.allclose(centers[15], (1750.0, 1750.0))  # p16


class TestEggboxBed:
    def test_dtype_and_finite(self) -> None:
        cx = np.array([500.0, 750.0, 1900.0])
        cy = np.array([500.0, 750.0, 1900.0])
        zb = eggbox_bed(
            cx,
            cy,
            domain_l=DOMAIN_L,
            n_per_side=N_PER_SIDE,
            ne_rise_m=NE_RISE,
            dep_radius_m=DEP_RADIUS,
            dep_depth_m=DEP_DEPTH,
        )
        assert zb.dtype == np.float32
        assert np.all(np.isfinite(zb))

    def test_depression_center_is_full_depth_below_local_plane(self) -> None:
        # At a depression center the bed sits exactly dep_depth below the
        # local (depression-free) plateau+rise plane.
        cx = np.array([750.0])
        cy = np.array([750.0])
        zb = eggbox_bed(
            cx,
            cy,
            domain_l=DOMAIN_L,
            n_per_side=N_PER_SIDE,
            ne_rise_m=NE_RISE,
            dep_radius_m=DEP_RADIUS,
            dep_depth_m=DEP_DEPTH,
        )
        plane = NE_RISE * (750.0 + 750.0) / (2.0 * DOMAIN_L)
        assert float(zb[0]) == np.float32(plane - DEP_DEPTH)

    def test_rim_matches_plateau(self) -> None:
        # Just outside a depression radius the bed is back on the plane (the
        # raised-cosine bowl reaches zero at dep_radius_m).
        center = (750.0, 750.0)
        cx = np.array([center[0] + DEP_RADIUS + 1.0])
        cy = np.array([center[1]])
        zb = eggbox_bed(
            cx,
            cy,
            domain_l=DOMAIN_L,
            n_per_side=N_PER_SIDE,
            ne_rise_m=NE_RISE,
            dep_radius_m=DEP_RADIUS,
            dep_depth_m=DEP_DEPTH,
        )
        plane = NE_RISE * (float(cx[0]) + float(cy[0])) / (2.0 * DOMAIN_L)
        assert float(zb[0]) == np.float32(plane)

    def test_northeast_is_higher_than_southwest(self) -> None:
        # The NE rise makes the far corner structurally the highest ground
        # (why EA points 15 & 16 stay dry). Compare plateau cells clear of
        # any depression.
        zb = eggbox_bed(
            np.array([500.0, 1500.0]),
            np.array([500.0, 1500.0]),
            domain_l=DOMAIN_L,
            n_per_side=N_PER_SIDE,
            ne_rise_m=NE_RISE,
            dep_radius_m=DEP_RADIUS,
            dep_depth_m=DEP_DEPTH,
        )
        assert float(zb[1]) > float(zb[0])

    def test_depressions_are_local_minima(self) -> None:
        # Each depression center is lower than a plateau point midway to its
        # NE neighbour (disconnected sinks, not a connected trench).
        zb = eggbox_bed(
            np.array([750.0, 1000.0]),
            np.array([750.0, 1000.0]),
            domain_l=DOMAIN_L,
            n_per_side=N_PER_SIDE,
            ne_rise_m=NE_RISE,
            dep_radius_m=DEP_RADIUS,
            dep_depth_m=DEP_DEPTH,
        )
        assert float(zb[0]) < float(zb[1])


class TestHydrograph:
    def test_injected_volume_matches_triangle_area(self) -> None:
        spec = _spec()
        # Symmetric triangle sampled at phase midpoints integrates to
        # ~0.5 * peak * base.
        expected = 0.5 * spec.inflow_peak_m3s * spec.hydrograph_base_s
        assert abs(_injected_volume(spec) - expected) / expected < 0.02

    def test_phase_durations_sum_to_t_end(self) -> None:
        spec = _spec()
        phases = _hydrograph_phases(spec)
        total = sum(p.duration_s for p in phases)
        assert total == spec.t_end_s

    def test_settle_tail_has_no_sources(self) -> None:
        spec = _spec()
        phases = _hydrograph_phases(spec)
        assert len(phases) == spec.hydrograph_phases + 1
        assert list(phases[-1].sources) == []

    def test_discharge_is_nonnegative_and_peaks_midway(self) -> None:
        spec = _spec()
        inflow = [sum(s.discharge_m3s for s in p.sources) for p in _hydrograph_phases(spec)[:-1]]
        assert all(q >= 0.0 for q in inflow)
        peak_idx = int(np.argmax(inflow))
        assert abs(peak_idx - (spec.hydrograph_phases - 1) / 2.0) <= 0.5

"""Tests for the flood-propagation benchmark (EA Test 4, 2010 dataset).

Pure-NumPy validation of the published-dataset loaders, the phase schedule,
and the axisymmetric reference the harness gates against — no GPU, no solver
run. The solver-facing gates live in bench/flood_propagation.py (manual,
GPU-required).
"""

# pyright: reportPrivateUsage=false
# The harness's dataset helpers are module-private (leading underscore);
# exercising them here is intentional, so silence the private-usage warning.

from __future__ import annotations

import numpy as np
import pytest

from deceris.inundation.bench.common import solve_radial_inflow
from deceris.inundation.bench.flood_propagation import (
    DEFAULT_DATASET_DIR,
    DOMAIN_X_M,
    DOMAIN_Y_M,
    INFLOW_LINE_Y_M,
    RADIAL_VALID_UNTIL_S,
    PropagationSpec,
    _gauge_radii,
    _hydrograph_phases,
    _injected_volume,
    _load_gauges,
    _load_hydrograph,
    _max_rel_error,
)

from ._fixtures import requires_ea_dataset

# Published hydrograph: 0 -> 20 m^3/s over 55 min, 180 min plateau, 60 min recession.
PUBLISHED_VOLUME_M3 = 0.5 * 20.0 * 55.0 * 60.0 + 20.0 * 180.0 * 60.0 + 0.5 * 20.0 * 60.0 * 60.0


def _spec(**over: object) -> PropagationSpec:
    base: dict[str, object] = {
        "name": "propagation",
        "nx": 200,
        "ny": 400,
        "domain_x_m": DOMAIN_X_M,
        "domain_y_m": DOMAIN_Y_M,
        "manning_n": 0.05,
        "t_end_s": 5.0 * 3600.0,
        "probe_time_s": 2.0 * 3600.0,
        "dt_max": 2.0,
        "cfl_interval": 10,
        "dataset_dir": DEFAULT_DATASET_DIR,
    }
    base.update(over)
    return PropagationSpec(**base)  # pyright: ignore[reportArgumentType]


def _cheap_reference(**over: object) -> object:
    """Small, fast reference solve for logic tests."""
    kwargs: dict[str, object] = {
        "hydrograph_t_s": np.array([0.0, 60.0, 600.0], dtype=np.float64),
        "hydrograph_q_m3s": np.array([0.0, 40.0, 40.0], dtype=np.float64),
        "source_radius_m": 10.0,
        "manning_n": 0.05,
        "g": 9.81,
        "r_max": 400.0,
        "n_cells": 200,
    }
    kwargs.update(over)
    return solve_radial_inflow([300.0, 600.0], **kwargs)  # pyright: ignore[reportArgumentType]


@requires_ea_dataset
class TestGauges:
    def test_six_published_points(self) -> None:
        assert _load_gauges(_spec().gauges_path).shape == (6, 2)

    def test_five_on_axis_one_diagonal(self) -> None:
        # Points 1-5 run east from the source at (0, 1000); point 6 sits at 45
        # degrees, which is what makes it an isotropy probe.
        gauges = _load_gauges(_spec().gauges_path)
        assert np.allclose(gauges[:5, 1], 1000.0)
        assert np.allclose(gauges[5], (300.0, 1300.0))

    def test_radii_from_the_inflow_line_centre(self) -> None:
        radii = _gauge_radii(_load_gauges(_spec().gauges_path))
        assert np.allclose(radii[:5], (50.0, 100.0, 200.0, 300.0, 400.0))
        assert float(radii[5]) == pytest.approx(np.hypot(300.0, 300.0))

    def test_all_gauges_are_inside_the_radial_window(self) -> None:
        # The reference only applies while the front is clear of the walls, so
        # every gauge must sit well inside the nearest-wall distance.
        radii = _gauge_radii(_load_gauges(_spec().gauges_path))
        assert float(radii.max()) < 0.5 * min(DOMAIN_X_M, DOMAIN_Y_M / 2.0)


@requires_ea_dataset
class TestHydrograph:
    def test_times_are_converted_from_minutes(self) -> None:
        t, q = _load_hydrograph(_spec().bc_path)
        assert t[-1] == pytest.approx(300.0 * 60.0)
        assert float(q.max()) == pytest.approx(20.0)

    def test_injected_volume_matches_the_published_hydrograph(self) -> None:
        # Midpoint sampling is volume-exact: every breakpoint is a whole minute.
        assert _injected_volume(_spec()) == pytest.approx(PUBLISHED_VOLUME_M3)

    def test_phase_durations_sum_to_t_end(self) -> None:
        spec = _spec()
        assert sum(p.duration_s for p in _hydrograph_phases(spec)) == pytest.approx(spec.t_end_s)

    def test_hydrograph_exactly_fills_the_run_without_a_settle_tail(self) -> None:
        # The published inflow ends exactly at t_end (300 min), so unlike Tests
        # 2 and 3 there is no quiescent tail phase — the run finishes on the
        # recession limb, still injecting a trickle.
        phases = _hydrograph_phases(_spec())
        assert len(phases) == 300
        tail = sum(s.discharge_m3s for s in phases[-1].sources)
        assert 0.0 < tail < 1.0

    def test_sources_sit_on_the_published_inflow_line(self) -> None:
        spec = _spec()
        peak = max(_hydrograph_phases(spec), key=lambda p: sum(s.discharge_m3s for s in p.sources))
        assert sum(s.discharge_m3s for s in peak.sources) == pytest.approx(20.0)
        ys = sorted(s.center_xy[1] for s in peak.sources)
        assert ys[0] >= INFLOW_LINE_Y_M[0]
        assert ys[-1] <= INFLOW_LINE_Y_M[1]
        assert all(s.center_xy[0] == pytest.approx(spec.inflow_x_m) for s in peak.sources)

    def test_shorter_run_truncates_the_hydrograph(self) -> None:
        short = _hydrograph_phases(_spec(t_end_s=600.0))
        assert sum(p.duration_s for p in short) == pytest.approx(600.0)


class TestRadialReference:
    def test_conserves_the_injected_volume(self) -> None:
        ref = _cheap_reference()
        dr = float(ref.r_m[1] - ref.r_m[0])  # pyright: ignore[reportAttributeAccessIssue]
        for k, t in enumerate(ref.times_s):  # pyright: ignore[reportAttributeAccessIssue]
            vol = float((ref.h_m[k] * ref.r_m).sum()) * dr * 2.0 * np.pi  # pyright: ignore[reportAttributeAccessIssue]
            expected = 40.0 * (float(t) - 30.0)  # ramp over the first 60 s
            assert vol == pytest.approx(expected, rel=2e-3)

    def test_depth_decreases_away_from_the_source(self) -> None:
        # Monotone outside the injection disc; inside it the distributed source
        # builds a small mound, so the peak sits a cell or two off centre.
        ref = _cheap_reference()
        h = ref.h_m[-1]  # pyright: ignore[reportAttributeAccessIssue]
        outside = (ref.r_m > 10.0) & (h > 1e-3)  # pyright: ignore[reportAttributeAccessIssue]
        assert bool(np.all(np.diff(h[outside]) <= 1e-9))

    def test_depths_are_non_negative_and_finite(self) -> None:
        ref = _cheap_reference()
        assert float(ref.h_m.min()) >= 0.0  # pyright: ignore[reportAttributeAccessIssue]
        assert bool(np.isfinite(ref.h_m).all())  # pyright: ignore[reportAttributeAccessIssue]

    def test_front_advances_monotonically_with_radius(self) -> None:
        # Arrival time must increase outward — the definition of a front.
        ref = _cheap_reference()
        arrival = ref.arrival_s  # pyright: ignore[reportAttributeAccessIssue]
        reached = np.isfinite(arrival)
        assert bool(np.all(np.diff(arrival[reached]) >= 0.0))

    def test_front_stays_inside_the_domain_and_moves(self) -> None:
        ref = _cheap_reference()
        arrival = ref.arrival_s  # pyright: ignore[reportAttributeAccessIssue]
        assert bool(np.isfinite(arrival).any())
        assert not bool(np.isfinite(arrival).all())  # the far field is still dry

    def test_later_sample_is_wetter_than_the_earlier_one(self) -> None:
        ref = _cheap_reference()
        assert float(ref.h_m[1].sum()) > float(ref.h_m[0].sum())  # pyright: ignore[reportAttributeAccessIssue]

    def test_grid_convergence(self) -> None:
        # A wetting front is a first-order-converging quantity, so refinement
        # shifts arrival by a few percent rather than pinning it. Measured 4-5%
        # between these two levels; the harness runs a much finer reference.
        coarse = _cheap_reference(n_cells=200)
        fine = _cheap_reference(n_cells=400)
        r = np.array([50.0, 100.0], dtype=np.float64)
        assert np.allclose(
            coarse.arrival_at(r),  # pyright: ignore[reportAttributeAccessIssue]
            fine.arrival_at(r),  # pyright: ignore[reportAttributeAccessIssue]
            rtol=0.08,
        )

    def test_deterministic(self) -> None:
        assert np.array_equal(
            _cheap_reference().h_m,  # pyright: ignore[reportAttributeAccessIssue]
            _cheap_reference().h_m,  # pyright: ignore[reportAttributeAccessIssue]
        )

    def test_rejects_unsorted_sample_times(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            solve_radial_inflow(
                [200.0, 100.0],
                hydrograph_t_s=np.array([0.0, 600.0]),
                hydrograph_q_m3s=np.array([40.0, 40.0]),
                source_radius_m=10.0,
                manning_n=0.05,
                g=9.81,
                r_max=400.0,
                n_cells=100,
            )

    def test_rejects_a_source_larger_than_the_domain(self) -> None:
        with pytest.raises(ValueError, match="source_radius_m"):
            solve_radial_inflow(
                [100.0],
                hydrograph_t_s=np.array([0.0, 600.0]),
                hydrograph_q_m3s=np.array([40.0, 40.0]),
                source_radius_m=500.0,
                manning_n=0.05,
                g=9.81,
                r_max=400.0,
                n_cells=100,
            )


class TestMaxRelError:
    def test_returns_a_scalar_not_an_array(self) -> None:
        # Guards the arrival-error metric: a misplaced float() around the array
        # instead of the reduction silently produced a TypeError at run time.
        err = _max_rel_error(np.array([11.0, 18.0]), np.array([10.0, 20.0]))
        assert isinstance(err, float)
        assert err == pytest.approx(0.1)

    def test_ignores_entries_the_front_never_reached(self) -> None:
        actual = np.array([11.0, np.inf])
        reference = np.array([10.0, 500.0])
        assert _max_rel_error(actual, reference) == pytest.approx(0.1)

    def test_infinite_when_nothing_is_comparable(self) -> None:
        assert _max_rel_error(np.array([np.inf]), np.array([10.0])) == float("inf")
        assert _max_rel_error(np.array([1.0]), np.array([0.0])) == float("inf")


class TestRadialWindow:
    def test_probe_default_is_inside_the_validity_window(self) -> None:
        assert _spec().probe_time_s <= RADIAL_VALID_UNTIL_S

    def test_window_ends_before_the_front_reaches_a_wall(self) -> None:
        # All three nearest walls are 1000 m from the source; the documented
        # window must be shorter than the time the front needs to get there.
        assert RADIAL_VALID_UNTIL_S < 5.0 * 3600.0

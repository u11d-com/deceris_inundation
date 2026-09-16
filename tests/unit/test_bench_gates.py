"""Tests for the shared gate primitives in bench/common.py.

These are the pieces every benchmark harness repeats inside its
``_evaluate_case``, which takes a live ``SWEWorkflow`` and therefore only runs
behind a GPU. Keeping them pure is what makes the scoring logic testable — a
misplaced parenthesis in one of these reductions previously shipped as a
run-time TypeError in EA Test 4's headline metric.
"""

from __future__ import annotations

import numpy as np
import pytest

# pytest.approx is typed only loosely in pytest 9.0.2 (untyped expected/rel/abs
# params), so every call site surfaces as "partially unknown" under strict.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
from inundation.bench.common import (
    check_state_health,
    l1_relative_error,
    max_relative_error,
    nearest_cell_indices,
    observed_order,
    volume_drift_rel,
)


class TestNearestCellIndices:
    def test_picks_the_closest_centroid(self) -> None:
        cx = np.array([0.0, 10.0, 20.0])
        cy = np.array([0.0, 0.0, 0.0])
        idx = nearest_cell_indices(cx, cy, np.array([[9.0, 0.0], [19.9, 0.0]]))
        assert idx.tolist() == [1, 2]

    def test_uses_both_axes(self) -> None:
        cx = np.array([0.0, 0.0])
        cy = np.array([0.0, 10.0])
        idx = nearest_cell_indices(cx, cy, np.array([[0.0, 9.0]]))
        assert idx.tolist() == [1]

    def test_returns_integer_indices(self) -> None:
        idx = nearest_cell_indices(np.array([0.0]), np.array([0.0]), np.array([[1.0, 1.0]]))
        assert idx.dtype == np.int64

    def test_rejects_mismatched_centroid_arrays(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            nearest_cell_indices(np.array([0.0, 1.0]), np.array([0.0]), np.array([[0.0, 0.0]]))

    def test_rejects_an_empty_mesh(self) -> None:
        with pytest.raises(ValueError, match="no cell centroids"):
            nearest_cell_indices(np.array([]), np.array([]), np.array([[0.0, 0.0]]))


class TestVolumeDriftRel:
    def test_relative_to_the_reference(self) -> None:
        assert volume_drift_rel(101.0, 100.0) == pytest.approx(0.01)

    def test_is_unsigned(self) -> None:
        assert volume_drift_rel(99.0, 100.0) == pytest.approx(0.01)

    def test_zero_reference_fails_rather_than_dividing(self) -> None:
        # A case that injected nothing must fail its mass gate, not pass it.
        assert volume_drift_rel(0.0, 0.0) == float("inf")
        assert volume_drift_rel(5.0, -1.0) == float("inf")


class TestCheckStateHealth:
    def test_clean_field_reports_no_reasons(self) -> None:
        finite, min_depth, reasons = check_state_health(np.array([0.0, 1.0, 2.0]))
        assert finite
        assert min_depth == 0.0
        assert reasons == []

    def test_flags_non_finite(self) -> None:
        finite, _min_depth, reasons = check_state_health(np.array([1.0, np.nan]))
        assert not finite
        assert "h_final_non_finite" in reasons

    def test_flags_negative_depth(self) -> None:
        _finite, min_depth, reasons = check_state_health(np.array([-1e-3, 1.0]))
        assert min_depth == pytest.approx(-1e-3)
        assert any(r.startswith("negative_depth") for r in reasons)

    def test_tolerance_admits_round_off_negatives(self) -> None:
        # Wetting/drying cases clamp at zero and can leave round-off below it.
        _finite, _min_depth, reasons = check_state_health(
            np.array([-1e-9, 1.0]), min_depth_tol=-1e-6
        )
        assert reasons == []

    def test_empty_field_is_vacuously_healthy(self) -> None:
        finite, min_depth, reasons = check_state_health(np.array([]))
        assert finite
        assert min_depth == 0.0
        assert reasons == []


class TestL1RelativeError:
    def test_normalises_by_the_reference(self) -> None:
        actual = np.array([1.1, 1.9])
        reference = np.array([1.0, 2.0])
        assert l1_relative_error(actual, reference) == pytest.approx(0.2 / 3.0)

    def test_exact_match_is_zero(self) -> None:
        reference = np.array([1.0, 2.0, 3.0])
        assert l1_relative_error(reference, reference) == 0.0

    def test_zero_reference_fails_rather_than_dividing(self) -> None:
        assert l1_relative_error(np.array([1.0]), np.array([0.0])) == float("inf")

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            l1_relative_error(np.array([1.0, 2.0]), np.array([1.0]))


class TestMaxRelativeError:
    def test_returns_a_scalar_worst_case(self) -> None:
        err = max_relative_error(np.array([11.0, 18.0]), np.array([10.0, 20.0]))
        assert isinstance(err, float)
        assert err == pytest.approx(0.1)

    def test_skips_entries_never_reached(self) -> None:
        # An unreached gauge is inf; it must not poison the reduction.
        err = max_relative_error(np.array([11.0, np.inf]), np.array([10.0, 500.0]))
        assert err == pytest.approx(0.1)

    def test_infinite_when_nothing_is_comparable(self) -> None:
        assert max_relative_error(np.array([np.inf]), np.array([10.0])) == float("inf")
        assert max_relative_error(np.array([1.0]), np.array([0.0])) == float("inf")

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            max_relative_error(np.array([1.0, 2.0]), np.array([1.0]))


class TestObservedOrder:
    def test_recovers_a_clean_first_order_sequence(self) -> None:
        dx = [8.0, 4.0, 2.0, 1.0]
        fit = observed_order(dx, [0.5 * h for h in dx])
        assert fit.order == pytest.approx(1.0)
        assert fit.pairwise_orders == pytest.approx([1.0, 1.0, 1.0])
        assert fit.monotone

    def test_recovers_second_order(self) -> None:
        dx = [4.0, 2.0, 1.0]
        fit = observed_order(dx, [h**2 for h in dx])
        assert fit.order == pytest.approx(2.0)

    def test_pairwise_orders_expose_a_flattening_curve(self) -> None:
        # First order down to a floor: the fitted slope averages the two regimes
        # away, the pairwise sequence is what shows the knee.
        fit = observed_order([8.0, 4.0, 2.0, 1.0], [8e-3, 4e-3, 2.05e-3, 2.0e-3])
        assert fit.pairwise_orders[0] == pytest.approx(1.0, abs=0.01)
        assert fit.pairwise_orders[-1] < 0.1
        assert fit.monotone

    def test_a_rising_error_is_reported_as_non_monotone(self) -> None:
        # The float32-floor signature: refinement makes it worse.
        fit = observed_order([8.0, 4.0, 2.0], [1e-5, 4e-5, 8e-4])
        assert not fit.monotone
        assert fit.order < 0.0

    def test_rejects_a_single_level(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            observed_order([1.0], [1.0])

    def test_rejects_dx_not_ordered_coarse_to_fine(self) -> None:
        with pytest.raises(ValueError, match="strictly decreasing"):
            observed_order([1.0, 2.0], [1.0, 2.0])

    def test_rejects_non_positive_errors(self) -> None:
        # A zero error has no logarithm; the caller must skip that metric.
        with pytest.raises(ValueError, match="finite and positive"):
            observed_order([2.0, 1.0], [1e-3, 0.0])

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            observed_order([4.0, 2.0, 1.0], [1.0, 2.0])

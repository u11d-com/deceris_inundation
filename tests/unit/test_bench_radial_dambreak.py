"""Tests for the radial (circular) dam-break reference in bench/common.py.

Pure-NumPy validation of the fine-grid axisymmetric finite-volume solver
that serves as the quasi-exact reference for the 2D circular dam-break — no
GPU, no solver run. The solver-facing gates live in bench/radial_dambreak.py
(manual, GPU-required).
"""

# pyright: reportUnknownMemberType=false
# pytest.approx's own stub signature is partially generic/unresolved
# (`expected: Unknown`) regardless of caller annotations — a pytest
# packaging gap, not fixable from this file.

from __future__ import annotations

import numpy as np
import pytest

from deceris.inundation.bench.common import (
    radial_dambreak_reference,
    solve_radial_dambreak,
)

G = 9.81
H_IN = 2.5
H_OUT = 0.5
R_DAM = 2.5
R_MAX = 20.0
T = 1.5


def _annular_volume(r_c: np.ndarray, h: np.ndarray) -> float:
    """2*pi * integral r h dr over the fine radial grid."""
    dr = float(r_c[1] - r_c[0])
    return float(2.0 * np.pi * np.sum(r_c * h) * dr)


class TestRadialSolve:
    def test_positive_and_finite(self) -> None:
        _r, h, u = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        assert np.all(h >= 0.0)
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(u))

    def test_conserves_annular_volume(self) -> None:
        # r-weighted FV form conserves annular mass telescopically; the wave
        # never reaches r_max, so the transmissive outer flux is inert.
        r_c, h, _u = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        h0 = np.where(r_c < R_DAM, H_IN, H_OUT)
        v0 = _annular_volume(r_c, h0)
        vt = _annular_volume(r_c, h)
        assert vt == pytest.approx(v0, rel=1e-4)

    def test_outer_region_undisturbed(self) -> None:
        r_c, h, _u = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        far = r_c > 0.9 * R_MAX
        assert np.allclose(h[far], H_OUT, atol=1e-6)

    def test_inner_rarefaction_lowers_center_depth(self) -> None:
        r_c, h, _u = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        center = h[r_c < 0.5]
        assert float(center.mean()) < H_IN

    def test_bore_moved_outward_but_inside_domain(self) -> None:
        r_c, h, _u = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        threshold = 1.2 * H_OUT
        wet = r_c[h > threshold]
        front = float(wet.max())
        assert front > R_DAM
        assert front < R_MAX

    def test_front_speed_below_physical_bound(self) -> None:
        # The bore cannot outrun the fastest characteristic of the reservoir.
        r_c, h, _u = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        front = float(r_c[h > 1.2 * H_OUT].max())
        max_speed = 2.0 * np.sqrt(G * H_IN)
        assert (front - R_DAM) / T < max_speed

    def test_grid_convergence_front(self) -> None:
        # Reference front is grid-independent to ~1% between 1000 and 2000 cells.
        r1, h1, _ = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        r2, h2, _ = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=2000
        )
        front1 = float(r1[h1 > 1.2 * H_OUT].max())
        front2 = float(r2[h2 > 1.2 * H_OUT].max())
        assert abs(front1 - front2) / front2 < 0.01

    def test_deterministic(self) -> None:
        a = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=500
        )
        b = solve_radial_dambreak(
            T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=500
        )
        assert np.array_equal(a[1], b[1])

    def test_rejects_nonpositive_time(self) -> None:
        with pytest.raises(ValueError, match="t must be positive"):
            solve_radial_dambreak(0.0, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX)

    def test_rejects_bad_radius(self) -> None:
        with pytest.raises(ValueError, match="need 0 < r_dam < r_max"):
            solve_radial_dambreak(T, h_in=H_IN, h_out=H_OUT, r_dam=R_MAX, g=G, r_max=R_MAX)


class TestRadialReferenceWrapper:
    def test_interpolates_onto_eval_radii(self) -> None:
        r_eval = np.linspace(0.1, R_MAX - 0.1, 100)
        h, u = radial_dambreak_reference(
            r_eval, T, h_in=H_IN, h_out=H_OUT, r_dam=R_DAM, g=G, r_max=R_MAX, n_cells=1000
        )
        assert h.shape == r_eval.shape
        assert u.shape == r_eval.shape
        assert np.all(h >= 0.0)
        # Outer sampled radii are still at the undisturbed layer.
        assert h[-1] == pytest.approx(H_OUT, abs=1e-3)

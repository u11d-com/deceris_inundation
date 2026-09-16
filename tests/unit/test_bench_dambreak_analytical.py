"""Tests for the Tier 1 analytical dam-break helpers in bench/common.py.

Pure-NumPy validation of the Ritter/Stoker closed-form solutions and the
synthetic channel-mesh builder — no GPU, no solver run. The solver-facing
gates live in bench/dambreak.py (manual, GPU-required).
"""

# pyright: reportUnknownMemberType=false
# pytest.approx's own stub signature is partially generic/unresolved
# (`expected: Unknown`) regardless of caller annotations — a pytest
# packaging gap, not fixable from this file (same note as
# test_inundation_bringup_dtslack.py).

from __future__ import annotations

import numpy as np
import pytest

from inundation.bench.common import (
    build_channel_mesh,
    ritter_solution,
    stoker_middle_depth,
    stoker_solution,
)

G = 9.81
H_UP = 10.0
H_DOWN = 1.0
DAM_X = 500.0
T = 20.0

X = np.linspace(0.0, 1000.0, 2001)


class TestRitter:
    def test_upstream_undisturbed(self) -> None:
        h, u = ritter_solution(X, T, h_up=H_UP, dam_x=DAM_X, g=G)
        c0 = np.sqrt(G * H_UP)
        upstream = X < DAM_X - c0 * T - 1.0
        assert np.allclose(h[upstream], H_UP)
        assert np.allclose(u[upstream], 0.0)

    def test_dry_beyond_front(self) -> None:
        h, u = ritter_solution(X, T, h_up=H_UP, dam_x=DAM_X, g=G)
        c0 = np.sqrt(G * H_UP)
        beyond = X > DAM_X + 2.0 * c0 * T + 1.0
        assert np.all(h[beyond] == 0.0)
        assert np.all(u[beyond] == 0.0)

    def test_depth_at_dam_is_4_9_h0(self) -> None:
        # Classical Ritter property: h(dam_x, t) = (4/9) h0 for all t > 0.
        h, _u = ritter_solution(np.array([DAM_X]), T, h_up=H_UP, dam_x=DAM_X, g=G)
        assert h[0] == pytest.approx(4.0 / 9.0 * H_UP, rel=1e-12)

    def test_profile_continuous_and_monotonic(self) -> None:
        h, _u = ritter_solution(X, T, h_up=H_UP, dam_x=DAM_X, g=G)
        assert np.all(np.diff(h) <= 1e-9)
        assert np.all(np.abs(np.diff(h)) < 0.1)  # no jumps at region seams

    def test_rejects_nonpositive_time(self) -> None:
        with pytest.raises(ValueError, match="t must be positive"):
            ritter_solution(X, 0.0, h_up=H_UP, dam_x=DAM_X, g=G)


class TestStoker:
    def test_middle_depth_bracketed(self) -> None:
        hm = stoker_middle_depth(H_UP, H_DOWN, G)
        assert H_DOWN < hm < H_UP

    def test_middle_depth_satisfies_matching_condition(self) -> None:
        hm = stoker_middle_depth(H_UP, H_DOWN, G)
        c0 = np.sqrt(G * H_UP)
        cm = np.sqrt(G * hm)
        u_rarefaction = 2.0 * (c0 - cm)
        u_shock = (hm - H_DOWN) * np.sqrt(G * (hm + H_DOWN) / (2.0 * hm * H_DOWN))
        assert u_rarefaction == pytest.approx(u_shock, rel=1e-9)

    def test_middle_depth_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="need 0 < h_down < h_up"):
            stoker_middle_depth(H_UP, 0.0, G)
        with pytest.raises(ValueError, match="need 0 < h_down < h_up"):
            stoker_middle_depth(H_UP, H_UP, G)

    def test_regions(self) -> None:
        h, u = stoker_solution(X, T, h_up=H_UP, h_down=H_DOWN, dam_x=DAM_X, g=G)
        c0 = np.sqrt(G * H_UP)
        hm = stoker_middle_depth(H_UP, H_DOWN, G)
        cm = np.sqrt(G * hm)
        um = 2.0 * (c0 - cm)
        s = um * hm / (hm - H_DOWN)

        upstream = X < DAM_X - c0 * T - 1.0
        assert np.allclose(h[upstream], H_UP)
        plateau = (X > DAM_X + (um - cm) * T + 1.0) & (X < DAM_X + s * T - 1.0)
        assert np.allclose(h[plateau], hm)
        assert np.allclose(u[plateau], um)
        downstream = X > DAM_X + s * T + 1.0
        assert np.allclose(h[downstream], H_DOWN)
        assert np.allclose(u[downstream], 0.0)

    def test_shock_satisfies_rankine_hugoniot(self) -> None:
        hm = stoker_middle_depth(H_UP, H_DOWN, G)
        cm = np.sqrt(G * hm)
        um = 2.0 * (np.sqrt(G * H_UP) - cm)
        s = um * hm / (hm - H_DOWN)
        # Mass: s (hm - hr) = hm um - hr*0
        assert s * (hm - H_DOWN) == pytest.approx(hm * um, rel=1e-9)
        # Momentum: s (hm um - 0) = hm um^2 + g/2 (hm^2 - hr^2)
        lhs = s * hm * um
        rhs = hm * um**2 + 0.5 * G * (hm**2 - H_DOWN**2)
        assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_approaches_ritter_as_h_down_vanishes(self) -> None:
        # hm scales ~ h_down^(1/3): 1e-12 downstream depth puts the residual
        # plateau at ~4e-4 m, below the 1e-3 comparison tolerance.
        h_stoker, _ = stoker_solution(X, T, h_up=H_UP, h_down=1e-12, dam_x=DAM_X, g=G)
        h_ritter, _ = ritter_solution(X, T, h_up=H_UP, dam_x=DAM_X, g=G)
        assert float(np.max(np.abs(h_stoker - h_ritter))) < 1e-3


class TestChannelMesh:
    def test_shapes_and_extent(self) -> None:
        nx, ny, length, width = 50, 4, 1000.0, 10.0
        verts, quads = build_channel_mesh(nx, ny, length, width)
        assert verts.shape == ((nx + 1) * (ny + 1), 2)
        assert quads.shape == (nx * ny, 4)
        assert verts[:, 0].min() == 0.0
        assert verts[:, 0].max() == pytest.approx(length)
        assert verts[:, 1].max() == pytest.approx(width)

    def test_quads_are_ccw_with_uniform_area(self) -> None:
        nx, ny, length, width = 20, 3, 200.0, 6.0
        verts, quads = build_channel_mesh(nx, ny, length, width)
        expected_area = (length / nx) * (width / ny)
        pts = verts[quads]  # (N, 4, 2)
        x, y = pts[..., 0].astype(np.float64), pts[..., 1].astype(np.float64)
        signed = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        assert np.all(signed > 0.0)
        assert np.allclose(signed, expected_area, rtol=1e-4)

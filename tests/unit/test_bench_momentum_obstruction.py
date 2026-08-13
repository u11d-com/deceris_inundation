"""Tests for the momentum-obstruction benchmark (EA Test 3).

Pure-NumPy validation of the synthetic prismatic bed builder and the
elevated-release initial condition — no GPU, no solver run. The solver-facing
invariant/qualitative gates live in bench/momentum_obstruction.py (manual,
GPU-required).
"""

# pyright: reportPrivateUsage=false
# The harness's release helpers are module-private (leading underscore);
# exercising them here is intentional, so silence the private-usage warning.

from __future__ import annotations

import numpy as np

from deceris.inundation.bench.common import sloping_obstruction_bed
from deceris.inundation.bench.momentum_obstruction import (
    BOWL_FLOOR_Z_M,
    CONTROL_X,
    CONTROL_Z,
    CREST_X_M,
    POINT1_X_M,
    POINT2_X_M,
    RESERVOIR_X0_M,
    RESERVOIR_X1_M,
    SHELF_Z_M,
    SILL_Z_M,
    ObstructionSpec,
    _bed_original,
    _control_initial_state,
    _control_wse,
    _initial_state,
    _original_cell_centroids,
    _release_mask,
    _released_volume,
)


def _spec(**over: object) -> ObstructionSpec:
    base: dict[str, object] = {
        "name": "obstruction",
        "nx": 150,
        "ny": 12,
        "channel_len_m": 300.0,
        "channel_width_m": 60.0,
        "manning_n": 0.03,
        "release_depth_m": 0.8,
        "t_end_s": 900.0,
        "dt_max": 2.0,
        "cfl_interval": 10,
    }
    base.update(over)
    return ObstructionSpec(**base)  # pyright: ignore[reportArgumentType]


class TestObstructionBed:
    def test_dtype_and_finite(self) -> None:
        cx = np.array([10.0, 127.0, 190.0, 250.0, 290.0])
        zb = sloping_obstruction_bed(cx, control_x=CONTROL_X, control_z=CONTROL_Z)
        assert zb.dtype == np.float32
        assert np.all(np.isfinite(zb))

    def test_control_points_reproduced(self) -> None:
        # Sampling exactly at the control x's returns the control z's.
        zb = sloping_obstruction_bed(
            np.asarray(CONTROL_X), control_x=CONTROL_X, control_z=CONTROL_Z
        )
        assert np.allclose(zb, np.asarray(CONTROL_Z, dtype=np.float32))

    def test_bowls_below_sill_below_shelf(self) -> None:
        # Ordering the momentum trap depends on: bowl floors < sill < shelf.
        zb = sloping_obstruction_bed(
            np.array([POINT1_X_M, CREST_X_M, POINT2_X_M, 10.0]),
            control_x=CONTROL_X,
            control_z=CONTROL_Z,
        )
        valley, sill, far_bowl, shelf = (float(v) for v in zb)
        assert valley == np.float32(BOWL_FLOOR_Z_M)
        assert far_bowl == np.float32(BOWL_FLOOR_Z_M)
        assert sill == np.float32(SILL_Z_M)
        assert shelf == np.float32(SHELF_Z_M)
        assert valley < sill < shelf

    def test_flat_bowl_floors(self) -> None:
        # The valley and far-bowl floors are genuinely flat.
        valley = sloping_obstruction_bed(
            np.array([100.0, 127.0, 155.0]), control_x=CONTROL_X, control_z=CONTROL_Z
        )
        far = sloping_obstruction_bed(
            np.array([225.0, 250.0, 295.0]), control_x=CONTROL_X, control_z=CONTROL_Z
        )
        assert np.allclose(valley, np.float32(BOWL_FLOOR_Z_M))
        assert np.allclose(far, np.float32(BOWL_FLOOR_Z_M))

    def test_flat_sill_top(self) -> None:
        # The sill top is flat (the obstruction crest, gauge sits on it).
        sill = sloping_obstruction_bed(
            np.array([176.0, 190.0, 204.0]), control_x=CONTROL_X, control_z=CONTROL_Z
        )
        assert np.allclose(sill, np.float32(SILL_Z_M))

    def test_shelf_is_flat_under_release_block(self) -> None:
        # The release block must stand on a level shelf.
        xs = np.linspace(RESERVOIR_X0_M + 1.0, RESERVOIR_X1_M - 1.0, 7)
        shelf = sloping_obstruction_bed(xs, control_x=CONTROL_X, control_z=CONTROL_Z)
        assert np.allclose(shelf, np.float32(SHELF_Z_M))

    def test_rejects_non_increasing_control_x(self) -> None:
        try:
            sloping_obstruction_bed(
                np.array([1.0]), control_x=(0.0, 0.0, 1.0), control_z=(0.0, 1.0, 2.0)
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-increasing control_x")


class TestRelease:
    def test_release_volume_matches_block(self) -> None:
        spec = _spec()
        expected = (RESERVOIR_X1_M - RESERVOIR_X0_M) * spec.channel_width_m * spec.release_depth_m
        assert abs(_released_volume(spec) - expected) / expected < 1e-12

    def test_release_block_is_still_and_on_shelf_only(self) -> None:
        spec = _spec()
        state = _initial_state(spec)
        h = np.asarray(state["h"], dtype=np.float32)
        hu = np.asarray(state["hu"], dtype=np.float32)
        hv = np.asarray(state["hv"], dtype=np.float32)
        mask = _release_mask(spec)
        assert np.all(h[mask] == np.float32(spec.release_depth_m))
        assert np.all(h[~mask] == 0.0)
        assert not hu.any()
        assert not hv.any()

    def test_release_mask_covers_full_width(self) -> None:
        # Prismatic release: every row of cells under the block is wet.
        spec = _spec()
        mask = _release_mask(spec).reshape(spec.ny, spec.nx)
        cols = mask.any(axis=0)
        assert np.array_equal(mask, np.tile(cols, (spec.ny, 1)))
        n_cols = round((RESERVOIR_X1_M - RESERVOIR_X0_M) / spec.dx_m)
        assert int(cols.sum()) == n_cols

    def test_static_ceiling_below_crest(self) -> None:
        # The discriminator: even if the whole release ponded in the valley,
        # the static surface would stay well below the sill crest. Integrate
        # the valley's hypsometry numerically from the same bed the solver
        # sees and require >= 0.15 m of freeboard at the all-in-valley level.
        spec = _spec()
        xs = np.linspace(0.0, spec.channel_len_m, 30001)
        zb = sloping_obstruction_bed(xs, control_x=CONTROL_X, control_z=CONTROL_Z).astype(
            np.float64
        )
        dx = float(xs[1] - xs[0])
        # Valley = the connected region below the crest, left of the sill top.
        valley = (zb < SILL_Z_M) & (xs < CREST_X_M)

        def valley_capacity(w: float) -> float:
            depth = np.clip(w - zb[valley], 0.0, None)
            return float(depth.sum() * dx * spec.channel_width_m)

        released = _released_volume(spec)
        ceiling = SILL_Z_M - 0.15
        assert valley_capacity(ceiling) > released


class TestStillWaterControl:
    def test_control_wse_below_crest(self) -> None:
        # The inertia-free endpoint (whole release at rest in the valley)
        # must sit clearly below the crest, or the case discriminates nothing.
        wse = _control_wse(_spec())
        assert wse < SILL_Z_M - 0.15

    def test_control_volume_matches_release(self) -> None:
        spec = _spec()
        state = _control_initial_state(spec)
        h = np.asarray(state["h"], dtype=np.float64)
        volume = float(h.sum()) * spec.dx_m * spec.dy_m
        released = _released_volume(spec)
        assert abs(volume - released) / released < 1e-3

    def test_control_is_still_level_valley_pond(self) -> None:
        spec = _spec()
        state = _control_initial_state(spec)
        h = np.asarray(state["h"], dtype=np.float64)
        hu = np.asarray(state["hu"], dtype=np.float32)
        hv = np.asarray(state["hv"], dtype=np.float32)
        assert not hu.any()
        assert not hv.any()
        cx, _cy = _original_cell_centroids(spec)
        # Use the harness bed (smoothed) so WSE matches how the state was built.
        zb = _bed_original(spec)
        wet = h > 0.0
        # Water only in the valley (left of the sill top, below the crest).
        assert np.all(cx[wet] < CREST_X_M)
        assert np.all(zb[wet] < SILL_Z_M)
        # Level surface: h + zb is constant across wet cells.
        wse = h[wet] + zb[wet]
        assert float(wse.max() - wse.min()) < 1e-6

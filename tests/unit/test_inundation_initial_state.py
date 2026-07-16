"""Tests for inundation initial-state preparation — solver-side loader only.

Extracted from the api repo's ``test_inundation_initial_state.py``; the
``deceris.tools.inundation_tool`` cases stay in the api repo because they
depend on API-side pipeline glue. Only the ``_load_initial_state`` white-box
tests live here — they exercise ``deceris.inundation.solver_workflow``.
"""

# pyright: reportPrivateUsage=false
# Deliberate white-box test of solver_workflow._load_initial_state (private).

from __future__ import annotations

import numpy as np
import pytest


def test_load_initial_state_ignores_saved_order_metadata() -> None:
    pytest.importorskip("hilbertcurve")

    from deceris.inundation.solver_workflow import _load_initial_state

    state: dict[str, object] = {
        "h": np.asarray([1.0, 0.0], dtype=np.float32),
        "hu": np.asarray([2.0, 3.0], dtype=np.float32),
        "hv": np.asarray([4.0, 5.0], dtype=np.float32),
        "saved_order": np.asarray("original"),
    }

    loaded = _load_initial_state(
        state,
        2,
        np.asarray([0.0, 0.0], dtype=np.float32),
        order="solver",
        id_col="cell_id",
        perm=np.asarray([0, 1], dtype=np.int32),
        dry_tol=1e-4,
    )

    assert loaded is not None
    h, hu, hv, n_mann = loaded
    assert h.tolist() == [1.0, 0.0]
    assert hu.tolist() == [2.0, 0.0]
    assert hv.tolist() == [4.0, 0.0]
    assert n_mann is None


def test_load_initial_state_ignores_benchmark_metadata_fields() -> None:
    """Benchmark metadata fields must be ignored rather than rejected."""
    pytest.importorskip("hilbertcurve")

    from deceris.inundation.solver_workflow import _load_initial_state

    state: dict[str, object] = {
        "h": np.asarray([1.0, 0.0], dtype=np.float32),
        "hu": np.asarray([2.0, 3.0], dtype=np.float32),
        "hv": np.asarray([4.0, 5.0], dtype=np.float32),
        "saved_order": np.asarray("solver"),
        "mesh_source": np.asarray("powiat-klodzki-mesh-5m.parquet"),
        "t_end_s": np.float64(259200),
        "dt_max": np.float64(0.25),
        "cfl_interval": np.int64(2000),
    }

    loaded = _load_initial_state(
        state,
        2,
        np.asarray([0.0, 0.0], dtype=np.float32),
        order="solver",
        id_col="cell_id",
        perm=np.asarray([0, 1], dtype=np.int32),
        dry_tol=1e-4,
    )

    assert loaded is not None
    h, hu, hv, n_mann = loaded
    assert h.tolist() == [1.0, 0.0]
    assert hu.tolist() == [2.0, 0.0]
    assert hv.tolist() == [4.0, 0.0]
    assert n_mann is None

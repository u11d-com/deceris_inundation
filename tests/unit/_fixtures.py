"""Shared test constants (migrated from swe_tuning.py self-check thresholds)."""

from pathlib import Path

import pytest

TEST_MESH_SMALL_FOOTPRINT = 100
TEST_MESH_LARGE_FOOTPRINT = 10_000
TEST_GPU_FREE_BYTES = 1000

# The published EA benchmark inputs needed by the loader tests are tracked in
# benchmark_assets/. Tests still skip when a partial source distribution omits it.
EA_DATASET_ROOT = Path(__file__).resolve().parents[2] / "benchmark_assets"

requires_ea_dataset = pytest.mark.skipif(
    not EA_DATASET_ROOT.is_dir(),
    reason=f"EA benchmark dataset not present at {EA_DATASET_ROOT}; unpack it to run this test",
)

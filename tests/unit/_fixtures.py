"""Shared test constants (migrated from swe_tuning.py self-check thresholds)."""

from pathlib import Path

import pytest

TEST_MESH_SMALL_FOOTPRINT = 100
TEST_MESH_LARGE_FOOTPRINT = 10_000
TEST_GPU_FREE_BYTES = 1000

# The published EA benchmark datasets are not redistributed with the repo (see
# .gitignore), so the tests that read them skip rather than fail on a clean
# checkout. Everything that can be exercised without the data still runs.
EA_DATASET_ROOT = Path(__file__).resolve().parents[2] / "Benchmarking_Model_Data"

requires_ea_dataset = pytest.mark.skipif(
    not EA_DATASET_ROOT.is_dir(),
    reason=f"EA benchmark dataset not present at {EA_DATASET_ROOT}; unpack it to run this test",
)

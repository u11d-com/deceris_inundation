"""Baseline SWE solver alias for benchmark comparisons."""

from __future__ import annotations

from .swe_gpu import SWESolver as SWESolverBaseline

__all__ = ["SWESolverBaseline"]

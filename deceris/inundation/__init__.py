"""Inundation tool — GPU-accelerated shallow water equation flood simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver_workflow import (
        PointSource,
        SimulationPhase,
        SWEWorkflow,
        WorkflowConfig,
        WorkflowResult,
    )

__all__ = [
    "PointSource",
    "SWEWorkflow",
    "SimulationPhase",
    "WorkflowConfig",
    "WorkflowResult",
]

_SOLVER_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _SOLVER_EXPORTS:
        from .solver_workflow import (  # pyright: ignore[reportMissingModuleSource]
            PointSource,
            SimulationPhase,
            SWEWorkflow,
            WorkflowConfig,
            WorkflowResult,
        )

        return {
            "PointSource": PointSource,
            "SimulationPhase": SimulationPhase,
            "SWEWorkflow": SWEWorkflow,
            "WorkflowConfig": WorkflowConfig,
            "WorkflowResult": WorkflowResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

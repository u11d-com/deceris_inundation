"""Vulkan/Kompute solver backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver import SWESolver

__all__ = ["SWESolver"]

_VULKAN_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _VULKAN_EXPORTS:
        from .solver import SWESolver  # pyright: ignore[reportMissingModuleSource]

        return {"SWESolver": SWESolver}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

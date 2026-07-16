"""GLSL compute shader compilation helpers for the SWE solver."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from .swe_shaders_cfl import (
    CFL_ACCUM_GLSL,
    CFL_REDUCE_GLSL,
    CFL_RESOLVE_GLSL,
    CFL_RESOLVE_TIME_GLSL,
    DT_RESET_GLSL,
    TIME_ADVANCE_GLSL,
)
from .swe_shaders_flux import FLUX_DTBUF_GLSL, FLUX_GLSL
from .swe_shaders_source import SOURCE_DTBUF_GLSL, SOURCE_GLSL
from .swe_shaders_update import UPDATE_DTBUF_GLSL, UPDATE_GLSL


def compile_glsl(glsl_src: str, label: str) -> bytes:
    """Compile GLSL compute shader source to SPIR-V bytes."""
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        raise RuntimeError("glslangValidator not found. Install with: brew install glslang")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{label}.comp")
        spv_path = os.path.join(tmpdir, f"{label}.spv")

        with open(src_path, "w") as fh:
            fh.write(glsl_src)

        result = subprocess.run(  # noqa: S603
            [glslang, "--target-env", "vulkan1.0", "-V", "-o", spv_path, src_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compiler failed for {label}:\n{result.stdout}\n{result.stderr}")

        with open(spv_path, "rb") as fh:
            return fh.read()


def compile_all() -> dict[str, bytes]:
    """Compile all SWE compute shaders and return label->SPIR-V map."""
    return {
        "flux": compile_glsl(FLUX_GLSL, "flux_kernel"),
        "flux_dtbuf": compile_glsl(FLUX_DTBUF_GLSL, "flux_dtbuf_kernel"),
        "update": compile_glsl(UPDATE_GLSL, "update_kernel"),
        "update_dtbuf": compile_glsl(UPDATE_DTBUF_GLSL, "update_dtbuf_kernel"),
        "cfl_accum": compile_glsl(CFL_ACCUM_GLSL, "cfl_accum_kernel"),
        "cfl_reduce": compile_glsl(CFL_REDUCE_GLSL, "cfl_reduce_kernel"),
        "cfl_resolve": compile_glsl(CFL_RESOLVE_GLSL, "cfl_resolve_kernel"),
        "dt_reset": compile_glsl(DT_RESET_GLSL, "dt_reset_kernel"),
        "cfl_resolve_time": compile_glsl(CFL_RESOLVE_TIME_GLSL, "cfl_resolve_time_kernel"),
        "time_advance": compile_glsl(TIME_ADVANCE_GLSL, "time_advance_kernel"),
        "source_dtbuf": compile_glsl(SOURCE_DTBUF_GLSL, "source_dtbuf_kernel"),
    }


__all__ = [
    "CFL_ACCUM_GLSL",
    "CFL_REDUCE_GLSL",
    "CFL_RESOLVE_GLSL",
    "CFL_RESOLVE_TIME_GLSL",
    "DT_RESET_GLSL",
    "FLUX_DTBUF_GLSL",
    "FLUX_GLSL",
    "SOURCE_DTBUF_GLSL",
    "SOURCE_GLSL",
    "TIME_ADVANCE_GLSL",
    "UPDATE_DTBUF_GLSL",
    "UPDATE_GLSL",
    "compile_all",
    "compile_glsl",
]

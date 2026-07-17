# SWE Solver Integration Workflow

This document explains how to integrate the mesh-based shallow-water GPU solver
into an existing Python application using the workflow wrapper in solver_workflow.py.

## 0. Environment, Packages, and Vulkan Setup

### 0.1 Python environment

Recommended Python: 3.10 to 3.12.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install core solver packages:

```bash
pip install -r requirements.txt
```

Install optional GIS/export packages (only if you need .gpkg/.shp IO or raster overlays):

```bash
pip install -r requirements-gis.txt
```

### 0.2 System dependencies

The solver needs:

- Vulkan runtime + ICD
- glslangValidator (for GLSL -> SPIR-V compilation)
- A GPU/driver stack that supports compute

macOS (Apple Silicon/Homebrew):

```bash
brew install glslang molten-vk vulkan-loader vulkan-tools
```

Linux (Ubuntu/Debian example):

```bash
sudo apt update
sudo apt install -y glslang-tools vulkan-tools libvulkan1 mesa-vulkan-drivers
```

Windows:

1. Install Vulkan SDK from LunarG.
2. Ensure glslangValidator is on PATH.
3. Update GPU drivers (NVIDIA/AMD/Intel) to latest Vulkan-capable versions.

### 0.3 Vulkan runtime configuration for kp

Set VK_ICD_FILENAMES before importing kp. This is especially important on macOS.

```python
import os

# macOS/Homebrew MoltenVK default path
os.environ.setdefault(
  "VK_ICD_FILENAMES",
  "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
)

import kp
```

If Homebrew is installed under /usr/local, use:

- /usr/local/etc/vulkan/icd.d/MoltenVK_icd.json

### 0.4 Quick validation checklist

Run these checks once in your deployment environment:

```bash
which glslangValidator
glslangValidator --version
```

```bash
python -c "import os; os.environ.setdefault('VK_ICD_FILENAMES','/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json'); import kp; print('kp import OK')"
```

If these fail, fix runtime/driver/ICD before integrating the app workflow.

## 1. What Was Prepared

The new workflow module provides:

- Typed request model for hydrograph phases and point inflows.
- One-time setup path for mesh loading, geometry preprocessing, shader compilation,
  and GPU solver allocation.
- Runtime execution API for phased simulations.
- Output object with snapshots, times, and volume metrics.

Primary classes and functions:

- WorkflowConfig
- PointSource
- SimulationPhase
- SWEWorkflow.prepare()
- SWEWorkflow.run()

## 2. Integration Pattern

Use a two-stage lifecycle in your application:

1. Startup stage (once): create and prepare workflow.
2. Request stage (many times): build phases from request payload and call run().

This avoids recompiling shaders and reallocating tensors per request.

## 3. Minimal App Wiring

```python
from solver_workflow import (
    WorkflowConfig,
    SWEWorkflow,
    SimulationPhase,
    PointSource,
)

# 1) Startup initialization
workflow = SWEWorkflow(
    WorkflowConfig(
        mesh_source="../../mesh/mesh_triangles_z2.shp",
        output_every=8000,
        dt_max=0.05,
        cfl_interval=1,
        progress=False,
    )
)
workflow.prepare()

# 2) Per request
phases = [
    SimulationPhase(
        duration_s=6000.0,
        sources=[PointSource(40.0, (349304.72, 266737.27), 2.0)],
    ),
    SimulationPhase(
        duration_s=3000.0,
        sources=[PointSource(120.0, (349304.72, 266737.27), 2.0)],
    ),
]

result = workflow.run(phases)

print(result.h_final.max())
print(result.volume_final_m3, result.volume_injected_m3)
```

## 4. Request Mapping Contract

Recommended request payload shape in an existing API:

```json
{
  "phases": [
    {
      "duration_s": 6000,
      "sources": [
        {"discharge_m3s": 40, "center_xy": [349304.72, 266737.27], "radius_m": 2.0}
      ]
    }
  ]
}
```

Map each source item directly to PointSource, and each phase to SimulationPhase.

## 5. Runtime and Reliability Notes

- Mesh coordinates and source coordinates must be in the same CRS/units.
- Keep cfl_interval at 1 when using strong, time-varying inflows.
- If run() is called concurrently, protect solver access with a lock or create one
  workflow instance per worker, because GPU tensors are mutable shared state.
- For low-latency APIs, execute run() in a worker process/thread and stream status
  separately.

## 6. Result Handling

WorkflowResult includes:

- snapshots: water depth arrays recorded at configured output interval.
- snap_times: simulation time corresponding to each snapshot.
- h_final: final depth state.
- volume_final_m3: final water volume inside mesh.
- volume_injected_m3: total imposed inflow volume.
- wall_seconds: wall-clock compute time.

Typical downstream actions:

- Convert h_final to polygons for GIS export.
- Build time animations from snapshots.
- Use volume metrics for sanity checks and acceptance tests.

## 7. Suggested Production Hardening

1. Cache prepared workflow by mesh id to avoid repeated compile/setup.
2. Add request validation for empty phases, negative duration, and invalid radius.
3. Add timeout and cancellation support around run() in your job layer.
4. Persist major run metadata (phase schedule, cfl params, mesh id) for auditability.

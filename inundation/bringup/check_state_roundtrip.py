"""State round-trip self-check: initial momentum and warm-start equivalence.

Verifies two properties that the benchmark suite does not cover, because every
harness starts from rest and runs uninterrupted:

- **Initial momentum survives.** ``WorkflowConfig.initial_state_source``
  accepts ``hu``/``hv``; every solver's ``run(resume=False)`` used to reset
  them to zero, silently discarding the caller's momentum (fixed 2026-08-19,
  see ``docs/planning/decisions.md``). This is the regression guard.
- **Warm start reproduces an uninterrupted run.** A Go/No-Go criterion in
  ``docs/architecture/validation-plan.md`` §Go-No-Go: running 0 -> T must
  match running 0 -> T/2, exporting ``(h, hu, hv)``, and resuming to T.
  Only testable since the Vulkan solvers gained ``download_hu``/``download_hv``.

Not bit-exact by construction: restarting restarts the adaptive-dt sequence,
so the two runs take different steps. The comparison is tolerance-based.

Standalone (no pytest, per ``bringup/`` convention)::

    python check_state_roundtrip.py
    python check_state_roundtrip.py --backend gpu_resident_batch --t-end 600
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from inundation.bench.common import build_channel_mesh, write_mesh_parquet
from inundation.workflow import (
    PointSource,
    SimulationPhase,
    SWEWorkflow,
    WorkflowConfig,
)

NX, NY = 60, 30
LENGTH_M, WIDTH_M = 600.0, 300.0
MANNING_N = 0.05
GRAVITY = 9.81

# Momentum must survive the first step, not merely be readable afterwards.
GATE_MOMENTUM_DECAY_REL = 0.05
# Warm start vs uninterrupted, relative to peak depth. Restarting resets the
# adaptive-dt sequence, but the CFL is a deterministic function of the state,
# so the resumed run retraces it: measured 4.8e-7 on MoltenVK.
GATE_WARM_START_REL = 1e-5


def _mesh(tmp: Path) -> Path:
    path = tmp / f"flat-{NX}x{NY}.parquet"
    verts, quads = build_channel_mesh(NX, NY, LENGTH_M, WIDTH_M)
    write_mesh_parquet(path, verts, quads, np.zeros(NX * NY, dtype=np.float32))
    return path


def _workflow(
    mesh: Path,
    backend: str,
    state: dict[str, object],
    *,
    order: str,
    t_end: float,
) -> SWEWorkflow:
    workflow = SWEWorkflow(
        WorkflowConfig(
            mesh_source=str(mesh),
            manning_n=MANNING_N,
            output_interval_s=t_end,
            dt_max=1.0,
            dt_init=1e-2,
            cfl_interval=10,
            initial_state_source=state,
            initial_state_order=order,
            solver_impl=backend,  # pyright: ignore[reportArgumentType]
        )
    )
    workflow.prepare()
    return workflow


def check_initial_momentum(mesh: Path, backend: str) -> bool:
    """Uniform sheet flow must decay by Manning friction, not vanish."""
    n = NX * NY
    h0, u0, t_end = 0.5, 1.0, 32.0
    state: dict[str, object] = {
        "h": np.full(n, h0, dtype=np.float32),
        "hu": np.full(n, h0 * u0, dtype=np.float32),
        "hv": np.zeros(n, dtype=np.float32),
    }
    workflow = _workflow(mesh, backend, state, order="original", t_end=t_end)
    result = workflow.run([SimulationPhase(duration_s=t_end, sources=[])])
    if result.hu_final is None:
        print("  FAIL: backend does not expose download_hu/hv")
        return False

    u_sim = float(np.median(np.asarray(result.hu_final) / np.maximum(result.h_final, 1e-9)))
    # 0-D reference: du/dt = -g n^2 u|u| / h^(4/3), integrated finely.
    u_ref, dt = u0, 1e-4
    for _ in range(int(t_end / dt)):
        u_ref /= 1.0 + dt * GRAVITY * MANNING_N**2 * abs(u_ref) / h0 ** (4.0 / 3.0)
    err = abs(u_sim - u_ref) / u_ref
    ok = err <= GATE_MOMENTUM_DECAY_REL
    print(f"  solver u = {u_sim:.6f} m/s, 0-D Manning reference = {u_ref:.6f} m/s")
    print(f"  relative error {err:.3e} <= {GATE_MOMENTUM_DECAY_REL} -> {'PASS' if ok else 'FAIL'}")
    if u_sim == 0.0:
        print("  (u is exactly zero: initial momentum was discarded, not decayed)")
    return ok


def _phases(t_end: float, half: float) -> tuple[list[SimulationPhase], list[SimulationPhase]]:
    """A source-driven schedule split at ``half``, and its uninterrupted twin."""
    source = PointSource(20.0, (LENGTH_M / 2.0, WIDTH_M / 2.0), 30.0)
    first = SimulationPhase(duration_s=half, sources=[source])
    second = SimulationPhase(duration_s=t_end - half, sources=[])
    return [first, second], [second]


def check_warm_start(mesh: Path, backend: str, t_end: float) -> bool:
    """0 -> T must match 0 -> T/2 then resume from the exported state."""
    n = NX * NY
    zeros = np.zeros(n, dtype=np.float32)
    dry: dict[str, object] = {"h": zeros.copy(), "hu": zeros.copy(), "hv": zeros.copy()}
    half = t_end / 2.0
    full_phases, tail_phases = _phases(t_end, half)

    continuous = _workflow(mesh, backend, dry, order="original", t_end=t_end)
    ref = continuous.run(full_phases)

    interrupted = _workflow(mesh, backend, dry, order="original", t_end=half)
    mid = interrupted.run([full_phases[0]])
    if mid.hu_final is None or mid.hv_final is None:
        print("  FAIL: backend does not expose download_hu/hv")
        return False

    # Downloads are in solver (Hilbert) order, so resume in that order.
    resumed_state: dict[str, object] = {
        "h": np.asarray(mid.h_final, dtype=np.float32).copy(),
        "hu": np.asarray(mid.hu_final, dtype=np.float32).copy(),
        "hv": np.asarray(mid.hv_final, dtype=np.float32).copy(),
    }
    resumed = _workflow(mesh, backend, resumed_state, order="solver", t_end=t_end - half)
    warm = resumed.run(tail_phases)

    scale = max(float(np.abs(ref.h_final).max()), 1e-12)
    err = float(np.abs(np.asarray(warm.h_final) - np.asarray(ref.h_final)).max()) / scale
    vol_err = abs(warm.volume_final_m3 - ref.volume_final_m3) / max(ref.volume_final_m3, 1e-12)
    ok = err <= GATE_WARM_START_REL
    print(f"  continuous volume = {ref.volume_final_m3:,.3f} m^3")
    print(f"  warm-start volume = {warm.volume_final_m3:,.3f} m^3 (rel diff {vol_err:.3e})")
    print(f"  max depth diff {err:.3e} <= {GATE_WARM_START_REL:.0e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="gpu_resident_batch")
    parser.add_argument("--t-end", type=float, default=600.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        mesh = _mesh(Path(tmp))
        print(f"[1/2] initial momentum survives run() ({args.backend})")
        ok_momentum = check_initial_momentum(mesh, args.backend)
        print(f"\n[2/2] warm start reproduces an uninterrupted run ({args.backend})")
        ok_warm = check_warm_start(mesh, args.backend, args.t_end)

    print(f"\nresult: {'PASS' if (ok_momentum and ok_warm) else 'FAIL'}")
    return 0 if (ok_momentum and ok_warm) else 1


if __name__ == "__main__":
    sys.exit(main())

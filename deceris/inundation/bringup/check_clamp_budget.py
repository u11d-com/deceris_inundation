"""Measure the mass budget of EA Test 4: clamp source vs residual sink.

The volume drift a benchmark reports is the *difference* of two non-conservative
terms of opposite sign, and their balance flips with both mesh and stopping time
(``docs/implementation/19-grid-convergence/results.md``). Only the difference is
observable from the aggregate, which is why refining the mesh moved the number
around without ever saying which term was responsible.

The positivity clamp ``max(h_new, 0.0)`` in ``vulkan/shaders/update.py`` is the
source and is now instrumented, so this closes the budget:

    V_final - V_injected  =  clamp_mass  +  residual

``clamp_mass`` is measured directly; ``residual`` is what is left over, and is
the term whose mechanism is still unidentified. Reporting them separately is the
point — a small aggregate can hide two large opposing terms.

Runs the published EA Test 4 configuration by importing the benchmark harness,
so the physics, hydrograph and mesh are exactly what the suite gates on.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from deceris.inundation.bench import flood_propagation as fp
from deceris.inundation.workflow import SWEWorkflow, WorkflowConfig

DEFAULT_OUTPUT_ROOT = ".tmp/clamp-budget"
# (nx, ny, t_end_s) — the levels the grid-convergence study measured, so the
# budget can be read against drifts that are already recorded. The dx=5/t=2h
# entry is the one that came out negative, and so is the discriminating case.
DEFAULT_LEVELS: tuple[tuple[int, int, float], ...] = (
    (100, 200, 7200.0),
    (200, 400, 7200.0),
    (100, 200, 18000.0),
    (200, 400, 18000.0),
)


@dataclass(frozen=True)
class BudgetRow:
    """Mass budget for one (mesh, stopping time)."""

    nx: int
    ny: int
    dx_m: float
    t_end_s: float
    steps: int
    volume_injected_m3: float
    volume_final_m3: float
    drift_m3: float
    clamp_mass_m3: float
    residual_m3: float
    drift_rel: float
    clamp_rel: float
    residual_rel: float


def _measure(nx: int, ny: int, t_end_s: float, output_dir: Path) -> BudgetRow:
    spec = fp.PropagationSpec(
        name="propagation",
        nx=nx,
        ny=ny,
        domain_x_m=fp.DOMAIN_X_M,
        domain_y_m=fp.DOMAIN_Y_M,
        manning_n=fp.MANNING_N,
        t_end_s=t_end_s,
        probe_time_s=min(fp.PROBE_TIME_S, t_end_s),
        dt_max=fp.DT_MAX_DEFAULT,
        cfl_interval=fp.CFL_INTERVAL_DEFAULT,
        dataset_dir=Path(fp.DEFAULT_DATASET_DIR),
    )
    mesh_path = fp._ensure_mesh(spec, output_dir)

    config = WorkflowConfig(
        mesh_source=str(mesh_path),
        manning_n=spec.manning_n,
        output_interval_s=t_end_s,
        dt_max=spec.dt_max,
        dt_init=fp.DT_INIT_DEFAULT,
        cfl_interval=spec.cfl_interval,
        progress=False,
        initial_state_source=fp._initial_state(spec),
        initial_state_order="original",
        solver_impl="gpu_resident_batch",
        track_clamp_mass=True,
    )
    workflow = SWEWorkflow(config)
    workflow.prepare()
    result = workflow.run(fp._hydrograph_phases(spec))

    if result.clamp_mass_m3 is None:
        raise RuntimeError("backend did not report clamp mass; is track_clamp_mass wired up?")

    injected = result.volume_injected_m3
    drift = result.volume_final_m3 - injected
    clamp = result.clamp_mass_m3
    return BudgetRow(
        nx=nx,
        ny=ny,
        dx_m=spec.domain_x_m / nx,
        t_end_s=t_end_s,
        steps=int(result.steps_total),
        volume_injected_m3=injected,
        volume_final_m3=result.volume_final_m3,
        drift_m3=drift,
        clamp_mass_m3=clamp,
        residual_m3=drift - clamp,
        drift_rel=drift / injected,
        clamp_rel=clamp / injected,
        residual_rel=(drift - clamp) / injected,
    )


def _print_report(rows: list[BudgetRow]) -> None:
    print()
    print("mass budget: drift = clamp + residual   (all volumes in m^3)")
    print(
        f"{'dx_m':>6} {'t_end_h':>8} {'steps':>8} {'drift':>12} "
        f"{'clamp':>12} {'residual':>12} {'drift_rel':>11}"
    )
    for r in rows:
        print(
            f"{r.dx_m:6.1f} {r.t_end_s / 3600.0:8.1f} {r.steps:8d} {r.drift_m3:+12.2f} "
            f"{r.clamp_mass_m3:+12.2f} {r.residual_m3:+12.2f} {r.drift_rel:+11.3e}"
        )
    print()
    for r in rows:
        if r.clamp_mass_m3 < 0.0:
            print(f"  !! dx={r.dx_m} t={r.t_end_s}: clamp mass is negative, which is impossible")


def main() -> int:
    parser = argparse.ArgumentParser(description="EA Test 4 clamp-vs-residual mass budget")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BudgetRow] = []
    for nx, ny, t_end_s in DEFAULT_LEVELS:
        print(f"[budget] nx={nx} ny={ny} t_end={t_end_s / 3600.0:.1f} h ...", flush=True)
        rows.append(_measure(nx, ny, t_end_s, output_dir))

    _print_report(rows)
    out = output_dir / "budget.json"
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"\n[summary] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

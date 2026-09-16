"""Mesh-refinement sweep over the benchmark harnesses.

Runs an existing harness at several resolutions and fits the observed order of
accuracy of each of its error metrics. Two questions motivate it:

1. **Is the scheme converging at the order it should?** HLL + SSP-RK2 without
   slope reconstruction is first order in space, so a smooth-region metric
   should show ``p ~ 1`` and a shock/front metric somewhat less. EA Test 4's
   front runs systematically 4-6% fast; whether that shrinks under refinement
   decides if it is discretisation error or a bias in the scheme.
2. **Is there an error floor?** Truncation error must fall monotonically under
   refinement. A metric that flattens or turns up is therefore evidence of a
   non-discretisation floor — which is the shape the EA Test 4 mass drift
   already showed (see ``docs/implementation/18-flood-propagation/results.md``).
   This sweep can *detect* such a floor in float32 alone; attributing it to
   float32 specifically rather than to the dry-front treatment or the known
   well-balance leak would need a float64 build of the same scheme.

Only cases whose reference is resolution-independent are swept: the analytical
dam-breaks (closed-form Stoker/Ritter, and a fixed 2000-cell axisymmetric solve
for the radial case) and EA Test 4 (fixed 1400-cell radial reference). Refining
the solver mesh therefore moves solver error alone.

Each level is a separate subprocess running the harness's own CLI, so the
metrics are exactly the ones the harness gates on — no reimplementation. Gate
failures at coarse levels are expected and are not treated as errors; only a
missing or unparseable summary is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inundation.bench.common import observed_order

# ── Sweep configuration ─────────────────────────────────────────────────────
# Metrics are the harness's own CaseMetrics fields. Ones that can be legitimately
# zero (isotropy_spread) are excluded: an order fit needs positive errors.


@dataclass(frozen=True)
class SweepCase:
    """One harness, the resolutions to run it at, and the metrics to fit."""

    name: str
    module: str
    domain_x_m: float  # for dx = domain_x_m / nx
    levels: tuple[tuple[int, int], ...]  # (nx, ny), coarse to fine
    metrics: tuple[str, ...]
    extra_args: tuple[str, ...] = ()


SWEEP_CASES: tuple[SweepCase, ...] = (
    SweepCase(
        name="dambreak",
        module="inundation.bench.dambreak",
        domain_x_m=1000.0,
        # 1D case: refine along the wave direction only, ny stays a strip.
        levels=((125, 5), (250, 5), (500, 5), (1000, 5), (2000, 5)),
        metrics=("l1_rel", "l2_rel", "front_rel_err", "volume_drift_rel"),
    ),
    SweepCase(
        name="radial_dambreak",
        module="inundation.bench.radial_dambreak",
        domain_x_m=40.0,
        levels=((100, 100), (200, 200), (400, 400), (800, 800)),
        metrics=("l1_rel", "l2_rel", "front_rel_err", "volume_drift_rel"),
    ),
    SweepCase(
        name="flood_propagation",
        module="inundation.bench.flood_propagation",
        domain_x_m=1000.0,
        levels=((50, 100), (100, 200), (200, 400), (400, 800)),
        # max_arrival_err_rel is included but reads against a floor: the harness
        # snapshots every 60 s, which is ~2% of the arrival time at the outer
        # gauges regardless of dx. depth_l1_rel and speed_l1_rel come from a
        # separate run that ends exactly at the probe time, so they are clean.
        metrics=(
            "depth_l1_rel",
            "speed_l1_rel",
            "max_arrival_err_rel",
            "volume_drift_rel",
        ),
    ),
)

DEFAULT_OUTPUT_ROOT = ".tmp/grid-convergence"
# The sweep measures error, not throughput, so one untimed repeat per level.
HARNESS_ARGS = ("--repeats", "1", "--warmup", "0")


@dataclass(frozen=True)
class LevelResult:
    """Metrics a harness reported for one sub-case at one resolution."""

    sweep_case: str
    sub_case: str  # the harness's own case name (dambreak runs stoker + ritter)
    backend: str
    nx: int
    ny: int
    dx_m: float
    wall_s: float
    metrics: dict[str, float]


def _run_level(case: SweepCase, nx: int, ny: int, output_root: Path) -> list[LevelResult]:
    """Run one harness at one resolution; return its per-sub-case metrics."""
    output_dir = output_root / case.name / f"{nx}x{ny}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        case.module,
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--output-dir",
        str(output_dir),
        *HARNESS_ARGS,
        *case.extra_args,
    ]
    print(f"[{case.name}] nx={nx} ny={ny} (dx={case.domain_x_m / nx:g} m) ...", flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, check=False)  # noqa: S603 - fixed argv, no shell
    wall_s = time.perf_counter() - t0

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"{case.name} at {nx}x{ny} produced no summary "
            f"(exit {completed.returncode}); see the output above"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    results: list[LevelResult] = []
    for entry in summary["correctness"]:
        metrics = {m: entry[m] for m in case.metrics if entry.get(m) is not None}
        results.append(
            LevelResult(
                sweep_case=case.name,
                sub_case=str(entry["case"]),
                backend=str(entry["backend"]),
                nx=nx,
                ny=ny,
                dx_m=case.domain_x_m / nx,
                wall_s=wall_s,
                metrics={k: float(v) for k, v in metrics.items()},
            )
        )
    print(f"[{case.name}] nx={nx} ny={ny} done in {wall_s:.1f} s", flush=True)
    return results


def _fit_table(results: list[LevelResult]) -> list[dict[str, object]]:
    """Fit an order per (sub_case, metric) over the levels that reported it."""
    fits: list[dict[str, object]] = []
    keys = sorted({(r.sweep_case, r.sub_case, r.backend) for r in results})
    for sweep_case, sub_case, backend in keys:
        series = sorted(
            (
                r
                for r in results
                if (r.sweep_case, r.sub_case, r.backend) == (sweep_case, sub_case, backend)
            ),
            key=lambda r: -r.dx_m,
        )
        metric_names = sorted({m for r in series for m in r.metrics})
        for metric in metric_names:
            usable = [r for r in series if metric in r.metrics]
            dx = [r.dx_m for r in usable]
            err = [r.metrics[metric] for r in usable]
            record: dict[str, object] = {
                "case": f"{sweep_case}/{sub_case}",
                "backend": backend,
                "metric": metric,
                "dx_m": dx,
                "error": err,
            }
            try:
                fit = observed_order(dx, err)
            except ValueError as exc:
                record |= {"order": None, "pairwise_orders": [], "monotone": None, "note": str(exc)}
            else:
                record |= {
                    "order": fit.order,
                    "pairwise_orders": fit.pairwise_orders,
                    "monotone": fit.monotone,
                }
            fits.append(record)
    return fits


def _print_report(results: list[LevelResult], fits: list[dict[str, object]]) -> None:
    print()
    print(f"{'case':28} {'dx_m':>8} {'nx':>6} {'ny':>6} {'wall_s':>8}")
    for r in sorted(results, key=lambda r: (r.sweep_case, r.sub_case, -r.dx_m)):
        print(
            f"{r.sweep_case + '/' + r.sub_case:28} {r.dx_m:8.4g} "
            f"{r.nx:6d} {r.ny:6d} {r.wall_s:8.1f}"
        )
    print()
    print("observed order p from error ~ dx**p (coarse -> fine)")
    print(f"{'case':28} {'metric':20} {'p':>6} {'monotone':>9}  errors")
    for f in fits:
        order = f["order"]
        # _fit_table writes fit.order (a float) into "order", or None when the fit
        # raised; the dict is heterogeneous, so the numeric arm needs a cast.
        p_txt = "  n/a" if order is None else f"{cast('float', order):6.2f}"
        errs = " ".join(f"{e:.3e}" for e in list(f["error"]))  # pyright: ignore[reportArgumentType]
        print(f"{f['case']!s:28} {f['metric']!s:20} {p_txt} {f['monotone']!s:>9}  {errs}")
        if f.get("note"):
            print(f"  !! {f['note']}")
        elif not f["monotone"]:
            print("  !! error does not fall monotonically under refinement (floor)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mesh-refinement sweep over the bench harnesses")
    parser.add_argument(
        "--cases",
        default=",".join(c.name for c in SWEEP_CASES),
        help=f"CSV of {tuple(c.name for c in SWEEP_CASES)}",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--max-levels",
        type=int,
        default=None,
        help="run only the first N (coarsest) levels of each case",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    by_name = {c.name: c for c in SWEEP_CASES}
    unknown = [c for c in wanted if c not in by_name]
    if unknown:
        print(f"unknown case(s) {unknown}; choose from {tuple(by_name)}")
        return 2

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[LevelResult] = []
    for name in wanted:
        case = by_name[name]
        levels = case.levels[: args.max_levels] if args.max_levels else case.levels
        for nx, ny in levels:
            results.extend(_run_level(case, nx, ny, output_root))

    fits = _fit_table(results)
    _print_report(results, fits)

    summary_path = output_root / "convergence.json"
    summary_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "case": f"{r.sweep_case}/{r.sub_case}",
                        "backend": r.backend,
                        "nx": r.nx,
                        "ny": r.ny,
                        "dx_m": r.dx_m,
                        "wall_s": r.wall_s,
                        "metrics": r.metrics,
                    }
                    for r in results
                ],
                "fits": fits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

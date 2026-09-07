"""Run the full benchmark suite (+ unit tests) and emit a standalone HTML report.

Each gated harness (`dambreak`, `radial_dambreak`, `floodplain_depressions`,
`momentum_obstruction`, `flood_propagation`) is run as a subprocess through its
own CLI with GIF rendering enabled, so the metrics and animations are exactly
the ones the harness itself produces — no reimplementation. The per-suite
``summary.json`` files, console logs and GIF animations are then folded into a
single self-contained ``report.html`` (GIFs are base64-inlined, so the file can
be shared as-is).

The report is written even when suites fail; the process exit code reflects the
overall verdict (0 = everything passed).

Run (macOS host, MoltenVK — needs an unsandboxed shell)::

    .venv/bin/python -m deceris.inundation.bench.report

or through the dev container::

    just benchmark-report
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

DEFAULT_OUTPUT_ROOT = ".tmp/bench-report"


@dataclass(frozen=True)
class SuiteSpec:
    """One benchmark harness and the flags needed to run it with animations."""

    name: str
    module: str
    title: str
    gif_args: tuple[str, ...]


SUITES: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        name="dambreak",
        module="deceris.inundation.bench.dambreak",
        title="1D dam-break (Stoker/Ritter analytical)",
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="radial_dambreak",
        module="deceris.inundation.bench.radial_dambreak",
        title="Radial dam-break (axisymmetric FV reference)",
        gif_args=("--gif", "--gif-2d"),
    ),
    SuiteSpec(
        name="floodplain_depressions",
        module="deceris.inundation.bench.floodplain_depressions",
        title="EA Test 2 — floodplain depressions",
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="momentum_obstruction",
        module="deceris.inundation.bench.momentum_obstruction",
        title="EA Test 3 — momentum over an obstruction",
        gif_args=("--gif",),
    ),
    SuiteSpec(
        name="flood_propagation",
        module="deceris.inundation.bench.flood_propagation",
        title="EA Test 4 — flood-front propagation",
        gif_args=("--gif",),
    ),
)


@dataclass
class SuiteResult:
    """Everything the report needs about one completed suite run."""

    spec: SuiteSpec
    exit_code: int
    wall_s: float
    log_text: str
    summary: dict[str, object] | None
    gif_paths: list[Path] = field(default_factory=lambda: list[Path]())

    @property
    def status(self) -> str:
        if self.exit_code == 0 and self.summary is not None:
            return "PASS"
        if self.exit_code == 1 and self.summary is not None:
            return "FAIL"
        return "ERROR"


@dataclass
class PytestResult:
    """Outcome of the unit-test run."""

    exit_code: int
    wall_s: float
    log_text: str


# ── Suite execution ──────────────────────────────────────────────────────────


def _run_suite(spec: SuiteSpec, output_root: Path, extra_args: list[str]) -> SuiteResult:
    output_dir = output_root / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        spec.module,
        "--output-dir",
        str(output_dir),
        *spec.gif_args,
        *extra_args,
    ]
    print(f"[{spec.name}] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd, check=False, capture_output=True, text=True
    )
    wall_s = time.perf_counter() - t0
    log_text = completed.stdout + (("\n" + completed.stderr) if completed.stderr else "")

    summary: dict[str, object] | None = None
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        parsed: object = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            summary = cast("dict[str, object]", parsed)

    result = SuiteResult(
        spec=spec,
        exit_code=completed.returncode,
        wall_s=wall_s,
        log_text=log_text,
        summary=summary,
        gif_paths=sorted(output_dir.glob("*.gif")),
    )
    print(
        f"[{spec.name}] {result.status} in {wall_s:.1f} s "
        f"(exit {completed.returncode}, {len(result.gif_paths)} gif(s))",
        flush=True,
    )
    return result


def _run_pytest(repo_root: Path) -> PytestResult:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", "--color=no"]
    print(f"[unit-tests] running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        cmd, check=False, capture_output=True, text=True, cwd=repo_root
    )
    wall_s = time.perf_counter() - t0
    log_text = completed.stdout + (("\n" + completed.stderr) if completed.stderr else "")
    verdict = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"[unit-tests] {verdict} in {wall_s:.1f} s (exit {completed.returncode})", flush=True)
    return PytestResult(exit_code=completed.returncode, wall_s=wall_s, log_text=log_text)


# ── HTML rendering ───────────────────────────────────────────────────────────

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
       background: #f6f7f9; color: #1f2328; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.2rem; margin-top: 2.2rem; }
.meta { color: #57606a; font-size: 0.85rem; }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 1rem;
         font-weight: 600; font-size: 0.8rem; color: #fff; vertical-align: middle; }
.badge.pass { background: #1a7f37; } .badge.fail { background: #cf222e; }
.badge.error { background: #bf8700; } .badge.skip { background: #57606a; }
table { border-collapse: collapse; margin: 0.6rem 0 1rem; font-size: 0.85rem;
        background: #fff; }
th, td { border: 1px solid #d0d7de; padding: 0.3rem 0.6rem; text-align: left;
         white-space: nowrap; }
th { background: #eaeef2; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.failed td { background: #ffebe9; }
details { margin: 0.6rem 0; }
details > summary { cursor: pointer; color: #57606a; font-size: 0.85rem; }
pre { background: #24292f; color: #d0d7de; padding: 0.8rem; border-radius: 6px;
      overflow-x: auto; font-size: 0.75rem; line-height: 1.4; }
.gifs { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; }
figure img { max-width: 520px; width: 100%; border: 1px solid #d0d7de;
             border-radius: 6px; }
figcaption { font-size: 0.8rem; color: #57606a; margin-top: 0.2rem; }
"""


def _badge(status: str) -> str:
    return f'<span class="badge {status.lower()}">{html.escape(status)}</span>'


def _fmt_value(value: object) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "&#10003;" if value else "&#10007;"
    if isinstance(value, float):
        return html.escape(f"{value:.6g}")
    if isinstance(value, list):
        items = cast("list[object]", value)
        return html.escape(", ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in items))
    return html.escape(str(value))


def _rows_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<p class='meta'>no entries</p>"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body_rows: list[str] = []
    for row in rows:
        failed = row.get("passed") is False
        cells = "".join(
            f"<td class='{'num' if isinstance(row.get(c), int | float) else ''}'>"
            f"{_fmt_value(row.get(c))}</td>"
            for c in columns
        )
        body_rows.append(f"<tr class='{'failed' if failed else ''}'>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _mapping_table(mapping: dict[str, object]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{_fmt_value(v)}</td></tr>"
        for k, v in mapping.items()
    )
    return f"<table><thead><tr><th>gate</th><th>threshold</th></tr></thead><tbody>{body}</tbody></table>"


def _as_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    entries = cast("list[object]", value)
    return [cast("dict[str, object]", e) for e in entries if isinstance(e, dict)]


def _log_details(title: str, log_text: str) -> str:
    return (
        f"<details><summary>{html.escape(title)}</summary>"
        f"<pre>{html.escape(log_text.strip() or '(no output)')}</pre></details>"
    )


def _gif_figure(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f"<figure><img loading='lazy' src='data:image/gif;base64,{data}' "
        f"alt='{html.escape(path.name)}'/>"
        f"<figcaption>{html.escape(path.name)} ({path.stat().st_size / 1e6:.1f} MB)</figcaption>"
        f"</figure>"
    )


def _suite_section(result: SuiteResult) -> str:
    parts: list[str] = [
        f"<h2 id='{html.escape(result.spec.name)}'>"
        f"{html.escape(result.spec.title)} {_badge(result.status)}</h2>",
        f"<p class='meta'>module <code>{html.escape(result.spec.module)}</code> · "
        f"wall {result.wall_s:.1f} s · exit {result.exit_code}</p>",
    ]
    if result.summary is not None:
        gates = result.summary.get("gates")
        if isinstance(gates, dict):
            parts.append("<h3>Gates</h3>")
            parts.append(_mapping_table(cast("dict[str, object]", gates)))
        correctness = _as_rows(result.summary.get("correctness"))
        if correctness:
            parts.append("<h3>Correctness</h3>")
            parts.append(_rows_table(correctness))
        performance = _as_rows(result.summary.get("performance"))
        if performance:
            parts.append("<h3>Performance</h3>")
            parts.append(_rows_table(performance))
    else:
        parts.append("<p class='meta'>no summary.json produced — see the log below</p>")
    if result.gif_paths:
        parts.append("<h3>Animations</h3><div class='gifs'>")
        parts.extend(_gif_figure(p) for p in result.gif_paths)
        parts.append("</div>")
    parts.append(_log_details("console log", result.log_text))
    return "\n".join(parts)


def _overview_table(pytest_result: PytestResult | None, results: list[SuiteResult]) -> str:
    rows: list[str] = []
    if pytest_result is not None:
        status = "PASS" if pytest_result.exit_code == 0 else "FAIL"
        rows.append(
            f"<tr><td><a href='#unit-tests'>unit tests (pytest)</a></td>"
            f"<td>{_badge(status)}</td><td class='num'>{pytest_result.wall_s:.1f}</td>"
            f"<td class='num'>&mdash;</td><td class='num'>&mdash;</td></tr>"
        )
    for r in results:
        correctness = _as_rows(r.summary.get("correctness")) if r.summary else []
        passed = sum(1 for row in correctness if row.get("passed") is True)
        cases = f"{passed}/{len(correctness)}" if correctness else "&mdash;"
        rows.append(
            f"<tr><td><a href='#{html.escape(r.spec.name)}'>{html.escape(r.spec.title)}</a></td>"
            f"<td>{_badge(r.status)}</td><td class='num'>{r.wall_s:.1f}</td>"
            f"<td class='num'>{cases}</td><td class='num'>{len(r.gif_paths)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>suite</th><th>status</th><th>wall s</th>"
        "<th>cases passed</th><th>gifs</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(pytest_result: PytestResult | None, results: list[SuiteResult]) -> str:
    """Render the full standalone report (GIFs inlined as data URIs)."""
    overall_pass = all(r.status == "PASS" for r in results) and (
        pytest_result is None or pytest_result.exit_code == 0
    )
    overall = _badge("PASS" if overall_pass else "FAIL")
    generated = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"<h1>Deceris inundation — benchmark report {overall}</h1>",
        f"<p class='meta'>generated {generated} · {html.escape(platform.platform())} · "
        f"python {html.escape(platform.python_version())}</p>",
        _overview_table(pytest_result, results),
    ]
    if pytest_result is not None:
        status = "PASS" if pytest_result.exit_code == 0 else "FAIL"
        sections.append(f"<h2 id='unit-tests'>Unit tests (pytest) {_badge(status)}</h2>")
        sections.append(
            f"<p class='meta'>wall {pytest_result.wall_s:.1f} s · exit {pytest_result.exit_code}</p>"
        )
        sections.append(_log_details("pytest output", pytest_result.log_text))
    sections.extend(_suite_section(r) for r in results)
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        "<title>Deceris inundation — benchmark report</title>"
        f"<style>{_CSS}</style></head><body><main>{body}</main></body></html>"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all gated benchmarks (+ unit tests) and emit an HTML report"
    )
    parser.add_argument(
        "--suites",
        default=",".join(s.name for s in SUITES),
        help=f"CSV of {tuple(s.name for s in SUITES)}",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeats", type=int, default=2, help=">=2 enables determinism checks")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--skip-unit-tests", action="store_true", help="skip the pytest run, benchmarks only"
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    wanted = [s.strip() for s in args.suites.split(",") if s.strip()]
    by_name = {s.name: s for s in SUITES}
    unknown = [s for s in wanted if s not in by_name]
    if unknown:
        print(f"unknown suite(s) {unknown}; choose from {tuple(by_name)}")
        return 2

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]

    pytest_result = None if args.skip_unit_tests else _run_pytest(repo_root)

    extra_args = ["--repeats", str(args.repeats), "--warmup", str(args.warmup)]
    results = [_run_suite(by_name[name], output_root, extra_args) for name in wanted]

    report_path = output_root / "report.html"
    report_path.write_text(render_html(pytest_result, results), encoding="utf-8")
    print(f"\n[report] {report_path} ({report_path.stat().st_size / 1e6:.1f} MB)")

    for r in results:
        print(f"  {r.spec.name:24} {r.status}")
    if pytest_result is not None:
        print(f"  {'unit-tests':24} {'PASS' if pytest_result.exit_code == 0 else 'FAIL'}")

    overall_ok = all(r.status == "PASS" for r in results) and (
        pytest_result is None or pytest_result.exit_code == 0
    )
    print(f"RESULT: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

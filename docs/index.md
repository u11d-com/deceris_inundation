# Documentation

deceris-inundation SWE mesh solver — canonical reference for internal devs
and AI agents. Audience: engineers integrating the solver + agents working
on the codebase. Optimized for search via [qmd](https://github.com/tobi/qmd).

## Tree

| Tier | Purpose | Docs |
| --- | --- | --- |
| `architecture/` | Solver internals: CUDA pipeline, sync rungs, multi-GPU halo, determinism, GPU numbers | [`overview.md`](architecture/overview.md), [`cuda-migration.md`](architecture/cuda-migration.md) |
| `integration/` | How to embed `SWEWorkflow` in a host Python app | [`workflow.md`](integration/workflow.md) |
| `bringup/` | GPU bring-up runbook, step-by-step | [`README.md`](bringup/README.md) |
| `bringup/plans/` | Forward-looking implementation plans: IPC ghost-write, overlap exchange, depth-2 ring, phase-2 compaction, perf roadmap | [`ipc-ghost-write.md`](bringup/plans/ipc-ghost-write.md), [`perf-roadmap.md`](bringup/plans/perf-roadmap.md), [`overlap-exchange.md`](bringup/plans/overlap-exchange.md), [`depth2-ghost-ring.md`](bringup/plans/depth2-ghost-ring.md), [`phase2-compaction.md`](bringup/plans/phase2-compaction.md) |
| `benchmarks/` | Validation + baseline result logs (lake-at-rest) | [`lake-at-rest-test.md`](benchmarks/lake-at-rest-test.md), [`lake-at-rest-baseline-results.md`](benchmarks/lake-at-rest-baseline-results.md) |
| `reference/` | Vulkan/Kompute build patches (macOS / CMake 4.x / MoltenVK) | [`kp-patches.md`](reference/kp-patches.md) |
| `archive/` | Historical originals preserving original paths + blame continuity. Superseded by canonical docs. QMD-excluded. | [`README.md`](archive/README.md) |

## Linting

`just mdlint` enforces the strict pymarkdown config at `.pymarkdown`
(see `dev` extra in `pyproject.toml`):

- Atx headings only, no inline HTML, no trailing spaces, fence languages required
- Line length 100 (prose only; code blocks + tables exempt)
- List indents consistent, ordered lists use 1/2/3, unordered use `-`
- Run as part of `just check`

Archive content is excluded by the linter — original prose formatting
preserved as-is for historical continuity.

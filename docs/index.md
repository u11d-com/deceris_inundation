# Documentation

<<<<<<< HEAD
| Folder | Purpose |
| --- | --- |
| [`architecture/`](architecture/) | Current Vulkan solver design and invariants |
| [`planning/`](planning/) | Roadmap and settled decisions |
| [`implementation/`](implementation/) | Benchmark plans, measurements, and conclusions |
| [`reference/`](reference/) | Vulkan and Kompute build notes |
| [`archive/`](archive/) | Unrelated historical reference material |

Start with [`architecture/overview.md`](architecture/overview.md). Benchmark
evidence lives in [`implementation/benchmark.md`](implementation/benchmark.md).
=======
deceris-inundation SWE mesh solver — canonical reference for internal devs
and AI agents. **Agents: use the `qmd` skill to search/read this tree —
don't grep by hand. Re-embed via the same skill after any structural
change (new/moved/renamed files).**

## Tree

| Tier | Purpose | Docs |
| --- | --- | --- |
| `architecture/` | What the solver is, its requirements, and its CUDA design (pipeline, sync rungs, multi-GPU halo, determinism) | [`overview.md`](architecture/overview.md), [`cuda-migration.md`](architecture/cuda-migration.md), [`validation-plan.md`](architecture/validation-plan.md) |
| `planning/` | Single place for "what's next" and "why we decided X" | [`roadmap.md`](planning/roadmap.md), [`decisions.md`](planning/decisions.md) |
| `implementation/` | One-time node setup, the cross-phase benchmark ledger, and one numbered folder per engineering effort (`plan.md` + `results.md` each) | [`gpu-setup.md`](implementation/gpu-setup.md), [`benchmark.md`](implementation/benchmark.md), `00-lake-at-rest/` … `12-k-step-ghost-ring/` |
| `reference/` | Vulkan/Kompute build patches (macOS / CMake 4.x / MoltenVK) | [`kp-patches.md`](reference/kp-patches.md) |
| `archive/` | Historical originals preserving original paths + blame continuity. Superseded by canonical docs. QMD-excluded. | [`README.md`](archive/README.md) |

### `implementation/` folders

| # | Folder | Status |
| --- | --- | --- |
| 00 | `lake-at-rest/` | done |
| 01 | `nccl-graph-capture-spike/` | done |
| 02 | `multi-gpu-orchestration/` | done |
| 03 | `multi-gpu-load-imbalance-profiling/` | done |
| 04 | `multi-gpu-scaling-benchmark/` | done (superseded by 05/06) |
| 05 | `overlap-exchange/` | done |
| 06 | `depth2-ghost-ring/` | done, default |
| 07 | `ipc-ghost-write/` | done, rejected (NCCL stays default) |
| 08 | `gather-update-fusion/` | done, default |
| 09 | `perf-measurements/` | done |
| 10 | `wet-dry-compaction/` | not started (primary next lever) |
| 11 | `ncu-audit-l2-pinning/` | not started |
| 12 | `k-step-ghost-ring/` | not started |

See [`implementation/benchmark.md`](implementation/benchmark.md) for the
full scoreboard and [`planning/roadmap.md`](planning/roadmap.md) for
sequencing/priority.

## Maintaining this tree

- **New engineering effort** → new numbered `implementation/NN-<slug>/`
  folder (`plan.md` + `results.md`), add a row to
  [`implementation/benchmark.md`](implementation/benchmark.md) and the
  folder table above.
- **Settled tradeoff/decision** → append to
  [`planning/decisions.md`](planning/decisions.md) (never edit past
  entries), linking the `implementation/` evidence.
- **Priorities/sequencing change** → edit
  [`planning/roadmap.md`](planning/roadmap.md) only.
- **Architecture docs** describe current design/requirements only — no
  status prose or result tables; link to `decisions.md` / `implementation/`
  instead.
- **New top-level `docs/` folder** → register it in `.qmd/index.yml` and
  the tree table above.
- Run `just mdlint`, then **re-embed via the `qmd` skill**, after any edit
  here.

## Linting

`just mdlint` enforces the strict pymarkdown config at `.pymarkdown`
(see `dev` extra in `pyproject.toml`):

- Atx headings only, no inline HTML, no trailing spaces, fence languages required
- Line length 100 (prose only; code blocks + tables exempt)
- List indents consistent, ordered lists use 1/2/3, unordered use `-`
- Run as part of `just check`

Archive content is excluded by the linter — original prose formatting
preserved as-is for historical continuity.
>>>>>>> 2f36b6f (fix: Docs rework 2)

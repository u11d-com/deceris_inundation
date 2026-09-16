# Archive

Historical originals of the canonical docs, preserved 1:1 in their original
in-monorepo paths under `deceris/inundation/...` (paths as they were before
this repo was extracted from the deceris api monorepo; the package now lives
at `inundation/`). Each file here is byte-for-byte
identical to the version that was renamed + moved into the canonical tree
during the 2026-Q3 docs rework.

## Purpose

- **Blame continuity**: `git log --follow` on the canonical file resolves
  back through this snapshot, so future archaeologists can find the original
  authoring history even after the move.
- **Original formatting**: archive docs are excluded from the pymarkdown
  linter (`docs/archive/**` in `.pymarkdown` config + `--exclude` glob on
  the `just mdlint` recipe). Editorial decisions about prose line lengths,
  inline HTML, list marker styles, etc. are preserved exactly as written.
- **QMD isolation**: archive is a separate QMD collection (`docs-archive`,
  `includeByDefault: false`). Only surface archive results when explicitly
  queried with `-c docs-archive`.

## 1:1 path map

The "Archive (original)" column lists the docs' original monorepo paths at
extraction time (pre-rename `deceris/inundation/...`). The historical files
themselves are not tracked in this repo; only this mapping survives.

| Canonical (new) | Original monorepo path |
| --- | --- |
| `docs/architecture/overview.md` | `archive/deceris/inundation/ARCHITECTURE.md` |
| `docs/architecture/cuda-migration.md` | `archive/deceris/inundation/CUDA_MIGRATION_PLAN.md` |
| `docs/integration/workflow.md` | `archive/deceris/inundation/INTEGRATION_WORKFLOW.md` |
| `docs/benchmarks/lake-at-rest-test.md` | `archive/deceris/inundation/LAKE_AT_REST_TEST.md` |
| `docs/benchmarks/lake-at-rest-baseline-results.md` | `archive/deceris/inundation/LAKE_AT_REST_BASELINE_RESULTS.md` |
| `docs/bringup/README.md` | `archive/deceris/inundation/bringup/README.md` |
| `docs/bringup/plans/ipc-ghost-write.md` | `archive/deceris/inundation/bringup/IPC_GHOST_WRITE_PLAN.md` |
| `docs/bringup/plans/perf-roadmap.md` | `archive/deceris/inundation/bringup/PERF_ROADMAP.md` |
| `docs/bringup/plans/overlap-exchange.md` | `archive/deceris/inundation/bringup/OVERLAP_EXCHANGE_PLAN.md` |
| `docs/bringup/plans/depth2-ghost-ring.md` | `archive/deceris/inundation/bringup/DEPTH2_GHOST_PLAN.md` |
| `docs/bringup/plans/phase2-compaction.md` | `archive/deceris/inundation/bringup/PHASE2_COMPACTION_PLAN.md` |
| `docs/reference/kp-patches.md` | `archive/deceris/inundation/kp-patches/README.md` |

If a canonical doc ever needs to revert, the original is one `cp` away.
If the archive needs a re-rename to match a future canonical path, do that
here too — the 1:1 correspondence is the archive's only invariant.

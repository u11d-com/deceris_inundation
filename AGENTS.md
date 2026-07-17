# AGENTS.md

## Dev environment

This repo runs a persistent dev container (Docker Compose) that mirrors the
Apptainer/GPU production environment. All `just` recipes automatically exec
into it via `docker_prefix` in the `justfile` — no per-command container
spin-up, no code copied into the image (bind-mounted at `/workspace`).

Start the container once at the beginning of a session:

```sh
just up
```

Stop it when done:

```sh
just down
```

Inside VS Code's devcontainer (or any environment with `IN_CONTAINER=1`
already set), `just` recipes run natively with no docker exec — the
container detection is automatic.

If `just up` hasn't been run and the `dev` container isn't running, any
`just` recipe that needs it (`lint`, `format`, `typecheck`, `test`, etc.)
will fail with a `docker compose exec` error. That's expected — run `just up`
first.

The image itself (`deceris-inundation:dev`) must be built ahead of time:

```sh
just image-build
```

## Verification commands

Run before considering any change complete:

```python
just lint          # ruff check
just format        # ruff format (whole repo)
just format-check  # ruff format --check (CI-style, no writes)
just typecheck      # pyright, strict mode
just mdlint         # pymarkdownlnt over docs/ + root README.md + AGENTS.md
just test           # pytest
just check          # all of the above, in order
```

opencode is configured to auto-run `just format-file <file>` (ruff) after
edits, and to use `just pyright-lsp` for live pyright diagnostics — both exec
into the same persistent `dev` container.

## Shared modules

Cross-module constants + helpers in `swe_tuning.py`. Benchmark
fingerprint + invariant/compare/deterministic/plot helpers in
`benchmark_common.py`. **Import from these; do not redefine locally.**

## Documentation

**Reading/searching `docs/`: use the `qmd` skill** — it's indexed for
search and retrieval; don't grep/read the tree by hand. **After adding,
moving, or renaming any doc: re-run the `qmd` skill's update/embed step**
so the index stays current — this is a required step of any docs change,
not optional cleanup.

`docs/` tiers — each answers a different question:

| Tier | Answers | Update when |
|---|---|---|
| `architecture/` | What is this + why is it designed this way | Design changes (rare) |
| `planning/roadmap.md` | What's next, in what order | Priorities shift, phase starts/completes |
| `planning/decisions.md` | What was decided + why (append-only) | A tradeoff gets settled |
| `implementation/NN-*/` | How a specific effort was built + what it measured | Every new engineering effort/experiment |
| `implementation/benchmark.md` | Ledger: one row per effort, current status | Every time an `NN-*/results.md` changes |
| `reference/` | Third-party build/setup notes | Rare |
| `archive/` | Frozen originals, do not edit | Never (except full re-rework) |

Rules:

- New engineering effort → new `implementation/NN-<slug>/` folder, next
  sequential number, `plan.md` + `results.md` (stub `not started` if work
  hasn't begun). Add a row to `implementation/benchmark.md` and to
  `docs/index.md`'s folder table.
- Decision made (tradeoff resolved, default flipped, path rejected) →
  append to `planning/decisions.md`, link the `implementation/` evidence.
  Never rewrite past entries — it's a log.
- Priorities change → edit `planning/roadmap.md` only.
- `architecture/*.md` = current design/requirements only — no embedded
  status prose or result tables. Link to `decisions.md` / `implementation/`
  instead.
- New top-level `docs/` folder → register in `.qmd/index.yml` and
  `docs/index.md`'s tree table.
- Run `just mdlint` after any docs edit, then re-embed via `qmd`.

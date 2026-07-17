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

Pymarkdown config at `.pymarkdown` (strict defaults + 4 rule overrides:
MD003 atx, MD007 indent:4, MD013 line_length:100 prose-only, MD024
siblings_only). `docs/archive/**` excluded via `--exclude` glob (frozen
historical snapshot, original prose preserved). When editing docs, also
rewrite cross-references to new canonical paths under `docs/<tier>/...`.

## Shared modules

Cross-module constants + helpers in `swe_tuning.py`. Benchmark
fingerprint + invariant/compare/deterministic/plot helpers in
`benchmark_common.py`. **Import from these; do not redefine locally.**

Verification: `just check`.

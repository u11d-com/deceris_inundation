image := "deceris-inundation:dev"
in_container := env_var_or_default("IN_CONTAINER", "")
docker_prefix := if in_container == "1" { "" } else { "docker compose exec -T dev" }

# Build the dev image (system deps + venv + kp), frozen lock. Run whenever
# Dockerfile / pyproject.toml / uv.lock / kp-patches / build-kp-linux.sh change.
image-build:
    docker build -t {{image}} .

# Start the persistent dev container (docker-compose.yml). Run this once at
# the start of a dev session; all other recipes exec into it via
# docker_prefix instead of spinning up a throwaway container per command.
up:
    docker compose up -d

# Stop the persistent dev container.
down:
    docker compose down

# Regenerate uv.lock from pyproject.toml. Uses a plain uv image (pinned to
# the same uv version as the Dockerfile) instead of the dev image, since the
# dev image's own build depends on uv.lock already existing (COPY + --frozen)
# — resolving the lock can't depend on the frozen install it produces.
lock:
    docker run --rm -v "$(pwd):/workspace" -w /workspace \
        ghcr.io/astral-sh/uv:0.11.21-python3.12-bookworm uv lock

lint:
    {{docker_prefix}} uv run ruff check .

format:
    {{docker_prefix}} uv run ruff format .

format-check:
    {{docker_prefix}} uv run ruff format --check .

format-file file:
    {{docker_prefix}} uv run ruff format {{file}}

pyright-lsp:
    {{docker_prefix}} uv run pyright-langserver --stdio

typecheck:
    {{docker_prefix}} uv run pyright

test:
    {{docker_prefix}} uv run pytest

# Full CI gate.
check: lint format-check typecheck test


benchmark-lake *args:
    {{docker_prefix}} uv run python -m deceris.inundation.benchmark_lake_at_rest {{args}}


shell:
    {{docker_prefix}} bash

# Build the Apptainer SIF for production from the already-built dev image.
apptainer-build:
    apptainer build inundation.sif inundation.def

# Run a command inside the SIF, bind-mounting the repo at /workspace like the
# dev container. e.g. `just apptainer-run -- python -m deceris.inundation.benchmark_lake_at_rest`
apptainer-run *args:
    apptainer exec --nv --bind .:/workspace inundation.sif {{args}}
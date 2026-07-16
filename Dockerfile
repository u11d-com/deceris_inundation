FROM ubuntu:24.04

# Base image for the inundation solver dev/CI environment. Bakes system deps,
# the Python venv (all extras), and the Kompute (kp) Vulkan backend build.
#
# Consumers of this image:
#   - justfile:            docker run --rm -v "$(pwd):/workspace" ... uv run ...
#   - .devcontainer:        VS Code Dev Containers, same image, code bind-mounted
#   - Apptainer (GPU):     inundation.def, Bootstrap: docker-daemon FROM deceris-inundation:dev
#
# Code is NEVER copied into this image — it's bind-mounted at /workspace at
# runtime by all three consumers. Only the files needed to resolve+build the
# venv (pyproject.toml, uv.lock, build-kp-linux.sh, kp-patches/) are copied in.
#
# Production uses the host's NVIDIA driver through Apptainer passthrough.

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        libvulkan1 \
        libvulkan-dev \
        vulkan-tools \
        glslang-tools \
        cmake \
        patch \
        libx11-6 \
        libxext6 \
        libxcb1 \
        libxau6 \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv for reproducible builds.
RUN curl -LsSf https://astral.sh/uv/0.11.21/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /opt/build
COPY pyproject.toml uv.lock ./
COPY deceris/inundation/build-kp-linux.sh ./
COPY deceris/inundation/kp-patches ./kp-patches

RUN uv sync --frozen --python 3.12 --no-install-project --all-extras \
    && chmod +x build-kp-linux.sh \
    && ./build-kp-linux.sh /opt/build/.venv

ENV PATH=/opt/build/.venv/bin:$PATH
ENV PYTHONPATH=/workspace
ENV LD_LIBRARY_PATH=/opt/nvidia/lib
ENV UV_PROJECT_ENVIRONMENT=/opt/build/.venv

WORKDIR /workspace
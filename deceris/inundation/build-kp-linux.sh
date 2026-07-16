#!/usr/bin/env bash
# build-kp-linux.sh — Download, patch, build, and install kp 0.9.0 into a Linux venv.
#
# Linux twin of build-kp.sh (which is macOS/Homebrew-only). Intended for use
# inside the inundation apptainer image build (%post), where system Vulkan
# packages (libvulkan-dev, glslang-tools, cmake, patch) are installed via apt
# before this script runs. Unlike macOS, no MoltenVK ICD is involved here —
# the Linux Vulkan loader discovers ICDs via VK_ICD_FILENAMES/icd.d at
# runtime, not at build time, so this script does not set or check it.
#
# Usage:
#   build-kp-linux.sh [<venv-path>]
#
# Arguments:
#   venv-path   Path to the target Python venv (default: /opt/build/.venv).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${PATCHES_DIR:-${SCRIPT_DIR}/kp-patches}"
DEFAULT_VENV="/opt/build/.venv"
VENV="${1:-${DEFAULT_VENV}}"
PYTHON="${VENV}/bin/python"

KP_VERSION="0.9.0"
KP_TARBALL="kp-${KP_VERSION}.tar.gz"
KP_DIR="kp-${KP_VERSION}"

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[build-kp-linux] $*"; }
error() { echo "[build-kp-linux] ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || error "'$1' not found. Install with: $2"
}

# ── Already installed? ────────────────────────────────────────────────────────

if "${PYTHON}" -c "import kp" 2>/dev/null; then
    KP_INSTALLED=$("${PYTHON}" -c "import kp; print(kp.__version__)" 2>/dev/null || echo "unknown")
    info "kp ${KP_INSTALLED} already installed in ${VENV} — skipping build."
    exit 0
fi

# ── Prerequisite checks ───────────────────────────────────────────────────────

info "Checking prerequisites ..."
require_cmd cmake "apt-get install -y cmake"
require_cmd patch "apt-get install -y patch"
require_cmd "${PYTHON}" "create the venv first"

# uv-created venvs don't ship pip — bootstrap it offline via ensurepip
# (bundled wheel, no network involved) rather than reaching for uv/curl.
"${PYTHON}" -m ensurepip --upgrade

if ! ldconfig -p 2>/dev/null | grep -q libvulkan; then
    error "libvulkan not found. Install with: apt-get install -y libvulkan-dev libvulkan1"
fi

if [[ ! -f "${PATCHES_DIR}/setup.py.patch" \
   || ! -f "${PATCHES_DIR}/Manager.cpp.patch" \
   || ! -f "${PATCHES_DIR}/Sequence.cpp.patch" \
   || ! -f "${PATCHES_DIR}/main.cpp.patch" ]]; then
    error "Patch files not found in '${PATCHES_DIR}'. Is the repo checkout intact?"
fi

info "Prerequisites OK."

# ── Build in a temp directory ─────────────────────────────────────────────────

WORK_DIR="$(mktemp -d -t build-kp-linux-XXXXXX)"
trap 'info "Cleaning up ${WORK_DIR} ..."; rm -rf "${WORK_DIR}"' EXIT

info "Working directory: ${WORK_DIR}"

# Download sdist
info "Downloading kp==${KP_VERSION} source distribution ..."
"${PYTHON}" -m pip download "kp==${KP_VERSION}" --no-deps --no-binary :all: -d "${WORK_DIR}"

TARBALL_PATH="${WORK_DIR}/${KP_TARBALL}"
[[ -f "${TARBALL_PATH}" ]] || error "Expected tarball not found: ${TARBALL_PATH}"

# Extract
info "Extracting ..."
tar -xzf "${TARBALL_PATH}" -C "${WORK_DIR}"
SRC_DIR="${WORK_DIR}/${KP_DIR}"
[[ -d "${SRC_DIR}" ]] || error "Extracted source dir not found: ${SRC_DIR}"

# Apply patches
info "Applying setup.py patch ..."
patch "${SRC_DIR}/setup.py" < "${PATCHES_DIR}/setup.py.patch"

info "Applying Manager.cpp patch ..."
patch "${SRC_DIR}/src/Manager.cpp" < "${PATCHES_DIR}/Manager.cpp.patch"

info "Applying Sequence.cpp patch ..."
patch "${SRC_DIR}/src/Sequence.cpp" < "${PATCHES_DIR}/Sequence.cpp.patch"

info "Applying Python binding patch ..."
patch "${SRC_DIR}/python/src/main.cpp" < "${PATCHES_DIR}/main.cpp.patch"

# Build and install
info "Building and installing kp==${KP_VERSION} (this takes a few minutes) ..."
"${PYTHON}" -m pip install "${SRC_DIR}"

info "Build complete."

# ── Validate ──────────────────────────────────────────────────────────────────
# Skipped here (unlike build-kp.sh): at container *build* time there is no
# guarantee a Vulkan ICD / GPU is visible (no --nv, no bind-mounted host
# driver libs yet), so kp.Manager() would fail even on a correct build.
# Validate at *runtime* instead (see bringup/README.md's Vulkan check step).
info "kp ${KP_VERSION} installed (import-only check) ..."
"${PYTHON}" -c "import kp; print(f'[build-kp-linux] kp {kp.__version__} import OK')"

info "kp ${KP_VERSION} installed successfully."

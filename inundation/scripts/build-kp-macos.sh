#!/usr/bin/env bash
# build-kp-macos.sh — Download, patch, build, and install kp 0.9.0 into a
# macOS venv, backed by MoltenVK (Vulkan-over-Metal).
#
# macOS twin of build-kp-linux.sh. Intended for local dev boxes (Apple
# Silicon or Intel) where kp is not built into a container image. Requires
# Homebrew prerequisites:
#
#   brew install cmake molten-vk vulkan-headers vulkan-loader vulkan-tools glslang
#
# Unlike Linux, macOS Vulkan support comes via MoltenVK's ICD, which must be
# pointed at explicitly through VK_ICD_FILENAMES/VULKAN_SDK both at build
# time (CMake needs to find the Vulkan SDK) and at runtime (kp.Manager()
# needs to find the ICD). This script sets both. Because a real GPU/ICD is
# available at build time on a dev box (unlike the container-build case in
# build-kp-linux.sh), validation here actually exercises kp.Manager().
#
# Usage:
#   build-kp-macos.sh [<venv-path>]
#
# Arguments:
#   venv-path   Path to the target Python venv (default: <repo-root>/.venv).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${PATCHES_DIR:-${SCRIPT_DIR}/../vulkan/patches}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_VENV="${REPO_ROOT}/.venv"
VENV="${1:-${DEFAULT_VENV}}"
PYTHON="${VENV}/bin/python"

KP_VERSION="0.9.0"
KP_TARBALL="kp-${KP_VERSION}.tar.gz"
KP_DIR="kp-${KP_VERSION}"

# ── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[build-kp-macos] $*"; }
error() { echo "[build-kp-macos] ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || error "'$1' not found. Install with: $2"
}

# ── Locate the MoltenVK ICD ───────────────────────────────────────────────────

BREW_PREFIX=""
if command -v brew &>/dev/null; then
    BREW_PREFIX="$(brew --prefix 2>/dev/null || true)"
fi

ICD_CANDIDATES=(
    "${BREW_PREFIX}/etc/vulkan/icd.d/MoltenVK_icd.json"
    "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
    "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json"
)

VK_ICD_FILENAMES=""
for candidate in "${ICD_CANDIDATES[@]}"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
        VK_ICD_FILENAMES="${candidate}"
        break
    fi
done

VULKAN_SDK="${BREW_PREFIX:-/opt/homebrew}"

# ── Already installed? ────────────────────────────────────────────────────────

if "${PYTHON}" -c "import kp" 2>/dev/null; then
    KP_INSTALLED=$("${PYTHON}" -c "import kp; print(kp.__version__)" 2>/dev/null || echo "unknown")
    info "kp ${KP_INSTALLED} already installed in ${VENV} — skipping build."
    exit 0
fi

# ── Prerequisite checks ───────────────────────────────────────────────────────

info "Checking prerequisites ..."
require_cmd cmake "brew install cmake"
require_cmd patch "xcode-select --install (or brew install gpatch)"
require_cmd "${PYTHON}" "create the venv first (uv sync)"

# uv-created venvs don't ship pip — bootstrap it offline via ensurepip
# (bundled wheel, no network involved) rather than reaching for uv/curl.
"${PYTHON}" -m ensurepip --upgrade

if [[ -z "${VK_ICD_FILENAMES}" ]]; then
    error "MoltenVK ICD not found. Install with: brew install molten-vk vulkan-headers vulkan-loader vulkan-tools glslang"
fi

if [[ ! -f "${PATCHES_DIR}/setup.py.patch" \
   || ! -f "${PATCHES_DIR}/Manager.cpp.patch" \
   || ! -f "${PATCHES_DIR}/Sequence.cpp.patch" \
   || ! -f "${PATCHES_DIR}/main.cpp.patch" ]]; then
    error "Patch files not found in '${PATCHES_DIR}'. Is the repo checkout intact?"
fi

info "Prerequisites OK. Using MoltenVK ICD: ${VK_ICD_FILENAMES}"

# ── Build in a temp directory ─────────────────────────────────────────────────

WORK_DIR="$(mktemp -d -t build-kp-macos-XXXXXX)"
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

# The bundled fmt/vulkan.hpp headers trigger -Wdeprecated-declarations and
# -Wdeprecated-literal-operator under recent Xcode/libc++, and kp's
# CMakeLists.txt enables -Werror, turning these into hard build failures.
# Not covered by the repo's patch files (Linux toolchains don't hit this),
# so downgrade them back to warnings via CXXFLAGS instead of patching
# setup.py directly.
EXTRA_CXXFLAGS="-Wno-error=deprecated-declarations -Wno-error=deprecated-literal-operator"

# Build and install
info "Building and installing kp==${KP_VERSION} (this takes a few minutes) ..."
VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" VULKAN_SDK="${VULKAN_SDK}" CXXFLAGS="${EXTRA_CXXFLAGS}" \
    "${PYTHON}" -m pip install "${SRC_DIR}"

info "Build complete."

# ── Validate ──────────────────────────────────────────────────────────────────
# Unlike build-kp-linux.sh, a real GPU/ICD is available at build time on a
# dev box, so we validate with an actual kp.Manager() round-trip rather than
# an import-only check.
info "Validating kp.Manager() against MoltenVK ..."
VK_ICD_FILENAMES="${VK_ICD_FILENAMES}" "${PYTHON}" -c "
import os
os.environ.setdefault('VK_ICD_FILENAMES', '${VK_ICD_FILENAMES}')
import kp
mgr = kp.Manager()
t = mgr.tensor([1.0, 2.0, 3.0])
print(f'[build-kp-macos] kp {kp.__version__} OK: {t.data()}')
"

info "kp ${KP_VERSION} installed successfully."
info "Remember to set VK_ICD_FILENAMES=${VK_ICD_FILENAMES} when running kp at runtime."

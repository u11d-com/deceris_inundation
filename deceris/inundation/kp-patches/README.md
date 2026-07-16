This directory contains patches required to build kp==0.9.0 on macOS ARM64 with CMake 4.x and MoltenVK.

## Patches

### 1. setup.py — Fix missing comma + CMake 4.x compat

File: `setup.py`

Two changes on lines 47-51:
- Add missing comma after `-DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON` (was
  being concatenated with `-DPYTHON_EXECUTABLE=...` as one string, causing a
  malformed CMake flag)
- Add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to allow the bundled pybind11 v2.9.2
  (which uses `cmake_minimum_required(VERSION 2.8.12)`) to build under CMake 4.x

### 2. src/Manager.cpp — MoltenVK portability enumeration

File: `src/Manager.cpp`, `createInstance()` method (~line 167)

MoltenVK is a Vulkan "portability driver". Since Vulkan 1.1, applications must:
1. Enable the `VK_KHR_portability_enumeration` instance extension
2. Set `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` on `VkInstanceCreateInfo.flags`

Without this, `vkEnumeratePhysicalDevices` returns an invalid instance error and
`kp.Manager()` hangs indefinitely on macOS.

The fix detects whether the extension is available and enables it if so.

## Build Instructions (macOS ARM64)

```sh
# Prerequisites
brew install python@3.12 cmake molten-vk vulkan-headers vulkan-loader vulkan-tools glslang

# Download kp source
pip3 download kp==0.9.0 --no-deps --no-binary :all: -d ./kp-sdist
tar xzf kp-sdist/kp-0.9.0.tar.gz

# Apply patches (or use the patch files in this directory)
patch kp-0.9.0/setup.py < setup.py.patch
patch kp-0.9.0/src/Manager.cpp < Manager.cpp.patch

# Build and install
cd kp-0.9.0
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
  VULKAN_SDK=/opt/homebrew \
  pip install .

# Validate
python -c "
import os
os.environ.setdefault('VK_ICD_FILENAMES', '/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json')
import kp
mgr = kp.Manager()
t = mgr.tensor([1.0, 2.0, 3.0])
print(f'kp {kp.__version__} OK: {t.data()}')
"
```

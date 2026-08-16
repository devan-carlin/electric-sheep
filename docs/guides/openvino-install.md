# OpenVINO Install Guide (Linux)

**Created**: 2026-08-16
**Hardware**: ai-server (Ubuntu 26.04, 4x Intel Arc Pro B70, Threadripper PRO 3945WX)
**Upstream reference**: [openvino/docs/dev/build_linux.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md)

---

## Overview

OpenVINO is Intel's inference runtime for CPU, GPU, and NPU. Two install paths:

1. **Prebuilt wheel (recommended first)** — `pip install openvino`. Fast, no compiler needed.
2. **Build from source** — needed for custom plugins, patches, or a version newer than the wheel.

Upstream validation matrix (from build_linux.md): Ubuntu 18.04/20.04, RHEL 8.2. Our Ubuntu 26.04 is newer than validated — expect to hit dependency issues and resolve them manually if the helper script lags.

---

## Path 1: Prebuilt Wheel

```bash
python3 -m venv .venv-ov
source .venv-ov/bin/activate
pip install openvino
```

Verify:

```bash
python3 -c "import openvino as ov; print(ov.get_version())"
```

List supported devices:

```bash
python3 -c "import openvino as ov; print(ov.Core().available_devices)"
```

Expected: `CPU` always; `GPU` only if the Intel GPU Compute Runtime (OpenCL) driver is present and the GPU is supported by the OpenVINO GPU plugin.

---

## Path 2: Build from Source

### Software Requirements

- CMake 3.26 or higher
- GCC 7.5 or higher
- Python 3.9 - 3.12 (for the Python API)
- Optional: Intel GPU Compute Runtime for OpenCL Driver (enables GPU inference)

### Steps

```bash
# 1. Clone and init submodules
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git submodule update --init --recursive

# 2. Install build dependencies (needs sudo)
sudo ./install_build_dependencies.sh

# 3. Create build folder
mkdir build && cd build

# 4. Configure and build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

Use `nproc` to size the parallel job count; cap it on memory-constrained boxes (e.g. `--parallel 8`).

### Important: Disable the oneAPI Environment First

Upstream explicitly warns: **disable the oneAPI environment before compiling OpenVINO from source on Linux, as it may cause build failures.**

On this box that means do NOT have `source /opt/intel/oneapi/setvars.sh` active in the build shell. Use a clean shell:

```bash
# if setvars was already sourced:
deactivate 2>/dev/null; unset IPEXROOT ONEAPI_ROOT 2>/dev/null
# or just open a fresh terminal that never sourced setvars.sh
```

This matters here because our normal workflow sources oneAPI setvars for the XPU toolchain.

### Build Outputs

- `bin/intel64/Release/` — runtime libraries and tools (`benchmark_app`, `query`, `mo`/`ovc`)
- `lib/intel64/Release/` — shared libraries

Smoke test with the query tool:

```bash
export LD_LIBRARY_PATH=$PWD/bin/intel64/Release:$LD_LIBRARY_PATH
./bin/intel64/Release/query
```

---

## Additional Build Options

### Python API

Add to the cmake step:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON ..
# pin an exact interpreter (cmake >= 3.16):
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=/usr/bin/python3.12 ..
```

After build, either export env vars:

```bash
export PYTHONPATH=<openvino_repo>/bin/intel64/Release/python:<openvino_repo>/tools/ovc:$PYTHONPATH
export LD_LIBRARY_PATH=<openvino_repo>/bin/intel64/Release:$LD_LIBRARY_PATH
export PATH=<openvino_repo>/tools/ovc/openvino/tools/ovc:$PATH
```

or build and install a wheel:

```bash
# at cmake time:
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON ..
# wheel build requirements:
pip install -r <openvino source tree>/src/bindings/python/wheel/requirements-dev.txt
# after build:
pip install <openvino_repo>/build/wheel/openvino-*.whl
```

### Custom Compilation (faster builds)

CMake options exist to skip plugins/frontends you do not need. See:
[cmake_options_for_custom_compilation.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md)

### IA32 Toolchain

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=<openvino_repo>/cmake/toolchains/ia32.linux.toolchain.cmake ..
```

Not relevant for this box (x86_64 only).

---

## Arc B70 (Battlemage) Notes

- The OpenVINO GPU plugin requires the Intel GPU Compute Runtime (OpenCL) driver. The `xe` driver stack on this box provides Level Zero; confirm the OpenCL runtime is installed separately if `GPU` does not appear in `available_devices`.
- Battlemage (Xe2) support in the OpenVINO GPU plugin is newer than Alchemist — verify with `query` / `benchmark_app` before assuming GPU inference works. If the plugin does not list the B70, CPU inference still works.
- Do not mix an active oneAPI setvars environment with the source build (see above).

---

## Related Docs in the OpenVINO Repo

- [build.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md) — build overview (all platforms)
- [cmake_options_for_custom_compilation.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md) — CMake option reference
- [get_started.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/get_started.md) — developer quick start
- [docs/dev/index.md](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/index.md) — developer documentation index
- [Building OpenVINO on CentOS 7 (wiki)](https://github.com/openvinotoolkit/openvino/wiki/Building-OpenVINO-on-CentOS-7-Guide) — legacy distro notes

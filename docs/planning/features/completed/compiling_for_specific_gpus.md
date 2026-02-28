# Plan: Compile TRITON-SWMM for Specific GPUs (Single-Arch Builds)

**Status:** Draft (implementation-ready)
**Owner:** Toolkit maintainers
**Created:** 2026-02-11

## Goal

Update the TRITON-SWMM compilation scripts generated in
`src/TRITON_SWMM_toolkit/system.py` to align with hardware-specific CUDA builds.
Kokkos in this repo rejects multi-arch CUDA builds, so we will compile **one
GPU architecture at a time** and select the appropriate flags based on
`gpu_hardware`.

## Target GPU Build Script (Template)

```bash
TRITON_DIR=/path/to/triton
BUILD_DIR=/path/to/triton/build_tritonswmm_gpu_<arch>

module purge
module load <from additional_modules config>

MPICXX=$(which mpic++)
MPICC=$(which mpicc)
NVCC=$(which nvcc)

mkdir -p "$BUILD_DIR"
rm -rf "$BUILD_DIR"/CMakeFiles "$BUILD_DIR"/CMakeCache.txt \
        "$BUILD_DIR"/Makefile "$BUILD_DIR"/cmake_install.cmake
cd "$BUILD_DIR"

cmake \
    -DTRITON_ENABLE_SWMM=ON \
    -DTRITON_SWMM_FLOODING_DEBUG=ON \
    -DTRITON_IGNORE_MACHINE_FILES=ON \
    -DTRITON_BACKEND=CUDA \
    -DTRITON_ARCH=<ARCH_FROM_GPU_HARDWARE> \
    -DTRITON_COMPILER_FLAGS='-DACTIVE_GPU=1' \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_<ARCH_FLAG>=ON \
    -DKokkos_ENABLE_OPENMP=OFF \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CXX_COMPILER="$MPICXX" \
    -DCMAKE_C_COMPILER="$MPICC" \
    -DCMAKE_CXX_FLAGS='-O3' \
    "$TRITON_DIR"

make -j4

echo "Build finished"
```

## Files in Scope

- `src/TRITON_SWMM_toolkit/system.py`
  - `TRITONSWMM_system._compile_backend()` (TRITON-SWMM builds)
  - `TRITONSWMM_system._compile_triton_only_backend()` (TRITON-only builds)

## Mapping: Script Lines → Code Changes

### 1) Module Environment
**Script:**
```bash
module purge
module load <from additional_modules config>
```
**Code updates:**
- Keep module loading driven by
  `additional_modules_needed_to_run_TRITON_SWMM_on_hpc`.
- Ensure a `module purge` precedes the module load block.

### 2) Compiler Bindings
**Script:**
```bash
MPICXX=$(which mpic++)
MPICC=$(which mpicc)
NVCC=$(which nvcc)
```
**Code updates:**
- Add these environment variables to the GPU build script for both TRITON-SWMM
  and TRITON-only paths.
- Inject CMake toolchain flags:
  - `-DCMAKE_CUDA_COMPILER="$NVCC"`
  - `-DCMAKE_CXX_COMPILER="$MPICXX"`
  - `-DCMAKE_C_COMPILER="$MPICC"`

### 3) Build Directory + Cleanup
**Script:**
```bash
mkdir -p "$BUILD_DIR"
rm -rf "$BUILD_DIR"/CMakeFiles "$BUILD_DIR"/CMakeCache.txt \
        "$BUILD_DIR"/Makefile "$BUILD_DIR"/cmake_install.cmake
cd "$BUILD_DIR"
```
**Code updates:**
- Ensure GPU scripts in `system.py` use this exact clean-up block (currently
  matches pattern but should be kept in sync for GPU path).
- Keep `TRITON_DIR` and `BUILD_DIR` set explicitly at the top of the script.

### 4) GPU-Specific CMake Flags (Single-Arch)
**Script:**
```bash
-DTRITON_ENABLE_SWMM=ON
-DTRITON_SWMM_FLOODING_DEBUG=ON
-DTRITON_IGNORE_MACHINE_FILES=ON
-DTRITON_BACKEND=CUDA
-DTRITON_ARCH=<ARCH_FROM_GPU_HARDWARE>
-DTRITON_COMPILER_FLAGS='-DACTIVE_GPU=1'
-DKokkos_ENABLE_CUDA=ON
-DKokkos_ARCH_<ARCH_FLAG>=ON
-DKokkos_ENABLE_OPENMP=OFF
-DCMAKE_CUDA_COMPILER="$NVCC"
-DCMAKE_CXX_COMPILER="$MPICXX"
-DCMAKE_C_COMPILER="$MPICC"
-DCMAKE_CXX_FLAGS='-O3'
```
**Code updates:**
- Replace the current GPU `cmake_flags` string in `_compile_backend` with a
  dynamically generated string derived from `gpu_hardware`.
- Update TRITON-only GPU compilation (`_compile_triton_only_backend`) with the
  same GPU CMake flags, except use:
  - `-DTRITON_ENABLE_SWMM=OFF`
  - Keep `TRITON_BACKEND=CUDA` and the selected `TRITON_ARCH` / Kokkos flag.

### 5) CMake Invocation Path
**Script:**
```bash
cmake ... "$TRITON_DIR"
```
**Code updates:**
- Replace the `..` CMake source path in GPU scripts with the explicit
  `"$TRITON_DIR"` path (matches A6000 script).

### 6) Completion Marker
**Script:**
```bash
echo "Build finished"
```
**Code updates:**
- Replace `echo 'script finished'` with `echo "Build finished"` to keep the
  build success marker consistent with the A6000 script.

## GPU Hardware → Flag Mapping

Kokkos in this repo does **not** support multi-arch CUDA builds. Select exactly
one `Kokkos_ARCH_*` flag and matching `TRITON_ARCH` based on `gpu_hardware`:

| gpu_hardware | TRITON_ARCH | Kokkos flag | Notes |
| --- | --- | --- | --- |
| `rtx3090`, `a6000` | `AMPERE80` | `Kokkos_ARCH_AMPERE86=ON` | Ampere 8.6 |
| `a100` | `AMPERE80` | `Kokkos_ARCH_AMPERE80=ON` | Ampere 8.0 |
| `h100`, `h200` | `HOPPER90` | `Kokkos_ARCH_HOPPER90=ON` | Hopper 9.0 |
| `v100` | `VOLTA70` | `Kokkos_ARCH_VOLTA70=ON` | TRITON_ARCH unverified; test first |
| `rtx2080` | `TURING75` | `Kokkos_ARCH_TURING75=ON` | TRITON_ARCH unverified; test first |

> Note: `TRITON_ARCH` is a **single value**, even for Ampere GPUs. Use
> `AMPERE80` and vary only the Kokkos flag between `AMPERE80` and `AMPERE86`.
> For V100/RTX2080, consider using a fallback `TRITON_ARCH=AMPERE80` if
> unverified values cause issues, while keeping the Kokkos flag for tuning.

## Implementation Steps

1. **Update TRITON-SWMM GPU compile script**
   - Edit `_compile_backend()` GPU branch.
   - Inject module purge + config-driven module loads.
   - Add `MPICXX`, `MPICC`, `NVCC` variables (use `which nvcc`).
   - Build GPU CMake flags from `gpu_hardware` mapping above.
   - Use `"$TRITON_DIR"` as the CMake source path.
   - Replace completion marker with `Build finished`.

2. **Update TRITON-only GPU compile script**
   - Edit `_compile_triton_only_backend()` GPU branch with the same
     module/compiler blocks and CMake flags, but keep
     `-DTRITON_ENABLE_SWMM=OFF`.
   - Use the same CMake source path and completion marker.

3. **Keep CPU path unchanged (for now)**
   - CPU compilation remains as-is to avoid unintended impacts.

4. **Config decision (required)**
   - Decide how `gpu_hardware` is supplied during compilation.
     - Option A: mirror `gpu_hardware` in `cfg_system` for compile-time access.
     - Option B: pass `gpu_hardware` through from analysis config when calling
       compilation.
   - Document the chosen source in the compilation methods.

## Risks / Notes

- Kokkos multi-arch builds are not supported in this repo; using multiple
  `Kokkos_ARCH_*` flags will error out.
- Ensure the `gpu_hardware` mapping stays in sync with real cluster hardware.
- Always use per-GPU build directories to avoid cached CMake defaults (e.g.,
  `TRITON_BACKEND` reverting to SERIAL).
- `TRITON_COMPILER_FLAGS='-DACTIVE_GPU=1'` is preserved to match the working
  script; ensure downstream build logic expects this flag.

## Success Criteria

- Generated GPU compilation scripts in `system.py` select exactly one Kokkos
  arch flag based on `gpu_hardware`.
- TRITON-SWMM and TRITON-only GPU builds both use the same toolchain setup and
  hardware-derived flags.
- CPU compilation flow remains unaffected.

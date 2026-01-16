#!/bin/bash
module purge
module load PrgEnv-amd Core/24.07 cmake/3.27.9 craype-accel-amd-gfx90a

export ROCM_PATH=${CRAY_AMD_COMPILER_PREFIX}
export CXX=CC
export MPICH_GPU_SUPPORT_ENABLED=1
unset HSA_XNACK
export TRITON_BACKEND="Kokkos_ENABLE_HIP"
export TRITON_ARCH="Kokkos_ARCH_AMD_GFX90A"
export TRITON_CXX_FLAGS="-DTRITON_HIP_LAUNCHER;-O3;-ffast-math;-I${ROCM_PATH}/include;-D__HIP_ROCclr__;-D__HIP_ARCH_GFX90A__=1;--rocm-path=${ROCM_PATH};--offload-arch=gfx90a;-Wno-unused-result;-Wno-macro-re
defined"
export TRITON_LINK_FLAGS="--rocm-path=${ROCM_PATH};-L${ROCM_PATH}/lib;-lamdhip64"
export TRITON_DEBUG=OFF
export CRAYPE_LINK_TYPE=dynamic
unset HAVE_GDAL
unset CXXFLAGS
unset FFLAGS
unset F77FLAGS
unset F90FLAGS 

# my modifications
export CRAY_CPU_TARGET=x86-64
export MPICH_DIR=/opt/cray/pe/mpich/8.1.31/ofi/amd/6.0
export HIP_LIB_PATH=/opt/rocm-6.2.4/lib

model=${COMPILED_MODEL_DIR}
# cd ${model}
# source modules_frontier.sh
swmm_build="${model}/Stormwater-Management-Model/build"

# create build folder
mkdir -p "${swmm_build}"
cd "${swmm_build}" || exit 1

# cmake
cmake "${model}/Stormwater-Management-Model"
cmake --build .

# set library path
export LD_LIBRARY_PATH="${swmm_build}/bin:$LD_LIBRARY_PATH"

# compile SWMM with TRITON
cd "${model}" || exit 1
make frontier_swmm_gpu
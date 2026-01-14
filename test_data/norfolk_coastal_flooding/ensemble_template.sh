#!/bin/bash
#SBATCH -o _slurm_outputs/%x/%A_outputs_%a_%N.out
#SBATCH -e _slurm_outputs/%x/%A_errors_%a_%N.out
#SBATCH -A ${allocation}
#SBATCH -t ${time}
#SBATCH -p ${partition}
#SBATCH -N ${nodes}
#SBATCH --exclusive
# SBATCH --cpus-per-task=${cpus_per_task}   # CPU=1, GPU=threads per GPU
# SBATCH --ntasks-per-node=${ntasks_per_node} # GPU=tasks per node, CPU optional
#${gpu_toggle}SBATCH --gres=${gres}
#SBATCH --mail-user=***REMOVED***@virginia.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug

# cd /lustre/orion/***REMOVED***/proj-shared/***REMOVED***
# salloc -A ***REMOVED*** -p batch -t 0-02:00:00 -N 1 --exclusive -q debug

set -euo pipefail

# Load modules
module purge
module load PrgEnv-amd Core/24.07 cmake/3.27.9 craype-accel-amd-gfx90a
module load miniforge3/23.11.0
DIR=~/.conda/envs/running_swmm
conda activate triton_swmm_toolkit

echo "Node CPUs: $(nproc)"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"

export ROCM_PATH=${CRAY_AMD_COMPILER_PREFIX}
export CXX=CC
export MPICH_GPU_SUPPORT_ENABLED=1
export TRITON_BACKEND="Kokkos_ENABLE_HIP"
export TRITON_ARCH="Kokkos_ARCH_AMD_GFX90A"
export TRITON_CXX_FLAGS="-DTRITON_HIP_LAUNCHER;-O3;-ffast-math;-I${ROCM_PATH}/include;-D__HIP_ROCclr__;-D__HIP_ARCH_GFX90A__=1;--rocm-path=${ROCM_PATH};--offload-arch=gfx90a;-Wno-unused-result;-Wno-macro-re
defined"
export TRITON_LINK_FLAGS="--rocm-path=${ROCM_PATH};-L${ROCM_PATH}/lib;-lamdhip64"
export TRITON_DEBUG=OFF
export CRAYPE_LINK_TYPE=dynamic
export CRAY_CPU_TARGET=x86-64
export MPICH_DIR=/opt/cray/pe/mpich/8.1.31/ofi/amd/6.0
export HIP_LIB_PATH=/opt/rocm-6.2.4/lib

# Ensemble simulations from template
${run_command}
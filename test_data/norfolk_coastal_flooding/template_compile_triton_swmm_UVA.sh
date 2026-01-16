#!/bin/bash
module load gcc openmpi

model=${COMPILED_MODEL_DIR}
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
make ${MAKE_COMMAND}
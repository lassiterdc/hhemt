import os
import pytest
import socket
import subprocess
import time
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")

# ijob \
#   -A ***REMOVED*** \
#   -p interactive \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=24

# module purge
# module load gompi/14.2.0_5.0.7 miniforge
# source activate triton_swmm_toolkit
# export PYTHONNOUSERSITE=1

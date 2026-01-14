import os
import pytest
import socket


pytestmark = pytest.mark.skipif(
    "SLURM_JOB_ID" not in os.environ,
    reason="Skipping this test script because it requires SLURM/HPC environment",
)

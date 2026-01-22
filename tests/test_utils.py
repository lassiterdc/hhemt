import os
import pytest
import socket


def uses_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def is_scheduler_context() -> bool:
    scheduler_vars = (
        "SLURM_JOB_ID",  # SLURM
        "PBS_JOBID",  # PBS
        "LSB_JOBID",  # LSF
        "COBALT_JOBID",  # Cobalt
    )
    return any(v in os.environ for v in scheduler_vars)


def on_frontier() -> bool:
    return "frontier" in socket.getfqdn()


def on_UVA_HPC() -> bool:
    return "virginia" in socket.getfqdn()

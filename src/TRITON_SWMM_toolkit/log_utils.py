"""
Logging utilities for workflow traceability.

Provides helper functions for logging workflow context in runner scripts,
enabling easy correlation between master workflow logs, SLURM logs, and
runner script logs during debugging.
"""

import logging
import os
from pathlib import Path


def log_workflow_context(logger: logging.Logger):
    """
    Log SLURM/workflow context for traceability.

    Should be called at the start of each runner script to create an audit trail
    linking runner logs back to their master Snakemake workflow run.

    Only logs detailed context when running in a SLURM environment. For local runs,
    logs a simple message instead to avoid cluttering logs with "N/A" values.

    This logs (when in SLURM):
    - Master SLURM job ID (for correlating with workflow_batch_*.out logs)
    - Child SLURM job ID (for correlating with .snakemake/slurm_logs/)
    - Node name (which HPC node executed this)
    - Partition (which SLURM partition)
    - CPU allocation (for debugging resource issues)

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to write context information to

    Examples
    --------
    >>> import logging
    >>> from TRITON_SWMM_toolkit.log_utils import log_workflow_context
    >>>
    >>> logger = logging.getLogger(__name__)
    >>> log_workflow_context(logger)
    """
    # Only log detailed context if running in SLURM
    in_slurm = "SLURM_JOB_ID" in os.environ

    if not in_slurm:
        logger.info("Execution environment: local (not SLURM)")
        return

    logger.info("=" * 60)
    logger.info("WORKFLOW CONTEXT")
    logger.info("=" * 60)

    # Master workflow info (for linking to workflow_batch_*.out)
    # In array jobs, SLURM_ARRAY_JOB_ID is the master
    # In batch_job mode, each job is independent (no master)
    master_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "N/A")
    if master_job_id == "N/A":
        # Not an array job - check if we're in a nested SLURM context
        # (master workflow also runs in SLURM with batch_job mode)
        master_job_id = os.environ.get("SLURM_JOB_ACCOUNT", "N/A")

    logger.info(f"Master SLURM job ID: {master_job_id}")

    # Child job info (current SLURM job running this script)
    child_job_id = os.environ.get("SLURM_JOB_ID", "local")
    logger.info(f"Current SLURM job ID: {child_job_id}")

    # Execution environment
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME", "local")
    logger.info(f"Node: {node}")

    partition = os.environ.get("SLURM_JOB_PARTITION", "local")
    logger.info(f"Partition: {partition}")

    # Resource allocation (useful for debugging allocation issues)
    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE", "N/A")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "N/A")
    ntasks = os.environ.get("SLURM_NTASKS", "N/A")
    logger.info(
        f"CPUs: {cpus_on_node} total ({ntasks} tasks × {cpus_per_task} CPUs/task)"
    )

    # GPU allocation (if applicable)
    gpus = os.environ.get("SLURM_JOB_GPUS", "N/A")
    if gpus != "N/A":
        logger.info(f"GPUs: {gpus}")

    logger.info("=" * 60)

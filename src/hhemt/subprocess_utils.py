"""
Subprocess utilities for unified logging across the TRITON-SWMM toolkit.

Provides utilities for running subprocesses with "tee" logging -
writing to both a local log file AND stdout so Snakemake can capture.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional


def run_subprocess_with_tee(
    cmd: List[str],
    logfile: Path,
    env: Optional[Dict] = None,
    cwd: Optional[Path] = None,
    echo_to_stdout: bool = True,
) -> subprocess.Popen:
    """
    Run a subprocess with output written to both a file AND stdout.

    This enables:
    - Detailed scenario-level logs preserved at logfile path
    - Output echoed to stdout so Snakemake captures it in centralized logs

    Parameters
    ----------
    cmd : List[str]
        Command to execute
    logfile : Path
        Path to write local log file
    env : dict, optional
        Environment variables for subprocess
    cwd : Path, optional
        Working directory for subprocess
    echo_to_stdout : bool
        If True, echo output to stdout (default True)

    Returns
    -------
    subprocess.Popen
        The process object (already waited for completion)
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=merged_env,
        cwd=str(cwd) if cwd else None,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Stream output to both file and stdout
    with open(logfile, "w") as lf:
        if proc.stdout:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                if echo_to_stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()

    proc.wait()
    if proc.stdout:
        proc.stdout.close()
    return proc

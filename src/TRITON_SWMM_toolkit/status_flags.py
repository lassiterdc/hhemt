"""Toolkit-managed status-flag writes.

Replaces rule-shell `touch {output}` directives in `workflow.py`. Provides a
single atomic-write helper that emits the bare `_status/*.flag` marker
(Snakemake-visible per `--rerun-triggers mtime input`) alongside an optional
`_status/*.flag.json` payload sidecar carrying per-rule diagnostic provenance.

The bare flag's mtime is decoupled from the sidecar's content — sidecar
re-emission does NOT bump the flag's mtime, so downstream rules do not
cascade-rerun on payload-only changes.

Module per D-FlagStorage Option 1 and D-FlagPayloadSchema Option 1.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_status_flag(
    flag_path: Path,
    *,
    rule_name: str,
    model_type: str | None = None,
    sa_id: str | None = None,
    event_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write a `_status/*.flag` marker + `*.flag.json` sidecar.

    Both files are written via `tempfile.mkstemp(dir=flag_path.parent) + os.replace`
    so the temp lives on the same filesystem as the target (POSIX-atomic on
    Lustre/GPFS). The bare flag is zero-byte; payload lives in the sidecar.

    Parameters
    ----------
    flag_path : Path
        Absolute path to the `_status/*.flag` file. Parent directory is created
        if it does not exist.
    rule_name : str
        The Snakemake rule that produced this flag (e.g., `run_tritonswmm`,
        `process_sa_42_evt_3`, `delete_subanalysis_5`).
    model_type, sa_id, event_id : str | None
        Per-rule diagnostic provenance; included in the sidecar payload when
        provided. None values are omitted from the JSON.
    extra : dict[str, Any] | None
        Optional caller-provided fields appended to the sidecar payload under
        the `"extra"` key.

    Raises
    ------
    OSError
        On filesystem write failure. The function logs and re-raises; partial
        state is cleaned up via the temp-file's `try/finally`.
    """
    flag_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write the bare flag (zero-byte, Snakemake-visible).
    fd, tmp = tempfile.mkstemp(prefix=".tmp.flag.", suffix="", dir=flag_path.parent)
    try:
        os.close(fd)
        os.replace(tmp, flag_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

    # 2. Write the sidecar payload (decoupled mtime).
    sidecar = flag_path.with_suffix(flag_path.suffix + ".json")
    payload: dict[str, Any] = {
        "rule_name": rule_name,
        "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if model_type is not None:
        payload["model_type"] = model_type
    if sa_id is not None:
        payload["sa_id"] = sa_id
    if event_id is not None:
        payload["event_id"] = event_id
    if extra:
        payload["extra"] = extra

    fd, tmp = tempfile.mkstemp(prefix=".tmp.flagjson.", suffix=".json", dir=sidecar.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, sidecar)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

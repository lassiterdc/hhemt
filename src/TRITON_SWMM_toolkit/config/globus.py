"""Pydantic models for Globus transfer configuration.

These models use plain BaseModel (NOT cfgBaseModel) because cfgBaseModel
validates all Path fields exist on disk — HPC paths like /lustre/orion/...
or /scratch/... do not exist on the local machine and would raise ValueError
at load time.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class GlobusEndpoints(BaseModel):
    """Source and destination Globus collection UUIDs for a transfer."""

    model_config = ConfigDict(extra="forbid")

    source_uuid: str
    destination_uuid: str


class GlobusTransferItem(BaseModel):
    """A single source→destination path pair within a transfer."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    destination_path: str
    recursive: bool = True


class GlobusTransferSpec(BaseModel):
    """Complete specification for a Globus transfer loaded from YAML.

    Attributes:
        label:        Human-readable label shown in Globus task monitor.
        endpoints:    Source and destination collection UUIDs.
        items:        List of path pairs to transfer.
        sync_level:   Globus sync level (0=exists, 1=size, 2=mtime, 3=checksum).
                      Default 2 (mtime) skips files that haven't changed.
        notify_on_succeeded: Send Globus email notification on success.
        notify_on_failed:    Send Globus email notification on failure.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    endpoints: GlobusEndpoints
    items: list[GlobusTransferItem]
    sync_level: int = 2
    notify_on_succeeded: bool = False
    notify_on_failed: bool = True
    deadline_minutes: Optional[int] = None

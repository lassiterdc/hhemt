"""Pydantic models for Globus transfer configuration.

These models use plain BaseModel (NOT cfgBaseModel) because cfgBaseModel
validates all Path fields exist on disk — HPC paths like /lustre/orion/...
or /scratch/... do not exist on the local machine and would raise ValueError
at load time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


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
    deadline_minutes: int | None = None


# ---------------------------------------------------------------------------
# Helpers for PostRunTransferConfig
# ---------------------------------------------------------------------------

_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[/\\]")


def _normalize_wsl_path(path: str) -> str:
    """Translate a Windows-style path to a WSL mount path.

    Accepts ``D:\\Dropbox\\foo`` or ``D:/Dropbox/foo`` and returns
    ``/mnt/d/Dropbox/foo``.  Paths that are already POSIX are returned
    unchanged.

    Raises:
        ValueError: If the path uses an unsupported format (e.g. UNC paths).
    """
    if path.startswith("\\\\") or path.startswith("//"):
        raise ValueError(f"UNC paths are not supported for WSL translation: {path}")
    m = _WINDOWS_DRIVE_RE.match(path)
    if m:
        drive = m.group(1).lower()
        rest = path[m.end() :].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path


def _get_endpoint_uuids(system: str) -> tuple[str, str]:
    """Return ``(source_uuid, scratch_base)`` for *system*.

    Raises:
        ConfigurationError: If *system* is unknown or its UUID is None.
    """
    from TRITON_SWMM_toolkit.constants import GLOBUS_SYSTEM_ENDPOINTS
    from TRITON_SWMM_toolkit.exceptions import ConfigurationError

    entry = GLOBUS_SYSTEM_ENDPOINTS.get(system)
    if entry is None:
        raise ConfigurationError(
            field="system",
            message=(f"Unknown Globus system '{system}'. " f"Valid systems: {sorted(GLOBUS_SYSTEM_ENDPOINTS)}"),
        )
    source_uuid, scratch_base = entry
    if source_uuid is None:
        raise ConfigurationError(
            field="system",
            message=f"Globus collection UUID for '{system}' is not configured.",
        )
    return source_uuid, scratch_base


# ---------------------------------------------------------------------------
# User-facing transfer config
# ---------------------------------------------------------------------------

# Default directories excluded from post-run transfers.
DEFAULT_EXCLUDE_PATTERNS = [
    "subanalyses/",
    "sims/out_triton/",
    "sims/out_tritonswmm/",
    "sims/out_swmm/",
]


class PostRunTransferConfig(BaseModel):
    """User-facing configuration for automatic post-run Globus transfers.

    This model captures *user intent* and translates it into a
    :class:`GlobusTransferSpec` via :meth:`to_transfer_spec`.

    Attributes:
        destination_root: Local destination root (Windows or POSIX path).
            A subdirectory named after the analysis_id is created under this.
        system: HPC system name (e.g. ``"frontier"``, ``"uva"``).
        exclude_patterns: Directory patterns to exclude.
            Defaults to raw output dirs.
        include_sims: If set, transfer only these simulation event indices.
        conflict_policy: What to do if the destination subdirectory exists.
        sync_level: Globus sync level (0=exists, 1=size, 2=mtime, 3=checksum).
        label: Human-readable label for the Globus task.
        wait_for_transfer: Block until the transfer completes.
        timeout_minutes: Max wait time before raising TimeoutError.
    """

    model_config = ConfigDict(extra="forbid")

    destination_root: str
    system: str
    exclude_patterns: list[str] = DEFAULT_EXCLUDE_PATTERNS
    include_sims: list[int] | None = None
    conflict_policy: Literal["prompt", "archive", "clear"] = "prompt"
    sync_level: int = 0
    label: str | None = None
    wait_for_transfer: bool = True
    timeout_minutes: int | None = None

    @model_validator(mode="after")
    def _validate_system(self) -> PostRunTransferConfig:
        """Eagerly validate that the system has a known endpoint UUID."""
        _get_endpoint_uuids(self.system)
        return self

    def to_transfer_spec(
        self,
        analysis_dir: Path,
        analysis_id: str,
    ) -> GlobusTransferSpec:
        """Build a :class:`GlobusTransferSpec` from user intent.

        Args:
            analysis_dir: Absolute path to the analysis directory on the HPC.
            analysis_id: Analysis identifier used as the destination subdirectory
                name.

        Returns:
            A fully-populated :class:`GlobusTransferSpec` ready for
            :meth:`GlobusTransferManager.transfer`.
        """
        from TRITON_SWMM_toolkit.constants import DESKTOP_GLOBUS_COLLECTION_UUID

        source_uuid, _scratch_base = _get_endpoint_uuids(self.system)
        dest_root = _normalize_wsl_path(self.destination_root)
        dest_path = f"{dest_root.rstrip('/')}/{analysis_id}"
        source_path = str(analysis_dir)

        # Build transfer items
        items: list[GlobusTransferItem] = []
        if self.include_sims is not None:
            # Transfer specific simulation directories only
            for sim_idx in self.include_sims:
                items.append(
                    GlobusTransferItem(
                        source_path=f"{source_path}/sims/{sim_idx}",
                        destination_path=f"{dest_path}/sims/{sim_idx}",
                        recursive=True,
                    )
                )
            # Also transfer top-level files (configs, logs, status)
            items.append(
                GlobusTransferItem(
                    source_path=source_path,
                    destination_path=dest_path,
                    recursive=True,
                )
            )
        else:
            items.append(
                GlobusTransferItem(
                    source_path=source_path,
                    destination_path=dest_path,
                    recursive=True,
                )
            )

        label = self.label or f"TRITON-SWMM {analysis_id} → {os.path.basename(dest_root)}"

        return GlobusTransferSpec(
            label=label,
            endpoints=GlobusEndpoints(
                source_uuid=source_uuid,
                destination_uuid=DESKTOP_GLOBUS_COLLECTION_UUID,
            ),
            items=items,
            sync_level=self.sync_level,
            notify_on_succeeded=False,
            notify_on_failed=True,
        )

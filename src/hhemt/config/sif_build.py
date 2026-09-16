"""sif_build_config: VENUE and RESOURCES for `hhemt build-sifs`. Identity is never in this file."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class sif_build_config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sif_root: Path = Field(..., description="Root under which the builder writes and the resolver looks.")
    partition: str = Field(..., description="Build partition on the BUILD host (must be fakeroot-capable).")
    cpus: int = 16
    mem_mb: int = 65536
    walltime_min: int = 480  # a6000 measured 5h45m; TIMEOUT is retriable (retries: 1)
    recipes_dir: Path | None = None  # None -> the package's sif/recipes
    toolkit_root: Path = Field(..., description="The git checkout that IS the running toolkit.")

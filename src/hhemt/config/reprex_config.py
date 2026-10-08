"""reprex_config.yaml Pydantic model (reproducibility-system C8, ADR-10).

The reprex_config is the minimal field set a target user supplies to run a
reprex bundle on THEIR system: the USER-bucket fields (host-local account /
login node / scratch / home / SIF path) plus the HPC-revisable SELECTORS
(target partition). Field membership is a projection of the ADR-10 taxonomy
(reprex_taxonomy.all_field_bucket == "user") ∪ the HPC partition selectors --
"deterministic what is user-specific and what is machine-specific" (O-m).

Plain BaseModel (NOT cfgBaseModel): the local paths it names are the TARGET
user's, validated for shape at load, not existence-checked against the
producer's tree (mirrors the CaseManifest / Globus-model precedent, Gotcha 1).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class reprex_config(BaseModel):
    """Minimal target-user field set to run a reprex bundle on a foreign system."""

    model_config = ConfigDict(extra="forbid")

    # ---- USER-bucket (host-local; never inherited from the bundle) ----
    default_account: str = Field(..., description="Target user's HPC allocation/account.")
    login_node: str | None = Field(None, description="Target cluster login node hostname.")
    sif_root: Path = Field(
        ...,
        description=(
            "Target-local directory under which fetched or rebuilt SIFs live at their identity path (found by digest)."
        ),
    )
    scratch_dir: Path | None = Field(None, description="Target-local scratch base for run outputs.")

    # ---- HPC-revisable SELECTORS (target partition axis) ----
    target_ensemble_partition: str = Field(..., description="Target partition for the ensemble sims.")
    target_setup_and_analysis_processing_partition: str | None = Field(
        None, description="Target partition for setup + processing (defaults to the ensemble partition)."
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_retired_sif_path(cls, data):
        # HARD REFUSAL, matching the ContainerSpec and ContainerRef siblings. `sif_path` was
        # RENAMED rather than removed, and a rename is the error shape where `extra="forbid"`
        # alone misleads: the reader gets "sif_root Field required" beside "sif_path Extra
        # inputs are not permitted", with nothing stating that one replaced the other, and has
        # to infer the substitution from two errors that read as independent.
        if isinstance(data, dict) and "sif_path" in data:
            raise ValueError(
                "reprex_config.sif_path was RETIRED (hhemt SIF quest, 2026-09) and RENAMED to "
                "sif_root: an image is resolved by identity under a DIRECTORY, never named by a "
                "path to one file. Rename sif_path to sif_root and point it at the directory "
                "holding the fetched .sif and its .manifest.json."
            )
        return data

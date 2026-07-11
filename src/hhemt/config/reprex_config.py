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

from pydantic import BaseModel, ConfigDict, Field


class reprex_config(BaseModel):
    """Minimal target-user field set to run a reprex bundle on a foreign system."""

    model_config = ConfigDict(extra="forbid")

    # ---- USER-bucket (host-local; never inherited from the bundle) ----
    default_account: str = Field(..., description="Target user's HPC allocation/account.")
    login_node: str | None = Field(None, description="Target cluster login node hostname.")
    sif_path: Path = Field(..., description="Target-local absolute path to the fetched, signed SIF.")
    scratch_dir: Path | None = Field(None, description="Target-local scratch base for run outputs.")

    # ---- HPC-revisable SELECTORS (target partition axis) ----
    target_ensemble_partition: str = Field(..., description="Target partition for the ensemble sims.")
    target_setup_and_analysis_processing_partition: str | None = Field(
        None, description="Target partition for setup + processing (defaults to the ensemble partition)."
    )

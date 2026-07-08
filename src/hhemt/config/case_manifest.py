"""Pydantic model for the case-study manifest (ADR-12).

A case study's ``case.yaml`` is a *provenance descriptor of REMOTE artifacts* —
the heavy input data lives in a Hydroshare resource, not on the local machine at
config-load time. For that reason this model subclasses plain ``pydantic.BaseModel``
(NOT ``cfgBaseModel``): ``cfgBaseModel._check_paths_exist`` raises ``ValueError`` for
any ``Path``-typed field whose target does not exist on disk, which is exactly wrong
for a manifest of not-yet-downloaded data. This mirrors the Globus config models'
deliberate plain-``BaseModel`` choice (architecture Gotcha 1).

``extra="forbid"`` is set locally so a typo'd key in ``case.yaml`` fails loudly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseManifest(BaseModel):
    """Validated schema for ``test_data/{case_name}/case.yaml`` (ADR-12)."""

    model_config = ConfigDict(extra="forbid")

    case_name: str = Field(description="Human-readable case-study name.")
    res_identifier: str = Field(description="Hydroshare resource id (32-hex) for the heavy input data.")
    doi: str | None = Field(
        default=None,
        description=(
            "DOI of the durable data deposit (e.g. '10.5281/zenodo.1234567'). "
            "Preferred over host-native ids for citation and long-term resolution."
        ),
    )
    pid: str | None = Field(
        default=None,
        description=(
            "Host-native record identifier when a DOI is absent (e.g. a Zenodo record id). "
            "host determines interpretation."
        ),
    )
    resource_version: str | None = Field(
        default=None,
        description="Hydroshare resource version label this manifest was pinned to.",
    )
    description: str | None = Field(default=None, description="One-line description of the case study.")
    citation: str | None = Field(default=None, description="Citation string for the hosted data resource.")
    manifest: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of bag-relative filename -> hex sha256. Empty dict means "
            "the manifest has not yet been populated (schema-valid placeholder)."
        ),
    )
    host: Literal["hydroshare", "zenodo"] = Field(
        default="hydroshare",
        description=(
            "Data-host backend. Widen this Literal (and flip to a scheme-keyed PID resolver) "
            "when a THIRD host is added."
        ),
    )

    @model_validator(mode="after")
    def _check_host_identifier(self) -> CaseManifest:
        if self.host == "zenodo" and not (self.doi or self.pid):
            raise ValueError("host='zenodo' requires a 'doi' or 'pid'.")
        return self

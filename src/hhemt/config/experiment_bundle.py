"""Pydantic model for the per-experiment bundle descriptor (`experiment.yaml`).

An experiment bundle is a self-describing directory in the private deployment
estate. Its descriptor declares: the experiment's identity, the input datasets it
needs (including datasets too large to git-track and not yet DOI-deposited), the
per-cluster destinations those inputs must reach, the container that executes it,
and a RESOLVABLE toolkit pin.

Base-class choice mirrors ``config/case_manifest.py`` (architecture Gotcha 1): this
is a plain ``BaseModel``, NOT ``cfgBaseModel``, because ``cfgBaseModel``'s
``_check_paths_exist`` field validator rejects any ``Path`` whose target is absent —
exactly wrong for a ``${VAR}``-templated producer path and for HPC destinations that
do not exist on the authoring machine. Paths are typed ``str`` here for the same
reason; expansion and existence are the runner's concern, not the schema's.

``extra="forbid"`` so a typo'd key in ``experiment.yaml`` fails loudly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetRef(BaseModel):
    """One input dataset an experiment needs.

    The two-field split is load-bearing and is the D6 amendment:

    - ``local_path`` is where the PRODUCER reads the bytes. It is optional and
      ``${VAR}``-templated. It MUST NOT be a literal operator path — a
      ``/home/{user}/...`` value is meaningless to a third party AND trips the
      zero-user-info blocklist scan.
    - ``deposit`` declares the bytes are in-scope for the PUBLISH payload. Without
      it, a descriptor that scrubs ``local_path`` produces a bundle that passes the
      zero-user-info gate and cannot be reproduced from.

    The ``doi``/``pid``/``host``/``sha256`` quartet mirrors ``CaseManifest``'s
    vocabulary so the pre-DOI -> post-DOI transition is a field-fill, not a schema
    break. Nothing reads them until the DOI-ingestion plan's Phase 3 lands.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Stable key for this input within the bundle.")
    local_path: str | None = Field(
        default=None,
        description=(
            "Producer-side read location, ${VAR}-templated (e.g. "
            "'${HHEMT_DATA_ROOT}/weather/design_storm_timeseries_SSR.nc'). Optional: "
            "absent once the dataset resolves by DOI. Never a literal operator path."
        ),
    )
    deposit: bool = Field(
        default=False,
        description=(
            "True => these bytes are part of the publish payload (publish_analysis "
            "deposits them). False => referenced only. Required True for any input a "
            "third party cannot otherwise obtain."
        ),
    )
    sha256: str | None = Field(default=None, description="Hex sha256 of the file, when pinned.")
    doi: str | None = Field(default=None, description="DOI of the durable deposit, once minted.")
    pid: str | None = Field(default=None, description="Host-native record id when no DOI exists.")
    host: Literal["hydroshare", "zenodo"] | None = Field(
        default=None, description="Data-host backend; interpretation key for pid."
    )
    destinations: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Cluster-name -> absolute on-cluster destination path. Consumed by the "
            "provisioning CLI (routed follow-up); declared here so the descriptor is "
            "the single source of experiment identity."
        ),
    )

    @model_validator(mode="after")
    def _check_resolvable(self) -> DatasetRef:
        if not self.local_path and not (self.doi or self.pid):
            raise ValueError(
                f"DatasetRef {self.name!r}: needs a local_path or a doi/pid — otherwise nothing can resolve the bytes."
            )
        if self.local_path and self.local_path.startswith("/home/"):
            raise ValueError(
                f"DatasetRef {self.name!r}: local_path must be ${{VAR}}-templated, not a "
                f"literal operator path ({self.local_path!r}). A literal path is "
                "unreproducible for a third party and trips the zero-user-info gate."
            )
        return self


class ToolkitPin(BaseModel):
    """A RESOLVABLE pin to the toolkit that produced/reproduces this experiment.

    ``version`` is primary because PyPI forbids version re-upload, so the identifier
    resolves to a durable ARTIFACT rather than to a repository. ``commit``/``tag`` are
    advisory only: the estate's own ``toolkit_pin.yaml`` pinned ``b570228``, which no
    longer resolves (``git cat-file -t`` -> rc=128) after a history rewrite. A git SHA
    is never the sole anchor.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="PyPI version, e.g. '0.1.0'.")
    sdist_sha256: str | None = Field(default=None, description="sha256 of the PyPI sdist.")
    commit: str | None = Field(default=None, description="ADVISORY ONLY — may be orphaned.")
    tag: str | None = Field(default=None, description="ADVISORY ONLY.")


class ContainerRef(BaseModel):
    """Reference to the SIF that executes this experiment.

    Deliberately a REFERENCE, not a copy. The SIF sha256's durable home is the
    RO-Crate's ``SoftwareApplication`` node, which the reprex two-part verify already
    reads. ``ContainerSpec`` itself carries no sha256 field. Minting a third copy here
    would create a drift surface with no reader.
    """

    model_config = ConfigDict(extra="forbid")

    def_recipe: str = Field(description="Repo-relative .def path, e.g. 'containers/uva-cuda.def'.")
    sha256_source: Literal["ro-crate"] = Field(
        default="ro-crate",
        description="Where the authoritative SIF digest lives. Only 'ro-crate' is valid.",
    )


class ExperimentBundle(BaseModel):
    """Validated schema for an estate `experiments/{slug}/experiment.yaml`."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(description="Stable slug; must equal the containing directory name.")
    description: str = Field(description="One-line description.")
    system_config: str = Field(description="Bundle-relative path to the system config YAML.")
    analysis_config: str = Field(description="Bundle-relative path to the analysis config YAML.")
    hpc_system_config: dict[str, str] = Field(
        default_factory=dict,
        description="Cluster-name -> estate-relative hpc_system_config path.",
    )
    inputs: list[DatasetRef] = Field(default_factory=list, description="Input datasets.")
    toolkit_pin: ToolkitPin
    container: ContainerRef | None = Field(default=None, description="None => native execution.")

    @model_validator(mode="after")
    def _check_deposit_coverage(self) -> ExperimentBundle:
        undepositable = [i.name for i in self.inputs if not i.deposit and not (i.doi or i.pid)]
        if undepositable:
            raise ValueError(
                f"inputs {undepositable!r} are neither deposited nor DOI-resolvable — "
                "a third party could not obtain them, so the bundle is not reproducible."
            )
        return self

"""Operator-authored bundle exclude-config (ADR-20, as amended 2026-07-14).

A reprex bundle is SELF-CONTAINED by default (ADR-9): every cfg-declared input is copied
into the bundle. Excluding an input is a governed, lower-priority OPT-OUT for data that is
oversized, private, or third-party/IP-restricted. This module defines the operator-authored
YAML that drives that opt-out.

Why a STANDALONE file passed by path, and not an ``analysis_config`` field
-------------------------------------------------------------------------
``bundle/_emit.py::_copy_configs_with_relative_paths`` serializes
``cfg_analysis.model_dump(mode="json")`` INTO the bundle, and ``_scrub_user_bucket_fields``
nulls only ``"user"``-bucket fields. An exclusion block living on ``analysis_config`` would
bucket ``"experiment"``, ride into the bundle, and be reconstituted on the consumer's
machine — so a consumer's re-emit would silently re-exclude an input that now exists
locally while re-emitting an ``input_deposit`` block pointing at the ORIGINAL operator's
deposit ("re-emit poisoning"). Exclusion is an emit-time OPERATOR policy, not a property of
the experiment.

Why a plain ``BaseModel`` and not ``cfgBaseModel``
-------------------------------------------------
``cfgBaseModel._check_paths_exist`` validates that Path fields exist on disk. This config
describes REMOTE artifacts that by definition are NOT on disk — the existence gate is
exactly wrong here. ``extra="forbid"`` is retained (the ``CaseManifest``/ADR-12 precedent)
so a typo'd key fails loudly rather than being silently ignored.

The `input_deposit` split (what the operator supplies vs. what the toolkit computes)
-----------------------------------------------------------------------------------
The toolkit COMPUTES ``relpath`` (the bundle-relative location the reconstituted config
resolves the input at), ``sha256``, and ``accessed``. The operator supplies the facts only
they hold:

- ``citation``   REQUIRED — what the input is, who holds it, and how a third party obtains it.
- ``contentUrl`` optional — a direct-download URL. **Its presence or absence IS the
  machine-readable fetchable bit.** Present => ``hhemt ingest`` fetches and sha256-verifies
  it automatically. Absent => ingest FAILS CLOSED with the citation, telling the consumer
  exactly how to obtain the file and where to put it. Omitting it is the CORRECT, supported
  choice for licensed/IP-restricted data — not a degraded one.
- ``url``        optional — a landing/description page (RO-Crate's own gloss for this field
  is "e.g. direct download is not available").
- ``identifier`` optional — the deposit's DOI/PID. A CITATION coordinate only, never a fetch
  coordinate.

Operator ordering (a precondition, not a toolkit affordance): the excluded input must
ALREADY have a durable record before this config is authored. The toolkit has no per-file
deposit helper and cannot mint one. Operator-owned data is deposited as its own resource
first; third-party data is referenced at its ORIGINAL source, and the toolkit MUST NOT
redeposit it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hhemt.bundle._path_policy import (
    _EXCLUDABLE_CATALOG,
    _PATH_FIELD_POLICY,
    PathPolicy,
)


class ExcludedInputRef(BaseModel):
    """The operator-supplied coordinates for ONE excluded input FILE.

    This is a FILE-level record, not a record-level one — it is deliberately NOT
    ``CaseManifest``-shaped. ``CaseManifest`` describes a whole remote deposit; this
    describes a single file within (or outside) one.
    """

    model_config = ConfigDict(extra="forbid")

    citation: str = Field(
        description=(
            "REQUIRED. What this input is, who holds it, and how a third party obtains it. "
            "This is the payload of the fail-closed message when the input is unfetchable, "
            "and it is what makes a restricted input HONEST rather than merely broken."
        )
    )
    contentUrl: str | None = Field(  # noqa: N815 — RO-Crate/schema.org property name
        default=None,
        description=(
            "Direct-download URL. Following it (allowing redirects) MUST download the file "
            "itself — a URL that lands on an HTML splash page is a contract violation, not a "
            "contentUrl. PRESENT => the consumer's ingest fetches + sha256-verifies it. "
            "ABSENT => ingest fails closed with the citation (the correct outcome for "
            "licensed/IP data the toolkit must not redistribute)."
        ),
    )
    url: str | None = Field(
        default=None,
        description="Landing/description page for this file (used when direct download is unavailable).",
    )
    identifier: str | None = Field(
        default=None,
        description="The deposit's DOI/PID. Citation coordinate ONLY — never used to fetch.",
    )


class BundleExcludeConfig(BaseModel):
    """The operator-authored exclude-config (``hhemt bundle --exclude-config {path}``).

    ``exclusions`` maps a cfg FIELD NAME (from ``hhemt bundle --list-excludable``) to the
    coordinates for the input(s) that field names:

    - a SCALAR field maps to one ``ExcludedInputRef``;
    - a LIST field (``BUNDLE_RELATIVE_LIST``, e.g. ``static_plot_configs``) maps to a
      ``{basename: ExcludedInputRef}`` dict — one entry per element, matched by file name,
      because one excluded list-field emits N ``input_deposit`` blocks.
    """

    model_config = ConfigDict(extra="forbid")

    exclusions: dict[str, ExcludedInputRef | dict[str, ExcludedInputRef]] = Field(
        default_factory=dict,
        description="cfg field name -> coordinates (scalar) or {basename: coordinates} (list field).",
    )

    @model_validator(mode="after")
    def _validate_against_catalog(self) -> BundleExcludeConfig:
        """Reject unknown, non-excludable, or shape-mismatched entries — loudly.

        A silent no-op here is the worst failure available: the operator believes an input
        was excluded by reference, the emit carries it anyway (or drops it with no block),
        and the defect surfaces only after a DOI has been minted.
        """
        for field_name, ref in self.exclusions.items():
            entry = _EXCLUDABLE_CATALOG.get(field_name)
            if entry is None:
                raise ValueError(
                    f"'{field_name}' is not an excludable input. It is either not a "
                    f"cfg-declared bundle-carried input, or the name is misspelled. "
                    f"Run `hhemt bundle --list-excludable` for the authoritative menu."
                )
            if not entry.excludable:
                raise ValueError(
                    f"'{field_name}' is catalogued as NOT excludable: {entry.reproducibility_cost}"
                )

            is_list_field = _PATH_FIELD_POLICY.get(field_name) is PathPolicy.BUNDLE_RELATIVE_LIST
            if is_list_field and not isinstance(ref, dict):
                raise ValueError(
                    f"'{field_name}' is a LIST field, so its exclusion must map each element "
                    f"by file name: {{basename: {{citation: ..., contentUrl: ...}}}}. One "
                    f"input_deposit block is emitted per element."
                )
            if not is_list_field and isinstance(ref, dict):
                raise ValueError(
                    f"'{field_name}' is a scalar field, so its exclusion takes the coordinates "
                    f"directly ({{citation: ..., contentUrl: ...}}), not a {{basename: ...}} map."
                )
        return self

    def refs_for(self, field_name: str, basename: str) -> ExcludedInputRef | None:
        """Return the coordinates for one excluded FILE, or None if not excluded.

        ``basename`` selects the element for a LIST field and is ignored for a scalar.
        """
        ref = self.exclusions.get(field_name)
        if ref is None:
            return None
        if isinstance(ref, dict):
            return ref.get(basename)
        return ref

    def excludes(self, field_name: str) -> bool:
        """True if this config opts the named cfg field out of self-contained carriage."""
        return field_name in self.exclusions

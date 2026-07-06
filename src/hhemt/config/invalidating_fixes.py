"""ADR-17 invalidating-fix registry loader (Pydantic).

Loads the package-data registry (``src/hhemt/invalidating_fixes.yaml``) and validates
each entry, failing loud on a malformed hand-edit. A pure INPUT-schema module (it
describes bug-fix commits, not on-disk analysis layout), so it is NON_BREAKING-
allowlisted in ``_layout_relevant_files.yaml`` and drives NO LAYOUT_VERSION bump.

The canonical ``RecomputeAction`` enum is IMPORTED from ``recompute.py`` (Phase 2
owns it, ADR-16). It is NEVER re-declared here — a duplicated enum that drifts is
the integration risk the master's Risks note guards against.
"""

from __future__ import annotations

import warnings
from importlib import resources
from pathlib import Path

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, field_validator, model_validator

from hhemt.recompute import RecomputeAction  # FROZEN ADR-16 enum — import, never re-declare

_REGISTRY_FILENAME = "invalidating_fixes.yaml"


class InvalidatingFix(BaseModel):
    """One registered calculation-invalidating fix.

    Plain ``BaseModel`` (describes a commit, not a local path). The Phase-2 resolver
    (``recompute.py::_resolve_registry_matches``) consumes ``commit_id``, ``severity``,
    ``recommended_action`` and ``affected_version_range``; the pure RO-Crate projection
    (``metadata.py::project_invalidating_fix``) additionally emits ``introduced_in_version``,
    ``description`` and ``significance``.
    """

    commit_id: str
    severity: str  # {"warning", "error"} — ADR-17 authoritative enum (D1)
    affected_scope: str  # descriptive tier ({scenario, consolidation}) — not classifier-evaluated
    recommended_action: RecomputeAction
    # T1 Option C: REQUIRED PEP 440 SpecifierSet and the SOLE evaluated version predicate.
    affected_version_range: str
    # DESCRIPTIVE metadata only — the version that SHIPPED the fix (a point); NEVER
    # evaluated by the classifier; retained for the projection's version-context field.
    introduced_in_version: str
    description: str
    significance: str

    @field_validator("affected_version_range")
    @classmethod
    def _validate_specifier_set(cls, value: str) -> str:
        """Fail loud when ``affected_version_range`` is not a parseable SpecifierSet."""
        try:
            SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise ValueError(
                f"affected_version_range {value!r} is not a valid PEP 440 SpecifierSet (e.g. '>=0.8.0,<0.9.3'): {exc}"
            ) from exc
        return value

    @model_validator(mode="after")
    def _warn_version_context_inconsistency(self) -> InvalidatingFix:
        """SOFT drift signal (never raises).

        A fix normally ships at the exclusive upper bound of its affected range
        (``affected < introduced_in_version``). When the range's ``<`` upper bound
        disagrees with ``introduced_in_version``, warn — it is a descriptive-drift
        signal, not a hard error (``introduced_in_version`` is not classifier-evaluated,
        so a mismatch must not block the load).
        """
        try:
            spec = SpecifierSet(self.affected_version_range)
            introduced = Version(self.introduced_in_version)
        except (InvalidSpecifier, InvalidVersion):
            return self  # hard failures are the field validators' concern; skip the soft check
        for clause in spec:
            if clause.operator == "<" and Version(clause.version) != introduced:
                warnings.warn(
                    f"invalidating-fix {self.commit_id[:7]}: affected_version_range upper "
                    f"bound {clause.version!r} != introduced_in_version "
                    f"{self.introduced_in_version!r} (descriptive drift; a fix normally "
                    "ships at the exclusive upper bound of its affected range)",
                    stacklevel=2,
                )
        return self


class InvalidatingFixRegistry(BaseModel):
    """The whole registry: a schema version plus the list of fixes."""

    schema_version: int
    fixes: list[InvalidatingFix]

    @model_validator(mode="after")
    def _reject_duplicate_commit_ids(self) -> InvalidatingFixRegistry:
        """Ambiguity-refusal (C10): two entries for one commit have no defined
        severity/scope/action winner. Mirrors ``version_migration/registry.py``'s
        duplicate-``(version_from, version_to)`` rejection (per-entry Pydantic type
        validation does not catch a cross-entry duplicate)."""
        seen: set[str] = set()
        for fix in self.fixes:
            if fix.commit_id in seen:
                raise ValueError(
                    f"duplicate commit_id {fix.commit_id!r} in the invalidating-fix "
                    "registry (each fix commit must appear at most once)"
                )
            seen.add(fix.commit_id)
        return self


def _default_registry_text() -> str:
    """Read the package-data registry YAML as text (install-location-independent)."""
    return (resources.files("hhemt") / _REGISTRY_FILENAME).read_text()


def load_invalidating_fixes(registry_path: Path | None = None) -> InvalidatingFixRegistry:
    """Load + validate the invalidating-fix registry, failing loud on a bad entry.

    With no argument, reads the shipped package-data registry (the path the Phase-2
    resolver seam and the Phase-4 skew check both use). ``registry_path`` overrides
    the source for testing against a fixture YAML. A malformed / empty-top-level YAML
    raises here; the Phase-2 caller (``recompute.py::_load_invalidating_fix_registry``)
    degrades that to ``[]`` so a bad registry never crashes analysis construction
    (C-NON-BLOCKING-BUGEMIT).
    """
    text = Path(registry_path).read_text() if registry_path is not None else _default_registry_text()
    raw = yaml.safe_load(text)
    if raw is None:
        raise ValueError(
            "invalidating_fixes.yaml parsed to None (file empty or top-level null); "
            "expected a mapping with 'schema_version' and 'fixes'."
        )
    return InvalidatingFixRegistry.model_validate(raw)

"""Cross-experiment bundle metadata-compatibility checker (PIP-1, Phase 1).

Reads each input bundle's identity surface, compares the bundles field-by-field,
and classifies every divergence into a ``CompatibilitySeverity`` keyed on the
field's ADR-10 USER/HPC/EXPERIMENT taxonomy bucket.

DISTINCT from ADR-17's bug-registry ``severity`` (output-invalidation):
``CompatibilitySeverity`` is COMBINE-ADMISSIBILITY (can these N experiments
share one report?). See the decision doc
``compatibilityseverity is orthogonal to adr17 severity``.

Read surface (as-built, verified 2026-07-04). The pre-foundation skeleton assumed
the ADR-6 embedded JSON-LD core carried the identity fields; the LANDED foundation
does not. The RO-Crate embedded core + co-located ``ro-crate-metadata.json`` carry
only name/analysis_id/system_id/schemaVersion/softwareVersion/variableMeasured/
per-part sha256/hasPart — NOT the model toggles, sensitivity axes, or ``case_name``
(``analysis._case_manifest`` is never set at emit, so the core's ``name`` degrades
to ``analysis_id``; ``case.yaml`` is NOT copied into the bundle). HPC identity
(gpu_hardware/partition/account) migrated off ``system_config`` to
``hpc_system_config``, which is NOT serialized into the bundle — so no HPC field
reaches the checker (the ``_classify`` "hpc" branch is unit-tested directly, never
reached via a real read). Therefore the comparison surface is sourced from the
bundle's ``cfg_system.yaml`` + ``cfg_analysis.yaml`` (written by
``_emit._copy_configs_with_relative_paths``) plus ``schemaVersion`` from the
RO-Crate sidecar for version-skew. Always-divergent identifiers (``analysis_id``,
toolkit git-sha) are EXCLUDED from the divergence surface; ``analysis_id`` is kept
only as the bundle LABEL. ``case_name`` is not bundled today, so it is not sourced
(bundling ``case.yaml`` so ``case_name`` becomes comparable is a routed follow-up).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

# blocked-on-FOUNDATION (A2) — RESOLVED: the landed taxonomy is
# ``hhemt.config.reprex_taxonomy.field_bucket``. It is bound via a FUNCTION-LOCAL
# import in ``_field_bucket`` below (a top-level import would form a cycle once
# Phase 3 wires this module into ``bundle.__init__`` — see the reprex_taxonomy
# docstring).

# Bundle-relative filenames of the checker's read surface.
_CFG_SYSTEM_FILENAME = "cfg_system.yaml"
_CFG_ANALYSIS_FILENAME = "cfg_analysis.yaml"
_ROCRATE_FILENAME = "ro-crate-metadata.json"

# Curated comparison surface (the bundled identity + sensitivity fields). Kept
# small and explicit so the divergence report stays signal-rich rather than
# flooding on every incidental config difference. Extend here (not in the
# classifier) when a new bundled identity field warrants comparison.
_CFG_SYSTEM_COMPARISON_FIELDS: tuple[str, ...] = (
    "toggle_triton_model",
    "toggle_tritonswmm_model",
    "toggle_swmm_model",
    "target_dem_resolution",
)
_CFG_ANALYSIS_COMPARISON_FIELDS: tuple[str, ...] = (
    "weather_events_to_simulate",
    "sensitivity_analysis",
)

# Keys returned by ``_read_jsonld_core`` that label a bundle but must NEVER be
# compared for divergence (they are always-divergent between two distinct bundles).
_RESERVED_LABEL_FIELDS: frozenset[str] = frozenset({"analysis_id"})


class CompatibilitySeverity(str, Enum):  # noqa: UP042 -- (str, Enum) is deliberate (NOT StrEnum); see recompute.py
    """Combine-admissibility severity (NOT ADR-17 output-invalidation severity)."""

    INFORMATIONAL = "informational"  # expected divergence (HPC bucket): UVA vs Frontier
    WARNING = "warning"  # surfaced, non-blocking (sensitivity axis; version-skew)
    BLOCKING = "blocking"  # aborts the combine (EXPERIMENT-IDENTITY, or USER bucket)


@dataclass(frozen=True)
class CompatibilityDivergence:
    field_name: str
    bucket: str  # "user" | "hpc" | "experiment"
    severity: CompatibilitySeverity
    bundle_a: str  # analysis_id of the left bundle
    bundle_b: str
    value_a: object
    value_b: object


@dataclass
class CompatibilityReport:
    divergences: list[CompatibilityDivergence] = field(default_factory=list)

    @property
    def blocking(self) -> list[CompatibilityDivergence]:
        return [d for d in self.divergences if d.severity is CompatibilitySeverity.BLOCKING]

    @property
    def is_compatible(self) -> bool:
        return not self.blocking


# Experiment-IDENTITY fields whose divergence is BLOCKING (different experiment
# entirely). Sensitivity-axis fields (columns/rows) are WARNING. This realizes
# the research-reproducibility threshold (A2 severity decision).
#
# ``case_name`` is retained defensively: it is not bundled today (so
# ``_read_jsonld_core`` never sources it), but if a future phase copies
# ``case.yaml`` into the bundle the field will correctly classify as BLOCKING
# without a taxonomy change here.
_EXPERIMENT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "case_name",
        "weather_events_to_simulate",
        "toggle_tritonswmm_model",
        "toggle_triton_model",
        "toggle_swmm_model",
    }
)


def _field_bucket(field_name: str) -> str:
    """Return the ADR-10 taxonomy bucket for a field.

    Delegates config Path fields (``weather_events_to_simulate``,
    ``sensitivity_analysis``) to the landed ``reprex_taxonomy.field_bucket`` via a
    FUNCTION-LOCAL import (a top-level import forms an import cycle once Phase 3
    wires this module into ``bundle.__init__`` — see the reprex_taxonomy
    docstring). ``field_bucket`` is Path-field-scoped and raises ``KeyError`` on
    every non-path field (toggles, ``target_dem_resolution``, ``schemaVersion``,
    identity fields); those default to the "experiment" surface, with identity
    handled in ``_classify`` via ``_EXPERIMENT_IDENTITY_FIELDS``. There is no HPC
    classification path over a real read (no HPC field reaches the checker).
    """
    from hhemt.config.reprex_taxonomy import field_bucket  # function-local: avoids the bundle.__init__ import cycle

    try:
        return field_bucket(field_name)  # config Path fields only
    except KeyError:
        return "experiment"  # non-path fields default to the experiment surface; identity handled in _classify


def _classify(field_name: str, bucket: str) -> CompatibilitySeverity:
    if bucket == "user":
        return CompatibilitySeverity.BLOCKING  # impossible under zero-user-info
    if bucket == "hpc":
        return CompatibilitySeverity.INFORMATIONAL  # expected (UVA vs Frontier)
    # experiment bucket:
    if field_name in _EXPERIMENT_IDENTITY_FIELDS:
        return CompatibilitySeverity.BLOCKING  # different experiment
    return CompatibilitySeverity.WARNING  # sensitivity column/row divergence


def _load_yaml(path: Path) -> dict:
    """Load a bundled cfg YAML into a dict; return {} if the file is absent."""
    if not path.exists():
        return {}
    import yaml  # function-local, mirroring _emit.py's deferred yaml import

    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _rocrate_schema_version(rocrate_path: Path) -> object | None:
    """Return the root Dataset's ``schemaVersion`` (=layout_version) from the
    co-located RO-Crate sidecar, or None if the sidecar is absent/unparseable.

    The root Dataset is the ``@id == "./"`` entity per ``metadata.build_analysis_crate``.
    """
    try:
        doc = json.loads(rocrate_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for entity in doc.get("@graph", []):
        if entity.get("@id") == "./":
            return entity.get("schemaVersion")
    return None


def _bundle_label(bundle_root: Path) -> str:
    """Return a stable label for a bundle (its ``analysis_id`` from
    ``bundle_manifest.json``), falling back to the bundle directory name.

    Used ONLY to label divergences — never compared for divergence.
    """
    manifest = bundle_root / BUNDLE_MANIFEST_FILENAME
    if manifest.exists():
        try:
            aid = json.loads(manifest.read_text()).get("analysis_id")
        except (json.JSONDecodeError, OSError):
            aid = None
        if aid:
            return str(aid)
    return bundle_root.name


def _read_jsonld_core(bundle_root: Path) -> dict:
    """Build a flat {field_name: value} comparison dict for one bundle.

    Sources the curated identity + sensitivity surface from the bundle's
    ``cfg_system.yaml`` (model toggles + ``target_dem_resolution``) and
    ``cfg_analysis.yaml`` (``weather_events_to_simulate`` + ``sensitivity_analysis``),
    plus ``schemaVersion`` from the RO-Crate sidecar when present (version-skew).
    Always-divergent identifiers (``analysis_id``, toolkit git-sha) are EXCLUDED
    from the divergence surface; ``analysis_id`` is returned only under the
    reserved ``"analysis_id"`` key that ``check_bundle_compatibility`` uses for
    labeling, never for divergence. ``case_name`` is not bundled today and is not
    sourced (see module docstring).

    A missing cfg file yields no fields from it (not an error) — the checker
    compares whatever identity surface each bundle actually carries.
    """
    sysd = _load_yaml(bundle_root / _CFG_SYSTEM_FILENAME)
    anad = _load_yaml(bundle_root / _CFG_ANALYSIS_FILENAME)

    core: dict = {}
    for fname in _CFG_SYSTEM_COMPARISON_FIELDS:
        if fname in sysd:
            core[fname] = sysd[fname]
    for fname in _CFG_ANALYSIS_COMPARISON_FIELDS:
        if fname in anad:
            core[fname] = anad[fname]

    rocrate = bundle_root / _ROCRATE_FILENAME
    if rocrate.exists():
        schema_version = _rocrate_schema_version(rocrate)
        if schema_version is not None:
            core["schemaVersion"] = schema_version

    # Reserved label — never compared (see _RESERVED_LABEL_FIELDS).
    core["analysis_id"] = _bundle_label(bundle_root)
    return core


def check_bundle_compatibility(bundle_roots: list[Path]) -> CompatibilityReport:
    """Compare N bundles' metadata pairwise; return a CompatibilityReport.

    Iterates each pair, unions their comparison field sets (excluding reserved
    label keys), and emits a CompatibilityDivergence (classified via _classify)
    for each field whose values differ. R2/R3: callers abort on report.blocking.
    """
    report = CompatibilityReport()
    roots = sorted(bundle_roots)
    cores = {r: _read_jsonld_core(r) for r in roots}
    for i, a in enumerate(roots):
        for b in roots[i + 1 :]:
            ca, cb = cores[a], cores[b]
            comparison_fields = (set(ca) | set(cb)) - _RESERVED_LABEL_FIELDS
            for fname in sorted(comparison_fields):
                va, vb = ca.get(fname), cb.get(fname)
                if va != vb:
                    bucket = _field_bucket(fname)
                    report.divergences.append(
                        CompatibilityDivergence(
                            field_name=fname,
                            bucket=bucket,
                            severity=_classify(fname, bucket),
                            bundle_a=ca.get("analysis_id", str(a)),
                            bundle_b=cb.get("analysis_id", str(b)),
                            value_a=va,
                            value_b=vb,
                        )
                    )
    return report

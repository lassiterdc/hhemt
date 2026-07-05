"""combine_bundle() + CombinedBundle (PIP-1, Phase 3).

combine_bundle(bundle_paths, output_path=None) -> CombinedBundle:
  1. load + compatibility-check the N bundles (Phase 1); abort on blocking.
  2. merge their consolidated trees (Phase 2).
  3. render the combined report through the `combined` ReportingSet (Phase 4).
  4. emit a NEW STANDALONE COMBINED BUNDLE (a flat hasPart-by-reference
     Provenance-Run-Crate over the N intact child crates) and return a
     CombinedBundle handle.

The emission is a standalone bundle (NOT a report-only object); CombinedBundle
mirrors Bundle's consume surface so the combined report is regenerable and the
bundle is iterable.

Phase-3 scope (as-built): steps 1-2 and the step-4 EMIT (child-crate copy,
flat-hasPart combined ``ro-crate-metadata.json``, ``combined_compatibility.json``
read-model, ``bundle_manifest.json`` via the dedicated
``_write_combined_bundle_manifest``) are complete and byte-deterministic (CR4).
The step-3 cross-experiment RENDER (``_render_combined_report``) and the two
``CombinedBundle`` regen seams are the Phase-4 wiring point: they raise
``NotImplementedError`` until the ``combined`` ReportingSet lands. Phase-3 tests
monkeypatch these seams to isolate the orchestration + emit from the render.

Determinism (CR4): ro-crate-py stamps a per-run ``datePublished`` wall-clock on
the root Dataset at ``crate.metadata.generate()`` time, and ``_emit._write_bundle_manifest``
stamps a per-run ``created_at_utc`` — two volatile surfaces that would make two
combines of the same N bundles diverge byte-wise. Both are neutralized here: the
combined crate strips ``metadata._VOLATILE_PROV_KEYS`` (which includes
``datePublished``) before serialization, and ``_write_combined_bundle_manifest``
OMITS ``created_at_utc`` (its deterministic provenance surface is ``combined_of``).
Inputs are sorted before harvest and the archive reuses ``_emit._emit_bundle_zip``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from hhemt.bundle._combine_merge import _experiment_id, merge_experiment_trees
from hhemt.bundle._compatibility import CompatibilityReport, check_bundle_compatibility
from hhemt.bundle._emit import _get_toolkit_git_sha
from hhemt.exceptions import ConfigurationError
from hhemt.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
)

# Combined-bundle layout constants.
_CHILD_CRATES_SUBDIR = "child_crates"
_COMBINED_ROCRATE_FILENAME = "ro-crate-metadata.json"
_COMBINED_COMPAT_FILENAME = "combined_compatibility.json"
# Provenance Run Crate profile — the combined tier is a base RO-Crate 1.2 +
# Provenance-Run-Crate (NOT Workflow-Run-Crate; A8 / master D3 / ADR-9 flip-condition).
_COMBINED_CONFORMS_TO = (
    {"@id": "https://w3id.org/ro/crate/1.2"},
    {"@id": "https://w3id.org/ro/wfrun/provenance/0.5"},
)


def combine_bundle(
    bundle_paths: list[Path],
    output_path: Path | None = None,
) -> CombinedBundle:
    roots = sorted(Path(p).resolve() for p in bundle_paths)
    if len(roots) < 2:
        raise ConfigurationError(
            field="bundle_paths",
            message=f"combine_bundle needs >=2 bundles, got {len(roots)}.",
            config_path=None,
        )
    report = check_bundle_compatibility(roots)
    if not report.is_compatible:
        blocking = "; ".join(
            f"{d.field_name} ({d.bucket}): {d.bundle_a}={d.value_a!r} vs {d.bundle_b}={d.value_b!r}"
            for d in report.blocking
        )
        raise ConfigurationError(
            field="bundle_paths",
            message=f"Bundles are not combine-compatible (blocking divergences): {blocking}",
            config_path=None,
        )
    merged = merge_experiment_trees(roots)  # consumed by the emit-time render step
    if output_path is None:
        sha = _get_toolkit_git_sha(strict=False)
        output_path = roots[0].parent / f"combined_{len(roots)}bundles_{sha}"
    output_path = Path(output_path)
    _emit_combined_bundle(roots, merged, report, output_path)
    return CombinedBundle.from_directory(output_path)


def _combined_experiment_ids(roots: list[Path]) -> list[str]:
    """Stable, unique per-experiment ids for the combined bundle.

    Uses the SAME identity source as the Phase-2 merge (``_combine_merge._experiment_id``
    = each bundle's ``analysis_id``) with the same collision-suffix rule, so the
    manifest's ``experiment_ids`` and the ``child_crates/{eid}`` directory names
    line up with the merged tree's ``experiment_{eid}`` nodes.
    """
    ids: list[str] = []
    used: set[str] = set()
    for i, r in enumerate(roots):
        eid = _experiment_id(r)
        if eid in used:
            eid = f"{eid}__{i}"  # collision-safe unique id (mirrors merge_experiment_trees)
        used.add(eid)
        ids.append(eid)
    return ids


def _emit_combined_bundle(
    roots: list[Path],
    merged,
    report: CompatibilityReport,
    output_path: Path,
) -> None:
    """Emit the standalone combined bundle (flat hasPart-by-reference RO-Crate).

    Copies each input bundle intact under output_path/child_crates/{experiment_id}/,
    writes the combined ro-crate-metadata.json (root Dataset hasPart -> each child
    crate by reference, NOT N mainEntity — cardinality-1, A8), persists the
    CompatibilityReport as a read-model (combined_compatibility.json) for the
    Phase-4 renderer, writes bundle_manifest.json via a dedicated
    _write_combined_bundle_manifest (NOT verbatim _write_bundle_manifest, which is
    single-analysis-shaped — per CR5; the combined manifest carries experiment_ids /
    child_crates / bundle_schema_version / a deterministic combined_of and OMITS the
    volatile created_at_utc per CR4), and renders the combined report.
    Reuses _emit_bundle_zip's determinism contract when a zip container is requested.

    RENDER DISPATCH (design note, F-I #4): the combined figures are NEW (do not
    pre-exist), so the cross-experiment renderer must EXECUTE here — unlike
    Bundle.regenerate_report, which runs ``snakemake --report`` over already-touched
    pre-rendered figures. The renderer reads ``analysis.analysis_paths.analysis_dir``;
    a CombinedBundle is not a TRITONSWMM_analysis and exposes only ``.root``. Drive
    the render with a minimal render-context object (a small dataclass or a narrow
    ``CombinedRenderContext`` Protocol exposing ONLY ``analysis_paths.analysis_dir ==
    output_path`` — NOT ``BundleableAnalysis``, which also requires ``_system`` /
    ``cfg_analysis`` the combined render has no single value for), passing an explicit
    ``report_cfg`` (``DEFAULT_REPORT_CONFIG`` for the first cut) as the renderer's
    required second argument, and invoking the ``combined`` set's renderer(s) directly
    at emit time, writing the figures + assembled combined HTML under
    ``output_path/plots/cross_experiment/`` and ``output_path/analysis_report.{html,zip}``.
    ``CombinedBundle.regenerate_report()`` re-invokes this same emit-time render path
    (there is no combined Snakefile in the first cut, so it does NOT use ``snakemake
    --report``). (Chosen over a combined-scoped Snakefile generator for the first cut;
    the generator sibling is a documented later option.)

    Phase-3 completes the copy + crate + read-model + manifest; the render is the
    ``_render_combined_report`` seam wired in Phase 4.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    experiment_ids = _combined_experiment_ids(roots)

    # 1. Copy each input bundle intact under child_crates/{experiment_id}/ so each
    #    child retains its own Workflow-Run-Crate (ADR-13 reconstitution; R7).
    crates_dir = output_path / _CHILD_CRATES_SUBDIR
    child_crates: list[str] = []
    for eid, root in zip(experiment_ids, roots, strict=True):
        dest = crates_dir / eid
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(root, dest)
        child_crates.append(f"{_CHILD_CRATES_SUBDIR}/{eid}")

    # 2. Combined crate + read-model + manifest (all byte-deterministic — CR4).
    _write_combined_rocrate(output_path, child_crates)
    _write_combined_compatibility(output_path, report)
    _write_combined_bundle_manifest(
        output_path,
        experiment_ids=experiment_ids,
        child_crates=child_crates,
        git_sha=_get_toolkit_git_sha(strict=False),
    )

    # 3. Cross-experiment render (Phase-4 seam).
    _render_combined_report(merged, report, output_path)


def _write_combined_rocrate(output_path: Path, child_crates: list[str]) -> None:
    """Write the combined ro-crate-metadata.json (flat hasPart-by-reference).

    A base RO-Crate 1.2 + Provenance-Run-Crate whose root Dataset hasPart-references
    each intact child crate directory (NOT N mainEntity — A8), plus one combine
    ``CreateAction`` recording the operation. Serialized via the toolkit's canonical
    JSON-LD post-process (sorted @graph + sorted keys), with ``metadata._VOLATILE_PROV_KEYS``
    (notably ro-crate-py's per-run ``datePublished``) stripped so two combines are
    byte-identical (CR4).
    """
    from rocrate.model.contextentity import ContextEntity
    from rocrate.rocrate import ROCrate

    from hhemt.metadata import _VOLATILE_PROV_KEYS, canonical_jsonld_from_doc

    crate = ROCrate()
    root = crate.root_dataset
    root["name"] = f"Combined cross-experiment bundle ({len(child_crates)} experiments)"
    root["conformsTo"] = [dict(c) for c in _COMBINED_CONFORMS_TO]

    child_refs: list[dict] = []
    for rel in child_crates:
        ds = crate.add_dataset(
            source=None,
            dest_path=rel,
            fetch_remote=False,
            validate_url=False,
            properties={"@type": "Dataset", "name": f"Child render bundle ({rel})"},
        )
        child_refs.append({"@id": ds.id})

    crate.add(
        ContextEntity(
            crate,
            "#hhemt-combine",
            properties={
                "@type": "SoftwareApplication",
                "name": "hhemt combine_bundle",
            },
        )
    )
    crate.add(
        ContextEntity(
            crate,
            "#combine",
            properties={
                "@type": "CreateAction",
                "name": "combine_bundle",
                "object": child_refs,  # the N intact child crates
                "result": {"@id": "analysis_report.zip"},  # produced by the Phase-4 render
                "instrument": {"@id": "#hhemt-combine"},
            },
        )
    )

    doc = crate.metadata.generate()
    for entity in doc["@graph"]:  # strip per-run wall-clocks (datePublished, ...) — CR4
        for volatile in _VOLATILE_PROV_KEYS:
            entity.pop(volatile, None)
    (output_path / _COMBINED_ROCRATE_FILENAME).write_text(canonical_jsonld_from_doc(doc))


def _write_combined_compatibility(output_path: Path, report: CompatibilityReport) -> None:
    """Persist the CompatibilityReport as a deterministic JSON read-model.

    Mirrors the single-analysis validation_report.json pattern (master (c)): the
    Phase-4 combined renderer REFERENCES this read-model rather than recomputing the
    compatibility check.
    """
    payload = {
        "is_compatible": report.is_compatible,
        "divergences": [
            {
                "field_name": d.field_name,
                "bucket": d.bucket,
                "severity": d.severity.value,
                "bundle_a": d.bundle_a,
                "bundle_b": d.bundle_b,
                "value_a": d.value_a,
                "value_b": d.value_b,
            }
            for d in report.divergences
        ],
    }
    (output_path / _COMBINED_COMPAT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def _write_combined_bundle_manifest(
    output_path: Path,
    *,
    experiment_ids: list[str],
    child_crates: list[str],
    git_sha: str,
) -> None:
    """Write the combined bundle_manifest.json (CR5 content model, CR4 determinism).

    Distinct from _emit._write_bundle_manifest, which stamps one ``analysis_id`` +
    a single-analysis ``source_paths_by_renderer`` (neither fits an N-experiment
    bundle) AND a volatile ``created_at_utc`` wall-clock. This writer carries the
    N-experiment identity surface (``experiment_ids`` / ``child_crates`` /
    ``combined_of``) at the same ``BUNDLE_SCHEMA_VERSION`` the local toolkit emits,
    and OMITS ``created_at_utc`` so two combines are byte-identical (CR4). The
    deterministic ``combined_of`` list IS the provenance surface; child crates
    retain their own timestamps.
    """
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "combined": True,
        "toolkit_git_sha": git_sha,
        "experiment_ids": list(experiment_ids),
        "child_crates": list(child_crates),
        "combined_of": list(experiment_ids),  # deterministic provenance surface (CR4)
    }
    (output_path / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _render_combined_report(merged, report: CompatibilityReport, output_path: Path) -> None:
    """Execute the cross-experiment ``combined`` ReportingSet render at emit time.

    Phase-4 seam. See ``_emit_combined_bundle``'s RENDER DISPATCH note: the combined
    figures are NEW, so the renderer must execute here (writing figures under
    ``output_path/plots/cross_experiment/`` and the assembled combined HTML/zip under
    ``output_path/analysis_report.{html,zip}``), driven by a minimal render-context
    exposing only ``analysis_paths.analysis_dir == output_path``. Wired once the
    ``combined`` ReportingSet lands in Phase 4.
    """
    raise NotImplementedError(
        "The cross-experiment `combined` ReportingSet render is wired in Phase 4 "
        "(combined ReportingSet + cross-family panel). Phase 3 emits the standalone "
        "combined crate, compatibility read-model, and manifest; the render executes here."
    )


class CombinedBundle:
    """Consume-side handle for a standalone combined bundle (mirrors Bundle)."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @classmethod
    def from_directory(cls, path: Path | str) -> CombinedBundle:
        root = Path(path).resolve()
        manifest_path = root / BUNDLE_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {BUNDLE_MANIFEST_FILENAME} under {root}.")

        from hhemt.bundle import BundleSchemaError

        manifest = json.loads(manifest_path.read_text())
        version = manifest.get("bundle_schema_version")
        if version != BUNDLE_SCHEMA_VERSION:
            raise BundleSchemaError(
                f"Combined bundle {root} has bundle_schema_version={version} "
                f"!= local BUNDLE_SCHEMA_VERSION={BUNDLE_SCHEMA_VERSION}."
            )
        return cls(root=root)

    @property
    def root(self) -> Path:
        return self._root

    def regenerate_report(self, *, format: Literal["html", "zip"] = "zip") -> Path:
        """Regenerate the combined report from the bundled data (mirrors Bundle.regenerate_report)."""
        raise NotImplementedError(  # delegates to the combined-report render path (Phase 4 wires the set)
            "CombinedBundle.regenerate_report is wired in Phase 4 alongside the `combined` ReportingSet render path."
        )

    def eda(self, *, plots_only: bool = True):
        """Regenerate the EDA surface from the bundled data (mirrors Bundle.eda)."""
        raise NotImplementedError("CombinedBundle.eda is wired in Phase 4 alongside the combined render path.")

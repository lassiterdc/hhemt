"""Render-bundle emission helpers — the portable artifact for local
renderer iteration.

Public surface (called from Analysis.bundle_report_data() and
TRITONSWMM_sensitivity_analysis.bundle_report_data()):

  emit_bundle(analysis, output_path) -> Path

Helpers (private to this module):

  _harvest_and_copy_sources(...)
  _rewrite_paths_to_relative(...)
  _write_bundle_manifest(...)
  _emit_bundle_zip(...)

This module is opt-in only. Importing it does not trigger any side
effects; the only entry point is emit_bundle(). The method is invoked
from Analysis.bundle_report_data() (and the sensitivity parallel),
which is in turn invoked only from the `hhemt bundle`
CLI command. It is NEVER invoked by Analysis.run() or
Analysis.submit_workflow().
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tempfile
import warnings
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hhemt.bundle._path_policy import (
    _PATH_FIELD_POLICY,
    _SELF_CONTAINED_POLICIES,
    PathPolicy,
    RewriteResult,
    enumerate_path_fields,
)
from hhemt.report_renderers._figure_emission import (
    harvest_source_paths,
)
from hhemt.version_migration.constants import (
    BUNDLE_BASELINE_SUBDIR,
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_OUTPUT_SUBDIR,
    BUNDLE_PLOTS_SUBDIR,
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_STATUS_SUBDIR,
    LAYOUT_VERSION,
    VERSION_FILE_NAME,
)

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.base import cfgBaseModel
    from hhemt.config.bundle_exclude import BundleExcludeConfig


def emit_bundle(
    analysis: TRITONSWMM_analysis,
    output_path: Path | None = None,
    exclude_config: Path | BundleExcludeConfig | None = None,
) -> Path:
    """Emit a portable render bundle from a completed HPC analysis.

    The bundle file set is the union of source paths declared via
    prov.artist().add_channel(...) calls during the most recent
    render_report() execution, harvested from *.manifest.json sidecars
    under {analysis_dir}/plots/. Configs are rewritten to relative paths.
    The HPC-baseline analysis_report.{html,zip} are preserved under
    bundle_baseline/.

    Args:
        exclude_config: The ADR-20 governed opt-out — a path to an operator-authored
            exclude-config YAML (or an already-validated ``BundleExcludeConfig``). When
            given, the named inputs are NOT carried; each emits an ``input_deposit``
            by-reference block into ``bundle_manifest.json`` and a URL-bearing ``File``
            part into the crate instead. Omit it (the default) and the bundle is
            SELF-CONTAINED: every cfg-declared input is carried (ADR-9).
    """
    analysis_dir = analysis.analysis_paths.analysis_dir
    plots_dir = analysis_dir / BUNDLE_PLOTS_SUBDIR
    if not plots_dir.exists() or not list(plots_dir.rglob("*.manifest.json")):
        raise FileNotFoundError(
            f"No *.manifest.json sidecars found under {plots_dir}. "
            f"Bundle emission requires a completed render_report(). "
            f"Run analysis.render_report() on HPC first."
        )

    sources_by_renderer = harvest_source_paths(plots_dir, analysis_dir)
    git_sha = _get_toolkit_git_sha()
    analysis_id = analysis.cfg_analysis.analysis_id

    if output_path is None:
        output_path = (
            analysis_dir / BUNDLE_OUTPUT_SUBDIR
            / f"{analysis_id}_{git_sha}_v{BUNDLE_SCHEMA_VERSION}.zip"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Function-local import: config/bundle_exclude.py imports hhemt.bundle._path_policy
    # (a leaf), and reprex_taxonomy.py documents the invariant that no module reachable
    # from hhemt.bundle.__init__ may import hhemt.config.* at module scope. Same discipline
    # as the function-local `all_field_bucket` import in _scrub_user_bucket_fields.
    from hhemt.config.bundle_exclude import BundleExcludeConfig

    if isinstance(exclude_config, Path | str):
        import yaml

        exclude_config = BundleExcludeConfig.model_validate(
            yaml.safe_load(Path(exclude_config).read_text()) or {}
        )

    with _staging_dir(output_path.parent) as staging:
        _harvest_and_copy_sources(sources_by_renderer, analysis_dir, staging)
        _copy_bundle_baseline(analysis_dir, staging)
        aggregated_invariants = _copy_configs_with_relative_paths(analysis, staging)
        _emit_resolved_brand_theme(analysis, staging)
        _copy_supporting_files(analysis, staging)
        input_deposits = _copy_declared_inputs(analysis, staging, exclude_config)
        _emit_runnable_template_set(staging)
        _upgrade_crate_to_workflow_run_crate(staging)
        _annotate_crate_excluded_inputs(staging, input_deposits)
        _write_bundle_manifest(
            staging,
            sources_by_renderer=sources_by_renderer,
            analysis_id=analysis_id,
            git_sha=git_sha,
            bundle_root_invariants=aggregated_invariants,
            input_deposits=input_deposits,
        )
        _emit_bundle_zip(staging, output_path)

    return output_path


def _harvest_and_copy_sources(
    sources_by_renderer: dict[str, list[Path]],
    analysis_dir: Path,
    staging: Path,
) -> None:
    """Copy each declared source path into the staging dir, preserving
    its relative position under analysis_dir.

    A declared source that does not exist on disk is SKIPPED (with a warning),
    not fatal. ADR-6 D3 lets renderers declare an expected source unconditionally
    even when it is legitimately absent — e.g. ``disk_utilization`` declares
    ``_status/_du.json`` and renders a "re-run processing to populate" placeholder
    when the sentinel is missing (a normal state for a sensitivity master, whose
    consolidate path does not write the analysis-scope DU sentinel). Hard-raising
    on such a source made ``bundle_report_data()`` crash on every sensitivity
    master. The skip keeps the emit side consistent with the declare side; the
    warning preserves auditability (a source that vanished after render shows up
    here rather than failing silently)."""
    for paths in sources_by_renderer.values():
        for src in paths:
            try:
                rel = src.resolve().relative_to(analysis_dir.resolve())
            except ValueError:
                rel = Path("external") / src.name
            if not src.exists():
                warnings.warn(
                    f"Bundle harvest: declared source {rel} does not exist on disk; "
                    f"skipping it (ADR-6 D3 permits renderers to declare an expected "
                    f"source unconditionally even when absent, e.g. _status/_du.json "
                    f"on a sensitivity master). If this source was expected to be "
                    f"present, it may have been removed after render_report().",
                    stacklevel=2,
                )
                continue
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)


def _copy_bundle_baseline(analysis_dir: Path, staging: Path) -> None:
    baseline = staging / BUNDLE_BASELINE_SUBDIR
    baseline.mkdir(parents=True, exist_ok=True)
    for fmt in ("html", "zip"):
        src = analysis_dir / f"analysis_report.{fmt}"
        if src.exists():
            shutil.copy2(src, baseline / src.name)


def _copy_configs_with_relative_paths(
    analysis: TRITONSWMM_analysis, staging: Path
) -> dict[str, dict]:
    """Copy cfg_system.yaml and cfg_analysis.yaml with all Pydantic
    ``Path``-typed fields rewritten per the per-field policy table in
    ``_path_policy._PATH_FIELD_POLICY``. Returns a dict keyed by cfg
    attribute (``cfg_system`` / ``cfg_analysis``) mapping to the
    ``RewriteResult.invariants`` dict from that cfg's rewrite — consumed
    by the Phase 3 manifest extension (``bundle_root_invariants``)."""
    import yaml

    aggregated: dict[str, dict] = {}
    for cfg_attr, filename in (
        ("cfg_system", "cfg_system.yaml"),
        ("cfg_analysis", "cfg_analysis.yaml"),
    ):
        cfg = (
            analysis._system.cfg_system
            if cfg_attr == "cfg_system"
            else analysis.cfg_analysis
        )
        cfg_dict = cfg.model_dump(mode="json")
        result = _rewrite_paths_to_relative(
            cfg_dict,
            cfg_model=type(cfg),
            analysis_dir=analysis.analysis_paths.analysis_dir,
            system_directory=analysis._system.cfg_system.system_directory,
        )
        # `result.invariants` is consumed by the Phase 3 manifest extension
        # (bundle_root_invariants). Plan Phase 3 captures it per-cfg.
        aggregated[cfg_attr] = result.invariants
        scrubbed = _scrub_user_bucket_fields(result.cfg_dict, cfg_model=type(cfg))
        (staging / filename).write_text(
            yaml.safe_dump(scrubbed, sort_keys=False)
        )
    return aggregated


def _scrub_user_bucket_fields(
    cfg_dict: dict, *, cfg_model: type[cfgBaseModel]
) -> dict:
    """Null every non-Path config field whose reprex bucket is ``"user"`` (ADR-9
    all-field scrub / C-ZERO-USER-INFO).

    The Path fields are already scrubbed by their ``_PATH_FIELD_POLICY`` entry (the
    two software-dir fields null via ``IS_NONE_ACCEPTABLE``). This SUPERSET pass walks
    the remaining NON-Path fields and nulls any that bucket ``"user"`` — a
    guard-and-verify pass today (no non-path field is ``"user"``) and a leak-catcher
    for any future ``"user"`` field. A key that is not a bucketed config field (e.g. a
    nested sub-model like ``crs``) raises ``KeyError`` from ``all_field_bucket`` and is
    skipped; the reprex_taxonomy totality test guarantees every real field is bucketed.
    """
    from hhemt.config.reprex_taxonomy import (  # function-local: acyclicity (never module-top)
        all_field_bucket,
    )

    path_fields = set(enumerate_path_fields(cfg_model))
    out = dict(cfg_dict)
    for name in list(out.keys()):
        if name in path_fields:
            continue  # already handled by the _PATH_FIELD_POLICY rewrite above
        try:
            bucket = all_field_bucket(name)
        except KeyError:
            continue  # not a bucketed scalar config field (defensive; nested sub-models)
        if bucket == "user":
            out[name] = None
    return out


def _rewrite_paths_to_relative(
    cfg_dict: dict,
    cfg_model: type[cfgBaseModel],
    analysis_dir: Path,
    system_directory: Path,
) -> RewriteResult:
    """Rewrite absolute Path fields in ``cfg_dict`` per the per-field
    policy table.

    Uses Pydantic v2 model introspection (``model_fields``) to enumerate
    Path-typed field names; each is then routed through its
    ``PathPolicy`` entry in ``_PATH_FIELD_POLICY``. Fields not declared
    on ``cfg_model`` as ``Path`` / ``Optional[Path]`` are passed through
    unchanged.

    Returns:
        ``RewriteResult`` with the rewritten ``cfg_dict`` and a
        per-policy ``invariants`` dict suitable for inclusion in the
        Phase 3 ``bundle_manifest.json`` ``bundle_root_invariants`` key.
    """
    analysis_root = analysis_dir.resolve()
    system_root = system_directory.resolve()
    path_fields = enumerate_path_fields(cfg_model)
    invariants: dict[str, list[str]] = {policy.value: [] for policy in PathPolicy}
    out = dict(cfg_dict)

    for name in path_fields:
        if name not in _PATH_FIELD_POLICY:
            raise KeyError(
                f"Pydantic field '{name}' on {cfg_model.__name__} is typed "
                f"as Path/Optional[Path] but has no entry in "
                f"_PATH_FIELD_POLICY. Add a policy entry in "
                f"bundle/_path_policy.py."
            )
        policy = _PATH_FIELD_POLICY[name]
        value = out.get(name)
        new_value = _apply_policy(
            value,
            policy=policy,
            analysis_root=analysis_root,
            system_root=system_root,
        )
        out[name] = new_value
        invariants[policy.value].append(name)

    return RewriteResult(cfg_dict=out, invariants=invariants)


def _apply_policy(
    value: Any,
    *,
    policy: PathPolicy,
    analysis_root: Path,
    system_root: Path,
) -> Any:
    """Apply a single ``PathPolicy`` to a cfg field value."""
    if policy is PathPolicy.FORCED_DOT:
        return "."
    if policy is PathPolicy.IS_NONE_ACCEPTABLE:
        return None
    if policy is PathPolicy.HELPER_RESOLVED:
        # Reserved for future runtime-derived fields. Pass through
        # unchanged in Phase 1.
        return value
    if policy is PathPolicy.BUNDLE_RELATIVE_LIST:
        # list[Path] field (e.g. static_plot_configs). A None or empty
        # list serializes to []. Otherwise rewrite each element through
        # the same absolute-to-relative logic as the scalar policy — the
        # scalar branch below only handles a single str and would pass a
        # list through unrewritten (absolute paths leaking into the bundle).
        if not value:
            return []
        return [
            _rewrite_absolute_to_relative(
                elem,
                analysis_root=analysis_root,
                system_root=system_root,
            )
            for elem in value
        ]
    if value is None:
        if policy is PathPolicy.BUNDLE_RELATIVE_OR_NONE:
            return None
        # BUNDLE_RELATIVE on a None value is a misconfiguration — the
        # field is declared required (Path, not Optional[Path]) yet
        # serialized as None. Pass through; Pydantic load-side will
        # reject it at consume time.
        return value
    return _rewrite_absolute_to_relative(
        value,
        analysis_root=analysis_root,
        system_root=system_root,
    )


def _rewrite_absolute_to_relative(
    value: Any,
    *,
    analysis_root: Path,
    system_root: Path,
) -> Any:
    """Rewrite an absolute path string to its bundle-relative form.

    Resolution order: ``analysis_root`` first, then ``system_root``, then
    ``external/{filename}`` fallback. Mirrors the
    ``_harvest_and_copy_sources`` fallback at the file-copy layer so the
    cfg and the staging-dir layout agree on where the file lives.
    Non-string values are passed through unchanged.
    """
    if not isinstance(value, str):
        return value
    try:
        p = Path(value)
        if not p.is_absolute():
            return value
        pr = p.resolve()
        if pr.is_relative_to(analysis_root):
            return str(pr.relative_to(analysis_root))
        if pr.is_relative_to(system_root):
            return str(pr.relative_to(system_root))
        return str(Path("external") / pr.name)
    except (ValueError, OSError):
        return value


def _emit_resolved_brand_theme(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    """D-9: serialize the RESOLVED brand theme into the bundle and repoint the
    bundled cfg_analysis.yaml::brand_theme at it, so regenerate_report reproduces
    the deploying lab's colors instead of an unresolvable HPC-side path.

    R-6: source the RESOLVED theme (self._brand_theme, set by run() from the
    explicit-override -> cfg-field -> default ladder) so a run(override_brand_theme=)
    override survives into the bundle. getattr-fallback to the cfg field/default
    covers a fresh-instance bundle emit that never had run() called (mirrors the
    render_report EDIT 3c pattern).
    """
    import yaml

    from ..config.brand_theme import DEFAULT_BRAND_THEME
    from ..config.loaders import load_brand_theme

    resolved = getattr(analysis, "_brand_theme", None)
    if resolved is None:
        cfg_brand = analysis.cfg_analysis.brand_theme
        resolved = load_brand_theme(cfg_brand) if cfg_brand is not None else DEFAULT_BRAND_THEME
    (staging / "brand_theme.resolved.yaml").write_text(
        yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False)
    )
    # Repoint the already-written bundled cfg_analysis.yaml at the sidecar so the
    # consume-side load_brand_theme(cfg_analysis.brand_theme) resolves locally.
    cfg_path = staging / "cfg_analysis.yaml"
    cfg_dict = yaml.safe_load(cfg_path.read_text())
    cfg_dict["brand_theme"] = "brand_theme.resolved.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))


def _copy_supporting_files(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    analysis_dir = analysis.analysis_paths.analysis_dir
    # Snakefile is renamed to Snakefile.source so the bundle's regen-only
    # generated Snakefile (written by bundle/snakefile_generator.py at consume
    # time) can take the canonical "Snakefile" filename. The .source variant is
    # preserved for debugging value per the Phase 2 D1 resolution.
    snakefile_src = analysis_dir / "Snakefile"
    if snakefile_src.exists():
        shutil.copy2(snakefile_src, staging / "Snakefile.source")
    for fname in (
        VERSION_FILE_NAME,
        "scenario_status.csv",
        "sensitivity_analysis_definition.csv",
        "ro-crate-metadata.json",
    ):
        src = analysis_dir / fname
        if src.exists():
            shutil.copy2(src, staging / fname)
    # NOTE: the weather-events CSV (cfg_analysis.weather_events_to_simulate) is no
    # longer copied here. It is a BUNDLE_RELATIVE cfg-declared input, so the
    # self-contained harvest (_copy_declared_inputs) carries it to EXACTLY the
    # location _rewrite_paths_to_relative writes into the cfg (external/{name} when
    # it lives outside analysis_dir). The old ad-hoc copy targeted the bundle ROOT
    # (Path(src.name)) for out-of-tree CSVs, which disagreed with the cfg rewrite's
    # external/{name} — reconciled by routing every declared input through the one
    # policy-driven copy path.
    status_dir = analysis_dir / BUNDLE_STATUS_SUBDIR
    if status_dir.exists():
        shutil.copytree(status_dir, staging / BUNDLE_STATUS_SUBDIR, dirs_exist_ok=True)
    plots_dir = analysis_dir / BUNDLE_PLOTS_SUBDIR
    if plots_dir.exists():
        shutil.copytree(plots_dir, staging / BUNDLE_PLOTS_SUBDIR, dirs_exist_ok=True)


def _sha256_file(path: Path) -> str:
    """Streaming 1 MiB-chunk sha256 of a file.

    The chunk size is load-bearing: the ``case manifest sha256 uses streaming chunked
    read byte identical to whole file`` stipulation binds every sha256 the toolkit emits
    or verifies, so the emit-side digest and the ingest-side
    ``_fetch_file_by_url(expected_sha256=...)`` verification agree by construction.
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_declared_inputs(
    analysis: TRITONSWMM_analysis,
    staging: Path,
    exclude_config: BundleExcludeConfig | None = None,
) -> list[dict]:
    """Self-contained emit (ADR-9): copy EVERY cfg-declared input file into the bundle
    at exactly the location ``_rewrite_paths_to_relative`` writes into the cfg, so a
    reconstituted config resolves every declared input on disk.

    Driven by the same ``_PATH_FIELD_POLICY`` table the rewrite consumes (one
    policy-driven copy path — never a second source list, so
    ``test_all_path_fields_have_policy``'s bidirectional guard already covers it).
    Overlaps with ``_harvest_and_copy_sources`` (a renderer-declared input that is also a
    cfg field) are idempotent overwrites. A declared-but-absent input is SKIPPED here;
    ``from_doi``'s fail-closed materialize gate raises on it at ingest.

    ADR-20 (as amended 2026-07-14) — the governed opt-out. When ``exclude_config`` names a
    field, its input(s) are NOT carried; instead an ``input_deposit`` block is returned for
    each, and ``_write_bundle_manifest`` records it. The block is FILE-level:
    ``{relpath, sha256, accessed, citation, contentUrl?, url?, identifier?}``. The toolkit
    computes ``relpath`` (via the SAME ``_rewrite_absolute_to_relative`` the cfg rewrite
    uses, so the block's path is byte-identical to where the reconstituted cfg looks),
    ``sha256``, and ``accessed``; the operator supplies the rest. Self-contained remains the
    default: no exclude-config => every input carried.

    Returns:
        The ``input_deposit`` blocks for the excluded inputs (empty when nothing is excluded).

    Raises:
        ConfigurationError: if an excluded input has no coordinate block, or resolves to a
            DIRECTORY (sha256 is undefined over a tree, so the by-reference record could not
            be integrity-pinned). Both are fail-closed: the toolkit MUST NOT emit a bundle
            that neither carries nor properly references a declared input — that defect
            would surface only after a DOI had been minted.
    """
    from hhemt.exceptions import ConfigurationError

    analysis_dir = analysis.analysis_paths.analysis_dir
    analysis_root = analysis_dir.resolve()
    system_root = analysis._system.cfg_system.system_directory.resolve()
    accessed = datetime.now(UTC).date().isoformat()
    deposits: list[dict] = []

    for cfg in (analysis._system.cfg_system, analysis.cfg_analysis):
        for name in enumerate_path_fields(type(cfg)):
            if _PATH_FIELD_POLICY.get(name) not in _SELF_CONTAINED_POLICIES:
                continue
            value = getattr(cfg, name, None)
            if value is None:
                continue
            excluded = exclude_config is not None and exclude_config.excludes(name)
            for elem in value if isinstance(value, list) else [value]:
                if elem is None:
                    continue
                src = Path(elem)
                if not src.is_absolute():
                    src = analysis_root / src
                src = src.resolve()
                if not src.exists():
                    continue  # declared-but-absent: ingest fail-closed handles it
                rel = _rewrite_absolute_to_relative(
                    str(src), analysis_root=analysis_root, system_root=system_root
                )
                if not isinstance(rel, str) or Path(rel).is_absolute():
                    continue  # unresolvable dest; leave for the ingest fail-closed gate

                if excluded:
                    ref = exclude_config.refs_for(name, src.name)
                    if ref is None:
                        raise ConfigurationError(
                            parameter=f"bundle_exclude_config.exclusions.{name}",
                            value=src.name,
                            reason=(
                                f"'{name}' is excluded but carries no coordinates for "
                                f"'{src.name}'. Every excluded input needs at least a "
                                f"'citation'; add a 'contentUrl' too if a consumer can "
                                f"download it directly. Without a block the bundle would "
                                f"neither carry nor reference this input."
                            ),
                        )
                    if src.is_dir():
                        raise ConfigurationError(
                            parameter=f"bundle_exclude_config.exclusions.{name}",
                            value=str(src),
                            reason=(
                                "excluded input resolves to a DIRECTORY; sha256 is undefined "
                                "over a tree, so the by-reference record cannot be "
                                "integrity-pinned. Deposit the directory as a single archive "
                                "and point the cfg field at that file, or carry it in-bundle."
                            ),
                        )
                    block = {
                        "relpath": rel,
                        "sha256": _sha256_file(src),
                        "accessed": accessed,
                        "citation": ref.citation,
                    }
                    # contentUrl present/absent IS the fetchable bit — omit the key entirely
                    # when absent rather than writing a null, so the manifest reads as a
                    # deliberate reference-only record.
                    for key in ("contentUrl", "url", "identifier"):
                        val = getattr(ref, key)
                        if val is not None:
                            block[key] = val
                    deposits.append(block)
                    continue

                dest = staging / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dest)

    return deposits


#: Bundle-root filenames for the minimal runnable set (reprex carriage, ADR-10).
REPREX_CONFIG_FILENAME = "reprex_config.yaml"
HPC_TEMPLATE_FILENAME = "hpc_system_config.template.yaml"
#: The workflow file the WRC mainEntity points at (the bundled, generated Snakefile).
_WORKFLOW_RELPATH = "Snakefile.source"


def _emit_runnable_template_set(staging: Path) -> None:
    """Carry the minimal runnable set into the bundle root: a ``reprex_config.yaml``
    template (the target user's USER-bucket + HPC-selector fields) and a scrubbed
    ``hpc_system_config`` template (the ``{your-allocation}``-placeholder form).

    Both files carry ONLY placeholders — never the producer's real account / login-node
    / SIF path (research-reproducibility finding 5: ship a template + report page, never
    the populated original), so the zero-user-info gate passes trivially. The
    reprex_config key set is sourced from ``reprex_config.model_fields`` so it stays in
    lockstep with the model if a field is added.
    """
    import yaml

    from hhemt.config.reprex_config import reprex_config

    placeholders = {
        "default_account": "{your-allocation}",
        "login_node": "{your-login-node}",
        "sif_path": "/scratch/{your-allocation}/tritonswmm.sif",
        "scratch_dir": "/scratch/{your-allocation}",
        "target_ensemble_partition": "{your-gpu-partition}",
        "target_setup_and_analysis_processing_partition": "{your-cpu-partition}",
    }
    reprex_template = {
        name: placeholders.get(name, "{fill-in}") for name in reprex_config.model_fields
    }
    (staging / REPREX_CONFIG_FILENAME).write_text(
        "# reprex_config.yaml — fill in with YOUR system's values, then run reprex().\n"
        "# USER-bucket (host-local) fields + the HPC-revisable partition selectors only.\n"
        + yaml.safe_dump(reprex_template, sort_keys=False)
    )
    hpc_template = {
        "default_account": "{your-allocation}",
        "login_node": "{your-login-node}",
        "sif_path": "/scratch/{your-allocation}/tritonswmm.sif",
    }
    (staging / HPC_TEMPLATE_FILENAME).write_text(
        "# hpc_system_config template — the HPC-specific info you must revise to run\n"
        "# this bundle on YOUR cluster. Placeholders only; contains zero producer info.\n"
        + yaml.safe_dump(hpc_template, sort_keys=False)
    )


def _upgrade_crate_to_workflow_run_crate(staging: Path) -> None:
    """Patch the bundle's copied ``ro-crate-metadata.json`` into a Workflow-Run-Crate
    (``mainEntity`` = the bundled ``Snakefile.source``), in place.

    Reuses the by-reference SIF + ``input_parts`` already in the copied sidecar (no
    ``sif_spec`` reconstruction — sidesteps the ``_case_manifest`` gap) and reserializes
    via the canonical byte-deterministic path. No-op when the crate sidecar or the
    ``Snakefile.source`` is absent (a bundle without a workflow file is not a WRC).
    """
    from hhemt.metadata import (
        canonical_jsonld_from_doc,
        upgrade_doc_to_workflow_run_crate,
    )

    sidecar = staging / "ro-crate-metadata.json"
    if not sidecar.exists() or not (staging / _WORKFLOW_RELPATH).exists():
        return
    doc = json.loads(sidecar.read_text())
    upgrade_doc_to_workflow_run_crate(doc, workflow_relpath=_WORKFLOW_RELPATH)
    sidecar.write_text(canonical_jsonld_from_doc(doc))


def reconstitute_runnable_config(
    bundle_root: Path,
    *,
    target_path: Path | None = None,
    software_dir: Path | None = None,
) -> Path:
    """Synthesize a runnable ``system_config.yaml`` from a reprex bundle (ADR-10, R5).

    Reads the bundle's scrubbed ``cfg_system.yaml`` and writes a ``system_config.yaml``
    whose bundle-relative Path fields are resolved to absolute paths under
    ``bundle_root`` (so the carried, self-contained inputs resolve at load-time), and
    every EXPERIMENT-bucket field is preserved verbatim. Returns the written path
    (``bundle_root/system_config.yaml`` unless ``target_path`` overrides).

    The two software-dir fields (``SWMM_software_directory`` /
    ``TRITONSWMM_software_directory``) are toolkit-owned OUTPUTS — build dirs the toolkit
    creates at target-side setup, exempt from the load-time existence check (D4). Their
    treatment splits by consume path:

    - ``software_dir=None`` (the RENDER path): both stay ``null``. Correct for
      bundle-local EDA/render (``load_eda_context`` never constructs a
      ``TRITONSWMM_system``).
    - ``software_dir`` provided (the RUN path, e.g. ``from_doi``): both are set to
      writable target-side subdirs (``software_dir/'tritonswmm'`` /
      ``software_dir/'swmm'``) so ``TRITONSWMM_system`` constructs (its ``__init__``
      hard-raises on a null ``TRITONSWMM_software_directory``) and a later ``run()`` can
      clone+build there. A runnable experiment needs REAL (not-yet-existing,
      existence-exempt) paths, not null — the null-ing is correct for render and wrong
      for run.
    """
    import yaml

    from hhemt.config.system import system_config

    bundle_root = Path(bundle_root).resolve()
    cfg_dict = yaml.safe_load((bundle_root / "cfg_system.yaml").read_text())
    path_fields = set(enumerate_path_fields(system_config))
    out = dict(cfg_dict)
    for name in path_fields:
        value = out.get(name)
        if value is None:
            continue
        if isinstance(value, list):  # BUNDLE_RELATIVE_LIST (none on system_config today)
            out[name] = [
                str((bundle_root / v).resolve()) if not Path(v).is_absolute() else v
                for v in value
            ]
            continue
        if isinstance(value, str) and not Path(value).is_absolute():
            out[name] = str((bundle_root / value).resolve())
    if software_dir is None:
        # RENDER path: toolkit-owned build dirs stay null (bundle-local EDA only).
        out["SWMM_software_directory"] = None
        out["TRITONSWMM_software_directory"] = None
    else:
        # RUN path: point the build dirs at a writable target-side location so a
        # TRITONSWMM_system constructs and run() can build there.
        software_dir = Path(software_dir).resolve()
        out["SWMM_software_directory"] = str(software_dir / "swmm")
        out["TRITONSWMM_software_directory"] = str(software_dir / "tritonswmm")
    target = target_path if target_path is not None else bundle_root / "system_config.yaml"
    Path(target).write_text(yaml.safe_dump(out, sort_keys=False))
    return Path(target)


def reconstitute_runnable_analysis_config(
    bundle_root: Path, *, target_path: Path | None = None
) -> Path:
    """Synthesize a runnable ``analysis_config.yaml`` from a reprex bundle — the
    analysis-side sibling of :func:`reconstitute_runnable_config`.

    The bundle's ``cfg_analysis.yaml`` stores its Path fields bundle-relative
    (``_PATH_FIELD_POLICY``): ``analysis_dir`` is the FORCED_DOT ``"."`` bundle-root
    marker, list fields (``static_plot_configs``) are element-wise relative, and every
    other declared input is ``BUNDLE_RELATIVE``/``BUNDLE_RELATIVE_OR_NONE``. This helper
    rewrites every non-absolute Path value to ``str((bundle_root / v).resolve())`` — which
    maps ``analysis_dir: "."`` to ``bundle_root`` (honoring the FORCED_DOT invariant) and
    rebases ``sensitivity_analysis`` and the rest onto the carried, self-contained inputs.
    It is the SINGLE rebase implementation: ``reprex()`` composes it rather than
    hand-rebasing ``sensitivity_analysis`` inline. Returns the written path
    (``bundle_root/analysis_config.yaml`` unless ``target_path`` overrides).
    """
    import yaml

    from hhemt.config.analysis import analysis_config

    bundle_root = Path(bundle_root).resolve()
    cfg_dict = yaml.safe_load((bundle_root / "cfg_analysis.yaml").read_text())
    path_fields = set(enumerate_path_fields(analysis_config))
    out = dict(cfg_dict)
    for name in path_fields:
        value = out.get(name)
        if value is None:
            continue
        if isinstance(value, list):  # BUNDLE_RELATIVE_LIST (e.g. static_plot_configs)
            out[name] = [
                str((bundle_root / v).resolve()) if not Path(v).is_absolute() else v
                for v in value
            ]
            continue
        if isinstance(value, str) and not Path(value).is_absolute():
            out[name] = str((bundle_root / value).resolve())
    target = (
        target_path if target_path is not None else bundle_root / "analysis_config.yaml"
    )
    Path(target).write_text(yaml.safe_dump(out, sort_keys=False))
    return Path(target)


def _annotate_crate_excluded_inputs(staging: Path, input_deposits: list[dict]) -> None:
    """Give each EXCLUDED input a by-reference ``File`` entity in the sidecar crate.

    ADR-20 originally ratified "a DESCRIPTIVE (no-``downloadUrl``) File part per excluded
    input". That clause was a NO-OP and was REVERSED (user-re-ratified 2026-07-14):
    ``build_analysis_crate``'s ``input_parts`` loop ALREADY emits a URL-less ``File`` with a
    bundle-relative ``@id`` for every CARRIED input, so a URL-less part distinguishes
    nothing — the published crate would have asserted a ``File`` at a bundle-relative path
    that is absent from the payload, with nothing marking it external and no way to obtain
    it. That is an immutable public FAIR defect inside a minted DOI.

    The corrected entity is the RO-Crate-native pattern for a file omitted for "licensing
    concerns, large data sizes, privacy" (the BagIt ``fetch.txt`` analogue): the ``@id``
    stays BUNDLE-RELATIVE (the file is conceptually payload — the reconstituted config
    resolves it there — merely not transferred), and the access/citation vocabulary is
    attached. ``contentUrl`` is omitted iff the input is unfetchable-by-design.

    Embedded-core safety: ``partition_core_vs_sidecar`` is an ALLOW-LIST filter and none of
    the added keys are in ``_EMBEDDED_PROV_KEYS``, so they are dropped from the
    deterministic embedded core automatically and live only in the sidecar crate — the
    FAIR-facing artifact. The D3-deferred embedded-core flip is still avoided at zero cost.
    """
    if not input_deposits:
        return
    crate_path = staging / "ro-crate-metadata.json"
    if not crate_path.exists():
        return  # no crate emitted for this analysis; nothing to annotate

    crate = json.loads(crate_path.read_text())
    graph = crate.get("@graph", [])
    by_id = {e.get("@id"): e for e in graph if isinstance(e, dict)}

    for block in input_deposits:
        entity = by_id.get(block["relpath"])
        if entity is None:
            entity = {"@id": block["relpath"], "@type": "File"}
            graph.append(entity)
        entity.setdefault("@type", "File")
        entity["sha256"] = block["sha256"]
        entity["sdDatePublished"] = block["accessed"]
        entity["citation"] = block["citation"]
        for key in ("contentUrl", "url", "identifier"):
            if key in block:
                entity[key] = block[key]
            else:
                entity.pop(key, None)

    crate["@graph"] = graph
    crate_path.write_text(json.dumps(crate, indent=2))


def _write_bundle_manifest(
    staging: Path,
    sources_by_renderer: dict[str, list[Path]],
    analysis_id: str,
    git_sha: str,
    bundle_root_invariants: dict | None = None,
    input_deposits: list[dict] | None = None,
) -> None:
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "layout_version": LAYOUT_VERSION,
        "toolkit_git_sha": git_sha,
        "analysis_id": analysis_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_paths_by_renderer": {
            name: [str(p) for p in paths]
            for name, paths in sources_by_renderer.items()
        },
    }
    if bundle_root_invariants is not None:
        manifest["bundle_root_invariants"] = bundle_root_invariants
    if input_deposits:
        # ADR-20: the by-reference record for each EXCLUDED input. Absent (not empty) when
        # the bundle is self-contained, so a self-contained manifest is byte-identical to
        # what it was before this feature existed.
        manifest["input_deposit"] = input_deposits
    (staging / BUNDLE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2)
    )


def _emit_bundle_zip(staging: Path, output_path: Path) -> None:
    # Emit a deterministic uncompressed zip from staging.
    #
    # Determinism is achieved via two mechanisms applied jointly:
    # (1) sorted file ordering (rglob output is sorted so the same
    #     staging tree always produces the same archive entry order);
    # (2) fixed date_time on every ZipInfo entry — the value
    #     (1980, 1, 1, 0, 0, 0) is the zipfile module's minimum
    #     valid date, eliminating mtime variability from real
    #     filesystem timestamps.
    #
    # Compression: ZIP_STORED (uncompressed at the file-byte level).
    # Same rationale as the prior tar format: zarr is the bulk of
    # bundle size and is internally chunked-compressed; external zip
    # compression adds CPU cost without size win.
    fixed_date_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_STORED
    ) as zf:
        for entry in sorted(staging.rglob("*")):
            if entry.is_dir():
                # Skip directory entries; zipfile reconstructs directory
                # structure from file paths at extraction time.
                continue
            arcname = entry.relative_to(staging)
            info = zipfile.ZipInfo(
                filename=str(arcname), date_time=fixed_date_time
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16  # rw-r--r-- file mode
            zf.writestr(info, entry.read_bytes())


def _get_toolkit_git_sha(strict: bool = True) -> str:
    """Resolve the toolkit's git SHA for bundle provenance.

    strict=True (emit-side): raise ConfigurationError if unavailable.
    strict=False (consume-side): return "unknown" if unavailable.
    """
    from hhemt.exceptions import ConfigurationError

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        if not sha:
            if not strict:
                return "unknown"
            raise ConfigurationError(
                field="toolkit_git_sha",
                message=(
                    "git rev-parse returned empty SHA — "
                    "toolkit may be in a detached state"
                ),
                config_path=None,
            )
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        if not strict:
            return "unknown"
        raise ConfigurationError(
            field="toolkit_git_sha",
            message=(
                "Cannot resolve toolkit git SHA for bundle provenance: "
                f"{exc}. Ensure git is installed and the hhemt "
                "package is installed from a git checkout (not a wheel)."
            ),
            config_path=None,
        ) from exc


@contextlib.contextmanager
def _staging_dir(parent: Path):
    """Sibling staging directory; cleaned up on exit."""
    with tempfile.TemporaryDirectory(
        prefix="bundle_staging_", dir=parent
    ) as tmp:
        yield Path(tmp)

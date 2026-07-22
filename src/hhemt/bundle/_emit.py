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


def emit_bundle(
    analysis: TRITONSWMM_analysis,
    output_path: Path | None = None,
) -> Path:
    """Emit a portable render bundle from a completed HPC analysis.

    The bundle file set is the union of source paths declared via
    prov.artist().add_channel(...) calls during the most recent
    render_report() execution, harvested from *.manifest.json sidecars
    under {analysis_dir}/plots/. Configs are rewritten to relative paths.
    The HPC-baseline analysis_report.{html,zip} are preserved under
    bundle_baseline/.
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
        output_path = analysis_dir / BUNDLE_OUTPUT_SUBDIR / f"{analysis_id}_{git_sha}_v{BUNDLE_SCHEMA_VERSION}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _staging_dir(output_path.parent) as staging:
        _harvest_and_copy_sources(sources_by_renderer, analysis_dir, staging)
        _copy_bundle_baseline(analysis_dir, staging)
        aggregated_invariants = _copy_configs_with_relative_paths(analysis, staging)
        _emit_resolved_brand_theme(analysis, staging)
        _copy_supporting_files(analysis, staging)
        _emit_hpc_identity(analysis, staging)
        _emit_runnable_template_set(staging)
        _upgrade_crate_to_workflow_run_crate(staging)
        _write_bundle_manifest(
            staging,
            sources_by_renderer=sources_by_renderer,
            analysis_id=analysis_id,
            git_sha=git_sha,
            bundle_root_invariants=aggregated_invariants,
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


def _copy_configs_with_relative_paths(analysis: TRITONSWMM_analysis, staging: Path) -> dict[str, dict]:
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
        cfg = analysis._system.cfg_system if cfg_attr == "cfg_system" else analysis.cfg_analysis
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
        (staging / filename).write_text(yaml.safe_dump(scrubbed, sort_keys=False))
    return aggregated


def _scrub_user_bucket_fields(cfg_dict: dict, *, cfg_model: type[cfgBaseModel]) -> dict:
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
    # case.yaml — the case manifest carries case_name (BLOCKING experiment-identity
    # field, already in _compatibility._EXPERIMENT_IDENTITY_FIELDS). Resolved from the
    # case_manifest_yaml constructor arg (D6); None on paths that do not set it.
    # BundleableAnalysis contract attr — direct access (not getattr-None) fails loud on a non-conforming input.
    case_yaml = analysis.case_manifest_yaml
    if case_yaml is not None and Path(case_yaml).exists():
        shutil.copy2(Path(case_yaml), staging / "case.yaml")
    # Copy the weather-events CSV referenced by cfg_analysis.weather_events_to_simulate.
    # The cfg rewrite preserves its relative position (typically directly under
    # analysis_dir for synth fixtures); the file itself must travel with the bundle
    # so analysis.py:164's pd.read_csv resolves at consume time.
    weather_events_csv = analysis.cfg_analysis.weather_events_to_simulate
    if weather_events_csv is not None and Path(weather_events_csv).exists():
        src = Path(weather_events_csv).resolve()
        try:
            rel = src.relative_to(analysis_dir.resolve())
        except ValueError:
            rel = Path(src.name)
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    status_dir = analysis_dir / BUNDLE_STATUS_SUBDIR
    if status_dir.exists():
        shutil.copytree(status_dir, staging / BUNDLE_STATUS_SUBDIR, dirs_exist_ok=True)
    plots_dir = analysis_dir / BUNDLE_PLOTS_SUBDIR
    if plots_dir.exists():
        shutil.copytree(plots_dir, staging / BUNDLE_PLOTS_SUBDIR, dirs_exist_ok=True)


#: Bundle-root filename for the scrubbed compute-config identity surface (combine-side).
HPC_IDENTITY_FILENAME = "hpc_system_config.identity.yaml"


def _emit_hpc_identity(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    """Emit the SCRUBBED compute-config identity surface for the combine-side checker.

    Allow-list-BY-CONSTRUCTION (D7): reach ONLY for the named compute-config-identity
    fields (``partitions`` map + ``gpu_allocation_flavor``) — the fields two experiments
    must agree on for an intercomparison to be meaningful — and NEVER emit any
    USER-bucket scalar (default_account / login_node / container.sif_path). No
    enumerate-and-null (that is fail-open against future schema drift). No-op when the
    analysis carries no hpc_system_config (local/native runs). Passes the zero-user-info
    gate trivially (it never writes a producer value). NOT the reprex template.
    """
    import yaml

    # BundleableAnalysis contract attr — direct access (not getattr-None) fails loud on a non-conforming input.
    cfg_hpc = analysis.cfg_hpc_system
    if cfg_hpc is None:
        return
    dumped = cfg_hpc.model_dump(mode="json")
    out = {k: dumped[k] for k in ("partitions", "gpu_allocation_flavor") if k in dumped}
    (staging / HPC_IDENTITY_FILENAME).write_text(yaml.safe_dump(out, sort_keys=False))


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
    reprex_template = {name: placeholders.get(name, "{fill-in}") for name in reprex_config.model_fields}
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


def reconstitute_runnable_config(bundle_root: Path, *, target_path: Path | None = None) -> Path:
    """Synthesize a runnable ``system_config.yaml`` from a reprex bundle (ADR-10, R5).

    Reads the bundle's scrubbed ``cfg_system.yaml`` and writes a ``system_config.yaml``
    whose bundle-relative Path fields are resolved to absolute paths under
    ``bundle_root`` (so the by-reference inputs the target user fetched into the bundle
    resolve at load-time), the two software-dir fields stay ``null`` (toolkit-owned,
    exempt from the load-time existence check via the Phase-1 D4 relaxation), and every
    EXPERIMENT-bucket field is preserved verbatim. Returns the written path
    (``bundle_root/system_config.yaml`` unless ``target_path`` overrides).
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
            out[name] = [str((bundle_root / v).resolve()) if not Path(v).is_absolute() else v for v in value]
            continue
        if isinstance(value, str) and not Path(value).is_absolute():
            out[name] = str((bundle_root / value).resolve())
    # Software dirs stay null: toolkit-owned, created at target-side setup, never bundled.
    out["SWMM_software_directory"] = None
    out["TRITONSWMM_software_directory"] = None
    target = target_path if target_path is not None else bundle_root / "system_config.yaml"
    Path(target).write_text(yaml.safe_dump(out, sort_keys=False))
    return Path(target)


def _write_bundle_manifest(
    staging: Path,
    sources_by_renderer: dict[str, list[Path]],
    analysis_id: str,
    git_sha: str,
    bundle_root_invariants: dict | None = None,
) -> None:
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "layout_version": LAYOUT_VERSION,
        "toolkit_git_sha": git_sha,
        "analysis_id": analysis_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_paths_by_renderer": {name: [str(p) for p in paths] for name, paths in sources_by_renderer.items()},
    }
    if bundle_root_invariants is not None:
        manifest["bundle_root_invariants"] = bundle_root_invariants
    (staging / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


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
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for entry in sorted(staging.rglob("*")):
            if entry.is_dir():
                # Skip directory entries; zipfile reconstructs directory
                # structure from file paths at extraction time.
                continue
            arcname = entry.relative_to(staging)
            info = zipfile.ZipInfo(filename=str(arcname), date_time=fixed_date_time)
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
                message=("git rev-parse returned empty SHA — toolkit may be in a detached state"),
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
    with tempfile.TemporaryDirectory(prefix="bundle_staging_", dir=parent) as tmp:
        yield Path(tmp)

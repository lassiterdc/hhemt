"""Render-bundle emission helpers — the portable artifact for local
renderer iteration.

Public surface (called from Analysis.bundle_report_data() and
TRITONSWMM_sensitivity_analysis.bundle_report_data()):

  emit_bundle(analysis, output_path) -> Path

Helpers (private to this module):

  _harvest_and_copy_sources(...)
  _rewrite_paths_to_relative(...)
  _write_bundle_manifest(...)
  _emit_bundle_tar(...)

This module is opt-in only. Importing it does not trigger any side
effects; the only entry point is emit_bundle(). The method is invoked
from Analysis.bundle_report_data() (and the sensitivity parallel),
which is in turn invoked only from the `TRITON_SWMM_toolkit bundle`
CLI command. It is NEVER invoked by Analysis.run() or
Analysis.submit_workflow().
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from TRITON_SWMM_toolkit.bundle._path_policy import (
    _PATH_FIELD_POLICY,
    PathPolicy,
    RewriteResult,
    enumerate_path_fields,
)
from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
    harvest_source_paths,
)
from TRITON_SWMM_toolkit.version_migration.constants import (
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
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.base import cfgBaseModel


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
        output_path = (
            analysis_dir / BUNDLE_OUTPUT_SUBDIR
            / f"{analysis_id}_{git_sha}_v{BUNDLE_SCHEMA_VERSION}.tar"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _staging_dir(output_path.parent) as staging:
        _harvest_and_copy_sources(sources_by_renderer, analysis_dir, staging)
        _copy_bundle_baseline(analysis_dir, staging)
        aggregated_invariants = _copy_configs_with_relative_paths(analysis, staging)
        _copy_supporting_files(analysis, staging)
        _write_bundle_manifest(
            staging,
            sources_by_renderer=sources_by_renderer,
            analysis_id=analysis_id,
            git_sha=git_sha,
            bundle_root_invariants=aggregated_invariants,
        )
        _emit_bundle_tar(staging, output_path)

    return output_path


def _harvest_and_copy_sources(
    sources_by_renderer: dict[str, list[Path]],
    analysis_dir: Path,
    staging: Path,
) -> None:
    """Copy each declared source path into the staging dir, preserving
    its relative position under analysis_dir."""
    for paths in sources_by_renderer.values():
        for src in paths:
            try:
                rel = src.resolve().relative_to(analysis_dir.resolve())
            except ValueError:
                rel = Path("external") / src.name
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                raise FileNotFoundError(
                    f"Bundle harvest declared source path {src} but the file does "
                    f"not exist. The renderer's manifest sidecar declared this "
                    f"source via emit_plot_with_sources; data corruption between "
                    f"render_report() completion and bundle emission is the most "
                    f"likely cause."
                )
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
        (staging / filename).write_text(
            yaml.safe_dump(result.cfg_dict, sort_keys=False)
        )
    return aggregated


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
    ):
        src = analysis_dir / fname
        if src.exists():
            shutil.copy2(src, staging / fname)
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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths_by_renderer": {
            name: [str(p) for p in paths]
            for name, paths in sources_by_renderer.items()
        },
    }
    if bundle_root_invariants is not None:
        manifest["bundle_root_invariants"] = bundle_root_invariants
    (staging / BUNDLE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2)
    )


def _emit_bundle_tar(staging: Path, output_path: Path) -> None:
    """Emit a deterministic uncompressed tar from staging."""
    with tarfile.open(output_path, "w") as tar:
        for entry in sorted(staging.rglob("*")):
            arcname = entry.relative_to(staging)
            tar.add(entry, arcname=str(arcname), recursive=False)


def _get_toolkit_git_sha(strict: bool = True) -> str:
    """Resolve the toolkit's git SHA for bundle provenance.

    strict=True (emit-side): raise ConfigurationError if unavailable.
    strict=False (consume-side): return "unknown" if unavailable.
    """
    from TRITON_SWMM_toolkit.exceptions import ConfigurationError

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
                f"{exc}. Ensure git is installed and the TRITON-SWMM_toolkit "
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

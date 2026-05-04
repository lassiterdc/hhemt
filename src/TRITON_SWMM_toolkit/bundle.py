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

from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
    harvest_source_paths,
)
from TRITON_SWMM_toolkit.version_migration.constants import (
    BUNDLE_SCHEMA_VERSION,
    LAYOUT_VERSION,
)

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis


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
    plots_dir = analysis_dir / "plots"
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
            analysis_dir / "render_bundle"
            / f"{analysis_id}_{git_sha}_v{BUNDLE_SCHEMA_VERSION}.tar"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _staging_dir(output_path.parent) as staging:
        _harvest_and_copy_sources(sources_by_renderer, analysis_dir, staging)
        _copy_bundle_baseline(analysis_dir, staging)
        _copy_configs_with_relative_paths(analysis, staging)
        _copy_supporting_files(analysis_dir, staging)
        _write_bundle_manifest(
            staging,
            sources_by_renderer=sources_by_renderer,
            analysis_id=analysis_id,
            git_sha=git_sha,
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
                continue
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)


def _copy_bundle_baseline(analysis_dir: Path, staging: Path) -> None:
    baseline = staging / "bundle_baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    for fmt in ("html", "zip"):
        src = analysis_dir / f"analysis_report.{fmt}"
        if src.exists():
            shutil.copy2(src, baseline / src.name)


def _copy_configs_with_relative_paths(
    analysis: TRITONSWMM_analysis, staging: Path
) -> None:
    """Copy cfg_system.yaml and cfg_analysis.yaml with all path-typed
    fields rewritten to be relative to the bundle root."""
    import yaml

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
        cfg_dict = _rewrite_paths_to_relative(
            cfg_dict,
            analysis_dir=analysis.analysis_paths.analysis_dir,
            system_directory=analysis._system.cfg_system.system_directory,
        )
        (staging / filename).write_text(yaml.safe_dump(cfg_dict, sort_keys=False))


def _rewrite_paths_to_relative(
    cfg_dict: Any,
    analysis_dir: Path,
    system_directory: Path,
) -> Any:
    """Recursively walk the config dict; rewrite absolute paths under
    analysis_dir or system_directory as relative to the corresponding root."""
    analysis_root = analysis_dir.resolve()
    system_root = system_directory.resolve()

    def _rewrite_one(value: Any) -> Any:
        if isinstance(value, str):
            try:
                p = Path(value)
                if not p.is_absolute():
                    return value
                pr = p.resolve()
                if pr.is_relative_to(analysis_root):
                    return str(pr.relative_to(analysis_root))
                if pr.is_relative_to(system_root):
                    return str(pr.relative_to(system_root))
            except (ValueError, OSError):
                pass
            return value
        if isinstance(value, dict):
            return {k: _rewrite_one(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_rewrite_one(v) for v in value]
        return value

    return _rewrite_one(cfg_dict)


def _copy_supporting_files(analysis_dir: Path, staging: Path) -> None:
    for fname in (
        "Snakefile",
        "_version.json",
        "scenario_status.csv",
        "sensitivity_analysis_definition.csv",
    ):
        src = analysis_dir / fname
        if src.exists():
            shutil.copy2(src, staging / fname)
    status_dir = analysis_dir / "_status"
    if status_dir.exists():
        shutil.copytree(status_dir, staging / "_status", dirs_exist_ok=True)
    plots_dir = analysis_dir / "plots"
    if plots_dir.exists():
        shutil.copytree(plots_dir, staging / "plots", dirs_exist_ok=True)


def _write_bundle_manifest(
    staging: Path,
    sources_by_renderer: dict[str, list[Path]],
    analysis_id: str,
    git_sha: str,
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
    (staging / "bundle_manifest.json").write_text(
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

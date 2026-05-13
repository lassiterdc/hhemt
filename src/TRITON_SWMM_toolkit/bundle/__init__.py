"""Bundle subsystem — render-bundle emission and the consume-side
``Bundle`` class.

Public surface
--------------
- ``Bundle`` — the consume-side entry point for a render bundle. Used by
  callers that already have a bundle on disk and want to regenerate the
  analysis report locally without re-running ``Analysis.run()``.
- ``emit_bundle`` — the emit-side helper. Invoked by
  ``Analysis.bundle_report_data()`` and
  ``TRITONSWMM_sensitivity_analysis.bundle_report_data()`` to produce a
  portable render bundle from a completed HPC analysis.

``Bundle`` is deliberately NOT a subclass of ``TRITONSWMM_analysis``.
Bundle outputs are pre-computed; ``Analysis.run()`` is not callable
against a bundle. The class boundary is the user's binding constraint
per the bundle-portable-report-regeneration plan's Friction 5 design
recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from TRITON_SWMM_toolkit.bundle._emit import (
    _get_toolkit_git_sha,
    emit_bundle,
)
from TRITON_SWMM_toolkit.subprocess_utils import run_subprocess_with_tee
from TRITON_SWMM_toolkit.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
)

__all__ = [
    "Bundle",
    "emit_bundle",
    "_get_toolkit_git_sha",
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_SCHEMA_VERSION",
]


class Bundle:
    """A portable render bundle, ready for local report regeneration."""

    def __init__(self, root: Path, manifest: dict) -> None:
        self._root = root.resolve()
        self._manifest = manifest

    @classmethod
    def from_directory(cls, path: Path | str) -> "Bundle":
        # Construct a Bundle from a bundle directory on disk.
        #
        # The directory must contain bundle_manifest.json. All paths
        # resolve via bundle.root at call time — no os.chdir, no
        # persisted absolute paths.
        #
        # Schema version validation: the manifest's
        # bundle_schema_version must be <= the locally-installed
        # toolkit's BUNDLE_SCHEMA_VERSION. A higher version means the
        # bundle was emitted by a newer toolkit; the local
        # installation cannot guarantee correct read.
        #
        # Legacy-bundle backward compatibility: pre-Plan-Phase-3
        # bundles lack the bundle_root_invariants manifest key; this
        # is treated as {} (no enforced invariants — legacy bundles
        # relied on consume-side cwd-based path resolution, which
        # Plan Phases 1 + 3 supersede).
        root = Path(path).resolve()
        manifest_path = root / BUNDLE_MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No {BUNDLE_MANIFEST_FILENAME} under {root}. "
                f"Is this a render bundle?"
            )
        manifest = json.loads(manifest_path.read_text())
        try:
            bundle_version = manifest["bundle_schema_version"]
        except KeyError as exc:
            raise ValueError(
                f"Malformed manifest at {manifest_path}: missing "
                f"required 'bundle_schema_version' field. Legacy "
                f"bundles (pre-Plan-Phase-3) lack bundle_root_invariants "
                f"but do contain bundle_schema_version; absence of this "
                f"field indicates the manifest was not produced by any "
                f"version of the toolkit's bundle emitter."
            ) from exc
        if bundle_version > BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Bundle schema version {bundle_version} exceeds locally "
                f"installed BUNDLE_SCHEMA_VERSION={BUNDLE_SCHEMA_VERSION}. "
                f"Upgrade the local toolkit installation to read this bundle."
            )
        invariants = manifest.get("bundle_root_invariants", {})
        if not isinstance(invariants, dict):
            raise ValueError(
                f"bundle_root_invariants in {manifest_path} must be a "
                f"dict, got {type(invariants).__name__}."
            )
        return cls(root=root, manifest=manifest)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> dict:
        return self._manifest

    def absolute(self, rel: str | Path) -> Path:
        """Resolve a bundle-relative path to an absolute path under
        ``bundle.root``."""
        return (self._root / rel).resolve()

    def _read_static_backend(self) -> Literal["matplotlib", "plotly"]:
        # Resolution order:
        #   1. cfg_report.yaml::interactive::static_backend — the
        #      F1-introduced canonical snapshot written by emit_bundle.
        #   2. cfg_analysis.yaml::report::interactive::static_backend —
        #      retained for forward compatibility with the future F2
        #      schema canonicalization.
        #   3. InteractiveBackendConfig().static_backend — Pydantic
        #      default ("plotly" per Plan Phase 2 D3 + Decision 4).
        # Private (leading underscore) but kept on the public Bundle
        # class so test code can monkey-patch for backend-override
        # coverage.
        import yaml
        cfg_report_path = self._root / "cfg_report.yaml"
        if cfg_report_path.exists():
            cfg_report_data = yaml.safe_load(cfg_report_path.read_text())
            interactive_section = cfg_report_data.get("interactive", {})
            if "static_backend" in interactive_section:
                return interactive_section["static_backend"]
        cfg_analysis_path = self._root / "cfg_analysis.yaml"
        if not cfg_analysis_path.exists():
            raise FileNotFoundError(
                f"No cfg_analysis.yaml under {self._root}. Bundle is "
                f"malformed or this is not a render bundle."
            )
        cfg_analysis_data = yaml.safe_load(cfg_analysis_path.read_text())
        report_section = cfg_analysis_data.get("report", {})
        interactive_section = report_section.get("interactive", {})
        if "static_backend" in interactive_section:
            return interactive_section["static_backend"]
        from TRITON_SWMM_toolkit.config.report import (
            InteractiveBackendConfig,
        )
        return InteractiveBackendConfig().static_backend

    def regenerate_report(
        self, *, format: Literal["html", "zip"] = "html"
    ) -> Path:
        """Regenerate the analysis report from bundled data.

        Phase 2 wires (a) the regeneration-scoped Snakefile generator
        and (b) the report-templates staging step. Phase 3 wires the
        subprocess invocation, CLI integration, and the
        ``_read_static_backend()`` cfg-read that derives the static
        backend from ``cfg_report.yaml``'s ``static_backend`` field
        (default ``"plotly"`` per Decision 4 / Plan Phase 2 D3).

        Parameters
        ----------
        format : {"html", "zip"}
            Final report format. Default ``"html"`` is the user-facing
            default at the Python API surface — distinct from an
            internal in-code default. The static backend is NOT a
            caller-facing parameter; it is derived from the bundle's
            cfg files via ``self._read_static_backend()`` so the
            user-visible default is config-controlled (per the
            project's no-in-code-defaults principle).
        """
        from TRITON_SWMM_toolkit.bundle.snakefile_generator import (
            write_regeneration_snakefile,
        )
        from TRITON_SWMM_toolkit.workflow import _emit_report_artifacts

        static_backend = self._read_static_backend()
        _emit_report_artifacts(self._root)
        write_regeneration_snakefile(self._root, static_backend=static_backend)

        # Defense-in-depth stale-lock check (per Decision 3.1A). CLI does
        # silent cleanup before reaching here; this check catches lock state
        # for Python-API callers that bypass the CLI (notebook usage).
        locks_dir = self._root / ".snakemake" / "locks"
        if locks_dir.exists() and any(locks_dir.iterdir()):
            lock_paths = sorted(p.name for p in locks_dir.iterdir())
            raise RuntimeError(
                f"Stale Snakemake locks under {locks_dir}: {lock_paths}. "
                f"Run `python -m snakemake --unlock --snakefile "
                f"{self._root}/Snakefile --directory {self._root}` to clear "
                f"them, or delete the locks/ directory manually. Locks are "
                f"left behind by an interrupted prior render."
            )

        # Determine final report output path under bundle root.
        output_path = self._root / f"analysis_report.{format}"

        # Build snakemake invocation. cwd= is set on the subprocess (not
        # the parent process) per R3 — no process-global os.chdir.
        # run_subprocess_with_tee is imported at module level so test
        # code can monkeypatch the binding site (bundle.__init__) — see
        # VMS-9 tests in tests/test_bundle.py.
        logs_dir = self._root / "_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Mark all existing plot outputs as up-to-date so --report does
        # not attempt to re-run plot rules against absent source data.
        # The bundle contains rendered plots, NOT the source datatree;
        # without --touch, snakemake's needrun check fires because no
        # .snakemake/metadata/ exists on a fresh local working directory.
        # Per knowledge doc `library/knowledge/snakemake/rerun triggers.md`
        # and the Snakemake `--touch` documented contract.
        touch_cmd = [
            "snakemake",
            "--snakefile", str(self._root / "Snakefile"),
            "--directory", str(self._root),
            "--cores", "1",
            "--touch",
            "--quiet",
        ]
        touch_proc = run_subprocess_with_tee(
            touch_cmd,
            logfile=logs_dir / "regenerate_touch.log",
            cwd=self._root,
            echo_to_stdout=False,
        )
        if touch_proc.returncode != 0:
            raise RuntimeError(
                f"snakemake --touch failed with exit code "
                f"{touch_proc.returncode}. See log: "
                f"{logs_dir / 'regenerate_touch.log'}"
            )
        cmd = [
            "snakemake",
            "--snakefile", str(self._root / "Snakefile"),
            "--directory", str(self._root),
            "--cores", "1",
            "--report",
            str(output_path) if format == "html"
            else str(self._root / "analysis_report.html"),
            "--quiet",
        ]
        proc = run_subprocess_with_tee(
            cmd,
            logfile=logs_dir / "regenerate.log",
            cwd=self._root,
            echo_to_stdout=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"snakemake regeneration failed with exit code "
                f"{proc.returncode}. See log: {logs_dir / 'regenerate.log'}"
            )

        # Handle format='zip': snakemake emitted analysis_report.html;
        # bundle it into a zip with any supporting files (per Plan Phase 4
        # `analysis_report.zip` shape — pre-Plan-Phase-4 this just zips
        # the HTML).
        if format == "zip":
            output_path = self._zip_html(
                self._root / "analysis_report.html"
            )
        return output_path

    def _zip_html(self, html_path: Path) -> Path:
        # Bundle the rendered HTML into a single zip at
        # {bundle_root}/analysis_report.zip. Plan Phase 4 may tighten
        # the contents to match its zip-emit determinism contract; this
        # is a minimal implementation for Plan Phase 3.
        import zipfile
        zip_path = self._root / "analysis_report.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(html_path, arcname=html_path.name)
        return zip_path

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
from typing import TYPE_CHECKING, Literal

from hhemt.bundle._emit import (
    _get_toolkit_git_sha,
    emit_bundle,
)

if TYPE_CHECKING:
    from hhemt.eda import EdaReportResult
from hhemt.bundle._combine import CombinedBundle, combine_bundle
from hhemt.subprocess_utils import run_subprocess_with_tee
from hhemt.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
)

__all__ = [
    "Bundle",
    "BundleSchemaError",
    "CombinedBundle",
    "combine_bundle",
    "emit_bundle",
    "_get_toolkit_git_sha",
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_SCHEMA_VERSION",
]


class BundleSchemaError(ValueError):
    """A bundle's bundle_schema_version does not match the locally-installed
    toolkit's BUNDLE_SCHEMA_VERSION. Distinct from generic malformed-manifest
    errors so callers can branch on schema-version mismatch specifically."""


class Bundle:
    """A portable render bundle, ready for local report regeneration."""

    def __init__(
        self,
        root: Path,
        manifest: dict,
        cfg_analysis: analysis_config | None = None,  # noqa: F821 — forward ref
    ) -> None:
        self._root = root.resolve()
        self._manifest = manifest
        self._cfg_analysis = cfg_analysis

    @classmethod
    def from_directory(cls, path: Path | str) -> Bundle:
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
            raise FileNotFoundError(f"No {BUNDLE_MANIFEST_FILENAME} under {root}. Is this a render bundle?")
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
            raise BundleSchemaError(
                f"Bundle {root} has bundle_schema_version="
                f"{bundle_version} which exceeds local "
                f"BUNDLE_SCHEMA_VERSION={BUNDLE_SCHEMA_VERSION}. "
                f"Upgrade the toolkit."
            )
        if bundle_version < BUNDLE_SCHEMA_VERSION:
            raise BundleSchemaError(
                f"Bundle {root} has bundle_schema_version="
                f"{bundle_version} which is below local "
                f"BUNDLE_SCHEMA_VERSION={BUNDLE_SCHEMA_VERSION}. "
                f"Pre-F2 bundles (v1) cannot load under post-F2 toolkit (v2) "
                f"because the bundle layout dropped its legacy peer-file for "
                f"report config and cfg_analysis.yaml gained a required "
                f"`report:` field. Re-emit the bundle from its source "
                f"analysis after running V0005 migration on the source dir."
            )
        invariants = manifest.get("bundle_root_invariants", {})
        if not isinstance(invariants, dict):
            raise ValueError(
                f"bundle_root_invariants in {manifest_path} must be a dict, got {type(invariants).__name__}."
            )
        # Load and Pydantic-validate the bundle's cfg_analysis.yaml at
        # construction time so downstream attribute access
        # (_read_static_backend reading
        # self._cfg_analysis.report.interactive.static_backend) is
        # guaranteed safe by R1's required-field contract.
        from hhemt.config.analysis import analysis_config
        from hhemt.config.loaders import yaml_to_model

        cfg_analysis_path = root / "cfg_analysis.yaml"
        if not cfg_analysis_path.exists():
            raise FileNotFoundError(
                f"Bundle at {root} is missing cfg_analysis.yaml — "
                f"required by R1 (analysis_config.report "
                f"load-time-required)."
            )
        cfg_analysis = yaml_to_model(cfg_analysis_path, analysis_config)
        return cls(root=root, manifest=manifest, cfg_analysis=cfg_analysis)

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

    def eda(self, *, plots_only: bool = True, notebook_filename: str | None = None) -> EdaReportResult:
        """Regenerate the EDA report locally from the bundled data (ADR-10).

        Delegates to the SAME eda/ free functions analysis.eda() uses, passing
        bundle.root as root and the bundled cfg. plots_only=True is the only
        supported mode — a Bundle has no source datatree, so calc cannot run;
        the EDA datasets (eda/<plot_id>.zarr) and rendered plots were carried into
        the bundle by the harvest chain when analysis.eda() ran pre-bundle.
        """
        from hhemt.eda import (
            EdaReportResult,
            render_eda_plots,
        )
        from hhemt.eda._html_export import export_eda_html
        from hhemt.eda._local_surface import emit_eda_local_surface
        from hhemt.eda._notebook import emit_eda_notebook

        if not plots_only:
            raise ValueError(
                "Bundle.eda() supports only plots_only=True — a bundle carries no "
                "source datatree, so the EDA calc stage cannot run."
            )
        eda_cfg = self._cfg_analysis.eda
        emit_eda_local_surface(self._root)
        plot_paths = render_eda_plots(self._root, cfg_analysis=self._cfg_analysis, eda_cfg=eda_cfg)
        notebook_path = emit_eda_notebook(
            self._root,
            cfg_analysis=self._cfg_analysis,
            eda_cfg=eda_cfg,
            is_bundle=True,
            notebook_filename=notebook_filename,
        )
        report_path = export_eda_html(notebook_path, root=self._root)
        return EdaReportResult(
            report_path=report_path,
            notebook_path=notebook_path,
            plot_paths=plot_paths,
            verdicts=[],
        )

    def _read_static_backend(self) -> Literal["matplotlib", "plotly"]:
        # Resolution (post-F2 rev v2): cfg_analysis.report is required by
        # analysis_config Pydantic schema (Phase 1, R1). The 3-step F1
        # resolution order is deleted along with the legacy report-config
        # peer file (Phase 3). Loading the bundle via Bundle.from_directory(...)
        # already validated cfg_analysis.yaml against analysis_config and
        # would have raised ValidationError if `report:` was absent — so
        # the attribute access below is guaranteed safe at this point.
        # Private (leading underscore) but kept on the public Bundle class
        # so test code can monkey-patch for backend-override coverage.
        return self._cfg_analysis.report.interactive.static_backend

    def regenerate_report(self, *, format: Literal["html", "zip"] = "zip") -> Path:
        """Regenerate the analysis report from bundled data.

        Phase 2 wires (a) the regeneration-scoped Snakefile generator
        and (b) the report-templates staging step. Phase 3 wires the
        subprocess invocation, CLI integration, and the
        ``_read_static_backend()`` cfg-read that derives the static
        backend from ``cfg_analysis.yaml``'s
        ``report.interactive.static_backend`` field (default ``"plotly"``
        per Decision 4 / Plan Phase 2 D3).

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
        from hhemt.bundle.snakefile_generator import (
            write_regeneration_snakefile,
        )
        from hhemt.workflow import _emit_report_artifacts

        static_backend = self._read_static_backend()
        from hhemt.config.brand_theme import DEFAULT_BRAND_THEME
        from hhemt.config.loaders import load_brand_theme
        from hhemt.workflow import _brand_theme_css_map

        _bt = self._cfg_analysis.brand_theme if self._cfg_analysis else None
        _theme = load_brand_theme(self._root / _bt) if _bt else DEFAULT_BRAND_THEME
        _emit_report_artifacts(self._root, brand_theme=_brand_theme_css_map(_theme))
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
            "--snakefile",
            str(self._root / "Snakefile"),
            "--directory",
            str(self._root),
            "--cores",
            "1",
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
        # Pass the actual output_path to --report regardless of format.
        # Snakemake auto-detects format from extension: ".html" → single-file
        # HTML; ".zip" → multi-file directory tree zip (small index report.html
        # + sibling data/ folder). The native zip shape matches user
        # expectation of "extracts to a folder with a small html + subfolders
        # with the plotting content."
        cmd = [
            "snakemake",
            "--snakefile",
            str(self._root / "Snakefile"),
            "--directory",
            str(self._root),
            "--cores",
            "1",
            "--report",
            str(output_path),
            "--report-stylesheet",
            str(self._root / "report" / "report.css"),
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

        # Apply React-bundle post-process surgery. ``bundle_mode=True`` drops
        # the bundle-irrelevant chrome (Workflow + Statistics menu items, the
        # empty-after-drops "General" ListHeading). Unconditional surgery
        # steps (initial-view to "metadata", navbar text, category order,
        # placeholder category, showCategory auto-pop, click delegate, About
        # drop, title clear) apply to both formats via either branch.
        from ..report_renderers._react_surgery import (
            apply_post_process_surgery,
            apply_post_process_surgery_to_zip,
        )

        # Navbar upper-left brand text from the bundled theme (D-6/D-9), defaulting
        # to the bundle's analysis_id; None falls back to the historical literal.
        _navbar = _theme.upper_left_text or (self._cfg_analysis.analysis_id if self._cfg_analysis else None)
        try:
            if format == "html":
                output_path.write_text(
                    apply_post_process_surgery(
                        output_path.read_text(),
                        bundle_mode=True,
                        navbar_text=_navbar,
                    )
                )
            else:
                apply_post_process_surgery_to_zip(
                    output_path,
                    bundle_mode=True,
                    navbar_text=_navbar,
                )
        except Exception:
            pass

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

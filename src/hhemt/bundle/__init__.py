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
- ``combine_bundle`` and ``CombinedBundle`` — combine two or more completed,
  unpacked render bundles into one standalone cross-experiment bundle, and the
  consume-side handle for that combined bundle (re-exported from
  ``bundle._combine``).
- ``BundleSchemaError`` — raised when a bundle's ``bundle_schema_version`` does
  not match the installed toolkit's (re-exported from ``hhemt.exceptions``).

``Bundle`` is deliberately NOT a subclass of ``TRITONSWMM_analysis``.
Bundle outputs are pre-computed; ``Analysis.run()`` is not callable
against a bundle, and a subclass would inherit it and every other
simulation-side method as a call that cannot succeed on a bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hhemt.bundle._emit import (
    _get_toolkit_git_sha,
    emit_bundle,
)
from hhemt.exceptions import BundleSchemaError  # re-export (moved to exceptions.py)

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
        """Construct a :class:`Bundle` from an unpacked bundle directory on disk.

        This is the entry point for consuming a render bundle locally — the call
        the how-to guides instruct you to make after transferring a bundle off
        an HPC system.

        Parameters
        ----------
        path : Path or str
            An UNPACKED bundle directory. It must contain
            ``bundle_manifest.json``; if you have a `.zip`, extract it first.

        Returns
        -------
        Bundle
            A handle whose paths all resolve via :attr:`root` at call time —
            no ``os.chdir``, no persisted absolute paths, so the directory is
            relocatable.

        Raises
        ------
        FileNotFoundError
            No ``bundle_manifest.json`` under ``path``, or no ``cfg_analysis.yaml``
            in the bundle.
        ValueError
            The manifest has no ``bundle_schema_version`` field, or its
            ``bundle_root_invariants`` value is not a dict.
        BundleSchemaError
            The bundle's ``bundle_schema_version`` DIFFERS from the locally
            installed toolkit's ``BUNDLE_SCHEMA_VERSION``, in either direction:
            a higher version was emitted by a newer toolkit (upgrade the
            toolkit), and a lower one by an older toolkit (re-emit the bundle
            from its source analysis). The match is exact, so an older bundle
            does not load.
        ConfigurationError
            The bundle's ``cfg_analysis.yaml`` fails the ``analysis_config``
            schema; ``load_bundle_config`` wraps the pydantic validation error.

        Notes
        -----
        A missing ``bundle_root_invariants`` manifest key is read as ``{}`` (no
        enforced invariants), and that default is DEFENSIVE: a bundle at the
        current ``BUNDLE_SCHEMA_VERSION`` always carries the key, because
        ``emit_bundle`` always passes the dict ``_copy_configs_with_relative_paths``
        returns to ``_write_bundle_manifest``, and the exact version match above
        rejects bundles at any other version. Only a manifest built some other
        way reaches the default.

        Examples
        --------
        >>> from hhemt.bundle import Bundle
        >>> b = Bundle.from_directory("path/to/bundle")
        >>> b.regenerate_report()
        """
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
                f"bundles (emitted before the bundle_root_invariants manifest key existed) lack it "
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
                f"An older bundle fails against a newer toolkit in one of exactly "
                f"two ways, and WHICH one depends on the bump. Either the cfg "
                f'CARRIES a key the model has since retired, which extra="forbid" '
                f"rejects; or it LACKS a key that has since become required. "
                f"Additive OPTIONAL fields break nothing -- a cfg missing them "
                f"takes their defaults -- so a version mismatch is not by itself "
                f"evidence that the cfg is wrong. Re-emit the bundle from its "
                f"source analysis with the current toolkit."
            )
        invariants = manifest.get("bundle_root_invariants", {})
        if not isinstance(invariants, dict):
            raise ValueError(
                f"bundle_root_invariants in {manifest_path} must be a dict, got {type(invariants).__name__}."
            )
        # Consume-side toolkit-identity comparison. The producing sha is already in the
        # manifest (emit_bundle writes it as the `toolkit_git_sha` key, through
        # _write_bundle_manifest in bundle/_emit.py); before this, nothing on the
        # consume side read it back, so a re-render months later on a drifted checkout
        # was silent. WARN, never raise: a drifted reader is a caveat on the output, not
        # a reason to refuse a bundle that opens fine -- and the in-repo fixtures carry
        # the literal non-sha sentinel "fixture", which the looks_like_sha guard skips.
        # Compare by PREFIX in BOTH directions: a manifest abbreviation of a different
        # length is the same commit, and a bare != reports that as divergence.
        import warnings

        from hhemt.container_labels import looks_like_sha

        _bundle_sha = manifest.get("toolkit_git_sha")
        _reader_sha = _get_toolkit_git_sha(strict=False)
        # An UNRESOLVABLE reader is a MISMATCH, not a silent pass. The guard used to
        # require BOTH sides to look like a sha before warning, and an unresolvable reader
        # resolves to the sentinel "unknown", which `looks_like_sha` rejects -- so the
        # guard fell silent exactly when the reader could say least about itself, and a
        # bundle naming one commit opened under a toolkit naming none with nothing in the
        # output recording it. The manifest side keeps its `looks_like_sha` gate, which is
        # what keeps the in-repo fixtures' literal "fixture" sentinel silent -- and, with no
        # key at all, what keeps the prefix test below from calling startswith(None).
        _bundle_is_sha = looks_like_sha(_bundle_sha)
        _agrees = (
            _bundle_is_sha
            and looks_like_sha(_reader_sha)
            and (_reader_sha.startswith(_bundle_sha) or _bundle_sha.startswith(_reader_sha))
        )
        if _bundle_is_sha and not _agrees:
            warnings.warn(
                f"Bundle {root} was produced by hhemt {_bundle_sha}, but the toolkit "
                f"reading it is {_reader_sha}. A re-render may differ from the bundled "
                f"figures. This is a provenance caveat, not a failure.",
                stacklevel=2,
            )
        # Load and Pydantic-validate the bundle's cfg_analysis.yaml at
        # construction time so downstream attribute access
        # (_read_static_backend reading
        # self._cfg_analysis.report.interactive.static_backend) is
        # guaranteed safe: `report` is a required field of analysis_config.
        from hhemt.config.analysis import analysis_config

        cfg_analysis_path = root / "cfg_analysis.yaml"
        if not cfg_analysis_path.exists():
            raise FileNotFoundError(
                f"Bundle at {root} is missing cfg_analysis.yaml — "
                f"required because report is a required field of analysis_config."
            )
        # Loaded as the validated MODEL deliberately, not as a dict: this file reads it
        # by attribute (.eda, .report.interactive.static_backend, .brand_theme,
        # .analysis_id) and passes the whole model to render_eda_plots and
        # emit_eda_notebook, so a dict would fail each of those reads with
        # AttributeError. load_bundle_config also rebases the model's Path fields
        # against the bundle root, which is what makes brand_theme -- the one Path
        # field this file opens -- resolvable here.
        from hhemt.bundle._path_policy import load_bundle_config

        cfg_analysis = load_bundle_config(cfg_analysis_path, analysis_config, root)
        return cls(root=root, manifest=manifest, cfg_analysis=cfg_analysis)

    @property
    def root(self) -> Path:
        """Absolute path to the unpacked bundle directory.

        Every bundle-relative path resolves against this at call time, which is
        what makes a bundle relocatable — move the directory and `root` moves
        with it. Use :meth:`absolute` to resolve a relative path against it.
        """
        return self._root

    @property
    def manifest(self) -> dict:
        """The parsed ``bundle_manifest.json``.

        Carries, among other keys, the bundle's ``bundle_schema_version``, the
        harvested file set, the root invariants and the producing
        ``toolkit_git_sha``. The dict is returned BY REFERENCE: mutating it changes
        this ``Bundle``'s in-memory copy, which later reads of this property see,
        and never writes back to disk.
        """
        return self._manifest

    def absolute(self, rel: str | Path) -> Path:
        """Resolve a bundle-relative path to an absolute path under
        ``bundle.root``."""
        return (self._root / rel).resolve()

    def eda(self, *, plots_only: bool = True, notebook_filename: str | None = None) -> EdaReportResult:
        """Regenerate the EDA report locally from the bundled data.

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

    def reprex(self, reprex_config, target_hpc_profile):
        """Validate round-trip runnability against a target HPC profile.

        Verifies the SIF (mandatory sha256 digest match when the crate references
        one, fail-closed; the toolkit does not PGP-sign SIFs and runs no signature
        check), re-aims ``validation.py`` preflight at ``target_hpc_profile``, and
        emits per-``(member_id, column)`` problem pairs plus per-field graduated
        experiment amendments for closest-possible cross-system reproduction. Mirrors ``Bundle.eda()`` —
        delegates to the ``bundle._reprex`` free function with ``bundle.root``.

        Parameters
        ----------
        reprex_config : hhemt.config.reprex_config.reprex_config
            The target user's minimal runnable-field set (account / login node /
            ``sif_root``, the directory their images are found under by digest /
            scratch + target partition selectors).
        target_hpc_profile : hhemt.config.hpc_system.hpc_system_config
            The reproducer's own HPC profile (partition caps, container spec).

        Returns
        -------
        hhemt.bundle._reprex.ReprexResult
        """
        from hhemt.bundle._reprex import reprex as _reprex

        return _reprex(self._root, reprex_config, target_hpc_profile)

    def _read_static_backend(self) -> Literal["matplotlib", "plotly"]:
        # cfg_analysis.report is a required field of the analysis_config
        # Pydantic schema, so there is no fallback resolution order here; an
        # earlier one that also read a separate report-config peer file was
        # removed together with that file. Loading the bundle via
        # Bundle.from_directory(...) already validated cfg_analysis.yaml
        # against analysis_config and would have raised ConfigurationError
        # (load_bundle_config wraps the pydantic validation error) if `report:`
        # was absent — so the attribute access below is guaranteed safe at this
        # point.
        # Private (leading underscore) but kept on the public Bundle class
        # so test code can monkey-patch for backend-override coverage.
        return self._cfg_analysis.report.interactive.static_backend

    _RENDER_OPENED_CFG_FIELDS = ("brand_theme",)
    """cfg_analysis fields the REGENERATE path OPENS, as opposed to reads as a scalar.

    A field belongs here iff regenerate_report (or something it calls) resolves it to a
    file and reads that file. One member today; that is the honest size of the set.
    """

    def _assert_render_inputs_present(self) -> None:
        """Fail closed, and actionably, when the bundle declares a render input it does
        not carry.

        Not a duplicate of experiments.py::_assert_declared_inputs_exist: that gate
        covers the DOI/run path and is not on this one, and it is scoped to the
        carried-input policies rather than to what render opens. Every field named here
        is ``excludable=False`` in the emit-side catalog, so a declared-but-absent value
        can only mean a broken bundle -- there is no by-reference reading to fall back
        on, and defaulting silently would emit a correct-looking report with wrong
        branding and no signal.

        Raises ProcessingError (CLI exit 5) rather than letting ``load_brand_theme``
        raise: that path surfaces a bare FileNotFoundError which, when the bundle sits
        under a temp root, ``_load_config`` decorates with a cross-node-visibility
        explanation that is actively wrong for a local re-render.

        KNOWN LIMITATION, stated rather than implied. Inside ``regenerate_report`` this
        raise leaves no half-written state: it is the third CALL, after
        ``assert_plots_match_running_build`` and ``_read_static_backend``, neither of
        which mutates (the imports between them do not mutate either). That property
        does NOT extend through the CLI verb: ``hhemt report-from-bundle``
        (``cli.py``) unlinks ``analysis_report.html`` and
        ``analysis_report.zip`` BEFORE calling ``regenerate_report``, so a raise here
        leaves those two files deleted. Pre-existing rather than introduced -- that flow
        destroyed the same two files and then raised ``FileNotFoundError`` from
        ``load_brand_theme`` -- and ``bundle_baseline/`` retains the HPC copies. Moving
        this check to ``from_directory`` or to the CLI verb ahead of its unlink is what
        would extend the property; neither is done here. NOTE the ordinal: a future edit
        that inserts a side effect into either of the two preceding calls, or between
        them, silently breaks the in-method half of this property.
        """
        from hhemt.exceptions import ProcessingError

        if self._cfg_analysis is None:
            return
        for name in self._RENDER_OPENED_CFG_FIELDS:
            value = getattr(self._cfg_analysis, name, None)
            if value is None:
                continue
            target = self._root / value
            if not target.exists():
                # Echo the DECLARED, bundle-relative form -- what an operator will
                # actually find in cfg_analysis.yaml. `value` here is the absolute path
                # rebase_bundle_relative_paths produced in memory, and quoting that
                # sends the reader searching the YAML for a string it does not contain.
                try:
                    declared = Path(target).relative_to(self._root)
                except ValueError:
                    declared = Path(value)
                raise ProcessingError(
                    operation="regenerate_report",
                    filepath=target,
                    reason=(
                        f"the bundle's cfg_analysis declares {name}: {declared} but the "
                        f"file is not present under the bundle root. {name} is not an "
                        f"excludable input, so it cannot be a by-reference deposit -- "
                        f"this bundle is incomplete. Re-emit it from the source analysis."
                    ),
                )

    def regenerate_report(self, *, format: Literal["html", "zip"] = "zip", declare_stale_plots: bool = False) -> Path:
        """Regenerate the analysis report from bundled data.

        Stages the report templates, writes a regeneration-scoped Snakefile
        (plot, rule-all and render-report rules only; no simulation,
        processing or consolidation rules), runs ``snakemake --touch`` and
        then ``snakemake --report`` as subprocesses rooted at the bundle, and
        applies the post-render surgery. The static backend comes from
        ``cfg_analysis.yaml``'s ``report.interactive.static_backend`` field
        through ``_read_static_backend()``.

        Parameters
        ----------
        format : {"html", "zip"}
            Final report format; the signature default is ``"zip"``. The
            static backend is NOT a caller-facing parameter; it is derived from
            the bundle's cfg files via ``self._read_static_backend()``, so it is
            set by the bundle's config rather than by an argument default here.
        declare_stale_plots : bool, default False
            Passed to ``assert_plots_match_running_build``. When False, a bundle
            whose figures were rendered by different code than the running
            toolkit raises ``StalePlotsError`` before any regeneration work.
            When True the mismatch is printed to this process's stdout (prefixed
            ``[render_report] DECLARED STALE PLOTS:``) and the regeneration
            proceeds. Nothing else keeps it: this method discards the message
            the guard returns, and the print runs outside the tee'd snakemake
            subprocesses, so it reaches no ``_logs/`` file and no report.

        Raises
        ------
        StalePlotsError
            The bundled figures do not match the running build and
            ``declare_stale_plots`` is False.
        ProcessingError
            From ``_assert_render_inputs_present``: the bundle declares a render
            input (``brand_theme``) that it does not carry. Also from the
            post-render surgery (``report_renderers._react_surgery``), when the
            rendered report carries the toggle machinery but a literal the surgery
            patches has drifted; the surgery is not wrapped here, so that error
            ends the call after ``snakemake --report`` has already written the
            un-surgeried report.
        ConfigurationError
            The bundled ``brand_theme`` file exists but fails the ``brand_theme``
            schema: ``load_brand_theme`` wraps the pydantic validation error.
            ``_assert_render_inputs_present`` checks only that the file is there.
            An empty theme file raises ``ValueError`` from the same loader instead.
        RuntimeError
            Stale Snakemake locks are present under ``.snakemake/locks``, or the
            ``snakemake --touch`` or ``snakemake --report`` subprocess exits
            non-zero.
        FileNotFoundError
            ``format="html"`` and ``snakemake --report`` exited 0 without writing
            ``analysis_report.html``: the html branch reads that file before the
            surgery. The zip branch does not raise on a missing archive -- the zip
            surgery returns without doing anything, and this method still returns the
            ``analysis_report.zip`` path.
        """
        from hhemt.provenance import assert_plots_match_running_build

        # Guard BEFORE any regeneration work: this seam re-renders the REPORT from a
        # bundle whose figures are already fixed, so the figures it will transcribe are
        # exactly the ones on disk now. (Contrast Analysis.render_report(), which may
        # itself rebuild the figures it would be judged against — that seam's placement
        # is a separate question and is deliberately not wired here.)
        assert_plots_match_running_build(self._root, declare_stale_plots=declare_stale_plots)

        from hhemt.bundle.snakefile_generator import (
            write_regeneration_snakefile,
        )
        from hhemt.workflow import _emit_report_artifacts

        static_backend = self._read_static_backend()
        from hhemt.config.brand_theme import DEFAULT_BRAND_THEME
        from hhemt.config.loaders import load_brand_theme
        from hhemt.workflow import _brand_theme_css_map

        # RENDER-PATH INPUT CHECK. load_bundle_config validates the archive record's
        # SHAPE under the `metadata` intent and asserts nothing about files on disk --
        # true for its other two callers, which read scalars only. This path is the
        # exception: it OPENS brand_theme. The requirement lives here, so the check does
        # too, rather than being smuggled into the loader's intent where it would be
        # wrong for everyone else.
        self._assert_render_inputs_present()
        _bt = self._cfg_analysis.brand_theme if self._cfg_analysis else None
        _theme = load_brand_theme(self._root / _bt) if _bt else DEFAULT_BRAND_THEME
        _emit_report_artifacts(self._root, brand_theme=_brand_theme_css_map(_theme))
        write_regeneration_snakefile(self._root, static_backend=static_backend)

        # Defense-in-depth stale-lock check. The CLI's report-from-bundle verb
        # clears stale locks itself before reaching here, after printing a warning
        # that names the locks directory; this check catches lock state for
        # Python-API callers that bypass the CLI (notebook usage).
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

        output_path = self._root / f"analysis_report.{format}"

        # Build snakemake invocation. cwd= is set on the subprocess, not
        # the parent process, so the caller's working directory never
        # changes (no process-global os.chdir).
        # run_subprocess_with_tee is imported at module level so test
        # code can monkeypatch the binding site (bundle.__init__), as
        # tests/test_bundle.py does.
        logs_dir = self._root / "_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Mark all existing plot outputs as up-to-date so --report does
        # not attempt to re-run plot rules against absent source data.
        # The bundle contains rendered plots, NOT the source datatree;
        # without --touch, snakemake's needrun check fires because no
        # .snakemake/metadata/ exists on a fresh local working directory.
        # This relies on Snakemake's documented `--touch` contract: it marks
        # outputs up to date without running their rules.
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
        # empty-after-drops "General" ListHeading), because a regenerated
        # bundle's Workflow and Statistics panels describe only the
        # regeneration DAG. The unconditional steps (title clear, About drop,
        # navbar text, category-sort comparator, toggle and render-guard
        # patches, showCategory auto-pop, click delegate, initial view to
        # "metadata", card-name humanization) apply to both formats via
        # either branch. The comparator ranks categories by the bundled config's
        # reporting-set order, resolved below as ``_category_order``.
        from ..report_renderers._react_surgery import (
            apply_post_process_surgery,
            apply_post_process_surgery_to_zip,
        )

        # Navbar upper-left brand text from the bundled theme, defaulting to the
        # bundle's analysis_id; None makes the surgery use its own fixed text,
        # "H&H Ensemble Modeling Toolkit".
        _navbar = _theme.upper_left_text or (self._cfg_analysis.analysis_id if self._cfg_analysis else None)
        # Resolve member_id card names to derived compute-config labels, from the
        # bundle's own scenario_status.csv. Threaded to BOTH branches so the html
        # and the zip cannot diverge on card names.
        from ..report_plot_ids import event_sidebar_labels_from_status, member_labels_from_status

        _member_labels = member_labels_from_status(self._root)
        _event_labels = event_sidebar_labels_from_status(self._root)
        # Sidebar order: the bundled config's active reporting set, resolved by the same
        # helper the two analysis render paths use, so the re-render ranks its categories
        # as the original render did. Without it the surgery's five-name default moves
        # "Workflow performance" and "Metadata" to the alphabetical tail. A Bundle built
        # without a config (never the case through from_directory) keeps that default.
        from types import SimpleNamespace

        from ..render_category_order import resolve_render_path_category_order

        _category_order = (
            resolve_render_path_category_order(SimpleNamespace(cfg_analysis=self._cfg_analysis))
            if self._cfg_analysis is not None
            else None
        )
        if format == "html":
            output_path.write_text(
                apply_post_process_surgery(
                    output_path.read_text(),
                    bundle_mode=True,
                    navbar_text=_navbar,
                    category_order=_category_order,
                    member_labels=_member_labels,
                    event_labels=_event_labels,
                )
            )
        else:
            apply_post_process_surgery_to_zip(
                output_path,
                bundle_mode=True,
                navbar_text=_navbar,
                category_order=_category_order,
                member_labels=_member_labels,
                event_labels=_event_labels,
            )

        return output_path

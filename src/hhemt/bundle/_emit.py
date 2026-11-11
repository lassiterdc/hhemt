"""Render-bundle emission helpers — the portable artifact for local
renderer iteration.

Public surface:

  emit_bundle(analysis, output_path=None, exclude_config=None, container_defs=None) -> Path

plus the two consume-side helpers reconstitute_runnable_config() and
reconstitute_runnable_analysis_config(). The emit pipeline's private helpers include
_harvest_sources, _rewrite_paths_to_relative, _write_bundle_manifest and
_emit_bundle_zip.

This module is opt-in only. Importing it does not trigger any side
effects. emit_bundle() is called only by deliberate, user-invoked bundling: the
`hhemt bundle` CLI command, Analysis.bundle_report_data() / reprex_bundle() /
publish_reprex_bundle() (and their sensitivity counterparts), and
publishing.publish_analysis(deposit_source="reprex_bundle"). It is NEVER invoked by
Analysis.run() or Analysis.submit_workflow().
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hhemt.bundle._archive import MODERATE, write_zip
from hhemt.bundle._path_policy import (
    _PATH_FIELD_POLICY,
    _SELF_CONTAINED_POLICIES,
    PathPolicy,
    RewriteResult,
    enumerate_path_fields,
    rebase_bundle_relative_paths,
)
from hhemt.bundle._placeholders import (
    SNIFF_BYTES,
    PlaceholderTable,
    build_placeholder_table,
    is_zarr_chunk_member_by_name,
    prefix_is_text,
    rewrite_producer_paths,
)
from hhemt.bundle._reprex_gate import (
    ZARR_ARRAY_METADATA_NAMES,
    ZARR_DECODE,
    ZeroUserInfoScanner,
    classify_zarr_array,
    decode_zarr_array_text,
    has_byte_order_mark,
    scan_binary_member,
)
from hhemt.report_renderers._figure_emission import (
    harvest_metadata_only_sources,
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


from hhemt.exceptions import StaleReadModelError
from hhemt.provenance import producing_stamp
from hhemt.utils import select_regenerable_figures

# The eda exemption is no longer spelled here. It lives once, in
# constants.UNREGENERABLE_ANALYSIS_SUBTREES, and reaches this site through
# utils.select_regenerable_figures -- so this module's former local alias for the
# eda subdirectory constant, and the import that fed it, are retired rather than
# kept beside a second source for one fact.


def _figure_stem(name: str) -> str:
    """Figure identity: the canonical plot ID's leading segment, or the bare stem.

    Canonical plot-ID stems (``report_plot_ids.py``) are
    `{renderer_kind}[__{descriptor}][__member.{id}][__evt.{id}]`, so the part
    before the first `__` identifies the figure family. Keying on this rather than on the
    full path is what lets BOTH extension siblings of a live figure survive: a rule declares
    one `output:` with a literal extension, and matplotlib emits .html AND .svg.
    """
    return name.split("__")[0] if "__" in name else name.rsplit(".", 1)[0]


def _declared_figure_stems(analysis_dir: Path) -> set[str] | None:
    """Figure stems every plot rule in the analysis Snakefile declares it produces.

    Returns None when no Snakefile exists, so the caller SKIPS pruning rather than reading
    "no declarations" as "everything is an orphan" — that collapse would delete every figure.
    """
    snakefile = analysis_dir / "Snakefile"
    if not snakefile.is_file():
        return None
    stems: set[str] = set()
    in_output = False
    for raw in snakefile.read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("output:"):
            in_output = True
            stripped = stripped[len("output:") :].strip()
        elif stripped and not raw.startswith((" ", "\t")):
            in_output = False
        if in_output and stripped:
            for tok in stripped.split(","):
                tok = tok.strip().strip("\"'")
                if tok and not tok.endswith(":"):
                    stems.add(_figure_stem(Path(re.sub(r"\{[^}]*\}", "*", tok)).name))
    return stems


def _prune_undeclared_figures(analysis_dir: Path, plots_dir: Path) -> list[str]:
    """Delete figures whose stem no plot rule declares, with their manifest sidecars.

    Returns the analysis-relative paths removed, for carriage in the bundle manifest. A
    warning would not do: _harvest_sources' leading comment records that a
    warnings.warn inside a Snakemake rule log "is lost by the time anyone reads the bundle",
    which is why declared_sources_absent is threaded into the manifest instead. A prune whose
    audit trail is lost is a silent deletion.

    plots/eda/ is EXEMPT: those figures come from analysis.eda(), a non-Snakemake in-process
    facade, so the Snakefile is not the authority for that subtree and their absence from it
    is not orphanhood. Without the exemption a live EDA figure such as
    eda_cross_hardware_magnitude, which no Snakefile rule declares, would be deleted as an
    orphan.
    """
    declared = _declared_figure_stems(analysis_dir)
    if declared is None or not plots_dir.exists():
        return []
    removed: list[str] = []
    # The walk comes from the ONE selector. The `keep` predicate is this caller's own
    # question -- a figure the Snakefile still declares is not an orphan -- and it stays
    # here rather than moving into the selector, which knows nothing about declarations.
    #
    # DELIBERATE NON-CONSUMPTION: this site does NOT read
    # constants.UNREGENERABLE_ANALYSIS_SUBTREES whole. The selector applies it, and because
    # this walk is rooted at plots/ the only member it can reach is plots/eda -- which is
    # the right subset, since eda_local/ is not a figure and an undeclared-FIGURE prune
    # has no business exempting it. The subset follows from the root, not from a skip list
    # anyone maintains here, which is why the inline exemption this replaced is gone.
    #
    # This prune runs against the LIVE analysis tree (PRUNE-BEFORE-HARVEST), so these are
    # DU-counted `plots/` bytes: route through du_sentinels.delete_and_account so the
    # analysis-scope DU sentinel is decremented by what is deleted.
    _prune_targets = select_regenerable_figures(
        analysis_dir,
        plots_dir,
        keep=lambda p: _figure_stem(p.name) in declared,
        on_select=lambda p: removed.append(str(p.relative_to(analysis_dir))),
    )
    from hhemt.du_sentinels import delete_and_account

    delete_and_account(_prune_targets, scope_dir=analysis_dir, scope="analysis")
    return removed


def _render_sha(plots_dir: Path, length: int = 12) -> str:
    """Short content hash over the RENDERED FIGURE SET under ``plots_dir`` (post-prune).

    Hashes relative path + file bytes for every non-sidecar file. Content-derived, not
    time-derived: two builds at one source pin were measured to produce different bytes,
    so a timestamp does not establish identity.

    HONEST LIMIT, stated rather than implied: this detects a CONTENT change, not a render
    EVENT. A deterministic renderer re-run over unchanged data yields the same hash, and
    that is the correct behaviour for the purpose — the failure being caught is stale
    figures under a fresh stamp, and figures that are byte-identical are not stale. What it
    cannot do is prove that a render was attempted.
    """
    import hashlib

    digest = hashlib.sha256()
    if not plots_dir.exists():
        return "norender"
    for path in sorted(plots_dir.rglob("*")):
        if path.is_dir() or path.name.endswith(".manifest.json"):
            continue
        digest.update(str(path.relative_to(plots_dir)).encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:length]


#: Relative paths of the one producer/transcriber pair that can silently disagree: the
#: Errors-and-Warnings figure TRANSCRIBES the persisted read-model (errors_and_warnings.py
#: reads validation_report.json via load_validation_report and never re-runs
#: validate_analysis at render time, so the figure is only as current as the persisted
#: read-model). Everything else a renderer reads is produced inside the DAG that renders it.
_EW_FIGURE_REL = "plots/errors_and_warnings/validation_report.html"
_VALIDATION_READ_MODEL_REL = "validation_report.json"


def _assert_report_not_older_than_read_model(analysis_dir: Path) -> None:
    """Refuse to ship an Errors-and-Warnings figure that predates its own read-model.

    INVARIANT: when both files exist, mtime(figure) >= mtime(read-model). NOT a strict
    `>`: two files written in the same filesystem timestamp tick are a correct tree, and
    asserting strict-later would encode one satisfying position rather than the invariant.

    Why this can happen at all: analysis.eda() re-persists validation_report.json near its
    end (``persist_validation_report``, inside ``eda()`` itself), and eda() is a
    non-Snakemake in-process facade, so it runs after the report DAG has already exited.
    eda() now re-renders this one figure right after that re-persist when the figure
    already exists; this guard is what catches any path that re-persists without doing
    so. Measured on
    generation e389264af7b9: figure 19:30, read-model 19:40 -- the figure shipped the
    previous generation's read-model, still carrying a check name this branch had retired.

    Absent on either side is a SKIP, not a refusal: no read-model means nothing to be stale
    against, and no figure means it is not in the bundle. The manifest-sidecar precondition
    below already owns the did-a-render-happen-at-all question.

    Runs BEFORE any staging copy, where mtimes are still real -- a bundle zip normalizes
    every timestamp to 1980-01-01, so this comparison is only meaningful on the live tree.
    """
    figure = analysis_dir / _EW_FIGURE_REL
    read_model = analysis_dir / _VALIDATION_READ_MODEL_REL
    if not figure.exists() or not read_model.exists():
        return
    fig_m, rm_m = figure.stat().st_mtime, read_model.stat().st_mtime
    if fig_m < rm_m:
        raise StaleReadModelError(
            f"{_EW_FIGURE_REL} is OLDER than {_VALIDATION_READ_MODEL_REL} by "
            f"{rm_m - fig_m:.0f}s -- the figure transcribes a superseded read-model and "
            "would ship one generation behind. REMEDY: the FIGURE must be rebuilt (its "
            "plot_errors_and_warnings rule re-executed, or the renderer re-invoked "
            "directly as analysis.eda() now does at its tail), then re-emit the bundle. "
            "analysis.render_report() does NOT do this and must not be reached for: it "
            "invokes `snakemake --report`, a post-execution render that re-executes no "
            "rule, so it leaves the figure's mtime untouched and this guard fires again "
            "(measured 2026-08-16 on a clean fixture root -- delta unchanged at 112s). "
            "No other single-call remedy is named here because none has been confirmed."
        )


def _crate_license_finding(analysis, staging: Path) -> str | None:
    """Return a human-readable crate-currency finding for the staged crate, or None.

    DETECTS at every bundle; REFUSES at none. The refusal lives at the publish boundary
    only, because that is the one door where the artifact becomes immutable. The
    asymmetry is not about how much a stale crate matters -- it is about what a wrong
    repair costs. The sibling gate above refuses on staleness against
    validation_report.json, which is MACHINE-GENERATED: no wrong-direction repair exists,
    so refusing there leaves only the correct path out. This check compares against
    dataset_license, which is HAND-AUTHORED config, where the wrong-direction repair is a
    one-line edit that is sometimes legitimately correct, corrupts the system of record
    rather than a derived copy, and is undetectable afterwards because both sides then
    agree. A refusal on this predicate would train that edit; a finding does not.

    Root resolution follows the spec's algorithm -- the metadata descriptor's `about`
    names the Root Data Entity -- rather than matching the literal "./". That is
    DEFENSIVE, not a fix for an observed skip: every crate this system emits today is
    "./"-rooted, because ro-crate-py hard-codes the descriptor's `about` to "./", and
    upgrade_doc_to_workflow_run_crate's own bare next() would raise on any other root
    before this function is reached. The literal match would therefore work today. It
    would work by depending on an upstream implementation detail that is not a guarantee,
    and resolving `about` costs nothing to be independent of it.
    """
    from hhemt.metadata import (
        _SPDX_LICENSE_TABLE,
        UnrecognizedCrateLicense,
        as_license_record,
        read_crate_license,
        resolve_dataset_license,
    )

    crate = Path(staging) / "ro-crate-metadata.json"
    if not crate.exists():
        return None
    graph = json.loads(crate.read_text()).get("@graph", [])
    descriptor = next(
        (e for e in graph if isinstance(e, dict) and str(e.get("@id", "")).endswith("ro-crate-metadata.json")),
        None,
    )
    if descriptor is None:
        return "ro-crate-metadata.json carries no metadata descriptor entity; root not resolvable"
    about = descriptor.get("about")
    root_id = about.get("@id") if isinstance(about, dict) else about
    root = next((e for e in graph if isinstance(e, dict) and e.get("@id") == root_id), None)
    if root is None:
        return f"metadata descriptor points at root {root_id!r}, which is absent from the @graph"
    lic = root.get("license")
    uri = lic.get("@id") if isinstance(lic, dict) else lic
    cfg_license = analysis.cfg_analysis.dataset_license
    # getattr, not a bare read: this is handed config-like objects that may predate the
    # details-file field (test doubles built from a SimpleNamespace), and a currency check
    # must not become an AttributeError on such an object.
    cfg_license_file = getattr(analysis.cfg_analysis, "dataset_license_file", None)
    # An unlicensed config with an unlicensed crate is the normal state of an analysis that
    # has not chosen a license (one is required only to publish), so it is not a finding.
    # Reporting it would put the same line in the manifest of every unpublished bundle.
    if uri is None and cfg_license is None and cfg_license_file is None:
        return None
    # The config side is rendered once, in the words of whichever field declares it: the two
    # fields are alternate sources of one license, so a message naming only dataset_license
    # would tell a details-file user to clear a field they never set.
    _cfg_plain = (
        "no license set"
        if cfg_license is None and cfg_license_file is None
        else (
            f"{cfg_license!r} ({_SPDX_LICENSE_TABLE[cfg_license]['uri']!r})"
            if cfg_license is not None
            else f"{str(cfg_license_file)!r} (dataset_license_file)"
        )
    )
    if uri is None:
        return (
            f"the crate root declares no license, but analysis_config declares "
            f"{_cfg_plain}; the crate predates the config. To write that license into the "
            f"crate, re-emit it with reprocess(start_with='consolidate', regenerate_existing=True) "
            f"before depositing. A plain reprocess does NOT re-emit it -- regenerate_existing "
            f"defaults to False, which deletes no consolidated zarr, so consolidation "
            f"early-returns. Clearing the analysis config's license instead makes the bundle "
            f"agree without a rebuild, but an analysis with no license cannot be published."
        )
    # Compared BY RECORD, through the same recognition rule the publish read-back uses, so
    # the bundle's verdict and publish's verdict cannot disagree about what a crate says. A
    # crate license this toolkit cannot name RAISES out of that rule instead of resolving to
    # a record, and that is not the same fact as a crate that declares none -- the flag
    # below carries the distinction across the record comparison, because both sides
    # normalize to None whenever the config also declares nothing and would then compare
    # equal. The flag is what keeps such a crate REPORTED as a finding rather than silently
    # shipped, and it reaches the `keep == ""` branch below, which exists for a license with
    # no id and no name to offer as a config value.
    unrecognized_crate_license = False
    try:
        crate_declared = read_crate_license(graph)
    except UnrecognizedCrateLicense:
        crate_declared = None
        unrecognized_crate_license = True
    crate_record = as_license_record(crate_declared)
    cfg_record = as_license_record(
        resolve_dataset_license(dataset_license=cfg_license, dataset_license_file=cfg_license_file)
    )
    if unrecognized_crate_license or cfg_record != crate_record:
        # Two remedies, the no-rebuild one first, as the publish refusal states them: keeping the
        # crate's license needs only the config line set back, while changing it needs a
        # re-consolidation. Which config field can express the crate's license follows the
        # crate's FORM -- an SPDX id goes in dataset_license, while a details-file license can
        # only be restored by pointing dataset_license_file at a file declaring the same terms.
        if isinstance(crate_declared, str):
            keep = (
                f"To keep the crate's license, set analysis_config.dataset_license to {crate_declared!r}; no "
                f"re-consolidation is needed. "
            )
        elif crate_declared is not None:
            keep = (
                "To keep the crate's license, set analysis_config.dataset_license_file to a details "
                f"file declaring name {crate_declared.name!r}"
                + (f" and link {crate_declared.link!r}" if crate_declared.link is not None else "")
                + "; no re-consolidation is needed. "
            )
        else:
            # A crate license outside both the table and its own entities has no id and no
            # name to set, so it is named by its URI and only the re-consolidation route is
            # offered.
            keep = ""
        return (
            f"crate root license {uri!r} disagrees with analysis_config "
            f"({_cfg_plain}); the crate predates the config. {keep}To change the license, re-emit "
            f"the crate with reprocess(start_with='consolidate', regenerate_existing=True) before "
            f"depositing. A plain reprocess does NOT re-emit it -- regenerate_existing defaults to "
            f"False, which deletes no consolidated zarr, so consolidation early-returns."
        )
    return None


#: The directory prefix, under ``plots/``, of a sensitivity analysis's per-member figures.
_MEMBER_FIGURE_PREFIX = ("sensitivity", "per_sim")

#: One flag name read by the refusal's fix hint, the override warning and the CLI help.
_ALLOW_STALE_FIGURES_FLAG = "--allow-stale-figures"


def _live_member_ids(obj: Any) -> frozenset[str]:
    """The member ids the experiment's setup table currently lists, as strings.

    ``emit_bundle`` receives one of two shapes. The sensitivity wrapper carries ``df_setup``
    on itself; the sensitivity master (a plain analysis, which is what ``hhemt bundle`` passes)
    carries it at ``.sensitivity.df_setup``. A non-sensitivity analysis carries neither, and
    ``self.sensitivity`` is unset unless the sensitivity toggle is on. No member set (neither
    resolves, or the index is empty) is an empty set: nothing can be stale against it.

    Ids are compared as strings because ``df_setup.index`` is the setup table's ``member_id``
    column, which the toolkit compares as ``astype(str)`` elsewhere.
    """
    df_setup = getattr(obj, "df_setup", None)
    if df_setup is None:
        df_setup = getattr(getattr(obj, "sensitivity", None), "df_setup", None)
    if df_setup is None:
        return frozenset()
    return frozenset(str(i) for i in df_setup.index)


def _stale_member_of(plots_dir: Path, figure: Path, live: frozenset[str]) -> str | None:
    """The member id of ``figure`` when it sits under a member directory NOT in ``live``."""
    parts = figure.relative_to(plots_dir).parts
    if len(parts) < len(_MEMBER_FIGURE_PREFIX) + 2 or parts[: len(_MEMBER_FIGURE_PREFIX)] != _MEMBER_FIGURE_PREFIX:
        return None
    member_dir = parts[len(_MEMBER_FIGURE_PREFIX)]
    if not member_dir.startswith("member-"):
        return None
    member_id = member_dir[len("member-") :]
    return None if member_id in live else member_id


def _check_stale_figures(analysis: Any, analysis_dir: Path, plots_dir: Path, allow_stale_figures: bool) -> list[str]:
    """Refuse a sensitivity analysis holding stale figures, or (with the override) report them.

    A STALE FIGURE is a regenerable figure file under ``plots/sensitivity/per_sim/member-{N}/``
    whose ``N`` is not in the live member set (``_live_member_ids``). Selection goes through
    ``select_regenerable_figures``, the toolkit's one walk for figure eligibility, with a
    ``keep`` predicate; that selector DELETES NOTHING and this check hands its result to no
    deleter, so the analysis tree is byte-identical whatever this returns or raises. A figure
    whose family no plot rule declares (``_declared_figure_stems``, the prune's own test) is the
    separate ``_prune_undeclared_figures`` case: the prune deletes it next, so it is not counted,
    warned about or listed here, even under a member directory.

    Returns the stale figures' analysis-relative paths (empty when none). Raises
    ``ConfigurationError(field="bundle_stale_figures")`` when there are some and
    ``allow_stale_figures`` is False. With the override it issues ONE warning naming the count
    and returns the paths, which ``emit_bundle`` records as ``stale_figures_included``.
    """
    live = _live_member_ids(analysis)
    if not live:
        return []
    # A stale member figure whose family no plot rule declares is the prune's case, not this
    # check's: `_prune_undeclared_figures` deletes it next, so it never reaches the bundle and must
    # not be counted, warned about or listed. The test is the prune's own (`_declared_figure_stems`
    # and `_figure_stem`), so the two cannot disagree. No Snakefile (None) means the prune skips, so
    # every stale figure survives into the bundle and is counted.
    declared = _declared_figure_stems(analysis_dir)
    stale: list[Path] = []
    by_member: dict[str, int] = {}

    def _record(figure: Path) -> None:
        member_id = _stale_member_of(plots_dir, figure, live)
        if member_id is not None:
            stale.append(figure)
            by_member[member_id] = by_member.get(member_id, 0) + 1

    select_regenerable_figures(
        analysis_dir,
        plots_dir,
        keep=lambda p: (
            _stale_member_of(plots_dir, p, live) is None
            or (declared is not None and _figure_stem(p.name) not in declared)
        ),
        on_select=_record,
    )
    if not stale:
        return []
    relative = [str(p.relative_to(analysis_dir)) for p in stale]
    if allow_stale_figures:
        _warn_stale_figures_included(len(relative))
        return relative

    from hhemt.exceptions import ConfigurationError

    member_ids = sorted(by_member, key=lambda m: (not m.isdigit(), int(m) if m.isdigit() else 0, m))
    setup_file = getattr(getattr(analysis, "cfg_analysis", None), "sensitivity_analysis", None)
    per_member = "\n".join(
        f"    plots/sensitivity/per_sim/member-{m}/   ({by_member[m]} figure files)" for m in member_ids
    )
    raise ConfigurationError(
        field="bundle_stale_figures",
        message=(
            f"This sensitivity analysis holds {len(relative)} stale figure file(s) for "
            f"{len(member_ids)} member(s) that are no longer in the setup table above: "
            f"member(s) {', '.join(member_ids)}. They are under:\n{per_member}\n"
            "  A stale figure shows a member this experiment no longer contains, and the data it was "
            "drawn from may be gone (the harvest warning names any declared source that is absent). "
            "Bundling it would put figures with no data behind them into the bundle and into any "
            "deposit made from it. Nothing was written or deleted."
        ),
        config_path=setup_file,
        fix_hint=(
            "re-render the figures for the current setup table with `hhemt run` (or `hhemt run-experiment`) "
            'and --override-force-rerun \'{"subject": "all", "stage": "render"}\', which deletes and '
            "rebuilds every regenerable figure; or put the member(s) back in the setup table. To bundle "
            f"anyway, re-run `hhemt bundle` with {_ALLOW_STALE_FIGURES_FLAG}, which bundles the stale "
            "figures as they are and lists them under stale_figures_included in bundle_manifest.json; "
            "from Python, pass allow_stale_figures=True to emit_bundle. `hhemt reprex emit` and "
            "`hhemt publish --reprex-bundle` take no such flag: clear the stale figures first."
        ),
    )


def _warn_stale_figures_included(count: int) -> None:
    """ONE warning naming how many stale figures the override carried (same form as W1)."""
    warnings.warn_explicit(
        message=(
            f"Bundle: {count} stale figure file(s) included by {_ALLOW_STALE_FIGURES_FLAG}; see stale_figures_included"
        ),
        category=UserWarning,
        filename=_HARVEST_WARNING_LABEL,
        lineno=0,
        module="hhemt.bundle._emit",
    )


def emit_bundle(
    analysis: TRITONSWMM_analysis,
    output_path: Path | None = None,
    exclude_config: Path | BundleExcludeConfig | None = None,
    container_defs: list[Path] | None = None,  # RETIRED: any non-empty value raises
    #   ConfigurationError below; kept in the signature only so callers get that message.
    omit_output_zarrs: bool = False,
    allow_stale_figures: bool = False,
) -> Path:
    """Emit a portable render bundle from a completed HPC analysis.

    The bundle file set is the union of source paths each renderer declared
    during the most recent render, namely the ``source_paths`` the renderer passed to
    ``emit_plot_with_sources`` plus any ``prov.artist().add_channel(...)``
    refs, harvested from *.manifest.json sidecars under
    {analysis_dir}/plots/ and {analysis_dir}/eda/, together with every
    cfg-declared input not excluded. Configs are rewritten to relative paths.
    The HPC-baseline analysis_report.html is preserved under bundle_baseline/. The
    baseline zip is not carried.

    Independent of any figure, the bundle also carries IN FULL ``analysis_datatree.zarr``, whose ``timeseries``
    branches hold the consolidated series, and every per-scenario time-series store that exists.
    The per-scenario stores are the ``ScenarioPaths`` fields named ``output_*_timeseries`` and
    ``output_*_time_series``. ``bundle_manifest.json`` names every carried store under
    ``carried_output_stores``.

    Every generated and every harvested text member of the archive has the
    producer's directories, interpreter, login name, and SLURM account replaced by
    the closed ``{hhemt_*}`` tokens of ``_placeholders`` as the member is written into the
    zip. Declared inputs, meaning the files ``_copy_declared_inputs`` carries, and binary
    members are copied through byte for byte.

    Parameters
    ----------
    exclude_config : Path or BundleExcludeConfig, or None
        Path to an operator-authored exclude-config YAML, or an already-validated
        ``BundleExcludeConfig``. When given, the named inputs are not carried. Each
        emits an ``input_deposit`` by-reference block into ``bundle_manifest.json`` and
        a URL-bearing ``File`` part into the crate instead. Omit the argument, the default, and
        the bundle is self-contained, and every config-declared input is carried.
    omit_output_zarrs : bool
        When True, every harvested ``.zarr`` store is carried SHALLOWLY, with zarr metadata
        documents only and no chunk files, and named in BOTH ``metadata_only_stores`` and
        ``omitted_output_zarrs`` in ``bundle_manifest.json``. ``combine`` refuses such a
        bundle, because ``combine`` refuses every ``metadata_only_stores`` entry, and the bundle's
        regeneration Snakefile refuses a plot rule that declared an omitted store in full.
        The default is False, and the emit is unchanged. The time-series stores of the analysis follow the flag
        too. Each ``.zarr`` among the stores is carried shallow and named in both keys.
    allow_stale_figures : bool
        A sensitivity analysis holding figures of members absent from the setup table is
        REFUSED by default, with ``ConfigurationError`` on field ``bundle_stale_figures``, before
        anything is deleted or staged. When True the emit proceeds, carries the stale figures as
        they are, issues a warning naming their count, and records their analysis-relative
        paths under ``stale_figures_included`` in ``bundle_manifest.json``. The emit deletes
        no file from the analysis tree on account of a stale figure.
    """
    # ARGUMENT validation, so it precedes every read of the tree: a caller passing a retired
    # parameter is told before a single stat.
    if container_defs:
        from hhemt.exceptions import ConfigurationError

        raise ConfigurationError(
            field="container_defs",
            message=(
                "container_defs was RETIRED (SIF quest, 2026-09): recipes are package data under "
                "hhemt/sif/recipes and a bundle carries the resolved images' .manifest.json set instead."
            ),
            config_path=None,
        )
    analysis_dir = analysis.analysis_paths.analysis_dir
    plots_dir = analysis_dir / BUNDLE_PLOTS_SUBDIR
    # Shipping-time staleness gate. Placed before the harvest so a stale tree is refused
    # rather than packed; mtimes are real here and normalized to 1980-01-01 inside the zip.
    _assert_report_not_older_than_read_model(analysis_dir)

    if not plots_dir.exists() or not list(plots_dir.rglob("*.manifest.json")):
        raise FileNotFoundError(
            f"No *.manifest.json sidecars found under {plots_dir}. "
            f"Bundle emission requires a completed render_report(). "
            f"Run analysis.render_report() on HPC first."
        )

    # Stale-figure refusal. Placed with the other shipping-time refusals, AFTER the
    # manifest-presence check and BEFORE the container-manifest step and the prune below: the
    # prune DELETES from the live analysis tree, so a refusal must precede it for the tree to
    # be untouched when the emit stops. Selection is read-only.
    stale_figures_included = _check_stale_figures(analysis, analysis_dir, plots_dir, allow_stale_figures)

    # A container-mode bundle carries the PRODUCER MANIFEST of the image the
    # SIMULATIONS RAN IN -- the CARRIED identity source from_doi consumes. Recipes are package
    # data, so no .def and no source tree are carried. Native bundles compute nothing here and
    # their manifest is byte-identical to before.
    #
    # PLACED HERE, ABOVE THE PRUNE, and the prune is the reason rather than the staging walk:
    # _prune_undeclared_figures DELETES orphan figures and their sidecars from the LIVE analysis
    # tree before anything is staged. Wasted staging I/O is recoverable; that deletion is not.
    # This gate reads no staged byte -- only the configs, the system log and sif_root -- so it
    # belongs beside the staleness gate above, with the other shipping-time refusals.
    sif_manifests = None
    if analysis.cfg_analysis.execution_environment == "container":
        sif_manifests = _emit_sif_manifests(analysis)

    # PRUNE BEFORE HARVEST. The harvest globs plots/**/*.manifest.json and
    # _copy_supporting_files copytrees the whole plots_dir, so a figure whose rule no longer
    # exists is collected into the manifest AND carried in the payload. Deleting figure +
    # sidecar from disk first fixes both populations. No filter on sources_by_renderer: its
    # values are the renderers' INPUT paths keyed by figure stem, disjoint from the outputs.
    pruned_orphan_figures = _prune_undeclared_figures(analysis_dir, plots_dir)
    sources_by_renderer = harvest_source_paths(plots_dir, analysis_dir)
    # SECOND HARVEST ROOT (bundle completeness). The EDA calc members write their own
    # manifest sidecars under {analysis_dir}/eda/ via emit_data_artifact_with_sources, and
    # those sidecars declare the RAW inputs each calc actually read. Nothing walked eda/
    # before this, so those inputs were never carried and eda/{plot_id}.zarr shipped as a
    # "derived" dataset nothing in the bundle could derive.
    #
    # harvest_source_paths is root-agnostic: its only position-specific branch re-roots
    # against members/member_{N} when the manifest sits under plots/sensitivity/per_sim/
    # member-{N}/, and it keys on the path's FIRST part being "sensitivity", which an
    # eda/ root never is. If a future derived/ reshape nests per-member artifacts under
    # this root, that branch becomes reachable here and would re-root paths that were
    # relativized against the primary -- re-check it then.
    #
    # Keys are NAMESPACED because an eda/ sidecar's stem equals its plots/eda/ figure
    # stem, so a plain dict merge would clobber one with the other. The map is written to
    # the bundle manifest and read by nothing, so the key form is free.
    for _stem, _paths in harvest_source_paths(analysis_dir / "eda", analysis_dir).items():
        sources_by_renderer[f"eda::{_stem}"] = _paths
    git_sha = _get_toolkit_git_sha()
    analysis_id = analysis.cfg_analysis.analysis_id

    if output_path is None:
        # `{toolkit_sha}` alone was truthful about the code and SILENT about the figures,
        # which is how three re-bundle rounds shipped the same stale plots under three fresh
        # names. render_sha hashes the RENDERED FIGURES (post-prune), not the harvested
        # inputs — hashing inputs cannot separate "re-rendered" from "re-zipped", since a
        # re-render from unchanged inputs leaves every input byte identical.
        render_sha = _render_sha(plots_dir)
        output_path = (
            analysis_dir / BUNDLE_OUTPUT_SUBDIR / f"{analysis_id}_{git_sha}_{render_sha}_v{BUNDLE_SCHEMA_VERSION}.zip"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Built once per emit, before anything is staged, from the analysis being emitted. It reads
    # nothing from the tree, so it is placed with the other per-emit inputs rather than inside the
    # staging block it is applied from.
    placeholders = build_placeholder_table(analysis)
    # The zero-user-info check is built here, beside the table and for the same reason, and BEFORE
    # anything is staged: a configured private list that cannot be used refuses the emit now rather
    # than after the archive has been written.
    scanner = ZeroUserInfoScanner.for_emit(analysis)

    # Function-local import: config/bundle_exclude.py imports hhemt.bundle._path_policy
    # (a leaf), and reprex_taxonomy.py documents the invariant that no module reachable
    # from hhemt.bundle.__init__ may import hhemt.config.* at module scope. Same discipline
    # as the function-local `all_field_bucket` import in _scrub_user_bucket_fields.
    from hhemt.config.bundle_exclude import BundleExcludeConfig

    if isinstance(exclude_config, Path | str):
        import yaml

        exclude_config = BundleExcludeConfig.model_validate(yaml.safe_load(Path(exclude_config).read_text()) or {})

    with _staging_dir(output_path.parent) as staging:
        # PLAN, never copy: harvested sources are byte-identical to their originals and
        # the zip normalizes every timestamp and mode, so materializing them under
        # render_bundle/ (the same small-file-create-bound mount as the analysis) bought
        # nothing. `_emit_bundle_zip` streams them from `streamed` at the end; only the
        # GENERATED files below (configs, crate, runnable set, manifest, baseline, plots/)
        # land in `staging`. The shallow set is computed over BOTH harvest roots at once
        # because a full declaration under one root must defeat a qualifier under the
        # other (harvest_metadata_only_sources).
        metadata_only = harvest_metadata_only_sources((plots_dir, analysis_dir / "eda"), analysis_dir)
        declared_sources_unresolved_outside_tree: list[str] = []
        streamed, declared_sources_absent, metadata_only_stores = _harvest_sources(
            sources_by_renderer,
            analysis_dir,
            metadata_only=metadata_only,
            omit_output_zarrs=omit_output_zarrs,
            unresolved_outside=declared_sources_unresolved_outside_tree,
        )
        # THE TIME-SERIES STORES, carried because they exist and not because a figure declares them. Planned
        # into the same `streamed` the harvest filled, so a store a figure also declares is planned once.
        carried_output_stores, carried_shallow_stores = _plan_time_series_carriage(
            analysis, analysis_dir, streamed, omit_output_zarrs=omit_output_zarrs
        )
        if omit_output_zarrs:
            metadata_only_stores = sorted(set(metadata_only_stores) | set(carried_shallow_stores))
        else:
            # A figure's qualifier may have named a carried store shallow, but it is carried in full here, and the
            # manifest must not call a full store metadata-only (`combine` refuses every name in that list).
            metadata_only_stores = [store for store in metadata_only_stores if store not in set(carried_output_stores)]
        # The omitted set is the shallow `.zarr` stores of THIS harvest under the flag: every
        # harvested `.zarr` source is shallow under it, so the filter selects exactly those.
        omitted_output_zarrs = (
            [store for store in metadata_only_stores if store.endswith(".zarr")] if omit_output_zarrs else None
        )
        _copy_bundle_baseline(analysis_dir, staging)
        _copy_reference_outputs(analysis, staging)
        aggregated_invariants, config_outputs = emit_relative_config_pair(analysis, staging)
        # THE NAMESPACE REFUSAL, before _copy_declared_inputs writes anything. The plan is
        # made here and again inside _copy_declared_inputs (planning copies nothing and
        # _copy_declared_inputs keeps its signature), so what is checked is what is written.
        planned_inputs, _excluded = _plan_declared_inputs(analysis, exclude_config)
        _assert_archive_namespace(config_outputs, planned_inputs, streamed)
        _assert_no_emit_name_clobber(analysis, config_outputs, planned_inputs)
        _emit_resolved_brand_theme(analysis, staging)
        _copy_supporting_files(analysis, staging)
        _emit_hpc_identity(analysis, staging)
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
            sif_manifests=sif_manifests,
            declared_sources_absent=declared_sources_absent,
            pruned_orphan_figures=pruned_orphan_figures,
            crate_license_finding=_crate_license_finding(analysis, staging),
            metadata_only_stores=metadata_only_stores,
            analysis_dir=analysis_dir,
            omitted_output_zarrs=omitted_output_zarrs,
            carried_output_stores=carried_output_stores,
            stale_figures_included=stale_figures_included,
            declared_sources_unresolved_outside_tree=declared_sources_unresolved_outside_tree,
            zero_user_info=scanner.manifest_record(),
        )
        # The declared inputs are the one member class the table never rewrites: leaving their
        # bytes alone keeps every recorded input digest true. Taken from the plan made above, so
        # what is exempt is exactly what `_copy_declared_inputs` wrote.
        declared_arcnames = frozenset(name for name, _src in _declared_destinations(planned_inputs))
        # The archive is written under a temporary name and renamed into place only after the
        # zero-user-info check has passed on every text member, so a refusal leaves nothing at
        # `output_path` that was written by this emit. The `finally` also removes the temporary
        # file when the zip writer fails for any other reason.
        partial_archive = output_path.with_name(f".{output_path.name}.partial")
        try:
            _emit_bundle_zip(
                staging,
                partial_archive,
                streamed=streamed,
                placeholders=placeholders,
                declared_arcnames=declared_arcnames,
                scanner=scanner,
            )
            os.replace(partial_archive, output_path)
        finally:
            # EXEMPT-DU: transient-intermediate
            partial_archive.unlink(missing_ok=True)

    return output_path


def _source_rel(resolved: Path, analysis_root: Path, *, name: str | None = None) -> Path:
    """Where a declared source lives in the bundle, relative to the master analysis directory.

    ``resolved`` under ``analysis_root`` keeps its position (so a sensitivity member's source
    reads ``members/member_{N}/...``); anything outside becomes ``external/{filename}``.
    ONE definition, called by ``_harvest_sources`` to name the archive entry and by
    ``_write_bundle_manifest`` to spell the manifest's source paths, so the manifest names
    exactly the entry the archive holds. ``name`` overrides the fallback filename: the
    harvest passes the UNRESOLVED source's name so a symlinked source keeps the archive name
    it has always had.
    """
    try:
        return resolved.relative_to(analysis_root)
    except ValueError:
        return Path("external") / (name if name is not None else resolved.name)


def _hold_identical_bytes(paths: list[Path]) -> bool:
    """Whether every path is a regular file and all of them hold the same bytes.

    Two claimants of one archive name at DIFFERENT resolved paths are still one file when their
    content is identical: the archive keeps one copy and loses nothing. Sizes are compared first,
    so a pair of different sizes never reads a byte; the sha256 (``_sha256_file``, streaming) is
    taken only when the sizes agree. The callers invoke this only for a name that already has
    more than one resolved claimant, so a full-carry store of hundreds of thousands of entries
    claimed once costs nothing.

    Returns False, never raises, for a claimant that is a directory or that cannot be read when
    compared, whether it vanished since the plan was made or is unreadable. The callers turn
    False into their ``ConfigurationError`` refusal, so a caller of ``emit_bundle`` sees the same exception
    type as before and never a bare ``OSError``.

    Deliberately NOT a recursive content comparison of directory claimants: a directory has no
    single digest, and the members of two directories already collide per file at their own
    names, where this function is called on each member.
    """
    try:
        if not all(p.is_file() for p in paths):
            return False
        if len({p.stat().st_size for p in paths}) != 1:
            return False
        return len({_sha256_file(p) for p in paths}) == 1
    except OSError:
        return False


def _plan_harvest_entry(entries: dict[str, Path], arcname: str, source: Path) -> None:
    """Record ``arcname -> source``, refusing when a DIFFERENT file already holds the name.

    Replaces ``entries.setdefault``, which kept the first source and silently discarded the
    second: after that call the loser is gone from the plan, so no check over the RETURNED
    plan can see it, and the archive dropped a declared file without a word. The same file
    reached twice (two renderers, or a symlink) is one file and is planned once, and so is a
    byte-identical file at another path, per ``_hold_identical_bytes``: the plan keeps the first.
    """
    held = entries.get(arcname)
    if held is None:
        entries[arcname] = source
        return
    if held.resolve() != source.resolve() and not _hold_identical_bytes([held, source]):
        from hhemt.exceptions import ConfigurationError

        raise ConfigurationError(
            field="source_paths_by_renderer",
            message=(
                f"two different files are declared under the one archive name {arcname!r}: "
                f"{held.resolve()} and {source.resolve()}. The bundle can hold only one of them there, "
                "and the other would be dropped silently."
            ),
            fix_hint="rename or relocate one of the two declared sources so their archive names differ",
        )


def _harvest_sources(
    sources_by_renderer: dict[str, list[Path]],
    analysis_dir: Path,
    *,
    metadata_only: frozenset[Path] = frozenset(),
    omit_output_zarrs: bool = False,
    unresolved_outside: list[str] | None = None,
) -> tuple[dict[str, Path], list[str], list[str]]:
    """PLAN the archive entries for every declared source path -- copy nothing.

    Returns ``(entries, declared_absent, metadata_only_stores)``. ``entries`` maps each
    archive name (POSIX, relative to the bundle root, preserving the source's position
    under ``analysis_dir``; ``external/{name}`` for a source outside it) to the ON-DISK
    file ``_emit_bundle_zip`` streams from. Each resolved path is walked ONCE however
    many figures declare it: the retired copy loop re-``copytree``'d a directory once
    per declaring figure, and ``harvest_source_paths`` dedupes only within a key.
    ``omit_output_zarrs`` forces every directory source whose name ends ``.zarr`` into the
    shallow set whether or not a sidecar qualified it (DR-3, C1(e)); False, the default,
    leaves the shallow set exactly ``metadata_only``.
    ``unresolved_outside`` is a list the caller owns. When it is given, a declared source that
    is ABSENT and whose resolved path lies OUTSIDE ``analysis_dir`` is appended to it as
    ``os.path.relpath(resolved, analysis_root)`` (the sidecar's own ``../`` spelling, which
    carries no producer root) and is NOT put in ``declared_absent``. A tree that was moved or
    copied to another depth resolves such a ``../`` path to nowhere, and recording it as a
    missing ``external/{name}`` file would contradict the declared-input copy, which carries a
    file under that archive name. Entries the list already held are kept; it leaves the call
    de-duplicated and sorted (``sorted(set(...))``). The sources THIS harvest found are
    announced by a second aggregated warning
    (``_warn_declared_sources_unresolved_outside_tree``) issued immediately after the in-tree
    one, and only when this harvest found any. When it is None the function behaves as it did
    before this keyword existed.
    A directory whose resolved path is in ``metadata_only`` contributes only its zarr
    metadata documents (``_zarr_metadata_filenames``); its bundle-root-relative path is
    returned in ``metadata_only_stores`` so the manifest can name it. Measured on the
    delivered 3,798-event campaign: the consolidated tree alone was 326,747 chunk files
    against a few hundred metadata documents.

    A declared source that does not exist on disk is SKIPPED (with ONE aggregated
    warning issued after the loop, ``_warn_declared_sources_absent``), not fatal.
    Renderers are allowed to declare an expected source unconditionally,
    even when it is legitimately absent — e.g. ``disk_utilization`` declares
    ``_status/_du.json`` and renders a "re-run processing to populate" placeholder
    when the sentinel is missing (a normal state for a sensitivity primary, whose
    consolidate path does not write the analysis-scope DU sentinel). Hard-raising
    on such a source made ``bundle_report_data()`` crash on every sensitivity
    primary. The skip keeps the emit side consistent with the declare side; the
    warning preserves auditability (a source that vanished after render shows up
    here rather than failing silently)."""
    # A declared source that is absent stays NON-FATAL (see above), but the evidence must
    # survive the run. Accumulate the skipped relative paths and return them so the caller can
    # record them in the bundle manifest — the manifest that made a false claim then also
    # carries the record of it. Without this the only trace is a warnings.warn on stderr
    # inside a Snakemake rule log, which is lost by the time anyone reads the bundle.
    declared_absent: list[str] = []
    outside_unresolved: list[str] = []
    entries: dict[str, Path] = {}
    shallow_stores: list[str] = []
    seen: set[Path] = set()
    analysis_root = analysis_dir.resolve()
    metadata_names = _zarr_metadata_filenames() if (metadata_only or omit_output_zarrs) else frozenset()
    for paths in sources_by_renderer.values():
        for src in paths:
            resolved = src.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rel = _source_rel(resolved, analysis_root, name=src.name)
            if not src.exists():
                if unresolved_outside is not None and not resolved.is_relative_to(analysis_root):
                    outside_unresolved.append(os.path.relpath(resolved, analysis_root))
                else:
                    declared_absent.append(str(rel))
                continue
            if src.is_dir():
                shallow = resolved in metadata_only or (omit_output_zarrs and src.name.endswith(".zarr"))
                if shallow:
                    shallow_stores.append(rel.as_posix())
                for member in sorted(src.rglob("*")):
                    if not member.is_file() or (shallow and member.name not in metadata_names):
                        continue
                    _plan_harvest_entry(entries, (rel / member.relative_to(src)).as_posix(), member)
            else:
                _plan_harvest_entry(entries, rel.as_posix(), src)
                for sidecar in shapefile_sidecars(src):
                    _plan_harvest_entry(entries, rel.with_suffix(sidecar.suffix).as_posix(), sidecar)
    if declared_absent:
        _warn_declared_sources_absent(sorted(set(declared_absent)))
    if unresolved_outside is not None:
        found = sorted(set(outside_unresolved))
        unresolved_outside.extend(found)
        unresolved_outside[:] = sorted(set(unresolved_outside))
        if found:
            _warn_declared_sources_unresolved_outside_tree(found)
    return entries, sorted(set(declared_absent)), sorted(shallow_stores)


#: The ``ScenarioPaths`` fields that name a per-scenario OUTPUT time-series store: ``output_`` prefix, then a name
#: ending ``_timeseries`` or ``_time_series``. The prefix matters: ``weather_timeseries`` and ``hyg_timeseries`` end the
#: same way and are INPUTS (the weather record and the hydrograph), which a bundle carries as declared inputs.
_SCENARIO_STORE_PREFIX = "output_"
_SCENARIO_STORE_SUFFIXES = ("_timeseries", "_time_series")


def _scenario_time_series_fields() -> tuple[str, ...]:
    """The names of the ``ScenarioPaths`` fields that hold a per-scenario time-series store, read from the class.

    Derived from the dataclass rather than listed, so a time-series store a later chunk adds to ``ScenarioPaths``
    under the same naming is carried without an edit here. Nine today (``output_tritonswmm_performance_timeseries``,
    ``output_tritonswmm_triton_timeseries``, ``output_tritonswmm_link_time_series``,
    ``output_tritonswmm_node_time_series``, ``output_triton_only_performance_timeseries``,
    ``output_triton_only_timeseries``, ``output_swmm_only_link_time_series``, ``output_swmm_only_node_time_series``,
    ``output_swmm_only_system_time_series``).
    """
    import dataclasses

    from hhemt.paths import ScenarioPaths

    return tuple(
        field.name
        for field in dataclasses.fields(ScenarioPaths)
        if field.name.startswith(_SCENARIO_STORE_PREFIX) and field.name.endswith(_SCENARIO_STORE_SUFFIXES)
    )


def _iter_scenario_paths(analysis: Any):
    """Yield the ``ScenarioPaths`` of every scenario of ``analysis`` whose simulation folder exists.

    The folder is named from the event's slug first, and a scenario object is built only for a folder that is already
    there, so an emit never creates a simulation folder or its disk-usage sentinel (the scenario constructor does both
    for a new folder). The object is a ``TRITONSWMM_scenario`` whose ``_create_directories`` does nothing: that is the
    one thing its constructor does that writes inside an EXISTING folder (it makes the scenario's empty working
    directories when they are missing), and an emit must neither create anything in the tree it packs nor need write
    access to it. The paths still come from the one place that defines them, ``scenario.py``. A sensitivity PRIMARY
    yields nothing: its scenarios belong to its members (see ``_plan_time_series_carriage`` for what that leaves
    uncarried). An object with no ``df_sims`` (the bare namespaces some tests pass) yields nothing.
    """
    cfg = getattr(analysis, "cfg_analysis", None)
    if cfg is not None and getattr(cfg, "toggle_sensitivity_analysis", False):
        return
    df_sims = getattr(analysis, "df_sims", None)
    if df_sims is None:
        return
    from hhemt.scenario import TRITONSWMM_scenario, compute_event_id_slug

    class _PathsOnlyScenario(TRITONSWMM_scenario):
        def _create_directories(self) -> None:
            """Create nothing: the emit reads the tree it packs."""

    simulations = Path(analysis.analysis_paths.simulation_directory)
    for event_iloc in df_sims.index:
        slug = compute_event_id_slug(analysis._retrieve_weather_indexer_using_integer_index(event_iloc))
        if (simulations / slug).is_dir():
            yield _PathsOnlyScenario(event_iloc, analysis).scen_paths


def _store_rel(src: Path, analysis_dir: Path, resolved: Path, analysis_root: Path) -> Path:
    """The archive-relative name of a carried store: its position under ``analysis_dir`` AS THE ANALYSIS NAMES IT.

    Taken from the unresolved path, because a store whose directory is a symlink (``sims`` moved to another volume
    and linked back) resolves OUTSIDE the analysis tree, where ``_source_rel`` would name it
    ``external/{file name}`` and the same store name of two scenarios would then claim one archive name. A store
    that is not lexically under ``analysis_dir`` takes the resolved spelling, as every harvested source does.
    """
    try:
        return Path(os.path.abspath(src)).relative_to(Path(os.path.abspath(analysis_dir)))
    except ValueError:
        return _source_rel(resolved, analysis_root, name=src.name)


def _plan_time_series_carriage(
    analysis: Any,
    analysis_dir: Path,
    entries: dict[str, Path],
    *,
    omit_output_zarrs: bool = False,
) -> tuple[list[str], list[str]]:
    """PLAN the archive entries of every time-series store of the analysis into ``entries``, independent of any figure.

    Carried: ``analysis_paths.analysis_datatree_zarr``, whose ``timeseries`` branches hold the consolidated series,
    and, for each scenario, every store named by
    ``_scenario_time_series_fields``. A store that is None (a disabled family) or absent on disk (a cleared scenario
    store) contributes nothing; when the scenario stores are cleared the consolidated store is the whole of the
    carriage, which is what clearing means. Each entry is named at its analysis-relative path through
    ``_plan_harvest_entry``, the planner the harvest uses, so a store a figure also declares, or two scenarios
    reaching one file, is planned once and two different files under one name are refused.

    Under ``omit_output_zarrs`` a ``.zarr`` store is planned SHALLOWLY (metadata documents only); a NetCDF file
    cannot be shallow and is planned whole. Returns ``(carried_output_stores, shallow_stores)``: every carried
    store and the shallow subset, as POSIX paths relative to the analysis directory, each sorted.

    What this does NOT carry, stated because it is a gap and not a choice: on a sensitivity analysis
    ``analysis_datatree_zarr`` is unbound on the primary and ``_iter_scenario_paths`` yields nothing, so the
    primary's ``experiment_datatree.zarr`` and every member's ``analysis_datatree.zarr`` and per-scenario stores
    are not planned here.
    """
    candidates: list[Path] = []
    consolidated = getattr(analysis.analysis_paths, "analysis_datatree_zarr", None)
    if consolidated is not None:
        candidates.append(Path(consolidated))
    fields = _scenario_time_series_fields()
    for scen_paths in _iter_scenario_paths(analysis):
        for name in fields:
            store = getattr(scen_paths, name, None)
            if store is not None:
                candidates.append(Path(store))
    analysis_root = analysis_dir.resolve()
    metadata_names = _zarr_metadata_filenames() if omit_output_zarrs else frozenset()
    carried: list[str] = []
    shallow_stores: list[str] = []
    seen: set[Path] = set()
    for src in candidates:
        resolved = src.resolve()
        if resolved in seen or not src.exists():
            continue
        seen.add(resolved)
        rel = _store_rel(src, analysis_dir, resolved, analysis_root)
        carried.append(rel.as_posix())
        if not src.is_dir():
            _plan_harvest_entry(entries, rel.as_posix(), src)
            continue
        shallow = omit_output_zarrs and src.name.endswith(".zarr")
        if shallow:
            shallow_stores.append(rel.as_posix())
        for member in sorted(src.rglob("*")):
            if not member.is_file() or (shallow and member.name not in metadata_names):
                continue
            _plan_harvest_entry(entries, (rel / member.relative_to(src)).as_posix(), member)
    return sorted(carried), sorted(shallow_stores)


#: The label ``_warn_declared_sources_absent`` prints where a warning prints a source file.
#: ONE name read by the issuer and asserted on by its test (F-D W1 / DR-4).
_HARVEST_WARNING_LABEL = "hhemt bundle emit"


def _warn_declared_sources_absent(absent: list[str]) -> None:
    """Issue ONE warning for every declared-but-absent source of a harvest (F-D W1, DR-4).

    ``warnings.warn_explicit`` with ``filename=_HARVEST_WARNING_LABEL`` and ``lineno=0``
    prints ``hhemt bundle emit:0: UserWarning: ...`` and NO source line, on every entry
    point. ``warnings.warn(..., stacklevel=n)`` cannot: whichever frame it attributes to
    is echoed (a Typer dispatcher's ``return callback(**use_params)`` on the CLI, a
    tuple-unpacking line in a toolkit caller otherwise), and that unrelated line is the
    defect this replaces. ``warn_explicit`` still raises a ``UserWarning`` under ``-W
    error`` and is still seen by ``pytest.warns``/``catch_warnings``. No ``registry`` is
    passed, so each harvest shows its warning rather than the once-per-location default
    deduplicating a loop over sweep members.
    """
    listing = ", ".join(absent)
    warnings.warn_explicit(
        message=(
            f"Bundle harvest: {len(absent)} declared source(s) do not exist on disk and "
            f"were skipped: {listing}. They are recorded under the manifest key "
            f"'declared_sources_absent'. A renderer may declare a source that is "
            f"legitimately absent (ADR-6 D3, e.g. _status/_du.json on a sensitivity "
            f"primary); whether each path above is legitimate is decided by its "
            f"declaring renderer's absent branch. A source that was expected to be "
            f"present may have been removed after render_report()."
        ),
        category=UserWarning,
        filename=_HARVEST_WARNING_LABEL,
        lineno=0,
        module="hhemt.bundle._emit",
    )


def _warn_declared_sources_unresolved_outside_tree(unresolved: list[str]) -> None:
    """Issue ONE warning for every out-of-tree declared source that did not resolve.

    The sibling of ``_warn_declared_sources_absent`` in the same ``warn_explicit`` form
    (``filename=_HARVEST_WARNING_LABEL``, ``lineno=0``, no ``registry``), for the other set:
    a source recorded as a ``../``-relative path that lands nowhere from THIS tree's location.
    Each of the two warnings names only its own set.
    """
    listing = ", ".join(unresolved)
    warnings.warn_explicit(
        message=(
            f"Bundle harvest: {len(unresolved)} declared source(s) outside the analysis directory "
            f"did not resolve from this tree's location ({listing}); see "
            f"declared_sources_unresolved_outside_tree. The tree may have been moved, or copied "
            f"to another depth, since its figures were rendered."
        ),
        category=UserWarning,
        filename=_HARVEST_WARNING_LABEL,
        lineno=0,
        module="hhemt.bundle._emit",
    )


def _zarr_metadata_filenames() -> frozenset[str]:
    """The filenames that are zarr METADATA documents, sourced from zarr's own constants.

    Not a hand list: a v3 store's sole document is ``zarr.json`` (the consolidated trees
    are v3 -- V0022), a v2 store's are ``.zarray``/``.zgroup``/``.zattrs`` plus the
    consolidated ``.zmetadata``; a list written from memory of one format carries
    nothing of the other. Function-local import so a zarr rename breaks ``hhemt
    bundle`` loudly here rather than every toolkit import at a module top. Lives
    bundle-side, not in a renderer: the renderer says only "metadata-only"; which
    files that means is the harvester's knowledge.
    """
    from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON

    return frozenset({ZARR_JSON, ZARRAY_JSON, ZGROUP_JSON, ZATTRS_JSON, ZMETADATA_V2_JSON})


def _copy_bundle_baseline(analysis_dir: Path, staging: Path) -> None:
    baseline = staging / BUNDLE_BASELINE_SUBDIR
    baseline.mkdir(parents=True, exist_ok=True)
    # The html page only. The zip is a binary member that a text rewrite cannot reach and that
    # the zero-user-info scan cannot read (its entries are deflated), and nothing in the toolkit
    # opens it; the html page is the one baseline the bundle's own tooling compares against.
    src = analysis_dir / "analysis_report.html"
    if src.exists():
        shutil.copy2(src, baseline / src.name)


#: The consolidated tree a ``test()`` group holds, whose ``summary`` branches are the reference when it is present.
_REFERENCE_TREE_NAME = "analysis_datatree.zarr"


def _copy_summary_branches(tree: Path, dest_tree: Path) -> None:
    """Copy every ``summary`` group of the consolidated tree ``tree`` into ``dest_tree``, chunk files included.

    A zarr version 3 store opens only through the group documents of every directory above a group, so each ancestor
    document is written beside the copied groups. The root document carries no attributes and no consolidated index,
    and an intermediate document carries no consolidated index either: the index lists the ``timeseries`` groups
    this copy leaves out, and a reader that followed it would look for them.
    """
    dest_tree.mkdir(parents=True, exist_ok=True)
    for summary in sorted(tree.glob("*/*/summary")):
        if not (summary / "zarr.json").is_file():
            continue
        rel = summary.relative_to(tree)
        for depth in range(len(rel.parts)):
            source_doc = tree.joinpath(*rel.parts[:depth], "zarr.json")
            target_doc = dest_tree.joinpath(*rel.parts[:depth], "zarr.json")
            if target_doc.exists():
                continue
            document = json.loads(source_doc.read_text(encoding="utf-8"))
            document.pop("consolidated_metadata", None)
            if depth == 0:
                document["attributes"] = {}
            target_doc.parent.mkdir(parents=True, exist_ok=True)
            target_doc.write_text(json.dumps(document), encoding="utf-8")
        # copyfile for the reason given in ``_copy_reference_outputs``.
        shutil.copytree(summary, dest_tree / rel, dirs_exist_ok=True, copy_function=shutil.copyfile)


def _copy_reference_outputs(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    """Carry the producer's test()-shaped physics summaries into
    ``reference_outputs/`` so a reproducer's ``analysis.test()`` output can be checked
    for per-arch within-family EXACT equality.

    Sourced from ``{analysis_dir}/_test/group_*`` — the SAME ``n_reporting_timesteps`` truncation the reproducer's
    test() applies, so the two are apples-to-apples (a FULL-matrix reference would not be: test() truncates
    the synth event to 13 of 181 weather frames, and the tracked vars are peaks). Physics summaries only (never the
    timing perf-timeseries or raw ``*_tseries.*``).

    A group whose consolidated ``analysis_datatree.zarr`` is present is carried as that tree's ``summary`` groups
    under ``reference_outputs/group_i/analysis_datatree.zarr``, because ``test()`` consolidates under the user's
    clearing switch and with the switch on the per-scenario summary stores no longer exist. A group with no such
    tree (one whose ``test()`` ran before the tree held summaries) is carried as its per-scenario
    ``sims/{event_id}/processed/*_summary.*`` stores. A group is never carried in both forms, which would store
    the same values twice.

    CONDITIONAL on ``{analysis_dir}/_test/`` existing: a bundle emitted without a prior
    producer test() carries no ``reference_outputs/`` and stays byte-identical to a bundle
    emitted before this carriage existed (the manifest is deliberately NOT extended — the
    subtree is discoverable by path). The
    subtree is group-indexed (``group_0``, ``group_1``, ...) so the reproducer maps its own
    ``_test/group_i`` to ``reference_outputs/group_i`` by the shared, deterministic group
    basename."""
    test_root = analysis.analysis_paths.analysis_dir / "_test"
    if not test_root.is_dir():
        return
    out_type = analysis.cfg_analysis.target_processed_output_type
    ref_root = staging / "reference_outputs"
    for group_dir in sorted(test_root.glob("group_*")):
        tree = group_dir / _REFERENCE_TREE_NAME
        if tree.is_dir():
            _copy_summary_branches(tree, ref_root / group_dir.name / _REFERENCE_TREE_NAME)
            continue
        for processed in sorted(group_dir.glob("sims/*/processed")):
            rel = processed.relative_to(test_root)  # group_i/sims/{event_id}/processed
            for summ in sorted(processed.glob(f"*_summary.{out_type}")):
                dest = ref_root / rel / summ.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                # copyfile, not copy2: the bundle zip stamps every entry with a fixed date and
                # mode, so the mtime and mode copy2 would carry are overwritten, and each
                # copystat is extra metadata calls per file on the small-file create path.
                if summ.is_dir():
                    shutil.copytree(summ, dest, dirs_exist_ok=True, copy_function=shutil.copyfile)
                else:
                    shutil.copyfile(summ, dest)


#: The two config files every deposit that carries configs writes, as ``(cfg attribute,
#: filename)`` pairs. ONE tuple, read by the writer (``emit_relative_config_pair``), by the
#: staged-names table below and by the data deposit's caller, so no site retypes either
#: filename and a rename lands in one place.
_CONFIG_PAIR: tuple[tuple[str, str], ...] = (
    ("cfg_system", "cfg_system.yaml"),
    ("cfg_analysis", "cfg_analysis.yaml"),
)


def _declared_preimage(value: Any, analysis_root: Path) -> Path:
    """The RESOLVED FILE a config path value names: the identity every step here shares.

    An absolute value resolves as written; a value already relative resolves against the
    analysis directory, exactly as ``_plan_declared_inputs`` has always located a declared
    input (``analysis_root / src`` then ``.resolve()``). One definition, because the refusal
    over archive names, the copy step and the rewrite must agree on what "one file" means:
    if they disagreed, a pair of spellings could be one file to the copy and two to the
    check, or the reverse, and the check would refuse a bundle that loses nothing or pass
    one that drops a file.
    """
    src = Path(value)
    if not src.is_absolute():
        src = analysis_root / src
    return src.resolve()


def _config_string_preimages(
    original: dict, rewritten: dict, cfg_model: type[cfgBaseModel], analysis_dir: Path
) -> list[tuple[str, Path]]:
    """Pair every REWRITTEN path string a config carries with the file it was written for.

    Only the fields whose policy carries an input into the bundle
    (``_SELF_CONTAINED_POLICIES``) are paired: the forced-dot and nulled fields rewrite to
    ``.`` or null whatever they held, so pairing them would report the analysis and system
    directories as two files claiming the name ``.``. A list field pairs element by element,
    because the rewrite maps each element to one output string.
    """
    analysis_root = analysis_dir.resolve()
    pairs: list[tuple[str, Path]] = []
    for name in enumerate_path_fields(cfg_model):
        if _PATH_FIELD_POLICY.get(name) not in _SELF_CONTAINED_POLICIES:
            continue
        before, after = original.get(name), rewritten.get(name)
        if before is None:
            continue
        befores = before if isinstance(before, list) else [before]
        afters = after if isinstance(after, list) else [after]
        for b, a in zip(befores, afters, strict=True):
            if b is None or not isinstance(a, str):
                continue
            pairs.append((a, _declared_preimage(b, analysis_root)))
    return pairs


def emit_relative_config_pair(
    analysis: TRITONSWMM_analysis, staging: Path
) -> tuple[dict[str, dict], dict[str, set[Path]]]:
    """Write ``cfg_system.yaml`` and ``cfg_analysis.yaml`` into ``staging`` with every
    Pydantic ``Path``-typed field rewritten per ``_path_policy._PATH_FIELD_POLICY``.

    The ONE writer of the relative-path config form. The render bundle calls it, and so does
    the data deposit (``publishing._deposit_set``), so the two deposits cannot drift in how
    they spell a path. The destination parameter is NAMED ``staging``, not ``dest`` or
    ``out``: the totality pass over this module's staged names keys on that spelling, and a
    renamed parameter would make this writer invisible to it.

    Returns ``(invariants, outputs)``:

    - ``invariants`` is keyed by cfg attribute (``cfg_system`` / ``cfg_analysis``) and holds
      each ``RewriteResult.invariants``; ``_write_bundle_manifest`` writes it as the
      manifest's ``bundle_root_invariants`` key.
    - ``outputs`` maps each REWRITTEN input path string to the set of files that produced it
      (see ``_declared_preimage``). Two different files under one string is a collision: the
      string names one place in the bundle and only one of the two files can be there.
    """
    import yaml

    analysis_dir = analysis.analysis_paths.analysis_dir
    system_directory = analysis._system.cfg_system.system_directory
    aggregated: dict[str, dict] = {}
    outputs: dict[str, set[Path]] = {}
    for cfg_attr, filename in _CONFIG_PAIR:
        cfg = analysis._system.cfg_system if cfg_attr == "cfg_system" else analysis.cfg_analysis
        cfg_dict = cfg.model_dump(mode="json")
        source_note = None
        if cfg_attr == "cfg_analysis":
            # The report section is the settings the report was rendered with, read from the
            # render's record in report_manifest.json; with no record the section stays the analysis
            # config's section and the comment line at the head of the file says so.
            from hhemt.provenance import read_report_settings_record

            record = read_report_settings_record(analysis_dir)
            if record is not None:
                cfg_dict["report"] = record["report_settings"]
                source_note = f"report settings: render record, source {record['report_settings_source']}"
            else:
                source_note = "report settings: analysis config, no render record"
        result = _rewrite_paths_to_relative(
            cfg_dict,
            cfg_model=type(cfg),
            analysis_dir=analysis_dir,
            system_directory=system_directory,
        )
        # `result.invariants` is captured per-cfg and written into the bundle manifest
        # as bundle_root_invariants (see _write_bundle_manifest).
        aggregated[cfg_attr] = result.invariants
        for out, preimage in _config_string_preimages(cfg_dict, result.cfg_dict, type(cfg), analysis_dir):
            outputs.setdefault(out, set()).add(preimage)
        scrubbed = _scrub_user_bucket_fields(result.cfg_dict, cfg_model=type(cfg))
        dumped = yaml.safe_dump(scrubbed, sort_keys=False)
        # yaml.safe_dump emits no comment, so the source line is prefixed to the dumped text.
        (staging / filename).write_text(f"# {source_note}\n{dumped}" if source_note else dumped)
    return aggregated, outputs


def config_string_collisions(outputs: dict[str, set[Path]]) -> list[tuple[str, list[Path]]]:
    """The rewritten strings that name more than one DIFFERENT file, each with its files sorted.

    Two fields naming the same file are one file and not a collision: the set holds one
    member. Only distinct resolved files under one string are reported.
    """
    return sorted((out, sorted(files)) for out, files in outputs.items() if len(files) > 1)


def format_collisions(collisions: list[tuple[str, list[Path]]]) -> str:
    """Render collisions naming BOTH (every) full resolved path, one line per colliding name."""
    return "; ".join(f"{name!r} is claimed by {' and by '.join(str(f) for f in files)}" for name, files in collisions)


def _scrub_user_bucket_fields(cfg_dict: dict, *, cfg_model: type[cfgBaseModel]) -> dict:
    """Null every non-Path config field whose reprex bucket is ``"user"``, so the
    bundled configs carry no host-local, user-identifying value.

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
        ``bundle_manifest.json`` ``bundle_root_invariants`` key.
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
        # Reserved for future runtime-derived fields. Passed through
        # unchanged.
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
    """Rewrite a path string to its bundle-relative form.

    Resolution order: ``analysis_root`` first, then ``system_root``, then
    ``external/{filename}`` fallback. Mirrors the ``_harvest_sources`` fallback at the
    archive-entry layer so the cfg and the archive layout agree on where the file lives.
    Non-string values are passed through unchanged.

    A value that is ALREADY relative is rewritten the way an absolute one is: it is first
    resolved to the file it names (``_declared_preimage``, against ``analysis_root``) and
    then placed by the same three rules. Passing it through untouched left
    ``../../../o1/w.nc`` in a deposit whose copy step puts that file at ``external/w.nc``,
    so the config and the archive disagreed about where the input lives.
    """
    if not isinstance(value, str):
        return value
    try:
        pr = _declared_preimage(value, analysis_root)
        if pr.is_relative_to(analysis_root):
            return str(pr.relative_to(analysis_root))
        if pr.is_relative_to(system_root):
            return str(pr.relative_to(system_root))
        return str(Path("external") / pr.name)
    except (ValueError, OSError):
        return value


def _emit_resolved_brand_theme(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    """Serialize the RESOLVED brand theme into the bundle and repoint the
    bundled cfg_analysis.yaml::brand_theme at it, so regenerate_report reproduces
    the deploying lab's colors instead of an unresolvable HPC-side path.

    Sources the RESOLVED theme (self._brand_theme, set by run() from the
    explicit-override -> cfg-field -> default ladder) so a run(override_brand_theme=)
    override survives into the bundle. getattr-fallback to the cfg field/default
    covers a fresh-instance bundle emit that never had run() called (the same
    fallback render_report uses for its fresh-instance render path).
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
    cfg_text = cfg_path.read_text()
    cfg_dict = yaml.safe_load(cfg_text)
    cfg_dict["brand_theme"] = "brand_theme.resolved.yaml"
    # yaml.safe_dump drops comments; keep the source line emit_relative_config_pair wrote at the head.
    first_line = cfg_text.split("\n", 1)[0]
    source_line = f"{first_line}\n" if first_line.startswith("# report settings:") else ""
    cfg_path.write_text(source_line + yaml.safe_dump(cfg_dict, sort_keys=False))


def _copy_supporting_files(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    analysis_dir = analysis.analysis_paths.analysis_dir
    # Snakefile is renamed to Snakefile.source so the bundle's regen-only
    # generated Snakefile (written by bundle/snakefile_generator.py at consume
    # time) can take the canonical "Snakefile" filename. The .source variant is
    # preserved for its debugging value.
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
    # The crate is carried SCRUBBED, not copied. The producer's own analysis_dir sidecar
    # legitimately holds full provenance; the BUNDLE copy must not, because it is the
    # artifact that gets deposited under a DOI. Measured on a real Norfolk emit: 213
    # entities carry `agent` -> {"@id": "#agent-{producer-host}"} and 213 carry
    # `startTime`, one distinct value each -- the largest single environment-sourced
    # population in the bundle, and one that no CONFIG-derived detector can reach: its
    # origin is socket.gethostname(), not any config field. Reaching it requires the
    # PERSISTED run record, NOT a live call at emit time -- provenance.py builds this
    # `agent` from `alog.workflow_submission_node.get()`, a LogField set once at
    # SUBMISSION time on the SUBMITTING host (workflow.py). emit_bundle runs wherever
    # `hhemt bundle` was invoked, typically a login node, so a live gethostname() here
    # would name a host that appears nowhere in the artifacts while missing every host
    # that does.
    #
    # The `agent` refs are additionally DANGLING (no #agent-* entity exists in the
    # @graph), so stripping them removes an unresolvable reference rather than
    # breaking a resolvable one.
    # `mainEntity` -- which experiments.py's from_doi ingest and bundle/_reprex.py both
    # require -- is not a volatile key and is untouched.
    #
    # NOT done via metadata._VOLATILE_GRAPH_KEYS: that set feeds _strip_volatile, whose
    # ONLY consumer is the equality test inside write_rocrate_sidecar's compare-and-write.
    # It governs which keys are ignored when deciding to SKIP a rewrite, never what is
    # written -- so populating it would leave this leak untouched AND make a sidecar that
    # differs only in agent/startTime compare equal, retaining a previous run's file.
    #
    # WHY THIS TIER HAS ITS OWN SET. _BUNDLE_TRANSFER_STRIP_KEYS is byte-equal to
    # metadata._VOLATILE_PROV_KEYS and to _COMBINED_EMIT_STRIP_KEYS today; that is timing,
    # not a shared rule, and the sets are not interchangeable. This tier TRANSPORTS an
    # existing crate rather than emitting one: the staging document below is a wholesale
    # copy of the analysis sidecar, and no root entity is constructed anywhere in this
    # module. An operation whose purpose is to move a description intact must not remove a
    # property the RO-Crate spec makes a MUST on the Root Data Entity -- `datePublished` --
    # so this set is the one that has to shrink, and it must shrink WITHOUT moving the
    # combined tier, which emits a first publication of its own and owes an authored value
    # instead. NOT YET DONE: _BUNDLE_TRANSFER_STRIP_KEYS still contains `datePublished`
    # today, so the bundled crate's root currently lacks it; removing it from this set is
    # pending.
    crate_src = analysis_dir / "ro-crate-metadata.json"
    if crate_src.exists():
        from hhemt.metadata import _BUNDLE_TRANSFER_STRIP_KEYS, canonical_jsonld_from_doc

        crate_doc = json.loads(crate_src.read_text())
        for entity in crate_doc.get("@graph", []):
            for volatile_key in _BUNDLE_TRANSFER_STRIP_KEYS:
                entity.pop(volatile_key, None)
        (staging / "ro-crate-metadata.json").write_text(canonical_jsonld_from_doc(crate_doc))
    # case.yaml — the case manifest carries case_name (BLOCKING experiment-identity
    # field, already in _compatibility._EXPERIMENT_IDENTITY_FIELDS). Resolved from the
    # case_manifest_yaml constructor arg; None on paths that do not set it.
    # BundleableAnalysis contract attr — direct access (not getattr-None) fails loud on a non-conforming input.
    case_yaml = analysis.case_manifest_yaml
    if case_yaml is not None and Path(case_yaml).exists():
        shutil.copy2(Path(case_yaml), staging / "case.yaml")
    # NOTE: the weather-events CSV (cfg_analysis.weather_events_to_simulate) is no
    # longer copied here. It is a BUNDLE_RELATIVE cfg-declared input, so the
    # self-contained harvest (_copy_declared_inputs) carries it to EXACTLY the
    # location _rewrite_paths_to_relative writes into the cfg (external/{name} when
    # it lives outside analysis_dir). The old ad-hoc copy targeted the bundle ROOT
    # (Path(src.name)) for out-of-tree CSVs, which disagreed with the cfg rewrite's
    # external/{name} — reconciled by routing every declared input through the one
    # policy-driven copy path.
    # This site carries ONLY the DU sentinel, never the _status/ tree: the one consumer
    # of _du.json is report_renderers/disk_utilization.py, which resolves
    # {analysis_dir}/_status/_du.json against the BUNDLE ROOT at regenerate time (and
    # cross_experiment_disk_utilization.py reads the same file per child crate). The bare
    # `.flag` completion markers are never carried -- bundle/snakefile_generator
    # ._status_flags_for returns () on that basis. The `*.flag.json` payload sidecars and
    # `_status/_job_index.json` DO reach a bundle, but through the manifest HARVEST, not
    # this copy: report_renderers/workflow_performance.py reads them at regenerate time
    # and therefore declares every one it opens, and _harvest_sources streams
    # exactly the declared set -- 22,794 sidecars on a 3,798-event campaign. Their
    # payloads carry `written_at` and `slurm_job_id` (status_flags.write_status_flag);
    # replacing that per-file declaration with a scrubbed read-model is a routed
    # follow-up, not something this copy controls.
    status_dir = analysis_dir / BUNDLE_STATUS_SUBDIR
    du_sentinel = status_dir / "_du.json"
    if du_sentinel.exists():
        (staging / BUNDLE_STATUS_SUBDIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(du_sentinel, staging / BUNDLE_STATUS_SUBDIR / "_du.json")
    plots_dir = analysis_dir / BUNDLE_PLOTS_SUBDIR
    if plots_dir.exists():
        # copyfile for the same reason as the reference-output copy: the zip overwrites
        # every stamp, so copy2's per-file copystat is cost with no effect on the bundle.
        shutil.copytree(plots_dir, staging / BUNDLE_PLOTS_SUBDIR, dirs_exist_ok=True, copy_function=shutil.copyfile)


#: Bundle-root filename for the scrubbed compute-config identity surface (combine-side).
HPC_IDENTITY_FILENAME = "hpc_system_config.identity.yaml"


def _emit_hpc_identity(analysis: TRITONSWMM_analysis, staging: Path) -> None:
    """Emit the SCRUBBED compute-config identity surface for the combine-side checker.

    Allow-list-BY-CONSTRUCTION: reach ONLY for the named compute-config-identity
    fields (``partitions`` map + ``gpu_allocation_flavor``) — the fields two experiments
    must agree on for an intercomparison to be meaningful — and NEVER emit any
    USER-bucket scalar (default_account / login_node / container.sif_root). No
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


def _sha256_file(path: Path) -> str:
    """Streaming 1 MiB-chunk sha256 of a file.

    Streaming bounds memory on multi-GB inputs. The digest equals a whole-file sha256
    whatever the chunk size, so it agrees with the ingest-side
    ``_fetch_file_by_url(expected_sha256=...)`` verification, which reads in the same
    1 MiB chunks.
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Reader sidecars a shapefile needs or uses, matched case-insensitively. ``.shx`` and ``.dbf`` are required for a
#: read; ``.prj`` and ``.cpg`` carry the CRS and the text encoding. Spatial-index files (``.sbn``, ``.sbx``,
#: ``.qix``) are not carried because nothing reads them, and ``.shp.xml`` (and any other ``.xml``) is never carried
#: because ArcGIS metadata can hold author and contact fields.
SHAPEFILE_REQUIRED_SIDECARS = (".shx", ".dbf")
SHAPEFILE_OPTIONAL_SIDECARS = (".prj", ".cpg")


def _sidecar_of(shp: Path, extension: str) -> Path | None:
    """The sidecar of ``shp`` with ``extension`` (any letter case), or None. Same stem, same directory."""
    parent = Path(shp).parent
    if not parent.is_dir():
        return None
    wanted = Path(shp).stem.lower() + extension
    for candidate in sorted(parent.iterdir()):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate
    return None


def shapefile_missing_required_sidecars(shp: Path) -> list[str]:
    """The required sidecar extensions (``.shx``, ``.dbf``) ``shp`` lacks; empty for a non-shapefile path."""
    if Path(shp).suffix.lower() != ".shp":
        return []
    return [e for e in SHAPEFILE_REQUIRED_SIDECARS if _sidecar_of(shp, e) is None]


def shapefile_sidecars(path: Path) -> list[Path]:
    """The sidecar files an input ``.shp`` must be carried with, beside it under the same stem.

    The ONE helper every site that turns an input path into an archive member goes through, so a site cannot
    carry a ``.shp`` alone. A path that is not a ``.shp`` has none. A ``.shp`` missing its ``.shx`` or ``.dbf`` is a
    producer error and raises ``ConfigurationError`` naming the file: a bundle carrying it could not be read.
    """
    from hhemt.exceptions import ConfigurationError

    path = Path(path)
    if path.suffix.lower() != ".shp":
        return []
    missing = shapefile_missing_required_sidecars(path)
    if missing:
        raise ConfigurationError(
            field="declared_input",
            message=(
                f"shapefile {path} has no {' or '.join(missing)} beside it, so it cannot be read; "
                "the bundle would carry an unreadable input"
            ),
            fix_hint="Restore the missing sidecar files next to the .shp, then emit again.",
        )
    found = [_sidecar_of(path, e) for e in (*SHAPEFILE_REQUIRED_SIDECARS, *SHAPEFILE_OPTIONAL_SIDECARS)]
    return [f for f in found if f is not None]


@dataclass(frozen=True)
class _PlannedInput:
    """One declared input the emit will carry: where it is, and where it lands in the bundle."""

    src: Path
    rel: str
    is_dir: bool


def _plan_declared_inputs(
    analysis: TRITONSWMM_analysis,
    exclude_config: BundleExcludeConfig | None = None,
) -> tuple[list[_PlannedInput], list[dict]]:
    """PLAN every cfg-declared input -- copy nothing.

    Driven by the same ``_PATH_FIELD_POLICY`` table the rewrite consumes (one
    policy-driven path -- never a second source list, so ``test_all_path_fields_have_policy``'s
    bidirectional guard already covers it). Returns ``(planned, deposits)``: ``planned`` is
    each input that will be CARRIED, with its destination relative to the bundle root, and
    ``deposits`` is the ``input_deposit`` block of each EXCLUDED input.

    Split out of ``_copy_declared_inputs`` so the destinations are known BEFORE anything is
    written: ``emit_bundle`` compares them against the harvest plan, the rewritten config
    strings and the emit's own staged names, and refuses a collision while the staging folder
    still holds nothing the refusal would have to unwind.

    The governed opt-out. When ``exclude_config`` names a field, its input(s) are not
    planned for carriage; an ``input_deposit`` block is returned for each. The block is
    FILE-level: ``{relpath, sha256, accessed, citation, contentUrl?, url?, identifier?}``.
    The toolkit computes ``relpath`` (via the SAME ``_rewrite_absolute_to_relative`` the cfg
    rewrite uses, so the block's path is byte-identical to where the reconstituted cfg
    looks), ``sha256`` and ``accessed``; the operator supplies the rest.

    Raises:
        ConfigurationError: if an excluded input has no coordinate block, or resolves to a
            DIRECTORY (sha256 is undefined over a tree, so the by-reference record could not
            be integrity-pinned). Both are fail-closed: the toolkit MUST NOT emit a bundle
            that neither carries nor properly references a declared input -- that defect
            would surface only after a DOI had been minted.
    """
    from hhemt.exceptions import ConfigurationError

    analysis_dir = analysis.analysis_paths.analysis_dir
    analysis_root = analysis_dir.resolve()
    system_root = analysis._system.cfg_system.system_directory.resolve()
    accessed = datetime.now(UTC).date().isoformat()
    planned: list[_PlannedInput] = []
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
                src = _declared_preimage(elem, analysis_root)
                if not src.exists():
                    continue  # declared-but-absent: ingest fail-closed handles it
                rel = _rewrite_absolute_to_relative(str(src), analysis_root=analysis_root, system_root=system_root)
                if not isinstance(rel, str) or Path(rel).is_absolute():
                    continue  # unresolvable dest; leave for the ingest fail-closed gate

                if excluded:
                    ref = exclude_config.refs_for(name, src.name)
                    if ref is None:
                        raise ConfigurationError(
                            field=f"bundle_exclude_config.exclusions.{name}",
                            message=(
                                f"'{name}' is excluded but carries no coordinates for "
                                f"'{src.name}'. Every excluded input needs at least a "
                                f"'citation'; add a 'contentUrl' too if a consumer can "
                                f"download it directly. Without a block the bundle would "
                                f"neither carry nor reference this input."
                            ),
                        )
                    if src.is_dir():
                        raise ConfigurationError(
                            field=f"bundle_exclude_config.exclusions.{name}",
                            message=(
                                f"excluded input {src} resolves to a DIRECTORY; sha256 is undefined "
                                "over a tree, so the by-reference record cannot be "
                                "integrity-pinned. Deposit the directory as a single archive "
                                "and point the cfg field at that file, or carry it in-bundle."
                            ),
                        )
                    # BY REFERENCE a ``.shp`` is one file: the block pins the ``.shp`` and nothing else, so its
                    # sidecars are not carried and the consumer must supply them beside it. Behaviour kept; the
                    # ingest gate refuses a declared ``.shp`` that lacks its ``.shx`` or ``.dbf``.
                    block = {
                        "relpath": rel,
                        "sha256": _sha256_file(src),
                        "accessed": accessed,
                        "citation": ref.citation,
                    }
                    # contentUrl present/absent IS the fetchable bit -- omit the key entirely
                    # when absent rather than writing a null, so the manifest reads as a
                    # deliberate reference-only record.
                    for key in ("contentUrl", "url", "identifier"):
                        val = getattr(ref, key)
                        if val is not None:
                            block[key] = val
                    deposits.append(block)
                    continue

                planned.append(_PlannedInput(src=src, rel=rel, is_dir=src.is_dir()))
                for sidecar in shapefile_sidecars(src):
                    planned.append(
                        _PlannedInput(src=sidecar, rel=str(Path(rel).with_suffix(sidecar.suffix)), is_dir=False)
                    )

    return planned, deposits


def _copy_declared_inputs(
    analysis: TRITONSWMM_analysis,
    staging: Path,
    exclude_config: BundleExcludeConfig | None = None,
) -> list[dict]:
    """Self-contained emit: copy EVERY cfg-declared input file into the bundle
    at exactly the location ``_rewrite_paths_to_relative`` writes into the cfg, so a
    reconstituted config resolves every declared input on disk.

    Executes the plan ``_plan_declared_inputs`` makes. Overlaps with ``_harvest_sources``
    (a renderer-declared input that is also a cfg field) resolve to this staged copy at the
    archive (``_emit_bundle_zip``: staging wins an archive-name collision); ``emit_bundle``
    has already refused the case where the two are DIFFERENT files. A declared-but-absent
    input is SKIPPED here; ``from_doi``'s fail-closed materialize gate raises on it at ingest.

    Returns:
        The ``input_deposit`` blocks for the excluded inputs (empty when nothing is excluded).
    """
    planned, deposits = _plan_declared_inputs(analysis, exclude_config)
    for item in planned:
        dest = staging / item.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item.is_dir:
            # copyfile, not copy2: _emit_bundle_zip stamps a fixed date_time and
            # 0o644 on every entry, so the mtime/mode/xattr copystat would preserve
            # are overwritten by construction -- and each copystat is three extra
            # metadata syscalls per file on the small-file create path that bounds
            # this emit. copytree still copystats each DIRECTORY once (stdlib tail).
            shutil.copytree(item.src, dest, dirs_exist_ok=True, copy_function=shutil.copyfile)
        else:
            shutil.copyfile(item.src, dest)

    return deposits


#: Bundle-root filenames for the minimal runnable set (the reprex carriage).
REPREX_CONFIG_FILENAME = "reprex_config.yaml"
HPC_TEMPLATE_FILENAME = "hpc_system_config.template.yaml"
#: The workflow file the WRC mainEntity points at (the bundled, generated Snakefile).
_WORKFLOW_RELPATH = "Snakefile.source"


@dataclass(frozen=True)
class _StagedName:
    """One name ``emit_bundle`` writes into its staging folder itself.

    ``kind`` is ``"file"`` or ``"tree"`` (a directory the emit fills). ``verbatim`` is the
    ONLY exception to the rule that a declared input may not take the name: given
    ``(analysis_dir, case_manifest_yaml, rest)`` -- ``rest`` being the part of the destination
    under a tree row, and empty for a file row -- it returns the file the emit itself copies
    to that name, or None when the emit writes the name from nothing on disk. A declared
    input whose resolved source IS that file loses nothing by taking the name; any other file
    would silently replace, or be replaced by, the emit's own.
    """

    name: str
    kind: str
    verbatim: Callable[[Path, Path | None, str], Path | None] | None = None


#: The names ``emit_bundle`` writes into its staging folder: a CLOSED set, so it is a table
#: and not a search to be repeated. The totality test in ``tests/test_publish_deposit_paths.py``
#: parses this module for every write into the staging folder and fails when the writers and this
#: table disagree, in either direction. A declared input (or a config string) that equals a
#: file row, or lies under a tree row, refuses unless its source is the row's ``verbatim``.
_EMIT_STAGED_NAMES: tuple[_StagedName, ...] = (
    _StagedName(_CONFIG_PAIR[0][1], "file"),
    _StagedName(_CONFIG_PAIR[1][1], "file"),
    _StagedName("brand_theme.resolved.yaml", "file"),
    _StagedName("Snakefile.source", "file", lambda a, c, r: a / "Snakefile"),
    _StagedName(VERSION_FILE_NAME, "file", lambda a, c, r: a / VERSION_FILE_NAME),
    _StagedName("scenario_status.csv", "file", lambda a, c, r: a / "scenario_status.csv"),
    _StagedName(
        "sensitivity_analysis_definition.csv", "file", lambda a, c, r: a / "sensitivity_analysis_definition.csv"
    ),
    _StagedName("ro-crate-metadata.json", "file"),
    _StagedName("case.yaml", "file", lambda a, c, r: c),
    _StagedName(HPC_IDENTITY_FILENAME, "file"),
    _StagedName(REPREX_CONFIG_FILENAME, "file"),
    _StagedName(HPC_TEMPLATE_FILENAME, "file"),
    _StagedName(BUNDLE_MANIFEST_FILENAME, "file"),
    _StagedName(BUNDLE_PLOTS_SUBDIR, "tree", lambda a, c, r: a / BUNDLE_PLOTS_SUBDIR / r),
    _StagedName("reference_outputs", "tree"),
    _StagedName(
        BUNDLE_STATUS_SUBDIR,
        "tree",
        lambda a, c, r: a / BUNDLE_STATUS_SUBDIR / r if r == "_du.json" else None,
    ),
    _StagedName(BUNDLE_BASELINE_SUBDIR, "tree"),
)


def _staged_name_match(dest: str) -> tuple[_StagedName, str] | None:
    """The table row a bundle-relative destination falls on, with the part under a tree row."""
    for row in _EMIT_STAGED_NAMES:
        if row.kind == "file" and dest == row.name:
            return row, ""
        if row.kind == "tree" and (dest == row.name or dest.startswith(row.name + "/")):
            return row, dest[len(row.name) + 1 :]
    return None


def _declared_destinations(planned: list[_PlannedInput]) -> list[tuple[str, Path]]:
    """Every file a plan will write, as ``(bundle-relative name, source file)``.

    FILE granularity: a directory input contributes one name per member, because two
    directories merging into one tree collide per file, not per directory. The sources are NOT
    resolved here: a full-carry directory can hold hundreds of thousands of members, and a
    source only needs resolving when its NAME is claimed twice (``_assert_archive_namespace``).
    """
    out: list[tuple[str, Path]] = []
    for item in planned:
        if not item.is_dir:
            out.append((item.rel, item.src))
            continue
        for member in sorted(item.src.rglob("*")):
            if member.is_file():
                out.append(((Path(item.rel) / member.relative_to(item.src)).as_posix(), member))
    return out


def _assert_archive_namespace(
    config_outputs: dict[str, set[Path]],
    planned: list[_PlannedInput],
    harvest_entries: dict[str, Path],
) -> None:
    """Refuse when two DIFFERENT files reach ONE name in the bundle root's archive namespace.

    The namespace has three producers: the config strings, the destinations
    ``_copy_declared_inputs`` writes, and the archive names ``_harvest_sources`` plans (merged
    at ``_emit_bundle_zip``, where staging wins). Each of the three is a place the parent
    commit silently kept one file and dropped the other. Two fields naming the same file
    are ONE file and not a collision, and neither are claimants at different paths whose bytes
    are identical (``_hold_identical_bytes``). A claimant that is a directory, or that cannot be
    read when compared, is never identical and is refused here with the same error.

    Sources are resolved, and their bytes compared, LAZILY, only for a name with more than one
    claimant: the harvest plan of a full-carry store is hundreds of thousands of entries, nearly
    all of them claimed once.
    """
    from hhemt.exceptions import ConfigurationError

    claims: dict[str, list[tuple[Path, str]]] = {}

    def _claim(name: str, source: Path, producer: str) -> None:
        claims.setdefault(name, []).append((source, producer))

    for out, files in config_outputs.items():
        for f in files:
            _claim(out, f, "a config path")
    for name, source in _declared_destinations(planned):
        _claim(name, source, "a declared input")
    for name, source in harvest_entries.items():
        _claim(name, source, "a harvested renderer source")

    clashes: dict[str, dict[Path, str]] = {}
    for name, claimants in claims.items():
        if len(claimants) < 2:
            continue
        by_source: dict[Path, str] = {}
        for source, producer in claimants:
            by_source.setdefault(source.resolve(), producer)
        if len(by_source) > 1 and not _hold_identical_bytes(list(by_source)):
            clashes[name] = by_source
    if clashes:
        detail = "; ".join(
            f"{name!r} is claimed by "
            + " and by ".join(f"{src} ({producer})" for src, producer in sorted(by_source.items()))
            for name, by_source in sorted(clashes.items())
        )
        raise ConfigurationError(
            field="bundle_archive_namespace",
            message=(
                "two different files would take one name in the bundle, and the other would be dropped "
                f"silently: {detail}."
            ),
            fix_hint="rename or relocate one of the files so the two bundle names differ",
        )


def _assert_no_emit_name_clobber(
    analysis: TRITONSWMM_analysis,
    config_outputs: dict[str, set[Path]],
    planned: list[_PlannedInput],
) -> None:
    """Refuse a declared input or config path that takes a name the emit writes itself.

    Direction does not matter: a name written BEFORE ``_copy_declared_inputs`` is overwritten
    by the declared input, and one written AFTER overwrites it. Either way a file is lost.
    The one exception is a source that IS the file the emit copies to that name
    (``_StagedName.verbatim``), which loses nothing.
    """
    from hhemt.exceptions import ConfigurationError

    analysis_dir = analysis.analysis_paths.analysis_dir.resolve()
    case_yaml = getattr(analysis, "case_manifest_yaml", None)
    case_path = Path(case_yaml) if case_yaml is not None else None

    candidates: list[tuple[str, Path, str]] = [
        (out, f, "a config path") for out, files in config_outputs.items() for f in files
    ]
    candidates += [(name, src, "a declared input") for name, src in _declared_destinations(planned)]
    for name, source, producer in candidates:
        hit = _staged_name_match(name)
        if hit is None:
            continue
        row, rest = hit
        verbatim = row.verbatim(analysis_dir, case_path, rest) if row.verbatim is not None else None
        if verbatim is not None and source.resolve() == verbatim.resolve():
            continue
        # Name the emit's own file too when it has one: "one of two files is lost" is only
        # actionable if the message says which two.
        emit_side = (
            f"the emit copies {verbatim.resolve()} to that name"
            if verbatim is not None
            else f"the emit writes {row.name!r} itself"
        )
        raise ConfigurationError(
            field="bundle_staged_names",
            message=(
                f"{producer} ({source.resolve()}) would take the bundle name {name!r}, but {emit_side}. "
                "One of the two files would be lost without a message."
            ),
            fix_hint="rename or relocate the file so its bundle name is not one the emit writes",
        )


def _emit_runnable_template_set(staging: Path) -> None:
    """Carry the minimal runnable set into the bundle root: a ``reprex_config.yaml``
    template (the target user's USER-bucket + HPC-selector fields) and a scrubbed
    ``hpc_system_config`` template (the ``{your-allocation}``-placeholder form).

    Both files carry ONLY placeholders — never the producer's real account / login-node
    / SIF path (ship a template, never
    the populated original), so the zero-user-info gate passes trivially. The
    reprex_config key set is sourced from ``reprex_config.model_fields`` so it stays in
    lockstep with the model if a field is added.
    """
    import yaml

    from hhemt.config.reprex_config import reprex_config

    placeholders = {
        "default_account": "{your-allocation}",
        "login_node": "{your-login-node}",
        "sif_root": "/scratch/{your-allocation}/sifs",
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
    # Shape MUST match the live hpc_system_config model (config/hpc_system.py): it is
    # extra="forbid" with REQUIRED hpc_name + partitions, and sif_path is nested under
    # `container:` (ContainerSpec). The prior flat {default_account, login_node, sif_path}
    # form raised a 3-error ValidationError (hpc_name missing, partitions missing,
    # sif_path extra_forbidden) — it was not loadable as an hpc_system_config at all.
    hpc_template = {
        "hpc_name": "{your-cluster-name}",
        "default_account": "{your-allocation}",
        "login_node": "{your-login-node}",
        "gpu_allocation_flavor": "gres",
        "partitions": {
            "{your-gpu-partition}": {
                "gpus_per_node": 4,
                "gpu_hardware": "{your-gpu-arch}",
                "gpu_compilation_backend": "CUDA",
                "max_runtime": 4320,
            },
        },
        "container": {
            "sif_root": "/scratch/{your-allocation}/sifs",
            "gpu_flag": "--nv",
        },
    }
    (staging / HPC_TEMPLATE_FILENAME).write_text(
        "# hpc_system_config template — the HPC-specific info you must revise to run\n"
        "# this bundle on YOUR cluster. Placeholders only; contains zero producer info.\n"
        "#\n"
        "# This is a SHAPE SKETCH, not a fill-three-blanks config: `partitions` must\n"
        "# describe YOUR cluster's real partitions and caps. Start from a worked example\n"
        "# in the hhemt repo -- test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml\n"
        "# (NVIDIA/gres/pmix) or hpc_system_config_frontier.yaml (AMD/gpus/Cray-MPICH) --\n"
        "# and pass the result via `hhemt ingest --hpc-system-config ...`.\n"
        "#\n"
        "# The `container:` block is consumed ONLY when the analysis config sets\n"
        "# execution_environment: container. On ingest, hhemt writes container.sif_root into a\n"
        "# DERIVED copy and places (or rebuilds) each carried image at its identity path under it\n"
        "# (ADR-21) -- your file is never modified.\n" + yaml.safe_dump(hpc_template, sort_keys=False)
    )


def _emit_sif_manifests(analysis: TRITONSWMM_analysis) -> list[dict]:
    """The CARRIED identity source: the producer ``.manifest.json`` of the image the
    SIMULATIONS RAN IN, resolved by the digest SETUP RECORDED (``system_log.sif_sha256``) rather
    than by an identity recomputed from the running toolkit -- the same digest-keyed lookup
    ``_reprex.py::_verify_sif`` performs on the consume side, so both ends of the round trip name
    one image by construction.

    The premise is narrower than the one it replaces, and that is the whole fix. Setup and the
    simulations are ONE workflow run at ONE toolkit version, so the digest setup recorded names
    what RAN. The emit is a separate act later on whatever checkout the operator drives, which is
    where "emit runs on the producer's git checkout, so recompute is total" failed: a toolkit
    advanced mid-campaign moves ``hhemt_sha`` -- the one identity input no config pins -- and the
    emit then demands an image nothing ever ran.

    Fails closed in four states, none of which names a recomputed identity and none of which
    names ``hhemt build-sifs``: a rebuild produces a different blob with a different digest, so it
    is not the remedy for any of them.
    """
    from hhemt.config.hpc_system import resolve_gpu_target
    from hhemt.exceptions import ConfigurationError
    from hhemt.sif.identity import derive_family, find_by_sha256
    from hhemt.sif.pins import resolve_full_sha

    cfg_hpc = analysis.cfg_hpc_system
    cspec = getattr(cfg_hpc, "container", None)
    if cspec is None:
        raise ConfigurationError(
            field="container", message="container-mode bundle with no container: block", config_path=None
        )
    cfg_system = analysis._system.cfg_system
    log = analysis._system.log
    # Setup and emit are different processes (different SLURM jobs on HPC), so the in-memory log
    # is not the one setup wrote. Same reason provenance._sif_spec_from_system_log refreshes.
    log.refresh()
    log_path = getattr(log, "logfile", "the system log")
    digest = log.sif_sha256.get()

    # (a) Nothing recorded. The capture in setup_workflow's container branch is ITSELF
    # identity-keyed -- resolve_sif(..., analysis._sif_identity_for(target_partition)) -- and
    # gated on the resolved path being a file, so a naive "re-run setup" at a NEWER toolkit
    # resolves a path holding no image, skips the capture silently, and lands back here. Both
    # remedies below carry the precondition that makes them work.
    if not digest:
        raise ConfigurationError(
            field="container.sif_root",
            message=(
                f"cannot bundle: no SIF digest is recorded in {log_path}. A container-mode bundle "
                "carries the image the simulations RAN IN, and this analysis recorded none -- the "
                "setup ran before the digest capture existed, the image was a sandbox directory "
                "(no single blob to hash), or the digest read failed. REMEDY, and its precondition "
                "is load-bearing because the capture resolves the image by the RUNNING toolkit's "
                "identity: either (1) check out the toolkit commit the image was built at and "
                "re-run the setup rule there, or (2) build an image at the running commit and "
                "re-run the setup rule. Re-running setup at a toolkit whose identity resolves to "
                "no image on disk records nothing, and this refusal recurs unchanged."
            ),
            config_path=None,
        )

    # (b) Recorded, but the image it names is not here. The remedy wording mirrors
    # _reprex.py::_verify_sif's so a producer reading this and a reproducer reading that are told
    # the same thing: obtain the reference image, do not rebuild one.
    found = find_by_sha256(cspec.sif_root, str(digest))
    if found is None:
        raise ConfigurationError(
            field="container.sif_root",
            message=(
                f"cannot bundle: the digest recorded in {log_path} is {digest}, but no manifest "
                f"under {cspec.sif_root} records that sha256. The image the simulations ran in is "
                "not at its identity path (or sif_root moved). Restore that image and its "
                ".manifest.json to sif_root. A rebuild does NOT recover it -- it produces a "
                "different blob with a different digest."
            ),
            config_path=None,
        )
    _sif_path, manifest = found
    ident = manifest.get("identity") or {}

    # (d) The carried identity must be one THIS analysis could have used. Checked BEFORE the
    # coverage arm so that by the time (c) runs, mpi/accel provably agree and (c) reduces to the
    # gpu_hardware question it is actually about. Unconditional, with no best-effort branch: this
    # function returns list[dict] and has no channel to disclose a skip, so a skipped check reads
    # downstream exactly like a passed one. triton_sha costs no git call in container mode --
    # sif.pins.resolve_full_sha returns a 40-hex ref as-is.
    mpi, accel = derive_family(cspec)
    for field_name, carried, expected in (
        ("mpi_family", ident.get("mpi_family"), mpi),
        ("accel", ident.get("accel"), accel),
        ("swmm_tag", ident.get("swmm_tag"), cfg_system.SWMM_tag_key),
        (
            "triton_sha",
            ident.get("triton_sha"),
            resolve_full_sha(cfg_system.TRITONSWMM_software_directory, str(cfg_system.TRITONSWMM_branch_key)),
        ),
    ):
        if carried != expected:
            raise ConfigurationError(
                field="container.sif_root",
                message=(
                    f"cannot bundle: the image recorded for this analysis ({digest}, from "
                    f"{log_path}) carries {field_name}={carried!r}, but this analysis's config "
                    f"declares {expected!r}. CAUSE: one system_directory holds one system log, and "
                    "analysis_dir defaults to system_directory/ANALYSIS_ID, so several analyses "
                    "under one system_directory SHARE this digest slot and the last setup to run "
                    "wins. Two sensitivity build targets can also differ in solver pin while "
                    "presenting the same (mpi_family, accel, gpu_hardware) tuple, because that "
                    "tuple is the dedup key and the pin is not part of it. Give this analysis its "
                    "own system_directory, or re-run its setup so the recorded digest is its own."
                ),
                config_path=None,
            )

    # (c) One recorded digest is ONE image. It serves the matrix iff every partition the matrix
    # requires resolves to the identity the carried image has. Partition COUNT is not the test:
    # under a CPU container spec the partition contributes nothing to the identity at all
    # (derive_identity discards it), so a cpu-spec matrix over any number of partitions is
    # single-image by construction.
    for part in sorted(_matrix_required_partitions(analysis.cfg_analysis, cfg_hpc)):
        hw = resolve_gpu_target(cfg_hpc, part)[0] if accel != "cpu" else None
        if hw != ident.get("gpu_hardware"):
            raise ConfigurationError(
                field="container.sif_root",
                message=(
                    f"cannot bundle: partition {part!r} requires an image for "
                    f"({mpi}, {accel}, {hw or 'cpu'}), but the image recorded for this analysis "
                    f"({digest}, from {log_path}) is for ({ident.get('mpi_family')}, "
                    f"{ident.get('accel')}, {ident.get('gpu_hardware') or 'cpu'}). The system log "
                    "records ONE SIF digest, so a matrix needing more than one image cannot be "
                    "bundled from it. Split the matrix by hardware and bundle each arm."
                ),
                config_path=None,
            )
    return [manifest]


def _matrix_required_partitions(cfg_analysis, cfg_hpc) -> set[str]:
    """Distinct partition NAMES the analysis matrix requires: the primary ensemble partition
    plus, for a sensitivity analysis, every distinct per-row ``hpc.partition`` /
    ``analysis.hpc_ensemble_partition`` value in the setup table (CSV or XLSX — when an
    XLSX is given it is the source of truth). Takes CONFIGS, not an analysis, so ``hhemt.sif.plan``
    and ``hhemt.sif.preflight`` can call it. RAISES on a read failure — the SIF planner must
    not silently under-plan; ``_matrix_required_arches`` below is the non-raising variant."""
    if cfg_hpc is None:
        return set()
    partitions: set[str] = set()
    _primary = cfg_analysis.hpc_ensemble_partition
    if _primary:
        partitions.add(str(_primary))
    if getattr(cfg_analysis, "toggle_sensitivity_analysis", False):
        _ref = cfg_analysis.sensitivity_analysis
        if _ref and Path(_ref).is_file():
            import pandas as pd

            _df = pd.read_excel(_ref) if str(_ref).lower().endswith((".xlsx", ".xls")) else pd.read_csv(_ref)
            for _col in ("hpc.partition", "analysis.hpc_ensemble_partition"):
                if _col in _df.columns:
                    partitions |= {str(v) for v in _df[_col].dropna().tolist() if str(v).strip()}
    return partitions


def _matrix_required_arches(analysis: TRITONSWMM_analysis) -> set[str]:
    """Distinct ``gpu_hardware`` arches (a100/a6000/...) the analysis matrix requires,
    best-effort: ``_matrix_required_partitions`` mapped via ``resolve_gpu_target[0]``.
    Returns ``set()`` on any read failure rather than raising. Nothing in ``src/`` calls it
    today; its only caller is ``tests/test_experiments_from_doi.py``.
    None/CPU arches (partitions with no GPU) are excluded."""
    from hhemt.config.hpc_system import resolve_gpu_target

    cfg_hpc = analysis.cfg_hpc_system
    if cfg_hpc is None:
        return set()
    try:
        partitions = _matrix_required_partitions(analysis.cfg_analysis, cfg_hpc)
    except Exception:
        return set()  # cannot enumerate -> do not block emit; ingest guard backstops
    return {hw for hw in (resolve_gpu_target(cfg_hpc, p)[0] for p in partitions) if hw}


def _upgrade_crate_to_workflow_run_crate(staging: Path) -> None:
    """Patch the bundle's copied ``ro-crate-metadata.json`` into a Workflow-Run-Crate
    (``mainEntity`` = the bundled ``Snakefile.source``), in place.

    Reuses the by-reference SIF + ``input_parts`` already in the copied sidecar (no
    ``sif_spec`` reconstruction) and reserializes
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
    """Synthesize a runnable ``system_config.yaml`` from a reprex bundle.

    Reads the bundle's scrubbed ``cfg_system.yaml`` and writes a ``system_config.yaml``
    whose bundle-relative Path fields are resolved to absolute paths under
    ``bundle_root`` (so the carried, self-contained inputs resolve at load-time), and
    every EXPERIMENT-bucket field is preserved verbatim. Returns the written path
    (``bundle_root/system_config.yaml`` unless ``target_path`` overrides).

    The two software-dir fields (``SWMM_software_directory`` /
    ``TRITONSWMM_software_directory``) are toolkit-owned OUTPUTS — build dirs the toolkit
    creates at target-side setup, exempt from the load-time existence check
    (``toolkit_owned_output``). Their
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
    out = rebase_bundle_relative_paths(cfg_dict, system_config, bundle_root)
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


def reconstitute_runnable_analysis_config(bundle_root: Path, *, target_path: Path | None = None) -> Path:
    """Synthesize a runnable ``analysis_config.yaml`` from a reprex bundle — the
    analysis-side sibling of :func:`reconstitute_runnable_config`.

    The bundle's ``cfg_analysis.yaml`` stores its Path fields bundle-relative
    (``_PATH_FIELD_POLICY``): ``analysis_dir`` is the FORCED_DOT ``"."`` bundle-root
    marker, list fields (``static_plot_configs``) are element-wise relative, and every
    other declared input is ``BUNDLE_RELATIVE``/``BUNDLE_RELATIVE_OR_NONE``. This helper
    rewrites every non-absolute Path value to ``str((bundle_root / v).resolve())`` — which
    maps ``analysis_dir: "."`` to ``bundle_root`` (honoring the FORCED_DOT invariant) and
    rebases ``sensitivity_analysis`` and the rest onto the carried, self-contained inputs.
    The rebase itself is delegated to ``_path_policy.rebase_bundle_relative_paths``, which
    is the SINGLE rebase implementation shared with the system-side sibling and with the
    three render-path readers; this function composes it and writes the result, and
    ``reprex()`` composes THIS rather than hand-rebasing ``sensitivity_analysis`` inline.
    Returns the written path
    (``bundle_root/analysis_config.yaml`` unless ``target_path`` overrides).
    """
    import yaml

    from hhemt.config.analysis import analysis_config

    bundle_root = Path(bundle_root).resolve()
    cfg_dict = yaml.safe_load((bundle_root / "cfg_analysis.yaml").read_text())
    out = rebase_bundle_relative_paths(cfg_dict, analysis_config, bundle_root)
    target = target_path if target_path is not None else bundle_root / "analysis_config.yaml"
    Path(target).write_text(yaml.safe_dump(out, sort_keys=False))
    return Path(target)


def _annotate_crate_excluded_inputs(staging: Path, input_deposits: list[dict]) -> None:
    """Give each EXCLUDED input a by-reference ``File`` entity in the sidecar crate.

    An earlier design called for "a DESCRIPTIVE (no-``downloadUrl``) File part per excluded
    input". That was a NO-OP and was REVERSED on 2026-07-14:
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
    FAIR-facing artifact. The embedded core therefore needs no change for this annotation.
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
    sif_manifests: list[dict] | None = None,
    declared_sources_absent: list[str] | None = None,
    pruned_orphan_figures: list[str] | None = None,
    crate_license_finding: str | None = None,
    metadata_only_stores: list[str] | None = None,
    analysis_dir: Path | None = None,
    omitted_output_zarrs: list[str] | None = None,
    carried_output_stores: list[str] | None = None,
    stale_figures_included: list[str] | None = None,
    declared_sources_unresolved_outside_tree: list[str] | None = None,
    zero_user_info: dict[str, str] | None = None,
) -> None:
    """Write ``bundle_manifest.json`` into ``staging``.

    ``analysis_dir`` is OPTIONAL. Given, each source in ``source_paths_by_renderer`` is
    spelled the way the archive names it (``_source_rel``): master-analysis-relative for a
    source under it (so a sensitivity member's source reads ``members/member_{N}/...``) and
    ``external/{filename}`` outside it, which keeps the producer's absolute layout out of a
    deposited manifest. Omitted, sources are written as given, and an ABSOLUTE source then
    raises ``ValueError``: it could only be written as the producer's own path, which is the
    leak this keyword exists to close. Callers that pass relative or empty sources are
    unaffected.
    """

    def _spell(source: Path) -> str:
        p = Path(source)
        if not p.is_absolute():
            return str(source)
        if analysis_dir is None:
            raise ValueError(
                f"source path {p} is absolute and no analysis_dir was given to relativize it against; "
                "pass analysis_dir so the manifest names the archive entry rather than the producer's path"
            )
        return _source_rel(p.resolve(), analysis_dir.resolve(), name=p.name).as_posix()

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "layout_version": LAYOUT_VERSION,
        "toolkit_git_sha": git_sha,
        # The BUNDLE-stage provenance-stamp capture site. `toolkit_git_sha` is NOT this
        # stamp under another name: it is one field where the stamp is three, and it
        # carries no version and no dirty flag. Minting the full triple here is what moves
        # `bundle` from uncaptured to captured; teaching the READER to accept the single
        # field instead would fabricate the two missing values, which is the
        # never-infer-one-stage's-stamp-from-another rule _collect_stage_stamps states.
        **producing_stamp(),
        "analysis_id": analysis_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_paths_by_renderer": {name: [_spell(p) for p in paths] for name, paths in sources_by_renderer.items()},
    }
    if bundle_root_invariants is not None:
        manifest["bundle_root_invariants"] = bundle_root_invariants
    if declared_sources_absent:
        # Renderers that DECLARED a source which was not on disk at harvest time.
        # Non-fatal (see _harvest_sources), but recorded here so the state is auditable from the
        # shipped artifact rather than only from a warning in a rule log. Absent (not
        # empty) when every declared source resolved, so a clean bundle's manifest is
        # byte-identical to what it was before this key existed.
        manifest["declared_sources_absent"] = declared_sources_absent
    if declared_sources_unresolved_outside_tree:
        # Out-of-tree sources a sidecar declared that did not resolve from THIS tree's
        # location (a moved or re-depthed tree): spelled as the sidecar's own ``../`` path, never as an
        # ``external/`` archive name, because the declared-input copy may carry the file under
        # that name. Absent (not empty) when none, as declared_sources_absent is.
        manifest["declared_sources_unresolved_outside_tree"] = declared_sources_unresolved_outside_tree
    if crate_license_finding:
        # The staged crate's license disagrees with the carried cfg_analysis.yaml, or the
        # crate's root could not be resolved to check. Non-fatal HERE and refused only at
        # the publish boundary: the bundle is inspectable and re-derivable, a DOI deposit
        # is neither. Same absent-when-clean property as the key above, for the same
        # byte-identity reason.
        manifest["crate_license_finding"] = crate_license_finding
    if pruned_orphan_figures:
        # The prune's audit trail, carried IN the bundle for the same reason
        # declared_sources_absent is: a warnings.warn inside a Snakemake rule log is lost by
        # the time anyone reads the bundle, and a reader finding 28 fewer figures than a
        # prior generation needs the removal list here rather than in a rotated job log.
        # Absent (not empty) when nothing was pruned, so a clean manifest stays
        # byte-identical to what it was before this key existed.
        manifest["pruned_orphan_figures"] = pruned_orphan_figures
    if input_deposits:
        # The by-reference record for each EXCLUDED input. Absent (not empty) when
        # the bundle is self-contained, so a self-contained manifest is byte-identical to
        # what it was before this feature existed.
        manifest["input_deposit"] = input_deposits
    if sif_manifests:
        # The producer manifests of the image the matrix resolves to (one: _emit_sif_manifests
        # refuses a matrix needing more). ABSENT (not empty) for a native bundle, so a native
        # manifest is byte-identical to before — same rule as input_deposit.
        # BUNDLE_SCHEMA_VERSION went 5 -> 6 when this key was added.
        manifest["sif_manifests"] = sif_manifests
    if metadata_only_stores:
        # Stores carried SHALLOWLY (zarr metadata documents only, no chunk files) because
        # every figure that declared them read only their metadata. A reader that opens
        # one of these for DATA gets fill values, not an error (measured: NaN, no raise),
        # so the manifest names them. Absent (not empty) when no store is shallow, so a
        # full-carry manifest is byte-identical to what it was before this key existed.
        manifest["metadata_only_stores"] = metadata_only_stores
    if omitted_output_zarrs:
        # The stores `--omit-output-zarrs` left out (a subset of metadata_only_stores, which
        # also names them so `combine` refuses). The regeneration Snakefile reads THIS key to
        # refuse a plot rule that declared one in full. Absent (not empty) without the flag,
        # so a default manifest is byte-identical to what it was before this key existed.
        manifest["omitted_output_zarrs"] = omitted_output_zarrs
    if carried_output_stores:
        # The time-series stores the emit carried because they exist (`analysis_datatree.zarr` and each scenario's
        # time-series stores), independent of any figure, as analysis-relative paths. Under `--omit-output-zarrs` a
        # `.zarr` among them is shallow and is also named in `omitted_output_zarrs`. Absent (not empty) when the
        # analysis has none, so a manifest without time-series stores is byte-identical to what it was before this
        # key existed. No BUNDLE_SCHEMA_VERSION bump: the key is additive and no consumer reads it.
        manifest["carried_output_stores"] = carried_output_stores
    if stale_figures_included:
        # The analysis-relative paths of the stale figures (figures of members no longer in the
        # setup table) that `--allow-stale-figures` carried into this bundle. Absent (not
        # empty) when none were carried, so a clean manifest is byte-identical to what it was
        # before this key existed.
        manifest["stale_figures_included"] = stale_figures_included
    if zero_user_info is not None:
        # What the emit-time check ran, layer by layer, so a reader of the shipped manifest can tell a
        # bundle certified by the generic patterns and the configs' own values alone (`private_list`
        # is `not configured`) from one that was also searched for a private list's tokens.
        manifest["zero_user_info"] = zero_user_info
    (staging / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))


def _emit_bundle_zip(
    staging: Path,
    output_path: Path,
    *,
    streamed: dict[str, Path] | None = None,
    placeholders: PlaceholderTable | None = None,
    declared_arcnames: frozenset[str] = frozenset(),
    scanner: ZeroUserInfoScanner | None = None,
) -> None:
    # Emit a deterministic zip from staging PLUS the streamed plan.
    #
    # `streamed` maps archive name -> on-disk source for the harvested sources, which
    # are never copied into staging (they are byte-identical to their originals and
    # the zip normalizes every timestamp and mode). Staging wins an archive-name
    # collision. The only collision that can reach this point is a cfg-declared input that a
    # renderer also declared (`_copy_declared_inputs` carries it into staging) with the SAME
    # bytes, and that is true by construction: `emit_bundle` refuses, before anything is
    # copied, every case where two DIFFERENT files would take one name
    # (`_assert_archive_namespace`, and `_plan_harvest_entry` inside the harvest). First-wins
    # is therefore a dedupe, not a choice.
    #
    # The bytes are written by `bundle._archive.write_zip` under the MODERATE policy
    # (DEFLATE level 6 for every entry except zarr chunk files, which are STORED because
    # the toolkit already compresses each chunk). Where the bytes are READ from is not a
    # determinism input; the entry set, the order and the stamps are, and `write_zip` fixes
    # all three: entries sorted by PATH PARTS (`a.split("/")`, the order
    # `sorted(staging.rglob("*"))` gave before the streamed plan existed, so an archive with
    # no streamed entries keeps its order), date 1980-01-01 (the zipfile module's minimum
    # valid date, which removes real filesystem timestamps) and mode 0o644 on every entry,
    # each entry streamed in 1 MiB blocks so no file is held whole in memory.
    #
    # Determinism contract, restated for compression: the same entries, in the same order,
    # with the same stamps and the same content bytes, always; the same ARCHIVE bytes
    # whenever the same zlib build compresses them. DEFLATE output is not specified across
    # zlib implementations, so two machines with different zlib builds can write archives
    # that differ byte for byte while every entry's content is identical. The repeat-emission
    # tests compare two emits in one process and so hold by construction. Deliberately NOT
    # ZIP_STORED any more: a stored member zip carries the raw solver files at full size
    # through every later copy and share of the bundle.
    #
    # Bytes are STREAMED through zf.open(info, "w") rather than read whole: with
    # info.file_size set from stat() first, this is exactly what ZipFile.writestr does
    # internally (it sets zinfo.file_size = len(data) and calls open(zinfo, "w")), so the
    # archive is byte-identical to the writestr form while never holding a file in memory.
    #
    # With a `placeholders` table the one exception is a TEXT member, which is read whole,
    # rewritten, and archived at the size of the rewritten bytes. That work is the `transform`
    # handed to `write_zip`, which calls it with the open handle of each path source. Each member
    # is opened exactly once: the text sniff reads its prefix from the SAME handle `write_zip`
    # then rewinds and streams when the transform answers None, because a bundle holds on the
    # order of 350,000 harvested files and a second open per member would bring back the
    # per-file cost this streamed emit exists to avoid. A member is left byte for byte alone when
    # it is a declared input (`declared_arcnames`) or a zarr chunk (classed binary by its name,
    # so its prefix is never read); it is still compressed as the policy says, as is every
    # rewritten member. With no table and no scanner there is no transform at all, which is the
    # unchanged emitter `_combine` relies on.
    #
    # With a `scanner` every text member is also checked, AFTER the rewrite, from the same bytes
    # that are written (declared inputs included: they are not rewritten, so a producer value in one
    # is exactly what the check exists to find). The members that are NOT text are checked too, and
    # by what they hold rather than by their bytes: a zarr array of text values is decoded
    # (`_decode_and_scan_zarr_arrays`), and any other NUL-prefixed member is read through its
    # decompressor when the standard library has one and in overlapping blocks otherwise
    # (`scan_binary_member`). The chunk files of a zarr array are never raw-scanned. Hits accumulate
    # across the whole archive and the refusal is raised once, after the zip is closed, so the
    # message names every offending member.
    entries: dict[str, Path] = {
        entry.relative_to(staging).as_posix(): entry for entry in sorted(staging.rglob("*")) if not entry.is_dir()
    }
    for arcname, src in (streamed or {}).items():
        entries.setdefault(arcname, src)
    rewrite = placeholders is not None and not placeholders.is_empty
    transform = None
    if rewrite or scanner is not None:
        # Looked up only when a member can be inspected: it imports zarr, which an emit with no table
        # and no scanner (`_combine`) never needed.
        metadata_names = _zarr_metadata_filenames()
        # With a scanner, the values the bytes of a zarr array only encode are DECODED and scanned here,
        # and the array's chunk files are then left out of the binary-member scan below.
        array_dirs = (
            _decode_and_scan_zarr_arrays(entries, scanner, metadata_names) if scanner is not None else frozenset()
        )

        def rewrite_and_scan(arcname: str, handle) -> bytes | None:
            """The text rewrite and scan of one member: the bytes to archive, or None to archive the file as is."""
            if (scanner is None and arcname in declared_arcnames) or is_zarr_chunk_member_by_name(
                arcname, metadata_names
            ):
                return None
            prefix = handle.read(SNIFF_BYTES)
            if has_byte_order_mark(prefix):
                # A member that begins with a UTF-16 or UTF-32 byte-order mark is tested BEFORE the NUL sniff:
                # a UTF-16 file of non-Latin text can hold no NUL byte in its first block and would be read
                # as UTF-8 text. A marked member is never rewritten; the shared binary path decodes it
                # strictly, or raw-scans it as an opaque binary when its body does not decode.
                if scanner is not None and not _is_under_any(arcname, array_dirs):
                    scan_binary_member(scanner, arcname, handle)
                return None
            if not prefix_is_text(prefix):
                if scanner is not None and not _is_under_any(arcname, array_dirs):
                    # A binary member that is not a zarr chunk: read through its decompressor when the
                    # standard library has one, in overlapping blocks otherwise. The handle is rewound
                    # by the scan and again by `write_zip` when this answers None.
                    scan_binary_member(scanner, arcname, handle)
                return None
            data = prefix + handle.read()
            if rewrite and arcname not in declared_arcnames:
                # The whole-path rule runs on the table's output, inside the same guard: a declared
                # input is archived byte for byte, so a token-shaped path in one is not rewritten.
                data = rewrite_producer_paths(placeholders.apply(data))
            if scanner is not None:
                scanner.record(arcname, data)
            return data

        transform = rewrite_and_scan

    write_zip(entries.items(), output_path, policy=MODERATE, transform=transform)
    if scanner is not None:
        scanner.raise_if_hits(None, "bundle emit zero-user-info check")


def _is_under_any(arcname: str, directories: frozenset[str]) -> bool:
    """True when some ancestor directory of ``arcname`` is in ``directories``."""
    parent = arcname.rpartition("/")[0]
    while parent:
        if parent in directories:
            return True
        parent = parent.rpartition("/")[0]
    return False


def _decode_and_scan_zarr_arrays(
    entries: dict[str, Path], scanner: ZeroUserInfoScanner, metadata_names: frozenset[str]
) -> frozenset[str]:
    """Decode every carried zarr array of text values and record its values with ``scanner``.

    Returns the archive directories of ALL the zarr arrays among ``entries`` (text-typed or not), so the
    caller leaves their chunk files out of the binary-member scan: the values of a text array are read
    here, and a numeric chunk's compressed bytes would only produce false positives. An array is read
    only when at least one of its chunk files is among ``entries``; a store carried SHALLOWLY (metadata
    documents only) has no values in the bundle to check, and reading the source store would scan values
    that are not shipped. Each hit is recorded under ``{array directory} (decoded values)``.
    """
    array_dirs: dict[str, Path] = {}
    for arcname, src in entries.items():
        if arcname.rsplit("/", 1)[-1] in ZARR_ARRAY_METADATA_NAMES:
            directory = Path(src).parent
            if classify_zarr_array(directory) is not None:
                array_dirs[arcname.rpartition("/")[0]] = directory
    if not array_dirs:
        return frozenset()
    carried: set[str] = set()
    known = frozenset(array_dirs)
    for arcname in entries:
        if arcname.rsplit("/", 1)[-1] in metadata_names:
            continue
        parent = arcname.rpartition("/")[0]
        while parent:
            if parent in known:
                carried.add(parent)
                break
            parent = parent.rpartition("/")[0]
    for arcname_dir in sorted(carried):
        directory = array_dirs[arcname_dir]
        if classify_zarr_array(directory) == ZARR_DECODE:
            scanner.record(f"{arcname_dir} (decoded values)", decode_zarr_array_text(directory))
    return known


def _toolkit_source_dir() -> Path:
    """The directory of the INSTALLED toolkit source. The anchor for every git query.

    Anchoring on __file__ (not the process CWD) is load-bearing: real runs are driven
    from a private estate repo, and an unanchored `git` resolves THAT repo
    — measured 2026-07-15, `_get_toolkit_git_sha()` from a foreign repo's CWD returned
    the foreign SHA while hhemt was imported from a different checkout.
    """
    return Path(__file__).resolve().parent


def _get_toolkit_git_sha(strict: bool = True) -> str:
    """Resolve the toolkit's git SHA (12-hex) for bundle provenance.

    Delegates to ``validation.running_identity`` — the ONE source of the running toolkit's
    commit — and keeps this function's two contracts: strict=True (emit-side)
    raises ConfigurationError when the toolkit has no identity; strict=False
    (consume-side) returns "unknown". The 12-hex width is this function's consumers'
    (bundle manifest, crate, combine) and is a truncation of the one 40-hex value, never a
    second git read.
    """
    from hhemt.exceptions import ConfigurationError
    from hhemt.validation import running_identity

    try:
        return running_identity().sha[:12]
    except ConfigurationError as exc:
        if not strict:
            return "unknown"
        raise ConfigurationError(
            field="toolkit_git_sha",
            message=(
                "Cannot resolve the toolkit sha for bundle provenance: "
                f"{exc}. Install hhemt from a git checkout (not a wheel) or run inside the SIF."
            ),
            config_path=None,
        ) from exc


@contextlib.contextmanager
def _staging_dir(parent: Path):
    """Sibling staging directory; cleaned up on exit."""
    with tempfile.TemporaryDirectory(prefix="bundle_staging_", dir=parent) as tmp:
        yield Path(tmp)

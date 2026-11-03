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

Phase-3/4/5 scope (as-built): steps 1-2 and the step-4 EMIT are byte-deterministic
(CR4). The step-3 cross-experiment RENDER (``_render_combined_report``) direct-renders
the three ``combined`` ReportingSet figures (compatibility, intercomparison, aggregate
errors-and-warnings) in-process (they are NEW and do not pre-exist), then emits a
FIRST-CLASS Snakemake ``--report`` via ``bundle.combined_snakefile_generator`` (reusing
the single-bundle report machinery: ``_emit_report_artifacts`` + ``write_combined_snakefile``
+ ``snakemake --touch`` + ``snakemake --report`` + ``_react_surgery``). That generator ALSO
HARVESTS each ``child_crates/{eid}/``'s already-rendered ``plots/`` figures into the SAME
combined Snakefile as native ``report()`` items under per-experiment sidebar categories
(``category={eid}`` / ``subcategory={child category}``), so the ONE combined report carries
every experiment's content natively — NO ``index.html`` front door, NO per-child report
regen (Option B, superseding the earlier Option-A out-link design). CR4 is unaffected:
the deterministic-emit test monkeypatches ``_render_combined_report`` to a no-op before the
byte-identity check, and child figures are referenced in place (no new files under the root).

Determinism (CR4): ro-crate-py stamps a per-run ``datePublished`` wall-clock on
the root Dataset at ``crate.metadata.generate()`` time, and ``_emit._write_bundle_manifest``
stamps a per-run ``created_at_utc`` — two volatile surfaces that would make two
combines of the same N bundles diverge byte-wise. Both are neutralized here: the
combined crate strips ``metadata._COMBINED_EMIT_STRIP_KEYS`` (which includes
``datePublished``) before serialization, and ``_write_combined_bundle_manifest``
OMITS ``created_at_utc`` (its deterministic provenance surface is ``combined_of``).

``_COMBINED_EMIT_STRIP_KEYS`` is this tier's OWN set and is not interchangeable with
``_BUNDLE_TRANSFER_STRIP_KEYS``, which the bundle tier uses, even though the two are
byte-equal today. This tier constructs a fresh ``ROCrate()`` and emits a FIRST
publication OF THE COMBINATION, so it owes an authored ``datePublished`` rather than an
inherited one; the bundle tier transports an existing crate and owes preservation. The
RO-Crate spec makes ``datePublished`` a MUST on the Root Data Entity, so neither tier may
simply omit it, and the two tiers cannot discharge that MUST by the same rule.
Inputs are sorted before harvest and the archive reuses ``_emit._emit_bundle_zip``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from hhemt.bundle._archive import MODERATE, CompressionPolicy, _confined_target, extract_zip, render_set, write_zip
from hhemt.bundle._combine_merge import _experiment_id, _resolve_root_tree
from hhemt.bundle._compatibility import (
    _CFG_SYSTEM_FILENAME,
    CompatibilityReport,
    _load_yaml,
    check_bundle_compatibility,
)
from hhemt.bundle._emit import _get_toolkit_git_sha
from hhemt.exceptions import ConfigurationError, ProcessingError
from hhemt.provenance import producing_stamp
from hhemt.utils import ROOT_TREE_NAMES, open_store
from hhemt.version_migration.constants import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
)

# Combined-bundle layout constants.
_CHILD_CRATES_SUBDIR = "child_crates"
_MEMBER_ZIP_DIR = ".member_zips"
_COMBINED_ROCRATE_FILENAME = "ro-crate-metadata.json"
_COMBINED_COMPAT_FILENAME = "combined_compatibility.json"
# Provenance Run Crate profile — the combined tier is a base RO-Crate 1.2 +
# Provenance-Run-Crate (NOT Workflow-Run-Crate; A8 / primary D3 / ADR-9 flip-condition).
_COMBINED_CONFORMS_TO = (
    {"@id": "https://w3id.org/ro/crate/1.2"},
    {"@id": "https://w3id.org/ro/wfrun/provenance/0.5"},
)


class _BundleSource:
    """One input bundle as the combine reads it: a directory or a member zip.

    The combine needs three things from an input: the names of its files, the bytes of a
    named file (the manifest, before anything is extracted), and a way to put a chosen
    subset of its files on disk. Putting those behind one class keeps every later step
    (identity, compatibility, the provenance gate, the renderers) working on a folder, and
    leaves one place to change when a source is read from somewhere else.

    A source holds NO open file between calls. A zip source opens its archive for each call
    and closes it again, so while the combine walks its inputs one at a time only one
    member's table of contents is in memory, never all of them.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def names(self) -> list[str]:
        """Every file in the bundle, as a path relative to the bundle root, ``/``-separated."""
        raise NotImplementedError

    def read_bytes(self, name: str) -> bytes:
        """The bytes of the file ``name``; ``KeyError`` when the bundle has no such file."""
        raise NotImplementedError

    def extract(self, dest: Path, members: list[str]) -> None:
        """Write the files ``members`` under the directory ``dest``, keeping their relative paths."""
        raise NotImplementedError


class _DirectorySource(_BundleSource):
    """A bundle that is an unpacked directory."""

    def names(self) -> list[str]:
        # Directory links are followed, because the `shutil.copytree` this replaces copied what a
        # symlinked subdirectory points at: not following them would drop its files (a `plots/`
        # linked to a shared folder, say) from the combined child with no error. A link that points
        # at the directory being walked or at one of its ancestors is skipped, since following it
        # would list the same files without end.
        found: list[str] = []
        for dirpath, dirs, files in os.walk(self.path, followlinks=True):
            here = os.path.realpath(dirpath)
            for name in list(dirs):
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    target = os.path.realpath(full)
                    if here == target or here.startswith(target + os.sep):
                        dirs.remove(name)
            rel = Path(dirpath).relative_to(self.path)
            found.extend((rel / name).as_posix() for name in files)
        return sorted(found)

    def read_bytes(self, name: str) -> bytes:
        try:
            return (self.path / name).read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(name) from exc

    def extract(self, dest: Path, members: list[str]) -> None:
        # Directories are made serially and the copies run on threads: a bundle holds tens of
        # thousands of small files, and creating them one after another is what bounds the copy.
        for parent in sorted({(dest / name).parent for name in members}):
            parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(shutil.copyfile, self.path / name, dest / name) for name in members]
            for future in futures:
                future.result()


class _ZipSource(_BundleSource):
    """A bundle that is a member zip (the archive ``emit_bundle`` writes, bundle files at its root)."""

    def names(self) -> list[str]:
        with zipfile.ZipFile(self.path) as zf:
            return [name for name in zf.namelist() if not name.endswith("/")]

    def read_bytes(self, name: str) -> bytes:
        with zipfile.ZipFile(self.path) as zf:
            return zf.read(name)

    def extract(self, dest: Path, members: list[str]) -> None:
        extract_zip(self.path, dest, members=members)


def _open_bundle_source(path: Path) -> _BundleSource:
    """The ``_BundleSource`` for an input path: a directory, or a zip file; anything else is refused."""
    if path.is_dir():
        return _DirectorySource(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return _ZipSource(path)
    raise ConfigurationError(
        field="bundle_paths",
        message=f"{path} is neither a bundle directory nor a bundle zip file.",
        config_path=None,
    )


def _refuse_metadata_only_source(source: _BundleSource) -> None:
    """Refuse a bundle whose ROOT consolidated tree was carried without its chunk files.

    A store carried metadata-only (zarr metadata documents and no chunk files) re-opens
    cleanly and returns fill values for every data read, with no error. The combine reads
    ``max_wlevel_m`` and ``max_flow_cms`` out of the root tree, so a metadata-only child would
    yield ``identical: True, max_abs_diff: 0.0`` on every pair. The refusal reads the
    bundle's own manifest, so it can run over every input before any file of any bundle is
    written anywhere, and before the working folder exists.
    """
    try:
        shallow = json.loads(source.read_bytes(BUNDLE_MANIFEST_FILENAME)).get("metadata_only_stores", []) or []
    except (KeyError, ValueError):
        return  # no manifest, or an unreadable one: the bundle loader reports it with its own message
    carried = sorted(set(shallow) & set(ROOT_TREE_NAMES))
    if carried:
        raise ProcessingError(
            operation="combine: child store carried metadata-only",
            filepath=source.path,
            reason=(
                f"{source.path.name} names {carried[0]} in its {BUNDLE_MANIFEST_FILENAME} "
                "'metadata_only_stores' list, so the bundle carries its zarr metadata "
                "documents without any chunk files; reading it for data returns fill "
                "values rather than raising. Re-emit the child from a tree whose reporting "
                "set declares the store in full, or run the combine against the producer's "
                "full analysis directory."
            ),
        )


def _stage_sources(inputs: list[Path], crates_dir: Path) -> list[str]:
    """Put each input's render set under ``crates_dir/{experiment_id}/``; return the ids in input order.

    Inputs are taken one at a time: the source is opened, its render set is extracted, and
    nothing of it stays open before the next input begins. The
    extraction goes to a provisional folder first because the experiment id is read from the
    extracted ``cfg_analysis.yaml`` by the same loader every later step uses, and that
    loader needs a directory; the folder is then renamed (one rename, no copy) to the id.
    A repeated id gets the input's index as a suffix, as the combined manifest expects.
    """
    crates_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    used: set[str] = set()
    for i, path in enumerate(inputs):
        source = _open_bundle_source(path)
        names = source.names()
        incoming = crates_dir / f".incoming_{i}"
        source.extract(incoming, render_set(names))
        eid = _experiment_id(incoming)
        if eid in used:
            eid = f"{eid}__{i}"
        used.add(eid)
        incoming.rename(crates_dir / eid)
        ids.append(eid)
    return ids


def _check_root_tree(root: Path) -> None:
    """Refuse a child that ships no root consolidated tree, and prove the tree's root group opens.

    This replaces opening and stamping every child's whole tree (which the render never
    read): the existence gate (``FileNotFoundError`` when the bundle ships none of the root
    tree names) and a read of the root group alone.
    """

    store = _resolve_root_tree(root)
    open_store(str(store), kind="dataset", group="/", chunks=None).close()


def combine_bundle(
    bundle_paths: list[Path],
    output_path: Path | None = None,
    *,
    declare_solver_split: bool = False,
    workdir: Path | None = None,
    unzipped: bool = False,
) -> CombinedBundle:
    """Combine N completed render bundles into one standalone combined bundle.

    Ingests two or more **completed** render bundles from different experiments, each
    given as an unpacked bundle directory or as a bundle ``.zip`` file, and emits a single
    standalone bundle carrying one cross-experiment report. The pipeline is: (1) refuse any
    input whose root tree was carried metadata-only, then open each input in turn and extract
    its render set (everything except raw solver output) into a private working folder under
    ``child_crates/{experiment_id}/``; (2) check metadata compatibility across the extracted
    bundles and ABORT on any blocking divergence (an EXPERIMENT-identity field mismatch),
    then check solver provenance and that each ships a root consolidated tree; (3) write the
    read-models and render the ``combined`` reporting set against the working folder; (4)
    write the combined bundle at ``output_path``. No input is modified and the working folder
    is removed on success, on failure and on SIGTERM.

    The output is one zip by default (``{output_path}.zip``; an ``output_path`` that already ends
    in ``.zip`` is used as given). It holds the root files (crate, read-models, manifest, report,
    ``plots/``) and, per experiment, ``child_crates/{experiment_id}.zip`` plus the two re-rendered
    child figures as ``child_crates/{experiment_id}.refreshed/{path}``. With ``unzipped=True`` the
    output is a folder at ``output_path`` holding each experiment's full bundle under
    ``child_crates/{experiment_id}/``. An existing output is replaced, and a failed or killed
    combine leaves no loadable output.

    Both single-analysis bundles (which ship ``analysis_datatree.zarr`` at their
    root) and sensitivity-primary bundles (``sensitivity_datatree.zarr``) are
    accepted; the root tree is resolved by existence.

    Parameters
    ----------
    bundle_paths : list of Path
        Paths to at least two bundles: unpacked bundle directories or bundle ``.zip`` files.
    output_path : Path, optional
        Destination directory for the combined bundle. When omitted, a sibling of
        the first bundle named ``combined_{N}bundles_{toolkit_git_sha}`` is used.
    declare_solver_split : bool, optional
        Accept bundles pinned to different solver versions on purpose.
    workdir : Path, optional
        Directory under which the private working folder is created. Defaults to the
        system temporary directory.
    unzipped : bool, optional
        Write a folder instead of a zip.

    Returns
    -------
    CombinedBundle
        Consume-side handle for the emitted combined bundle: ``root`` is the ``.zip`` path and
        ``is_archive`` is true for a zipped output; ``root`` is the folder for ``unzipped=True``.

    Raises
    ------
    ConfigurationError
        If fewer than two bundles are supplied, if an input is neither a directory nor a
        zip, if the bundles carry a blocking compatibility divergence (they do not describe
        the same experiment family), or if bundles sharing one ``TRITONSWMM_branch_key``
        carry differing ``triton_producing_sha`` or commit-date root attrs (one pin label
        would cover two builds). The last check has no waiver; ``declare_solver_split`` does
        not apply to it.
    ProcessingError
        If an input's root consolidated tree was carried metadata-only.

    Examples
    --------
    >>> from hhemt.bundle import combine_bundle
    >>> combined = combine_bundle([Path("bundle_a"), Path("bundle_b")])  # doctest: +SKIP
    >>> combined.regenerate_report()  # doctest: +SKIP
    """
    inputs = sorted(Path(p).resolve() for p in bundle_paths)
    if len(inputs) < 2:
        raise ConfigurationError(
            field="bundle_paths",
            message=f"combine_bundle needs >=2 bundles, got {len(inputs)}.",
            config_path=None,
        )
    # Refuse a metadata-only input before anything is written: one pass over every input's own
    # manifest, ahead of the working folder, so a refused combine leaves no trace.
    for path in inputs:
        _refuse_metadata_only_source(_open_bundle_source(path))
    # The prefix names the SLURM job when there is one, so a folder left behind by a killed job
    # can be attributed to it.
    job = os.environ.get("SLURM_JOB_ID") or "local"
    unzipped = unzipped is True
    if output_path is None:
        output_path = inputs[0].parent / f"combined_{len(inputs)}bundles_{_get_toolkit_git_sha(strict=False)}"
    output_path = Path(output_path)
    if not unzipped and output_path.suffix != ".zip":
        output_path = output_path.with_name(output_path.name + ".zip")
    _sweep_stale_working_folders(workdir)
    working = Path(tempfile.mkdtemp(prefix=f"hhemt_combine_{job}_", dir=None if workdir is None else str(workdir)))
    partial = output_path.with_name(output_path.name + ".partial")
    previous = _install_sigterm_cleanup([working, partial])
    try:
        eids = _stage_sources(inputs, working / _CHILD_CRATES_SUBDIR)
        roots = [working / _CHILD_CRATES_SUBDIR / eid for eid in eids]
        report = check_bundle_compatibility(roots, declare_solver_split=declare_solver_split is True)
        if not report.is_compatible:
            blocking = "; ".join(
                f"{d.field_name} ({d.bucket}): {d.bundle_a}={d.value_a!r} vs {d.bundle_b}={d.value_b!r}"
                for d in report.blocking
            )
            _hint = ""
            if any(d.field_name == "TRITONSWMM_branch_key" for d in report.blocking):
                _hint = (
                    "\n\nThese bundles were built at DIFFERENT pinned solver versions. If that is "
                    "deliberate — e.g. a clean arm at a fix's ancestor against a resume arm at the fix — "
                    "re-run with `hhemt combine --declare-solver-split` (or "
                    "`combine_bundle(..., declare_solver_split=True)`). The divergence is still reported "
                    "and still appears in the combined report; declaring it only stops it aborting the "
                    "combine. Do NOT pass it to silence a split you did not intend: two arms at different "
                    "solvers answer different questions, and combining them unknowingly publishes a "
                    "silently-wrong figure."
                )
            raise ConfigurationError(
                field="bundle_paths",
                message=f"Bundles are not combine-compatible (blocking divergences): {blocking}{_hint}",
                config_path=None,
            )
        # Agreement gate, beside the compatibility gate and before the output path is touched: a
        # disagreement is a property of the input bundles, decidable from their root-tree attrs alone.
        # Placed here rather than in the read-model writer, which runs after the read-models start to
        # be written and would leave partly written files. No waiver flag.
        _solver_provenance(roots)
        for root in roots:
            _check_root_tree(root)
        member_zips = None if unzipped else _member_zips(inputs, eids, working)
        names = [f"{_CHILD_CRATES_SUBDIR}/{eid}" if unzipped else f"{_CHILD_CRATES_SUBDIR}/{eid}.zip" for eid in eids]
        _emit_combined_bundle(roots, None, report, working, child_names=names)
        if unzipped:
            _publish_folder(working, inputs, eids, output_path, partial)
        else:
            _publish_zip(working, eids, member_zips, output_path)
    finally:
        signal.signal(signal.SIGTERM, previous) if previous is not None else None
        shutil.rmtree(working, ignore_errors=True)  # EXEMPT-DU: transient-intermediate
    if unzipped:
        return CombinedBundle.from_directory(output_path)
    return CombinedBundle(output_path, is_archive=True)


def _install_sigterm_cleanup(paths: list[Path]):
    """Make SIGTERM remove ``paths`` and then take its normal course; return the previous handler.

    SLURM sends SIGTERM at a job's time limit, and the default action ends the process with no
    ``finally`` run, which would leave the working folder (a memory-backed ``/tmp`` on a compute
    node) behind. The handler removes the folders, restores the previous handler and sends the
    signal again, so the process still terminates as it would have. Returns ``None`` and installs
    nothing off the main thread, where Python does not allow a handler.
    """
    import threading

    if threading.current_thread() is not threading.main_thread():
        return None
    previous = signal.getsignal(signal.SIGTERM)

    def _handler(signum, _frame):
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)  # EXEMPT-DU: transient-intermediate
            else:
                path.unlink(missing_ok=True)  # EXEMPT-DU: transient-intermediate
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _handler)
    return previous


def _sweep_stale_working_folders(workdir: Path | None) -> None:
    """Remove ``hhemt_combine_*`` folders left by combines that can no longer be running.

    A folder is stale when its job id (the segment after the prefix) is not in ``squeue``, or,
    for a folder with no SLURM job (``local``) or when ``squeue`` cannot be read, when it is
    older than one day. A SIGKILL leaves a folder that no handler could remove, and this is
    what removes it at the next combine.
    """
    base = Path(workdir) if workdir is not None else Path(tempfile.gettempdir())
    prefix = "hhemt_combine_"
    try:
        candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    except OSError:
        return
    if not candidates:
        return
    live: set[str] | None
    try:
        out = subprocess.run(["squeue", "-h", "-o", "%i"], capture_output=True, text=True, timeout=10, check=True)
        live = set(out.stdout.split())
    except (OSError, subprocess.SubprocessError):
        live = None
    for path in candidates:
        job = path.name[len(prefix) :].split("_", 1)[0]
        if job.isdigit() and live is not None:
            stale = job not in live
        else:
            try:
                stale = time.time() - path.stat().st_mtime > 86400
            except OSError:
                continue
        if stale:
            shutil.rmtree(path, ignore_errors=True)  # EXEMPT-DU: transient-intermediate


def _zip_is_under_policy(path: Path) -> bool:
    """Whether every entry of the zip at ``path`` is compressed as the moderate policy says."""
    with zipfile.ZipFile(path) as zf:
        return all(
            info.compress_type == MODERATE.compress_type_for(info.filename)
            for info in zf.infolist()
            if not info.is_dir()
        )


def _member_zips(inputs: list[Path], eids: list[str], working: Path) -> list[Path]:
    """The zip that carries each member inside the combined zip: the input itself if it is a zip.

    A directory input has no zip, so its full bundle is written as one under the moderate policy
    (in a folder of the working folder that the published output skips), which makes every member
    a zip and lets the combined zip hold them all the same way.
    """
    out: list[Path] = []
    for path, eid in zip(inputs, eids, strict=True):
        if path.is_file():
            out.append(path)
            continue
        source = _DirectorySource(path)
        dest = working / _MEMBER_ZIP_DIR / f"{eid}.zip"
        dest.parent.mkdir(exist_ok=True)
        write_zip(((name, path / name) for name in source.names()), dest)
        out.append(dest)
    return out


class _StoredNamesPolicy(CompressionPolicy):
    """The moderate policy, except that the named archive entries are STORED.

    The named entries are member zips already compressed entry by entry; compressing them again
    would cost CPU for nothing and change the bytes of the zip inside.
    """

    stored_names: frozenset[str] = frozenset()

    def compress_type_for(self, arcname: str) -> int:
        if arcname in self.stored_names:
            return zipfile.ZIP_STORED
        return super().compress_type_for(arcname)


def _root_files(working: Path) -> list[Path]:
    """Every file of the working folder that belongs at the output's root, sorted."""
    skip = {_CHILD_CRATES_SUBDIR, _MEMBER_ZIP_DIR}
    found: list[Path] = []
    for dirpath, dirs, files in os.walk(working):
        if Path(dirpath) == working:
            dirs[:] = [d for d in dirs if d not in skip]
        found.extend(Path(dirpath) / f for f in files)
    return sorted(found)


def _refreshed_figures(working: Path, eid: str) -> list[str]:
    """The re-rendered child figures that exist for ``eid``, as paths relative to its folder."""
    child = working / _CHILD_CRATES_SUBDIR / eid
    return [rel for _module, rel in _PURE_DATA_CHILD_RENDERERS if (child / rel).is_file()]


def _publish_zip(working: Path, eids: list[str], member_zips: list[Path], output_path: Path) -> None:
    """Write the combined zip at ``output_path``; it replaces any existing file by rename on success."""
    entries: list[tuple[str, Path]] = [(f.relative_to(working).as_posix(), f) for f in _root_files(working)]
    stored: set[str] = set()
    for eid, member in zip(eids, member_zips, strict=True):
        arc = f"{_CHILD_CRATES_SUBDIR}/{eid}.zip"
        entries.append((arc, member))
        if _zip_is_under_policy(member):
            stored.add(arc)
        for rel in _refreshed_figures(working, eid):
            entries.append(
                (f"{_CHILD_CRATES_SUBDIR}/{eid}.refreshed/{rel}", working / _CHILD_CRATES_SUBDIR / eid / rel)
            )
    policy = _StoredNamesPolicy(level=MODERATE.level, store_zarr_chunks=MODERATE.store_zarr_chunks)
    object.__setattr__(policy, "stored_names", frozenset(stored))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_zip(entries, output_path, policy=policy)


def _copy_files(pairs: list[tuple[Path, Path]]) -> None:
    """Copy each ``(source, destination)`` file with ``copyfile`` on eight threads."""
    for parent in sorted({dest.parent for _src, dest in pairs}):
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(shutil.copyfile, src, dest) for src, dest in pairs]:
            future.result()


def _publish_folder(working: Path, inputs: list[Path], eids: list[str], output_path: Path, partial: Path) -> None:
    """Write the combined folder at ``output_path`` through ``{output}.partial``, then rename it into place.

    Each zip member is extracted whole into ``child_crates/{eid}/`` and each directory member is
    copied whole; the re-rendered child figures are then laid over them. An existing output is
    renamed aside before the new folder takes its name, and removed afterwards.
    """
    from hhemt.utils import fast_rmtree

    if partial.exists():
        fast_rmtree(partial)  # EXEMPT-DU: transient-intermediate
    partial.mkdir(parents=True)
    try:
        _copy_files([(f, partial / f.relative_to(working)) for f in _root_files(working)])
        for path, eid in zip(inputs, eids, strict=True):
            dest = partial / _CHILD_CRATES_SUBDIR / eid
            if path.is_file():
                extract_zip(path, dest)
            else:
                names = _DirectorySource(path).names()
                _copy_files([(path / n, dest / n) for n in names])
            _copy_files(
                [(working / _CHILD_CRATES_SUBDIR / eid / rel, dest / rel) for rel in _refreshed_figures(working, eid)]
            )
        aside = None
        if output_path.exists():
            aside = output_path.with_name(f"{output_path.name}.old_{os.getpid()}")
            os.rename(output_path, aside)
        os.rename(partial, output_path)
        if aside is not None:
            fast_rmtree(aside)  # EXEMPT-DU: bundle-root
    except BaseException:
        if partial.exists():
            fast_rmtree(partial)  # EXEMPT-DU: transient-intermediate
        raise


def _combined_experiment_ids(roots: list[Path]) -> list[str]:
    """Stable, unique per-experiment ids for the combined bundle.

    Uses each bundle's ``analysis_id`` (``_combine_merge._experiment_id``) with the same
    collision-suffix rule ``_stage_sources`` applies, so the manifest's ``experiment_ids``
    and the ``child_crates/{eid}`` directory names agree.
    """
    ids: list[str] = []
    used: set[str] = set()
    for i, r in enumerate(roots):
        eid = _experiment_id(r)
        if eid in used:
            eid = f"{eid}__{i}"  # collision-safe unique id (mirrors _stage_sources)
        used.add(eid)
        ids.append(eid)
    return ids


def _emit_combined_bundle(
    roots: list[Path],
    merged,
    report: CompatibilityReport,
    output_path: Path,
    tree_cache: dict[str, object] | None = None,
    child_names: list[str] | None = None,
) -> None:
    """Build the standalone combined bundle (flat hasPart-by-reference RO-Crate) in ``output_path``.

    ``combine_bundle`` calls this with the private working folder as ``output_path`` and
    publishes the finished folder afterwards, so the combine's output path is touched only
    after the render succeeds. A child already at ``output_path/child_crates/{experiment_id}``
    (which is where the combine extracts it) is used in place; a child anywhere else is copied
    there intact.

    Writes the combined ro-crate-metadata.json (root Dataset hasPart -> each child
    crate by reference, NOT N mainEntity — cardinality-1, A8), persists the
    CompatibilityReport as a read-model (combined_compatibility.json) for the
    Phase-4 renderer, writes bundle_manifest.json via a dedicated
    _write_combined_bundle_manifest (NOT verbatim _write_bundle_manifest, which is
    single-analysis-shaped — per CR5; the combined manifest carries experiment_ids /
    child_crates / bundle_schema_version / a deterministic combined_of and OMITS the
    volatile created_at_utc per CR4), and renders the combined report.
    ``merged`` is unused and kept only so existing callers' positional arguments still line
    up; pass ``None``. ``child_names`` are the archive-relative names the crate and manifest give
    the children (``child_crates/{eid}.zip`` for a zipped output); the default names the folders.
    The manifest is written after the report renders, so it never describes a bundle whose
    render failed.

    RENDER DISPATCH (F-I #4, Phase 5; Option B): the combined figures are NEW (do not
    pre-exist), so ``_render_combined_report`` direct-renders them in-process with a
    minimal ``_CombinedRenderContext`` (exposing only ``analysis_paths.analysis_dir ==
    output_path``), then emits a first-class Snakemake ``--report`` reusing the
    single-bundle machinery via ``combined_snakefile_generator``. The cross-experiment
    figures land under ``plots/cross_experiment/``; the generator ADDITIONALLY harvests each
    ``child_crates/{eid}/``'s already-rendered ``plots/`` figures in place as per-experiment
    ``report()`` sidebar sections, and the assembled report lands under
    ``analysis_report.{html,zip}`` (NO ``index.html``, NO per-child report regen).
    ``CombinedBundle.regenerate_report()`` re-invokes the same render path against the bundle
    root (the renderers read the on-disk read-models; the harvest re-globs ``child_crates/``,
    so there is no re-merge).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    experiment_ids = _combined_experiment_ids(roots)
    # Each child is opened at most once across both read-model writers.
    tree_cache = {} if tree_cache is None else tree_cache

    # 1. Each child sits intact under child_crates/{experiment_id}/ so it retains its own
    #    Workflow-Run-Crate (ADR-13 reconstitution; R7).
    crates_dir = output_path / _CHILD_CRATES_SUBDIR
    child_crates: list[str] = []
    for eid, root in zip(experiment_ids, roots, strict=True):
        dest = crates_dir / eid
        # The combine extracts each child straight to `dest`; copying a folder onto itself
        # would delete it first, so a child already there is left alone.
        if dest.exists() and dest.resolve() != Path(root).resolve():
            # EXEMPT-DU: bundle-root
            shutil.rmtree(dest)
        if not dest.exists():
            shutil.copytree(root, dest)
        child_crates.append(f"{_CHILD_CRATES_SUBDIR}/{eid}")
    if child_names is not None:
        child_crates = list(child_names)

    # 2. Combined crate + read-model + manifest (all byte-deterministic — CR4).
    _write_combined_rocrate(output_path, child_crates)
    _write_combined_compatibility(output_path, report)
    _write_combined_intercomparison(output_path, roots, tree_cache)
    # The cross-solver read-model, and with it `distinct_solver_count` -- the count the combined
    # report's cross-solver figure is gated on, by both the report rules and the direct
    # render in `_render_combined_report`. Written BEFORE the render step below
    # so a consumer on the generation path finds it, and left on disk afterwards so
    # `regenerate_report()` re-renders from it with no re-merge.
    _write_cross_solver_comparison(output_path, roots, tree_cache)
    # 3. Cross-experiment render (Phase-4 seam), then the manifest.
    _render_combined_report(merged, report, output_path)
    _write_combined_bundle_manifest(
        output_path,
        experiment_ids=experiment_ids,
        child_crates=child_crates,
        git_sha=_get_toolkit_git_sha(strict=False),
        roots=roots,
    )


def _open_child_tree(source: Path, eid: str, cache: dict[str, object] | None = None):
    """Open one child's ROOT consolidated tree (``source`` is the child's folder), once per combine.

    Both read-model writers read child stores through this one function, so a store that
    both need is opened a single time (they pass the same ``cache``), and the way a child's
    store is reached is a single site to change. ``eid`` is the cache key and names the child
    in an error. Returns the opened ``DataTree``, or ``None`` when the child ships no root tree.
    """

    from hhemt.utils import resolve_experiment_tree

    store = resolve_experiment_tree(source)
    if not store.exists():
        return None
    if cache is not None and eid in cache:
        return cache[eid]
    dt = open_store(str(store))
    if cache is not None:
        cache[eid] = dt
    return dt


def _write_combined_rocrate(output_path: Path, child_crates: list[str]) -> None:
    """Write the combined ro-crate-metadata.json (flat hasPart-by-reference).

    A base RO-Crate 1.2 + Provenance-Run-Crate whose root Dataset hasPart-references
    each intact child crate directory (NOT N mainEntity — A8), plus one combine
    ``CreateAction`` recording the operation. Serialized via the toolkit's canonical
    JSON-LD post-process (sorted @graph + sorted keys), with
    ``metadata._COMBINED_EMIT_STRIP_KEYS`` (notably ro-crate-py's per-run
    ``datePublished``) stripped so two combines are byte-identical (CR4).

    This tier's set is SEPARATE from the bundle tier's ``_BUNDLE_TRANSFER_STRIP_KEYS`` and
    must stay separate. A combined crate is a first publication of the combination, so it
    owes an authored ``datePublished`` under the RO-Crate MUST rather than an inherited
    one; stripping the key is what this tier does BEFORE it has an authored value to put
    there, and the two halves must be able to move independently of the bundle tier.
    """
    from rocrate.model.contextentity import ContextEntity
    from rocrate.rocrate import ROCrate

    from hhemt.metadata import _COMBINED_EMIT_STRIP_KEYS, canonical_jsonld_from_doc

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
        for volatile in _COMBINED_EMIT_STRIP_KEYS:
            entity.pop(volatile, None)
    (output_path / _COMBINED_ROCRATE_FILENAME).write_text(canonical_jsonld_from_doc(doc))


def _write_combined_compatibility(output_path: Path, report: CompatibilityReport) -> None:
    """Persist the CompatibilityReport as a deterministic JSON read-model.

    Mirrors the single-analysis validation_report.json pattern (primary (c)): the
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


#: Read-model filename the cross_experiment_intercomparison renderer projects.
_COMBINED_INTERCOMPARISON_FILENAME = "combined_intercomparison.json"

#: Key-result variables compared clean-vs-resume cross-bundle (mirrors
#: eda.cross_sim_identity.TRACKED_VARS): peak flood depth + peak conduit flow. Each var
#: maps to a TUPLE of candidate child node-paths tried in order — the coupled tier
#: (tritonswmm/*) first, the pure-TRITON tier (triton_only/triton) as fallback — because a
#: pure-TRITON primary stores max_wlevel_m under triton_only/triton, NOT tritonswmm/triton
#: (processing_analysis.py NODE_PATHS). max_flow_cms is coupled-only (no SWMM link tier on a
#: pure-TRITON primary), so its candidate tuple has the single coupled path and is simply
#: absent — and correctly skipped — for a pure-TRITON primary.
_INTERCOMPARISON_VARS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tritonswmm/triton", "triton_only/triton"), "max_wlevel_m"),
    (("tritonswmm/swmm_link",), "max_flow_cms"),
)


def _bundle_role_from_status(root: Path) -> str:
    """clean iff every bundled scenario_status.csv ``n_resumes`` == 0; resume iff any > 0.

    Combine-first: each experiment bundle is a single-arm primary (one all-clean sweep, one
    all-resume sweep). scenario_status.csv is bundled by ``_emit._copy_supporting_files``.
    Mirrors ``eda.compute_sensitivity``'s ``n_resumes`` clean/resume classifier convention.
    """
    import csv

    p = root / "scenario_status.csv"
    if not p.exists():
        return "clean"  # no resume evidence -> treat as clean
    max_r = 0
    try:
        with p.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    max_r = max(max_r, int(float(row.get("n_resumes") or 0)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return "clean"
    return "resume" if max_r > 0 else "clean"


# `_config_identity_from_node_attrs` was LIFTED to `eda.cross_sim_identity` so the WRITER of
# the pair records here and the RENDERER that joins against them share ONE definition. A
# second implementation drifting from this one would produce an empty join and a summary
# table of blank magnitude columns, raising nothing. Imported function-locally at its call
# site below.


def _load_intercomparison_subs(root: Path, cache: dict[str, object] | None = None) -> dict:
    """``{config_identity_str: {var_name: DataArray}}`` for one bundle's
    ``sensitivity_datatree.zarr``.

    Walks the same tree convention as ``eda._config_diff._load_subs`` (top-level groups
    ``/member_{id}``; the key-result arrays live under ``/member_{id}/tritonswmm/{triton,swmm_link}``).
    One representative sub per compute-config (sorted-first, deterministic).

    CORRECTED 2026-07-21 — the collapse is NO LONGER silent. This docstring previously
    asserted "replicates of one config are byte-identical, so a representative suffices."
    That premise is FALSE on a resume arm: measured, it holds 14/14 on the clean sweep but
    FAILS on 6 of 14 resume configs (gpu_1, gpu_3, gpu_4, hybrid_12, hybrid_13, mpi_10;
    max |delta| 5.9e-7 .. 1.2e-6 m). Because sorting puts ``_r1`` before ``_r2``, every
    ``_r2`` was discarded — so 10 of the 14 truncation-asymmetric members never
    reached the read-model, and nothing recorded that they had been dropped.

    The representative collapse is RETAINED (the pairing is per-config by construction),
    but the discarded ``member_id``s are now returned alongside the payload so a consumer can
    DISCLOSE the collapse rather than silently under-report. Keying on ``member_id`` and
    reporting replicate spread as a first-class column is the fuller fix and is tracked
    separately — it changes the read-model schema and every consumer of it.

    Returns ``(subs, collapsed)`` where ``collapsed`` maps ``{config_identity_str:
    [discarded member_id, ...]}``, empty when no replicate was dropped.
    """
    from hhemt.eda.cross_sim_identity import config_identity_from_node_attrs
    from hhemt.utils import resolve_experiment_tree

    store = resolve_experiment_tree(root)
    out: dict = {}
    collapsed: dict = {}
    if not store.exists():
        return out, collapsed
    # A child store carried SHALLOWLY (zarr metadata documents only, no chunk files) re-opens
    # cleanly and returns FILL VALUES for every data read -- measured, no raise. This function
    # reads max_wlevel_m/max_flow_cms out of the store below, so a shallow child would yield
    # `identical: True, max_abs_diff: 0.0` on every pair with no error and no warning. REFUSE
    # rather than skip: a skipped child yields an EMPTY pair list, which the roll-up renders as
    # "no pairs" -- the same false-clean shape the refusal exists to prevent.
    child_manifest = root / BUNDLE_MANIFEST_FILENAME
    if child_manifest.exists():
        try:
            shallow_stores = json.loads(child_manifest.read_text()).get("metadata_only_stores", [])
        except (OSError, ValueError):
            shallow_stores = []
        if store.relative_to(root).as_posix() in shallow_stores:
            raise ProcessingError(
                operation="combine: child store carried metadata-only",
                filepath=store,
                reason=(
                    f"{root.name} names this store in its {BUNDLE_MANIFEST_FILENAME} "
                    "'metadata_only_stores' list, so the bundle carries its zarr metadata "
                    "documents without any chunk files; reading it for data returns fill "
                    "values rather than raising. Re-emit the child from a tree whose reporting "
                    "set declares the store in full, or run the combine against the producer's "
                    "full analysis directory."
                ),
            )
    dt = _open_child_tree(root, root.name, cache)
    for g in sorted(dt.groups):
        if g.count("/") != 1 or not g.startswith("/member_"):
            continue
        cfg_key = config_identity_from_node_attrs(dict(dt[g].attrs))
        if cfg_key in out:
            # First (sorted) representative per config wins. RECORD the discard — a
            # replicate dropped without a trace is how the resume-arm spread went
            # unreported (see docstring).
            collapsed.setdefault(cfg_key, []).append(g.lstrip("/"))
            continue
        vars_by_name: dict = {}
        for candidate_children, var in _INTERCOMPARISON_VARS:
            for child in candidate_children:
                try:
                    child_node = dt[f"{g}/{child}"]
                except KeyError:
                    continue
                if var in child_node.data_vars:
                    vars_by_name[var] = child_node[var]
                    break  # first present candidate wins (coupled tier before pure-TRITON)
        if vars_by_name:
            out[cfg_key] = vars_by_name
    return out, collapsed


def _write_combined_intercomparison(
    output_path: Path, roots: list[Path], cache: dict[str, object] | None = None
) -> None:
    """Derive clean-vs-resume per-compute-config identity CROSS-BUNDLE into
    ``combined_intercomparison.json`` (the read-model the ``cross_experiment_intercomparison``
    renderer projects).

    Combine-first (the SOLE clean-vs-resume vehicle): the two bundles are single-arm primaries —
    one all-clean (``n_resumes==0``), one all-resume (``n_resumes>0``). Classify each bundle
    from its bundled ``scenario_status.csv``, pair each compute-config present in BOTH (keyed on
    the run_mode/n_mpi/n_omp/n_gpus/n_nodes/partition identity), and compare each key-result
    summary (``max_wlevel_m`` + ``max_flow_cms``) per event via ``compare_variable_exact`` (the
    Phase-1 compliant kernel: ``xr.align`` exact + transpose + dtype gate + ``equal_nan``; the
    primary Assumption establishes consolidation is value-preserving, so reading the consolidated
    tree here is stipulation-compliant). Records ``{config, variable, event_iloc, identical,
    max_abs_diff}``. Read-only over the child bundles; deterministic ordering. Does NOT read any
    ``eda_resume_sensitivity.verdict.json`` (a single-arm primary skips that member and writes no
    verdict). Writes an EMPTY-but-honest ``pairs`` list when the two bundles share no
    compute-config, a bundle carries no consolidated tree, or the pair is not clean+resume.
    """
    from hhemt.eda.cross_sim_identity import compare_variable_exact

    experiment_ids = _combined_experiment_ids(roots)
    roles = [_bundle_role_from_status(root) for root in roots]
    experiments = [{"experiment": eid, "role": role} for eid, role in zip(experiment_ids, roles, strict=True)]

    # X2 (iter-3): pair WITHIN each model so combined_intercomparison.json carries BOTH
    # TRITON-SWMM and TRITON pairs; the cross_experiment_intercomparison renderer already
    # groupBy: "model" (Q6b), so this yields a per-model section pair with NO renderer change.
    # Supersedes the clean[0]/resume[0] single-model pairing (the "multi-model pairing" the
    # retired comment named). Fungible: identical compare_variable_exact kernel per model.
    _MODEL_BY_TOKEN = (("_tritonswmm", "TRITON-SWMM"), ("_triton", "TRITON"))

    def _model_of_root(root: Path) -> str:
        eid = _combined_experiment_ids([root])[0]
        for _tok, _label in _MODEL_BY_TOKEN:
            if eid.endswith(_tok):
                return _label
        return ""

    pairs: list[dict] = []
    _clean_collapsed_all: dict = {}
    _resume_collapsed_all: dict = {}
    _by_model_role: dict[str, dict[str, list[Path]]] = {}
    for _root, _role in zip(roots, roles, strict=True):
        _by_model_role.setdefault(_model_of_root(_root), {}).setdefault(_role, []).append(_root)
    for _model, _rr in sorted(_by_model_role.items()):
        _cl, _rs = _rr.get("clean", []), _rr.get("resume", [])
        if not (_cl and _rs):
            continue
        clean_subs, clean_collapsed = _load_intercomparison_subs(_cl[0], cache)
        resume_subs, resume_collapsed = _load_intercomparison_subs(_rs[0], cache)
        _clean_collapsed_all.update(clean_collapsed)
        _resume_collapsed_all.update(resume_collapsed)
        for cfg_key in sorted(set(clean_subs) & set(resume_subs)):
            c_vars, r_vars = clean_subs[cfg_key], resume_subs[cfg_key]
            for _child, var in _INTERCOMPARISON_VARS:
                c_da, r_da = c_vars.get(var), r_vars.get(var)
                if c_da is None or r_da is None or "event_iloc" not in c_da.dims:
                    continue
                for e in [int(v) for v in c_da["event_iloc"].values]:
                    try:
                        cmp = compare_variable_exact(c_da.sel(event_iloc=e), r_da.sel(event_iloc=e))
                    except (ValueError, KeyError):
                        continue
                    mad = cmp["max_abs_diff"]
                    pairs.append(
                        {
                            "config": cfg_key,
                            "variable": var,
                            "event_iloc": int(e),
                            "identical": bool(cmp["identical"]),
                            "model": _model,
                            # NaN (incomparable coords) -> None so the JSON is valid + honest.
                            "max_abs_diff": (None if mad != mad else float(mad)),
                        }
                    )
    payload = {
        "experiments": sorted(experiments, key=lambda x: x["experiment"]),
        "pairs": sorted(pairs, key=lambda p: (p.get("model", ""), p["config"], p["variable"], p["event_iloc"])),
        # Replicate collapse is DISCLOSED, not silent. `pairs` above carries one
        # representative sub per compute-config; any replicate dropped to get there is
        # listed here so a renderer can state the collapse. The premise that replicates
        # are byte-identical is FALSE on a resume arm (measured 6 of 14 configs differ),
        # so an undisclosed collapse under-reports the very spread a resume-validity
        # experiment exists to measure. See _load_intercomparison_subs.
        "replicates_collapsed": {
            "clean": {k: sorted(v) for k, v in sorted(_clean_collapsed_all.items())},
            "resume": {k: sorted(v) for k, v in sorted(_resume_collapsed_all.items())},
        }
        if (_clean_collapsed_all or _resume_collapsed_all)
        else {},
    }
    (output_path / _COMBINED_INTERCOMPARISON_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


#: Model arm -> the consolidated-tree PERFORMANCE group that arm writes, keyed exactly as
#: ``processing_analysis._MODE_TO_TREE_PATH`` writes it. The arm is taken from which group
#: exists rather than from an analysis-id suffix: the suffix is a naming convention of the analysis
#: id that may change, while the group path is written by the consolidation itself.
_PERFORMANCE_GROUP_BY_ARM: dict[str, str] = {
    "tritonswmm": "tritonswmm/performance",
    "triton": "triton_only/performance",
}


def _solver_branch_key(root: Path) -> str:
    """The bundle's pinned solver ref, or ``""`` when it carries none.

    Read from the SAME field the combine's compatibility gate compares
    (``TRITONSWMM_branch_key`` in the bundled ``cfg_system.yaml``), so the read-model's solver
    identity cannot disagree with the gate that decided these bundles were compatible.

    The value is a git ref, not the identity token Decision 4 specifies for the estate
    (``{sha8}_{YYYYMMDD}``, package 2a.1, not yet landed), which the estate will pass as the
    ``variant`` infix on the analysis NAME: the toolkit passes this ref to ``git checkout`` when it
    builds a solver. A reader expecting the token here would find a bare sha; see the read-model's
    ``solver_format`` key, which states which one this writer found rather than leaving the
    consumer to guess.
    """
    sysd = _load_yaml(root / _CFG_SYSTEM_FILENAME)
    return str(sysd.get("TRITONSWMM_branch_key") or "").strip()


#: Root attrs `processing_analysis._stamp_triton_provenance` stamps on the consolidated tree from the
#: compile-time system log: the solver clone's HEAD sha and its `YYYYMMDD` committer date.
_ROOT_SHA_ATTR = "triton_producing_sha"
_ROOT_DATE_ATTR = "triton_producing_commit_date"


def _usable_date(raw: object) -> str | None:
    """``raw`` as a real ``YYYYMMDD`` calendar date, or ``None``.

    A date that is present but not an 8-digit calendar date is treated as NOT captured rather than
    sorted as text: the figure is to order its series by this value (chunk 1b.4, not yet landed), and a
    malformed string would take a position set by its characters instead of by when the commit happened.
    """
    from datetime import datetime

    text = str(raw or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return None
    return text


def _root_provenance_attrs(root: Path) -> tuple[str | None, str | None]:
    """``(triton_producing_sha, commit date)`` from one bundle's ROOT consolidated tree, else ``None``s.

    Reads only the root group's attrs, so it works on a store carried metadata-only (attrs are zarr
    metadata, never chunks) and opens no member. A bundle with no tree, an unreadable root, or an
    unstamped root yields ``(None, None)``: absence is the normal state for every bundle consolidated
    before the capture existed, a container-mode bundle (no clone to read) and an intermittent
    capture miss, and callers treat it as "not captured", never as a value.
    """

    from hhemt.utils import resolve_experiment_tree

    store = resolve_experiment_tree(root)
    if not store.exists():
        return None, None
    try:
        attrs = dict(open_store(str(store), kind="dataset", group="/", chunks=None).attrs)
    except Exception:  # noqa: BLE001 -- an unreadable root is "not captured", not a combine failure
        return None, None
    sha = str(attrs.get(_ROOT_SHA_ATTR) or "").strip() or None
    return sha, _usable_date(attrs.get(_ROOT_DATE_ATTR))


def _solver_provenance(roots: list[Path]) -> dict[str, dict]:
    """Aggregate each SOLVER's provenance across its bundles; RAISE when one pin names two builds.

    A solver is one ``TRITONSWMM_branch_key``, and one solver is normally two bundles (one per
    model arm). The commit date belongs to the commit, and both arms built the same commit, so a
    capture missing from one arm is ABSORBED by the other. A solver is undated only when none of
    its bundles carries the date.

    Two *present* values that differ -- for the date or for ``triton_producing_sha`` -- mean one pin
    label covers two builds, and the function raises ``ConfigurationError`` naming both bundles and
    both values instead of picking one. A value present in one bundle and absent in another is not
    a disagreement. There is deliberately NO waiver: ``declare_solver_split`` exists because a set
    of DIFFERENT pins can be a deliberate design, while one pin covering two builds cannot be, and a
    waiver would publish one series that silently mixes them. Reads root-tree attrs only and writes
    nothing, so a caller can run it before any output path exists.

    Returns ``{pin: {"date", "label", "triton_producing_sha", "dated_bundles", "undated_bundles"}}``
    in sorted-pin order. ``label`` is the first 8 characters of the pin, then ``_`` and the date
    when dated: the form Decision 4 specifies for the estate's ``_identity_token(pin)`` (package
    2a.1, not yet landed), so a figure label and an experiment-name infix agree once it lands. A
    display value only; the cell identity stays the pin.
    """
    rows: dict[str, list[tuple[str, str | None, str | None]]] = {}
    for root in roots:
        pin = _solver_branch_key(root)
        if not pin:
            continue
        sha, date = _root_provenance_attrs(root)
        rows.setdefault(pin, []).append((root.name, sha, date))

    out: dict[str, dict] = {}
    problems: list[str] = []
    for pin, bundles in sorted(rows.items()):
        agreed: dict[str, str | None] = {}
        for kind, idx in (("triton_producing_sha", 1), ("commit date", 2)):
            present = [(name, row[idx]) for row in bundles for name in (row[0],) if row[idx] is not None]
            values = sorted({v for _, v in present})
            if len(values) > 1:
                shown = "; ".join(f"{name}={value!r}" for name, value in present)
                problems.append(f"solver {pin!r} carries differing {kind} values ({shown})")
            agreed[kind] = values[0] if values else None
        date = agreed["commit date"]
        out[pin] = {
            "date": date,
            "label": f"{pin[:8]}_{date}" if date else pin[:8],
            "triton_producing_sha": agreed["triton_producing_sha"],
            "dated_bundles": sorted(name for name, _, d in bundles if d is not None),
            "undated_bundles": sorted(name for name, _, d in bundles if d is None),
        }
    if problems:
        raise ConfigurationError(
            field="bundle_paths",
            message=(
                "Bundles that share one TRITONSWMM_branch_key were built from different solver commits: "
                + " | ".join(problems)
                + ". One pin label would cover two builds, so the combine refuses rather than choosing "
                "one; this check has no waiver (--declare-solver-split does not apply). Re-run the "
                "disagreeing arm at the intended pin."
            ),
            config_path=None,
        )
    return out


def _solver_order(info: dict[str, dict]) -> list[str]:
    """Pins in the order the figure is to show them: dated ascending by date, then undated by identity.

    Ground for undated-last: a dated series' position is an age claim, so the dated series stay
    contiguous from the oldest and no undated series sits among them. Ties on date fall to the pin.
    """
    dated = sorted((pin for pin, v in info.items() if v["date"]), key=lambda pin: (info[pin]["date"], pin))
    undated = sorted(pin for pin, v in info.items() if not v["date"])
    return dated + undated


def _member_wallclock(dt, member_group: str) -> tuple[str, float | None]:
    """``(model arm, wall-clock seconds)`` for one member node, or ``("", None)``.

    ``Total`` is REDUCED BY MAX over whatever dims survive. The performance summary is already
    summed over ``timestep_min`` by ``_export_performance_summary``, so a member carrying one
    simulation has nothing left to reduce and the max is the identity. A member carrying
    several events would otherwise have their wall-clocks SUMMED, which reports the number of
    events as cost; max answers "the slowest single simulation at this config", which is the
    quantity that is comparable across solvers.

    A non-finite result is returned as ``None`` rather than as a number, because zarr's fill
    value for a float variable is NaN and reporting it as a measurement would be the exact
    "ran and produced nothing" ambiguity the figure's ``not_run`` marker exists to remove.
    """
    import numpy as np

    for arm, rel in _PERFORMANCE_GROUP_BY_ARM.items():
        try:
            node = dt[f"{member_group}/{rel}"]
        except KeyError:
            continue
        if "Total" not in node.data_vars:
            continue
        values = np.asarray(node["Total"].values, dtype=float).ravel()
        finite = values[np.isfinite(values)]
        return arm, (float(finite.max()) if finite.size else None)
    return "", None


def _write_cross_solver_comparison(
    output_path: Path, roots: list[Path], cache: dict[str, object] | None = None
) -> None:
    """Derive the cross-SOLVER read-model into ``cross_solver_comparison.json``.

    WHY A COMBINE-TIME READ-MODEL AND NOT A RENDER-TIME SCAN. The figure's question -- did the
    solver change move the measurement? -- spans bundles, so no single child can answer it and
    the merge is the only place holding them together. Deriving it here also keeps
    ``CombinedBundle.regenerate_report()`` cheap: a re-render re-reads this file and re-merges
    nothing.

    TWO OUTPUTS IN ONE FILE, because they share a lifetime and must never disagree.
    ``cells`` is the figure's data, one ``(solver, model_arm, config)`` cell each.
    ``distinct_solver_count`` is the predicate package 1b's Definition of Done turns on (the
    figure must be inert on a single-solver combine). It is published IN THIS ARTIFACT rather
    than returned in memory because ``regenerate_report()`` re-invokes the render path with no
    re-merge, so an in-memory channel would be missing on exactly the re-render path that must
    also stay inert. The consuming chunk (1b.4, not yet landed) is to skip the
    ``cross_solver_comparison`` selection when this count is below 2.

    ``distinct_solver_count`` COUNTS DISTINCT BUNDLE SOLVER REFS, not cells, so it is written
    even when no cell could be derived. A single-solver combine must still produce the file:
    to a consumer that checks only for the file's presence, an absent file and a count of two
    are indistinguishable, and the failure would be a vacuously rendered figure rather than a
    missing one.

    Degradation is per bundle and RECORDED, never silent. A bundle with no solver ref, no
    consolidated tree, or a metadata-only store contributes its solver to the count and no
    cells, and the reason lands in ``skipped_bundles`` so a reader of the read-model learns
    why instead of reading an omission as an absence of simulations. The delivered renderer does
    not display that key.
    """
    from hhemt.eda.cross_sim_identity import config_identity_from_node_attrs
    from hhemt.report_renderers.cross_solver_comparison import READ_MODEL_FILENAME
    from hhemt.utils import resolve_experiment_tree

    solvers = [_solver_branch_key(root) for root in roots]
    # Per-solver date/label, aggregated over each solver's bundles. Raises on a disagreement, so a
    # direct caller cannot publish a read-model whose solver silently mixes two builds; the combine
    # entry point has already run this same call before any write (see `combine_bundle`).
    solver_info = _solver_provenance(roots)
    distinct = _solver_order(solver_info)

    cells: list[dict] = []
    skipped: list[dict] = []
    collapsed: dict[str, list[str]] = {}
    arms: set[str] = set()
    configs: set[str] = set()

    for root, solver in zip(roots, solvers, strict=True):
        if not solver:
            skipped.append({"bundle": root.name, "reason": "no TRITONSWMM_branch_key in cfg_system.yaml"})
            continue
        store = resolve_experiment_tree(root)
        if not store.exists():
            skipped.append({"bundle": root.name, "reason": "no consolidated tree"})
            continue
        # A store carried SHALLOWLY (zarr metadata documents only) re-opens cleanly and yields
        # FILL VALUES on every data read rather than raising. `_load_intercomparison_subs` REFUSES
        # that case because fill values there produce a false "identical" verdict; here a fill
        # value would produce a false MEASUREMENT, so the bundle is skipped with its reason
        # recorded instead of contributing a cell.
        manifest = root / BUNDLE_MANIFEST_FILENAME
        if manifest.exists():
            try:
                shallow = json.loads(manifest.read_text()).get("metadata_only_stores", []) or []
            except (OSError, ValueError):
                shallow = []
            if store.relative_to(root).as_posix() in shallow:
                skipped.append({"bundle": root.name, "reason": "consolidated tree carried metadata-only"})
                continue

        dt = _open_child_tree(root, root.name, cache)
        for g in sorted(dt.groups):
            if g.count("/") != 1 or not g.startswith("/member_"):
                continue
            arm, seconds = _member_wallclock(dt, g)
            if not arm:
                continue
            cfg_key = config_identity_from_node_attrs(dict(dt[g].attrs))
            if any(c["solver"] == solver and c["model_arm"] == arm and c["config"] == cfg_key for c in cells):
                # First (sorted) representative per (solver, arm, config) wins; the discarded
                # member ids are recorded rather than dropped, mirroring
                # `_load_intercomparison_subs`, because an undisclosed replicate collapse
                # under-reports the very spread a comparison exists to show.
                collapsed.setdefault(f"{solver}|{arm}|{cfg_key}", []).append(g.lstrip("/"))
                continue
            arms.add(arm)
            configs.add(cfg_key)
            cells.append(
                {
                    "solver": solver,
                    "model_arm": arm,
                    "config": cfg_key,
                    "status": "ok" if seconds is not None else "not_run",
                    "value": seconds,
                }
            )

    payload: dict = {
        "metric": "wallclock_seconds",
        "metric_label": "Wall-clock (s), max over a member's simulations",
        # Which identity the `solver` field carries, stated rather than left for a consumer to
        # infer: this writer emits whatever `TRITONSWMM_branch_key` holds, which is the pinned
        # git ref and NOT the experiment token Decision 4 specifies for the estate
        # (`{sha8}_{YYYYMMDD}`, package 2a.1, not yet landed).
        "solver_format": "pin",
        # Intended figure order: dated solvers ascending by date, then undated by identity. The
        # delivered renderer does NOT read `solver_info` or this list's position and re-sorts by
        # parsing each token; ordering by `solver_info` is chunk 1b.4's change (not yet landed).
        # The list is written in the intended order so a reader of the file sees that sequence.
        "solvers": distinct,
        # Per solver: its commit date (or None when no bundle of that solver carried it), its legend
        # label, and which bundles did and did not carry the date. The cell identity above stays the
        # pin for every cell, so no capture state can change a solver's series or `not_run` count.
        "solver_info": {pin: solver_info[pin] for pin in distinct},
        "model_arms": sorted(arms),
        "configs": sorted(configs),
        "distinct_solver_count": len(distinct),
        "cells": sorted(cells, key=lambda c: (c["model_arm"], c["config"], c["solver"])),
    }
    if skipped:
        payload["skipped_bundles"] = skipped
    if collapsed:
        payload["replicates_collapsed"] = {k: v for k, v in sorted(collapsed.items())}
    (output_path / READ_MODEL_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_combined_bundle_manifest(
    output_path: Path,
    *,
    experiment_ids: list[str],
    child_crates: list[str],
    git_sha: str,
    roots: list[Path],
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
    from hhemt.bundle import _dependency  # function-local: avoids a top-level bundle import cycle

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "combined": True,
        "toolkit_git_sha": git_sha,
        # The three-field stamp, matching the single-bundle sibling. `toolkit_git_sha`
        # above is NOT this stamp under another name: it is one field where the stamp is
        # three, and it carries no version and no dirty flag.
        **producing_stamp(),
        "experiment_ids": list(experiment_ids),
        "child_crates": list(child_crates),
        # Lineage stamp (CR4-deterministic; no wall-clock): per-child {experiment_id, role, identity}
        # so the combined bundle self-documents WHICH clean + WHICH resume experiment (case/sha/role)
        # it was built from — the reproducibility-from-one-entry provenance surface.
        "combined_of": [
            {
                "experiment_id": eid,
                "role": _dependency.classify_bundle_role(root),
                "identity": _dependency.read_bundle_identity(root).model_dump(exclude_none=True),
            }
            for eid, root in zip(experiment_ids, roots, strict=True)
        ],
    }
    (output_path / BUNDLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class _CombinedRenderContext:
    """Minimal render-context for the emit-time combined render.

    Exposes ONLY ``analysis_paths.analysis_dir == output_path`` (NOT a
    BundleableAnalysis, which also requires ``_system`` / ``cfg_analysis`` the
    combined render has no single value for — see the RENDER DISPATCH note).
    """

    analysis_dir: Path

    @property
    def analysis_paths(self) -> SimpleNamespace:
        return SimpleNamespace(analysis_dir=self.analysis_dir)


# Pure-data child report renderers: read ONLY bundle-local files (no TRITONSWMM_system),
# so they can be re-rendered at combine time to refresh a child's harvested-in-place figure
# with the CURRENT toolkit's renderer code (e.g. b4's n_resumes column) even on a SCRUBBED
# child whose cfg_system carries null software dirs. (renderer_module, output_relpath).
_PURE_DATA_CHILD_RENDERERS: tuple[tuple[str, str], ...] = (
    ("scenario_status_appendix", "plots/appendix/scenario_status.html"),
    ("errors_and_warnings", "plots/errors_and_warnings/validation_report.html"),
)


@dataclass(frozen=True)
class _ChildReportRenderContext:
    """System-free render context for a child bundle's pure-data report figures.

    Mirrors load_eda_context's config-only load (yaml_to_model, NO TRITONSWMM_system):
    a scrubbed cfg_system (null software dirs) is never constructed, so the pure-data
    renderers run bundle-locally. Exposes exactly the attributes scenario_status_appendix
    and errors_and_warnings read: analysis_paths.analysis_dir, analysis_id, cfg_analysis.
    """

    analysis_dir: Path
    cfg_analysis: object

    @property
    def analysis_paths(self) -> SimpleNamespace:
        return SimpleNamespace(analysis_dir=self.analysis_dir)

    @property
    def analysis_id(self) -> str:
        return str(getattr(self.cfg_analysis, "analysis_id", "") or "")


def _rerender_child_report_figures(bundle_root: Path) -> None:
    """Re-render each child bundle's pure-data report figures IN PLACE at combine time.

    Refreshes the harvested-in-place child figures (scenario_status appendix +
    errors_and_warnings) with the current toolkit's renderer code, so the ONE combined
    --report reflects e.g. b4's n_resumes column regardless of when the child bundle was
    produced. Reads ONLY bundle-local files (no TRITONSWMM_system) via a config-only
    context, so a scrubbed child (null software dirs) re-renders cleanly. Best-effort per
    (child, renderer): a re-render failure leaves the child's already-rendered figure in
    place (never worse than the harvest-in-place baseline). CR4-safe: reached only through
    _render_combined_report, which the byte-identity test monkeypatches to a no-op.
    """
    import importlib

    from hhemt.config.analysis import analysis_config

    crates = bundle_root / _CHILD_CRATES_SUBDIR
    if not crates.exists():
        return
    for child in sorted(p for p in crates.iterdir() if p.is_dir()):
        cfg_path = child / "cfg_analysis.yaml"
        if not cfg_path.exists():
            continue
        try:
            # report_config is itself a cfgBaseModel, so the armed "*" existence
            # validator DOES run on it; it is safe ONLY because report_config carries
            # ZERO Path fields TODAY. analysis_config.brand_theme shows that shape is
            # already live one model over, so ADDING a Path field to report_config
            # invalidates this reasoning at that moment.
            # SILENT-FAILURE WARNING: this load sits inside its OWN try whose except is
            # the very next statement, so a raising reader skips the CHILD WHOLESALE and
            # never reaches the per-renderer except-continue further down. Either way the
            # STALE HARVESTED figure survives and every gate stays green. Verify by
            # asserting the re-render HAPPENED, never by the suite staying green.
            from hhemt.bundle._path_policy import load_bundle_config

            cfg_analysis = load_bundle_config(cfg_path, analysis_config, child)
        except Exception:
            continue
        ctx = _ChildReportRenderContext(analysis_dir=child, cfg_analysis=cfg_analysis)
        for renderer_module, out_rel in _PURE_DATA_CHILD_RENDERERS:
            out_path = child / out_rel
            if not out_path.exists():
                continue  # child's active set did not emit this figure
            try:
                module = importlib.import_module(f"hhemt.report_renderers.{renderer_module}")
                module.render(ctx, cfg_analysis.report, out_path)
            except Exception:
                continue  # leave the already-rendered figure in place


def _render_combined_report(merged, report, output_path: Path, *, formats: tuple[str, ...] = ("html", "zip")) -> None:
    """Render the combined cross-experiment figures, then emit a REAL Snakemake
    ``--report`` (first-class chrome/nav/CSS), reusing the single-bundle machinery.

    Step 1 direct-renders the two ``combined`` ReportingSet figures — they are NEW
    (do not pre-exist), unlike ``Bundle.regenerate_report`` whose figures are bundled
    — writing ``plots/cross_experiment/{compatibility,intercomparison}.html`` plus
    their manifest sidecars. Step 2 hands off to
    ``render_combined_report_via_snakemake``: stage report.css + captions + the
    combined workflow-description, write the combined regeneration Snakefile,
    ``snakemake --touch`` the pre-rendered figures, then ``snakemake --report`` +
    React surgery per requested format. ``merged`` / ``report`` stay unused (the
    renderers read the on-disk read-models), so ``CombinedBundle.regenerate_report``
    re-invokes with ``None, None`` and no re-merge.
    """
    import importlib

    from hhemt.bundle.combined_snakefile_generator import (
        _active_combined_selections,
        render_combined_report_via_snakemake,
    )
    from hhemt.config.report import DEFAULT_REPORT_CONFIG

    # Option B: NO child-report regen and NO index.html front door. The per-experiment
    # sidebar sections harvest each child's ALREADY-rendered plots/ figures in place
    # (combined_snakefile_generator._harvest_per_experiment_rule_specs), so the ONE
    # combined --report carries every experiment's content natively.
    ctx = _CombinedRenderContext(analysis_dir=output_path)
    # The selections come from the generator's helper rather than the raw registry so a
    # conditional figure (the cross-solver one) is NOT written to disk when its predicate is
    # false; the report rules read the same helper, so files and rules cannot disagree.
    for sel in _active_combined_selections(output_path):
        for tmpl in sel.rule_spec_template:
            module = importlib.import_module(f"hhemt.report_renderers.{tmpl.renderer_module}")
            rel = tmpl.output_path_template.replace("__OUTPUT_EXT__", ".html")
            fig_path = output_path / rel
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            module.render(ctx, DEFAULT_REPORT_CONFIG, fig_path)
    # F1: refresh each child's pure-data report figures (scenario_status appendix +
    # errors_and_warnings) with the current toolkit's renderers BEFORE the harvest
    # --touch indexes them, so b4's n_resumes column (and any current renderer change)
    # reaches the combined report even on a scrubbed child. System-free; CR4-safe (this
    # whole function is monkeypatched out in the byte-identity test).
    _rerender_child_report_figures(output_path)
    render_combined_report_via_snakemake(output_path, formats=formats)


#: Record written beside an unpacked zip that says which zip it came from, so a repeat unpack can
#: tell the same file (reuse what is there) from a different file of the same name (start over).
_UNPACK_RECORD = ".hhemt_unpack_source.json"


def _zip_identity(zip_path: Path) -> dict[str, int]:
    """The size and modification time of ``zip_path``: what makes it the same file again."""
    st = zip_path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _prepare_unpack_dir(zip_path: Path, unpack_dir: Path) -> bool:
    """Decide whether ``unpack_dir`` can be reused for ``zip_path``; clear it when it cannot.

    Reuse needs the record this module writes after an unpack and that record naming this exact
    zip (size and mtime). A missing or unreadable record, or a record of another file, means
    the folder's contents say nothing reliable about this zip (a replacement zip of the same
    name would otherwise keep every old file of the same size), so the folder is removed.
    Returns True when the folder is kept.
    """
    from hhemt.utils import fast_rmtree

    record = unpack_dir / _UNPACK_RECORD
    if unpack_dir.is_dir() and record.is_file():
        try:
            if json.loads(record.read_text()) == _zip_identity(zip_path):
                return True
        except (OSError, ValueError):
            pass
    if unpack_dir.exists():
        fast_rmtree(unpack_dir)  # EXEMPT-DU: bundle-root
    return False


def _write_unpack_record(zip_path: Path, unpack_dir: Path) -> None:
    """Record, in ``unpack_dir``, the identity of the zip it was unpacked from."""
    (unpack_dir / _UNPACK_RECORD).write_text(json.dumps(_zip_identity(zip_path)))


def unpack_member_zip(zip_path: Path) -> Path:
    """Unpack a member bundle zip into ``{zip stem}/`` beside it and return that folder.

    Files are created on eight threads. A repeat call into a folder this function unpacked from
    the same zip re-extracts only files that are missing or whose size differs from the zip's;
    a record absent, or a different zip of the same name, starts from an empty folder. The
    record is written last, so a call that failed part way never counts as an unpack.
    """
    zip_path = Path(zip_path).resolve()
    unpack = zip_path.parent / zip_path.stem
    reuse = _prepare_unpack_dir(zip_path, unpack)
    extract_zip(zip_path, unpack, only_missing_or_resized=reuse)
    _write_unpack_record(zip_path, unpack)
    return unpack


def is_combined_manifest(manifest: dict) -> bool:
    """Whether a parsed ``bundle_manifest.json`` is a combined bundle's (it carries ``child_crates``)."""
    return "child_crates" in manifest


class CombinedBundle:
    """Consume-side handle for a standalone combined bundle (mirrors Bundle)."""

    def __init__(self, root: Path, *, is_archive: bool = False) -> None:
        self._root = root.resolve()
        self._is_archive = is_archive

    @property
    def is_archive(self) -> bool:
        """True when ``root`` is the combined ``.zip`` file rather than an unpacked folder."""
        return self._is_archive

    @classmethod
    def from_directory(cls, path: Path | str) -> CombinedBundle:
        """Construct a :class:`CombinedBundle` from an unpacked combined-bundle directory.

        The combined-bundle counterpart of :meth:`hhemt.bundle.Bundle.from_directory`,
        for a bundle produced by ``hhemt combine`` rather than by a single
        analysis.

        Parameters
        ----------
        path : Path or str
            An UNPACKED combined-bundle directory containing
            ``bundle_manifest.json`` and a ``child_crates/`` directory holding
            one intact crate per combined experiment.

        Returns
        -------
        CombinedBundle
            A handle exposing :meth:`regenerate_report`.

        Raises
        ------
        FileNotFoundError
            No ``bundle_manifest.json`` under ``path``.
        BundleSchemaError
            The manifest's ``bundle_schema_version`` does not EXACTLY equal the
            local ``BUNDLE_SCHEMA_VERSION``. Note this is stricter than
            :meth:`Bundle.from_directory`, which admits older bundles: a
            combined bundle must match exactly, so one emitted by any other
            toolkit version must be re-combined from its children rather than
            read here.

        Notes
        -----
        ``eda()`` is deliberately unavailable on a combined bundle — there is no
        aggregate EDA surface across experiments. Run ``.eda()`` on each intact
        child bundle instead.
        """
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

    @classmethod
    def from_zip(cls, path: Path | str) -> CombinedBundle:
        """Unpack a combined ``.zip`` beside itself and return a handle on the unpacked folder.

        The render set of the bundle is extracted into ``{zip stem}/`` next to the zip, the
        place ``hhemt report-from-bundle`` unpacks a member zip: the root files, and for each
        experiment the render set (everything but raw ``out_*/bin`` solver output) of its nested
        member zip at ``child_crates/{experiment_id}/``, with the zip's
        ``child_crates/{experiment_id}.refreshed/`` figures laid over it. A nested member
        zip that was stored as one compressed entry is inflated to a temporary file first, and
        that file is removed once its render set is out. A repeat call reuses what a previous
        call extracted when the zip is the same file (size and modification time), restoring
        any missing or resized file; a different zip of the same name starts from an empty folder.

        Parameters
        ----------
        path : Path or str
            A combined bundle zip written by ``hhemt combine``.

        Returns
        -------
        CombinedBundle
            A handle on the unpacked folder (``is_archive`` false); ``regenerate_report()``
            on it re-renders in place.

        Raises
        ------
        FileNotFoundError
            ``path`` is not a file, or the zip has no ``bundle_manifest.json``.
        BundleSchemaError
            The combined manifest's ``bundle_schema_version`` differs from this toolkit's.
        """
        zip_path = Path(path).resolve()
        if not zip_path.is_file():
            raise FileNotFoundError(f"No combined bundle zip at {zip_path}.")
        unpack = zip_path.parent / zip_path.stem
        reuse = _prepare_unpack_dir(zip_path, unpack)
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        if BUNDLE_MANIFEST_FILENAME not in names:
            raise FileNotFoundError(f"No {BUNDLE_MANIFEST_FILENAME} in {zip_path}.")
        prefix = f"{_CHILD_CRATES_SUBDIR}/"
        root_names = [n for n in names if not n.startswith(prefix)]
        extract_zip(zip_path, unpack, members=root_names, only_missing_or_resized=reuse)
        scratch = unpack / ".member_tmp"
        try:
            for arc in sorted(
                n for n in names if n.startswith(prefix) and n.endswith(".zip") and "/" not in n[len(prefix) :]
            ):
                eid = arc[len(prefix) : -len(".zip")]
                # One member at a time: the nested zip becomes a file (inflated if it was stored
                # as a compressed entry), its render set is extracted, and the file is removed.
                extract_zip(zip_path, scratch, members=[arc])
                nested = scratch / arc
                with zipfile.ZipFile(nested) as nz:
                    nested_names = [n for n in nz.namelist() if not n.endswith("/")]
                extract_zip(
                    nested,
                    unpack / _CHILD_CRATES_SUBDIR / eid,
                    members=render_set(nested_names),
                    only_missing_or_resized=reuse,
                )
                nested.unlink()  # EXEMPT-DU: transient-intermediate
                refreshed = f"{prefix}{eid}.refreshed/"
                with zipfile.ZipFile(zip_path) as zf:
                    for name in (n for n in names if n.startswith(refreshed)):
                        # The same confinement `extract_zip` applies: a name that climbs with `..` is
                        # placed inside the child folder, never beside or above it.
                        target = _confined_target(unpack / _CHILD_CRATES_SUBDIR / eid, name[len(refreshed) :])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zf.read(name))
        finally:
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)  # EXEMPT-DU: transient-intermediate
        _write_unpack_record(zip_path, unpack)
        return cls.from_directory(unpack)

    @property
    def root(self) -> Path:
        """Absolute path to the unpacked combined-bundle directory.

        Its ``child_crates/{experiment_id}/`` subdirectories hold one intact
        crate per combined experiment; the cross-experiment report is rendered
        by harvesting their already-rendered figures in place.
        """
        return self._root

    def regenerate_report(self, *, format: Literal["html", "zip"] = "zip", declare_stale_plots: bool = False) -> Path:
        """Regenerate the combined report from the bundled data (mirrors Bundle.regenerate_report).

        Re-invokes the SAME render path against the bundle root — a REAL
        ``snakemake --report`` over the combined Snakefile (first-class chrome/nav/CSS).
        The renderers read the bundled ``combined_{compatibility,intercomparison}.json``
        read-models, so no re-merge is needed.
        """
        from hhemt.provenance import assert_plots_match_running_build

        if self._is_archive:
            # A combined zip is re-rendered from its unpacked render set, which `from_zip` lays
            # beside the zip; the handle for that folder does the render.
            return CombinedBundle.from_zip(self._root).regenerate_report(
                format=format, declare_stale_plots=declare_stale_plots
            )
        assert_plots_match_running_build(self._root, declare_stale_plots=declare_stale_plots)
        _render_combined_report(None, None, self._root, formats=(format,))
        report = self._root / f"analysis_report.{format}"
        if not report.exists():
            raise FileNotFoundError(f"Combined report {report} was not produced.")
        return report

    def eda(self, *, plots_only: bool = True):
        """Combined bundles have no aggregate EDA surface (see child bundles)."""
        raise NotImplementedError(
            "Combined bundles have no aggregate EDA surface. The cross-experiment "
            "byte-identity panel is deferred (R6); run .eda() on each child bundle "
            "under child_crates/{experiment_id}/ via Bundle.from_directory(...)."
        )

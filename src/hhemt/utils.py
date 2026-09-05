import datetime
import importlib.util
import json
import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any, Literal

import xarray as xr
import yaml
from platformdirs import user_data_dir

#: Root consolidated-store names, in RESOLUTION PRIORITY order. V0021 unified BOTH prior
#: root stores -- a regular analysis's ``analysis_datatree.zarr`` and a sensitivity
#: master's ``sensitivity_datatree.zarr`` -- into one ``experiment_datatree.zarr`` per
#: experiment. The two retired names are RETAINED because a render bundle is a tarball no
#: migration reaches, and an analysis dir that predates the migration still carries one.
#:
#: ORDER IS LOAD-BEARING and is not alphabetical, historical, or arbitrary. Every consumer
#: routed through ``resolve_experiment_tree`` wants the EXPERIMENT-LEVEL AGGREGATE. On a
#: root carrying BOTH retired names the aggregate is the sensitivity master's store, so
#: ``sensitivity_datatree.zarr`` MUST precede ``analysis_datatree.zarr``: the latter at an
#: experiment root is that experiment's own single-analysis tree, which is the aggregate
#: only when it is the sole retired name present.
#:
#: Per-MEMBER stores under ``members/member_N/analysis_datatree.zarr`` are NOT renamed by
#: V0021 and must not be resolved through here.
ROOT_TREE_NAMES = (
    "experiment_datatree.zarr",
    "sensitivity_datatree.zarr",
    "analysis_datatree.zarr",
)
EXPERIMENT_TREE_NAME = ROOT_TREE_NAMES[0]


def resolve_experiment_tree(root: str | Path) -> Path:
    """Resolve an experiment's ROOT consolidated store by EXISTENCE.

    Returns the first existing candidate in ``ROOT_TREE_NAMES`` priority order -- see
    that tuple's note on why the order is not interchangeable. When NONE exists the
    unified name is returned rather than None, so a caller's own absent-tree branch keeps
    its `.exists()` shape and reports against the canonical name instead of a retired one.
    """
    root = Path(root)
    for name in ROOT_TREE_NAMES:
        cand = root / name
        if cand.exists():
            return cand
    return root / EXPERIMENT_TREE_NAME


class BatchJobSubmissionError(Exception):
    """
    Custom exception for batch job submission failures.

    Provides detailed information about why a batch job submission failed,
    including the script path, command, dependency information, and stderr output.
    """

    def __init__(
        self,
        script_path: Path,
        command: list,
        return_code: int,
        stderr: str,
        dependent_job_id: int | str | list | None = None,
        dependency_type: str = "afterok",
    ):
        self.script_path = script_path
        self.command = command
        self.return_code = return_code
        self.stderr = stderr
        self.dependent_job_id = dependent_job_id
        self.dependency_type = dependency_type

        # Format the error message
        error_lines = [
            "Failed to submit batch job script",
            f"  Script: {script_path}",
        ]

        if dependent_job_id:
            error_lines.append(f"  Dependency: {dependency_type}:{dependent_job_id}")

        error_lines.extend(
            [
                f"  Command: {' '.join(str(c) for c in command)}",
                f"  Return code: {return_code}",
            ]
        )

        if stderr.strip():
            error_lines.append(f"  Error output:\n{self._indent_text(stderr)}")

        self.message = "\n".join(error_lines)
        super().__init__(self.message)

    @staticmethod
    def _indent_text(text: str, indent: str = "    ") -> str:
        """Indent each line of text for better readability."""
        return "\n".join(indent + line for line in text.split("\n"))


def fast_rmtree(
    path: str | Path,
    *,
    missing_ok: bool = True,
    onerror: Callable | None = None,
    analysis_dir: str | Path | None = None,
) -> int:
    """Fast, cross-platform directory delete. Returns bytes reclaimed.

    Uses OS-native delete commands for speed; falls back to shutil.rmtree.

    When `analysis_dir` is provided AND `path` is not itself the analysis_dir,
    `du_sentinels.restamp_parent_sentinels(path, analysis_dir=...)` is invoked
    after the delete completes so parent-scope DU sentinels stay accurate. The
    `path == analysis_dir` short-circuit converts the EXEMPT-site convention
    (root-wipe — re-stamping a directory being deleted is meaningless) from a
    prose comment into grep-detectable code (SE F-I Flag 2).

    Parameters
    ----------
    path : str | Path
        Directory path to delete.
    missing_ok : bool
        If True, silently return when path does not exist.
    onerror : callable, optional
        Error handler passed to shutil.rmtree (fallback only).
    analysis_dir : str | Path, optional
        Root of the analysis tree this delete is scoped to. When provided,
        parent-scope DU sentinels under `analysis_dir` are re-stamped after
        the delete; when None, no re-stamping occurs (the caller is responsible
        for sentinel accuracy out-of-band).
    """
    path = Path(path)

    if not path.exists():
        if missing_ok:
            return 0
        raise FileNotFoundError(path)

    # Measured BEFORE the delete -- afterwards there is nothing to measure. This is
    # the pre-delete walk the docstring prices, over a tree about to be walked for
    # deletion anyway.
    freed = 0
    if path.is_symlink() or path.is_file():
        try:
            freed = path.stat().st_size
        except OSError:
            pass
    else:
        for _f in path.rglob("*"):
            try:
                if _f.is_file():
                    freed += _f.stat().st_size
            except OSError:
                pass

    if path.is_symlink() or path.is_file():
        path.unlink()
        _restamp_after_mutation(path, analysis_dir)
        return freed

    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "rmdir", "/s", "/q", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["rm", "-rf", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        shutil.rmtree(path, onerror=onerror)

    _restamp_after_mutation(path, analysis_dir)
    return freed


def _restamp_after_mutation(path: Path, analysis_dir: str | Path | None) -> None:
    """Re-stamp parent DU sentinels for `path` under `analysis_dir`.

    No-op when `analysis_dir` is None, when `path == analysis_dir` (root-wipe
    short-circuit per SE F-I Flag 2), or when path resolution fails. Imports
    `restamp_parent_sentinels` lazily to keep `utils.py` free of a top-level
    dependency on `du_sentinels.py`.
    """
    if analysis_dir is None:
        return
    try:
        path_resolved = Path(path).resolve()
        analysis_resolved = Path(analysis_dir).resolve()
    except OSError:
        return
    if path_resolved == analysis_resolved:
        return
    if not analysis_resolved.exists():
        return
    from hhemt.du_sentinels import restamp_parent_sentinels

    restamp_parent_sentinels(Path(path), analysis_dir=analysis_resolved)


def _recover_and_clear_publish_temps(final, aside, tmp, *, analysis_dir=None) -> None:
    """Steps 0 and 1 of the crash-safe publish. Shared by the callable-wrapping
    form below and by the inline form in process_simulation, so the .aside/.tmp
    preference has exactly ONE implementation -- two copies is how it gets undone.

    THE PREFERENCE IS ORDER-DERIVED, NOT PROVENANCE-DERIVED. Read this before
    changing either branch.

    An earlier draft preferred .aside on the ground that it is "trustworthy by
    construction -- only ever a renamed complete `final`". That rests on the
    ABSENT-or-COMPLETE invariant, WHICH THIS MODULE ESTABLISHES, so it does not
    hold over the stores this change INHERITS: on the first run after it ships,
    every `final` on disk was written by the unprotected path and may be a
    silent-fill partial. Measured: with an inherited partial at .aside and a
    complete store at .tmp, that draft restored the partial and deleted the
    complete one, announcing that it was restoring the last known-good copy.
    That is a regression against doing nothing -- clearing both would have left
    `final` absent and let the rule re-run.

    What holds instead is an ORDERING fact about this process: .aside is created
    ONLY at step 3, which runs only after write_fn(tmp) RETURNED (and, in the
    inline form, only after the streaming loop completed). So

        aside.exists() and not final.exists()   ==>   .tmp is COMPLETE

    and in that state .tmp is the newer complete store and .aside is whatever
    `final` happened to be. Prefer .tmp.

    RECOVERY PUBLISHES, and that is a deliberate widening of what this function
    does. The os.rename(tmp, final) below promotes a PREVIOUS invocation's store
    to `final`. After a force-rerun that store may reflect a superseded input
    set, so a reader between the recovery and the current write's own publish can
    see stale-but-complete data. It is transient unless the current write also
    fails with retries exhausted; the alternative -- discarding a complete store
    to leave `final` absent -- loses data the system still has.

    The asymmetry stays real everywhere else. On the step-2 crash path .aside
    does NOT exist, a mid-write .tmp is incomplete, and nothing can tell it from
    a complete one -- so it is cleared, unconditionally, in every branch.
    """
    if aside.exists():
        if not final.exists():
            if tmp.exists():
                # Died in the step3->step4 gap: .tmp is the complete new store.
                os.rename(tmp, final)
                fast_rmtree(aside, analysis_dir=analysis_dir)
            else:
                warnings.warn(
                    f"Recovering {aside}: a previous publish died mid-swap with no "
                    f"replacement store present, so it is being restored to "
                    f"{final.name}.",
                    stacklevel=2,
                )
                os.rename(aside, final)
        else:
            # Died in the step4->step5 gap: the aside is superseded.
            fast_rmtree(aside, analysis_dir=analysis_dir)
    if tmp.exists():
        warnings.warn(f"Discarding un-publishable partial store {tmp}.", stacklevel=2)
        fast_rmtree(tmp, analysis_dir=analysis_dir)


def _publish_store_crash_safe(write_fn, fname_out, *, analysis_dir=None) -> None:
    """Build a zarr store under a temp name and publish it by rename.

    GUARANTEE, and it is exactly this one: `fname_out` is either ABSENT or a
    COMPLETE store, never an INCOMPLETE one. It is NOT guaranteed that a complete
    store survives somewhere after any crash -- on the FRESH path (no prior store)
    a crash before the rename leaves nothing, and that is the relaunch's own path.

    WHY THIS EXISTS: no sound completeness detector is possible. zarr omits an
    all-fill chunk by default (write_empty_chunks unset anywhere in this tree), so
    a legitimately dry corner of a flood-depth field is byte-identical to a killed
    write, and a missing chunk READS as the fill value rather than raising.
    Measured under a real SIGKILL: the store is present-but-incomplete for ~28% of
    a direct mode="w" write, opens without error, reports the full expected length,
    and returns NaN for the rows whose chunks never landed.

    THE GUARANTEE IS SINGLE-WRITER. Two processes publishing the same `final` can
    interleave steps 3 and 4 and defeat it. Do not read that as "the toolkit
    prevents this": the arbitration is partial and checkable. Snakemake's
    output-vs-output lock covers concurrent rules in ONE workflow, and
    SnakemakeWorkflowBuilder._orchestrator_liveness_gate covers a second driver --
    but reprocess runs with --nolock and that gate refuses on a live DRIVER, not
    on live WORKERS, so two publishers can still reach one store. The failure it
    degrades to is a loud rename/replace error, which is strictly better than
    today's silent interleave, and that is the whole claim.

    STEPS 0/3/4 ARE ONE INTERLOCK, NOT THREE PIECES OF HYGIENE. Step 0 guarantees
    no `.aside` survives into step 3; step 3's rename is what frees `final` so
    step 4's os.replace is legal (both raise "Directory not empty" onto a
    non-empty target). "Optimising away" step 0 breaks step 3 on the SECOND
    recovery run, not the first -- which is why it would survive testing.

    COST: 1.00x peak on the fresh path (the temp occupies bytes the store would
    have taken anyway) and 2.00x on the rewrite path, where old and new coexist
    until step 5.
    """
    final = Path(fname_out)
    aside = final.with_name(final.name + ".aside")
    tmp = final.with_name(final.name + ".tmp")

    _recover_and_clear_publish_temps(final, aside, tmp, analysis_dir=analysis_dir)
    write_fn(tmp)  # step 2
    if final.exists():
        os.rename(final, aside)  # step 3
    os.replace(tmp, final)  # step 4
    if aside.exists():
        fast_rmtree(aside, analysis_dir=analysis_dir)  # step 5


def chapters_dir_for(fname_out) -> Path:
    """Sibling directory holding this store's per-chapter parts."""
    root = Path(fname_out)
    return root.with_name(root.name + ".chapters")


def unified_flag_for(fname_out) -> Path:
    root = Path(fname_out)
    return root.with_name(root.name + ".done")


def chapter_store_for(chapters: Path, k: int) -> Path:
    return chapters / f"chapter_{k:05d}.zarr"


def chapter_flag_for(chapters: Path, k: int) -> Path:
    return chapters / f"chapter_{k:05d}.done"


def verify_and_flag_chapter(store: Path, flag: Path, n_expected: int) -> None:
    """Open the chapter store, confirm it is readable, THEN write the flag.

    THE ORDER IS THE CONTRACT: write -> verify -> flag -> record -> clear raw. The
    RECORD step appends the producing build to the chapter set's provenance history
    and is what lets the chapter-set guard compare a resume against the builds that
    actually flagged chapters. It follows the flag deliberately: recording before the
    flag lands would name a build for a chapter a failed verify never flagged. The flag is a
    positive completion marker in the Gotcha-40 sense, and the clear that follows
    it destroys the raw inputs, so the flag must not be written on an unchecked
    store.

    WHAT THE FLAG ATTESTS, stated exactly, because an earlier docstring overclaimed
    it. It attests that `to_zarr` RETURNED and that the resulting store OPENS. It
    does NOT attest that the store's data is complete, and the size check below is
    near-tautological on the success path: zarr writes the declared shape before the
    chunk bytes, so if `to_zarr` returned the size is right by construction, and if
    it did not return the flag is never written and `reap_unflagged_chapters`
    removes the store. With `write_empty_chunks=False` an absent chunk is
    byte-indistinguishable from a dry corner of a flood field, so no cheap check can
    do better. The open() is retained because it is not vacuous -- it catches a
    store that returned but cannot be read back -- and the size check is retained
    because it costs one metadata read and catches a stale store at the same index.
    """
    from hhemt.exceptions import ProcessingError

    ds = xr.open_zarr(store, consolidated=False)
    try:
        n = ds.sizes.get("timestep_min", 0)
        if n != n_expected:
            raise ProcessingError(
                operation="verify_and_flag_chapter",
                filepath=str(store),
                reason=f"chapter covers {n} timesteps, expected {n_expected}",
            )
    finally:
        ds.close()
    tmp = flag.with_suffix(flag.suffix + ".tmp")
    tmp.write_text("ok", encoding="utf-8")
    os.replace(tmp, flag)
    # RECORD THE PRODUCING BUILD, and only now. The chapter-set guard compares the
    # running build against this history, so the history must be the provenance of
    # the FLAGGED chapters -- not a log of every invocation that entered the writer.
    # Recorded here rather than at the two call sites because those two blocks are a
    # measured drift surface (they have already diverged once) and a third flush site
    # would silently miss it. Function-local import, matching ProcessingError above;
    # provenance imports utils function-locally too, so no module-load edge is added.
    from hhemt.provenance import record_chapter_build

    record_chapter_build(flag.parent)


def completed_chapters(chapters: Path) -> dict:
    """Chapter index -> store path, for FLAGGED chapters whose store still exists."""
    if not chapters.exists():
        return {}
    out = {}
    for f in sorted(chapters.glob("chapter_*.done")):
        k = int(f.stem.split("_")[1])
        store = chapter_store_for(chapters, k)
        if store.exists():
            out[k] = store
    return out


def covered_timesteps(chapters: Path) -> set:
    """The set of timestep_min VALUES already published by flagged chapters.

    A SET OF COORDINATE VALUES, never a COUNT. The loop that consumes this skips
    timesteps on three paths (a missing raw file, a variable that loaded nothing, a
    chunk that loaded nothing), so the number of published timesteps and their
    POSITION in `timestep_list` diverge whenever any timestep is skipped -- which on
    a partially-written raw directory is the expected shape, not a corner case. A
    count used as a slice index is silently wrong there; a value set is correct
    under skipped timesteps, a non-contiguous chapter set, and chapters written out
    of order.

    DEPENDENCY THIS RESTS ON, named because nothing protects it: exact set membership
    requires that `timestep_min` round-trips through zarr UNENCODED. return_dic_zarr_encodings
    does not encode it today -- its float32 branch iterates ds.data_vars and its coordinate
    loop handles only Unicode -- but NO spec in this set touches that module and NO test pins
    the property, so a one-line edit there would silently break resume by making
    `t not in _covered_ts` true for timesteps that ARE covered.
    """
    out = set()
    for store in completed_chapters(chapters).values():
        ds = xr.open_zarr(store, consolidated=False)
        try:
            out.update(ds["timestep_min"].values.tolist())
        finally:
            ds.close()
    return out


def reap_unflagged_chapters(chapters: Path, *, analysis_dir=None) -> None:
    """STATE 3: a chapter store with no flag was interrupted mid-write. Delete it.

    Never deletes a FLAGGED chapter -- after a raw clear a flagged chapter is the
    SOLE copy of its timesteps, and deleting one is unrecoverable without re-running
    the simulation.
    """
    if not chapters.exists():
        return
    for store in sorted(chapters.glob("chapter_*.zarr")):
        k = int(store.stem.split("_")[1])
        if not chapter_flag_for(chapters, k).exists():
            warnings.warn(f"Discarding unflagged (interrupted) chapter store {store}.", stacklevel=2)
            fast_rmtree(store, analysis_dir=analysis_dir)


def clear_raw_for_timesteps(df_outputs, timesteps, *, analysis_dir=None) -> int:
    """Delete ONLY the raw per-timestep files this chapter consumed. Returns bytes.

    DELIBERATELY NOT `process_simulation._clear_raw_outputs`, and it MUST NOT be
    refactored to share that helper's allowlist. `_CLEAR_RAW_DELETE_SUBDIRS`
    contains `cfg/` (the config_NNNN.cfg HOTSTART CHECKPOINTS a walltime-killed sim
    resumes from) and `performance/` (the per-checkpoint files V0008's
    _aggregate_perf_tseries merges for wallclock). Calling it per chapter would
    RAISE RuntimeError on a resumable multi-allocation run and SILENTLY destroy both
    on a single-allocation one.

    This helper preserves them BY CONSTRUCTION rather than by omission from a delete
    list: it deletes only paths named in `df_outputs`, which contains the
    per-variable per-timestep data files and nothing else.
    """
    freed = 0
    for path in df_outputs.loc[list(timesteps)].to_numpy().ravel():
        p = Path(path)
        if not p.exists():
            continue
        # Routed through fast_rmtree rather than a bare unlink + wrapper call. Three
        # reasons, and the first is a correctness one: restamp_parent_sentinels has NO
        # None guard (`if not analysis_dir.exists()`), and analysis_dir defaults to None
        # here, so calling it directly -- the change that would satisfy the DU checker's
        # _is_restamp_call most obviously -- raises AttributeError on every caller that
        # omits analysis_dir. fast_rmtree handles a FILE, guards None via
        # _restamp_after_mutation, and RETURNS the bytes it freed ([Q232]), which
        # replaces the manual stat.
        freed += fast_rmtree(p, analysis_dir=analysis_dir)
    return freed


def merge_chapters_to_unified(chapters: Path, fname_out, *, analysis_dir=None) -> None:
    """STATES 4/5/6: concatenate flagged chapters into the unified store.

    A flagless unified store is a merge that was interrupted; it is deleted and
    re-merged rather than trusted, because nothing distinguishes it from a complete
    one. The chapters are still present -- the interlock holds them until the
    unified flag lands, which is what makes re-merge possible at all.

    CONTIGUITY IS ASSERTED, not assumed. `completed_chapters` admits an interior
    hole: a flag whose store was later removed leaves a gap, and concatenating
    sorted(parts) over a gapped set yields a unified store missing an interior
    timestep range and then FLAGS IT COMPLETE. The index set must be exactly
    range(len(parts)).
    """
    from hhemt.exceptions import ProcessingError

    final = Path(fname_out)
    flag = unified_flag_for(final)
    if final.exists() and not flag.exists():
        warnings.warn(f"Discarding un-flagged (interrupted) unified store {final}; re-merging.", stacklevel=2)
        fast_rmtree(final, analysis_dir=analysis_dir)
    parts = completed_chapters(chapters)
    if not parts:
        raise ProcessingError(
            operation="merge_chapters_to_unified",
            filepath=str(chapters),
            reason="no flagged chapter stores to merge",
        )
    if sorted(parts) != list(range(len(parts))):
        raise ProcessingError(
            operation="merge_chapters_to_unified",
            filepath=str(chapters),
            reason=(
                f"chapter index set {sorted(parts)} is not contiguous from 0; a flagged "
                "chapter's store is missing and merging would publish a store with an "
                "interior gap and flag it complete"
            ),
        )
    ds = xr.concat(
        [xr.open_zarr(parts[k], consolidated=False) for k in sorted(parts)],
        dim="timestep_min",
    )
    # `open_zarr` returns DASK-backed arrays chunked on each chapter's STORED zarr
    # grid, not on the chapter extent -- so a 132-timestep chapter stored at chunk 17
    # contributes (17 x7, 13) and the next chapter's chunks follow it, putting a SHORT
    # chunk in the interior. Zarr permits a short FINAL chunk only, and the interior
    # short chunk also straddles the inherited grid, which xarray >= 2026.4 refuses via
    # validate_grid_chunks_alignment. Dropping `encoding['chunks']` alone does NOT fix
    # this: it removes the target grid and leaves zarr to derive one from the same
    # non-uniform dask chunks, which then fails the uniformity requirement instead.
    # Making the concat-axis dask grid UNIFORM at the inherited length satisfies both
    # and preserves the published store's chunking ON THE CONCAT AXIS. That scope is
    # deliberate: this touches `timestep_min` only, and a NON-concat axis whose grid
    # differs across chapters is NOT covered. `xr.concat` unifies a non-concat axis to
    # the FINER grid while chapter 0's encoding survives, so the same refusal would
    # fire if chapter 0 were the COARSER side. It cannot be, under this writer: chunk
    # coarseness falls as a chapter's time extent grows, chapters flush at a threshold
    # so only the LAST is short, and chapter 0 is therefore never strictly coarser.
    # Derive from data_vars ONLY -- coords carry their own grids and would send every
    # case down the fallback branch, collapsing the time axis to a single chunk.
    _time_chunks = {
        int(_v.encoding["chunks"][_v.dims.index("timestep_min")])
        for _v in ds.data_vars.values()
        if "timestep_min" in _v.dims and _v.encoding.get("chunks")
    }
    if len(_time_chunks) == 1:
        ds = ds.chunk({"timestep_min": _time_chunks.pop()})
    elif _time_chunks:
        for _v in (*ds.variables.values(), *ds.coords.values()):
            _v.encoding.pop("chunks", None)
            _v.encoding.pop("preferred_chunks", None)
        ds = ds.chunk({"timestep_min": min(_time_chunks)})
    ds.to_zarr(final, mode="w", consolidated=False)
    tmp = flag.with_suffix(flag.suffix + ".tmp")
    tmp.write_text("ok", encoding="utf-8")
    os.replace(tmp, flag)
    # STATE 6: chapters die ONLY now, after the unified flag.
    fast_rmtree(chapters, analysis_dir=analysis_dir)


def fix_line_endings(file_path, target_ending="\n"):
    """
    Convert line endings in a file to the target format.
    Only rewrites the file if line endings are incorrect.

    Args:
        file_path (str): Path to the file to fix
        target_ending (str): Target line ending ('\n' for LF, '\r\n' for CRLF)

    Returns:
        bool: True if file was modified, False if already correct
    """
    try:
        # Read file in binary mode
        with open(file_path, "rb") as f:
            original_content = f.read()

        # Normalize to LF first, then convert to target
        normalized = original_content.replace(b"\r\n", b"\n")  # CRLF -> LF
        normalized = normalized.replace(b"\r", b"\n")  # CR -> LF

        # Convert to target ending if needed
        if target_ending == "\r\n":
            normalized = normalized.replace(b"\n", b"\r\n")

        # Only write if content changed
        if normalized != original_content:
            with open(file_path, "wb") as f:
                f.write(normalized)
            print(f"✓ Fixed line endings in: {file_path}")
            return True
        else:
            # print(f"✓ Already correct: {file_path}")
            return False

    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def run_bash_script(
    bash_script: Path,
    dependent_job_id: int | str | list | None = None,
    dependency_type: Literal["afterok", "afterany"] = "afterok",
    verbose: bool = True,
):
    cmd = ["sbatch"]
    dpdndncy = ""
    if dependent_job_id:
        if isinstance(dependent_job_id, list):
            dependent_job_id = ",".join(dependent_job_id)
        cmd.append(
            f"--dependency={dependency_type}:{dependent_job_id}",
        )
        dpdndncy = f"\n dependent on job {dependent_job_id} using dependency={dependency_type}"
    cmd.append(str(bash_script))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise BatchJobSubmissionError(
            script_path=bash_script,
            command=cmd,
            return_code=e.returncode,
            stderr=e.stderr,
            dependent_job_id=dependent_job_id if dependent_job_id else None,
            dependency_type=dependency_type,
        ) from e

    job_id = proc.stdout.strip().split()[-1]
    if verbose:
        print(f"Submitted script {bash_script}{dpdndncy}\njob id: {job_id}", flush=True)
    return job_id


def archive_directory_contents(dir: Path):
    archive_dir = dir / "_archive"
    archive_dir.mkdir(exist_ok=True, parents=True)
    for item in dir.iterdir():
        if item.name == "_archive":
            continue
        shutil.move(str(item), archive_dir / item.name)


def create_mask_from_shapefile(da_to_mask, shapefile_path=None, series_single_row_of_gdf=None):  # , COORD_EPSG):
    # da_to_mask, shapefile_path = da_sim_wlevel, f_mitigation_aois
    og_shape = da_to_mask.shape
    import geopandas as gpd
    import rasterio.features
    from shapely.geometry import mapping

    if shapefile_path is not None:
        gdf = gpd.read_file(shapefile_path)
        shapes = [mapping(geom) for geom in gdf.geometry]  # Convert geometries to GeoJSON-like format
    if series_single_row_of_gdf is not None:
        shapes = [mapping(series_single_row_of_gdf.geometry)]
    mask = rasterio.features.geometry_mask(
        shapes,
        transform=da_to_mask.rio.transform(),
        invert=True,
        out_shape=(og_shape),
    )
    return mask


def read_yaml(f_yaml: Path | str):
    return yaml.safe_load(Path(f_yaml).read_text())


def write_yaml(data: dict, f_yaml: Path | str):
    with open(f_yaml, "w") as file:
        yaml.dump(data, file)
    return


def get_package_root(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Package {package_name} not found")
    return Path(spec.origin).parent


def get_package_data_root(package_name) -> Path:
    return Path(user_data_dir(package_name))


def fill_template(f_template: Path, mapping: dict):
    with open(f_template) as T:
        template = Template(T.read())
        filled = template.safe_substitute(mapping)
    return filled


def create_from_template(f_template: Path, mapping: dict, f_out: Path):
    filled = fill_template(f_template, mapping)
    f_out.parent.mkdir(parents=True, exist_ok=True)
    with open(f_out, "w+") as f1:
        f1.write(filled)
    return filled


def find_all_keys_in_template(f_template):
    with open(f_template) as f:
        text = f.read()
    keys = re.findall(r"\{([^}]+)\}", text)
    unique_keys = list(dict.fromkeys(keys))
    return unique_keys


def load_json(file: Path):
    with open(file) as f:
        log = json.load(f)
    return log


def write_json(data: dict, file: Path):
    file.parent.mkdir(exist_ok=True, parents=True)
    pid = os.getpid()
    tmp_path = file.with_suffix(file.suffix + f".{pid}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(file)


def write_json_exclusive(data: dict, file: Path):
    """Create `file` with `data`, refusing if the name already exists.

    The counterpart to `write_json` for the ONE case where a caller believes the
    target is absent. `write_json` uses `os.replace`, which is unconditional and
    therefore cannot distinguish creating a file from destroying one; this asks
    the filesystem, via `O_CREAT | O_EXCL`, and raises `FileExistsError` when the
    belief is wrong. A caller that inferred absence from a failed read has no
    other way to check that inference -- a second read is the same instrument
    that already gave the wrong answer.
    """
    file.parent.mkdir(exist_ok=True, parents=True)
    fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())


def replace_substring_in_file(file_path, old_substring, new_substring, verbose=False):
    """
    Replace all occurrences of old_substring with new_substring in a text file.

    Parameters:
        file_path (str): Path to the text file.
        old_substring (str): The substring to be replaced.
        new_substring (str): The substring to replace with.
    """
    # Read the file
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Replace substring
    content = content.replace(old_substring, new_substring)

    # Write back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    if verbose:
        print(f"Replaced '{old_substring}' with '{new_substring}' in {file_path}")


def read_text_file_as_string(file):
    with open(file) as f:
        contents = f.read()
    return contents


def current_datetime():
    return (
        datetime.datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds",
        )
    )


def current_datetime_string(filepath_friendly: bool = False):
    """
    Docstring for current_datetime_string

    Generates a datetime string following following  ISO 8601 format conventions.

    :param filepath_friendly: If True, colons are replaced with nothing, e.g.,
        2026-01-07T10:03:37-05:00 becomes 2026-01-07T100337-0500
    :type filepath_friendly: bool
    """
    dts = current_datetime()
    if filepath_friendly:
        dts = dts.replace(":", "")

    return dts


def string_to_datetime(dt: str):
    return datetime.datetime.fromisoformat(dt)


def read_header(file, nlines):
    lst_lines = []
    with open(file) as f:
        for _i in range(nlines):
            line = f.readline()
            if not line:
                break  # Stop if file has fewer than 6 lines
            lst_lines.append(line)
    return lst_lines


def read_text_file_as_list_of_strings(file):
    with open(file) as f:
        contents = f.readlines()
    return contents


_EMPTY_TRITON_LOG_FIELDS: dict[str, Any] = {
    "nTasks": None,
    "omp_threads_per_task": None,
    "gpus_per_task": None,
    "total_gpus": None,
    "gpu_backend": None,
    "build_type": None,
    "triton_git_version": None,
    "wall_time_s": None,
    "machine": None,
    "cpu": None,
    "gpu": None,
}


def parse_triton_log_file(log_file_path: Path) -> dict[str, Any]:
    #  TODO retrieveing wall time from the log path is bad because if the sim
    # is resumed from a hotstart file, the time will only reflect the
    # last chunk of the simulation since its restarting
    """
    Parse TRITON log.out file to extract actual compute resource usage.

    Parameters
    ----------
    log_file_path : Path
        Path to the log.out file

    Returns
    -------
    dict
        Dictionary containing:
        - nTasks: int - Number of MPI tasks
        - omp_threads_per_task: int - OpenMP threads per task
        - gpus_per_task: int - GPUs per task
        - total_gpus: int - Total GPUs used
        - gpu_backend: str - GPU backend (HIP/CUDA/none)
        - build_type: str - Build type (e.g., "CPU+OMP", "GPU+HIP")
        - triton_git_version: str - TRITON git version
        - wall_time_s: float - Total wall time in seconds
        - machine: str - Machine name
        - cpu: str - CPU model
        - gpu: str - GPU device model as reported by the runtime, or "none" on a
          CPU-only build. Absent (None) on any log predating the GPU emission.

    Returns None for all fields if file doesn't exist or parsing fails.
    """
    if not log_file_path.exists():
        return dict(_EMPTY_TRITON_LOG_FIELDS)

    try:
        content = read_text_file_as_string(log_file_path)

        # Initialize result dictionary with None values
        result = dict(_EMPTY_TRITON_LOG_FIELDS)

        # Parse each field using regex
        # Machine name
        match = re.search(r"Machine\s*:\s*(.+)", content)
        if match:
            result["machine"] = match.group(1).strip()  # type: ignore

        # CPU model
        match = re.search(r"^CPU\s*:\s*(.+)$", content, re.M)
        if match:
            result["cpu"] = match.group(1).strip()  # type: ignore

        match = re.search(r"^GPU\s*:\s*(.+)$", content, re.M)
        if match:
            result["gpu"] = match.group(1).strip()  # type: ignore

        # nTasks
        match = re.search(r"nTasks\s*:\s*(\d+)", content)
        if match:
            result["nTasks"] = int(match.group(1))  # type: ignore

        # OMP threads per task
        match = re.search(r"OMP threads per task\s*:\s*(\d+)", content)
        if match:
            result["omp_threads_per_task"] = int(match.group(1))  # type: ignore

        # GPUs per task (handle "0 (CPU-only)" case)
        match = re.search(r"GPUs per task\s*:\s*(\d+)", content)
        if match:
            result["gpus_per_task"] = int(match.group(1))  # type: ignore

        # GPU backend
        match = re.search(r"GPU backend\s*:\s*(\S+)", content)
        if match:
            result["gpu_backend"] = match.group(1).strip()  # type: ignore

        # Total GPUs
        match = re.search(r"Total GPUs\s*:\s*(\d+)", content)
        if match:
            result["total_gpus"] = int(match.group(1))  # type: ignore

        # TRITON git version
        match = re.search(r"TRITON_GIT_VERSION\s*:\s*(.+)", content)
        if match:
            result["triton_git_version"] = match.group(1).strip()  # type: ignore

        # Build type
        match = re.search(r"Build type\s*:\s*(.+)", content)
        if match:
            result["build_type"] = match.group(1).strip()  # type: ignore

        # Wall time
        match = re.search(r"TRITON total wall time \[s\]\s*:\s*([\d.]+)", content)
        if match:
            result["wall_time_s"] = float(match.group(1))  # type: ignore

        return result

    except Exception as e:
        warnings.warn(
            f"Failed to parse TRITON log file {log_file_path}: {str(e)}",
            UserWarning,
            stacklevel=2,
        )
        return dict(_EMPTY_TRITON_LOG_FIELDS)


def return_dic_zarr_encodings(
    ds: xr.Dataset, clevel: int = 5, *, store_float32: bool = False, time_chunk: int | None = None
) -> dict:
    """
    Create a dictionary of Zarr encodings for an xarray Dataset.

    Uses Blosc compression for numeric variables and preserves
    maximum string length for Unicode coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to encode.
    clevel : int, default=5
        Compression level for Blosc.

    Returns
    -------
    encoding : dict
        Dictionary suitable for xarray.to_zarr(..., encoding=encoding)
    """
    encoding = {}

    # Compressor for numeric data
    import zarr

    compressor = zarr.codecs.BloscCodec(  # type: ignore
        cname="zstd",
        clevel=clevel,
        shuffle=zarr.codecs.BloscShuffle.shuffle,  # type: ignore
    )

    # Handle data variables
    for var in ds.data_vars:  # type: ignore
        dtype_kind = ds[var].dtype.kind
        if dtype_kind in {"i", "u", "f"}:  # int / unsigned int / float
            enc = {"compressors": compressor}
            if store_float32 and dtype_kind == "f":
                enc["dtype"] = "float32"
            if time_chunk is not None and "timestep_min" in ds[var].dims:
                ax = ds[var].dims.index("timestep_min")
                chunks = list(ds[var].shape)
                chunks[ax] = time_chunk
                enc["chunks"] = tuple(chunks)
            encoding[var] = enc
        # Optionally handle other types if needed

    # Handle coordinate encoding
    for coord in ds.coords:  # type: ignore
        dtype_kind = ds[coord].dtype.kind  # type: ignore
        if dtype_kind == "U":  # Unicode string coordinates
            max_len_arr = ds[coord].str.len().max()
            max_len = int(max_len_arr.compute() if hasattr(max_len_arr.data, "compute") else max_len_arr.values)
            encoding[coord] = {"dtype": f"<U{max_len}"}  # type: ignore

    return encoding


def return_dic_autochunk(ds):
    chunk_dict = {}
    for var in ds.dims:
        chunk_dict[var] = "auto"
    return chunk_dict


def estimate_timesteps_per_chunk(
    rds_dem: xr.DataArray,
    n_variables: int,
    memory_budget_MiB: float,
    dtype: Any = None,
) -> int:
    """
    Estimate how many timesteps can fit in memory budget.

    Uses simple memory arithmetic to calculate how many timesteps can be
    loaded simultaneously for all variables within the specified memory budget.
    This is used for chunked processing of TRITON binary outputs.

    Parameters
    ----------
    rds_dem : xr.DataArray
        DEM raster with x and y coordinates (used to get grid dimensions)
    n_variables : int
        Number of variables per timestep (e.g., 4 for H, QX, QY, MH)
    memory_budget_MiB : float
        Target memory budget in MiB
    dtype : np.dtype or None
        Data type (default: np.float64). If None, uses float64.

    Returns
    -------
    int
        Number of timesteps per chunk (minimum 1)

    Examples
    --------
    >>> # For a 513x526 grid with 4 variables and 200 MiB budget
    >>> chunk_size = estimate_timesteps_per_chunk(
    ...     rds_dem=dem,
    ...     n_variables=4,
    ...     memory_budget_MiB=200.0
    ... )
    >>> # Returns number of timesteps that fit in 200 MiB

    Notes
    -----
    Memory calculation:
        memory_per_timestep = n_variables × n_y × n_x × bytes_per_element
        timesteps_per_chunk = memory_budget / memory_per_timestep

    This simple approach is appropriate for timeseries processing where we need
    to determine how much data to load BEFORE creating the dataset. For chunking
    existing datasets for zarr writes, use compute_optimal_chunks() instead.
    """
    import numpy as np

    if dtype is None:
        dtype = np.float64

    n_y = len(rds_dem.y)
    n_x = len(rds_dem.x)
    bytes_per_element = np.dtype(dtype).itemsize

    # Memory for ONE timestep across ALL variables
    memory_per_timestep_bytes = n_variables * n_y * n_x * bytes_per_element
    memory_per_timestep_MiB = memory_per_timestep_bytes / (1024**2)

    # How many timesteps fit in budget?
    timesteps_per_chunk = int(memory_budget_MiB / memory_per_timestep_MiB)

    # Ensure at least 1 timestep per chunk
    return max(1, timesteps_per_chunk)


def prev_power_of_two(n: int | float) -> int:
    """
    Return the largest power of 2 less than or equal to n.

    Parameters
    ----------
    n : int or float
        Input number (must be positive)

    Returns
    -------
    int
        Largest power of 2 <= n

    Examples
    --------
    >>> prev_power_of_two(100)
    64
    >>> prev_power_of_two(256)
    256
    """
    n = int(n)
    if n < 1:
        return 1
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n.bit_length() - 1)


def ds_memory_req_MiB(ds: xr.Dataset) -> float:
    """
    Calculate memory requirement of xarray Dataset in MiB.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to measure

    Returns
    -------
    float
        Memory requirement in MiB
    """
    return ds.nbytes / 1024**2


def compute_optimal_chunks(
    ds: xr.Dataset,
    spatial_coords: list[str] | str | None,
    max_mem_usage_MiB: float,
    spatial_coord_size: int = 65536,  # 256x256 for x,y coords
    verbose: bool = True,
) -> dict | str:
    """
    Compute optimal chunk sizes for writing xarray datasets to disk.

    This function determines chunk sizes that:
    1. Keep memory usage under max_mem_usage_MiB
    2. Use efficient spatial chunks (~256x256 for x,y)
    3. Handle sparse multi-dimensional coordinates (sensitivity analysis)

    Extracted from processing_analysis.py to make it reusable for both
    per-simulation processing and analysis-level consolidation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to compute chunks for
    spatial_coords : List[str] | str | None
        Spatial coordinate names (e.g., ['x', 'y'] or 'node_id').
        If None, returns 'auto'.
    max_mem_usage_MiB : float
        Maximum memory per chunk in MiB
    spatial_coord_size : int
        Target total cells per spatial chunk (default 65536 = 256^2)
    verbose : bool
        Print chunk information if True

    Returns
    -------
    dict or "auto"
        Chunk specification for each dimension

    Examples
    --------
    >>> # For TRITON spatial outputs
    >>> chunks = compute_optimal_chunks(
    ...     ds=ds_triton,
    ...     spatial_coords=["x", "y"],
    ...     max_mem_usage_MiB=200.0
    ... )

    >>> # For SWMM node outputs
    >>> chunks = compute_optimal_chunks(
    ...     ds=ds_swmm_nodes,
    ...     spatial_coords="node_id",
    ...     max_mem_usage_MiB=200.0
    ... )

    Notes
    -----
    This function is used for chunking EXISTING datasets for zarr writes.
    For determining how many timesteps to load during processing, use
    estimate_timesteps_per_chunk() instead.
    """

    # Handle non-spatial data (e.g., performance summaries)
    if spatial_coords is None:
        if verbose:
            print("spatial_coords are None. Returning chunks = 'auto'", flush=True)
        return "auto"

    if isinstance(spatial_coords, str):
        spatial_coords = [spatial_coords]

    # Validation: Check that all spatial coords exist in dataset
    missing_coords = [c for c in spatial_coords if c not in ds.coords]
    if missing_coords:
        error_msg = (
            f"Spatial coordinates {missing_coords} not found in dataset. "
            f"Available coordinates: {list(ds.coords.keys())}"
        )
        raise ValueError(error_msg)

    size_per_spatial_coord = spatial_coord_size ** (1 / len(spatial_coords))

    if len(spatial_coords) not in [1, 2]:
        raise ValueError("Spatial dimension can only be 1 or 2 dimensional")

    lst_non_spatial_coords = []
    for coord in ds.coords:
        if coord not in spatial_coords and coord in ds.dims:
            lst_non_spatial_coords.append(coord)

    # Categorize variables by whether they have spatial dimensions
    spatial_vars = []
    nonspatial_vars = []  # system-wide vars
    for var in ds.data_vars:
        var_dims = set(ds[var].dims)
        if any(coord in var_dims for coord in spatial_coords):
            spatial_vars.append(var)
        else:
            nonspatial_vars.append(var)

    # Get average bytes per element (for float64/float32 estimation)
    # Use first spatial variable if available, otherwise use a default
    if spatial_vars:
        sample_var = ds[spatial_vars[0]]
        bytes_per_element = sample_var.dtype.itemsize
    else:
        bytes_per_element = 8  # default to float64

    # Calculate spatial chunk size first (fixed target)
    chunks: dict = {}
    spatial_chunk_points = 1
    for coord in spatial_coords:
        coord_len = len(ds[coord])
        chunk_size = int(min(size_per_spatial_coord, coord_len))
        chunks[coord] = chunk_size
        spatial_chunk_points *= chunk_size

    # Calculate non-spatial budget accounting for heterogeneous variable shapes
    # Chunk memory = (n_spatial_vars * spatial_points * nonspatial_points +
    #                 n_nonspatial_vars * nonspatial_points) * bytes_per_element
    # Solving for nonspatial_points:
    # nonspatial_points = max_mem_bytes /
    #                     (bytes_per_element * (n_spatial_vars * spatial_points + n_nonspatial_vars))

    bytes_available = max_mem_usage_MiB * 1024**2

    # Calculate the "weight" of one nonspatial point in the chunk
    # Each nonspatial point contributes:
    # - spatial_chunk_points elements for each spatial variable
    # - 1 element for each non-spatial variable
    elements_per_nonspatial_point = len(spatial_vars) * spatial_chunk_points + len(nonspatial_vars)

    if elements_per_nonspatial_point > 0:
        target_nonspatial_points = bytes_available / (bytes_per_element * elements_per_nonspatial_point)
        target_nonspatial_points = max(1, int(target_nonspatial_points))
    else:
        # Edge case: no variables (shouldn't happen in practice)
        target_nonspatial_points = 1

    # Use power-of-2 for better compression
    target_nonspatial_chunk = prev_power_of_two(target_nonspatial_points)

    # Sort non-spatial coords by size (largest first) for better chunking
    sorted_nonspatial = sorted(
        lst_non_spatial_coords,
        key=lambda c: len(ds[c]),
        reverse=True,
    )

    nonspatial_chunk_product = 1
    for coord in sorted_nonspatial:
        coord_len = len(ds[coord])

        # Determine chunk size for this dimension
        if nonspatial_chunk_product >= target_nonspatial_chunk:
            # Already reached target, chunk remaining dims minimally
            chunk_size = 1
        elif coord_len == 1:
            # Singleton dimension
            chunk_size = 1
        else:
            # Calculate how much "budget" remains for chunking
            remaining_budget = target_nonspatial_chunk // nonspatial_chunk_product
            chunk_size = min(coord_len, prev_power_of_two(remaining_budget))
            # Ensure at least some chunking for large dimensions
            if chunk_size < 1:
                chunk_size = 1

        chunks[coord] = chunk_size
        nonspatial_chunk_product *= chunk_size

    # Build test slice to verify memory usage
    test_slice = {}
    for coord, chunk_size in chunks.items():
        test_slice[coord] = slice(0, min(chunk_size, len(ds[coord])))

    # Estimate test chunk memory without forcing rechunking (avoid dask overhead)
    test_ds = ds.isel(test_slice)
    test_size_MiB = ds_memory_req_MiB(test_ds)

    # Validation: Check chunk efficiency
    if test_size_MiB < 1:
        msg = (
            f"Warning: chunks are less than 1 MiB ({test_size_MiB:.3f} MiB), "
            "which could lead to inefficient reading and writing. "
            f"Consider increasing max_mem_usage_MiB or spatial_coord_size."
        )
        print(msg, flush=True)

    if test_size_MiB > max_mem_usage_MiB * 1.2:
        msg = (
            f"Chunk size ({test_size_MiB:.1f} MiB) exceeds "
            f"max_mem_usage_MiB ({max_mem_usage_MiB} MiB). "
            f"Chunks: {chunks}"
        )
        raise ValueError(msg)

    if verbose:
        print(
            f"Memory per chunk: {test_size_MiB:.3f} MiB\nChunks: {chunks}",
            flush=True,
        )

    return chunks


def get_file_size_MiB(f: Path):
    if f.name.split(".")[-1] == "zarr":
        size_bytes = zarr_size_bytes(f)
    else:
        size_bytes = f.stat().st_size
    return size_bytes / 1024**2


def zarr_size_bytes(zarr_path: Path) -> int:
    return sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())


def write_datatree_zarr(
    tree: "xr.DataTree",
    fname_out: Path,
    compression_level: int = 5,
    analysis_dir=None,
) -> None:
    """Write a DataTree to a hierarchical zarr store.

    Builds per-group encoding dicts from `return_dic_zarr_encodings()`
    applied to each populated leaf node's dataset.
    """
    encoding: dict = {}
    for path, node in tree.subtree_with_keys:
        if node.has_data:
            key = path if path.startswith("/") else f"/{path}"
            encoding[key] = return_dic_zarr_encodings(node.dataset, compression_level)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not have a Zarr V3 specification.*",
            category=Warning,
        )
        _publish_store_crash_safe(
            lambda _dest: tree.to_zarr(_dest, mode="w", encoding=encoding, consolidated=False),
            fname_out,
            analysis_dir=analysis_dir,
        )


def write_zarr(ds, fname_out, compression_level, chunks: str | dict = "auto", analysis_dir=None):
    encoding = return_dic_zarr_encodings(ds, compression_level)
    if chunks == "auto":
        chunks = return_dic_autochunk(ds)
    ds = ds.chunk(chunks)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not have a Zarr V3 specification.*",
            category=Warning,
        )
        _publish_store_crash_safe(
            lambda _dest: ds.to_zarr(_dest, mode="w", encoding=encoding, consolidated=False),
            fname_out,
            analysis_dir=analysis_dir,
        )


def write_zarr_then_netcdf(ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"):
    # encoding = return_dic_zarr_encodings(ds, compression_level)
    if chunks == "auto":
        chunks = return_dic_autochunk(ds)
    ds = ds.chunk(chunks)
    # first write to zarr, then write to netcdf
    write_zarr(ds, f"{fname_out}.zarr", compression_level, chunks)
    # open and write
    ds = xr.open_dataset(
        f"{fname_out}.zarr",
        engine="zarr",
        chunks="auto",
        consolidated=False,
        decode_timedelta=False,
    )
    write_netcdf(ds, fname_out, compression_level, chunks)
    # delete zarr
    try:
        # EXEMPT-DU: transient-intermediate
        fast_rmtree(f"{fname_out}.zarr")
    except Exception as e:
        print(f"Could not remove zarr folder {fname_out}.zarr due to error {e}")
    return


def return_dic_netcdf_encodings(ds: xr.Dataset, clevel: int = 5) -> dict:
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind in {"i", "u", "f"}:
            encoding[var] = {"zlib": True, "complevel": clevel, "shuffle": True}
    # Coordinates usually don’t need compression
    return encoding


def write_netcdf(ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"):
    encoding = return_dic_netcdf_encodings(ds, compression_level)
    if chunks == "auto":
        chunk_dict = return_dic_autochunk(ds)
    else:
        chunk_dict = chunks
    try:
        ds = ds.chunk(chunk_dict)
    except NotImplementedError:
        ds = ds.copy(deep=False)
    ds.to_netcdf(fname_out, encoding=encoding, engine="h5netcdf")
    return


def paths_to_strings(obj: Any) -> Any:
    """
    Recursively convert all pathlib.Path objects to strings
    in arbitrarily nested dictionaries / containers.
    """
    if isinstance(obj, Path):
        return str(obj)

    elif isinstance(obj, dict):
        return {k: paths_to_strings(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [paths_to_strings(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(paths_to_strings(v) for v in obj)

    elif isinstance(obj, set):
        return {paths_to_strings(v) for v in obj}

    return obj


def convert_datetime_to_str(obj: Any) -> Any:
    """
    Recursively convert all datetime objects to ISO format strings
    in arbitrarily nested dictionaries / containers.

    This ensures that datetime objects can be serialized to JSON
    when writing xarray datasets to zarr format.
    """
    import pandas as pd

    # Handle datetime objects
    if isinstance(obj, datetime.datetime | pd.Timestamp):
        return obj.isoformat()

    elif isinstance(obj, dict):
        return {k: convert_datetime_to_str(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_datetime_to_str(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_datetime_to_str(v) for v in obj)

    elif isinstance(obj, set):
        return {convert_datetime_to_str(v) for v in obj}

    return obj

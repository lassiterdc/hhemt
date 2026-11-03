"""Tests for the copy primitive at the bundle emit's two remaining per-file copy sites.

``_copy_reference_outputs`` (a directory summary through ``copytree``, a file summary through a
single copy) and ``_copy_supporting_files`` (the ``plots/`` tree through ``copytree``) used to
copy with ``shutil.copy2``. The bundle zip overwrites every entry's date and mode, so the
timestamps and mode ``copy2`` carries are discarded, and each ``copystat`` is extra metadata
calls on the small-file create path that bounds the emit. Both sites now copy with
``shutil.copyfile``.

Every test builds small files under ``tmp_path`` and calls the two private helpers on a
``SimpleNamespace`` stand-in for the analysis, so none compiles or runs a solver and all run on a
laptop. Run from a worktree as ``PYTHONPATH=${PWD}/src pytest tests/test_emit_copy_function.py``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle._emit import _copy_reference_outputs, _copy_supporting_files

# An mtime far from any time a copy could be made at, so a preserved stamp is unmistakable.
_OLD_MTIME = 1_000_000_000  # 2001-09-09


def _analysis(analysis_dir: Path, out_type: str = "zarr") -> SimpleNamespace:
    """The three attributes the two helpers read from an analysis."""
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=SimpleNamespace(target_processed_output_type=out_type),
        case_manifest_yaml=None,
    )


def _age(path: Path) -> None:
    os.utime(path, (_OLD_MTIME, _OLD_MTIME))


def _reference_inputs(analysis_dir: Path, out_type: str) -> list[Path]:
    """One ``_test`` group with a summary of the requested type; returns the source FILES."""
    processed = analysis_dir / "_test" / "group_0" / "sims" / "ev1" / "processed"
    processed.mkdir(parents=True)
    if out_type == "zarr":
        store = processed / "a_summary.zarr"
        (store / "c" / "0").mkdir(parents=True)
        files = [store / "zarr.json", store / "c" / "0" / "0"]
        files[0].write_text("{}")
        files[1].write_bytes(b"chunk")
    else:
        files = [processed / "a_summary.nc"]
        files[0].write_bytes(b"netcdf bytes")
    for file in files:
        _age(file)
    return files


def _plots_inputs(analysis_dir: Path) -> list[Path]:
    plots = analysis_dir / "plots"
    (plots / "sub").mkdir(parents=True)
    files = [plots / "fig.png", plots / "sub" / "fig.manifest.json"]
    files[0].write_bytes(b"png")
    files[1].write_text("{}")
    for file in files:
        _age(file)
    return files


def _forbid_per_file_copystat(monkeypatch) -> None:
    """Make any per-file ``copy2``, ``copy`` or file ``copystat`` raise for the rest of the test.

    ``copytree`` binds its default ``copy_function=copy2`` when ``shutil`` is imported, so patching
    ``shutil.copy2`` does not reach a ``copytree`` call that leaves the default; ``copy2`` itself
    calls ``shutil.copystat`` through the module namespace, so refusing a ``copystat`` whose source
    is a FILE reaches every route to the per-file stamp copy. Directory ``copystat`` calls (the
    one ``copytree`` makes per directory) are passed through, because they are not per-file work.
    """
    real_copystat = shutil.copystat

    def _copystat(src, dst, *args, **kwargs):
        if os.path.isfile(src):
            raise AssertionError(f"copystat reached for the file {src}: this copy site must not copy stamps per file")
        return real_copystat(src, dst, *args, **kwargs)

    def _refuse(name: str):
        def _raiser(*_args, **_kwargs):
            raise AssertionError(f"shutil.{name} reached: this copy site must not use it")

        return _raiser

    monkeypatch.setattr(shutil, "copystat", _copystat)
    monkeypatch.setattr(shutil, "copy2", _refuse("copy2"))
    monkeypatch.setattr(shutil, "copy", _refuse("copy"))


@pytest.mark.parametrize("out_type", ["zarr", "nc"])
def test_reference_outputs_copy_does_not_reach_copy2(tmp_path, monkeypatch, out_type):
    """Property: ``_copy_reference_outputs`` copies no file with ``copy2``, ``copy`` or a file ``copystat``.

    Class: REGRESSION-shaped discriminator. Before this change both arms (a directory summary
    through ``copytree``'s default copy function, a file summary through a direct ``copy2``)
    copy stamps per file and the guard reds. The input class is the processed output type, both
    values, so the two code arms are each reached. The plausible wrong implementation is leaving
    either arm on ``copy2``. Why required: the copies sit on the small-file create path and their
    stamps are discarded by the zip.

    A second correct implementation is a hand-written read-and-write loop, which reaches none of
    the guarded names, so the test still passes under it.
    """
    analysis_dir = tmp_path / "analysis"
    _reference_inputs(analysis_dir, out_type)
    _forbid_per_file_copystat(monkeypatch)
    staging = tmp_path / "staging"
    _copy_reference_outputs(_analysis(analysis_dir, out_type), staging)
    assert any(p.is_file() for p in staging.rglob("*")), "the copy must carry at least one file"


def test_plots_copy_does_not_reach_copy2(tmp_path, monkeypatch):
    """Property: ``_copy_supporting_files`` copies the ``plots/`` tree without per-file stamp copies.

    Class: REGRESSION-shaped discriminator: before this change ``copytree`` used its default
    ``copy2`` for every figure and the guard reds. Only ``plots/`` exists under the stand-in
    analysis directory, so the helper's other single-file ``copy2`` calls (Snakefile, version file,
    status csv, case manifest, disk-usage sentinel) are not reached and the guard fires only for
    the tree copy. The plausible wrong implementation is ``copytree`` without a ``copy_function``.
    Why required: ``plots/`` holds a file per figure and per manifest sidecar, the largest per-file
    population this helper copies.

    A second correct implementation is ``shutil.copytree`` with a hand-written copy function that
    reaches none of the guarded names.
    """
    analysis_dir = tmp_path / "analysis"
    _plots_inputs(analysis_dir)
    _forbid_per_file_copystat(monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()
    _copy_supporting_files(_analysis(analysis_dir), staging)
    assert (staging / "plots" / "sub" / "fig.manifest.json").is_file()


def test_both_sites_copy_with_copyfile(tmp_path, monkeypatch):
    """Property: every file the two sites carry is copied by ``shutil.copyfile``.

    Class: NEW CAPABILITY, pinning the primitive the approved design names. Not established as a
    discriminator on its own: ``copy2`` and ``copy`` end in ``copyfile``, so a spy on ``copyfile``
    alone is green on the old code too. This test therefore also forbids per-file stamp copies, and
    what it adds over the two tests above is the positive claim that the primitive reached is
    ``copyfile``. The plausible wrong implementation is a hand-written loop, which the design does
    not name; it is rejected here and only here, so the two tests above accept such an
    implementation while this one pins the stated primitive.

    A second correct implementation under this test is ``shutil.copyfile`` called through
    ``copytree``'s ``copy_function`` or directly, which is what both arms use.
    """
    analysis_dir = tmp_path / "analysis"
    sources = _reference_inputs(analysis_dir, "zarr") + _plots_inputs(analysis_dir)
    seen: list[str] = []
    real_copyfile = shutil.copyfile

    def _spy(src, dst, *args, **kwargs):
        seen.append(str(src))
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", _spy)
    _forbid_per_file_copystat(monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()
    analysis = _analysis(analysis_dir)
    _copy_reference_outputs(analysis, staging)
    _copy_supporting_files(analysis, staging)
    assert sorted(seen) == sorted(str(s) for s in sources)


@pytest.mark.parametrize("site", ["reference_outputs", "plots"])
def test_copies_carry_the_bytes_but_not_the_source_mtime(tmp_path, site):
    """Property: the copied files hold the source bytes and take a fresh mtime, not the source's.

    Class: NEW CAPABILITY and a behavioural discriminator that does not name the primitive: under
    ``copy2`` the destination's mtime equals the source's (2001-09-09 here), under any copy that
    does not copystat it is the copy time. The input class is the two sites; the satisfying input
    is that the bytes still match exactly. The plausible wrong implementation is ``copy2`` kept at
    either site. Why required: the zip stamps every entry itself, so a preserved mtime has no
    effect on the bundle and its cost is the point of the change.

    Not established: this does not prove the copy skipped ``copystat``'s other work (mode, flags),
    only that the mtime step did not run. A second correct implementation is ``shutil.copyfile``
    or a hand-written loop; both give a fresh mtime and identical bytes.
    """
    analysis_dir = tmp_path / "analysis"
    if site == "reference_outputs":
        sources = _reference_inputs(analysis_dir, "zarr")
    else:
        sources = _plots_inputs(analysis_dir)
    staging = tmp_path / "staging"
    staging.mkdir()
    analysis = _analysis(analysis_dir)
    if site == "reference_outputs":
        _copy_reference_outputs(analysis, staging)
        root = analysis_dir / "_test" / "group_0" / "sims" / "ev1" / "processed"
        dest_root = staging / "reference_outputs" / "group_0" / "sims" / "ev1" / "processed"
    else:
        _copy_supporting_files(analysis, staging)
        root = analysis_dir / "plots"
        dest_root = staging / "plots"
    for source in sources:
        dest = dest_root / source.relative_to(root)
        assert dest.read_bytes() == source.read_bytes(), dest
        assert int(dest.stat().st_mtime) != _OLD_MTIME, f"{dest} kept the source mtime"

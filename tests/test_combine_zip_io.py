"""Tests for ``combine_bundle`` taking member zips or directories and working in a private folder.

Covers the combine's input side: each input is a bundle directory or a member zip, its render set
(everything but raw solver output) is extracted into a working folder, the checks and read-models
run against that folder, and nothing is written at the output until the render has succeeded.

Every bundle here is a small stub built under ``tmp_path``: a manifest, two config files, a tiny
consolidated tree, a status CSV and a few raw-output files. The seams that need a real analysis
config (experiment identity, the compatibility checker, the combined manifest) and the Snakemake
render are replaced with small stand-ins; extraction, the metadata-only refusal, the root-tree
check, the read-model writers, the working folder and the publish step are the real code. No
solver is compiled or run, so every test runs on a laptop. Run as
``PYTHONPATH=${PWD}/src pytest tests/test_combine_zip_io.py``.

Each test docstring names its property, its class (NEW CAPABILITY unless marked: the behaviour
did not exist before this chunk), the plausible wrong implementation that turns it red, and a
second correct implementation that stays green.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import yaml

from hhemt.bundle import _combine as C
from hhemt.exceptions import ProcessingError
from hhemt.version_migration.constants import BUNDLE_SCHEMA_VERSION, COMBINED_BUNDLE_SCHEMA_VERSION

_CPU = {
    "run_mode": "clean",
    "n_mpi_procs": 1,
    "n_omp_threads": 1,
    "n_gpus": 0,
    "n_nodes": 1,
    "hpc.partition": "standard",
}
_PIN = "a38338b09e62e57c936f51516bbdbe495d89a546"


def _stub_bundle(root: Path, eid: str, *, resumes: int, shallow: bool = False, wlevel: float = 1.0) -> Path:
    """A stub bundle directory with raw solver output, a render set, and a consolidated tree."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                "metadata_only_stores": ["analysis_datatree.zarr"] if shallow else [],
            }
        )
    )
    (root / "cfg_system.yaml").write_text(f"TRITONSWMM_branch_key: {_PIN}\n")
    (root / "cfg_analysis.yaml").write_text(yaml.safe_dump({"analysis_id": eid}))
    (root / "scenario_status.csv").write_text(f"event,n_resumes\n0,{resumes}\n")
    (root / "plots").mkdir(exist_ok=True)
    (root / "plots" / "fig.txt").write_text(f"figure of {eid}\n")
    for i in range(3):
        raw = root / "sims" / str(i) / "out_triton" / "bin"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"H_{i}.bin").write_bytes(os.urandom(2048))
        (root / "sims" / str(i) / "out_triton" / "run.cfg").write_text(f"cfg {i}\n")
    groups = {
        "/": xr.Dataset(),
        "/member_0": xr.Dataset(attrs=_CPU),
        "/member_0/tritonswmm": xr.Dataset(),
        "/member_0/tritonswmm/triton": xr.Dataset(
            {"max_flood_depth": ("event_iloc", np.asarray([wlevel, wlevel + 1.0]))},
            coords={"event_iloc": [0, 1]},
        ),
        "/member_0/tritonswmm/performance": xr.Dataset({"Total": ("Rank", np.asarray([10.0]))}),
    }
    xr.DataTree.from_dict(groups).to_zarr(root / "analysis_datatree.zarr", mode="w", consolidated=False)
    return root


def _zip_dir(src: Path, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())
    return dest


def _digest_tree(root: Path) -> dict[str, tuple[str, int]]:
    """``{relative path: (sha256, mtime_ns)}`` for every file under ``root``."""
    return {
        p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def seams(monkeypatch):
    """Stand-ins for the seams that need a real analysis config or Snakemake; everything else is real."""
    render_calls: list[dict] = []

    def fake_experiment_id(root: Path) -> str:
        return yaml.safe_load((Path(root) / "cfg_analysis.yaml").read_text())["analysis_id"]

    def fake_compat(roots, declare_solver_split=False):
        return SimpleNamespace(is_compatible=True, divergences=[], blocking=[])

    def fake_manifest(output_path, *, experiment_ids, child_crates, git_sha, roots):
        (output_path / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "bundle_schema_version": COMBINED_BUNDLE_SCHEMA_VERSION,
                    "experiment_ids": list(experiment_ids),
                    "child_crates": list(child_crates),
                },
                sort_keys=True,
            )
        )

    def fake_render(merged, report, output_path, **kwargs):
        render_calls.append(
            {
                "root": Path(output_path),
                "children": sorted(p.name for p in (Path(output_path) / "child_crates").iterdir()),
                "files": sorted(
                    p.relative_to(output_path).as_posix() for p in Path(output_path).rglob("*") if p.is_file()
                ),
            }
        )

    monkeypatch.setattr(C, "_experiment_id", fake_experiment_id)
    monkeypatch.setattr(C, "check_bundle_compatibility", fake_compat)
    monkeypatch.setattr(C, "_write_combined_bundle_manifest", fake_manifest)
    monkeypatch.setattr(C, "_render_combined_report", fake_render)
    return SimpleNamespace(render_calls=render_calls)


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    """One clean and one resume bundle of the same model arm, both as directories."""
    a = _stub_bundle(tmp_path / "in" / "clean", "exp_clean_tritonswmm", resumes=0)
    b = _stub_bundle(tmp_path / "in" / "resume", "exp_resume_tritonswmm", resumes=1)
    return a, b


def test_a_combine_leaves_every_input_directory_and_zip_unchanged(tmp_path, seams):
    """Property: after a combine every input zip and input directory has the same bytes and mtimes as before.

    Class: REGRESSION guard on today's copy path (a discriminator against any implementation that moves
    or edits an input). Wrong implementation: extracting by renaming the input's folder into the working
    folder, or writing the working folder next to the input; red because the directory vanishes or its
    mtimes change. Reachability: one directory and one zip are inputs, the two kinds the combine accepts.
    Both-states anchor: the check compares hashes and mtimes of the inputs themselves. Invocation: the
    module is new in this chunk's path, so discrimination is the mutation `_DirectorySource.extract`
    using `shutil.move`. Second correct implementation: copying the whole input into a temporary folder
    and extracting from the copy.
    """
    clean, resume = _pair(tmp_path)
    zip_in = _zip_dir(resume, tmp_path / "in" / "resume.zip")
    # the zip is the stand-in for the second input; the directory copy stays out of the call
    before = {"dir": _digest_tree(clean), "zip": hashlib_sha(zip_in), "zip_mtime": zip_in.stat().st_mtime_ns}
    out = tmp_path / "out"
    C.combine_bundle([clean, zip_in], out, workdir=tmp_path)
    assert _digest_tree(clean) == before["dir"]
    assert hashlib_sha(zip_in) == before["zip"] and zip_in.stat().st_mtime_ns == before["zip_mtime"]
    assert out.with_name("out.zip").is_file()


def hashlib_sha(path: Path) -> str:
    """sha256 of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_working_folder_holds_the_render_set_and_the_unzipped_output_holds_full_members(tmp_path, seams):
    """Property: children are extracted without ``out_*/bin`` files, and the render runs on that folder.

    Class: NEW CAPABILITY. Wrong implementation: extracting every file (the raw ``.bin`` files appear),
    or rendering against the output path; red on the raw-file assertion and on the root assertion.
    Reachability: the stubs carry three raw files each under ``sims/{i}/out_triton/bin`` and a
    sibling ``run.cfg`` that must survive. Both-states anchor: the render recorder reads the folder at
    render time. Second correct implementation: extracting everything and deleting ``out_*/bin`` after.
    """
    clean, resume = _pair(tmp_path)
    out = tmp_path / "out"
    C.combine_bundle([clean, resume], out, workdir=tmp_path, unzipped=True)
    (call,) = seams.render_calls
    assert call["root"] != out.resolve() and call["root"].parent == tmp_path.resolve()
    assert call["children"] == ["exp_clean_tritonswmm", "exp_resume_tritonswmm"]
    files = call["files"]
    assert not any("/out_triton/bin/" in f for f in files)
    assert "child_crates/exp_clean_tritonswmm/sims/0/out_triton/run.cfg" in files
    assert "child_crates/exp_clean_tritonswmm/analysis_datatree.zarr/zarr.json" in files
    # the unzipped output holds each FULL member (raw files included), as the input does
    assert len(list(out.rglob("H_*.bin"))) == 6
    assert len(list((clean / "sims").rglob("H_*.bin"))) == 3


def test_a_directory_and_a_zip_of_the_same_bundle_give_identical_combined_root_files(tmp_path, seams):
    """Property: a bundle given as a directory and as its zip gives byte-identical read-models and root files.

    Class: NEW CAPABILITY. Wrong implementation: a zip source whose names or bytes differ from the
    directory's (a leading folder kept, a file missed); red on the file-set or byte comparison.
    Reachability: both runs combine the same two bundles, once as directories and once with one of
    them zipped. Both-states anchor: two combined outputs are compared with each other. Second correct
    implementation: unzipping each zip to a folder first and treating it as a directory input.
    """
    clean, resume = _pair(tmp_path)
    zip_resume = _zip_dir(resume, tmp_path / "in" / "resume.zip")
    out_dirs, out_mixed = tmp_path / "o_dirs", tmp_path / "o_mixed"
    C.combine_bundle([clean, resume], out_dirs, workdir=tmp_path, unzipped=True)
    C.combine_bundle([clean, zip_resume], out_mixed, workdir=tmp_path, unzipped=True)

    def root_files(out: Path) -> dict[str, str]:
        return {p.name: hashlib_sha(p) for p in sorted(out.iterdir()) if p.is_file()}

    a, b = root_files(out_dirs), root_files(out_mixed)
    assert "combined_intercomparison.json" in a and "cross_solver_comparison.json" in a
    assert a == b
    child_a = {
        p.relative_to(out_dirs).as_posix(): hashlib_sha(p)
        for p in (out_dirs / "child_crates").rglob("*")
        if p.is_file()
    }
    child_b = {
        p.relative_to(out_mixed).as_posix(): hashlib_sha(p)
        for p in (out_mixed / "child_crates").rglob("*")
        if p.is_file()
    }
    assert {k.replace("o_dirs", "") for k in child_a} == {k.replace("o_mixed", "") for k in child_b}
    assert child_a == child_b


def test_read_models_are_identical_between_full_members_and_the_render_set(tmp_path, seams, monkeypatch):
    """Property: read-models and root files match whether the working folder holds FULL members or the render set.

    Class: REGRESSION guard for the claim that no combine reader opens raw solver output; the
    must-fire control below shows the comparison can fail. Wrong implementation: a render set that
    drops a file a reader opens (here the control drops ``scenario_status.csv``); red on the control
    run. Reachability: the stubs carry clean and resume arms, so the status CSV decides the pairing
    in ``combined_intercomparison.json``. Both-states anchor: three outputs are compared with each
    other, never with a stored expectation. Second correct implementation: a render set computed
    from a deny-list of file suffixes instead of the ``out_*/bin`` path rule. Not covered: the
    Snakemake report itself, which this test replaces with a recorder.
    """
    clean, resume = _pair(tmp_path)

    def root_files(out: Path) -> dict[str, str]:
        return {p.name: hashlib_sha(p) for p in sorted(out.iterdir()) if p.is_file()}

    out_set = tmp_path / "o_set"
    C.combine_bundle([clean, resume], out_set, workdir=tmp_path, unzipped=True)

    monkeypatch.setattr(C, "render_set", lambda names: list(names))
    out_full = tmp_path / "o_full"
    C.combine_bundle([clean, resume], out_full, workdir=tmp_path, unzipped=True)
    assert root_files(out_set) == root_files(out_full)

    def without_status(names):
        return [n for n in names if n != "scenario_status.csv" and "/out_" not in n]

    monkeypatch.setattr(C, "render_set", without_status)
    out_control = tmp_path / "o_control"
    C.combine_bundle([clean, resume], out_control, workdir=tmp_path, unzipped=True)
    assert root_files(out_control) != root_files(out_set)


def test_a_metadata_only_member_zip_is_refused_before_the_working_folder_or_output_exists(tmp_path, seams, monkeypatch):
    """Property: a member zip whose manifest names its root tree metadata-only is refused with nothing created.

    Class: REGRESSION guard for the existing refusal operation text, now raised before extraction
    (the refusal ran only at read-model time before). Wrong implementation: refusing after the
    working folder is made, or only for a directory input; red on the working-folder listing and on
    the zip case. Reachability: the second input is a ZIP whose manifest lists ``analysis_datatree.zarr``
    under ``metadata_only_stores``. Both-states anchor: ``mkdtemp`` is never called, ``workdir`` is empty
    and the output absent after the raise. Second correct implementation: reading the manifest from the zip's central
    directory with ``zipfile`` before any ``mkdtemp``.
    """
    clean = _stub_bundle(tmp_path / "in" / "clean", "exp_clean_tritonswmm", resumes=0)
    shallow_dir = _stub_bundle(tmp_path / "in" / "shallow", "exp_resume_tritonswmm", resumes=1, shallow=True)
    shallow_zip = _zip_dir(shallow_dir, tmp_path / "in" / "shallow.zip")
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "out"
    made: list[str] = []
    real_mkdtemp = C.tempfile.mkdtemp
    monkeypatch.setattr(C.tempfile, "mkdtemp", lambda *a, **k: made.append("mkdtemp") or real_mkdtemp(*a, **k))
    with pytest.raises(ProcessingError, match="metadata-only") as excinfo:
        C.combine_bundle([clean, shallow_zip], out, workdir=work)
    assert excinfo.value.operation == "combine: child store carried metadata-only"
    assert list(work.iterdir()) == [] and not out.exists()
    with pytest.raises(ProcessingError):
        C.combine_bundle([clean, shallow_dir], out, workdir=work)
    assert list(work.iterdir()) == [] and not out.exists()
    assert made == [], "the working folder was created before the refusal"
    assert seams.render_calls == []


def test_each_child_tree_is_opened_once_across_the_combine(tmp_path, seams, monkeypatch):
    """Property: a combine over N bundles opens each child's consolidated tree once in total.

    Class: NEW CAPABILITY (a discriminator: the same count measured on the tree before this chunk
    is pasted in the round record). Wrong implementation: opening again in each read-model writer
    or in a merge step; red because the count exceeds N. Reachability: the two stubs are a clean
    and a resume arm of one model, so BOTH writers read both stores. Both-states anchor: the count
    is of ``xarray.open_datatree`` calls themselves. Second correct implementation: holding the opened
    trees in a dict owned by the combine entry point and passing it to both writers.
    """
    clean, resume = _pair(tmp_path)
    calls: list[str] = []
    real = xr.open_datatree

    def counting(path, *a, **k):
        calls.append(str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(xr, "open_datatree", counting)
    C.combine_bundle([clean, resume], tmp_path / "out", workdir=tmp_path)
    assert len(calls) == 2
    assert len({Path(c).parent.name for c in calls}) == 2


def test_a_failed_render_leaves_no_output_and_no_working_folder(tmp_path, seams, monkeypatch):
    """Property: when the render raises, no output folder exists and the working folder is removed.

    Class: NEW CAPABILITY (discriminator: before this chunk a failed render left a partial output
    with ``child_crates/``). Wrong implementation: writing the output first or skipping the cleanup
    on error; red on either assertion. Reachability: the render seam raises after the read-models
    exist. Both-states anchor: the filesystem is read after the raise. Second correct implementation:
    building into ``output.partial`` and renaming at the end.
    """
    clean, resume = _pair(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("render failed")

    monkeypatch.setattr(C, "_render_combined_report", boom)
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="render failed"):
        C.combine_bundle([clean, resume], out, workdir=work)
    assert not out.exists()
    assert list(work.iterdir()) == []


def test_a_zip_input_that_is_not_a_bundle_archive_is_refused_with_a_clear_error(tmp_path, seams):
    """Property: an input that is neither a directory nor a zip raises ``ConfigurationError`` naming it.

    Class: NEW CAPABILITY. Wrong implementation: passing a plain file to ``Bundle`` and surfacing
    ``FileNotFoundError: No bundle_manifest.json`` for something that was never a bundle; red on
    the exception type. Reachability: a text file is an ordinary mistake on a command line.
    Both-states anchor: the type and the path in the message. Second correct implementation: checking
    ``zipfile.is_zipfile`` and ``is_dir`` in the CLI layer before the call.
    """
    from hhemt.exceptions import ConfigurationError

    clean, _ = _pair(tmp_path)
    junk = tmp_path / "notes.txt"
    junk.write_text("not a bundle")
    with pytest.raises(ConfigurationError, match="notes.txt"):
        C.combine_bundle([clean, junk], tmp_path / "out", workdir=tmp_path)


def test_two_inputs_with_one_experiment_id_get_distinct_child_folders(tmp_path, seams):
    """Property: two bundles with the same ``analysis_id`` become ``eid`` and ``eid__{index}`` child folders.

    Class: NEW CAPABILITY. Wrong implementation: extracting both to ``child_crates/{eid}`` (the
    second overwrites the first); red on the child list. Reachability: a bundle and its copy carry
    one id. Both-states anchor: the recorder lists the folder names at render time. Second correct
    implementation: numbering every child, not only the repeat.
    """
    a = _stub_bundle(tmp_path / "in" / "a", "exp_same_tritonswmm", resumes=0)
    b = _stub_bundle(tmp_path / "in" / "b", "exp_same_tritonswmm", resumes=1)
    C.combine_bundle([a, b], tmp_path / "out", workdir=tmp_path)
    (call,) = seams.render_calls
    assert call["children"] == ["exp_same_tritonswmm", "exp_same_tritonswmm__1"]


def test_combine_command_accepts_member_zips_and_passes_workdir(tmp_path, monkeypatch):
    """Property: the CLI accepts a ``.zip`` among the bundle arguments and forwards ``--workdir``.

    Class: NEW CAPABILITY. Wrong implementation: keeping ``file_okay=False`` on the argument (typer
    rejects the zip before the call) or dropping ``workdir`` on the way to ``combine_bundle``; red on
    the exit code and on the recorded argument. Reachability: one zip and one directory are
    arguments. Both-states anchor: the recorded call is what ``combine_bundle`` received. Second
    correct implementation: a click callback validating each path itself.
    """
    from typer.testing import CliRunner

    import hhemt.bundle as bundle_pkg
    from hhemt.cli import app

    clean = _stub_bundle(tmp_path / "in" / "clean", "exp_clean_tritonswmm", resumes=0)
    zip_in = _zip_dir(_stub_bundle(tmp_path / "in" / "r", "exp_resume_tritonswmm", resumes=1), tmp_path / "r.zip")
    seen: dict = {}

    def fake_combine(bundles, output_path=None, *, declare_solver_split=False, workdir=None, unzipped=False):
        seen.update(
            bundles=list(bundles), workdir=workdir, output=output_path, unzipped=unzipped, split=declare_solver_split
        )
        return SimpleNamespace(root=tmp_path / "out")

    monkeypatch.setattr(bundle_pkg, "combine_bundle", fake_combine)
    work = tmp_path / "work"
    work.mkdir()
    result = CliRunner().invoke(app, ["combine", str(clean), str(zip_in), "--workdir", str(work), "--unzipped"])
    assert result.exit_code == 0, result.output
    assert [Path(p).name for p in seen["bundles"]] == ["clean", "r.zip"]
    assert seen["workdir"] == work
    assert seen["unzipped"] is True and seen["split"] is False


def test_an_input_without_a_manifest_is_reported_by_the_bundle_loader_and_leaves_no_working_folder(tmp_path):
    """Property: a directory with no manifest passes the metadata-only pass and fails with the loader message.

    Class: NEW CAPABILITY (the pre-pass must not raise ``KeyError`` for an absent manifest). Wrong
    implementation: letting the manifest read's ``KeyError`` escape the pre-pass; red on the
    exception type. Reachability: two directories that are not bundles are an ordinary input
    mistake, and no seam is patched, so the real identity loader runs. Both-states anchor: the
    raised type and message come from ``Bundle.from_directory``; ``workdir`` is empty afterwards.
    Second correct implementation: checking for the manifest name in the source's file list first.
    """
    for name in ("a", "b"):
        (tmp_path / "in" / name).mkdir(parents=True)
        (tmp_path / "in" / name / "cfg_system.yaml").write_text(f"TRITONSWMM_branch_key: {_PIN}\n")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(FileNotFoundError, match="bundle_manifest.json"):
        C.combine_bundle([tmp_path / "in" / "a", tmp_path / "in" / "b"], tmp_path / "out", workdir=work)
    assert list(work.iterdir()) == [] and not (tmp_path / "out").exists()


def test_a_directory_link_that_points_at_an_ancestor_is_skipped_and_the_listing_ends(tmp_path):
    """Property: a symlink to the bundle or to a directory above the walk is not followed, so ``names()`` ends.

    Class: NEW CAPABILITY (the guard that comes with following directory links). Wrong
    implementation: ``os.walk(..., followlinks=True)`` with no guard; it descends ``loop/loop/...``
    until the path is too long and raises ``OSError``; red on the raise. Reachability: a link from
    a subfolder back to the bundle root, and one to a sibling directory that holds no cycle (which
    must still be followed). Both-states anchor: the listing is compared with a literal. Second
    correct implementation: remembering the real path of every directory entered and skipping one
    already seen on the current path.
    """
    bundle = tmp_path / "bundle"
    (bundle / "sub").mkdir(parents=True)
    (bundle / "top.txt").write_text("t")
    (bundle / "sub" / "inner.txt").write_text("i")
    os.symlink(bundle, bundle / "sub" / "loop")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "s.txt").write_text("s")
    os.symlink(shared, bundle / "linked")
    assert C._DirectorySource(bundle).names() == ["linked/s.txt", "sub/inner.txt", "top.txt"]


def _zip_names_types(path: Path) -> dict[str, int]:
    with zipfile.ZipFile(path) as zf:
        return {i.filename: i.compress_type for i in zf.infolist()}


def _policy_zip(src: Path, dest: Path) -> Path:
    """A member zip written the way ``write_zip`` writes one (moderate policy)."""
    from hhemt.bundle._archive import write_zip

    write_zip(((p.relative_to(src).as_posix(), p) for p in sorted(src.rglob("*")) if p.is_file()), dest)
    return dest


def test_the_default_output_is_one_zip_with_member_zips_root_files_and_refreshed_figures(tmp_path, seams):
    """Property: the default output is ``{output}.zip`` with member zips, root files and refreshed figures.

    Class: NEW CAPABILITY (discriminator: before this chunk the output was a folder). Wrong
    implementation: a folder, or members spread as ``child_crates/{eid}/...``; red on the entry list.
    Reachability: one input is a policy zip (copied whole, stored), one is a STORED-entry zip
    (an old member; compressed as one entry) and the refreshed figure is a file the child holds at
    the re-render path. Both-states anchor: the entry names are compared with a literal set;
    ``return_value.root`` and ``is_archive`` read from the returned handle. Second correct
    implementation: assembling the zip with ``zipfile`` directly.
    """
    clean, resume = _pair(tmp_path)
    fig = clean / "plots" / "appendix" / "scenario_status.html"
    fig.parent.mkdir(parents=True)
    fig.write_text("figure")
    new_zip = _policy_zip(clean, tmp_path / "in" / "clean.zip")
    old_zip = tmp_path / "in" / "resume.zip"
    with zipfile.ZipFile(old_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(resume.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(resume).as_posix())
    out = tmp_path / "out"
    cb = C.combine_bundle([new_zip, old_zip], out, workdir=tmp_path)
    zip_path = tmp_path / "out.zip"
    assert cb.root == zip_path.resolve() and cb.is_archive is True
    types = _zip_names_types(zip_path)
    names = set(types)
    assert {"bundle_manifest.json", "combined_compatibility.json", "combined_intercomparison.json"} <= names
    assert not any(n.startswith("child_crates/") and n.count("/") > 1 and ".refreshed" not in n for n in names)
    assert {"child_crates/exp_clean_tritonswmm.zip", "child_crates/exp_resume_tritonswmm.zip"} <= names
    assert "child_crates/exp_clean_tritonswmm.refreshed/plots/appendix/scenario_status.html" in names
    assert types["child_crates/exp_clean_tritonswmm.zip"] == zipfile.ZIP_STORED
    assert types["child_crates/exp_resume_tritonswmm.zip"] == zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read("child_crates/exp_clean_tritonswmm.zip") == new_zip.read_bytes()
        inner = zf.read("child_crates/exp_resume_tritonswmm.zip")
        assert hashlib.sha256(inner).hexdigest() == hashlib_sha(old_zip)
        manifest = json.loads(zf.read("bundle_manifest.json"))
    assert manifest["child_crates"] == [
        "child_crates/exp_clean_tritonswmm.zip",
        "child_crates/exp_resume_tritonswmm.zip",
    ]
    assert not out.exists()


def test_an_output_ending_zip_is_used_as_given_and_a_rerun_replaces_it_without_a_partial(tmp_path, seams):
    """Property: ``--output x.zip`` is used as given; a second combine replaces it and leaves no ``.partial``.

    Class: NEW CAPABILITY. Wrong implementation: appending ``.zip`` again (``x.zip.zip``) or
    refusing to overwrite; red on the file list. Reachability: the same two directory inputs
    combined twice. Both-states anchor: the directory listing is read after each run. Second
    correct implementation: unlink then write.
    """
    clean, resume = _pair(tmp_path)
    out = tmp_path / "x.zip"
    C.combine_bundle([clean, resume], out, workdir=tmp_path)
    first = out.read_bytes()
    out.write_bytes(b"stale")
    C.combine_bundle([clean, resume], out, workdir=tmp_path)
    assert out.read_bytes() != b"stale" and zipfile.is_zipfile(out)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("x")) == ["x.zip"]
    assert first[:2] == b"PK"


def test_unzipped_replaces_an_existing_folder_and_holds_refreshed_figures_over_full_members(tmp_path, seams):
    """Property: ``unzipped=True`` replaces an existing folder, leaves no ``.partial``, keeps refreshed figures.

    Class: NEW CAPABILITY. Wrong implementation: merging into the old folder (the stale file
    survives) or leaving ``.partial``; red on the stale-file and listing assertions. Reachability:
    the old folder holds a file the new output lacks. Both-states anchor: the filesystem is read.
    Second correct implementation: removing the old folder first, then renaming.
    """
    clean, resume = _pair(tmp_path)
    fig = clean / "plots" / "appendix" / "scenario_status.html"
    fig.parent.mkdir(parents=True)
    fig.write_text("figure")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    cb = C.combine_bundle([clean, resume], out, workdir=tmp_path, unzipped=True)
    assert cb.is_archive is False and cb.root == out.resolve()
    assert not (out / "stale.txt").exists()
    assert (out / "child_crates/exp_clean_tritonswmm/plots/appendix/scenario_status.html").read_text() == "figure"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("out")) == ["out"]


def test_a_sigterm_during_the_render_removes_the_working_folder_and_still_terminates(tmp_path, seams, monkeypatch):
    """Property: SIGTERM mid-combine removes the working folder and the process still dies of SIGTERM.

    Class: NEW CAPABILITY (discriminator: a ``finally`` alone does not run on SIGTERM). Wrong
    implementation: no handler (folder left behind), or a handler that swallows the signal
    (the process lives on); red on the folder check or the exit status. Reachability: the child
    process sends itself SIGTERM from inside the render seam. Both-states anchor: the child's
    return code and the working directory listing, read by this process. Second correct
    implementation: a handler that raises ``SystemExit`` after removing the folder.
    """
    import subprocess
    import sys
    import textwrap

    clean, resume = _pair(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    script = tmp_path / "child.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, signal, sys
            sys.path.insert(0, {str(Path(__file__).parent.parent / "tests")!r})
            from pathlib import Path
            from types import SimpleNamespace
            import yaml
            from hhemt.bundle import _combine as C
            C._experiment_id = lambda r: yaml.safe_load((Path(r)/"cfg_analysis.yaml").read_text())["analysis_id"]
            C.check_bundle_compatibility = lambda roots, declare_solver_split=False: SimpleNamespace(
                is_compatible=True, divergences=[], blocking=[]
            )
            def boom(*a, **k):
                os.kill(os.getpid(), signal.SIGTERM)
                import time; time.sleep(5)
            C._render_combined_report = boom
            C.combine_bundle([Path({str(clean)!r}), Path({str(resume)!r})], Path({str(tmp_path / "o")!r}),
                             workdir=Path({str(work)!r}))
            """
        )
    )
    env = dict(os.environ, PYTHONPATH=str(Path(C.__file__).parents[2]))
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, timeout=120)
    assert proc.returncode == -15, proc.stderr[-400:]
    assert list(work.iterdir()) == []
    assert not (tmp_path / "o.zip").exists() and not (tmp_path / "o.zip.partial").exists()


def test_stale_working_folders_are_removed_at_start_and_live_ones_are_kept(tmp_path, seams, monkeypatch):
    """Property: a ``hhemt_combine_*`` folder of a job absent from ``squeue`` is removed; a listed job's is kept.

    Class: NEW CAPABILITY. Wrong implementation: no sweep, or one that removes every such folder
    (a running job's working folder would vanish); red on either assertion. Reachability: three
    folders, for a dead job, a live job and a ``local`` folder made old by its mtime; ``squeue`` is
    stubbed (the venue has none). Both-states anchor: the directory listing. Second correct
    implementation: asking ``squeue -j`` once per folder.
    """
    import time

    work = tmp_path / "work"
    work.mkdir()
    for name in (
        "hhemt_combine_111_aaaa",
        "hhemt_combine_222_bbbb",
        "hhemt_combine_local_cccc",
        "hhemt_combine_local_dddd",
    ):
        (work / name).mkdir()
        (work / name / "f").write_text("x")
    old = time.time() - 3 * 86400
    os.utime(work / "hhemt_combine_local_cccc", (old, old))
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="222\n", returncode=0))
    clean, resume = _pair(tmp_path)
    C.combine_bundle([clean, resume], tmp_path / "out", workdir=work)
    left = sorted(p.name for p in work.iterdir())
    assert left == ["hhemt_combine_222_bbbb", "hhemt_combine_local_dddd"]


def test_the_combined_zip_is_deterministic_for_one_set_of_inputs(tmp_path, seams):
    """Property: two combines of the same inputs give byte-identical zips in one process.

    Class: REGRESSION guard on the combine's determinism contract (CR4). Wrong implementation:
    a wall-clock or temp-folder name written into an entry; red on the byte comparison.
    Reachability: the manifest and crate stand-ins are deterministic and the real read-model
    writers run. Both-states anchor: two outputs compared with each other. Second correct
    implementation: comparing entry digests rather than archive bytes.
    """
    clean, resume = _pair(tmp_path)
    C.combine_bundle([clean, resume], tmp_path / "a.zip", workdir=tmp_path)
    C.combine_bundle([clean, resume], tmp_path / "b.zip", workdir=tmp_path)
    assert hashlib_sha(tmp_path / "a.zip") == hashlib_sha(tmp_path / "b.zip")


def _combined_zip(tmp_path: Path, *, old_member: bool = False) -> tuple[Path, Path, Path]:
    """Combine a folder input and a zip input into ``tmp_path/out.zip``; return (zip, clean, resume)."""
    clean, resume = _pair(tmp_path)
    fig = clean / "plots" / "appendix" / "scenario_status.html"
    fig.parent.mkdir(parents=True)
    fig.write_text("refreshed figure")
    if old_member:
        member = tmp_path / "in" / "resume.zip"
        with zipfile.ZipFile(member, "w", compression=zipfile.ZIP_STORED) as zf:
            for p in sorted(resume.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(resume).as_posix())
    else:
        member = _zip_dir(resume, tmp_path / "in" / "resume.zip")
    C.combine_bundle([clean, member], tmp_path / "out.zip", workdir=tmp_path)
    return tmp_path / "out.zip", clean, resume


@pytest.mark.parametrize("old_member", [False, True])
def test_from_zip_unpacks_root_files_and_each_members_render_set_with_refreshed_figures(tmp_path, seams, old_member):
    """Property: ``from_zip`` extracts root files and each member's render set, refreshed figures over it.

    Class: NEW CAPABILITY (discriminator: ``from_zip`` did not exist). Wrong implementation: extracting
    each nested member whole (the raw ``.bin`` files appear), skipping the refreshed overlay, or reading a
    deflated nested entry without inflating it; red on the file-set literals. Reachability: the folder
    input becomes a nested zip of its FULL bundle (raw files inside) and, with ``old_member``, the zip input
    is an old stored-entry zip written as one compressed entry (D17). Both-states anchor: the unpacked
    file set is compared with literals and the refreshed figure's bytes with the bytes written at the
    re-render path. Second correct implementation: unpacking each nested zip fully, then deleting
    ``out_*/bin``.
    """
    zip_path, clean, _resume = _combined_zip(tmp_path, old_member=old_member)
    # The refreshed entry must differ from the nested member's own copy for the overlay to be visible.
    refreshed_name = "child_crates/exp_clean_tritonswmm.refreshed/plots/appendix/scenario_status.html"
    rewritten = tmp_path / "rewritten.zip"
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            dst.writestr(info, b"re-rendered figure" if info.filename == refreshed_name else src.read(info.filename))
    rewritten.replace(zip_path)
    cb = C.CombinedBundle.from_zip(zip_path)
    unpack = tmp_path / "out"
    assert cb.root == unpack.resolve() and cb.is_archive is False
    files = {p.relative_to(unpack).as_posix() for p in unpack.rglob("*") if p.is_file()}
    assert "bundle_manifest.json" in files and "combined_intercomparison.json" in files
    for eid in ("exp_clean_tritonswmm", "exp_resume_tritonswmm"):
        assert f"child_crates/{eid}/sims/0/out_triton/run.cfg" in files
        assert f"child_crates/{eid}/analysis_datatree.zarr/zarr.json" in files
    assert not any("/bin/" in f for f in files if f.startswith("child_crates/"))
    assert not any(".member_tmp" in f for f in files)
    assert not any(f.endswith(".zip") and f.startswith("child_crates/") for f in files)
    assert (unpack / "child_crates/exp_clean_tritonswmm/plots/appendix/scenario_status.html").read_text() == (
        "re-rendered figure"
    )


def test_regenerate_report_on_an_archive_handle_and_on_from_zip_render_in_the_unpacked_folder(
    tmp_path, seams, monkeypatch
):
    """Property: ``regenerate_report`` works on the handle ``combine_bundle`` returns and on ``from_zip``'s.

    Class: NEW CAPABILITY (discriminator: before this chunk the archive handle raised
    ``NotImplementedError``). Wrong implementation: still raising, or rendering against the zip path;
    red on the exception or on the recorded root. Reachability: the render seam records the folder it is
    handed and writes the report file the method requires. Both-states anchor: the recorded root is the
    unpacked folder beside the zip. Second correct implementation: ``regenerate_report`` on the archive
    handle unpacking through the same helper ``from_zip`` uses.
    """
    import hhemt.provenance as prov

    zip_path, _c, _r = _combined_zip(tmp_path)
    handle = C.combine_bundle(
        [tmp_path / "in" / "clean", tmp_path / "in" / "resume.zip"], tmp_path / "again.zip", workdir=tmp_path
    )
    assert handle.is_archive is True
    seen: list[Path] = []

    def render(merged, report, root, **kw):
        seen.append(Path(root))
        (Path(root) / f"analysis_report.{kw['formats'][0]}").write_text("r")

    monkeypatch.setattr(C, "_render_combined_report", render)
    monkeypatch.setattr(prov, "assert_plots_match_running_build", lambda *a, **k: None)
    out = handle.regenerate_report(format="zip")
    assert seen == [(tmp_path / "again").resolve()] and out == (tmp_path / "again" / "analysis_report.zip").resolve()
    out2 = C.CombinedBundle.from_zip(zip_path).regenerate_report(format="html")
    assert seen[-1] == (tmp_path / "out").resolve() and out2.name == "analysis_report.html"


def test_report_from_bundle_routes_a_combined_zip_to_combinedbundle(tmp_path, seams, monkeypatch):
    """Property: ``hhemt report-from-bundle {combined}.zip`` renders through ``CombinedBundle`` and never ``Bundle``.

    Class: NEW CAPABILITY (discriminator: today a combined manifest fails ``Bundle.from_directory``).
    Wrong implementation: unpacking every zip with ``Bundle``; red on the ``Bundle`` spy or the exit
    code. Reachability: a real combined zip from a combine over stubs. Both-states anchor: the spies and
    the exit code. Second correct implementation: routing on a CLI flag instead of the manifest.
    """
    from typer.testing import CliRunner

    import hhemt.bundle as bundle_pkg
    from hhemt.cli import app

    zip_path, _c, _r = _combined_zip(tmp_path)
    called: list[str] = []
    monkeypatch.setattr(
        C.CombinedBundle, "regenerate_report", lambda self, **kw: called.append("combined") or self.root / "r.zip"
    )
    monkeypatch.setattr(bundle_pkg.Bundle, "from_directory", classmethod(lambda cls, p: called.append("bundle")))
    result = CliRunner().invoke(app, ["report-from-bundle", str(zip_path)])
    assert result.exit_code == 0, result.output
    assert called == ["combined"]

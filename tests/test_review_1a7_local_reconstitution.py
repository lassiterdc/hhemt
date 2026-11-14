"""Review of the local reconstitution: the shared post-extraction path, the destination, the reference comparison.

The chunk's own tests record the config paths the shared path hands the constructor and build the reference stores by
hand. These tests (a) record the ORDER of every step ``from_doi`` takes and what each step is handed, which holds on the
base commit and on the chunk's tree alike, (b) run the real emit's reference-output carriage into the real comparator,
so the path convention of the two sides is checked against each other and not against a docstring, (c) quantify the
comparator over dtype classes, and (d) put hostile destinations to ``extract_reprex_bundle`` inside ``tmp_path``.

No solver is compiled or run; every write is under ``tmp_path``.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation, What kills it, and a second
correct implementation under which it still passes.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle._reprex import extract_reprex_bundle
from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment
from tests.test_publish_deposit_paths import _world


def _extract_to(zip_path: Path, dest: Path) -> Path:
    """Extract to ``dest``; a tree whose extractor has no ``dest`` argument fails the test, not the collection."""
    try:
        return extract_reprex_bundle(zip_path, dest=dest)
    except TypeError as exc:
        raise AssertionError(f"extract_reprex_bundle takes no dest argument: {exc}") from exc


def compare_reference_outputs(*args, **kwargs):
    """Call the chunk's comparator, imported when called so a tree without it fails the test and not the collection."""
    try:
        from hhemt.bundle._reprex import compare_reference_outputs as real
    except ImportError as exc:
        raise AssertionError(f"compare_reference_outputs does not exist in hhemt.bundle._reprex: {exc}") from exc
    return real(*args, **kwargs)


def _emitted_zip(tmp_path: Path, *, test_store: bool = False, out_type: str = "zarr") -> tuple[Path, Path]:
    """A real emitted bundle over the bare analysis of ``_world``, with a Workflow-Run-Crate; returns (zip, analysis
    dir)."""
    from hhemt.bundle import _emit

    kwargs = {} if out_type == "zarr" else {"analysis_over": {"target_processed_output_type": out_type}}
    world = _world(tmp_path, **kwargs)
    (world.ad / "Snakefile").write_text("rule all:\n    input: []\n")
    crate = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"},
            {"@id": "./", "@type": "Dataset", "name": "T"},
        ],
    }
    (world.ad / "ro-crate-metadata.json").write_text(json.dumps(crate))
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    if test_store:
        _write_summary(world.ad / "_test" / "group_0" / "sims" / "event_index.0" / "processed", out_type, 1.5)
    return _emit.emit_bundle(world.analysis, tmp_path / "out" / "bundle.zip"), world.ad


def _write_summary(processed: Path, out_type: str, value: float, name: str = "TRITONSWMM_TRITON_summary") -> Path:
    processed.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({"max_flood_depth": (("y", "x"), np.array([[value, np.nan], [2.0, 3.0]]))})
    path = processed / f"{name}.{out_type}"
    if out_type == "zarr":
        ds.to_zarr(path, mode="w", consolidated=False, zarr_format=3)
    else:
        ds.to_netcdf(path, engine="h5netcdf")
    return path


# ---- (1) the shared post-extraction path ------------------------------------------------------------------------


def test_from_doi_takes_its_steps_in_the_base_order_with_the_base_arguments(tmp_path, monkeypatch):
    """Property: ``from_doi`` fetches, extracts, validates the bundle, gates on the workflow, reconstitutes the system
    config then the analysis config, checks the declared inputs, resolves the HPC config, and only then constructs the
    experiment; the system config is reconstituted with ``{bundle_root}/software`` when no ``software_dir`` is given and
    with the caller's when one is; the experiment's ``bundle_root`` is the extraction root.

    Class: REGRESSION - green on the base commit and on the chunk's tree; the plausible wrong refactor reorders two
        steps (the input check before the workflow gate), drops the default software directory, or constructs the
        experiment before the checks. It is red under that.
    Reachability: ``from_doi`` is the DOI entry point and its three round-trip tests cannot run on a laptop (compile
        guard), so no other local test orders these steps; artifact: the recorder below.
    Both-states anchor: the assertion turns on the recorded sequence, a property of the code at both commits.
    Invocation: not isolatable from this tree - discrimination shown instead: swapping the two reconstitute calls in
        ``_reconstitute_from_bundle_root`` reorders the log and reds this (measured by the reviewer).
    What kills it: any reorder, dropped or added step, or a changed argument.
    Second correct implementation: a pipeline that runs the same steps from a list of callables in the same order.
    """
    import hhemt.bundle as bundle_pkg
    import hhemt.bundle._emit as emit_mod
    import hhemt.bundle._reprex as reprex_mod

    zip_path, _ad = _emitted_zip(tmp_path)
    log: list[tuple] = []

    def wrap(owner, name, label, describe=lambda *a, **k: ()):
        real = getattr(owner, name)

        def inner(*args, **kwargs):
            log.append((label, *describe(*args, **kwargs)))
            return real(*args, **kwargs)

        monkeypatch.setattr(owner, name, inner)

    wrap(reprex_mod, "extract_reprex_bundle", "extract")
    wrap(bundle_pkg.Bundle, "from_directory", "schema_guard")
    wrap(emit_mod, "reconstitute_runnable_config", "config_system", lambda root, software_dir=None: (software_dir,))
    wrap(emit_mod, "reconstitute_runnable_analysis_config", "config_analysis")
    for name in ("_assert_bundle_has_workflow", "_assert_declared_inputs_exist", "_resolve_hpc_system_config"):
        raw = TRITON_SWMM_experiment.__dict__[name]
        real = raw.__func__

        kind = type(raw)

        def make(real=real, name=name, kind=kind):
            def inner(*args, **kwargs):
                log.append((name,))
                return real(*args, **kwargs)

            return kind(inner)

        monkeypatch.setattr(TRITON_SWMM_experiment, name, make())

    def _init(self, cfg_system_yaml, cfg_analysis_yaml, case_name=None, hpc_system_config_yaml=None):
        log.append(("construct", case_name, hpc_system_config_yaml))

    monkeypatch.setattr(TRITON_SWMM_experiment, "__init__", _init)
    monkeypatch.setattr(
        TRITON_SWMM_experiment,
        "_fetch_bundle_zip",
        classmethod(lambda cls, *a, **k: (log.append(("fetch",)), zip_path)[1]),
    )

    exp = TRITON_SWMM_experiment.from_doi(
        doi="10.5281/zenodo.1", host="zenodo", target_dir=tmp_path / "t", validate=False
    )
    root = exp.bundle_root
    assert root == tmp_path / "out" / "bundle"
    default_software = root / "software"
    assert log == [
        ("fetch",),
        ("extract",),
        ("schema_guard",),
        ("_assert_bundle_has_workflow",),
        ("config_system", default_software),
        ("config_analysis",),
        ("_assert_declared_inputs_exist",),
        ("_resolve_hpc_system_config",),
        ("construct", None, None),
    ]
    log.clear()
    explicit = tmp_path / "my_software"
    TRITON_SWMM_experiment.from_doi(
        doi="10.5281/zenodo.1", host="zenodo", target_dir=tmp_path / "t2", software_dir=explicit, validate=False
    )
    assert ("config_system", explicit) in log


# ---- (1b) the real emit's reference carriage into the real comparator -------------------------------------------


@pytest.mark.parametrize("out_type", ["zarr", "nc"])
def test_the_reference_stores_the_emit_carries_are_the_stores_the_comparator_reads(tmp_path, out_type):
    """Property: the producer's ``_test/group_i/sims/{event}/processed/*_summary.{out_type}`` stores, carried by the
        real
    emit, extracted, and compared with the same ``_test`` tree, pass; the same comparison against a ``_test`` tree whose
    value differs fails and names the store and the variable.

    Class: NEW CAPABILITY - the plausible wrong implementation reads a path convention the emit does not write (a
        different group directory, or a suffix filter narrower than the emit's ``*_summary.{out_type}``); it is red
        under it because the comparator then finds no reference store and raises. The chunk's own tests write the
        reference tree by hand, so neither side of the seam was exercised against the other.
    Reachability: ``_copy_reference_outputs`` is part of every emit whose analysis has a ``_test/`` directory, and a
        reproducer's ``analysis.test()`` writes ``{analysis_dir}/_test/group_{i}`` as each sub-analysis's directory
        (``analysis.py``, the ``group_id`` loop); artifact: this test runs the first and the source shows the second.
    Both-states anchor: the assertion turns on pass versus fail of one comparison over a tree, not on a message.
    Invocation: not isolatable from this tree - discrimination shown instead: a comparator glob of ``group_*/sims/*/
        processed/*_summary.zarr`` greens the zarr arm and reds the nc arm (measured by the reviewer).
    What kills it: a path-convention mismatch between emit and comparator, in either out_type.
    Second correct implementation: walk ``reference_outputs`` with ``os.walk`` for directories or files named
        ``*_summary.*`` and map each to the same relative path under ``_test``; same verdicts.
    """
    zip_path, ad = _emitted_zip(tmp_path, test_store=True, out_type=out_type)
    bundle_root = _extract_to(zip_path, tmp_path / "unpacked")
    rel = f"group_0/sims/event_index.0/processed/TRITONSWMM_TRITON_summary.{out_type}"
    assert (bundle_root / "reference_outputs" / rel).exists(), "control: the emit carried the reference store"
    assert compare_reference_outputs(bundle_root, ad).passed
    other = tmp_path / "rerun"
    _write_summary(other / "_test" / "group_0" / "sims" / "event_index.0" / "processed", out_type, 9.5)
    result = compare_reference_outputs(bundle_root, other)
    assert not result.passed
    assert rel in result.summary() and "max_flood_depth" in result.summary()


def test_a_reproducer_that_never_ran_test_fails_the_comparison_and_does_not_pass_it(tmp_path):
    """Property: with a bundle that carries references and an analysis directory with no ``_test/`` at all, the
    comparison returns a failing verdict naming each reference store as not produced.

    Class: NEW CAPABILITY - the plausible wrong implementation treats an absent re-run tree as nothing to compare and
        passes. It is red under it.
    Reachability: a reproducer who runs the comparison before ``analysis.test()``; artifact: this test.
    Both-states anchor: turns on ``passed`` being False and the store being named, not on wording.
    Invocation: not isolatable from this tree - discrimination shown instead: a comparator that returns no verdict for
        a missing ``_test`` reds this (the bundle then reports a pass on an empty list only if ``passed`` ignores it).
    What kills it: an early return on a missing ``_test``.
    Second correct implementation: raise ``ProcessingError`` naming the missing re-run tree; this test then needs
        ``pytest.raises``, so it accepts either a failing verdict or that error.
    """
    zip_path, _ad = _emitted_zip(tmp_path, test_store=True)
    bundle_root = _extract_to(zip_path, tmp_path / "unpacked")
    empty = tmp_path / "fresh_analysis"
    empty.mkdir()
    try:
        result = compare_reference_outputs(bundle_root, empty)
    except ProcessingError:
        return
    assert not result.passed
    assert "TRITONSWMM_TRITON_summary.zarr" in result.summary()


# ---- (3) the comparator over dtype classes ----------------------------------------------------------------------

_KINDS = {
    "float32": (np.array([0.5, np.nan, 2.0], dtype="float32"), 3),
    "float64": (np.array([0.1, np.nan, 2.0], dtype="float64"), 3),
    "int64": (np.array([1, 2, 3], dtype="int64"), 3),
    "uint8": (np.array([1, 2, 3], dtype="uint8"), 3),
    "bool": (np.array([True, False, True]), 3),
    "text": (np.array(["a", "b", "c"]), 3),
    "datetime": (np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[ns]"), 3),
    "timedelta": (np.array([1, 2, 3], dtype="timedelta64[s]"), 3),
}


def _store_with(tmp: Path, values: np.ndarray, out_type: str) -> Path:
    processed = tmp / "_test" / "group_0" / "sims" / "event_index.0" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({"v": ("t", values)})
    path = processed / f"S_summary.{out_type}"
    if out_type == "zarr":
        ds.to_zarr(path, mode="w", consolidated=False, zarr_format=3)
    else:
        ds.to_netcdf(path, engine="h5netcdf")
    return path


def _perturbed(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    if out.dtype.kind == "b":
        out[0] = not out[0]
    elif out.dtype.kind in "US":
        out[0] = "z"
    elif out.dtype.kind in "mM":
        out[0] = out[2]
    elif out.dtype.kind == "f":
        out[0] = np.nextafter(out[0], np.float32(10) if out.dtype == np.float32 else 10.0)
    else:
        out[0] = out[0] + 1
    return out


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_every_dtype_class_passes_when_identical_and_fails_when_one_element_differs(tmp_path, kind):
    """Property: for each of eight dtype classes (two float widths, signed and unsigned integers, bool, text, datetime
    with a NaT, timedelta), a store compared with an identical copy passes, and the same store compared with a copy
    whose first element differs by the smallest representable step fails and names the variable.

    Class: NEW CAPABILITY - the plausible wrong implementation branches on dtype kind and compares one kind loosely
        or not at all (for example datetimes or booleans through the float path); it is red under it for that kind.
    Reachability: a summary store holds float peaks, integer counts, text ids and datetime coordinates; artifact: the
        golden multi-sim summaries carry coordinates of these classes.
    Both-states anchor: turns on pass versus fail across a one-element change, whatever the dtype.
    Invocation: not isolatable from this tree - discrimination shown instead: removing the datetime branch, so a
        NaT compares unequal to itself, reds the datetime arm (measured by the reviewer).
    What kills it: a dtype branch that equates unequal values, or fails equal ones (the identical arm).
    Second correct implementation: compare with ``xarray.testing.assert_identical`` per variable and translate its
        failure into a verdict; same pass and fail outcomes for these inputs.
    """
    values, _ = _KINDS[kind]
    bundle = tmp_path / "bundle"
    staged = _store_with(tmp_path / "staged", values, "zarr")
    ref_root = bundle / "reference_outputs" / "group_0" / "sims" / "event_index.0" / "processed"
    ref_root.mkdir(parents=True)
    staged.rename(ref_root / "S_summary.zarr")
    same_dir, diff_dir = tmp_path / "same", tmp_path / "diff"
    _store_with(same_dir, values, "zarr")
    _store_with(diff_dir, _perturbed(values), "zarr")
    assert compare_reference_outputs(bundle, same_dir).passed, "identical store must pass"
    result = compare_reference_outputs(bundle, diff_dir)
    assert not result.passed, "control: a one-element change must be seen"
    assert "'v'" in result.summary()


# ---- (2) hostile destinations to the extraction -----------------------------------------------------------------


def _tiny_zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("bundle_manifest.json", "{}")
    return path


@pytest.mark.parametrize("levels_up", [0, 1])
def test_extraction_never_deletes_a_directory_that_holds_the_zip_it_is_extracting(tmp_path, levels_up):
    """Property: when the destination named for an extraction is the directory that contains the zip, or an ancestor of
    that directory, the zip file still exists after the call, whether the call extracts, refuses or raises.

    Class: REGRESSION - green at the base commit (the base had no destination argument; its sibling destination
        ``zip.with_suffix("")`` can never contain the zip) and red on the chunk's tree: ``extract_reprex_bundle``
            removes
        an existing ``dest`` with ``shutil.rmtree`` before it opens the zip, so a destination that holds the zip
        deletes the zip and every other file beside it.
    Reachability: ``from_bundle_zip(zip, target_dir=T)`` sets ``dest = T / zip stem``; with the zip at
        ``D/{stem}/{stem}.zip`` (the layout ``emit_bundle`` gives ``bundles/{name}.zip`` inside a directory of that
            name,
        or any user who keeps ``x/x.zip``) and ``T = D``, ``dest`` is the zip's own directory; with the zip two levels
        down, ``dest`` is an ancestor; artifact: this test, inside ``tmp_path``.
    Both-states anchor: turns on the zip still existing afterwards, not on whether the call raises.
    Invocation: not isolatable from this tree - discrimination shown instead: refusing, before any deletion, a ``dest``
        that equals or contains ``zip_path.parent`` greens this test (measured by the reviewer).
    What kills it: an unconditional ``rmtree`` of a caller-named directory.
    Second correct implementation: extract into a fresh temporary sibling and rename it over ``dest`` only after the
        extraction succeeded and ``dest`` is verified not to contain the zip; the zip survives in every case.
    """
    holder = tmp_path / "stem"
    zip_path = _tiny_zip(holder / "stem.zip" if levels_up == 0 else holder / "inner" / "stem.zip")
    # levels_up 0: the destination is the zip's own directory; 1: an ancestor of it
    try:
        extract_reprex_bundle(zip_path, dest=holder)
    except Exception:  # noqa: BLE001 - only the zip's survival is asserted
        pass
    assert zip_path.exists(), "the zip being extracted was deleted by the extraction"


def test_from_bundle_zip_with_a_target_dir_named_for_the_zips_grandparent_keeps_the_zip_and_its_neighbours(tmp_path):
    """Property: ``from_bundle_zip(zip, target_dir=T)`` where ``T / zip stem`` is the directory holding the zip leaves
    the zip and every file beside it in place, whether it extracts elsewhere, refuses, or raises.

    Class: REGRESSION - the base commit has no ``from_bundle_zip``, so the test is green there (the attribute is
        missing and the call is not reached); on the chunk's tree the extraction removes ``T / stem`` before it opens
        the zip, which deleted the zip and a neighbouring user file (measured, ``FileNotFoundError`` raised after).
    Reachability: a user who keeps ``x/x.zip`` and passes ``target_dir`` as the directory above ``x``; artifact:
        this test.
    Both-states anchor: turns on the zip and the neighbour still existing, not on what the call raises.
    Invocation: not isolatable from this tree - discrimination shown instead: refusing a destination that contains the
        zip, before deletion, greens this test (measured by the reviewer).
    What kills it: an unconditional ``rmtree`` of ``target_dir / stem``.
    Second correct implementation: extract into a fresh temporary directory under ``target_dir`` and rename it over the
        destination only when the destination does not contain the zip.
    """
    holder = tmp_path / "stem"
    zip_path = _tiny_zip(holder / "stem.zip")
    neighbour = holder / "other.txt"
    neighbour.write_text("a file the user keeps beside the zip")
    try:
        TRITON_SWMM_experiment.from_bundle_zip(zip_path, target_dir=tmp_path, validate=False)
    except Exception:  # noqa: BLE001 - only what survives is asserted
        pass
    assert zip_path.exists() and neighbour.exists()

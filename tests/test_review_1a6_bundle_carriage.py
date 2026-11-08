"""Review of the bundle's time-series carriage: the real emit over a real analysis tree carries every store.

The chunk's own tests (``tests/test_bundle_timeseries_carriage.py``) run the real emit over a bare namespace whose
scenario enumeration is replaced by a stand-in, and populate one of the eight store fields. These tests run
``emit_bundle`` unchanged over the multi-sim golden analysis, a real ``TRITONSWMM_analysis`` whose three scenarios hold
all eight stores each, so the enumeration, the planner, the manifest and the archive are exercised together on a tree
    the
chunk's author did not build. No solver is compiled or run; every write is under ``tmp_path``.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What kills it, and a
second correct implementation under which it still passes.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import xarray as xr

from hhemt.bundle import _emit
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

#: The glob that names the per-scenario time-series stores by the FILE NAME the scenario constructor gives them, not by
#: the dataclass fields the emit reads: every one of the eight ends ``_tseries.zarr`` and no summary or input does.
STORE_GLOB = "*/processed/*_tseries.zarr"
EXPECTED_STORES_PER_SCENARIO = 8


def _prepare_figure(analysis) -> Path:
    """Write the one figure sidecar the emit requires, declaring no source, so only the carriage is under test."""
    ad = Path(analysis.analysis_paths.analysis_dir)
    plots = ad / "plots"
    plots.mkdir(exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    return ad


def _on_disk_stores(analysis) -> dict[str, Path]:
    """Every ``*_tseries.zarr`` directory under ``sims/`` plus the consolidated store, keyed by archive name."""
    ad = Path(analysis.analysis_paths.analysis_dir).resolve()
    sims = Path(analysis.analysis_paths.simulation_directory).resolve()
    found = {p.relative_to(ad).as_posix(): p for p in sims.glob(STORE_GLOB) if p.is_dir()}
    consolidated = ad / "analysis_timeseries.zarr"
    if consolidated.is_dir():
        found[consolidated.relative_to(ad).as_posix()] = consolidated
    return found


def _files_under(store: Path) -> dict[str, Path]:
    """UNIT: files (not directories) below ``store``, keyed by the path relative to it."""
    return {p.relative_to(store).as_posix(): p for p in sorted(store.rglob("*")) if p.is_file()}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(zf: zipfile.ZipFile) -> dict:
    return json.loads(zf.read(BUNDLE_MANIFEST_FILENAME))


def test_real_emit_over_the_golden_carries_every_store_file_for_file(golden_multisim_analysis, tmp_path):
    """Property: every time-series store on disk is named under ``carried_output_stores``, no other path is, and the
    archive holds exactly the files of each store, with every chunk file byte-identical to its source.

    Class: REGRESSION is not the claim here; NEW CAPABILITY - the plausible wrong implementation is a carriage that
        iterates a subset of the eight store fields (for example the first two) or plans a store's directory but not its
        files. It is red under it: the chunk's own arms populate one field, so a subset passes them.
    Reachability: the golden's three scenarios each hold all eight stores (``tritonswmm`` triton, performance, node,
        link; ``triton`` only and its performance; ``swmm`` only node and link), so a subset implementation drops
        stores that exist; artifact: the matcher ``*/processed/*_tseries.zarr`` over ``sims/`` returns 24 directories.
    Both-states anchor: the assertion compares the carried set with the set the FILE NAMES give, whatever the number of
        stores, and compares archive members with disk files, whatever their count.
    Invocation isolated: not isolatable from this tree - discrimination shown instead: restricting
        ``_scenario_time_series_fields`` to its first two names turns this red (measured by the reviewer, recorded in
        its file).
    What kills it: a subset of the fields; a planner that omits chunk files; a manifest naming a store the archive
        lacks.
    Second correct implementation: enumerate the stores by globbing ``sims/*/processed/*_tseries.*`` instead of
        reading ``ScenarioPaths`` fields; the carried set and the archive are the same, so this passes.
    """
    analysis = golden_multisim_analysis
    ad = _prepare_figure(analysis)
    stores = _on_disk_stores(analysis)
    per_scenario = [name for name in stores if name.startswith("sims/")]
    expected_count = 3 * EXPECTED_STORES_PER_SCENARIO
    assert len(per_scenario) == expected_count, f"control: the golden's store count: {len(per_scenario)}"
    out = _emit.emit_bundle(analysis, tmp_path / "bundle.zip")
    with zipfile.ZipFile(out) as zf:
        manifest = _manifest(zf)
        names = zf.namelist()
        assert sorted(manifest["carried_output_stores"]) == sorted(stores), "the carried set is the on-disk set"
        for rel, store in stores.items():
            sources = _files_under(store)
            held = sorted(n[len(rel) + 1 :] for n in names if n.startswith(rel + "/") and not n.endswith("/"))
            assert held == sorted(sources), f"{rel}: archive members differ from the files on disk"
            meta = _emit._zarr_metadata_filenames()
            chunks = [r for r in sources if r.rsplit("/", 1)[-1] not in meta]
            assert chunks, f"control: {rel} holds chunk files, so the byte comparison below compares some"
            for r in chunks:
                assert _sha(zf.read(f"{rel}/{r}")) == _sha(sources[r].read_bytes()), f"{rel}/{r}"
    assert ad.exists()


def test_the_extracted_golden_stores_open_and_equal_their_sources(golden_multisim_analysis, tmp_path):
    """Property: every carried store, extracted from the archive, opens as a zarr store and equals its source.

    Class: NEW CAPABILITY - the plausible wrong implementation ships metadata the placeholder rewrite damaged, or
        ships a store whose metadata and chunks come from different scenarios; it is red under it because the store
        then fails to open or holds other values.
    Reachability: the emit rewrites every text member, and a zarr store's ``zarr.json`` or ``.zattrs`` is one; the
        golden stores carry real attributes, so the rewrite runs over real metadata; artifact: the store count below
        is 24.
    Both-states anchor: the assertion turns on value equality of the opened datasets, whatever the metadata bytes are.
    Invocation isolated: not isolatable from this tree - discrimination shown instead: a carriage that names every
        store but plans no files for the ``*_link_tseries.zarr`` stores turns this red (measured by the reviewer,
        recorded in its file), and the chunk's own tests stay green under it.
    What kills it: archive names that cross scenarios; metadata that no longer parses.
    Second correct implementation: carry each store as a single tar member and unpack it on read; the opened values are
        the same, so this passes.
    """
    analysis = golden_multisim_analysis
    _prepare_figure(analysis)
    stores = _on_disk_stores(analysis)
    out = _emit.emit_bundle(analysis, tmp_path / "bundle.zip")
    root = tmp_path / "extracted"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(root)
    assert len(stores) >= 3 * EXPECTED_STORES_PER_SCENARIO
    kwargs = {"consolidated": False, "decode_times": False, "decode_timedelta": False}
    for rel, store in stores.items():
        got = xr.open_zarr(root / rel, **kwargs).load()
        want = xr.open_zarr(store, **kwargs).load()
        xr.testing.assert_equal(got, want)


@pytest.mark.parametrize("flag", [False, True])
def test_the_manifest_and_the_archive_agree_for_every_carried_store_under_both_flag_states(
    golden_multisim_analysis, tmp_path, flag
):
    """Property: for every name in ``carried_output_stores`` the archive holds at least one member under it; under
    ``omit_output_zarrs`` every member under it is a zarr metadata document and the name is also in
    ``omitted_output_zarrs`` and ``metadata_only_stores``; without the flag it is in neither.

    Class: REGRESSION - the manifest is the reader's account of the archive, and a name the archive does not back is a
        false claim. The plausible wrong implementation lists a store it planned shallowly with no metadata, or lists
        a store without planning it.
    Reachability: both flag states are parameters of ``emit_bundle``; artifact: the golden emits under each (this test).
    Both-states anchor: the assertion turns on the relation between two artifacts of one emit, whichever the flag.
    Invocation isolated: not isolatable from this tree - discrimination shown instead: dropping the
        ``metadata_only_stores`` union under the flag, and separately planning the stores in full under the flag,
        each turn the flag arm red (both measured by the reviewer, recorded in its file).
    What kills it: a manifest/archive mismatch in either state; chunk files shipping under the flag.
    Second correct implementation: list the stores by reading the archive back after writing it; the relation holds by
        construction, so this passes.
    """
    analysis = golden_multisim_analysis
    _prepare_figure(analysis)
    out = _emit.emit_bundle(analysis, tmp_path / "bundle.zip", omit_output_zarrs=flag)
    meta = _emit._zarr_metadata_filenames()
    with zipfile.ZipFile(out) as zf:
        manifest = _manifest(zf)
        names = [n for n in zf.namelist() if not n.endswith("/")]
    carried = manifest["carried_output_stores"]
    assert len(carried) >= 3 * EXPECTED_STORES_PER_SCENARIO, "control: the golden has stores to carry"
    omitted = manifest.get("omitted_output_zarrs", [])
    shallow = manifest.get("metadata_only_stores", [])
    chunk_stores = 0
    for store in carried:
        below = [n for n in names if n.startswith(store + "/")]
        assert below, f"{store} is named but the archive holds nothing under it"
        chunks = [n for n in below if n.rsplit("/", 1)[-1] not in meta]
        chunk_stores += bool(chunks)
        if flag:
            assert chunks == [], f"{store}: chunk files ship under omit_output_zarrs"
            assert store in omitted and store in shallow
        else:
            assert store not in omitted and store not in shallow
    if not flag:
        assert chunk_stores > 0, "control: without the flag the stores ship chunk files"


@pytest.mark.skipif(os.geteuid() == 0, reason="a root process ignores directory permissions")
def test_emit_succeeds_on_a_read_only_simulations_tree_as_it_did_before(golden_multisim_analysis, tmp_path):
    """Property: an analysis tree whose simulation directories cannot be written to still emits, and carries the same
    stores it carries when the tree is writable.

    Class: REGRESSION - red on the tree under review, green at its base commit (both outputs recorded in the
        reviewer's file). The shape says an analysis without such stores "emits exactly as before"; the carriage builds
        a ``TRITONSWMM_scenario`` per event, whose constructor creates the scenario's empty working directories when
        they are missing, so an emit that needed no write before now needs one.
    Reachability: the golden's own scenarios lack their ``logs`` directories (the constructor adds them, which the
        chunk's own enumeration test asserts as ``new_dirs``), so an analysis restored from an archive that drops empty
            directories
        and then mounted read-only reaches the failure; artifact: the PermissionError on
        ``sims/event_index.0/logs`` recorded in the reviewer's file.
    Both-states anchor: the assertion turns on the locked emit returning an archive with the same members and the same
        carried set as the writable emit of the same tree, a relation that holds at the base commit (both carry no key)
        and under a correct fix; it does not turn on any message text or on the key existing.
    Invocation isolated: not isolatable from this tree - discrimination shown instead: reading the stores' locations
        without constructing a scenario (a read-only derivation of ``processed/{name}``) greens it.
    What kills it: constructing a scenario object, or any other write, inside the emit.
    Second correct implementation: enumerate by globbing ``sims/*/processed/*_tseries.zarr``; no write happens, the
        carried set is the same, and this passes.
    """
    analysis = golden_multisim_analysis
    _prepare_figure(analysis)
    sims = Path(analysis.analysis_paths.simulation_directory)
    locked = [p for p in sims.rglob("*") if p.is_dir()] + [sims]
    # The locked emit runs FIRST: a writable emit before it would create the missing working directories and leave
    # nothing for the locked one to fail on.
    for directory in locked:
        directory.chmod(0o555)
    try:
        locked_out = _emit.emit_bundle(analysis, tmp_path / "locked.zip")
    finally:
        for directory in locked:
            directory.chmod(0o755)
    writable_out = _emit.emit_bundle(analysis, tmp_path / "writable.zip")
    with zipfile.ZipFile(locked_out) as zf_locked, zipfile.ZipFile(writable_out) as zf_writable:
        assert _manifest(zf_locked).get("carried_output_stores") == _manifest(zf_writable).get("carried_output_stores")
        assert zf_locked.namelist() == zf_writable.namelist()


def test_a_symlinked_simulations_directory_carries_its_stores_at_their_analysis_relative_names(
    golden_multisim_analysis, tmp_path
):
    """Property: when ``sims`` is a symlink to a directory outside the analysis tree, the emit still succeeds and
        carries
    each scenario's stores under ``sims/{event}/processed/...``, the position they have under the analysis directory
    as the analysis names it, and no member lands under ``external/``.

    Class: NEW CAPABILITY - the plausible wrong implementation resolves each store path and takes its position relative
        to the resolved analysis directory, which sends every store of a symlinked ``sims`` to ``external/{file name}``,
        where the same file name of two scenarios is two different files under one archive name. It is red under that
        implementation: ``emit_bundle`` raises ``ConfigurationError`` (field ``source_paths_by_renderer``), where the
        base commit emitted this tree without error.
    Reachability: ``sims`` is the analysis's own simulation directory, and a relocation of it to another volume leaves a
        symlink at that name; the placeholder table already writes a path "as given and, when different, as resolved:
        a symlinked scratch directory"; artifact: the reviewer's file records the ``ConfigurationError`` on this tree
        and the clean emit of the same tree at the base commit.
    Both-states anchor: the assertion turns on where chunk files of the stores sit in the archive and on the absence of
        any ``external/`` member, never on the exception text.
    Invocation isolated: not isolatable from this tree - discrimination shown instead: naming each carried store by
        its unresolved position under the analysis directory (``src.relative_to(analysis_dir)``, the position the
        shape calls analysis-relative) greens this test; recorded in the reviewer's file.
    What kills it: resolving before relativizing in ``_plan_time_series_carriage``.
    Second correct implementation: enumerate each scenario folder with ``os.scandir`` on the unresolved ``sims`` path
        and relativize with ``os.path.relpath``; the archive names are the same, and this passes.
    """
    analysis = golden_multisim_analysis
    ad = _prepare_figure(analysis)
    sims = ad / "sims"
    away = tmp_path / "elsewhere" / "sims"
    away.parent.mkdir()
    sims.rename(away)
    sims.symlink_to(away, target_is_directory=True)
    wanted = sorted(p.relative_to(away).as_posix() for p in away.glob(STORE_GLOB))
    assert len(wanted) == 3 * EXPECTED_STORES_PER_SCENARIO, "control: the relocated tree holds the stores"
    out = _emit.emit_bundle(analysis, tmp_path / "bundle.zip")
    meta = _emit._zarr_metadata_filenames()
    with zipfile.ZipFile(out) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    assert [n for n in names if n.startswith("external/")] == []
    for rel in wanted:
        chunks = [n for n in names if n.startswith(f"sims/{rel}/") and n.rsplit("/", 1)[-1] not in meta]
        assert chunks, f"no chunk file of sims/{rel} in the archive"

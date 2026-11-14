"""Reviewer arms for the out-of-tree absent-source classification of the bundle harvest.

Problem this file solves: a sidecar records each source outside the analysis directory as a
``../``-relative path. Harvested from a tree sitting at another depth, such a path lands nowhere;
the harvest used to report it as a missing ``external/{name}`` file although the declared-input
copy carries a file under that archive name. ``_harvest_sources(..., unresolved_outside=[])`` now
records it under its own list, with its own aggregated warning and its own manifest key.

The coder's own tests drive ``_harvest_sources`` and ``_write_bundle_manifest`` separately, so the
lines inside ``emit_bundle`` that create the list, hand it to the harvest and hand it on to the
manifest writer are covered by nothing. These arms drive the REAL ``emit_bundle`` (through the
``_world`` builder of ``tests/test_publish_deposit_paths.py``) and attack the classification
boundary and the warning's form with inputs the coder's single tree does not contain.

Solver-free: every arm writes ``tmp_path`` files only, no compile, no simulation, and no fixture
from the synthetic-model tier is requested.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

from hhemt.bundle import _emit
from hhemt.bundle._emit import _HARVEST_WARNING_LABEL, _harvest_sources
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME
from tests.test_publish_deposit_paths import _world

KEY = "declared_sources_unresolved_outside_tree"
UNRESOLVED = "../moved_away/matrix.csv"


def _real_emit(tmp_path: Path, declared: list[str]):
    """Run the real ``emit_bundle`` over a world whose one sidecar declares ``declared``.

    The world holds one real out-of-tree file, ``../outside/events.nc``. Returns
    ``(archive member names, parsed bundle_manifest.json, harvest warnings issued by the emit)``.
    """
    world = _world(tmp_path)
    (world.out / "events.nc").write_text("x", encoding="utf-8")
    plots = world.ad / "plots"
    plots.mkdir()
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": declared}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _emit.emit_bundle(world.analysis, world.base / "bundle.zip")
    harvest = [w for w in caught if str(w.message).startswith("Bundle harvest:")]
    with zipfile.ZipFile(out) as zf:
        return zf.namelist(), json.loads(zf.read(BUNDLE_MANIFEST_FILENAME)), harvest


def test_a_real_emit_hands_the_unresolved_list_to_the_harvest_and_on_to_the_manifest(tmp_path):
    """Property: through ``emit_bundle`` an absent out-of-tree source reaches the manifest under its own key only.

    Class: every sidecar declaration of a real ``emit_bundle`` (in-tree absent, out-of-tree absent,
    out-of-tree present, here), plus two controls that differ in WHICH kinds are present.
    UNIT: manifest keys and archive member names; warnings are counted per aggregated message.
    Why required: the list is created, passed to the harvest and passed to the manifest writer by
    three separate lines inside ``emit_bundle``; a test that calls the harvest and the writer
    directly reads none of them, so a dropped line ships a manifest that still calls the relocated
    source a missing ``external/`` file, with every other test green.
    What kills it: omitting ``unresolved_outside=`` from the ``_harvest_sources`` call in
    ``emit_bundle`` (the source falls back into ``declared_sources_absent`` as ``external/matrix.csv``),
    or omitting ``declared_sources_unresolved_outside_tree=`` from the ``_write_bundle_manifest``
    call (the key never appears), or having ``emit_bundle`` issue a third warning of its own.
    Second correct implementation: having ``_harvest_sources`` return the list as a fourth tuple
    element and ``emit_bundle`` unpack it passes unchanged, since the test reads only the archive,
    the manifest and the warnings.
    """
    names, manifest, harvest = _real_emit(tmp_path / "mixed", ["_status/_du.json", UNRESOLVED, "../outside/events.nc"])
    assert manifest[KEY] == [UNRESOLVED]
    assert manifest["declared_sources_absent"] == ["_status/_du.json"], (
        "the in-tree absent source stays under its own key"
    )
    assert "external/events.nc" in names, "the present out-of-tree source is still carried as external/{name}"
    assert "external/matrix.csv" not in names
    assert len(harvest) == 2, [str(w.message)[:90] for w in harvest]
    assert "declared_sources_absent" in str(harvest[0].message) and KEY not in str(harvest[0].message)
    assert KEY in str(harvest[1].message) and "_du.json" not in str(harvest[1].message)

    _names, only_out, harvest = _real_emit(tmp_path / "only_out", [UNRESOLVED])
    assert only_out[KEY] == [UNRESOLVED]
    assert "declared_sources_absent" not in only_out, "absent, not empty, when no in-tree source is missing"
    assert len(harvest) == 1 and KEY in str(harvest[0].message)

    _names, clean, harvest = _real_emit(tmp_path / "clean", ["../outside/events.nc"])
    assert KEY not in clean and "declared_sources_absent" not in clean, "a clean bundle's manifest carries neither key"
    assert harvest == []


def test_classification_follows_the_resolved_root_not_a_spelling_of_it(tmp_path):
    """Property: in-tree versus out-of-tree is decided on RESOLVED paths and by path components, not by string prefix.

    Class: spellings of the analysis root and of its neighbours that a string comparison confuses:
    the root reached through a symlink alias, a sibling directory whose name begins with the root's
    name, and an in-tree source spelled through the alias.
    UNIT: entries of the returned ``declared_absent`` list and of the ``unresolved_outside`` list.
    Why required: a source misfiled as in-tree is reported as a missing ``external/{name}`` file,
    which is the false alarm this classification exists to remove; one misfiled as out-of-tree
    hides a real in-tree loss under a message that blames the tree's location.
    What kills it: ``str(resolved).startswith(str(analysis_root))`` (files ``analysis_old/x.csv`` as
    in-tree), or comparing the UNRESOLVED source or root (the alias spelling then disagrees with
    the real one, filing in-tree sources as out-of-tree), or a ``os.path.relpath`` taken from the
    unresolved root (the ``../`` depth then differs).
    Second correct implementation: ``try: resolved.relative_to(analysis_root) except ValueError``
    passes unchanged, since the test reads only the two returned lists.
    """
    real_root = tmp_path / "site" / "analysis"
    (real_root / "_status").mkdir(parents=True)
    (tmp_path / "site" / "analysis_old").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_root)
    sources = {
        "r1": [
            real_root / "_status" / "_du.json",  # in-tree absent, spelled through the real root
            alias / "_status" / "_du2.json",  # in-tree absent, spelled through the alias
            tmp_path / "site" / "analysis_old" / "x.csv",  # absent, a sibling sharing the root's name prefix
            tmp_path / "site" / "moved" / "matrix.csv",  # absent, an ordinary out-of-tree neighbour
        ]
    }
    unresolved: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _entries, absent, _shallow = _harvest_sources(sources, alias, unresolved_outside=unresolved)

    assert absent == ["_status/_du.json", "_status/_du2.json"]
    assert unresolved == ["../analysis_old/x.csv", "../moved/matrix.csv"]


def test_an_absent_out_of_tree_store_under_the_omit_flag_is_in_neither_the_shallow_set_nor_the_manifest_omitted_list(
    tmp_path,
):
    """Property: an unresolved out-of-tree ``.zarr`` is reported under its own key, never as an omitted store.

    Class: absent directory-store sources (a name ending ``.zarr``) outside the analysis directory,
    harvested with the omit flag, which forces every harvested ``.zarr`` into the shallow set.
    UNIT: entries of the returned shallow-store list and of ``unresolved_outside``.
    Why required: the shallow list feeds ``omitted_output_zarrs`` and the regeneration Snakefile's
    refusal; a store that is not in the archive named there refuses a rule over a file nobody
    omitted, and the manifest would claim to have omitted what it never held.
    What kills it: recording the store as shallow before the existence check, or classifying it
    after the shallow bookkeeping.
    Second correct implementation: filtering the shallow list against the entries actually planned,
    after the loop, passes unchanged, since the test reads only the returned lists.
    """
    root = tmp_path / "site" / "analysis"
    root.mkdir(parents=True)
    store = tmp_path / "site" / "moved" / "ext.zarr"
    unresolved: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entries, absent, shallow = _harvest_sources(
            {"r": [store]}, root, omit_output_zarrs=True, unresolved_outside=unresolved
        )
    assert unresolved == ["../moved/ext.zarr"]
    assert absent == [] and shallow == [] and entries == {}


def test_the_second_warning_takes_the_harvest_warning_form_and_shows_on_every_harvest(tmp_path):
    """Property: the unresolved-source warning carries the harvest label, line 0, and is shown by EVERY harvest.

    Class: repeated harvests in one process under Python's default warning filter (a sweep emits
    one harvest per member from the same call site), each with the same unresolved source.
    UNIT: warnings recorded for the unresolved kind, counted per harvest.
    Why required: the in-tree warning is issued so that no source line is echoed and no
    once-per-location default hides a later harvest; a sibling issued by ``warnings.warn`` echoes an
    unrelated source line on every entry point and is shown once per process, so the second member
    of a sweep loses its report.
    What kills it: ``warnings.warn(..., stacklevel=n)``, a ``warn_explicit`` given a shared
    ``registry``, or a different label or line number.
    Second correct implementation: any issuer whose recorded ``filename`` equals the harvest label,
    ``lineno`` is 0 and which is not deduplicated across calls passes unchanged.
    """
    root = tmp_path / "site" / "analysis"
    root.mkdir(parents=True)
    sources = {"r": [tmp_path / "site" / "moved" / "matrix.csv"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(2):
            _harvest_sources(sources, root, unresolved_outside=[])
    mine = [w for w in caught if KEY in str(w.message)]
    assert len(mine) == 2, [str(w.message)[:60] for w in caught]
    assert all(w.filename == _HARVEST_WARNING_LABEL and w.lineno == 0 for w in mine)
    assert all(issubclass(w.category, UserWarning) for w in mine)

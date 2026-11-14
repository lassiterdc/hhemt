"""Reviewer tests for the bundle member census: positions the census tests do not reach.

``tests/test_synth_bundle_member_census.py`` proves a full emitted bundle leaks nothing by running one
scan over every member. These tests pin the three ways that scan could still be vacuous while every one of
its own arms stays green:

- a scan that reads only the sniffed prefix of a text member, so a value past the first 8192 bytes is
  never seen (the census arms plant their value at byte 0);
- a scanner that holds layer (a) only, because the census arms build it from an emitter and then assert
  nothing about what layer (b) was given (the unsubstituted bundle is found leaking by layer (a) alone);
- a bundle that no longer carries the config pair, the crate or a zarr metadata document, the three
  members the replaced deposit test scanned, because the census scans whatever members exist and never
  names them.

Producer-looking values are built from fragments; this file is itself scanned by the repository's
anonymization guard. The solver-bearing arms reuse the census module's ``emitted`` fixture and run on
Rivanna scratch only; the solver-free arms run anywhere.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._placeholders import SNIFF_BYTES, prefix_is_text
from tests import test_synth_bundle_member_census as _census
from tests.test_synth_bundle_member_census import (
    PLANTED_HOME,
    _fresh_scanner,
    _in_zarr_store,
    _layer_a_scanner,
    _write_zip,
    census_archive,
)

#: The census module's two-bundle fixture, bound under its own name so pytest registers it for this module.
emitted = _census.emitted

#: The members the replaced deposit test scanned, as they are named inside an emitted bundle.
POPULATION_NAMES = ("cfg_analysis.yaml", "cfg_system.yaml", "ro-crate-metadata.json")

#: A configured absolute path under no per-user root, so layer (a) cannot see it and only layer (b) can.
CONFIGURED_PATH = "/" + "data" + "/" + "zzproj" + "/" + "run1" + "/" + "in.tif"


def unscanned_population(archive: Path) -> list[str]:
    """The names, from the replaced test's population, that ``archive`` lacks or carries as a non-text member."""
    missing: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for name in POPULATION_NAMES:
            if name not in names or not prefix_is_text(zf.read(name)[:SNIFF_BYTES]):
                missing.append(name)
        has_store_metadata = any(
            name.rsplit("/", 1)[-1] == "zarr.json"
            and _in_zarr_store(name)
            and prefix_is_text(zf.read(name)[:SNIFF_BYTES])
            for name in names
        )
        if not has_store_metadata:
            missing.append("a zarr.json document under a *.zarr store")
    return missing


def _emitter_holding(path: str) -> SimpleNamespace:
    """A stand-in emitter whose system config holds ``path`` as a string leaf, the way a real config does."""

    class _Dump:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def model_dump(self, mode: str = "json") -> dict:
            return self._payload

    return SimpleNamespace(
        _system=SimpleNamespace(cfg_system=_Dump({"dem": path})), cfg_analysis=_Dump({}), cfg_hpc_system=None
    )


def test_the_census_reads_a_text_member_past_the_sniff_window(tmp_path):
    """Property: a producer value sitting past the first ``SNIFF_BYTES`` of a text member is reported.

    Class: NEW CAPABILITY - the plausible wrong implementation sniffs a member's prefix and then scans only
        that prefix (``scan_member(name, data[:SNIFF_BYTES])``), and is red under it: the value is at offset
        ``2 * SNIFF_BYTES`` and the census arms of the coder's file all plant theirs at offset 0.
    Reachability: a generated Snakefile or a harvested ``.inp`` runs to tens of kilobytes and names the
        analysis directory in its later rules; the emit reads such a member whole (``data = prefix +
        fh.read()``) and scans all of it, artifact ``_emit.py`` ``_emit_bundle_zip``.
    Both-states anchor: the assertion turns on where the value sits in the member, not on wording; the same
        member with the value at offset 0 is reported by both the correct and the mutant census.
    Invocation isolated: a pure function of the archive bytes; no tree, environment or working directory.
    What kills it: scanning the sniffed prefix instead of the member; scanning a fixed number of leading
        lines.
    Second correct implementation: unpack the archive to a directory and call ``scan_member`` on each file's
        ``read_bytes()``, which reads the whole member too.
    """
    filler = b"x = 1\n" * SNIFF_BYTES
    deep = _write_zip(tmp_path / "deep.zip", {"Snakefile.source": filler + f"cd {PLANTED_HOME}\n".encode()})
    assert len(filler) > 2 * SNIFF_BYTES
    found = census_archive(deep, _layer_a_scanner(), frozenset())
    assert list(found.leaking) == ["Snakefile.source"], found.leaking


def test_a_scanner_built_from_an_emitter_flags_a_configured_path_that_layer_a_cannot_see(tmp_path):
    """Property: the census scanner holds layer (b): a configured absolute path is reported under its layer.

    Class: NEW CAPABILITY - the plausible wrong implementation builds ``_fresh_scanner`` as a layer-(a)-only
        scanner (or from an object that lacks the configs, which ``for_emit`` tolerates by contributing
        nothing), and is red under it: the member holds a path under no per-user root, which only layer (b)
        can match.
    Reachability: ``ZeroUserInfoScanner.for_emit`` reads ``_system.cfg_system`` and ``cfg_analysis`` by
        ``getattr`` and skips an absent one silently (``_reprex_gate.py``), so a wrong emitter degrades the
        scan to layer (a) with no error; the full-bundle arms cannot tell, because layer (a) alone already
        finds the per-user paths of a Rivanna scratch directory.
    Both-states anchor: the assertion turns on the layer that reports the member; the same bytes are clean
        under a layer-(a)-only scanner (the control asserted below), so the difference is the configs.
    Invocation isolated: a stand-in emitter and a hand-built archive; no tree, environment or fixture.
    What kills it: ``_fresh_scanner`` returning ``ZeroUserInfoScanner()``; passing it an object with no
        configs.
    Second correct implementation: build the scanner with ``ZeroUserInfoScanner.for_emit`` directly on the
        emitter inside the test, which holds the same layer-(b) values.
    """
    archive = _write_zip(tmp_path / "cfg.zip", {"Snakefile.source": f"input: {CONFIGURED_PATH}\n".encode()})
    assert census_archive(archive, _layer_a_scanner(), frozenset()).leaking == {}
    found = census_archive(archive, _fresh_scanner(_emitter_holding(CONFIGURED_PATH)), frozenset())
    assert list(found.leaking) == ["Snakefile.source"], found.leaking
    assert found.leaking["Snakefile.source"][0].startswith(f"Snakefile.source: {G.LAYER_CONFIG_VALUES}: ")


def test_the_population_check_names_each_member_of_the_replaced_test_that_an_archive_lacks(tmp_path):
    """Property: ``unscanned_population`` is empty for an archive that carries the three members as text plus a
        store metadata document, and names exactly the member that is absent or binary.

    Class: NEW CAPABILITY - the plausible wrong implementation reports presence from the member count alone
        (any non-empty text population), and is red under it because an archive without the crate still has
        hundreds of text members.
    Reachability: the crate is copied into the bundle only ``if crate_src.exists()`` (``_emit.py``), so a
        bundle emitted without the analysis sidecar lacks it and the census, which scans what exists,
        stays green.
    Both-states anchor: the assertion turns on membership of one named member, with every other member held.
    Invocation isolated: hand-built archives only.
    What kills it: dropping the name test; treating a NUL-prefixed config as present.
    Second correct implementation: open each name with ``ZipFile.getinfo`` and sniff its first bytes.
    """
    text = b"plain text\n"
    members = {
        "cfg_analysis.yaml": text,
        "cfg_system.yaml": text,
        "ro-crate-metadata.json": text,
        "analysis_datatree.zarr/zarr.json": b"{}",
        "analysis_datatree.zarr/c/0": b"\x00\x01",
    }
    whole = _write_zip(tmp_path / "whole.zip", members)
    no_crate = _write_zip(
        tmp_path / "no_crate.zip", {k: v for k, v in members.items() if k != "ro-crate-metadata.json"}
    )
    binary_cfg = _write_zip(tmp_path / "binary_cfg.zip", {**members, "cfg_system.yaml": b"\x00\x01"})
    no_store = _write_zip(
        tmp_path / "no_store.zip", {k: v for k, v in members.items() if not k.startswith("analysis_datatree.zarr")}
    )
    assert unscanned_population(whole) == []
    assert unscanned_population(no_crate) == ["ro-crate-metadata.json"]
    assert unscanned_population(binary_cfg) == ["cfg_system.yaml"]
    assert unscanned_population(no_store) == ["a zarr.json document under a *.zarr store"]


def test_the_emitted_bundle_carries_as_text_the_members_the_replaced_deposit_test_scanned(emitted):
    """Property: the emitted bundle holds the config pair, the crate and a zarr store metadata document as text.

    Class: NEW CAPABILITY - the plausible wrong implementation is a bundle that omits one of them (the
        crate copy is conditional on the analysis sidecar), and is green under the census, which scans
        whatever members exist; this test is red under it.
    Reachability: the replaced test scanned exactly these members of a deposit set it built itself; the
        census replaces it over a bundle the fixture produces, so their presence is the fixture's fact and
        not the test's construction.
    Both-states anchor: the assertion turns on the archive's member names and sniffed class, not on any
        wording.
    Invocation isolated: not isolatable from this tree (a solver-bearing fixture); discrimination is shown by
        the hand-built arm above, which is red when one member is removed.
    What kills it: an emit that stops writing the crate, the config pair or the store metadata documents.
    Second correct implementation: read the bundle manifest's file list instead of the archive's names.
    """
    assert unscanned_population(emitted.clean) == []


def test_the_unsubstituted_bundle_trips_a_layer_b_check_and_not_only_a_pattern(emitted):
    """Property: with substitution off, the census reports at least one member under the config-values layer.

    Class: NEW CAPABILITY - the plausible wrong implementation is a census scanner that holds layer (a)
        only, and is red under it: the full-bundle arms of the coder's file pass on layer (a) alone, since
        the unsubstituted bundle holds per-user scratch paths that the path pattern finds.
    Reachability: the unsubstituted Snakefile names the analysis directory, which is a string leaf of the
        analysis config and so a layer-(b) value.
    Both-states anchor: the assertion turns on which layer reports, with the archive held fixed.
    Invocation isolated: not isolatable from this tree; discrimination is shown by the solver-free arm
        ``test_a_scanner_built_from_an_emitter_flags_a_configured_path_that_layer_a_cannot_see``.
    What kills it: ``_fresh_scanner`` built from the wrong object or without configs.
    Second correct implementation: assert on ``len(scanner._config_checks) > 0`` before the scan.
    """
    raw = census_archive(emitted.raw, _fresh_scanner(emitted.emitter), emitted.declared)
    by_config_value = sorted(
        member for member, hits in raw.leaking.items() if any(f": {G.LAYER_CONFIG_VALUES}: " in hit for hit in hits)
    )
    assert by_config_value, "no member was reported by layer (b); the scanner may hold layer (a) only"

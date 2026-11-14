"""A full emitted bundle carries no producer-specific value in any text member, and says so member by member.

``emit_bundle`` rewrites the producer's directories, interpreter, login name and SLURM account to the
closed ``{hhemt_*}`` tokens as it writes each text member, then scans every text member with three
layers and refuses the archive on a hit. This module is the independent proof that the refusal's
population is the bundle and not a sample of it: it emits the two full synthetic bundles (a multi-sim
analysis and a sensitivity analysis), unpacks each, and scans EVERY member again with layers built
from the fixture's own configs and the running login name, plus the private list when
``$HHEMT_REPREX_BLOCKLIST`` is set. Three properties are asserted per bundle:

- no text member leaks (the census);
- every binary member is a zarr chunk under a harvested store or a declared input (the only member
  classes the emit copies through byte for byte, so the only places a value could hide unrewritten);
- the same scan finds a leak when substitution is switched off (the must-fire arm), so a zero above
  is a statement about the bundle and not about a scan that cannot fire.

The synthetic-fixture arms build and run the solver, so they run on Rivanna scratch only, with
``HHEMT_COMPILE_VENUE=toolchain``; the fixture is requested inside the module fixture AFTER the venue
check, so collecting this module on a laptop builds nothing. The census helper is exercised by
solver-free arms over hand-built archives, which run anywhere.

Producer-looking values (per-user paths, a Windows home, an address) are BUILT FROM FRAGMENTS below,
never written as literals: this file is itself scanned by the repository's anonymization guard, which
refuses a literal that matches its patterns.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What
kills it, and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._placeholders import (
    SNIFF_BYTES,
    PlaceholderTable,
    is_zarr_chunk_member_by_name,
    prefix_is_text,
)
from hhemt.bundle._reprex_gate import ZeroUserInfoScanner

SYNTH_FIXTURES = ("rendered_synth_multi_sim", "rendered_synth_sensitivity")


# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------


def _unix(root: str, user: str, rest: str = "x") -> str:
    """A path under a per-user root, assembled so no literal of the form appears in this file."""
    return "/" + root + "/" + user + "/" + rest


PLANTED_HOME = _unix("home", "someone")
PLANTED_SCRATCH = _unix("scratch", "someone", "run")


# --------------------------------------------------------------------------------------
# The census: one pass over every member of an unpacked archive
# --------------------------------------------------------------------------------------


@dataclass
class MemberCensus:
    """What a scan of one archive found. Hits carry a layer and a class label or value index, never a value."""

    text_members: int = 0
    binary_members: int = 0
    zarr_chunk_members: int = 0
    leaking: dict[str, list[str]] = field(default_factory=dict)
    binary_outside: list[str] = field(default_factory=list)


def _in_zarr_store(arcname: str) -> bool:
    """True when some directory on the member's path is a ``*.zarr`` store."""
    return any(part.endswith(".zarr") for part in arcname.split("/")[:-1])


def census_archive(archive: Path, scanner: ZeroUserInfoScanner, declared: frozenset[str]) -> MemberCensus:
    """Scan every member of ``archive`` and classify the members that were not scanned.

    A zarr chunk (classed binary by its name, as the emit classes it) and a member whose first
    ``SNIFF_BYTES`` carry a NUL byte are binary and are not scanned; every other member is text and is
    scanned in full, declared inputs included, because the emit scans those too. A binary member is
    reported in ``binary_outside`` unless it sits under a ``*.zarr`` store or is a declared input.
    """
    metadata_names = _emit._zarr_metadata_filenames()
    census = MemberCensus()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir():
                continue
            if is_zarr_chunk_member_by_name(name, metadata_names):
                census.binary_members += 1
                census.zarr_chunk_members += 1
                continue
            data = zf.read(name)
            if not prefix_is_text(data[:SNIFF_BYTES]):
                census.binary_members += 1
                if not _in_zarr_store(name) and name not in declared:
                    census.binary_outside.append(name)
                continue
            census.text_members += 1
            hits = scanner.scan_member(name, data)
            if hits:
                census.leaking[name] = [hit.render() for hit in hits]
    return census


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _layer_a_scanner() -> ZeroUserInfoScanner:
    """A scanner holding layer (a) only: nothing is read from a config and no private list is loaded."""
    return ZeroUserInfoScanner()


# --------------------------------------------------------------------------------------
# The census helper, over hand-built archives (solver-free)
# --------------------------------------------------------------------------------------


def test_the_census_reports_each_leaking_text_member_by_name_and_layer_and_clears_a_token_form(tmp_path):
    """Property: a text member holding a per-user path is reported under its own name and layer; the same
        member holding the placeholder token instead is not.

    Class: NEW CAPABILITY - the plausible wrong implementation is a census that scans only a fixed set of
        names (the config pair and the crate, which is what the replaced test did), and is red under it
        because the leak sits in a member of another name.
    Reachability: a member named ``Snakefile.source`` carries the path; the layer-(a) path class fires on a
        home-directory form and the member is plain text, so the sniff admits it to the scan.
    Both-states anchor: the assertion turns on the member's content (a path versus its token), not on any
        wording the change introduced.
    Invocation isolated: a pure function of the archive bytes; no tree, environment or working directory.
    What kills it: scanning a named subset of members; skipping members whose name has no suffix.
    Second correct implementation: unpack to a directory and walk it with ``assert_bundle_zero_user_info``
        per member, which reports the same members.
    """
    leaky = _write_zip(
        tmp_path / "leaky.zip",
        {"Snakefile.source": f"cd {PLANTED_HOME}/work\n".encode(), "notes.txt": b"nothing private here\n"},
    )
    clean = _write_zip(
        tmp_path / "clean.zip",
        {"Snakefile.source": b"cd {hhemt_home}/work\n", "notes.txt": b"nothing private here\n"},
    )
    found = census_archive(leaky, _layer_a_scanner(), frozenset())
    assert list(found.leaking) == ["Snakefile.source"], found.leaking
    assert found.leaking["Snakefile.source"][0].startswith("Snakefile.source: pattern_rules: ")
    assert found.text_members == 2
    assert census_archive(clean, _layer_a_scanner(), frozenset()).leaking == {}


def test_the_census_scans_a_zarr_metadata_document_but_not_a_zarr_chunk(tmp_path):
    """Property: a zarr store's metadata document is scanned as text and its chunks are counted, not scanned.

    Class: NEW CAPABILITY - the plausible wrong implementation treats every member under a ``.zarr`` store
        as binary, and is red under it because a recorded source path in ``zarr.json`` is then never seen.
    Reachability: ``zarr.json`` is one of the store's metadata names, so the emit sniffs it and scans it;
        the chunk ``0`` is classed binary by name and carries a path-shaped byte string.
    Both-states anchor: the assertion turns on which of the two same-store members is scanned.
    Invocation isolated: a pure function of the archive bytes.
    What kills it: classing the whole store as binary; classing a member as a chunk by its basename alone.
    Second correct implementation: classify by the sniff alone and count a chunk as text when its first
        bytes hold no NUL, which scans the chunk too and still finds the metadata leak.
    """
    archive = _write_zip(
        tmp_path / "store.zip",
        {
            "analysis_datatree.zarr/zarr.json": ('{"source_file": "' + PLANTED_SCRATCH + '"}').encode(),
            "analysis_datatree.zarr/c/0": PLANTED_SCRATCH.encode(),
        },
    )
    found = census_archive(archive, _layer_a_scanner(), frozenset())
    assert list(found.leaking) == ["analysis_datatree.zarr/zarr.json"], found.leaking
    assert found.zarr_chunk_members == 1 and found.text_members == 1 and found.binary_outside == []


def test_the_census_names_a_binary_member_that_is_neither_a_zarr_chunk_nor_a_declared_input(tmp_path):
    """Property: a binary member outside the zarr stores and the declared inputs is reported; those two
        classes are not.

    Class: NEW CAPABILITY - the plausible wrong implementation counts every binary member as accounted
        for, and is red under it because an undeclared binary (a nested archive, say) passes silently.
    Reachability: three NUL-prefixed members, one under a store, one named as a declared input, one neither.
    Both-states anchor: the assertion turns on membership in the two permitted classes.
    Invocation isolated: a pure function of the archive bytes and the declared-name set.
    What kills it: dropping the declared-name test; dropping the store test; reporting nothing.
    Second correct implementation: collect all binary members first and subtract the two permitted sets.
    """
    nul = b"\x00\x01\x02"
    archive = _write_zip(
        tmp_path / "binary.zip",
        {"a.zarr/c/0": nul, "external/dem.tif": nul, "nested/other.bin": nul, "readme.txt": b"plain\n"},
    )
    found = census_archive(archive, _layer_a_scanner(), frozenset({"external/dem.tif"}))
    assert found.binary_outside == ["nested/other.bin"], found.binary_outside
    assert found.binary_members == 3 and found.zarr_chunk_members == 1


def test_the_census_finds_a_private_list_token_only_when_a_list_is_loaded(tmp_path):
    """Property: a layer-(c) token is reported with its 1-based index when the list is loaded, and nothing
        is reported for the same bytes when it is not.

    Class: NEW CAPABILITY - the plausible wrong implementation drops layer (c) from the scan, and is red
        under it because the token member is then clean.
    Reachability: the scanner is built with one private pattern, the way ``for_consume`` builds it from a
        carrier; the member holds the token as a whole word.
    Both-states anchor: the assertion turns on the presence of the loaded list, with the member unchanged.
    Invocation isolated: no environment variable is read; the pattern list is passed in.
    What kills it: ignoring the private checks; matching the token inside a longer word.
    Second correct implementation: load the list from a carrier file named by the variable and scan.
    """
    archive = _write_zip(tmp_path / "token.zip", {"notes.txt": b"the zztokenalpha site\n"})
    loaded = ZeroUserInfoScanner(_private_checks=G._private_patterns(["zztokenalpha"]))
    found = census_archive(archive, loaded, frozenset())
    assert found.leaking == {"notes.txt": ["notes.txt: private_list: private list token #1"]}, found.leaking
    assert census_archive(archive, _layer_a_scanner(), frozenset()).leaking == {}


# --------------------------------------------------------------------------------------
# The two full bundles (Rivanna scratch: solver-bearing fixtures)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=SYNTH_FIXTURES)
def emitted(request, tmp_path_factory):
    """Emit the fixture's full bundle twice: as the toolkit emits it, and with substitution switched off.

    The synthetic fixture is requested HERE, after the venue check, so a laptop that collects this module
    builds nothing. The second emit replaces the placeholder table with an empty one and lets the emit's
    own refusal pass, which is the only way to obtain an archive that still holds the producer's values.
    """
    from hhemt._compile_venue import venue_is_permitted

    if not venue_is_permitted():
        pytest.skip("OWED on Rivanna scratch: needs HHEMT_COMPILE_VENUE=toolchain (a solver-bearing fixture)")
    analysis = request.getfixturevalue(request.param)
    emitter = analysis.sensitivity if getattr(analysis, "sensitivity", None) is not None else analysis
    out = tmp_path_factory.mktemp(f"census_{request.param}")
    clean = _emit.emit_bundle(emitter, out / "clean.zip")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_emit, "build_placeholder_table", lambda _analysis: PlaceholderTable.from_values({}))
        mp.setattr(ZeroUserInfoScanner, "raise_if_hits", lambda self, *args, **kwargs: None)
        raw = _emit.emit_bundle(emitter, out / "raw.zip")
    planned, _excluded = _emit._plan_declared_inputs(emitter, None)
    declared = frozenset(name for name, _src in _emit._declared_destinations(planned))
    return SimpleNamespace(name=request.param, emitter=emitter, clean=clean, raw=raw, declared=declared)


def _fresh_scanner(emitter) -> ZeroUserInfoScanner:
    """Layers (a) and (b) from the emitter's own configs and the login name, plus (c) when the variable is set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ZeroUserInfoScanner.for_emit(emitter)


def _private_list_members(census: MemberCensus) -> list[str]:
    """The members whose hits include a layer-(c) hit."""
    return sorted(m for m, hits in census.leaking.items() if any(f": {G.LAYER_PRIVATE_LIST}: " in h for h in hits))


def _report(label: str, census: MemberCensus) -> None:
    print(
        f"CENSUS {label}: text_members={census.text_members} binary_members={census.binary_members} "
        f"zarr_chunk_members={census.zarr_chunk_members} leaking_text_members={len(census.leaking)} "
        f"binary_outside={len(census.binary_outside)}"
    )


def test_a_full_emitted_bundle_has_no_leaking_text_member(emitted):
    """Property: no text member of the full emitted bundle holds a producer-specific value.

    Class: NEW CAPABILITY - the plausible wrong implementation scans the config pair, the crate and the
        zarr store (the replaced test's population), and is green under it while the Snakefile, the
        baseline page, the status csv and the copied SWMM inputs are unchecked; the must-fire arm below is
        red under a scan that cannot reach them.
    Reachability: the bundle is emitted from a completed, rendered synthetic analysis, so its members
        include the generated Snakefile, the baseline report page and the harvested figure sources, each
        of which the producer's directories reach before substitution.
    Both-states anchor: the assertion turns on the member bytes after the rewrite; the paired arm scans the
        same members before it.
    Invocation isolated: not isolatable from this tree; discrimination is shown instead by the must-fire
        arm, whose only difference is the substitution switch.
    What kills it: any change that leaves a producer directory, interpreter, login name or account in a
        text member of the archive; shrinking the scanned member population.
    Second correct implementation: unpack the archive and run ``assert_bundle_zero_user_info`` on the
        directory, which reads the same members and reports the same leaks.
    """
    census = census_archive(emitted.clean, _fresh_scanner(emitted.emitter), emitted.declared)
    _report(f"{emitted.name} substituted", census)
    assert census.text_members > 0, "the text-member population is zero; the absence below would be vacuous"
    assert census.leaking == {}, f"leaking text members: {sorted(census.leaking)}"


def test_a_full_emitted_bundle_has_no_binary_member_outside_zarr_stores_and_declared_inputs(emitted):
    """Property: every binary member is a zarr chunk under a harvested store or a declared input.

    Class: NEW CAPABILITY - the plausible wrong implementation treats "binary" as "cannot leak" without
        naming what binary members exist, and is red under it when a nested archive (the baseline zip the
        emit no longer carries) or any other undeclared binary member appears in the bundle.
    Reachability: the declared inputs of a synthetic analysis are a DEM, a land-use raster and a weather
        NetCDF, which are binary and carried byte for byte; the harvested stores hold the chunks.
    Both-states anchor: the assertion turns on the set of binary member names, whatever their count.
    Invocation isolated: not isolatable from this tree; the hand-built archive arm above shows the
        reporting works on a member of neither class.
    What kills it: carrying a nested archive or any other undeclared binary file in the bundle.
    Second correct implementation: derive the permitted names from the manifest's input records instead
        of from the emit's declared-input plan.
    """
    census = census_archive(emitted.clean, _fresh_scanner(emitted.emitter), emitted.declared)
    assert census.binary_members > 0, "no binary member was seen; the zarr stores did not reach the bundle"
    assert census.binary_outside == [], f"binary members outside the permitted classes: {census.binary_outside}"


def test_the_same_scan_finds_a_leak_when_substitution_is_switched_off(emitted):
    """Property: with the placeholder rewrite disabled, the census finds at least one leaking text member.

    Class: NEW CAPABILITY - the control for the two arms above. The plausible wrong implementation is a
        scan that cannot see the producer's values at all, and is green on the substituted bundle for that
        reason; this arm is red under it.
    Reachability: the bundle's generated Snakefile holds the producer's interpreter and directories before
        the rewrite, so the layer-(a) and layer-(b) patterns have something to find.
    Both-states anchor: the assertion turns on the substitution switch alone; the scanner, the analysis and
        the member classification are the same objects as in the substituted arm.
    Invocation isolated: the two archives come from one analysis in one process; only the table differs.
    What kills it: a scanner built so that it matches nothing; the rewrite running in the disabled emit.
    Second correct implementation: write the unsubstituted bundle by copying the staging tree directly and
        scan that, which holds the same producer values.
    """
    scanner = _fresh_scanner(emitted.emitter)
    raw = census_archive(emitted.raw, scanner, emitted.declared)
    clean = census_archive(emitted.clean, _fresh_scanner(emitted.emitter), emitted.declared)
    _report(f"{emitted.name} substitution disabled", raw)
    assert len(raw.leaking) >= 1, "the scan found nothing in an unsubstituted bundle; it cannot fire"
    assert len(raw.leaking) > len(clean.leaking), (len(raw.leaking), len(clean.leaking))


def test_a_private_list_token_is_found_in_the_unsubstituted_bundle_and_not_in_the_emitted_one(
    emitted, tmp_path, monkeypatch
):
    """Property: a private-list token naming the fixture's own run directory is absent from the emitted
        bundle's text members and present in the unsubstituted one.

    Class: NEW CAPABILITY - the plausible wrong implementation loads no list when the variable is set, and
        is red under it because the unsubstituted bundle then shows no layer-(c) hit.
    Reachability: the session directory that holds the fixture's analysis is named from the fixture, is a
        component of every absolute producer path the rewrite replaces, and is written into a synthetic
        carrier file built here; no real private list is read or set.
    Both-states anchor: the assertion turns on the substitution switch, with the carrier held fixed.
    Invocation isolated: the variable is set for this test only, and the archives are not re-emitted.
    What kills it: leaving the session directory in a text member; a gate that ignores the variable.
    Second correct implementation: use the login name as the token and scan the same pair of archives.
    """
    analysis_dir = Path(emitted.emitter.analysis_paths.analysis_dir)
    token = next((part for part in analysis_dir.parts if part.startswith("rendered_synth")), None)
    assert token is not None, f"no fixture-named directory in {analysis_dir}; the token cannot be derived"
    carrier = tmp_path / "carrier.txt"
    carrier.write_text("# min-tokens: 1\n" + token + "\n", encoding="utf-8")
    monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", str(carrier))
    scanner = _fresh_scanner(emitted.emitter)
    assert scanner.private_list_state == G.PRIVATE_LIST_CHECKED
    clean = census_archive(emitted.clean, scanner, emitted.declared)
    raw = census_archive(emitted.raw, scanner, emitted.declared)
    in_clean, in_raw = _private_list_members(clean), _private_list_members(raw)
    print(f"CENSUS {emitted.name} private-list arm: substituted={len(in_clean)} unsubstituted={len(in_raw)}")
    assert in_clean == [], in_clean
    assert len(in_raw) >= 1, "the synthetic carrier token never fired on the unsubstituted bundle"

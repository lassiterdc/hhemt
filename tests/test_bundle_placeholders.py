"""Producer values in a bundle's text members are replaced by closed ``{hhemt_*}`` tokens at archive-write time.

SOLVER-FREE: every arm builds a ``PlaceholderTable`` from plain strings, writes small files under
``tmp_path`` and runs ``_emit_bundle_zip`` (or, for the wiring arm, the real ``emit_bundle`` over a
temporary tree). No fixture here compiles or runs a solver.

Producer-looking values (home directories, scratch paths, a login name) are BUILT FROM FRAGMENTS
below, never written as literals, so this file is not itself a hit for the repository's
anonymization guard.

Each test docstring states Property, Class, Why required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _placeholders as P
from hhemt.bundle._emit import _copy_bundle_baseline, _emit_bundle_zip
from hhemt.bundle._reprex_gate import _SNIFF_BYTES, _is_text

# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------

LOGIN = "carol" + "ine"  # a login name of at least four characters
HOME = "/ho" + "me/" + LOGIN
SCRATCH = "/scr" + "atch/" + LOGIN
SYSTEM_DIR = SCRATCH + "/runs"
ANALYSIS_DIR = SYSTEM_DIR + "/a1"  # the analysis directory lies INSIDE the system directory
TOOLKIT = HOME + "/code/hhemt"
CONDA_ENV = TOOLKIT + "/workflow/envs/hhemt.yaml"  # the conda path lies INSIDE the toolkit directory
PYTHON = HOME + "/miniconda3/envs/e/bin/python"  # the interpreter lies INSIDE the home directory
ACCOUNT = "hydro-lab"

ALL_VALUES = {
    P.TOKEN_ANALYSIS_DIR: [ANALYSIS_DIR],
    P.TOKEN_SYSTEM_DIR: [SYSTEM_DIR],
    P.TOKEN_TOOLKIT_DIR: [TOOLKIT],
    P.TOKEN_PYTHON: [PYTHON],
    P.TOKEN_CONDA_ENV: [CONDA_ENV],
    P.TOKEN_HOME: [HOME],
    P.TOKEN_USER: [LOGIN],
    P.TOKEN_SLURM_ACCOUNT: [ACCOUNT],
}


def _table(**overrides) -> P.PlaceholderTable:
    values = dict(ALL_VALUES)
    values.update(overrides)
    return P.PlaceholderTable.from_values(values)


def _rewrite(text: str, table: P.PlaceholderTable | None = None) -> str:
    return (table or _table()).apply(text.encode()).decode()


def _put(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def _read_zip(zpath: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zpath) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


# --------------------------------------------------------------------------------------
# Table rows: each replaces its value
# --------------------------------------------------------------------------------------

ROW_CASES = [
    (P.TOKEN_ANALYSIS_DIR, ANALYSIS_DIR),
    (P.TOKEN_SYSTEM_DIR, SYSTEM_DIR),
    (P.TOKEN_TOOLKIT_DIR, TOOLKIT),
    (P.TOKEN_PYTHON, PYTHON),
    (P.TOKEN_CONDA_ENV, CONDA_ENV),
    (P.TOKEN_HOME, HOME),
    (P.TOKEN_SLURM_ACCOUNT, ACCOUNT),
]


@pytest.mark.parametrize(("token", "value"), ROW_CASES, ids=[t for t, _ in ROW_CASES])
def test_each_row_replaces_its_value_wherever_it_stands_alone(token, value):
    """Property: a value of the table is replaced by its token in two differently positioned contexts.

    Class: REGRESSION - red on the parent, which has no ``_placeholders`` module (ImportError at collection).
    Reachability: a text member holding the value as a shell word or as a quoted YAML scalar reaches the
        substitution; artifact: the census rows of the bundled Snakefile and config pair.
    Both-states anchor: the assertion turns on the VALUE disappearing, a property of the member's bytes, not on
        the token's spelling alone (the absence assertion fails if the value is merely appended to).
    Invocation isolated: pure function of a table and bytes; no tree, path or environment is consulted.
    What kills it: a table that omits the row, a regex that anchors on whitespace only (the quoted case), a
        replacement that is not the row's own token.
    Second correct implementation: one ``bytes.replace`` per row applied longest first.
    """
    as_shell_word = _rewrite(f"cd {value} && run")
    as_quoted_scalar = _rewrite(f'key: "{value}"\n')
    assert as_shell_word == f"cd {token} && run"
    assert as_quoted_scalar == f'key: "{token}"\n'
    assert value not in as_shell_word + as_quoted_scalar


def test_the_user_row_replaces_a_login_name_only_in_its_three_contexts():
    """Property: the login name becomes ``{hhemt_user}`` as a whole path segment, an e-mail local part, or a
        ``--user`` / ``-u`` / ``user=`` value, and nowhere else.

    Class: NEW CAPABILITY - plausible wrong implementation is a bare substring replace of the login name; it
        reds the "unchanged" arm (the name inside a longer word). Output pasted in the round file.
    Reachability: each context is a form the census found in the bundled Snakefile and status page.
    Both-states anchor: the unchanged arm asserts that bytes holding the name inside a longer word are the
        SAME before and after, which holds in both the old and the new code only if the new code refuses it.
    What kills it: a substring replace; dropping a context; a lookahead that lets the name continue.
    Second correct implementation: one combined regex with named alternatives instead of three sequential ones.
    """
    token = P.TOKEN_USER
    replaced = {
        f"/data/{LOGIN}/x": f"/data/{token}/x",
        f"mail {LOGIN}@example.org now": f"mail {token}@example.org now",
        f"srun --user {LOGIN} x": f"srun --user {token} x",
        f"srun --user={LOGIN} x": f"srun --user={token} x",
        f"srun -u {LOGIN} x": f"srun -u {token} x",
        f"owner user={LOGIN};": f"owner user={token};",
    }
    for before, after in replaced.items():
        assert _rewrite(before) == after, before
    # Same bytes in, same bytes out: the name continues into a longer word, or stands as plain prose.
    for untouched in (f"{LOGIN}ville", f"x{LOGIN}", f"the {LOGIN} said", f"/data/{LOGIN}x/y", f"--user {LOGIN}x"):
        assert _rewrite(untouched) == untouched, untouched


# --------------------------------------------------------------------------------------
# Order: real nesting
# --------------------------------------------------------------------------------------


def test_longest_first_holds_for_each_real_nesting():
    """Property: where one value lies inside another, the INNER occurrence takes its own token and the OUTER
        occurrence takes the outer token, with no mangled suffix.

    Class: REGRESSION - red on the parent (no module); NEW CAPABILITY - plausible wrong implementation is one
        ``replace`` per row in table order (analysis, system, toolkit, python, conda, home): it rewrites the
        toolkit prefix of the conda path and the home prefix of the interpreter, and this test reds under it.
    Reachability: the three nestings are the ones the census measured (19 of 19 toolkit hits inside the conda
        path; every interpreter under home; the analysis directory under the system directory).
    Both-states anchor: the expected strings name the LONGER token for the longer value, a property of the
        table's contents and not of any wording this change introduces.
    Invocation isolated: pure function of a table and bytes.
    What kills it: shortest-first or table-order substitution.
    Second correct implementation: sorting the rows by length and applying one ``bytes.replace`` each.
    """
    assert _rewrite(CONDA_ENV) == P.TOKEN_CONDA_ENV
    assert _rewrite(TOOLKIT + "/src") == P.TOKEN_TOOLKIT_DIR + "/src"
    assert _rewrite(PYTHON) == P.TOKEN_PYTHON
    assert _rewrite(HOME + "/other/bin") == P.TOKEN_HOME + "/other/bin"
    assert _rewrite(ANALYSIS_DIR) == P.TOKEN_ANALYSIS_DIR
    assert _rewrite(SYSTEM_DIR + "/sibling") == P.TOKEN_SYSTEM_DIR + "/sibling"
    assert _rewrite(ANALYSIS_DIR + "/plots/f.html") == P.TOKEN_ANALYSIS_DIR + "/plots/f.html"


def test_a_written_token_is_never_rewritten_and_application_is_idempotent():
    """Property: tokens already in a member, and Snakemake / Python brace syntax, are left alone, and applying
        the table twice equals applying it once.

    Class: NEW CAPABILITY - plausible wrong implementation is a table that treats the braces as a pattern or
        re-scans its own output; it reds the idempotence and the brace-syntax arms.
    Reachability: a bundle emitted from a bundle (combine) can hold tokens already; Snakefile.source holds
        ``{input}`` and ``{wildcards.event_id}``.
    Both-states anchor: output equals input for these strings in the parent too (it rewrites nothing), so the
        arm turns on the new code NOT damaging them.
    What kills it: an over-broad pattern, a two-pass substitution that matches inside a written token.
    Second correct implementation: a single-pass regex with a dictionary lookup.
    """
    text = "{input} {wildcards.event_id} {hhemt_home}/x {hhemt_unknown} {0:>4}"
    assert _rewrite(text) == text
    once = _rewrite(f"{HOME}/a {ANALYSIS_DIR}/b --user {LOGIN}")
    assert _rewrite(once) == once


def test_an_unknown_token_is_refused_because_the_set_is_closed():
    """Property: a table cannot be built with a token outside the closed set.

    Class: NEW CAPABILITY - plausible wrong implementation accepts any ``{hhemt_*}`` key; red under it.
    Reachability: a caller adding a row by passing a new key.
    Both-states anchor: the refusal turns on the key's membership in the closed tuple.
    What kills it: dropping the membership check.
    Second correct implementation: validating in ``from_values`` instead of ``__init__``.
    """
    with pytest.raises(ValueError, match="closed"):
        P.PlaceholderTable.from_values({"{hhemt_new}": [HOME]})


# --------------------------------------------------------------------------------------
# The guards: absolute-path guard scope and the four-character floor
# --------------------------------------------------------------------------------------


def test_the_absolute_path_guard_is_scoped_to_the_six_path_valued_rows():
    """Property: a non-absolute value in a PATH-VALUED row is never substituted, while a name of four or more
        characters in the NAME-VALUED user row is still substituted in its context.

    Class: NEW CAPABILITY - plausible wrong implementation applies the guard in every row; arm 2 reds under it
        (a login name is not an absolute path). A second wrong implementation applies no guard; arm 1 reds.
    Reachability: arm 1 is a sensitivity builder's bare ``python``; arm 2 is the login name in ``--user``.
    Both-states anchor: arm 1 asserts the bare word survives and arm 2 that the name does not; each turns on the
        guard's scope, in a differently positioned input in the SAME fragment.
    Invocation isolated: pure function of a table and bytes.
    What kills it: dropping the guard (arm 1) or widening it (arm 2).
    Second correct implementation: filtering the values before the table is built instead of inside it.
    """
    fragment = f"python -m hhemt --user {LOGIN} && {PYTHON} -m hhemt"
    table = _table(**{P.TOKEN_PYTHON: ["python"]})  # arm 1: the bare word is the row's value
    out = table.apply(fragment.encode()).decode()
    assert out.startswith("python -m hhemt "), "the bare word python must stay"
    assert f"--user {P.TOKEN_USER}" in out, "arm 2: the user row is not subject to the absolute-path guard"
    assert any(s.token == P.TOKEN_PYTHON and s.reason == "not an absolute path" for s in table.skipped)

    # The satisfying input for arm 1: the absolute interpreter in a sensitivity-shaped Snakefile fragment.
    absolute_table = _table()
    assert absolute_table.apply(fragment.encode()).decode().endswith(f"{P.TOKEN_PYTHON} -m hhemt")


@pytest.mark.parametrize("value", ["////", "/", "C:", "relative/dir", "python"])
def test_a_path_row_value_that_is_not_absolute_is_skipped(value):
    """Property: values the absolute-path rule rejects are not substituted in a path-valued row.

    Class: NEW CAPABILITY - plausible wrong implementation accepts any string of four or more characters
        (``////`` is a default report hatch value and would rewrite every occurrence of itself).
    Reachability: a config leaf of that shape reaching a table value.
    Both-states anchor: the text is unchanged, a property of both trees' output for these inputs.
    What kills it: a floor-only guard.
    Second correct implementation: ``pathlib.PurePath.is_absolute`` plus a non-separator check.
    """
    table = _table(**{P.TOKEN_HOME: [value]})
    probe = f"x {value} y"
    assert table.apply(probe.encode()).decode() == probe


def test_a_value_shorter_than_four_characters_is_skipped_and_logged_and_a_four_character_one_is_not(caplog):
    """Property: a three-character login name is not substituted and is logged by token and length; a
        four-character one is substituted.

    Class: NEW CAPABILITY - plausible wrong implementation has no floor (rewrites the name inside words) or
        a floor of five (does not substitute the four-character name).
    Reachability: a short login name or account.
    Both-states anchor: the pair of arms differ only in the value's length.
    What kills it: no floor, a floor off by one in either direction, no log record.
    Second correct implementation: a ``len(value.encode())`` floor, the same for ASCII values.
    """
    with caplog.at_level(logging.WARNING, logger=P.logger.name):
        short = _table(**{P.TOKEN_USER: ["abc"]})
    assert short.apply(b"/d/abc/x --user abc") == b"/d/abc/x --user abc"
    assert [(s.token, s.length) for s in short.skipped if s.token == P.TOKEN_USER] == [(P.TOKEN_USER, 3)]
    assert any(P.TOKEN_USER in r.getMessage() and "3 character" in r.getMessage() for r in caplog.records)
    assert "abc" not in "".join(r.getMessage() for r in caplog.records).replace("character", ""), (
        "the value itself must not be logged"
    )

    four = _table(**{P.TOKEN_USER: ["abcd"]})
    assert four.apply(b"/d/abcd/x") == f"/d/{P.TOKEN_USER}/x".encode()


# --------------------------------------------------------------------------------------
# Building the table from an analysis and the environment
# --------------------------------------------------------------------------------------


def _analysis(*, python_executable=None, hpc_account=None, with_system=True) -> SimpleNamespace:
    ns = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=Path(ANALYSIS_DIR)))
    if with_system:
        ns._system = SimpleNamespace(cfg_system=SimpleNamespace(system_directory=Path(SYSTEM_DIR)))
    if python_executable is not None:
        ns._python_executable = python_executable
    if hpc_account is not None:
        ns.cfg_hpc_system = SimpleNamespace(default_account=hpc_account)
    return ns


def test_build_resolves_python_like_the_base_builder_and_never_takes_a_bare_word(monkeypatch):
    """Property: the interpreter row is the configured interpreter when absolute, else ``sys.executable``.

    Class: NEW CAPABILITY - plausible wrong implementation reads the sensitivity builder's raw ``python``
        and makes the bare word a table value; red under it (the bare word would be in the table).
    Reachability: ``analysis._python_executable`` is ``python`` for every analysis today.
    Both-states anchor: the assertion is on the table's accepted value, with the configured-absolute case as
        the differently positioned satisfying input.
    What kills it: reading the configured value without the absolute check.
    Second correct implementation: delegating to the base builder's ``python_executable``.
    """
    monkeypatch.setattr(sys, "executable", PYTHON)
    for configured in ("python", "python3"):
        table = P.build_placeholder_table(_analysis(python_executable=configured))
        assert table.apply(f"{PYTHON} -m x python -m y".encode()) == f"{P.TOKEN_PYTHON} -m x python -m y".encode()
    other = HOME + "/alt/bin/python"
    table = P.build_placeholder_table(_analysis(python_executable=other))
    # The configured absolute interpreter is the row's value; the process interpreter is then just a path.
    assert table.apply(f"{other} {PYTHON}".encode()) == f"{P.TOKEN_PYTHON} {PYTHON}".encode()


def test_build_reads_directories_account_home_and_login_and_tolerates_a_partial_analysis(monkeypatch):
    """Property: each row is filled from its source, and an analysis lacking a source leaves that row empty
        instead of raising.

    Class: NEW CAPABILITY - plausible wrong implementation indexes ``analysis._system.cfg_system`` directly and
        raises ``AttributeError`` on the stand-ins the existing emit tests pass.
    Reachability: ``tests/test_bundle_stale_figures.py`` passes an analysis with no ``_system`` into
        ``emit_bundle``.
    Both-states anchor: the full analysis and the partial one are the two positions.
    What kills it: unguarded attribute access.
    Second correct implementation: ``try/except AttributeError`` around each source.
    """
    monkeypatch.setattr(sys, "executable", PYTHON)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(HOME)))
    monkeypatch.setattr(P.getpass, "getuser", lambda: LOGIN)
    full = P.build_placeholder_table(_analysis(hpc_account=ACCOUNT))
    out = full.apply(f"{ANALYSIS_DIR} {SYSTEM_DIR} {HOME} --user {LOGIN} --account={ACCOUNT}".encode()).decode()
    assert out == (
        f"{P.TOKEN_ANALYSIS_DIR} {P.TOKEN_SYSTEM_DIR} {P.TOKEN_HOME} --user {P.TOKEN_USER} "
        f"--account={P.TOKEN_SLURM_ACCOUNT}"
    )
    partial = P.build_placeholder_table(SimpleNamespace())
    # The directory rows are empty (nothing to take them from); the environment rows still work.
    rewritten = partial.apply(f"{SYSTEM_DIR}/x {HOME}/y".encode())
    assert P.TOKEN_SYSTEM_DIR.encode() not in rewritten and P.TOKEN_ANALYSIS_DIR.encode() not in rewritten
    assert P.TOKEN_HOME.encode() in rewritten


# --------------------------------------------------------------------------------------
# The zip writer
# --------------------------------------------------------------------------------------


def _staging_and_stream(tmp_path: Path, stream_count: int = 0):
    """A staging tree with one text member per interesting class and an optional streamed zarr store."""
    staging = tmp_path / "staging"
    text = f"dir {ANALYSIS_DIR}\nhome {HOME}\n"
    _put(staging / "Snakefile.source", text)
    _put(staging / "external" / "weather.csv", text)  # the declared input
    _put(staging / "binary.bin", b"\x00\x01" + text.encode())  # NUL in the prefix: binary
    streamed: dict[str, Path] = {}
    src_root = tmp_path / "src"
    zarr_dir = src_root / "store.zarr"
    for i in range(stream_count):
        # chunk-named members whose bytes are plain text holding a value: classed binary by NAME.
        streamed[f"store.zarr/c/{i}"] = _put(zarr_dir / "c" / str(i), text)
    streamed["store.zarr/zarr.json"] = _put(
        zarr_dir / "zarr.json", json.dumps({"attributes": {"source_file": ANALYSIS_DIR + "/in.nc"}})
    )
    streamed["inputs/model.inp"] = _put(src_root / "model.inp", text)
    return staging, streamed, text


def test_text_members_are_rewritten_and_declared_and_binary_members_are_not(tmp_path):
    """Property: staged and streamed TEXT members are rewritten; a declared input and a NUL-prefixed member
        keep their bytes.

    Class: REGRESSION - red on the parent, which has neither the keyword nor the module.
    Reachability: the census members: ``Snakefile.source`` (staged text), ``sims/.../hydraulics.inp``
        (streamed text), ``external/*`` (declared inputs).
    Both-states anchor: the declared member and the rewritten member hold IDENTICAL bytes at different archive
        names, so the only thing that differs is the exemption; the assertion turns on that exemption.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments.
    What kills it: rewriting every member, rewriting none, a sniff that ignores the prefix.
    Second correct implementation: the exemption applied by comparing resolved source paths to the declared
        sources instead of by archive name.
    """
    staging, streamed, text = _staging_and_stream(tmp_path)
    out = tmp_path / "b.zip"
    _emit_bundle_zip(
        staging,
        out,
        streamed=streamed,
        placeholders=_table(),
        declared_arcnames=frozenset({"external/weather.csv"}),
    )
    got = _read_zip(out)
    expected = f"dir {P.TOKEN_ANALYSIS_DIR}\nhome {P.TOKEN_HOME}\n".encode()
    assert got["Snakefile.source"] == expected
    assert got["inputs/model.inp"] == expected
    assert got["external/weather.csv"] == text.encode(), "a declared input is never rewritten"
    assert got["binary.bin"] == b"\x00\x01" + text.encode(), "a member with a NUL in its prefix is binary"


def test_an_emit_without_a_table_is_the_unchanged_emitter(tmp_path):
    """Property: with no table, ``_emit_bundle_zip`` copies every member byte for byte (the path ``_combine`` uses).

    Class: REGRESSION - holds before and after; it pins that the new keyword defaults leave callers alone.
    Reachability: ``_combine`` and two existing tests call the function without the keyword.
    Both-states anchor: the output equals the input in both trees.
    What kills it: defaulting the table to a built one.
    Second correct implementation: a separate legacy function for table-less callers.
    """
    staging, streamed, text = _staging_and_stream(tmp_path)
    out = tmp_path / "b.zip"
    _emit_bundle_zip(staging, out, streamed=streamed)
    got = _read_zip(out)
    assert got["Snakefile.source"] == text.encode()
    assert got["inputs/model.inp"] == text.encode()


def test_each_member_is_opened_once_and_sizes_and_bytes_are_deterministic(tmp_path, monkeypatch):
    """Property: over a store of 1,000 chunk-named files plus text members, every archive member is opened
        exactly once; a text member's entry size equals its rewritten length; two emits give one sha256;
        chunk-named members keep their bytes though those bytes are plain text holding a value.

    Class: NEW CAPABILITY - plausible wrong implementation sniffs by opening each member a second time (a
        ``_is_text(path)`` call then a copy) and reds the count; a second wrong implementation sniffs every
        member including chunks and reds the chunk-bytes arm.
    Reachability: the real harvest hands this function hundreds of thousands of chunk files.
    Both-states anchor: the count is a property of how members are read in both trees (one open each).
    Invocation isolated: the counter wraps ``Path.open`` for the paths of THIS test's members only.
    What kills it: a second open; reading the chunk's prefix.
    Not established: the entry-size assertion does not discriminate a rewritten member whose ``file_size`` was
        left at the ``stat`` value, because ``zipfile`` rewrites the header from the bytes it wrote when the
        output is seekable; it pins the visible result, not the assignment.
    Second correct implementation: ``os.open`` / ``io.open`` instead of ``Path.open`` for the single read.
    """
    staging, streamed, text = _staging_and_stream(tmp_path, stream_count=1000)
    members = {p for p in streamed.values()} | {p for p in staging.rglob("*") if p.is_file()}
    counts: dict[Path, int] = {}
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        if self in members:
            counts[self] = counts.get(self, 0) + 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    declared = frozenset({"external/weather.csv"})
    out_a, out_b = tmp_path / "a.zip", tmp_path / "b.zip"
    _emit_bundle_zip(staging, out_a, streamed=streamed, placeholders=_table(), declared_arcnames=declared)
    first_counts = dict(counts)
    counts.clear()
    _emit_bundle_zip(staging, out_b, streamed=streamed, placeholders=_table(), declared_arcnames=declared)
    monkeypatch.setattr(Path, "open", real_open)

    assert set(first_counts) == members and set(first_counts.values()) == {1}, "unit: opens per archive member"
    assert counts == first_counts
    assert hashlib.sha256(out_a.read_bytes()).hexdigest() == hashlib.sha256(out_b.read_bytes()).hexdigest()

    rewritten = f"dir {P.TOKEN_ANALYSIS_DIR}\nhome {P.TOKEN_HOME}\n".encode()
    with zipfile.ZipFile(out_a) as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            assert info.file_size == len(data)
            if info.filename.startswith("store.zarr/c/"):
                assert data == text.encode(), "a chunk is classed binary by name and keeps its bytes"
        assert zf.read("Snakefile.source") == rewritten
        assert info.file_size == len(zf.read(info.filename))


def test_a_sensitivity_zarr_json_source_file_is_rewritten_and_still_parses(tmp_path):
    """Property: the ``attributes.source_file`` of a zarr metadata document is rewritten, and the document is
        still valid JSON whose value begins with the analysis token.

    Class: REGRESSION - red on the parent (no rewrite); output pasted in the round file.
    Reachability: 24 ``zarr.json`` members of the sensitivity census carried a producer path in that key.
    Both-states anchor: the assertion parses the member and reads the key, which are properties of the
        document in both trees.
    What kills it: classing the metadata document as a chunk because it sits under a ``.zarr`` directory.
    Second correct implementation: rewriting the JSON through ``json.loads`` / ``json.dumps`` instead of bytes.
    """
    staging, streamed, _text = _staging_and_stream(tmp_path)
    out = tmp_path / "b.zip"
    _emit_bundle_zip(staging, out, streamed=streamed, placeholders=_table())
    doc = json.loads(_read_zip(out)["store.zarr/zarr.json"])
    assert doc["attributes"]["source_file"] == P.TOKEN_ANALYSIS_DIR + "/in.nc"


def test_the_byte_prefix_predicate_agrees_with_the_scan_gate_on_the_same_files(tmp_path):
    """Property: classifying from a prefix already read gives the same answer as ``_reprex_gate._is_text``.

    Class: NEW CAPABILITY, a fitness function - the emit's classification and the scan's must not drift.
    Reachability: a NUL as the 8192nd and 8193rd byte straddles the sniff boundary.
    Both-states anchor: both predicates are applied to the same files.
    What kills it: a different prefix length or a different NUL rule in either predicate.
    Second correct implementation: ``_is_text`` itself made the single definition, taking bytes.
    """
    cases = {
        "text": b"plain",
        "nul_first": b"\x00abc",
        "nul_last_sniffed": b"a" * (_SNIFF_BYTES - 1) + b"\x00",
        "nul_just_beyond": b"a" * _SNIFF_BYTES + b"\x00",
        "empty": b"",
    }
    for name, data in cases.items():
        path = _put(tmp_path / name, data)
        assert P.prefix_is_text(data[:_SNIFF_BYTES]) == _is_text(path), name


# --------------------------------------------------------------------------------------
# The baseline: html only
# --------------------------------------------------------------------------------------


def test_the_baseline_carries_the_html_page_and_not_the_zip(tmp_path):
    """Property: ``_copy_bundle_baseline`` copies ``analysis_report.html`` and does not copy the zip.

    Class: REGRESSION - red on the parent, which copies both.
    Reachability: both files sit beside each other in a rendered analysis directory.
    Both-states anchor: the assertion is on which files exist in the baseline directory.
    Invocation isolated: pure function of two directories.
    What kills it: iterating over both suffixes.
    Second correct implementation: copying the html and deleting a copied zip afterwards.
    """
    analysis_dir, staging = tmp_path / "a", tmp_path / "s"
    _put(analysis_dir / "analysis_report.html", "<html/>")
    _put(analysis_dir / "analysis_report.zip", b"PK")
    _copy_bundle_baseline(analysis_dir, staging)
    assert sorted(p.name for p in (staging / "bundle_baseline").iterdir()) == ["analysis_report.html"]

    only_zip = tmp_path / "b"
    _put(only_zip / "analysis_report.zip", b"PK")
    _copy_bundle_baseline(only_zip, tmp_path / "t")
    assert list((tmp_path / "t" / "bundle_baseline").iterdir()) == [], "a zip alone is not carried either"


def test_the_emit_docstring_and_the_bundle_help_name_the_html_baseline_only():
    """Property: the text that describes the bundle's baseline no longer says the zip is carried.

    Class: REGRESSION - red on the parent, whose docstring and help read ``analysis_report.{html,zip}``.
    Reachability: both texts are what a user reads before running ``hhemt bundle``.
    Both-states anchor: asserts the false claim's ABSENCE, which is a property of the parent's text.
    What kills it: leaving either text unchanged.
    Second correct implementation: any wording that names the html baseline and says the zip is not carried.
    """
    from hhemt import cli

    for doc in (_emit.emit_bundle.__doc__, cli.bundle_command.__doc__):
        assert "analysis_report.{html,zip}" not in doc
        assert "analysis_report.html" in doc


# --------------------------------------------------------------------------------------
# Wiring: the real emit_bundle
# --------------------------------------------------------------------------------------


def test_emit_bundle_applies_the_table_and_exempts_exactly_the_declared_inputs(tmp_path, monkeypatch):
    """Property: through the REAL ``emit_bundle`` over a temporary tree, a generated text member carries the
        analysis-directory token, a producer-free declared input is unchanged, a declared input holding the
        analysis directory refuses the emit instead of being rewritten, and the baseline holds the html page
        and no zip.

    Class: REGRESSION - red on the parent (the value reaches the archive and the zip is present).
    Reachability: ``emit_bundle`` is the only caller that builds the table and the declared set.
    Both-states anchor: in the second state the declared input and the generated member hold the same value;
        only the exemption separates a refusal from a token.
    Invocation isolated: no solver, no fixture; configs are real models over ``tmp_path``.
    What kills it: building the table but not passing it; passing an empty declared set; the exemption keyed on
        something other than the archive name.
    Second correct implementation: applying the table at staging time to generated files only.
    """
    from tests.test_publish_deposit_paths import _put as _put_world
    from tests.test_publish_deposit_paths import _run_emit, _world

    world = _world(tmp_path, analysis_over=None)
    weather = _put_world(world.out / "weather.csv", "forcing series\n")
    world.analysis.cfg_analysis = world.analysis.cfg_analysis.model_copy(update={"weather_timeseries": weather})
    _put_world(world.ad / "Snakefile", f"cd {world.ad}\n")
    _put_world(world.ad / "analysis_report.html", f"<p>{world.ad}</p>")
    _put_world(world.ad / "analysis_report.zip", b"PK\x03\x04")
    monkeypatch.setattr(_emit, "_get_toolkit_git_sha", lambda strict=True: "deadbeef0000")

    zpath = _run_emit(world)
    names = _read_zip(zpath)
    assert names["Snakefile.source"] == f"cd {P.TOKEN_ANALYSIS_DIR}\n".encode()
    assert names["bundle_baseline/analysis_report.html"] == f"<p>{P.TOKEN_ANALYSIS_DIR}</p>".encode()
    assert "bundle_baseline/analysis_report.zip" not in names
    declared = [n for n in names if n.endswith("weather.csv")]
    assert declared and names[declared[0]] == b"forcing series\n"
    # Declared inputs are never rewritten, so one that holds a producer value reaches the archive
    # unchanged and the zero-user-info check refuses the emit, naming it.
    from hhemt.exceptions import ProcessingError

    weather.write_text(f"forcing for {world.ad}\n")
    with pytest.raises(ProcessingError, match="external/weather.csv"):
        _run_emit(world)
    shutil.rmtree(world.base, ignore_errors=True)


def _fake_source_checkout(root: Path, monkeypatch) -> tuple[Path, Path]:
    """Point ``hhemt.__file__`` at ``{root}/checkout/src/hhemt`` and return (checkout root, package directory)."""
    import getpass

    import hhemt

    checkout = (root / "checkout").resolve()
    package = checkout / "src" / "hhemt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(hhemt, "__file__", str(package / "__init__.py"))
    # A login name under four characters is never substituted, so the temp path cannot be rewritten by it.
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    return checkout, package


def test_package_and_checkout_paths_take_separate_tokens_on_a_source_checkout(tmp_path, monkeypatch):
    """Property: on a source checkout, a path under the package directory is written ``{hhemt_package_dir}/...``,
    the checkout's conda environment file keeps ``{hhemt_conda_env}``, and any other checkout path is written
    ``{hhemt_toolkit_dir}/...``.

    Class: REGRESSION - red on the parent, where the package path is written ``{hhemt_toolkit_dir}/...`` and
        no ``{hhemt_package_dir}`` exists.
    Why required: with one token for two directories, ``{checkout}/src/hhemt/x`` and ``{checkout}/x`` could not
        be told apart, so a consumer that rehydrates gets the wrong file for one of them.
    Reachability: the real ``build_placeholder_table`` over a package location monkeypatched to a ``src/``
        layout; the three lines name one path of each kind.
    Both-states anchor: the assertion turns on which token the longer directory takes, a property of the table's
        contents and not of wording this change introduces.
    What kills it: a table whose toolkit row still lists the package directory, or one that lists the checkout
        root under the package token.
    Second correct implementation: a table that builds the two rows from one helper returning
        (package, checkout) pairs, with the same rewritten text.
    """
    checkout, package = _fake_source_checkout(tmp_path, monkeypatch)
    table = P.build_placeholder_table(SimpleNamespace())
    text = f"{package}/report_templates/x\n{checkout}/workflow/envs/hhemt.yaml\n{checkout}/tests/fixtures/h.yaml\n"
    assert table.apply(text.encode()).decode() == (
        f"{P.TOKEN_PACKAGE_DIR}/report_templates/x\n{P.TOKEN_CONDA_ENV}\n{P.TOKEN_TOOLKIT_DIR}/tests/fixtures/h.yaml\n"
    )


def test_package_path_is_a_path_valued_row_so_a_dot_or_hyphen_neighbour_is_handled_as_a_path(tmp_path):
    """Property: ``{hhemt_package_dir}`` is a path-valued row: a package path immediately followed by ``.`` (a
    ``.bak`` sibling) or preceded by ``-`` is rewritten, and a non-absolute package value is never substituted.

    Class: NEW CAPABILITY - the plausible wrong implementation leaves the token out of the path-valued set; the
        token then takes the name boundary rule, which leaves both neighbours unrewritten and admits a
        non-absolute value. Red under it (checked by removing the token from the set).
    Why required: a token missing from the set silently changes three behaviours at once, and the other
        tests only use the token on a plain ``/``-terminated path.
    Reachability: the path value is built from fragments; the two neighbour strings and the relative value are
        each reached by one assertion.
    Both-states anchor: the assertions turn on the neighbouring characters, not on any token spelling.
    What kills it: dropping ``TOKEN_PACKAGE_DIR`` from ``_PATH_VALUED_TOKENS``.
    Second correct implementation: a per-row ``kind`` attribute instead of a set, with the same outcomes.
    """
    package = TOOLKIT + "/src/hhemt"
    assert P.TOKEN_PACKAGE_DIR in P._PATH_VALUED_TOKENS
    table = P.PlaceholderTable.from_values({P.TOKEN_PACKAGE_DIR: [package]})
    assert table.apply(f"{package}.bak/x".encode()).decode() == f"{P.TOKEN_PACKAGE_DIR}.bak/x"
    assert table.apply(f"a-{package}/x".encode()).decode() == f"a-{P.TOKEN_PACKAGE_DIR}/x"
    relative = P.PlaceholderTable.from_values({P.TOKEN_PACKAGE_DIR: ["relative/pkg_dir"]})
    assert relative.is_empty and [s.reason for s in relative.skipped] == ["not an absolute path"]

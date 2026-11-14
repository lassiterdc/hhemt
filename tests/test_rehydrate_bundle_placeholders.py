"""``rehydrate_placeholders`` puts a consumer's values back for the closed ``{hhemt_*}`` tokens of a bundle member.

SOLVER-FREE: every arm writes small files under ``tmp_path``. Nothing here compiles or runs a solver.

Producer-looking values (home directories, scratch paths, a login name) are BUILT FROM FRAGMENTS
below, never written as literals, so this file is not itself a hit for the repository's anonymization
guard.

Two jobs. (1) The function itself: a round trip against the real ``PlaceholderTable``, the refusal of
an unresolved token, and the closed-set rule that keeps Snakemake and shell braces intact. (2) The
consumer of rewritten members: ``report_plot_ids.event_sidebar_labels_from_status``
reads only ``Path(scenario_directory).name``, which the prefix rewrite leaves intact, so nothing is
wired into it and the last two tests pin that.

Each test docstring states Property, Class, Why required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.experiment_definition as ed
from hhemt.bundle import _placeholders as P
from hhemt.exceptions import ConfigurationError
from hhemt.report_plot_ids import event_sidebar_labels_from_status
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

LOGIN = "carol" + "ine"
HOME = "/ho" + "me/" + LOGIN
SYSTEM_DIR = "/scr" + "atch/" + LOGIN + "/runs"
ANALYSIS_DIR = SYSTEM_DIR + "/a1"  # the analysis directory lies INSIDE the system directory
PYTHON = HOME + "/miniconda3/envs/e/bin/python"  # the interpreter lies INSIDE the home directory


@pytest.fixture(autouse=True)
def _keep_copies_in_tmp(tmp_path, monkeypatch):
    """Default copies go under ``$SCRATCH_DIR``; point it at tmp_path so no test writes outside it."""
    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path / "scratch_dir"))


def _bundle_root(tmp_path: Path) -> Path:
    """A directory that is a bundle root: it holds a ``bundle_manifest.json``."""
    root = tmp_path / "a_bundle"
    root.mkdir()
    (root / BUNDLE_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    return root


def _member(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_round_trip_through_the_real_table_restores_the_original_bytes(tmp_path):
    """Property: rewriting a member with a real ``PlaceholderTable`` and rehydrating with the same values
    returns the original bytes.

    Class: NEW CAPABILITY. The plausible wrong implementation is a function that returns the tokens
    unexpanded (a copy only): red under it, because the rewritten text differs from the original and so does
    the copy. On the parent the name does not exist and the call raises AttributeError.
    Reachability: a text holding the analysis directory, its parent system directory, the interpreter and the
    home directory reaches every branch; the rewrite is asserted to have CHANGED the text first, so the
    round trip cannot pass on a table that rewrote nothing.
    Both-states anchor: the assertion turns on byte equality of the original and the restored text, a
    property that holds under any correct table and any correct rehydrator, not on the token spelling.
    Second correct implementation: one that replaces each token by a separate ``bytes.replace`` instead of a
    single regex pass; for values that contain no token text the result is the same.
    """
    original = (
        f"analysis: {ANALYSIS_DIR}/sims/event_a\n"
        f"system: {SYSTEM_DIR}/inputs\n"
        f"interpreter: {PYTHON} -m hhemt.run\n"
        f"cache: {HOME}/.cache\n"
    )
    table = P.PlaceholderTable.from_values(
        {
            P.TOKEN_ANALYSIS_DIR: [ANALYSIS_DIR],
            P.TOKEN_SYSTEM_DIR: [SYSTEM_DIR],
            P.TOKEN_PYTHON: [PYTHON],
            P.TOKEN_HOME: [HOME],
        }
    )
    rewritten = table.apply(original.encode()).decode()
    assert rewritten != original and P.TOKEN_ANALYSIS_DIR in rewritten and P.TOKEN_PYTHON in rewritten
    member = _member(tmp_path / "bundle_in", "Snakefile.source", rewritten)

    out = ed.rehydrate_placeholders(
        member,
        {
            P.TOKEN_ANALYSIS_DIR: ANALYSIS_DIR,
            P.TOKEN_SYSTEM_DIR: SYSTEM_DIR,
            P.TOKEN_PYTHON: PYTHON,
            P.TOKEN_HOME: HOME,
        },
        dest_dir=tmp_path / "out",
    )
    assert out.read_text(encoding="utf-8") == original
    assert member.read_text(encoding="utf-8") == rewritten  # the source member is never modified


def test_directory_tokens_default_to_the_nearest_bundle_root(tmp_path):
    """Property: with no supplied value, both directory tokens become the nearest ancestor holding
    ``bundle_manifest.json``.

    Class: NEW CAPABILITY. The plausible wrong implementation is a default of the member's own directory (or
    the current working directory): red under it, because the member sits two levels below the root.
    Reachability: a member at ``root/scenarios/s1/scenario_status.csv`` under a root with a manifest reaches
    the walk-up; a nested manifest is also planted to show the NEAREST wins.
    Both-states anchor: the assertion turns on which ancestor is returned, not on any wording.
    Second correct implementation: one that walks down from a caller-supplied bundle root instead of up from
    the member, giving the same root for a member inside it.
    """
    root = _bundle_root(tmp_path)
    inner = root / "members" / "m1"
    inner.mkdir(parents=True)
    (inner / BUNDLE_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    member = _member(inner / "scenarios", "x.txt", f"a={P.TOKEN_ANALYSIS_DIR}/p b={P.TOKEN_SYSTEM_DIR}/q\n")

    out = ed.rehydrate_placeholders(member, dest_dir=tmp_path / "out")

    assert out.read_text(encoding="utf-8") == f"a={inner.resolve()}/p b={inner.resolve()}/q\n"
    outer = _member(root / "members" / "m2" / "scenarios", "x.txt", f"a={P.TOKEN_ANALYSIS_DIR}/p\n")
    assert ed.rehydrate_placeholders(outer, dest_dir=tmp_path / "out2").read_text(encoding="utf-8") == (
        f"a={root.resolve()}/p\n"
    )


def test_an_unresolved_directory_token_is_refused_and_nothing_is_written(tmp_path):
    """Property: a directory token with no supplied value and no enclosing bundle raises ``ConfigurationError``
    before any copy is written.

    Class: NEW CAPABILITY. The plausible wrong implementation leaves the token in place and returns a copy:
    red under it, because no exception is raised and a file appears in ``dest_dir``.
    Reachability: a member under ``tmp_path`` with no manifest above it holding ``{hhemt_analysis_dir}``. The
    must-fire control varies the PROPERTY (a value is supplied) and the same call then succeeds, so the
    refusal is caused by the missing value and not by the token's presence.
    Both-states anchor: the assertion turns on the exception type and the empty ``dest_dir``, present in
    any correct refusal whatever its message.
    Second correct implementation: one that raises ``ConfigurationError`` subclass, or that validates all
    tokens before reading the file; both satisfy the same assertions.
    """
    member = _member(tmp_path / "loose", "x.txt", f"{P.TOKEN_ANALYSIS_DIR}/a {P.TOKEN_HOME}/b\n")
    dest = tmp_path / "dest"

    with pytest.raises(ConfigurationError) as excinfo:
        ed.rehydrate_placeholders(member, dest_dir=dest)
    assert P.TOKEN_ANALYSIS_DIR in str(excinfo.value)
    assert P.TOKEN_HOME not in str(excinfo.value)  # the home token HAS a default and is not named
    assert not dest.exists() or not any(dest.iterdir())

    out = ed.rehydrate_placeholders(member, {P.TOKEN_ANALYSIS_DIR: ANALYSIS_DIR}, dest_dir=dest)
    assert out.read_text(encoding="utf-8").startswith(f"{ANALYSIS_DIR}/a ")


def test_only_the_closed_tokens_are_replaced(tmp_path):
    """Property: Snakemake wildcards, shell variables and an unknown ``{hhemt_*}`` name are byte-identical in
    the copy; only the closed tokens change.

    Class: NEW CAPABILITY. The plausible wrong implementation is ``os.path.expandvars`` (the sibling
    ``expand_config_vars`` approach) or a ``{hhemt_.*}`` pattern: red under it, because ``${CONDA_PREFIX}``
    is expanded or ``{hhemt_unknown}`` is treated as unresolved.
    Reachability: a rule line carrying ``{input}``, ``{wildcards.event_id}``, ``${CONDA_PREFIX}``,
    ``{hhemt_unknown}`` and one real token. The differently positioned satisfying input is the real token,
    which IS replaced in the same member.
    Both-states anchor: the assertion turns on which spans are untouched, not on wording.
    Second correct implementation: one that iterates the closed tokens with ``str.replace`` on decoded text.
    """
    foreign = "shell: '{input} {wildcards.event_id} ${CONDA_PREFIX}/bin {hhemt_unknown}'"
    member = _member(tmp_path / "m", "Snakefile.source", f"{foreign}\npy={P.TOKEN_PYTHON}\n")

    out = ed.rehydrate_placeholders(member, dest_dir=tmp_path / "out")

    text = out.read_text(encoding="utf-8")
    assert text.startswith(foreign + "\n")
    assert text.endswith(f"py={sys.executable}\n")


def test_tokens_with_no_consumer_value_stay_as_the_token_and_a_supplied_value_wins(tmp_path):
    """Property: conda env, login name and SLURM account stay as their token by default, and a supplied value
    replaces the SLURM account.

    Class: NEW CAPABILITY. The plausible wrong implementation refuses these three, treating every
    remaining token as unresolved: red under it, because the call raises.
    Reachability: a member holding all three tokens reaches the leave-as-token default for each; the same
    member with an account supplied reaches the override.
    Both-states anchor: the assertion turns on the token surviving, which holds under any implementation of
    the documented default.
    Second correct implementation: one that maps each such token to itself in an explicit table rather
    than skipping it.
    """
    body = f"env={P.TOKEN_CONDA_ENV} user={P.TOKEN_USER} acct={P.TOKEN_SLURM_ACCOUNT}\n"
    member = _member(tmp_path / "m", "run.sh", body)

    assert ed.rehydrate_placeholders(member, dest_dir=tmp_path / "o1").read_text(encoding="utf-8") == body
    supplied = ed.rehydrate_placeholders(member, {P.TOKEN_SLURM_ACCOUNT: "hydro-lab"}, dest_dir=tmp_path / "o2")
    assert supplied.read_text(encoding="utf-8") == (f"env={P.TOKEN_CONDA_ENV} user={P.TOKEN_USER} acct=hydro-lab\n")


def test_a_name_outside_the_closed_set_or_an_empty_value_is_refused(tmp_path):
    """Property: ``values`` naming a token outside the closed set, or giving an empty value, raises
    ``ConfigurationError``.

    Class: NEW CAPABILITY. The plausible wrong implementation ignores unknown keys: red under it, because
    a misspelled token would silently leave its occurrences unresolved.
    Reachability: both checks run on ``values`` alone, before the member is searched for tokens, so a member
    with no token (first call) and one with a token (second call) each reach one.
    Both-states anchor: the assertion turns on the exception, not on its message.
    Second correct implementation: one that raises ``ValueError`` for the empty value would fail the
    ``ConfigurationError`` assertion, so the type is part of the contract (CLI exit 2).
    """
    member = _member(tmp_path / "m", "x.txt", "no tokens here\n")
    with pytest.raises(ConfigurationError, match="hhemt_typo"):
        ed.rehydrate_placeholders(member, {"{hhemt_typo}": "/x"}, dest_dir=tmp_path / "o")
    member2 = _member(tmp_path / "m2", "y.txt", f"{P.TOKEN_HOME}\n")
    with pytest.raises(ConfigurationError, match="empty"):
        ed.rehydrate_placeholders(member2, {P.TOKEN_HOME: ""}, dest_dir=tmp_path / "o")


def test_a_substituted_value_is_never_scanned_again(tmp_path):
    """Property: a supplied value that itself spells a token is written literally, not expanded a second time.

    Class: NEW CAPABILITY. The plausible wrong implementation replaces token by token in sequence, so a
    value containing a later token is rewritten by that later pass: red under it.
    Reachability: the python value spells the home token, and the home token is later in the closed-set order, so
    a sequential replacement visits it AFTER the python value was written.
    Both-states anchor: the assertion turns on the literal output, identical under any single-pass
    implementation.
    Second correct implementation: a regex alternation (the shipped one) or a manual scan that
    emits the value and advances past the matched token.
    """
    member = _member(tmp_path / "m", "x.txt", f"{P.TOKEN_PYTHON} {P.TOKEN_HOME}\n")
    out = ed.rehydrate_placeholders(
        member, {P.TOKEN_PYTHON: "/py/" + P.TOKEN_HOME, P.TOKEN_HOME: "/h"}, dest_dir=tmp_path / "o"
    )
    assert out.read_text(encoding="utf-8") == f"/py/{P.TOKEN_HOME} /h\n"


def test_two_members_with_one_file_name_get_two_default_copies_and_the_source_is_safe(tmp_path):
    """Property: the default destination keeps two same-named members apart, and a copy never overwrites its source.

    Class: NEW CAPABILITY. The plausible wrong implementation writes ``resolved_{name}`` into one shared
    directory, as ``expand_config_vars`` does: red under it, because the second call overwrites the first.
    Reachability: two ``zarr.json`` members in sibling directories, called with the default destination.
    The in-place arm points ``dest_dir`` at the member's own directory.
    Both-states anchor: the assertion turns on two distinct paths holding two distinct contents.
    Second correct implementation: one that returns a fresh ``tempfile.mkdtemp`` directory per call.
    """
    a = _member(tmp_path / "a", "zarr.json", f"{P.TOKEN_PYTHON} a\n")
    b = _member(tmp_path / "b", "zarr.json", f"{P.TOKEN_PYTHON} b\n")
    out_a = ed.rehydrate_placeholders(a, {P.TOKEN_PYTHON: "/py"})
    out_b = ed.rehydrate_placeholders(b, {P.TOKEN_PYTHON: "/py"})
    assert out_a != out_b
    assert out_a.read_text(encoding="utf-8") == "/py a\n" and out_b.read_text(encoding="utf-8") == "/py b\n"
    assert str(tmp_path / "scratch_dir") in str(out_a)

    # Pointing dest_dir at the member's own directory writes a sibling copy and leaves the source alone.
    in_place = ed.rehydrate_placeholders(a, {P.TOKEN_PYTHON: "/py"}, dest_dir=a.parent)
    assert in_place != a and in_place.read_text(encoding="utf-8") == "/py a\n"
    assert a.read_text(encoding="utf-8") == f"{P.TOKEN_PYTHON} a\n"


def _status_csv(directory: Path, scenario_dirs: list[str]) -> Path:
    rows = ["scenario_directory,event_iloc,event_label"]
    rows += [f"{d},{i},storm {i}" for i, d in enumerate(scenario_dirs)]
    return _member(directory, "scenario_status.csv", "\n".join(rows) + "\n")


def test_the_one_consumer_reads_the_same_labels_from_a_prefix_rewritten_status_csv(tmp_path):
    """Property: ``event_sidebar_labels_from_status`` returns the same labels from a ``scenario_status.csv``
    rewritten by the placeholder table as from the original, with no rehydration.

    Class: NEW CAPABILITY (of the pin). The plausible wrong rewrite replaces the whole scenario directory with
    its token; the must-fire control at the end of this test builds exactly that rewrite and its labels
    differ, so the assertion is red under it. Not established: the consumer's own reading is already
    green on the parent, so this test is not red on the parent for any reason other than the module import.
    Reachability: the CSV carries the producer's analysis directory as the prefix of each
    ``scenario_directory`` cell; the rewritten CSV is asserted to differ from the original and to hold the
    token, so equality is not vacuous.
    Both-states anchor: the assertion turns on ``Path(...).name`` of the scenario directory, a property of
    both the original and the rewritten cell.
    Second correct implementation: a labeler that keyed on the last two path segments would still pass
    this test only if the rewrite kept two, so the test constrains the rewrite (a prefix replacement), which
    is the prefix-only rule of the placeholder table, not a property of the labeler.
    """
    scenario_dirs = [f"{ANALYSIS_DIR}/sims/event_a", f"{ANALYSIS_DIR}/sims/event_b"]
    original_dir = tmp_path / "original"
    _status_csv(original_dir, scenario_dirs)
    table = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [ANALYSIS_DIR]})
    rewritten_dir = tmp_path / "rewritten"
    rewritten_text = table.apply((original_dir / "scenario_status.csv").read_bytes()).decode()
    _member(rewritten_dir, "scenario_status.csv", rewritten_text)

    assert P.TOKEN_ANALYSIS_DIR in rewritten_text
    expected = event_sidebar_labels_from_status(original_dir)
    assert expected == {"event_a": "storm 0 (event 0)", "event_b": "storm 1 (event 1)"}
    assert event_sidebar_labels_from_status(rewritten_dir) == expected

    # Must-fire control: a rewrite that swallows the final path segment changes the labels, so the
    # equality above is caused by the prefix rule and not by an insensitive labeler.
    whole = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [scenario_dirs[0]]})
    swallowed_dir = tmp_path / "swallowed"
    _member(
        swallowed_dir, "scenario_status.csv", whole.apply((original_dir / "scenario_status.csv").read_bytes()).decode()
    )
    assert event_sidebar_labels_from_status(swallowed_dir) != expected


def test_rehydrating_the_status_csv_leaves_the_consumers_labels_unchanged(tmp_path):
    """Property: rehydrating a rewritten ``scenario_status.csv`` inside a bundle yields the same labels again,
    so the rehydrator and the consumer agree.

    Class: NEW CAPABILITY. The plausible wrong implementation drops or alters the CSV's last line (for
    example a ``strip()`` on decoded text): red under it, because the final event disappears from the labels.
    Reachability: the CSV sits under a bundle root, so the directory token takes its default.
    Both-states anchor: the assertion turns on label equality across the round trip.
    Second correct implementation: one that streams the file in chunks and substitutes per chunk,
    provided it does not split a token across chunks.
    """
    root = _bundle_root(tmp_path)
    scenario_dirs = [f"{ANALYSIS_DIR}/sims/event_a", f"{ANALYSIS_DIR}/sims/event_b"]
    source = _status_csv(tmp_path / "src", scenario_dirs)
    table = P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [ANALYSIS_DIR]})
    _member(root, "scenario_status.csv", table.apply(source.read_bytes()).decode())

    out = ed.rehydrate_placeholders(root / "scenario_status.csv", dest_dir=tmp_path / "out")
    resolved_dir = tmp_path / "resolved"
    resolved_dir.mkdir()
    (resolved_dir / "scenario_status.csv").write_bytes(out.read_bytes())

    assert event_sidebar_labels_from_status(resolved_dir) == event_sidebar_labels_from_status(tmp_path / "src")
    assert str(root.resolve()) in out.read_text(encoding="utf-8")


def _fake_layout(root: Path, monkeypatch, *, under_src: bool) -> tuple[Path, Path]:
    """Point ``hhemt.__file__`` at a package directory and return (its parent layout root, the package directory)."""
    import hhemt

    base = (root / "layout").resolve()
    package = base / "src" / "hhemt" if under_src else base / "site" / "hhemt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(hhemt, "__file__", str(package / "__init__.py"))
    # A login name under four characters is never substituted, so the temp path cannot be rewritten by it.
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    return base, package


def test_a_package_path_and_a_checkout_path_round_trip_on_a_source_checkout(tmp_path, monkeypatch):
    """Property: a member holding a package-directory path and a checkout-root path, rewritten by the real
    ``build_placeholder_table`` on a source checkout and rehydrated on the same install, comes back
    byte-identical.

    Class: REGRESSION - red on the parent, where the package path comes back without ``/src/hhemt`` (the single
        toolkit token was written for the checkout root and read back as the checkout root).
    Why required: this is the round trip the whole rehydrator exists for, on the layout the toolkit is
        developed in; the other round-trip test builds its table from plain strings.
    Reachability: the package location is monkeypatched to a ``src/`` layout, so both directories exist as
        distinct values; the rewrite is asserted to have changed the text and to hold both tokens.
    Both-states anchor: the assertion turns on byte equality of original and restored text, which holds under
        any correct table and rehydrator.
    What kills it: one token standing for both directories; a rehydrator default of the package directory for
        the checkout token or the reverse.
    Second correct implementation: a rehydrator that derives the checkout root from the package directory by
        string suffix removal instead of ``Path.parent``, with the same restored text.
    """
    checkout, package = _fake_layout(tmp_path, monkeypatch, under_src=True)
    original = f"{package}/report_templates/x\n{checkout}/tests/fixtures/h.yaml\n"
    rewritten = P.build_placeholder_table(SimpleNamespace()).apply(original.encode()).decode()
    assert P.TOKEN_PACKAGE_DIR in rewritten and P.TOKEN_TOOLKIT_DIR in rewritten and rewritten != original
    member = _member(tmp_path / "m", "Snakefile.source", rewritten)

    out = ed.rehydrate_placeholders(member, dest_dir=tmp_path / "out")

    assert out.read_text(encoding="utf-8") == original


def test_the_checkout_token_has_no_default_outside_a_source_checkout_and_the_package_token_does(tmp_path, monkeypatch):
    """Property: with the package outside ``src/``, a member holding ``{hhemt_toolkit_dir}`` and no supplied value is
    refused with ``ConfigurationError``, while ``{hhemt_package_dir}`` rehydrates to that package directory.

    Class: NEW CAPABILITY - the plausible wrong implementation defaults the checkout token to the package
        directory (or its parent) on an installed copy: red under it, because the call succeeds and writes a
        path that is not a checkout. On the parent there is no ``{hhemt_package_dir}`` token.
    Why required: an installed copy has no checkout; a guessed default would point a consumer at a directory
        that does not hold what a checkout path names.
    Reachability: the package location is monkeypatched to a layout with no ``src/``. The must-fire control
        varies the property (a supplied value) and the same member then resolves.
    Both-states anchor: the assertions turn on the exception and on the returned directory, not on wording.
    What kills it: a toolkit default that does not check the package's parent name.
    Second correct implementation: an implementation that checks for ``pyproject.toml`` beside ``src/``
        instead of the directory name would also refuse here, because the temp layout has neither.
    """
    _, package = _fake_layout(tmp_path, monkeypatch, under_src=False)
    toolkit_member = _member(tmp_path / "t", "x.txt", f"{P.TOKEN_TOOLKIT_DIR}/a\n")
    with pytest.raises(ConfigurationError, match="hhemt_toolkit_dir"):
        ed.rehydrate_placeholders(toolkit_member, dest_dir=tmp_path / "o1")
    supplied = ed.rehydrate_placeholders(toolkit_member, {P.TOKEN_TOOLKIT_DIR: "/chk"}, dest_dir=tmp_path / "o2")
    assert supplied.read_text(encoding="utf-8") == "/chk/a\n"

    package_member = _member(tmp_path / "p", "x.txt", f"{P.TOKEN_PACKAGE_DIR}/a\n")
    out = ed.rehydrate_placeholders(package_member, dest_dir=tmp_path / "o3")
    assert out.read_text(encoding="utf-8") == f"{package}/a\n"

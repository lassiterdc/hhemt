"""A path under the rewritten login segment becomes one ``{hhemt_producer_path}`` token; layer (b) skips mount roots.

SOLVER-FREE: every arm builds members on ``tmp_path`` and calls the placeholder module, the scanner, the
rehydrator or ``emit_bundle`` over a temporary tree. No fixture here compiles or runs a solver.

Producer-looking paths and addresses are BUILT FROM FRAGMENTS below, never written as literals, so this
file is not itself a hit for the repository's anonymization guard. The login-segment token is taken from
``_placeholders`` rather than typed beside a per-user root.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolation, What
kills it, and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import getpass
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _placeholders as P
from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.exceptions import ConfigurationError
from hhemt.experiment_definition import rehydrate_placeholders
from tests.test_publish_deposit_paths import _put, _world

USER = P.TOKEN_USER
TOKEN = "{hhemt_producer_path}"
LOGIN = "carol" + "ine"  # a login name of at least four characters, so the table rewrites it
SCRATCH_ROOT = "/scr" + "atch"  # a bare mount root and a per-user root at once
CHAIN_ROOT = "../../sfs/weka/scr" + "atch"  # a `../`-relative spelling of the same kind of root
INTERPRETER_PATH = f"{CHAIN_ROOT}/{USER}/b/.venv/bin/python"
PLAIN_PATH = f"{SCRATCH_ROOT}/{USER}/b/m.csv"
URL_WITH_LOGIN = "https://h/" + USER + "/repo"
DRIVE_PATH_WITH_LOGIN = "C:" + "/" + "Users" + "/" + USER + "/a"
TOKEN_PREFIXED_PATH = P.TOKEN_HOME + "/x/" + USER + "/z"
RAW_LOGIN_PATH = f"{SCRATCH_ROOT}/{LOGIN}/b/m.csv"


def _rule(text: str) -> str:
    return P.rewrite_producer_paths(text.encode()).decode()


def _login_table() -> P.PlaceholderTable:
    return P.PlaceholderTable.from_values({P.TOKEN_USER: [LOGIN]})


def _scanner_without_carrier(monkeypatch) -> G.ZeroUserInfoScanner:
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    return G.ZeroUserInfoScanner.for_consume()


def test_the_closed_set_holds_ten_tokens_and_the_module_docstring_names_ten():
    """Property: the closed set holds ten tokens including the producer-path token, which is not a path-valued
        row, and the module docstring says ten.

    Class: NEW CAPABILITY - the plausible wrong implementation is a ninth-token set with the rule hard-coded
        around it, and it reds on the length and on the membership.
    Reachability: the rehydrator and the table both read the tuple, so a token missing from it is unknown there.
    Both-states anchor: the nine earlier tokens are all still present beside the new one.
    Invocation isolated: reads module attributes only.
    What kills it: leaving the tuple at nine; listing the new token as a path-valued row, which would subject it
        to the absolute-path guard it has no value for.
    Second correct implementation: a tuple built by concatenating a nine-token base with the new one.
    """
    assert len(P.PLACEHOLDER_TOKENS) == 10, "unit: entries of the closed tuple"
    assert TOKEN in P.PLACEHOLDER_TOKENS
    assert TOKEN not in P._PATH_VALUED_TOKENS
    assert len(set(P.PLACEHOLDER_TOKENS)) == 10
    assert "these ten exact names" in (P.__doc__ or "")


@pytest.mark.parametrize("path", [INTERPRETER_PATH, PLAIN_PATH], ids=["relative-interpreter", "plain-absolute"])
def test_the_whole_path_rule_replaces_the_entire_path(path):
    """Property: an absolute path under the login segment, written absolutely or through a ``../`` chain, is
        replaced WHOLE by the producer-path token, with the surrounding text kept.

    Class: REGRESSION - red on the parent, where the function does not exist and the path stays as text.
    Reachability: the build estate's interpreter and an experiment's input file sit under the login segment
        and no table row covers them.
    Both-states anchor: the same sentence without a path is unchanged (second assertion).
    Invocation isolated: pure function over a string.
    What kills it: replacing only the login segment; stopping at the first following slash; keeping the chain.
    Second correct implementation: a tokenizer that splits on the end characters and replaces any word that
        holds the login-segment token.
    """
    assert _rule(f'run "{path}" now') == f'run "{TOKEN}" now'
    assert _rule(f"cmd {path}") == f"cmd {TOKEN}"
    assert _rule("no path here") == "no path here"


def test_the_whole_path_ends_at_the_first_end_character_and_not_before():
    """Property: the match ends at whitespace or one of the quote, bracket, separator and backtick characters,
        and a trailing dot or colon stays inside it.

    Class: NEW CAPABILITY - the plausible wrong implementation ends at the first non-path character class
        such as a dot, and leaves a producer file name behind the token.
    Reachability: a path is followed by a comma in a CSV row, a quote in a shell word and a bracket in a list.
    Both-states anchor: one path ended by each end character, and one whose tail holds a dot and a colon.
    Invocation isolated: pure function over strings.
    What kills it: an end class that omits a character; an end class that includes the dot.
    Second correct implementation: a lazy match followed by a lookahead for the end class.
    """
    for end in ('"', "'", "<", ">", ",", ";", ")", "]", "`", " ", "\n"):
        assert _rule(f"a {PLAIN_PATH}{end}z") == f"a {TOKEN}{end}z", repr(end)
    assert _rule(f"{SCRATCH_ROOT}/{USER}/b/m.v1.csv:tail") == TOKEN


def test_three_planted_controls_stay_unchanged_and_the_drive_path_is_refused_by_layer_a(monkeypatch):
    """Property: a URL carrying the login segment, a drive path carrying it and a path that follows another
        token are left as they are by the rule; layer (a) passes the URL and the token-prefixed path and
        refuses the drive path.

    Class: NEW CAPABILITY - the plausible wrong implementation matches any ``/`` and rewrites all three, or
        rewrites the drive path and so hides a Windows producer path the scan exists to find.
    Reachability: a generated page links a repository under the login name; a Windows path would be a producer
        path no row covers.
    Both-states anchor: the plain path beside each control IS rewritten in the same string.
    Invocation isolated: the rule is a pure function and the scanner reads no carrier.
    What kills it: a start rule without the five excluded left characters.
    Second correct implementation: a rule that parses the member into words and tests each word's first
        character against the same left-context set.
    """
    scanner = _scanner_without_carrier(monkeypatch)
    for control in (URL_WITH_LOGIN, DRIVE_PATH_WITH_LOGIN, TOKEN_PREFIXED_PATH):
        assert _rule(control) == control, control
        assert _rule(f"{control} {PLAIN_PATH}") == f"{control} {TOKEN}", control
    assert scanner.scan_member("m", URL_WITH_LOGIN.encode()) == []
    assert scanner.scan_member("m", TOKEN_PREFIXED_PATH.encode()) == []
    drive_hits = scanner.scan_member("m", DRIVE_PATH_WITH_LOGIN.encode())
    assert [hit.layer for hit in drive_hits] == [G.LAYER_PATTERNS], "unit: layer-(a) hits on the drive path"


def test_the_rule_matches_only_the_literal_token_form_and_apply_alone_does_not_run_it():
    """Property: a path still holding the raw login is not the rule's input, and the table's own ``apply``
        turns it into the login-segment token and stops there.

    Class: REGRESSION - red if the whole-path rule were folded into ``apply``, which is why it is a separate
        function.
    Reachability: ``apply`` runs first in the emit and its output is the rule's input.
    Both-states anchor: the same raw path, run through ``apply`` and then the rule, ends as the token.
    Invocation isolated: a table built from one plain value.
    What kills it: matching the raw login; running the rule inside ``apply``.
    Second correct implementation: a rule that receives the table and rewrites raw login paths itself.
    """
    table = _login_table()
    assert _rule(RAW_LOGIN_PATH) == RAW_LOGIN_PATH
    assert table.apply(RAW_LOGIN_PATH.encode()).decode() == f"{SCRATCH_ROOT}/{USER}/b/m.csv"
    assert table.apply(b"/d/" + LOGIN.encode() + b"/x").decode() == "/d/" + USER + "/x"
    assert P.rewrite_producer_paths(table.apply(RAW_LOGIN_PATH.encode())).decode() == TOKEN


def _emit_world(world, monkeypatch, name: str = "b.zip") -> Path:
    """The real ``emit_bundle`` over the world, with one renderer manifest declaring no sources."""
    monkeypatch.setattr(_emit, "_get_toolkit_git_sha", lambda strict=True: "deadbeef0000")
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    return _emit.emit_bundle(world.analysis, world.base / "bundles" / name)


@pytest.mark.filterwarnings("ignore:zero-user-info check")
def test_emit_bundle_writes_the_token_for_a_producer_path_that_a_bind_root_would_have_refused(tmp_path, monkeypatch):
    """Property: the real emit, over a generated member that holds a plain absolute path under the login
        segment and an HPC config whose bind root is a one-segment path, writes the producer-path token and
        completes with no zero-user-info hit.

    Class: REGRESSION - red on the parent, which refuses on the layer-(b) bind root.
    Reachability: a generated Snakefile names the producer's estate and the HPC config names the mount root
        every such path sits under.
    Both-states anchor: the same world emits with the same member minus the path, and the archive then holds
        no token.
    Invocation isolated: real configs over ``tmp_path``; the login name is pinned and no carrier is set.
    What kills it: never calling the rule from the emit; calling it after the scan; keeping the bind root a
        layer-(b) value.
    Second correct implementation: rewriting the staged files before zipping instead of the bytes written.
    """
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: LOGIN)
    world = _world(tmp_path)
    world.analysis.cfg_hpc_system = SimpleNamespace(model_dump=lambda mode="json": {"bind": SCRATCH_ROOT})
    _put(world.ad / "Snakefile", "run clean\n")
    clean = _emit_world(world, monkeypatch, "clean.zip")
    with zipfile.ZipFile(clean) as zf:
        assert TOKEN.encode() not in zf.read("Snakefile.source")

    _put(world.ad / "Snakefile", f"python {RAW_LOGIN_PATH} {CHAIN_ROOT}/{LOGIN}/b/.venv/bin/python\n")
    zpath = _emit_world(world, monkeypatch, "token.zip")
    with zipfile.ZipFile(zpath) as zf:
        member = zf.read("Snakefile.source").decode()
        manifest = json.loads(zf.read("bundle_manifest.json"))
    assert member == f"python {TOKEN} {TOKEN}\n", member
    assert manifest["zero_user_info"]["config_values"] == "checked"
    assert LOGIN not in member


def test_layer_b_skips_a_one_segment_configured_path_and_still_searches_a_two_segment_one(monkeypatch):
    """Property: with a one-segment configured path and a two-segment one, text holding a path beneath the
        first fires nothing and text holding the second fires layer (b); the extraction still returns both.

    Class: REGRESSION - red on the parent, where the one-segment value fires on every path beneath it.
    Reachability: a container bind source such as the mount root is a configured path.
    Both-states anchor: the two-segment value fires on the text that holds it (the must-fire control).
    Invocation isolated: a stand-in analysis holding only an HPC config dump; the login is pinned.
    What kills it: dropping the one-segment value inside ``config_path_values``, which breaks the landed
        extraction expectation; skipping every value.
    Second correct implementation: a skip keyed on the path's depth computed with ``PurePosixPath.parts``.
    """
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: "zz")
    one, two = "/sf" + "s", "/sf" + "s/x"
    dump = {"bind_roots": [one], "bind_pair": two}
    analysis = SimpleNamespace(cfg_hpc_system=SimpleNamespace(model_dump=lambda mode="json": dump))
    assert G.config_path_values([dump]) == [one, two]
    with pytest.warns(UserWarning):
        scanner = G.ZeroUserInfoScanner.for_emit(analysis)
    assert scanner.scan_member("m", f"see {one}/y".encode()) == []
    hits = scanner.scan_member("m", f"see {two}".encode())
    assert [hit.layer for hit in hits] == [G.LAYER_CONFIG_VALUES], "unit: layer-(b) hits on the two-segment text"


def test_rehydration_leaves_the_producer_path_token_in_place(tmp_path):
    """Property: a member holding the producer-path token is rehydrated to a copy that still holds it, with no
        ``ConfigurationError``, while an unresolved directory token in the same position is still refused.

    Class: REGRESSION - red on the parent, where the token is not in the closed set and a caller that names it
        gets an unknown-token refusal.
    Reachability: any consumer that parses a rewritten Snakefile or sidecar calls the rehydrator first.
    Both-states anchor: the control member holds a directory token with no default and no bundle root, and it
        raises.
    Invocation isolated: members on ``tmp_path``; no bundle manifest sits above them.
    What kills it: giving the token a default value; treating it as unresolved.
    Second correct implementation: a default that returns the token's own text.
    """
    member = _put(tmp_path / "m" / "Snakefile.source", f"python {TOKEN} --in {TOKEN}\n")
    resolved = rehydrate_placeholders(member, dest_dir=tmp_path / "out")
    assert resolved.read_bytes() == member.read_bytes()
    assert rehydrate_placeholders(member, {TOKEN: "/kept"}, dest_dir=tmp_path / "out2").read_text() == (
        "python /kept --in /kept\n"
    )
    control = _put(tmp_path / "m" / "other.source", f"cd {P.TOKEN_ANALYSIS_DIR}\n")
    with pytest.raises(ConfigurationError):
        rehydrate_placeholders(control, dest_dir=tmp_path / "out3")


def test_a_declared_input_is_archived_byte_identical_and_the_same_text_in_another_member_is_rewritten(
    tmp_path, monkeypatch
):
    """Property: a declared input whose text already holds the literal token form of a path under the login
        segment is archived byte for byte, while the same text in a member that is not declared becomes the
        producer-path token.

    Class: NEW CAPABILITY - the plausible wrong implementation calls the rule after the guard that exempts
        declared inputs, and rewrites the declared member too.
    Reachability: a declared text input can hold any text, including the token form.
    Both-states anchor: the second member in the same archive, with the same bytes, is rewritten.
    Invocation isolated: ``_emit_bundle_zip`` over a staging directory on ``tmp_path`` with a real scanner.
    What kills it: the rule applied unconditionally to every text member; the rule placed after the guard.
    Second correct implementation: applying the rule inside ``PlaceholderTable.apply`` with a declared-name
        check there.
    """
    text = f"cd {PLAIN_PATH}\n"
    staging = tmp_path / "staging"
    _put(staging / "declared.txt", text)
    _put(staging / "generated.txt", text)
    zpath = tmp_path / "out.zip"
    scanner = _scanner_without_carrier(monkeypatch)
    _emit_bundle_zip(
        staging,
        zpath,
        placeholders=_login_table(),
        declared_arcnames=frozenset({"declared.txt"}),
        scanner=scanner,
    )
    with zipfile.ZipFile(zpath) as zf:
        assert zf.read("declared.txt") == text.encode()
        assert zf.read("generated.txt").decode() == f"cd {TOKEN}\n"
    assert scanner.hits == []


CHAIN3 = "../" * 3
CHAIN7 = "../" * 7
OUTSIDE_NAME = "wea" + "ther/storm" + ".nc"


def _sidecar_member() -> str:
    """A figure sidecar's shape: a JSON list holding one outside-tree entry and one in-tree entry."""
    return json.dumps({"source_paths_relative": [CHAIN7 + OUTSIDE_NAME, "inputs/in_tree.csv"]})


def _svg_member() -> str:
    """An SVG whose embedded source text joins the entries with a semicolon."""
    return f"<svg><metadata><dc:source>{CHAIN3}dem.tif; inputs/in_tree.csv</dc:source></metadata></svg>"


def _html_member() -> str:
    """A report configuration cell: a code element holding a relative path outside the analysis tree."""
    return f"<tr><td><code>{CHAIN7}{OUTSIDE_NAME}</code></td></tr>"


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        (_sidecar_member, json.dumps({"source_paths_relative": [TOKEN, "inputs/in_tree.csv"]})),
        (_svg_member, f"<svg><metadata><dc:source>{TOKEN}; inputs/in_tree.csv</dc:source></metadata></svg>"),
        (_html_member, f"<tr><td><code>{TOKEN}</code></td></tr>"),
    ],
    ids=["json-sidecar", "svg-joined-source", "html-code-cell"],
)
def test_a_relative_chain_of_three_or_more_levels_is_replaced_whole_in_each_member_shape(member, expected):
    """Property: a relative path that climbs three or more levels and names something is replaced whole by the
        producer-path token in a JSON sidecar, an SVG's joined source text and an HTML code cell, and the
        in-tree neighbour in the same member is untouched.

    Class: REGRESSION - red on the parent, where none of the three holds the login segment and all stay as text.
    Reachability: the renderers write a source outside the analysis directory as a relative path, and the report
        prints a configured path the same way.
    Both-states anchor: the in-tree entry beside the chain is byte-identical in the same string.
    Invocation isolated: pure function over a string.
    What kills it: a rule keyed on the login segment only; an end set missing the quote, the angle bracket or
        the semicolon.
    Second correct implementation: parsing the JSON and rewriting entries, plus an HTML and an XML pass.
    """
    assert _rule(member()) == expected


@pytest.mark.parametrize(
    "text",
    [
        "../x/y.csv",
        "../../x/y.csv",
        f"<code>../../{OUTSIDE_NAME}</code>",
        f"{CHAIN3}{TOKEN}/y",
        f"{CHAIN3}{P.TOKEN_SLURM_ACCOUNT}/y",
        f"{P.TOKEN_HOME}/{CHAIN3}y",
        f"{CHAIN3}{P.TOKEN_HOME}/y",
        f"a/{CHAIN3}y",
        CHAIN3,
        "../" * 12,
    ],
    ids=[
        "one-level",
        "two-level",
        "two-level-in-html",
        "chain-then-token",
        "chain-then-account-token",
        "after-token",
        "chain-with-token",
        "tail",
        "bare",
        "bare-long",
    ],
)
def test_shorter_chains_tails_and_already_tokenised_strings_stay_byte_identical(text):
    """Property: one and two levels, a chain that already holds a token, a chain that is the tail of a longer
        path and a chain with nothing after it are left exactly as they are.

    Class: NEW CAPABILITY - the plausible wrong implementation tokenises any ``../`` and so rewrites the in-tree
        references of a neighbour, or rewrites a string a second time.
    Reachability: a bundle member holds short upward references and token-bearing strings beside the chains.
    Both-states anchor: the three-level chain with a name in the first parametrised test IS rewritten.
    Invocation isolated: pure function over a string.
    What kills it: a threshold below three; no check for an existing token; a start rule without the left
        context.
    Second correct implementation: counting levels in a callback and returning the match unchanged below three.
    """
    assert _rule(text) == text


def test_the_rewrite_is_idempotent_over_every_shape():
    """Property: rewriting twice gives what rewriting once gives, over the login path, the chains and a mix.

    Class: NEW CAPABILITY - the plausible wrong implementation leaves a fragment the second pass rewrites again.
    Reachability: the emit runs the rule once, but a bundle-derived tree is emitted again.
    Both-states anchor: each input differs from its output, so the rule did something to run twice.
    Invocation isolated: pure function over strings.
    What kills it: a replacement that itself matches the pattern.
    Second correct implementation: a single combined scan.
    """
    mixed = f"{_sidecar_member()}\n{_svg_member()}\n{_html_member()}\n{PLAIN_PATH} {INTERPRETER_PATH}\n"
    for text in (_sidecar_member(), _svg_member(), _html_member(), mixed):
        once = _rule(text)
        assert once != text
        assert _rule(once) == once


def test_a_declared_input_holding_a_chain_is_archived_byte_identical(tmp_path, monkeypatch):
    """Property: a declared input that holds a three-level chain is archived byte for byte, while the same text
        in a member that is not declared is rewritten.

    Class: NEW CAPABILITY - the plausible wrong implementation applies the chain rule outside the guard.
    Reachability: a declared text input can hold any text.
    Both-states anchor: the second member, with the same bytes, is rewritten.
    Invocation isolated: ``_emit_bundle_zip`` over a staging directory on ``tmp_path``.
    What kills it: running the chain rule after the guard or in the scanner's path.
    Second correct implementation: rewriting declared members too and restoring them from the source.
    """
    text = _html_member()
    staging = tmp_path / "staging"
    _put(staging / "declared.html", text)
    _put(staging / "generated.html", text)
    zpath = tmp_path / "out.zip"
    _emit_bundle_zip(
        staging,
        zpath,
        placeholders=_login_table(),
        declared_arcnames=frozenset({"declared.html"}),
    )
    with zipfile.ZipFile(zpath) as zf:
        assert zf.read("declared.html") == text.encode()
        assert zf.read("generated.html").decode() == _rule(text) != text


def test_a_sidecar_after_the_rewrite_lets_a_regenerated_snakefile_dry_run_complete(tmp_path):
    """Property: a regenerated plot rule whose sidecar source entry is the producer-path token completes a
        Snakemake dry run, as the entries already holding other tokens do. The rule is written by hand in the
        emitter's literal form; the emitter itself is exercised by the brace-token tests of the workflow module.

    Class: REGRESSION - guards the consumer: a token in a literal source list must not become a wildcard.
    Reachability: the rewritten sidecar's entries feed the rule's literal source list.
    Both-states anchor: the control rule with a plain relative entry also completes.
    Invocation isolated: a Snakefile on ``tmp_path`` run with the pinned Snakemake; no solver.
    What kills it: a rule emitter that formats the entry as a wildcard pattern.
    Second correct implementation: dropping token entries from the literal list before writing the rule.
    """
    import shutil
    import subprocess

    snakemake = shutil.which("snakemake")
    if snakemake is None:
        pytest.skip("snakemake is not on PATH")
    for name, entry in (("token", TOKEN), ("plain", "inputs/in_tree.csv")):
        out = tmp_path / name
        out.mkdir()
        (out / "Snakefile").write_text(
            "rule all:\n    input: 'fig.png'\n"
            "rule fig:\n    output: 'fig.png'\n"
            f"    params:\n        source_paths = lambda w: {[entry]!r}\n"
            "    shell: 'touch {output}'\n"
        )
        result = subprocess.run(
            [snakemake, "-n", "--cores", "1", "-s", str(out / "Snakefile"), "-d", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (name, result.stderr[-400:])


def test_a_chain_and_a_folder_in_front_of_a_token_are_replaced_whole():
    """Property: a chain, a folder name and then a token are replaced whole, so the folder name does not ship.

    Class: REGRESSION - red under a rule that leaves any chain holding a token anywhere.
    Reachability: the table rewrites an account name inside a relative path before this rule runs.
    Both-states anchor: the sibling control whose first name is the token stays byte-identical.
    Invocation isolated: pure function over strings.
    What kills it: an exemption keyed on a token anywhere in the span.
    Second correct implementation: testing the first segment after the levels in a callback.
    """
    assert _rule(f"{CHAIN3}norfolk/{P.TOKEN_SLURM_ACCOUNT}/x") == TOKEN
    assert _rule(f'"{CHAIN3}norfolk/{P.TOKEN_SLURM_ACCOUNT}/x"') == f'"{TOKEN}"'
    assert _rule(f"{CHAIN3}{P.TOKEN_SLURM_ACCOUNT}/norfolk/x") == f"{CHAIN3}{P.TOKEN_SLURM_ACCOUNT}/norfolk/x"
    once = _rule(f"{CHAIN3}norfolk/{P.TOKEN_SLURM_ACCOUNT}/x")
    assert _rule(once) == once

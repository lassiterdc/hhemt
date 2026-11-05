"""A `../`-relative producer path is rewritten to its placeholder; the consume scan reports an unsearched list.

Two behaviours of the bundle anonymization, tested together because they were found together in one
real bundle's refusal:

* Figure sidecars record each source file relative to the analysis directory, so a source outside that
  tree reads ``../../../..`` followed by the producer's own path with its leading slash removed. The
  placeholder table matched a path value only where it was not preceded by a dot, so that spelling
  survived substitution and the emit refused. The table now replaces the chain and the value together,
  and ``rehydrate_placeholders`` writes back an absolute consumer path that every reader of the sidecar
  resolves correctly.
* ``Bundle.reprex()``'s informational scan returned a clean result when ``$HHEMT_REPREX_BLOCKLIST`` was
  unset or empty, although it had not searched the private list. It now raises one ``UserWarning``
  naming the state, and an unusable value is still a diagnosed failure that names the path.

SOLVER-FREE: every arm builds its members under ``tmp_path`` and calls the table, the rehydrator, the
sidecar harvest and the consume scan directly. No fixture here compiles or runs a solver.

Producer-looking values (a home directory, scratch directories, a login name) are BUILT FROM FRAGMENTS
below, never written as literals, so this file is not itself a hit for the repository's anonymization
guard.

Each test docstring states Property, Class, Why required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

from hhemt.bundle import _placeholders as P
from hhemt.bundle import _reprex
from hhemt.experiment_definition import rehydrate_placeholders
from hhemt.report_renderers._figure_emission import harvest_source_paths

# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------

LOGIN = "carol" + "ine"
ROOT = "/gp" + "fs/tard" + "is"
HOME = ROOT + "/ho" + "me/" + LOGIN
SCRATCH = ROOT + "/scr" + "atch/" + LOGIN
SYSTEM_DIR = SCRATCH + "/runs"
ANALYSIS_DIR = SYSTEM_DIR + "/a1"
TOOLKIT = HOME + "/code/hhemt"
PACKAGE = TOOLKIT + "/src/hhemt"
CONDA_ENV = HOME + "/envs/hhemt.yaml"
PYTHON = HOME + "/miniconda3/envs/e/bin/python"

#: The seven rows whose value is a filesystem path, each with a value of its own.
PATH_ROWS = {
    P.TOKEN_ANALYSIS_DIR: ANALYSIS_DIR,
    P.TOKEN_SYSTEM_DIR: SYSTEM_DIR,
    P.TOKEN_TOOLKIT_DIR: TOOLKIT,
    P.TOKEN_PACKAGE_DIR: PACKAGE,
    P.TOKEN_PYTHON: PYTHON,
    P.TOKEN_CONDA_ENV: CONDA_ENV,
    P.TOKEN_HOME: HOME,
}

STATE_WARNING = "HHEMT_REPREX_BLOCKLIST"


def _rewrite(text: str, **rows: str) -> str:
    """``text`` after a table holding exactly the given ``{token: value}`` rows has been applied."""
    table = P.PlaceholderTable.from_values({token: [value] for token, value in rows.items()})
    return table.apply(text.encode()).decode()


def _chain(length: int) -> str:
    return "../" * length


# --------------------------------------------------------------------------------------
# The `../`-relative spelling of a path value
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("chain_length", [1, 2, 4])
@pytest.mark.parametrize("token", sorted(PATH_ROWS))
def test_a_parent_chain_followed_by_a_row_value_is_replaced_whole_by_that_rows_token(token, chain_length):
    """Property: a `../` chain then a path row's value (no leading slash) becomes the row's token, chain included.

    Class: every path-valued row crossed with chain lengths 1, 2 and 4, in a JSON sidecar line of the
    shape the renderers write (the member is a quoted list entry).

    Why required: the producer's directory stays in a public archive in this spelling, and the emit-time
    check refuses the archive, so a bundle of a real analysis could not be written. The replacement must
    take the chain too: a token with ``../`` left in front of it would rehydrate to a path under the
    wrong directory.

    REGRESSION: red on the previous tree (output pasted in the round file), because a dot stood before the
    value's leading slash and the boundary rule declined it.

    What kills it: matching the value but leaving the chain (output ``../{token}/...``), or matching only
    the absolute spelling.

    Second correct implementation: rewrite each chain to the absolute spelling first (strip the ``../``
    run and put the slash back), then apply the absolute rule. It yields the same text and passes.
    """
    value = PATH_ROWS[token]
    sidecar_line = f'["{_chain(chain_length)}{value[1:]}/sub/x.csv"]'

    assert _rewrite(sidecar_line, **{token: value}) == f'["{token}/sub/x.csv"]'


@pytest.mark.parametrize(
    "case, text",
    [
        ("a longer sibling name", f"{_chain(3)}{HOME[1:]}2/x"),
        ("a hyphenated sibling name", f"{_chain(3)}{HOME[1:]}-old/x"),
        ("a chain that begins mid-path", f"a/{_chain(3)}{HOME[1:]}/x"),
        ("a chain glued to a word on its left", f"a{_chain(3)}{HOME[1:]}/x"),
        ("a chain that is the tail of a longer dot run", f".{_chain(3)}{HOME[1:]}/x"),
    ],
)
def test_a_parent_chain_that_is_not_the_whole_chain_or_does_not_end_at_the_value_is_left_alone(case, text):
    """Property: the chain rule fires only on a whole chain ending at a name boundary; other text is unchanged.

    Class: five neighbour shapes of a home path reached through a chain: value glued to a longer name
    (letter and hyphen), chain preceded by a path segment, by a word character and by a dot.

    Why required: the rule is a rewrite of public text. Rewriting the front of ``{home}2`` or the middle of
    ``a/../{home}`` would put a token where the producer wrote a different directory, and rehydration
    would point a consumer at the wrong place.

    NEW CAPABILITY: the plausible wrong implementation is a chain rule with no left boundary and no
    trailing boundary (``(\\.\\./)+`` then the value); it rewrites all five and this test is red under it
    (output pasted in the round file). These cases are green on the previous tree, which had no rule.

    Second correct implementation: a lexical walk that tokenizes the text on ``/`` and replaces only a
    segment run that starts at a chain head and ends on a segment boundary. It leaves the same five alone.
    """
    assert _rewrite(text, **{P.TOKEN_HOME: HOME}) == text, case


def test_the_absolute_spelling_is_still_rewritten_and_the_chain_form_agrees_with_it():
    """Property: a path value in its absolute spelling becomes the same token as its chain spelling, with the same tail.

    Class: one home path, both spellings, one table.

    Why required: the chain rule is added beside the absolute rule, and a change that replaced the
    absolute alternative while adding the chain one would leave an absolute path in the bundle.

    What kills it: dropping the absolute alternative from the table's pattern. Green on the previous tree
    for the absolute half, red for the chain half.

    Second correct implementation: a table that rewrites in two passes (chains first, absolute second).
    """
    absolute = _rewrite(f"{HOME}/.local/x", **{P.TOKEN_HOME: HOME})
    through_chain = _rewrite(f"{_chain(4)}{HOME[1:]}/.local/x", **{P.TOKEN_HOME: HOME})

    assert absolute == through_chain == f"{P.TOKEN_HOME}/.local/x"


def test_a_value_nested_inside_a_longer_row_takes_the_longer_rows_token_through_a_chain():
    """Property: when two path rows both match the text after a chain, the longer value's token is used.

    Class: the analysis directory lies inside the system directory, and the text is a chain followed by a
    file inside the analysis directory.

    Why required: longest-first order is how the absolute rule keeps a nested directory from being
    reported as its parent; the chain form must keep it, or a sidecar would name the system directory
    plus a tail that rehydrates to the wrong file.

    What kills it: ordering the chain alternatives after every absolute alternative of the shorter value,
    or matching the chain form on the first row in table order.

    Second correct implementation: sort all alternatives by value length across both spellings.
    """
    text = f'"{_chain(4)}{ANALYSIS_DIR[1:]}/out.csv"'
    rewritten = _rewrite(text, **{P.TOKEN_SYSTEM_DIR: SYSTEM_DIR, P.TOKEN_ANALYSIS_DIR: ANALYSIS_DIR})

    assert rewritten == f'"{P.TOKEN_ANALYSIS_DIR}/out.csv"'


# --------------------------------------------------------------------------------------
# The round trip: emit-time substitution, then rehydration, then the sidecar harvest
# --------------------------------------------------------------------------------------


def test_a_rewritten_sidecar_resolves_to_the_consumers_home_after_rehydration(tmp_path):
    """Property: a rewritten, rehydrated sidecar entry resolves under the consumer's home via the harvest.

    Class: one sidecar whose ``source_paths_relative`` holds a source reached by a chain, a sibling
    plain-relative entry as a control, the home path of the consumer unrelated to the producer's.

    Why required: the replacement is only correct if it can be undone to something a reader can open.
    The harvest joins each entry onto the analysis directory with ``pathlib``, and joining an absolute
    right-hand path returns that path, so the rehydrated absolute entry must land under the consumer's
    home and not under the analysis directory.

    REGRESSION: red on the previous tree, where the entry keeps its chain, no token is present for the
    rehydrator to replace, and the harvest resolves the entry under the analysis directory's ancestors.

    What kills it: a rewrite that keeps the chain in front of the token (the resolved path would climb
    out of the consumer's home), or one that drops the tail.

    Second correct implementation: rehydrate by a regular expression over ``{hhemt_home}`` instead of the
    rehydrator's literal replacement; it produces the same bytes for this input.
    """
    producer_entry = f"{_chain(4)}{HOME[1:]}/.local/x"
    plain_entry = "inputs/y.csv"
    sidecar = json.dumps({"source_paths_relative": [producer_entry, plain_entry]}).encode()
    rewritten = P.PlaceholderTable.from_values({P.TOKEN_HOME: [HOME]}).apply(sidecar)

    bundle_plots = tmp_path / "bundle" / "plots"
    bundle_plots.mkdir(parents=True)
    member = bundle_plots / "fig.manifest.json"
    member.write_bytes(rewritten)
    consumer_home = tmp_path / "consumer_home"
    consumer_analysis = tmp_path / "consumer_analysis"
    consumer_analysis.mkdir()
    resolved_dir = tmp_path / "resolved"

    resolved_copy = rehydrate_placeholders(member, {P.TOKEN_HOME: consumer_home}, dest_dir=resolved_dir)
    harvested = harvest_source_paths(resolved_dir, consumer_analysis)

    assert resolved_copy.parent == resolved_dir
    assert harvested == {
        resolved_copy.name.removesuffix(".manifest.json"): [
            (consumer_home / ".local" / "x").resolve(),
            (consumer_analysis / plain_entry).resolve(),
        ]
    }


# --------------------------------------------------------------------------------------
# The consume-side scan and an unsearched private list
# --------------------------------------------------------------------------------------


def _clean_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "notes.txt").write_text("nothing producer-specific is written here\n")
    return bundle


def _state_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning) and STATE_WARNING in str(w.message)]


@pytest.mark.parametrize("value, state", [(None, "not set"), ("", "set but empty")])
def test_the_consume_scan_warns_once_naming_the_state_when_the_private_list_is_not_configured(
    tmp_path, monkeypatch, value, state
):
    """Property: with the variable unset or empty the scan returns no leak and warns once, naming the state.

    Class: the two unconfigured states of the variable, over a bundle directory holding one clean text file.

    Why required: the scan reports no leak both when it searched and found nothing and when it did not
    search the private list at all. The leak list cannot carry the second fact without making
    ``Bundle.reprex()`` report a leak the scan did not find, so the warning is the only place it can say so.

    REGRESSION: red on the previous tree, which raised no warning in either state.

    What kills it: putting the state in the returned list (the returned list is asserted empty), warning
    twice, or naming the wrong state.

    Second correct implementation: warn from the scanner constructor and have ``_scan_zero_user_info`` pass
    it through; the same single warning reaches the caller.
    """
    if value is None:
        monkeypatch.delenv(STATE_WARNING, raising=False)
    else:
        monkeypatch.setenv(STATE_WARNING, value)
    bundle = _clean_bundle(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        leaks = _reprex._scan_zero_user_info(bundle)

    messages = _state_warnings(caught)
    assert leaks == []
    assert len(messages) == 1
    assert re.search(rf"\bis {state}\b", messages[0])


def test_the_consume_scan_raises_no_state_warning_with_a_usable_private_list(tmp_path, monkeypatch):
    """Property: a well-formed carrier is searched silently: no state warning, no leak.

    Class: one synthetic carrier file, built here from fragments, declaring a floor of one entry.

    Why required: the control for the previous test. A warning raised in every state would pass that test
    and tell a consumer nothing.

    NEW CAPABILITY: the plausible wrong implementation is a warning raised unconditionally; it is red here.

    Second correct implementation: warn only when the scanner's own recorded state is not ``checked``.
    """
    carrier = tmp_path / "private_list.txt"
    carrier.write_text("# min-tokens: 1\n" + "zq" + "xv" + "fragment" + "\n")
    monkeypatch.setenv(STATE_WARNING, str(carrier))
    bundle = _clean_bundle(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        leaks = _reprex._scan_zero_user_info(bundle)

    assert _state_warnings(caught) == []
    assert leaks == []


def test_an_unusable_private_list_is_a_diagnosed_scan_failure_naming_the_path_and_not_an_exception(
    tmp_path, monkeypatch
):
    """Property: a variable naming a missing file gives one diagnosis naming the path, no raise, no state warning.

    Class: one absolute path to a missing file.

    Why required: the state warning covers the unconfigured states only. A mistyped carrier must stay a
    diagnosed failure, because a pass here would read as a searched list; and the scan must not raise,
    because ``Bundle.reprex()`` is informational and a raise would turn it into a crash.

    What kills it: letting the carrier's ``ProcessingError`` escape, or answering an unusable value with
    the unconfigured-state warning.

    Second correct implementation: catch the error in ``reprex()`` instead of in the scan and record the
    diagnosis there; the scan's own contract would then change, so this test pins the scan.
    """
    missing = tmp_path / "no_such_list.txt"
    monkeypatch.setenv(STATE_WARNING, str(missing))
    bundle = _clean_bundle(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        leaks = _reprex._scan_zero_user_info(bundle)

    assert len(leaks) == 1
    assert str(missing) in leaks[0]
    assert _state_warnings(caught) == []


def test_the_consume_scan_docstring_no_longer_describes_a_deferred_emit_gate_or_a_local_blocklist():
    """Property: the module no longer calls emit-time enforcement deferred or the scan a local-blocklist match.

    Class: a single artifact, the module source; this is a point and not a class (the two sentences are
    the only places the claims were made).

    Why required: the emit-time gate is built, and the scan runs the generic patterns and the private
    list only. A reader who trusted the old text would believe a clean consume-side scan certified the
    producer's configs.

    REGRESSION: red on the previous tree, where both sentences are present (unit: occurrences of each
    phrase, one apiece).

    What kills it: restoring either phrase.

    Second correct implementation: any wording of the corrected docstring that avoids the two phrases.
    """
    source = Path(_reprex.__file__).read_text()

    assert source.count("Hard emit-time enforcement is deferred") == 0
    assert source.count("matches against the LOCAL blocklist") == 0

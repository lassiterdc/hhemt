"""The `shared_runs_root` marker is derived from fixture closures and test bodies, never from a hand-kept list.

Problem this file pins. The synthetic builders write their analysis trees under one per-worktree runs root unless
`HHEMT_TEST_RUNS_ROOT_OVERRIDE` points elsewhere. The laptop lane has to keep the tests that touch that root off the
pool's per-test override directory, and a hand-kept list of those tests drifts the first time a fixture is added.
`tests/conftest.py::pytest_collection_modifyitems` therefore computes the set at collection (a fixture in the closure
calls a `retrieve_*` builder before any override is set, or the test's own body does) and gives each member an
`xdist_group`. These tests plant small modules in a throwaway rootdir, collect them with the real hook, and read the
marks it gave each node. The planted fixtures are stand-ins that are never executed, so no solver runs and no tree is
built.

Each planted test is named for the case it covers. `EXPECTED` is the full verdict table; the three mutant tests show
that dropping an arm of the derivation, or ignoring the override, changes that table, which is what makes the
positive tests able to fail.

Two spellings below are assembled from pieces on purpose. The suite partitioner (`src/hhemt/suite/partition.py::
file_trees`) reads a test file's text, attaches the file to every analysis tree whose catalog builder name occurs in
it, and refuses a file that names the dynamic fixture-lookup API unless a literal resolves. This file plants such
text only inside strings, so spelling those names whole would wire a light unit test into the heavy tree component.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import tests.conftest as suite_conftest
from tests.fixtures.test_case_catalog import Local_TestCases

pytest_plugins = ["pytester"]

# Builder and lookup names, spelled in pieces: see the module docstring.
_ALL_MODELS = "retrieve_" + "synth_all_models_test_case"
_MULTI_SIM = "retrieve_" + "synth_multi_sim_test_case"
_SENSITIVITY = "retrieve_" + "synth_cpu_config_sensitivity_case"
_DIRECT = "retrieve_" + "synth_TRITON_SWMM_test_case"
_DYNAMIC_LOOKUP = "get" + "fixturevalue"

_INNER_CONFTEST = """
import pytest

from tests.conftest import pytest_collection_modifyitems  # noqa: F401  (the hook under test)


@pytest.fixture
def tritonswmm_cpu_compiled():
    # Stand-in for the compile gate. Like the real one it builds the all-models tree, so it is a shared-root
    # fixture that the `_compiled` suffix marks as covered.
    Local_TestCases.@ALL@(start_from_scratch=False)


@pytest.fixture
def synthetic_multisim_completed(tritonswmm_cpu_compiled):
    Local_TestCases.@MULTI@(start_from_scratch=False)


@pytest.fixture
def rendered_synth_multi_sim(tritonswmm_cpu_compiled):
    Local_TestCases.@MULTI@(start_from_scratch=False)


@pytest.fixture
def wipes_all_models():
    Local_TestCases.@ALL@(start_from_scratch=True)


@pytest.fixture
def reads_sensitivity():
    Local_TestCases.@SENS@(start_from_scratch=False)


@pytest.fixture
def names_its_tree_directly():
    @DIRECT@(analysis_name="named_tree", start_from_scratch=False)


@pytest.fixture
def docstring_mentions_override():
    '''Explains that HHEMT_TEST_RUNS_ROOT_OVERRIDE redirects the tree. It sets nothing.'''
    Local_TestCases.@MULTI@(start_from_scratch=False)


@pytest.fixture
def override_by_setenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    Local_TestCases.@MULTI@(start_from_scratch=True)


@pytest.fixture
def override_by_context(tmp_path):
    with _runs_root_override_env(tmp_path):
        Local_TestCases.@MULTI@(start_from_scratch=True)


@pytest.fixture
def override_by_keyword(tmp_path):
    @DIRECT@(analysis_name="kw_tree", runs_root_override=tmp_path)


@pytest.fixture
def override_set_too_late(tmp_path, monkeypatch):
    Local_TestCases.@MULTI@(start_from_scratch=False)
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))


@pytest.fixture
def builds_nothing():
    return "no builder is called here"
"""

_INNER_TESTS = """
import pytest


def test_closure_wiper(wipes_all_models):
    pass


def test_closure_names_tree_directly(names_its_tree_directly):
    pass


def test_closure_docstring_does_not_count_as_override(docstring_mentions_override):
    pass


def test_closure_override_set_too_late(override_set_too_late):
    pass


def test_body_calls_builder():
    Local_TestCases.@MULTI@(start_from_scratch=False)


@pytest.mark.parametrize("fixture_name", ["wipes_all_models"])
def test_fetches_wiper_by_name(request, fixture_name):
    request.@GFV@(fixture_name)


def test_free(builds_nothing):
    pass


def test_closure_override_by_setenv(override_by_setenv):
    pass


def test_closure_override_by_context(override_by_context):
    pass


def test_closure_override_by_keyword(override_by_keyword):
    pass


def test_body_overrides_first(tmp_path, monkeypatch):
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    Local_TestCases.@MULTI@(start_from_scratch=False)


@pytest.mark.parametrize("label", ["not-a-fixture-name"])
def test_string_value_naming_no_fixture(label):
    pass


def test_compile_only(tritonswmm_cpu_compiled):
    pass


def test_compile_covered_completed(synthetic_multisim_completed):
    pass


def test_compile_covered_rendered(rendered_synth_multi_sim):
    pass


def test_compile_uncovered_one_tree(tritonswmm_cpu_compiled, wipes_all_models):
    pass


def test_compile_uncovered_two_trees(tritonswmm_cpu_compiled, wipes_all_models, reads_sensitivity):
    pass


def test_compile_body_builder(tritonswmm_cpu_compiled):
    Local_TestCases.@MULTI@(start_from_scratch=False)
"""

_BARE = "shared_runs_root"

# name -> (carries shared_runs_root, carries compile_tier, xdist_group names). One row per planted test. This is the
# table for a session that keeps both tiers (no `-m`): every marked test is grouped by tree. `LAPTOP_EXPECTED` below is
# the same project under `-m "not compile_tier"`.
EXPECTED: dict[str, tuple[bool, bool, tuple[str, ...]]] = {
    "test_closure_wiper": (True, False, (f"{_BARE}:synth_all_models+synth_sensitivity",)),
    "test_closure_names_tree_directly": (True, False, (f"{_BARE}:named_tree",)),
    "test_closure_docstring_does_not_count_as_override": (True, False, (f"{_BARE}:synth_multi_sim",)),
    "test_closure_override_set_too_late": (True, False, (f"{_BARE}:synth_multi_sim",)),
    "test_body_calls_builder": (True, False, (f"{_BARE}:synth_multi_sim",)),
    "test_fetches_wiper_by_name": (True, False, (f"{_BARE}:synth_all_models+synth_sensitivity",)),
    "test_free": (False, False, ()),
    "test_closure_override_by_setenv": (False, False, ()),
    "test_closure_override_by_context": (False, False, ()),
    "test_closure_override_by_keyword": (False, False, ()),
    "test_body_overrides_first": (False, False, ()),
    "test_string_value_naming_no_fixture": (False, False, ()),
    "test_compile_only": (True, True, ()),
    "test_compile_covered_completed": (True, True, (f"{_BARE}:synth_multi_sim",)),
    "test_compile_covered_rendered": (True, True, (f"{_BARE}:rendered_synth_multi_sim",)),
    "test_compile_uncovered_one_tree": (True, True, (f"{_BARE}:synth_all_models+synth_sensitivity",)),
    "test_compile_uncovered_two_trees": (True, True, (f"{_BARE}:synth_all_models+synth_sensitivity",)),
    "test_compile_body_builder": (True, True, (f"{_BARE}:synth_multi_sim",)),
}


# The laptop session (`-m "not compile_tier"`): no compile_tier test is selected, so every marked test outside that
# tier keeps the bare group and the compile_tier rows (deselected by pytest) carry none.
LAPTOP_EXPECTED: dict[str, tuple[bool, bool, tuple[str, ...]]] = {
    name: (marked, tier, () if tier else ((_BARE,) if marked else ()))
    for name, (marked, tier, _groups) in EXPECTED.items()
}


def _plant(text: str) -> str:
    for token, value in {
        "@ALL@": _ALL_MODELS,
        "@MULTI@": _MULTI_SIM,
        "@SENS@": _SENSITIVITY,
        "@DIRECT@": _DIRECT,
        "@GFV@": _DYNAMIC_LOOKUP,
    }.items():
        text = text.replace(token, value)
    return text


@pytest.fixture
def collect_planted(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Return a function that collects the planted module with the real hook and reads each node's marks.

    The optional `markexpr` is passed as the session's `-m`; items it deselects stay in the result, because pytest
    deselects after the hook has run and `inline_genitems` reports every collected item."""
    monkeypatch.delenv("HHEMT_FORBID_COMPILE", raising=False)  # the hook arms a guard when this is "1"

    def collect(markexpr: str | None = None) -> dict[str, tuple[bool, bool, tuple[str, ...]]]:
        """Collect the planted project under `markexpr`; map each test name to (marked, compile_tier, group names)."""
        pytester.makeconftest(_plant(_INNER_CONFTEST))
        pytester.makepyfile(test_planted=_plant(_INNER_TESTS))
        items, _ = pytester.inline_genitems(*(["-m", markexpr] if markexpr else []))
        return {
            item.originalname: (
                item.get_closest_marker("shared_runs_root") is not None,
                item.get_closest_marker("compile_tier") is not None,
                tuple(str(m.args[0]) for m in item.iter_markers("xdist_group")),
            )
            for item in items
        }

    return collect


def test_the_planted_verdict_table_is_what_the_derivation_gives(collect_planted):
    """Property: the derivation gives exactly the verdict table above, mark, tier and group for every planted test.

    Class: NEW CAPABILITY. The wrong implementation this reds under is a derivation that reads only the fixture
    closure, only the test body, or the names of fixtures instead of their source; the mutant tests below run those.
    A second correct implementation, a token-stream scan in place of `ast`, passes it unchanged."""
    assert collect_planted() == EXPECTED


def test_a_free_test_stays_unmarked_and_ungrouped(collect_planted):
    """Control: tests that build nothing, or set the override before the builder, carry neither mark nor group.

    Class: NEW CAPABILITY. Reds under a derivation that marks every test holding a `retrieve_*` name, which
    ignores the override. Varies the property under suspicion (override set first) rather than the token: the five
    controls all name a builder."""
    marks = collect_planted()
    controls = [
        "test_free",
        "test_closure_override_by_setenv",
        "test_closure_override_by_context",
        "test_closure_override_by_keyword",
        "test_body_overrides_first",
        "test_string_value_naming_no_fixture",
    ]
    assert [marks[name] for name in controls] == [(False, False, ())] * len(controls)


def test_no_compile_tier_test_carries_the_bare_group_and_each_uncovered_one_names_its_tree(collect_planted):
    """Property: the bare group is for tests outside compile_tier; a compile_tier test is grouped by uncovered tree.

    Class: NEW CAPABILITY. Reds under one bare group for every marked compile_tier test, which would pin all of
    them to one worker. The two-tree row shows a test that would carry two group names carries one naming both, and
    the one-tree row beside it carries that same name, because the tree it reaches is joined to the other by the
    two-tree test."""
    marks = collect_planted()
    compile_rows = {name: row for name, row in marks.items() if row[1]}
    assert compile_rows, "no compile_tier test was planted, so this test checks nothing"
    assert not [name for name, row in compile_rows.items() if _BARE in row[2]]
    assert {name: row[2] for name, row in compile_rows.items() if row[2]} == {
        "test_compile_uncovered_one_tree": (f"{_BARE}:synth_all_models+synth_sensitivity",),
        "test_compile_uncovered_two_trees": (f"{_BARE}:synth_all_models+synth_sensitivity",),
        "test_compile_body_builder": (f"{_BARE}:synth_multi_sim",),
        "test_compile_covered_rendered": (f"{_BARE}:rendered_synth_multi_sim",),
        "test_compile_covered_completed": (f"{_BARE}:synth_multi_sim",),
    }


def test_a_compile_tier_test_whose_shared_root_fixtures_are_all_covered_carries_no_group(collect_planted):
    """Property: a test whose only shared-root fixture is a `*_compiled` one carries no tree group; a
    `rendered_synth_*` holder carries that fixture's own group, and a test naming `synthetic_*_completed` itself joins
    that tree's group (their rows in `EXPECTED`).

    Class: NEW CAPABILITY. Reds when the `*_compiled` suffix is moved out of the covered set without the lock that
    covers it: adding the wiping fixture to the covered set makes `test_compile_uncovered_one_tree` lose its group,
    which the verdict table above rejects. This test itself checks only the `*_compiled` row; the other two are
    exercised by the verdict-table test, the grouped-rows test and the direct-naming test below, not by this one."""
    marks = collect_planted()
    covered = ["test_compile_only"]
    assert [marks[name] for name in covered] == [(True, True, ())] * len(covered)


def test_adding_a_wiping_fixture_to_the_covered_set_removes_its_group(collect_planted, monkeypatch):
    """Must-fire: the covered set is a claim that a lock exists, so widening it without one drops a needed group.

    Class: NEW CAPABILITY. The mutant adds `wipes_all_models` to the covered set, which is what declaring
    `synth_all_models_analysis` covered without building its lock would do. The compile_tier test that holds it then
    loses its tree group, so the verdict table reds. The set widened is the one `_is_covered_fixture` answers
    for (what a fixture reached through another fixture contributes), not `_COVERED_FIXTURE_NAMES`, which also
    decides what a test naming a session fixture itself joins."""
    real = suite_conftest._is_covered_fixture
    monkeypatch.setattr(suite_conftest, "_is_covered_fixture", lambda name: name == "wipes_all_models" or real(name))
    widened = collect_planted()
    assert EXPECTED["test_compile_uncovered_one_tree"][2] == (f"{_BARE}:synth_all_models+synth_sensitivity",)
    assert widened["test_compile_uncovered_one_tree"][2] == ()
    assert widened != EXPECTED


@pytest.mark.parametrize(
    ("mutant", "dropped"),
    [
        ("drops_fixture_arm", {"test_closure_wiper", "test_closure_names_tree_directly", "test_fetches_wiper_by_name"}),
        ("ignores_test_body_arm", {"test_body_calls_builder"}),
    ],
)
def test_a_derivation_that_drops_an_arm_gives_a_different_table(collect_planted, monkeypatch, mutant, dropped):
    """Must-fire: each arm is load-bearing, so removing it changes which tests are marked.

    Class: NEW CAPABILITY. The mutant is a wrapper over the real per-function scan that returns the empty set for
    the functions of one arm: fixtures (anything not named `test_*`) or test bodies. The verdict table then loses
    exactly the listed tests as marked, so the positive test above would red under it. Control: the unmutated table
    holds each listed test as marked."""
    real = suite_conftest._unoverridden_trees
    assert all(EXPECTED[name][0] for name in dropped), "control: the unmutated table must mark every listed test"

    def wrapper(func):
        """Stand in for the per-function scan: empty for the functions of the arm this mutant drops."""
        is_test = func.__name__.startswith("test_")
        skip = (mutant == "ignores_test_body_arm" and is_test) or (mutant == "drops_fixture_arm" and not is_test)
        return frozenset() if skip else real(func)

    monkeypatch.setattr(suite_conftest, "_unoverridden_trees", wrapper)
    mutated = collect_planted()
    assert mutated != EXPECTED
    assert {name for name in dropped if not mutated[name][0]} == dropped


def test_a_derivation_that_ignores_the_override_marks_the_overridden_tests(collect_planted, monkeypatch):
    """Must-fire: the override check is load-bearing, so a derivation that never sees it over-marks.

    Class: NEW CAPABILITY. The mutant answers False to "does this node name the override", which is the failure of
    a scan keyed on the builder token alone. The overridden controls then become marked, so the control test above
    would red under it."""
    monkeypatch.setattr(suite_conftest, "_mentions_override", lambda node: False)
    over_marked = collect_planted()
    assert over_marked["test_closure_override_by_setenv"][0] is True
    assert over_marked["test_body_overrides_first"][0] is True
    assert over_marked != EXPECTED


def test_the_retriever_set_is_the_catalogs_own_method_set():
    """Property: the builder names the scan recognises are exactly the catalog's `retrieve_*` methods plus the direct
    builder, so a method added to the catalog is recognised without an edit to the scan.

    Class: NEW CAPABILITY. Reds under a hard-coded name list that goes stale. Two-sided: the set is compared with
    what `dir(Local_TestCases)` reports, and the tree is spot-checked on two methods."""
    catalog = {name for name in dir(Local_TestCases) if name.startswith("retrieve_")}
    derived = suite_conftest._retriever_trees()
    assert catalog, "control: the catalog must expose at least one retrieve_* method"
    assert set(derived) - {_DIRECT} == catalog
    assert derived[_MULTI_SIM] == "synth_multi_sim"
    assert derived[_SENSITIVITY] == "synth_sensitivity"
    assert derived[_DIRECT] is None


def test_the_marker_is_registered(request):
    """Property: `shared_runs_root` is a registered marker, so `-m shared_runs_root` selects without a warning.

    Class: NEW CAPABILITY. Reds when the `pyproject.toml` markers table loses the entry."""
    registered = [line.split(":", 1)[0].strip() for line in request.config.getini("markers")]
    assert "shared_runs_root" in registered


# ---------------------------------------------------------------------------------------------------------------
# Reviewer tests (chunk 1a.1). Both plant modules in a throwaway rootdir and never execute a builder or a solver.
# ---------------------------------------------------------------------------------------------------------------

_HELPER_TESTS = """
import pytest


def _build_case():
    Local_TestCases.@MULTI@(start_from_scratch=False)


def _build_case_overridden(root):
    Local_TestCases.@MULTI@(start_from_scratch=False, runs_root_override=root)


def _builds_nothing():
    return "no builder is called here"


@pytest.fixture
def fixture_via_helper():
    _build_case()


def test_body_via_helper():
    _build_case()


def test_fixture_via_helper(fixture_via_helper):
    pass


def test_body_via_overridden_helper(tmp_path):
    _build_case_overridden(tmp_path)


def test_body_via_nonbuilding_helper():
    _builds_nothing()
"""

_CLASS_TESTS = """
import pytest


class TestIndentedSources:
    @pytest.fixture
    def class_wiper(self):
        Local_TestCases.@ALL@(start_from_scratch=True)

    def test_class_fixture(self, tritonswmm_cpu_compiled, class_wiper):
        pass

    def test_class_body(self, tritonswmm_cpu_compiled):
        Local_TestCases.@MULTI@(start_from_scratch=False)
"""


@pytest.fixture
def collect_text(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Return a function that collects one planted test module with the real hook and reads each node's marks."""
    monkeypatch.delenv("HHEMT_FORBID_COMPILE", raising=False)  # the hook arms a guard when this is "1"

    def collect(text: str) -> dict[str, tuple[bool, bool, tuple[str, ...]]]:
        """Collect `text` as the planted test module; map each test name to (marked, compile_tier, group names)."""
        pytester.makeconftest(_plant(_INNER_CONFTEST))
        pytester.makepyfile(test_planted=_plant(text))
        items, _ = pytester.inline_genitems()
        return {
            item.originalname: (
                item.get_closest_marker("shared_runs_root") is not None,
                item.get_closest_marker("compile_tier") is not None,
                tuple(str(m.args[0]) for m in item.iter_markers("xdist_group")),
            )
            for item in items
        }

    return collect


def test_a_builder_reached_through_a_same_module_helper_marks_the_test(collect_text):
    """Property: a test whose body, or a fixture in its closure, calls a module-level helper that calls a `retrieve_*`
    builder with no override is a shared-root test and carries the mark and the bare group.

    Class: REGRESSION, red on the tree as the coder left it. The input class is every test or fixture that reaches a
    builder through a helper function defined in (or imported by name into) its own module, which the source scan of
    the test function's own text cannot see. Measured reach at `38656ee7`: 7 collected tests in three files,
    `tests/test_synth_container_status.py` (3), `tests/test_in_rule_processing_snakefile.py` (2) and
    `tests/test_debug_flag_keep_previous_snapshots.py` (2), each unmarked and each writing into an exported override
    directory when run alone.

    Why it is required: the free tests' override directory is asserted EMPTY after the laptop run, and an unmarked
    test that builds through a helper fills it, so the 1a Definition of Done reds on exactly these tests.

    What kills it: resolving the called helper's own source (a `Name` call that resolves in the function's globals to a
    function) before declaring a test free. A second correct implementation, wrapping the builder at collection and
    recording which tests call it, passes unchanged because the assertion reads marks and not how they were derived."""
    marks = collect_text(_HELPER_TESTS)
    assert (marks["test_body_via_helper"], marks["test_fixture_via_helper"]) == ((True, False, (_BARE,)),) * 2


def test_a_helper_that_sets_the_override_or_builds_nothing_leaves_the_test_free(collect_text):
    """Control: a helper that passes `runs_root_override=` or calls no builder leaves the calling test unmarked.

    Class: NEW CAPABILITY, held green. Varies the property under suspicion (the helper reaches no UNOVERRIDDEN
    builder) rather than the token: the overridden helper names a builder. The plausible wrong implementation is a
    helper-following scan that marks any test calling a function whose body mentions a builder name. Green before and
    after the coder's next turn; the mutation that reds it is dropping the keyword check on the helper's call.

    Why it is required: over-marking pins a free test to the shared group and costs the laptop lane its parallelism.
    A second correct implementation that follows helpers by tracing the builder at runtime also leaves both free."""
    marks = collect_text(_HELPER_TESTS)
    assert (marks["test_body_via_overridden_helper"], marks["test_body_via_nonbuilding_helper"]) == (
        (False, False, ()),
    ) * 2


def test_class_based_fixtures_and_tests_carry_their_tree_group(collect_text, monkeypatch):
    """Property: a fixture or test defined as a class method, whose source is indented, is scanned like a module-level
    function, so a compile_tier test holding an uncovered class fixture gets that tree's group.

    Class: NEW CAPABILITY, held green. The wrong implementation is a scan that parses `inspect.getsource` output
    without dedenting it; every class-method source then fails to parse and the scan quietly returns no tree. Mutant,
    run in this test: `textwrap.dedent` replaced by the identity inside `tests/conftest.py`. At `38656ee7` that mutant
    drops the tree group from 8 collected tests in `tests/test_multi_model_integration.py` and leaves all 10 of the
    coder's tests green, because every planted function there is module-level.

    Why it is required: the loss is silent. Those 8 tests would run in no tree group and race the wipers of the same
    tree on a Rivanna chunk, with no failure anywhere. A second correct implementation, `ast.parse` over the file once
    and indexing function nodes by line number, passes the positive half unchanged."""
    expected = {
        "test_class_fixture": (True, True, (f"{_BARE}:synth_all_models",)),
        "test_class_body": (True, True, (f"{_BARE}:synth_multi_sim",)),
    }
    assert collect_text(_CLASS_TESTS) == expected
    monkeypatch.setattr(suite_conftest, "textwrap", type("NoDedent", (), {"dedent": staticmethod(lambda s: s)}))
    assert collect_text(_CLASS_TESTS) != expected


# ---------------------------------------------------------------------------------------------------------------
# Group naming by the session's own `-m` expression (the group rule), and where in the hook order it must happen.
# ---------------------------------------------------------------------------------------------------------------


def _trees_of(group: str) -> list[str]:
    """The trees a per-tree group name lists; the bare group lists none."""
    return group.split(":", 1)[1].split("+") if ":" in group else []


def test_the_session_expression_decides_between_tree_groups_and_the_bare_group(collect_planted):
    """Property: a session whose `-m` keeps a marked compile_tier test names groups by tree; one that keeps none gives
    every marked test outside compile_tier the bare group, and compile_tier tests get none.

    Class: NEW CAPABILITY. Three expressions are collected: none, one that selects both tiers, and the laptop's
    `not compile_tier`. The wrong implementation this reds under keys the naming on something else than the
    expression (the environment, or a constant), which gives the laptop expression tree groups or the both-tier
    expressions the bare group; the two mutant tests below run those. A second correct implementation, evaluating
    the expression with a hand-written parser instead of pytest's matcher, passes unchanged."""
    assert collect_planted() == EXPECTED
    assert collect_planted("compile_tier or not compile_tier") == EXPECTED
    assert collect_planted("not compile_tier") == LAPTOP_EXPECTED


def test_a_naming_that_ignores_the_expression_gives_the_laptop_tree_groups(collect_planted, monkeypatch):
    """Must-fire: answering "a compile_tier test is kept" whatever the expression gives the laptop tree groups.

    Class: NEW CAPABILITY. Control first: the unmutated laptop collection equals `LAPTOP_EXPECTED`, so a red below
    comes from the mutation and not from the planted project."""
    assert collect_planted("not compile_tier") == LAPTOP_EXPECTED
    monkeypatch.setattr(suite_conftest, "_session_keeps_compile_tier", lambda config, marked: True)
    mutated = collect_planted("not compile_tier")
    assert mutated != LAPTOP_EXPECTED
    assert mutated["test_closure_wiper"][2] == EXPECTED["test_closure_wiper"][2]


def test_a_naming_that_ignores_the_expression_gives_a_both_tier_session_the_bare_group(collect_planted, monkeypatch):
    """Must-fire: answering "no compile_tier test is kept" whatever the expression gives a both-tier session the
    bare group, which lets one tree's wipers and readers land on two workers.

    Class: NEW CAPABILITY. Control first: the unmutated both-tier collection equals `EXPECTED`."""
    assert collect_planted() == EXPECTED
    monkeypatch.setattr(suite_conftest, "_session_keeps_compile_tier", lambda config, marked: False)
    mutated = collect_planted()
    assert mutated != EXPECTED
    assert mutated["test_closure_wiper"][2] == (_BARE,)


def test_no_tree_is_in_two_groups_in_a_both_tier_session(collect_planted):
    """Property: every tree maps to exactly one group name across the marked tests of a both-tier session.

    Class: NEW CAPABILITY. The planted project holds a test reaching two trees and tests reaching one of them, so a
    naming built from each test's own tree set gives one tree two names (`...:a` and `...:a+b`). The check reads only
    the group names, so it does not depend on how the trees were found. Control: the same check on a table that
    does contain a tree in two names finds it, so the check can fail."""

    def trees_in_two_names(table) -> dict[str, set[str]]:
        """Map each tree that appears under more than one group name in `table` to those names."""
        names: dict[str, set[str]] = {}
        for _marked, _tier, groups in table.values():
            for group in groups:
                for tree in _trees_of(group):
                    names.setdefault(tree, set()).add(group)
        return {tree: found for tree, found in names.items() if len(found) > 1}

    assert trees_in_two_names(collect_planted()) == {}
    split = {"a": (True, True, (f"{_BARE}:t1",)), "b": (True, True, (f"{_BARE}:t1+t2",))}
    assert trees_in_two_names(split) == {"t1": {f"{_BARE}:t1", f"{_BARE}:t1+t2"}}


@pytest.mark.parametrize("override", [None, "/some/exported/override/dir"])
def test_the_group_naming_is_the_same_whether_or_not_the_override_is_exported(collect_planted, monkeypatch, override):
    """Property: the group tuples of the planted project are identical with `HHEMT_TEST_RUNS_ROOT_OVERRIDE` unset and
    exported, under no `-m` and under `not compile_tier`.

    Class: NEW CAPABILITY. The wrong implementation keys the naming on the environment, for example "per-tree groups
    unless the override is exported". Every worker and the controller collect the same items, but a harness chunk
    without isolation pops the variable while another process exports it, so such a naming would give one session two
    layouts and let one tree's wipers and readers land on two workers.

    Why it is required: DR-7 option 3 states that the naming reads no per-process state such as the environment, and no
    other test in this file exports the variable, so a naming that read it would pass all of them. Checkable against
    the code: `_session_keeps_compile_tier` and `_tree_group_names` read only the items and `config.option.markexpr`.

    What kills it: any read of the variable in the hook's naming path. The unset arm is the control, the exported arm
    the varied one; a mutant that also required the variable unset reds the exported arm only.

    A second correct implementation under which it passes: one that evaluates the expression with its own parser, as
    long as it still ignores the environment."""
    if override is None:
        monkeypatch.delenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", override)
    assert collect_planted() == EXPECTED
    assert collect_planted("not compile_tier") == LAPTOP_EXPECTED


_FIXTURE_PROJECT_CONFTEST = """
from tests.conftest import (  # noqa: F401  (the fixtures under test)
    _free_test_runs_root,
    _keep_free_tests_off_the_shared_runs_root,
    pytest_configure,
)
"""

_MARKED_WRITER_AND_UNMARKED_READER = """
import os
from pathlib import Path

import pytest

ENV = "HHEMT_TEST_RUNS_ROOT_OVERRIDE"


@pytest.mark.shared_runs_root
def test_marked_writes_under_the_slug_root(tmp_path):
    assert ENV not in os.environ
    tree = tmp_path / "synthetic_test_runs" / "tree"
    tree.mkdir(parents=True)
    (tree / "file.txt").write_text("x")


def test_unmarked_reads_only():
    assert Path(os.environ[ENV]).is_dir()
"""

_UNMARKED_WRITER = """
import os
from pathlib import Path


def test_unmarked_writes_into_its_runs_root():
    root = Path(os.environ["HHEMT_TEST_RUNS_ROOT_OVERRIDE"])
    (root / "synthetic_test_runs").mkdir()
"""

_SENTINEL_READERS = """
import os

import pytest

ENV = "HHEMT_TEST_RUNS_ROOT_OVERRIDE"


@pytest.mark.shared_runs_root
def test_marked_sees_the_sentinel():
    assert os.environ[ENV] == os.environ["SENTINEL"]


def test_unmarked_sees_the_sentinel():
    assert os.environ[ENV] == os.environ["SENTINEL"]
"""


@pytest.fixture
def run_fixture_project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Return a function that runs a planted project holding the free-test runs-root fixtures from `tests/conftest.py`.

    The outer test is itself an unmarked test, so the outer fixture has already exported the variable for it; the
    function clears it (or sets `start_env`) BEFORE the inner session starts, because the fixture under test decides
    from the value present at session start. The inner session runs in this process, so `monkeypatch` restores it."""

    def run(tests: str, start_env: str | None = None) -> pytest.RunResult:
        """Run the planted tests once; `start_env` is the value exported at session start, None for unset."""
        if start_env is None:
            monkeypatch.delenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", raising=False)
        else:
            monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", start_env)
        pytester.makeconftest(_FIXTURE_PROJECT_CONFTEST)
        pytester.makepyfile(test_inner=tests)
        return pytester.runpytest_inprocess("-p", "no:cacheprovider")

    return run


def test_a_marked_test_that_writes_under_the_runs_root_leaves_the_control_directory_empty(run_fixture_project):
    """Property: with the variable unset at session start, a marked stand-in that writes a tree sees the variable still
    unset and leaves the free-test directory empty, and an unmarked stand-in that only reads it passes.

    Class: NEW CAPABILITY. The wrong implementation exports the override for marked tests too, which sends the marked
    writer's tree into the control directory (the stand-in's own `ENV not in os.environ` assert reds first).

    Why it is required, checkable against the code: `_keep_free_tests_off_the_shared_runs_root` returns early for a
    test carrying `shared_runs_root`, and `_free_test_runs_root` asserts its directory empty at session end, so
    "all passed, no errors" is reachable only when the marked test wrote elsewhere.

    What kills it: dropping the marker check in the autouse fixture.

    A second correct implementation under which it passes: one that gives each marked test a private override of its
    own, if the stand-in's unset assertion were relaxed to "not the free-test directory"."""
    result = run_fixture_project(_MARKED_WRITER_AND_UNMARKED_READER)
    result.assert_outcomes(passed=2)


def test_an_unmarked_test_that_writes_into_its_runs_root_is_failed_naming_what_it_wrote(run_fixture_project):
    """Property: an unmarked stand-in that writes into the free-test directory errors, and the message names the
    node id and the entry it wrote.

    Class: NEW CAPABILITY, the must-fire pair of the test above: the same fixtures and the same directory, the writer
    unmarked. Control: the previous test passes with the same project.

    Why it is required, checkable against the code: a mis-derived free test would otherwise write into a directory
    nobody reads, and the laptop lane would stay green while the test still touches a shared tree.

    What kills it: removing the before/after comparison in the autouse fixture. The session-end assertion in
    `_free_test_runs_root` is redundant with it for a write made during a test and is not exercised here.

    A second correct implementation under which it passes: one that makes the directory read-only, so the write
    itself fails inside the test."""
    result = run_fixture_project(_UNMARKED_WRITER)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*not marked shared_runs_root but wrote*synthetic_test_runs*"])


def test_a_variable_exported_before_the_session_is_left_exactly_as_exported(run_fixture_project, monkeypatch, tmp_path):
    """Property: when the variable is exported at session start, a marked and an unmarked stand-in both see the
    sentinel value, unchanged.

    Class: NEW CAPABILITY. The wrong implementation redirects it for unmarked tests or unsets it for marked ones,
    which breaks a harness chunk's isolation root or a developer's own export.

    Why it is required, checkable against the code: `_free_test_runs_root` yields None when the stashed start value
    is not the unset marker, and the autouse fixture then does nothing at all.

    What kills it: redirecting unmarked tests, or unsetting for marked ones, when the variable was exported. Reading
    the variable at fixture time instead of in `pytest_configure` gives the same value and is not distinguished here.

    A second correct implementation under which it passes: one that never touches an exported variable."""
    sentinel = str(tmp_path / "sentinel")
    monkeypatch.setenv("SENTINEL", sentinel)
    result = run_fixture_project(_SENTINEL_READERS, start_env=sentinel)
    result.assert_outcomes(passed=2)


_RUNNABLE_CONFTEST = """
import pytest

@IMPORT_HOOK@


class Local_TestCases:
    @staticmethod
    def @MULTI@(**kwargs):
        return None


@pytest.fixture
def tritonswmm_cpu_compiled():
    return None


@pytest.fixture
def uses_the_tree():
    Local_TestCases.@MULTI@(start_from_scratch=False)
"""

_RUNNABLE_TESTS = """
def test_with_compile(tritonswmm_cpu_compiled, uses_the_tree):
    pass


def test_without_compile(uses_the_tree):
    pass
"""

_TRYFIRST_IMPORT = "from tests.conftest import pytest_collection_modifyitems  # noqa: F401"
_TRYLAST_DEFINE = """
import tests.conftest as suite_conftest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    suite_conftest.pytest_collection_modifyitems(config, items)
"""


@pytest.fixture
def run_under_loadgroup(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Return a function that runs the planted project at `-n 2 --dist loadgroup` and reads the group suffix xdist
    appends to each node id; the planted tests only `pass`, and the builder is a local no-op stand-in."""
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(repo_root), str(repo_root / "src")]))
    monkeypatch.setenv("HHEMT_DISABLE_RUN_DIR_REAPER", "1")
    monkeypatch.delenv("HHEMT_FORBID_COMPILE", raising=False)

    def run(hook_source: str, *extra: str) -> dict[str, str | None]:
        """Run once; map each planted test name to its xdist group suffix, or None when its node id has none."""
        pytester.makeconftest(_plant(_RUNNABLE_CONFTEST).replace("@IMPORT_HOOK@", hook_source))
        pytester.makepyfile(test_planted=_RUNNABLE_TESTS)
        result = pytester.runpytest_subprocess("-n", "2", "--dist", "loadgroup", "-v", "-p", "no:cacheprovider", *extra)
        found = re.findall(r"test_planted\.py::(\w+)(?:@(\S+))?", result.stdout.str())
        return {name: group or None for name, group in found}

    return run


def test_a_group_reaches_the_node_id_only_when_the_naming_runs_in_the_tryfirst_hook(run_under_loadgroup):
    """Property: at `-n 2 --dist loadgroup` a both-tier session suffixes both planted node ids with
    `@shared_runs_root:synth_multi_sim`, and the laptop expression suffixes the non-compile one with
    `@shared_runs_root`.

    Class: REGRESSION against placing the naming in a later hook. pytest-xdist's worker hook appends each node's
    `xdist_group` to its node id before a `trylast` or plain hook of ours runs, so a group added there never reaches
    `--dist loadgroup`. Control: the same project with the imported (`tryfirst`) hook gives the suffixes. Must-fire:
    the same project with the hook re-registered `trylast` gives no suffix at all.

    Why it is required: a late group looks right in `item.iter_markers` and does nothing, so the unit tests above
    pass while no run is ever grouped. A second correct implementation, computing the groups in a
    `pytest_collection` wrapper, passes: the assertion reads the node ids and not the hook."""
    tree = "shared_runs_root:synth_multi_sim"
    assert run_under_loadgroup(_TRYFIRST_IMPORT) == {"test_with_compile": tree, "test_without_compile": tree}
    laptop = run_under_loadgroup(_TRYFIRST_IMPORT, "-m", "not compile_tier")
    assert laptop == {"test_without_compile": "shared_runs_root"}
    assert run_under_loadgroup(_TRYLAST_DEFINE) == {"test_with_compile": None, "test_without_compile": None}


_DUMP_PLUGIN = """
import json
import os


def pytest_collection_finish(session):
    rows = [
        {
            "id": item.nodeid,
            "marked": item.get_closest_marker("shared_runs_root") is not None,
            "tier": item.get_closest_marker("compile_tier") is not None,
            "groups": [str(m.args[0]) for m in item.iter_markers("xdist_group")],
        }
        for item in session.items
    ]
    with open(os.environ["GROUP_DUMP"], "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
"""


def test_in_the_real_suite_no_tree_is_in_two_groups_and_every_marked_test_outside_the_tier_is_grouped(tmp_path):
    """Property: collecting this repository's own tests with no `-m`, every tree appears in exactly one group name,
    no marked test outside compile_tier is ungrouped, no compile_tier test carries the bare group, and an unmarked
    test carries no group.

    Class: NEW CAPABILITY, over the real marked set (about 236 tests at the time of writing) rather than a planted
    project. The wrong implementation is a naming built from each test's own tree set, which gives
    `synth_multi_sim` two names. Collection only: no test body runs, so no builder or solver is reached. The check
    reads group names alone, so it is independent of the scan that found the trees.

    What kills it: restoring the per-test naming. Control: the collected table must hold marked tests and at least
    one per-tree group, so an empty collection cannot pass it."""
    import json
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "group_dump_plugin.py").write_text(_DUMP_PLUGIN, encoding="utf-8")
    dump = tmp_path / "groups.json"
    env = {
        **os.environ,
        "GROUP_DUMP": str(dump),
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(repo_root / "src")]),
        "HHEMT_DISABLE_RUN_DIR_REAPER": "1",
    }
    env.pop("HHEMT_FORBID_COMPILE", None)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-p", "group_dump_plugin"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    rows = json.loads(dump.read_text(encoding="utf-8"))
    marked = [row for row in rows if row["marked"]]
    assert marked and any(":" in group for row in marked for group in row["groups"])
    names: dict[str, set[str]] = {}
    for row in marked:
        for group in row["groups"]:
            for tree in _trees_of(group):
                names.setdefault(tree, set()).add(group)
    assert {tree: found for tree, found in names.items() if len(found) > 1} == {}
    assert [row["id"] for row in marked if not row["tier"] and not row["groups"]] == []
    assert [row["id"] for row in marked if row["tier"] and _BARE in row["groups"]] == []
    assert [row["id"] for row in rows if not row["marked"] and row["groups"]] == []


_RENDERED_HOLDERS = """
import pytest


def test_rendered_and_wiper(rendered_synth_multi_sim, wipes_all_models):
    pass


def test_rendered_only(rendered_synth_multi_sim):
    pass


@pytest.mark.parametrize("fixture_name", ["rendered_synth_multi_sim"])
def test_rendered_by_name(request, fixture_name):
    request.@GFV@(fixture_name)
"""


def test_every_holder_of_a_rendered_fixture_shares_one_group_with_the_tree_a_sibling_reaches(collect_text):
    """Property: a test holding a rendered fixture together with an uncovered shared-tree builder, a test holding only
    that fixture, and a test fetching it by a parametrized name all carry the same single group, which names the
    fixture and the tree.

    Class: NEW CAPABILITY. Two wrong implementations: one names a rendered holder by its fixture alone, so the
    fixture-only holder keeps a group of its own and the tree's wiper can run on another worker than a holder that
    also reaches the tree; the other reads only the declared fixture names and so skips the holder that fetches the
    fixture by a parametrized name.

    Why it is required, checkable against the code: `pytest_collection_modifyitems` joins each test's rendered
    fixtures (`_rendered_fixtures`, which follows a parametrized fetch through `_fixture_funcs`) and its tree tokens
    into one component in `_tree_group_names`, and the real suite has four such tests (the from-DOI round trips),
    which no other test in this file plants.

    What kills it: leaving the rendered fixtures out of the joined tokens (the fixture-only and by-name holders then
    carry `shared_runs_root:rendered_synth_multi_sim`), or dropping the parametrized-fetch closure in
    `_fixture_funcs` (the by-name holder then carries no group). Each is shown red by a scratch copy of
    `tests/conftest.py`, outside the worktree.

    A second correct implementation under which it passes: one that gives rendered fixtures and trees one shared
    namespace of tokens and names each component by its sorted members."""
    marks = collect_text(_RENDERED_HOLDERS)
    group = f"{_BARE}:rendered_synth_multi_sim+synth_all_models"
    assert {name: row[2] for name, row in marks.items()} == {
        "test_rendered_and_wiper": (group,),
        "test_rendered_only": (group,),
        "test_rendered_by_name": (group,),
    }


# ---------------------------------------------------------------------------------------------------------------
# Which override a from-scratch build may wipe under (`_select_runs_root` in the case builder).
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def root_selection(tmp_path, monkeypatch):
    """Return a function that selects a runs root with the environment, the recorded process-level value and the
    scratch-directory factory under the test's control, plus the directories it set up.

    `harness` plays the node-local isolation root the chunk wrapper exports for the whole process; `scratch` is where
    the stand-in `mkdtemp` puts private roots, so nothing is created in the system temp dir."""
    from tests.fixtures import test_case_builder as builder

    harness = tmp_path / "harness"
    scratch = tmp_path / "scratch"
    harness.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(builder.tempfile, "mkdtemp", lambda prefix="": str(scratch / f"{prefix}private"))

    def select(*, env: Path | None, process_level: Path | None, **kwargs) -> Path:
        """Select a root: `env` is the variable's value now, `process_level` the value recorded at import."""
        if env is None:
            monkeypatch.delenv(builder._TEST_RUNS_ROOT_OVERRIDE_ENV, raising=False)
        else:
            monkeypatch.setenv(builder._TEST_RUNS_ROOT_OVERRIDE_ENV, str(env))
        monkeypatch.setattr(
            builder, "_PROCESS_LEVEL_RUNS_ROOT_OVERRIDE", None if process_level is None else str(process_level)
        )
        kwargs.setdefault("runs_root_override", None)
        kwargs.setdefault("share_scratch_root", False)
        return builder._select_runs_root(tmp_path / "slug", **kwargs)

    select.harness = harness
    select.scratch = scratch
    return select


def test_a_from_scratch_build_under_the_harness_level_override_does_not_use_the_shared_tree(root_selection):
    """Property: with the variable exported for the whole process, a `start_from_scratch=True` build is sent to a
    private root, so the `fast_rmtree` of its system directory cannot delete a tree other workers read.

    Class: REGRESSION, red on the builder before this change, which wiped under any override. The observed failure
    was workers of one chunk losing `synth_all_models/weather_events_to_simulate.csv` in setup. The test builds the
    marker file the shared tree would hold and checks the selected system directory is another directory.

    Why it is required, checkable against the code: `_select_runs_root` privatises when the environment value equals
    `_PROCESS_LEVEL_RUNS_ROOT_OVERRIDE`; the constructor wipes `runs_root / analysis_name`.

    What kills it: dropping the `harness_level` arm. Control: the same call without `start_from_scratch` returns the
    shared root, so the privatisation is the from-scratch branch and not a blanket redirect.

    A second correct implementation under which it passes: one that makes the wipe itself refuse a tree under the
    harness-level override."""
    harness = root_selection.harness
    marker = harness / "synthetic_test_runs" / "case" / "marker.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("shared")
    shared = harness / "synthetic_test_runs"
    kept = root_selection(env=harness, process_level=harness, start_from_scratch=False)
    assert kept == shared, "control: a cached build still uses the shared root"
    chosen = root_selection(env=harness, process_level=harness, start_from_scratch=True)
    assert chosen != shared and (chosen / "case") != (shared / "case")
    assert marker.read_text() == "shared"


def test_the_share_scratch_opt_out_still_builds_under_the_harness_level_override(root_selection):
    """Property: `share_scratch_root=True` keeps its meaning under the harness-level override: the build uses the
    shared root, because a later reader must see what it produces.

    Class: NEW CAPABILITY. The wrong implementation privatises whatever the flag says. Control: the previous test,
    which differs only in the flag.

    Why it is required, checkable against the code: the condition in `_select_runs_root` includes
    `not share_scratch_root`.

    What kills it: removing that term.

    A second correct implementation under which it passes: one that expresses the opt-out as a separate branch."""
    harness = root_selection.harness
    chosen = root_selection(env=harness, process_level=harness, start_from_scratch=True, share_scratch_root=True)
    assert chosen == harness / "synthetic_test_runs"


def test_an_override_a_fixture_set_around_one_build_is_used_as_given(root_selection, tmp_path):
    """Property: a value that is not the process-level one (a fixture's per-build directory, or an explicit
    `runs_root_override=`) is the root used, even for a from-scratch build, so later readers find the tree there.

    Class: NEW CAPABILITY. The wrong implementation privatises every environment override, which would send the
    builds the conftest fixtures make under `_runs_root_override_env` to a different directory than the one their
    readers use.

    Why it is required, checkable against the code: the fixture-level and explicit cases are the ones
    `_select_runs_root` leaves alone, and the conftest fixtures that use `_runs_root_override_env` depend on it.

    What kills it: treating any environment value as harness-level. Control: both arms run with a process-level value
    set, so only the difference between the values decides.

    A second correct implementation under which it passes: one that records a flag when the harness exports."""
    harness = root_selection.harness
    fixture_dir = tmp_path / "fixture_level"
    by_fixture = root_selection(env=fixture_dir, process_level=harness, start_from_scratch=True)
    assert by_fixture == fixture_dir / "synthetic_test_runs"
    explicit = root_selection(
        env=harness, process_level=harness, start_from_scratch=True, runs_root_override=tmp_path / "explicit"
    )
    assert explicit == tmp_path / "explicit" / "synthetic_test_runs"


def test_with_no_override_the_root_selection_is_the_old_one(root_selection):
    """Property: with the variable unset, a from-scratch build gets a private root and any other build the slug root.

    Class: NEW CAPABILITY, held as a control for the three tests above: the unchanged behaviour stays unchanged.

    Why it is required, checkable against the code: this is the pre-existing `effective is None` branch.

    What kills it: changing what an unset variable selects.

    A second correct implementation under which it passes: any that keeps the same two outcomes."""
    private = root_selection(env=None, process_level=None, start_from_scratch=True)
    assert private == root_selection.scratch / "hhemt-scratch-private" / "synthetic_test_runs"
    slug = root_selection(env=None, process_level=None, start_from_scratch=False)
    assert slug.name == "slug"


_DIRECT_COVERED_TESTS = """
import pytest


@pytest.fixture
def synthetic_multisim_completed(tritonswmm_cpu_compiled):
    Local_TestCases.@MULTI@(start_from_scratch=False)


@pytest.fixture
def synthetic_sensitivity_completed(tritonswmm_cpu_compiled):
    Local_TestCases.@SENS@(start_from_scratch=False)


@pytest.fixture
def synthetic_multisim_completed_isolated(synthetic_multisim_completed):
    return "a private copy"


@pytest.fixture
def synthetic_sensitivity_completed_isolated(synthetic_sensitivity_completed):
    return "a private copy"


def test_names_multisim(synthetic_multisim_completed):
    pass


def test_names_sensitivity(synthetic_sensitivity_completed):
    pass


def test_names_multisim_and_an_uncovered_builder(synthetic_multisim_completed, wipes_all_models):
    pass


def test_multisim_through_a_clone_only(synthetic_multisim_completed_isolated):
    pass


def test_sensitivity_through_a_clone_only(synthetic_sensitivity_completed_isolated):
    pass


def test_names_the_clone_and_the_fixture(synthetic_multisim_completed, synthetic_multisim_completed_isolated):
    pass
"""


def test_a_test_naming_a_covered_session_fixture_joins_the_trees_group_and_a_clone_only_test_does_not(collect_text):
    """Property: a compile_tier test that names `synthetic_multisim_completed` or `synthetic_sensitivity_completed` in
    its own parameters carries that tree's group; one that reaches the fixture only through a `*_isolated` clone
    carries none; one that names a covered fixture and an uncovered builder carries ONE name for both.

    Class: NEW CAPABILITY, measured on the cluster: a test that snapshots the session tree it names ran on another
    worker than the group whose members write workflow files into that tree, and failed in 5 of 5 runs, passing in
    2 of 2 once it joined the group. The covered exemption is right for the BUILD race (`_session_tree_lock`), not
    for a test that reads the tree in place.

    Why it is required, checkable against the code: `_directly_named_covered_trees` reads the fixtures named in the
    item's own `argnames` and adds their trees to a compile_tier test's tokens in `pytest_collection_modifyitems`.

    What kills it: the rule before this change (the two direct tests carry no group), a rule that groups every test
    holding a covered fixture in its closure (the clone-only tests then carry a group), and a rule keyed on the
    multisim fixture alone (the sensitivity test then carries none). Each is shown red by a scratch copy of the whole
    worktree.

    A second correct implementation under which it passes: one that reads the item's signature with
    `inspect.signature` instead of `_fixtureinfo.argnames`."""
    marks = collect_text(_DIRECT_COVERED_TESTS)
    joined = f"{_BARE}:synth_all_models+synth_multi_sim"
    assert {name: row[2] for name, row in marks.items()} == {
        "test_names_multisim": (joined,),
        "test_names_sensitivity": (f"{_BARE}:synth_sensitivity",),
        "test_names_multisim_and_an_uncovered_builder": (joined,),
        "test_multisim_through_a_clone_only": (),
        "test_sensitivity_through_a_clone_only": (),
        "test_names_the_clone_and_the_fixture": (joined,),
    }
    assert all(row[1] for row in marks.values()), "control: every planted test is compile_tier"

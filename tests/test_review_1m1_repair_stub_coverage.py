"""Reviewer test for the 1m.1 repair: the partition-caps test's no-op list names every validator it must silence.

The repaired test (`test_preflight_validate_checks_the_analysis_request_against_its_partition_caps`) runs the
real `preflight_validate` with each OTHER validator replaced by a no-op, from a hand-written tuple of names.
That tuple drifted once (chunk 1m.1 added `_validate_sensitivity_file_readable` and the tuple did not follow),
so this file derives the tuple's required content from `preflight_validate` itself.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import hhemt.validation as validation

_STUBBED_TEST = "test_preflight_validate_checks_the_analysis_request_against_its_partition_caps"
_TEST_FILE = Path(__file__).with_name("test_unread_settings_rulings.py")
# The one validator the stubbed test is ABOUT; it must stay real.
_UNDER_TEST = "_validate_partition_spec_caps_for_analysis"


def _validators_called_by_preflight(preflight_source: str) -> set[str]:
    """Names of the module-level `_validate_*` functions `preflight_validate` calls directly."""
    fn = ast.parse(textwrap.dedent(preflight_source)).body[0]
    return {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("_validate_")
    }


def _names_in_stub_tuples(test_source: str, test_name: str) -> set[str]:
    """Every string constant inside a tuple literal in the named test (the `for name in (...)` lists)."""
    fn = next(n for n in ast.walk(ast.parse(test_source)) if isinstance(n, ast.FunctionDef) and n.name == test_name)
    return {
        elt.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Tuple)
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }


def _unstubbed(preflight_source: str, test_source: str) -> set[str]:
    return (
        _validators_called_by_preflight(preflight_source)
        - {_UNDER_TEST}
        - _names_in_stub_tuples(test_source, _STUBBED_TEST)
    )


def test_the_partition_caps_test_silences_every_validator_preflight_calls_except_its_subject():
    """Property: no `_validate_*` call in `preflight_validate` other than the subject is left real in the stubbed test.

    Class: every direct `_validate_*` callee of `preflight_validate` (derived from its source, 13 today),
    against the string names in the stubbed test's tuples. Why it is required: a validator left real runs
    against the test's `SimpleNamespace` stub config and raises AttributeError, which is how CI red 5 arose
    (erratum E-13). What kills it: adding a `_validate_*` call to `preflight_validate` without adding its
    name to the test's tuple, or dropping a name from the tuple. A second correct implementation derives the
    tuple from `preflight_validate`'s source inside the stubbed test; this test would still pass.
    """
    missing = _unstubbed(inspect.getsource(validation.preflight_validate), _TEST_FILE.read_text())
    assert missing == set()


def test_the_coverage_derivation_fires_on_a_missing_name_and_stays_quiet_on_a_reordered_tuple():
    """Property: `_unstubbed` reports exactly the dropped name and nothing for a differently arranged complete tuple.

    Class: the repaired test's source with (a) one name deleted from the first position, (b) one name deleted
    from the last position, (c) the same names re-ordered across two tuples. Why it is required: a derivation
    that passes on the live file proves nothing until it is shown to fire (the must-fire control the
    first test's zero owes). What kills it: a derivation that reads only the first tuple element, or only
    the first tuple. A second correct implementation compares sets from `ast` over a different walk order;
    this test would still pass.
    """
    preflight = inspect.getsource(validation.preflight_validate)
    source = _TEST_FILE.read_text()
    first = '"_validate_sensitivity_file_readable",\n'
    last = '"_validate_timeseries_keep",\n'
    assert first in source and last in source
    assert _unstubbed(preflight, source.replace(first, "", 1)) == {"_validate_sensitivity_file_readable"}
    assert _unstubbed(preflight, source.replace(last, "", 1)) == {"_validate_timeseries_keep"}
    reordered = source.replace(first, "", 1).replace(last, last + "    " + first, 1)
    assert _unstubbed(preflight, reordered) == set()

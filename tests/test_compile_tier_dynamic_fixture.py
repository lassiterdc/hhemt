"""A test that fetches a compile-reaching fixture by a parametrized name must carry `compile_tier`.

`tests/conftest.py::pytest_collection_modifyitems` marks a test `compile_tier` when `tritonswmm_cpu_compiled` is in
its fixture closure. A test that calls `request.getfixturevalue(fixture_name)` with a `parametrize` value does not
list that fixture in its closure, so without the per-node extension it would escape the mark and run on the laptop.

Each case plants a small test module in a throwaway pytest rootdir, collects it with the real hook, and reads the
marks the hook gave each node. The planted modules define their own stand-in fixtures, so no solver is compiled
or run, and the hook is imported from `tests/conftest.py` rather than copied.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

_INNER_CONFTEST = '''
import pytest

from tests.conftest import pytest_collection_modifyitems  # noqa: F401  (the hook under test)


@pytest.fixture
def tritonswmm_cpu_compiled():
    return "stand-in for the coupled compile"


@pytest.fixture
def rendered_like(tritonswmm_cpu_compiled):
    """A fixture whose closure holds the compile, as every rendered_synth_* fixture does."""
    return "rendered"


@pytest.fixture
def cheap():
    return "cheap"
'''

_INNER_TESTS = """
import pytest


@pytest.mark.parametrize("fixture_name", ["rendered_like"])
def test_fetches_rendered_by_name(request, fixture_name):
    request.getfixturevalue(fixture_name)


@pytest.mark.parametrize("fixture_name", ["cheap"])
def test_fetches_cheap_by_name(request, fixture_name):
    request.getfixturevalue(fixture_name)


@pytest.mark.parametrize("label", ["not-a-fixture-name"])
def test_string_value_naming_no_fixture(label):
    pass


@pytest.mark.parametrize("count", [3])
def test_non_string_value(count):
    pass


def test_requests_the_compile_directly(tritonswmm_cpu_compiled):
    pass


def test_requests_nothing():
    pass
"""


@pytest.fixture
def planted_marks(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Collect the planted module with the real hook; map each planted test to whether it carries `compile_tier`."""
    monkeypatch.delenv("HHEMT_FORBID_COMPILE", raising=False)  # the hook arms a guard when this is "1"
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(test_planted=_INNER_TESTS)
    items, _ = pytester.inline_genitems()
    return {item.originalname: item.get_closest_marker("compile_tier") is not None for item in items}


def test_parametrized_getfixturevalue_of_a_rendered_fixture_is_marked(request):
    # Fetched by name on purpose. The suite partitioner (`src/hhemt/suite/partition.py::file_trees`) refuses a file
    # whose text contains `getfixturevalue` unless a string literal in the file names a fixture it can resolve. Every
    # other occurrence here is a docstring, a test name or planted source text, none of which is such a literal. This
    # lookup names `planted_marks`, defined above, which reaches no analysis tree.
    planted_marks = request.getfixturevalue("planted_marks")
    assert planted_marks["test_fetches_rendered_by_name"] is True


@pytest.mark.parametrize(
    "name",
    [
        "test_fetches_cheap_by_name",
        "test_string_value_naming_no_fixture",
        "test_non_string_value",
        "test_requests_nothing",
    ],
)
def test_controls_that_do_not_reach_the_compile_stay_unmarked(planted_marks, name):
    assert planted_marks[name] is False


def test_a_direct_request_of_the_compile_is_still_marked(planted_marks):
    assert planted_marks["test_requests_the_compile_directly"] is True

"""Reviewer test: the absent-source warning test takes its expected text from the stdlib formatter.

Property: the test that checks the formatted text of the aggregated absent-source warning gives the
same verdict whichever formatter an earlier test left installed on the process, and running it leaves
the process-wide warnings state as it found it.

Class: every formatter a prior import can install on ``warnings.formatwarning`` (the stdlib one, and
the one-line replacement that importing ``plotly.tools`` installs), crossed with the process-wide
state the test could disturb: the formatter, the display hook, the default action, the live filter
list, the base filter list the module-level import binds, and the once-registry.

Why it is required: the test passes alone and failed after any earlier test that imported plotly,
because the expected text came from the process formatter. A repair that still reads the process
formatter passes the test in its own file and fails in a session, so only a node that installs the
replacement formatter itself reproduces the order dependence. A repair that resets the process
formatter to get the right text would pass too, and leaves the process changed.

What kills it: reading the expected text from ``warnings.formatwarning``; reloading or replacing
``warnings`` in place; any step that changes the formatter, display hook, default action or filter
set of the process.

A second correct implementation that still passes: any route to the stdlib text that touches no
process-wide state, including a hard-coded expected string.

Every arm writes only ``tmp_path`` files; none compiles or runs a solver.
"""

from __future__ import annotations

import _warnings
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

# The stdlib formatter is read before plotly is imported, because importing ``plotly.tools`` replaces it.
_STDLIB_FORMATTER = warnings.formatwarning
_PLOTLY_TOOLS = pytest.importorskip("plotly.tools")
_TEST_FILE = Path(__file__).with_name("test_bundle_harvest_absent_warning.py")


def _reviewed_test() -> Any:
    spec = importlib.util.spec_from_file_location("_reviewed_absent_warning_test", _TEST_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.test_the_warning_carries_no_python_source_file


@pytest.fixture(params=["stdlib", "one_line"])
def installed_formatter(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Install the formatter named by the param for the duration of the node."""
    chosen = _PLOTLY_TOOLS.warning_on_one_line if request.param == "one_line" else _STDLIB_FORMATTER
    monkeypatch.setattr(warnings, "formatwarning", chosen)
    return str(request.param)


def _state() -> dict[str, Any]:
    return {
        "formatwarning": warnings.formatwarning,
        "showwarning": warnings.showwarning,
        "defaultaction": warnings.defaultaction,
        "filters": list(warnings.filters),
        "base_filters": list(_warnings.filters),
        "once": dict(warnings._onceregistry),
        "warnoptions": list(sys.warnoptions),
    }


def test_the_text_check_passes_whichever_formatter_is_installed(installed_formatter: str, tmp_path: Path) -> None:
    """CLASS: any formatter an earlier import left on the process."""
    _reviewed_test()(tmp_path)


def test_the_text_check_leaves_the_process_warning_state_as_found(installed_formatter: str, tmp_path: Path) -> None:
    """CLASS: any installed formatter; the formatter, hook, default action, filters and registry."""
    before = _state()
    try:
        _reviewed_test()(tmp_path)
    except AssertionError:
        pass  # the text check's own verdict is the first test's concern; only the state is judged here
    after = _state()
    assert after == before


def test_the_text_check_keeps_the_filter_multiset_when_options_name_an_existing_filter(
    installed_formatter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an interpreter option naming a filter the base list already holds, the count and the
    set of filters are unchanged, which is the residual the repair states; order is not asserted."""
    option = "ignore::DeprecationWarning"
    existing = ("ignore", None, DeprecationWarning, None, 0)
    monkeypatch.setattr(sys, "warnoptions", [option])
    base = _warnings.filters
    saved = list(base)
    try:
        base.insert(len(base), existing)
        before = list(base)
        try:
            _reviewed_test()(tmp_path)
        except AssertionError:
            pass  # the text check's own verdict is the first test's concern; only the filters are judged here
        after = list(_warnings.filters)
        assert len(after) == len(before)
        assert sorted(map(repr, after)) == sorted(map(repr, before))
    finally:
        base[:] = saved

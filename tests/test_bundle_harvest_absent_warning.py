"""F-D W1 (DR-4): a harvest with absent declared sources issues ONE aggregated warning that
names no source file, so no entry point echoes an unrelated source line under it.

`_harvest_sources` used to call `warnings.warn(..., stacklevel=2)` once per absent path.
`stacklevel=2` attributes the warning to `_harvest_sources`' caller, and the default
formatter then prints that caller's SOURCE LINE under the message -- `entries, declared_absent,
metadata_only_stores = _harvest_sources(` on a programmatic call, Typer's `return
callback(**use_params)` on the CLI. The echoed line is unrelated to the cause and is the
defect the observer reported (not the stack depth), so the fix is a warning with no source
file at all: `warnings.warn_explicit(..., filename="hhemt bundle emit", lineno=0)`.

Every test writes only `tmp_path` files; none compiles or runs a solver.
"""

from __future__ import annotations

import importlib.util
import linecache
import re
import warnings
from pathlib import Path

import pytest

from hhemt.bundle._emit import _harvest_sources

_ABSENT = ("_status/_du.json", "eda/a.zarr", "eda/b.zarr")


def _analysis(tmp_path: Path) -> tuple[Path, list[Path]]:
    """An analysis dir whose three declared sources are absent and one is present."""
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "eda").mkdir(parents=True)
    (analysis_dir / "_status").mkdir()
    present = analysis_dir / "eda" / "present.txt"
    present.write_text("x", encoding="utf-8")
    return analysis_dir, [present, *(analysis_dir / rel for rel in _ABSENT)]


def _harvest_recording(analysis_dir: Path, declared: list[Path]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _harvest_sources({"r1": declared, "r2": declared[:2]}, analysis_dir)
    return result, [w for w in caught if issubclass(w.category, UserWarning)]


def test_three_absent_sources_issue_one_warning_naming_all_three(tmp_path: Path) -> None:
    """CLASS: any harvest with k >= 2 absent declared sources.

    Pre-change: k warnings, one per path. Post-change: exactly one, naming every path and
    the manifest key where the same list is recorded.
    """
    analysis_dir, declared = _analysis(tmp_path)
    (_entries, absent, _shallow), caught = _harvest_recording(analysis_dir, declared)

    assert sorted(absent) == sorted(_ABSENT), "the recorded list is unchanged by the fix"
    assert len(caught) == 1, [str(w.message) for w in caught]
    message = str(caught[0].message)
    for rel in _ABSENT:
        assert rel in message
    assert "declared_sources_absent" in message
    assert re.search(r"(?<!\d)3(?!\d)", message), "the count 3 must appear as a bare integer token"


def test_the_warning_carries_no_python_source_file(tmp_path: Path) -> None:
    """CLASS: the formatter has no source line to echo.

    The default formatter reads `linecache.getline(filename, lineno)`. A filename that does
    not end `.py` names no file there, and `lineno=0` is never a line; both are asserted,
    and the formatted text is checked directly rather than inferred from the attributes.
    """
    analysis_dir, declared = _analysis(tmp_path)
    _result, caught = _harvest_recording(analysis_dir, declared)
    [w] = caught

    assert not str(w.filename).endswith(".py"), w.filename
    assert w.filename == "hhemt bundle emit"
    assert w.lineno == 0
    assert linecache.getline(str(w.filename), w.lineno) == ""
    # Importing plotly replaces the process-wide `warnings.formatwarning`, and the replacement wraps the
    # message in blank lines, so the text below is taken from a fresh copy of the stdlib module (public
    # importlib.util route) rather than from whatever formatter an earlier test left installed. The fresh
    # copy is not isolated: it shares the C-level filter list and the once-registry with the process, and
    # executing it re-runs the module's processing of `sys.warnoptions`. For an interpreter `-W` flag (or
    # PYTHONWARNINGS) that removes an equal existing filter and re-inserts it at the front, so the filter count
    # is unchanged, precedence against a later-inserted filter could differ, and the filter-mutation counter
    # moves, which makes a once-per-location warning already shown appear again. Under pytest `sys.warnoptions`
    # is empty unless Python itself is started that way.
    spec = importlib.util.find_spec("warnings")
    pristine_warnings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pristine_warnings)
    formatted = pristine_warnings.formatwarning(w.message, w.category, w.filename, w.lineno)
    assert formatted.startswith("hhemt bundle emit:0: UserWarning: ")
    assert formatted.count("\n") == 1, "message line only; no echoed source line follows it"


def test_a_clean_harvest_issues_no_warning(tmp_path: Path) -> None:
    """Differently-positioned satisfying input: nothing absent, so nothing is said."""
    analysis_dir, declared = _analysis(tmp_path)
    (_entries, absent, _shallow), caught = _harvest_recording(analysis_dir, declared[:1])

    assert absent == []
    assert caught == []


def test_a_single_absent_source_still_warns_once(tmp_path: Path) -> None:
    """Differently-positioned satisfying input: the k == 1 arm, which the aggregation
    must not silence (a bug that warned only for k >= 2 would pass the first test)."""
    analysis_dir, declared = _analysis(tmp_path)
    _result, caught = _harvest_recording(analysis_dir, declared[:2])

    assert len(caught) == 1
    assert re.search(r"(?<!\d)1(?!\d)", str(caught[0].message)), "the count 1 must appear as a bare integer token"
    assert "_status/_du.json" in str(caught[0].message)


def test_the_warning_is_still_an_error_under_w_error(tmp_path: Path) -> None:
    """`-W error` and `pytest.warns` keep seeing a UserWarning: logging was rejected for
    exactly this (DR-4 option 3), and `warn_explicit` must not have weakened it."""
    analysis_dir, declared = _analysis(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="declared_sources_absent"):
            _harvest_sources({"r": declared}, analysis_dir)


def test_each_harvest_shows_its_warning_rather_than_once_per_location(tmp_path: Path) -> None:
    """No `registry` is passed, so a loop over sweep members is not deduplicated away."""
    analysis_dir, declared = _analysis(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")  # the once-per-location mode a registry would feed
        for _ in range(2):
            _harvest_sources({"r": declared}, analysis_dir)

    assert len([w for w in caught if issubclass(w.category, UserWarning)]) == 2

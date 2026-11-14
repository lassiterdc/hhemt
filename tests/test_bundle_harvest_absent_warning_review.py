"""Reviewer test for chunk 1b.1 (F-D W1, DR-4): the ONE aggregated warning is complete at scale.

Property: however many declared sources are absent, and whatever characters their relative
paths carry, a harvest issues exactly one warning and that one warning names EVERY absent path
and the count. DR-4 asks for "the count, every absent path and the manifest key"; a warning
that shortens a long list for readability ("... and 57 more") satisfies a three-path test and
silently drops the paths the observer needs, because the manifest key is the only other place
they are recorded and a console reader does not open the manifest.

Class: any harvest with k >= 1 absent declared sources, k up to a sweep's member count, with
paths that contain spaces, brackets, a percent sign and a non-ASCII letter.

Why it is required: the coder's tests quantify over k = 3 and three plain paths, so a
truncating or an escaping implementation passes them.

What kills it: capping the listing at N paths, or passing the paths through a quoting or
escaping step that rewrites the characters. Either leaves a path out of the message text.

A second correct implementation that still passes: a message worded differently, with the
paths one per line or comma-separated, and the count as a bare number anywhere in the text.
This test asserts on the count as an integer token and on each path as a substring, never on
the sentence around them.

Every arm writes only ``tmp_path`` files; none compiles or runs a solver.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from hhemt.bundle._emit import _harvest_sources

#: Characters a regex, a printf and a shell would each treat specially, plus a non-ASCII letter.
_AWKWARD = ("eda/a [1].zarr", "eda/100%.zarr", "eda/café (x).zarr", "eda/q{2}.zarr")


def _declared(analysis_dir: Path, rels: list[str]) -> list[Path]:
    (analysis_dir / "eda").mkdir(parents=True, exist_ok=True)
    return [analysis_dir / rel for rel in rels]


@pytest.mark.parametrize("k", [1, 2, 60])
def test_one_warning_names_every_absent_path_and_the_count(tmp_path: Path, k: int) -> None:
    """CLASS: REGRESSION. Red on the pre-change tree: k warnings, not one, for k >= 2; for
    k == 1 the pre-change warning names the path and no count, so the count assertion fails."""
    analysis_dir = tmp_path / "analysis"
    rels = [f"eda/member-{i:03d}.zarr" for i in range(k)]
    declared = _declared(analysis_dir, rels)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _entries, absent, _shallow = _harvest_sources({"r": declared}, analysis_dir)
    harvest = [w for w in caught if issubclass(w.category, UserWarning)]

    assert sorted(absent) == sorted(rels)
    assert len(harvest) == 1, f"{len(harvest)} warnings for {k} absent sources"
    message = str(harvest[0].message)
    missing = [rel for rel in rels if rel not in message]
    assert not missing, f"{len(missing)} of {k} absent paths are not named: {missing[:3]}"
    assert re.search(rf"(?<!\d){k}(?!\d)", message), f"the count {k} is not named"


def test_awkward_path_characters_survive_into_the_message(tmp_path: Path) -> None:
    """CLASS: REGRESSION on paths whose characters a formatter may rewrite. Red on the
    pre-change tree: four warnings, not one."""
    analysis_dir = tmp_path / "analysis"
    declared = _declared(analysis_dir, list(_AWKWARD))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _harvest_sources({"r": declared}, analysis_dir)
    harvest = [w for w in caught if issubclass(w.category, UserWarning)]

    assert len(harvest) == 1, [str(w.message) for w in harvest]
    message = str(harvest[0].message)
    assert [rel for rel in _AWKWARD if rel not in message] == []

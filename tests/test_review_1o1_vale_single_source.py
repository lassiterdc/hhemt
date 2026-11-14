"""`.vale/README.md` must account for every `Hhemt*` style directory under `.vale/styles`.

Shape row: 1o.1 (approved shape S1 D6 / G6): hhemt's `.vale/styles/Hhemt*` is the one maintained copy of the
Hhemt styles, and the README names hhemt as their source and says no other copy is maintained.

Input class: every directory whose name starts with `Hhemt` under `.vale/styles`, against the README text.
Why required: a style the README does not name is a style a reader cannot tell is covered by the
single-source statement, so a copy of it elsewhere would read as legitimate; the README is the only place
the statement lives.
Kill: a README that omits one style directory (the fifth-style case) fails `test_readme_names_every_style`.
Second correct implementation (B5): a README that names the styles in a bulleted list, a prose sentence, or a
differently worded source statement still passes, because the check is on the directory name and two phrases.
Fitness function: a structural rule over the style directories with no reference to the problem domain.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_STYLES = _REPO / ".vale" / "styles"
_README = _REPO / ".vale" / "README.md"


def hhemt_style_names(styles: Path) -> list[str]:
    """The names of the `Hhemt*` style directories under `styles`."""
    return sorted(d.name for d in styles.iterdir() if d.is_dir() and d.name.startswith("Hhemt"))


def missing_style_mentions(readme_text: str, names: list[str]) -> list[str]:
    """The style names that `readme_text` does not name as a whole word."""
    return [n for n in names if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", readme_text)]


def states_single_source(readme_text: str) -> bool:
    """True when the README names hhemt as the maintainer and says no other copy is maintained."""
    lowered = readme_text.lower()
    return "hhemt" in lowered and "no other copy is maintained" in lowered


def test_population_is_nonempty() -> None:
    """Control: the enumeration reaches the four styles, so an empty pass is not vacuous."""
    assert len(hhemt_style_names(_STYLES)) >= 4


def test_readme_names_every_style() -> None:
    names = hhemt_style_names(_STYLES)
    assert missing_style_mentions(_README.read_text(encoding="utf-8"), names) == []


def test_readme_states_single_source() -> None:
    assert states_single_source(_README.read_text(encoding="utf-8"))


def test_detector_is_red_on_a_readme_omitting_a_style(tmp_path: Path) -> None:
    """NEW CAPABILITY: the plausible wrong README omits one style; the detector must name it."""
    styles = tmp_path / "styles"
    for n in ("HhemtDocs", "HhemtQuality", "HhemtMarkers", "HhemtContributing", "Other"):
        (styles / n).mkdir(parents=True)
    names = hhemt_style_names(styles)
    assert "Other" not in names
    full = " ".join(names)
    assert missing_style_mentions(full, names) == []
    for omitted in names:
        partial = " ".join(n for n in names if n != omitted)
        assert missing_style_mentions(partial, names) == [omitted]


@pytest.mark.parametrize("text", ["The hhemt repository maintains them.", "No other copy is maintained."])
def test_single_source_detector_needs_both_phrases(text: str) -> None:
    assert not states_single_source(text)

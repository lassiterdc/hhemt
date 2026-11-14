"""The unit of length is spelled "meters" in every tracked text file and in the configuration reference.

Three properties, each with a control that fires when the property is violated:

(a) No tracked text file (docs, source, tests, test data, identifiers and comments included) holds the other
    spelling of the word, in any compound ("kilometers" is refused as well). One lookup key must keep the name
    that a coordinate reference system gives its axis unit, and it is the only exemption.
(b) The configuration reference that the documentation build renders holds none.
(c) The ``manhole_diameter`` field states its unit.

The forbidden spelling is built at run time from fragments and planted only in temporary files, so this source holds
no literal of it. No solver compiles or runs.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_HOOK = REPO / "hooks" / "config_reference.py"

_STEM = "m" + "e" + "tr" + "e"
_FORBIDDEN = re.compile(_STEM + "s?", re.IGNORECASE)

#: (path, exact stripped line) pairs that may hold the spelling. The units lookup key must equal the unit name that
#: pyproj gives a CRS axis.
EXEMPTIONS: tuple[tuple[str, str], ...] = (("src/hhemt/units.py", '"' + _STEM + '": "m",'),)
EXEMPTIONS_LENGTH = 1


def scan_text(path: Path) -> list[tuple[int, str]]:
    """The (line number, stripped line) pairs of ``path`` that hold the spelling; a binary file yields none.

    The file is read as text and never parsed. Bytes that are not UTF-8 are replaced, so a stray byte beside the
    spelling is reported on its line and nothing raises.
    """
    data = path.read_bytes()
    if b"\x00" in data[:8000]:
        return []
    text = data.decode("utf-8", errors="replace")
    return [(n, line.strip()) for n, line in enumerate(text.split("\n"), start=1) if _FORBIDDEN.search(line)]


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [name.decode("utf-8", errors="replace") for name in out.split(b"\0") if name]


def _load_hook():
    spec = importlib.util.spec_from_file_location("config_reference", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config_reference"] = mod
    spec.loader.exec_module(mod)
    mod._load_local_src().bind_local_src(mod._SRC, {"hhemt"})
    return mod


def test_no_tracked_text_file_holds_the_other_spelling_outside_the_exemptions():
    """Property: over every tracked text file, the only lines holding the spelling are the exempted ones.

    Class: every ``git ls-files`` path that exists on disk and whose first 8,000 bytes hold no NUL byte.
    Kills it: a doc, docstring, message, comment or identifier spelled the other way.
    """
    assert len(EXEMPTIONS) == EXEMPTIONS_LENGTH
    exempt = set(EXEMPTIONS)
    found = []
    for name in _tracked_files():
        path = REPO / name
        if not path.is_file():
            continue
        for number, line in scan_text(path):
            if (name, line) not in exempt:
                found.append(f"{name}:{number}: {line[:120]}")
    assert found == [], "\n".join(found)


def test_each_exemption_matches_exactly_one_line_of_its_file():
    """Property: an exemption cannot go stale or widen without an edit a reviewer sees.

    Class: every pair of ``EXEMPTIONS``; the lines of its file that equal the pair's stripped line.
    """
    assert len(EXEMPTIONS) == EXEMPTIONS_LENGTH
    for name, line in EXEMPTIONS:
        hits = [text for _, text in scan_text(REPO / name) if text == line]
        assert len(hits) == 1, (name, line, len(hits))


def test_the_configuration_reference_holds_the_spelling_nowhere():
    """Property: the configuration reference as the documentation build renders it holds none.

    Class: the string that ``hooks/config_reference.py``'s ``_render()`` returns.
    """
    page = _load_hook()._render()
    assert page, "the rendered reference is empty, so the scan reads nothing"
    assert [line[:120] for line in page.split("\n") if _FORBIDDEN.search(line)] == []


def test_manhole_diameter_states_its_unit():
    """Property: the ``manhole_diameter`` description names the unit, spelled with "meters"."""
    from hhemt.config.analysis import analysis_config

    assert "meters" in (analysis_config.model_fields["manhole_diameter"].description or "")


@pytest.mark.parametrize(
    "word",
    [_STEM + "s", _STEM, "kilo" + _STEM + "s", "Centi" + _STEM.upper()],
    ids=["plural", "singular", "compound", "uppercase"],
)
def test_the_scanner_reports_each_form_of_the_spelling(tmp_path, word):
    """Control: a planted file holding the spelling is reported on its line, and one holding the good spelling is not.

    Kills it: a scanner anchored at the start of a word; one that is case-sensitive.
    """
    bad = tmp_path / "bad.txt"
    bad.write_text(f"first line\nthe depth is in {word}\n", encoding="utf-8")
    assert scan_text(bad) == [(2, f"the depth is in {word}")]
    good = tmp_path / "good.txt"
    good.write_text("the depth is in meters and kilometers\n", encoding="utf-8")
    assert scan_text(good) == []


def test_an_undecodable_byte_beside_the_spelling_is_reported_and_raises_nothing(tmp_path):
    """Control: a file with the byte 0xB3 and the spelling on one line is reported on that line."""
    path = tmp_path / "mixed.txt"
    path.write_bytes(b"ok\nx \xb3 in " + (_STEM + "s").encode() + b"\n")
    hits = scan_text(path)
    assert [n for n, _ in hits] == [2]


def test_a_binary_file_is_skipped(tmp_path):
    """Control: a file whose first bytes hold a NUL is not scanned."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00" + (_STEM + "s").encode())
    assert scan_text(path) == []


def test_a_second_spelling_line_in_a_copy_of_the_units_module_is_reported():
    """Control: the exemption covers exactly one line; a copy with a second line holding the key is flagged.

    Class: a temporary copy of ``units.py`` with the exempted line repeated.
    """
    import tempfile

    name, line = EXEMPTIONS[0]
    source = (REPO / name).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "units.py"
        copy.write_text(source + "\n    " + line + "\n", encoding="utf-8")
        hits = [text for _, text in scan_text(copy) if text == line]
    assert len(hits) == 2

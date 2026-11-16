"""No source file of the toolkit names a coastal term.

Every name the toolkit owns for an external boundary is neutral: a boundary applies a water level, a free outflow, a
slope or a Froude number, and nothing in the code says which kind of water body or storm produced it. This test reads
every ``src/hhemt/**/*.py``, splits identifiers at underscores and at camelCase boundaries and every other run of text
at non-letters, lower-cases each token, and fails on a token that is exactly ``tide``, ``tides``, ``surge`` or
``surges``, or that begins with ``storm`` and ends in one of those. There is no exclusion list: ``surgery``,
``surgeried``, ``identity`` and ``TIDAL`` (a SWMM keyword) are not those tokens.

A key that a refusal or a migration must still spell is assembled from parts that are not themselves such a
token, which is why this test needs no exception for them. No solver is compiled or run; no test here reads more than
the source text.
"""

from __future__ import annotations

import re
from pathlib import Path

import hhemt

SRC_ROOT = Path(hhemt.__file__).resolve().parent

_COASTAL_WORDS = ("tide", "tides", "surge", "surges")
# A letter run is split into tokens at camelCase boundaries: an acronym stays whole until a capital begins a
# lowercase run.
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
_LETTERS = re.compile(r"[A-Za-z]+")


def _is_coastal(token: str) -> bool:
    return token in _COASTAL_WORDS or (token.startswith("storm") and token.endswith(_COASTAL_WORDS))


def coastal_tokens(text: str) -> list[tuple[int, str]]:
    """The ``(line, token)`` of every coastal token of ``text``, in order."""
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        if "tide" not in lowered and "surge" not in lowered:
            continue
        for word in _LETTERS.findall(line):
            for part in _CAMEL.findall(word):
                if _is_coastal(part.lower()):
                    found.append((number, part.lower()))
    return found


def _source_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def test_no_source_token_is_a_coastal_word():
    offenders: list[str] = []
    for path in _source_files():
        raw = path.read_bytes().lower()
        if b"tide" not in raw and b"surge" not in raw:
            continue
        for number, token in coastal_tokens(path.read_text()):
            offenders.append(f"{path.relative_to(SRC_ROOT.parent.parent)}:{number}: {token}")
    assert not offenders, "coastal tokens in source:\n" + "\n".join(offenders)


def test_the_scan_reads_the_source_tree():
    # Control: a scan that read nothing would pass the test above for the wrong reason.
    assert len(_source_files()) > 100


def test_the_old_spellings_fire_the_guard():
    # Control that varies the property (the token is a coastal word), not the file: each coastal spelling fires.
    probe = "\n".join(
        [
            "toggle_storm_tide_boundary = True",
            "stormsurge_peak_m = 1.0",
            "frame['TideGate'] = 'NO'",
            "# the storm tide boundary",
            "x = surge_height",
        ]
    )
    tokens = {token for _line, token in coastal_tokens(probe)}
    assert {"tide", "stormsurge", "surge"} <= tokens


def test_the_near_misses_do_not_fire_the_guard():
    probe = "# un-surgeried ExperimentIdentity contentIdentifier TIDAL surgery identity stormwater"
    assert coastal_tokens(probe) == []

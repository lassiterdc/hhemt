"""Reviewer pins for the v0.3.0 release entry (the release entry), over classes the author's file does not quantify.

The author's file pins that named words are present in the entry. These tests pin what the entry CLAIMS
against the tree that ships with it: every backticked setting resolves to a config field or a removed key,
the removed keys are refused with a replacement message, the lock file carries the released version, and the
entry names no step of the release procedure that the developer owns. Every test reads text only.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_RELEASE = "0.3.0"
# The steps the approved shape (package 3a) keeps for the developer: tag, PyPI publish, GitHub release, DOI.
_DEVELOPER_STEPS = re.compile(r"\b(tag|tagged|pypi|github|publish|published|publishing|zenodo|doi)\b", re.IGNORECASE)


def _entry() -> str:
    text = (_REPO / "HISTORY.md").read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^## v{re.escape(_RELEASE)} \(\d{{4}}-\d{{2}}-\d{{2}}\)[ \t]*\n(.*?)(?=^## v|\Z)", text)
    assert match is not None, f"HISTORY.md has no '## v{_RELEASE} (YYYY-MM-DD)' entry"
    return match.group(1)


def _backticked(entry: str) -> list[str]:
    """Backticked identifiers only; a backticked file extension such as `.inp` names no setting."""
    return [t for t in re.findall(r"`([^`\n]+)`", entry) if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t)]


def _declared_fields(sources: list[str]) -> set[str]:
    """Names declared as a pydantic field line (`    name: type ...`) in the given source texts."""
    names: set[str] = set()
    for text in sources:
        names |= set(re.findall(r"(?m)^    ([A-Za-z_][A-Za-z0-9_]*)\s*:", text))
    return names


def _retired_keys(system_text: str) -> set[str]:
    block = re.search(r"_retired_template_keys = \{(.*?)\n\s*\}", system_text, re.S)
    assert block is not None, "system.py no longer declares _retired_template_keys"
    return set(re.findall(r'"([A-Za-z_]+)"\s*:', block.group(1)))


def _unresolved(tokens: list[str], system_text: str, analysis_text: str) -> list[str]:
    # The analysis config refuses its removed keys through `_REFUSED_BOUNDARY_KEYS`, whose spellings are assembled from
    # parts, so no text match over analysis.py can see them; the map itself is the source.
    from hhemt.config.analysis import _REFUSED_BOUNDARY_KEYS

    known = _declared_fields([system_text, analysis_text]) | _retired_keys(system_text) | set(_REFUSED_BOUNDARY_KEYS)
    return [t for t in tokens if t not in known]


def test_every_backticked_setting_in_the_entry_is_a_config_field_or_a_retired_key():
    """NEW CAPABILITY — plausible wrong implementation: a field given a new name after the entry was written.

    Quantifies over every backticked token of the v0.3.0 entry. Enforces approved-shape row: package 3a,
    'no claim the merged code does not make'. A setting the entry names that no config class declares and
    no removed-key map lists is a release note about a field that does not exist. The same check runs on the
    finished tree once the observation and raster fields land, which is when a rename of such a field would
    show. Killed by renaming `TRITON_hextra_m` in analysis.py or deleting `swmm_outfalls_to_couple` in system.py.
    """
    system_text = (_REPO / "src/hhemt/config/system.py").read_text(encoding="utf-8")
    analysis_text = (_REPO / "src/hhemt/config/analysis.py").read_text(encoding="utf-8")
    tokens = _backticked(_entry())
    assert len(tokens) >= 10, tokens  # the entry names the model, the outfall list, four removed keys, the settings
    assert _unresolved(tokens, system_text, analysis_text) == []


def test_the_resolution_check_flags_a_setting_no_class_declares():
    """Control for the test above: the instrument returns a nonzero result on a known-violating input."""
    system_text = (_REPO / "src/hhemt/config/system.py").read_text(encoding="utf-8")
    analysis_text = (_REPO / "src/hhemt/config/analysis.py").read_text(encoding="utf-8")
    assert _unresolved(["TRITON_courant_number", "TRITON_cfl_number"], system_text, analysis_text) == [
        "TRITON_cfl_number"
    ]


def test_every_retired_key_the_entry_names_is_refused_with_a_replacement_message():
    """REGRESSION-class pin over the removed-key map: the entry says such a config 'is refused with a message
    that says what replaces it'.

    Quantifies over the removed keys the entry names (the four SWMM and cfg-template keys). Each must be a key of
    `_retired_template_keys` in system.py, and the raise built from that map must carry the word 'Replacement'.
    Killed by dropping a key from the map or by rewording the raise so it names no replacement.
    """
    system_text = (_REPO / "src/hhemt/config/system.py").read_text(encoding="utf-8")
    named = {
        t
        for t in _backticked(_entry())
        if t in {"SWMM_hydraulics", "SWMM_hydrology", "SWMM_full", "triton_swmm_configuration_template"}
    }
    assert named == {"SWMM_hydraulics", "SWMM_hydrology", "SWMM_full", "triton_swmm_configuration_template"}
    assert named <= _retired_keys(system_text)
    assert re.search(
        r'raise ValueError\(\s*f"\'\{_k\}\' is not a system config field[^"]*"\s*f"Replacement: \{_use\}', system_text
    )


def test_the_lock_file_carries_the_released_version():
    """REGRESSION: the bump that edits pyproject.toml and CITATION.cff alone fails every hook that syncs with
    `uv sync --locked` (the author's round-1 obstruction), so the lock must agree with the declared version.

    Quantifies over the four copies of the version: pyproject.toml, CITATION.cff, the newest HISTORY heading and the
    `hhemt` package row of uv.lock. Killed by reverting any one of them to 0.2.1.
    """
    py = re.search(r'(?m)^version\s*=\s*"([^"]+)"', (_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    cff = re.search(r'(?m)^version:\s*"([^"]+)"', (_REPO / "CITATION.cff").read_text(encoding="utf-8"))
    lock = re.search(
        r'(?m)^name = "hhemt"\nversion = "([^"]+)"\nsource = \{ editable = "\." \}',
        (_REPO / "uv.lock").read_text(encoding="utf-8"),
    )
    head = re.search(r"(?m)^## v(\S+) \(", (_REPO / "HISTORY.md").read_text(encoding="utf-8"))
    assert py and cff and lock and head
    assert {py.group(1), cff.group(1), lock.group(1), head.group(1)} == {_RELEASE}


def test_the_entry_names_no_step_the_developer_keeps_for_the_release():
    """The tag, the PyPI publish, the GitHub release and the DOI are the developer's; the entry must not claim
    or imply one.

    Quantifies over the closed list of those steps (tag, PyPI, GitHub, publish, Zenodo, DOI), read as whole words. The
    author's file bans doi, zenodo, venue, citation and reaper but not tag, PyPI, GitHub or publish. Killed by adding
    'Tagged and published to PyPI.' to the entry.
    """
    assert _DEVELOPER_STEPS.findall(_entry()) == []


def test_the_release_step_instrument_flags_a_tagged_and_published_claim():
    """Control for the test above: a known-violating entry yields a nonzero result."""
    assert _DEVELOPER_STEPS.findall("Tagged and published to PyPI.") == ["Tagged", "published", "PyPI"]

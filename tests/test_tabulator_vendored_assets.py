"""Integrity and packaging pins for the vendored Tabulator assets (`[Q160]`(7)).

A vendored third-party asset whose integrity is never checked is a supply-chain hole
dressed as convenience: nothing else in the tree would notice if one of these files were
replaced, truncated, or silently re-fetched at a different version. These pins are the
check. They were captured from the published `tabulator-tables@6.4.0` distribution at
vendoring time and independently corroborated against the npm registry's own metadata.

The pins are deliberately duplicated from nothing — they are the ground truth, hand-
recorded here rather than computed from the files they guard, because a hash computed
from the artifact it verifies proves only that the file equals itself.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import pytest

from hhemt.report_renderers._tabulator_defaults import (
    _TABULATOR_VENDOR_DIR,
    _TABULATOR_VERSION,
    tabulator_head_assets,
)

#: filename -> (exact byte size, sha256) for tabulator-tables@6.4.0.
_PINS: dict[str, tuple[int, str]] = {
    "LICENSE": (1082, "191a2ee554684e1064c897b432f0e1bc6dfa714ca045d3f6ea2cf692cbd398b7"),
    "tabulator.min.js": (443166, "86df9b98a7cde1098d8cbc0f1916b6989971507984299bc0b4d289a63ed520a0"),
    "tabulator.min.css": (28375, "93ab046ce80d8c1933b06b30d530b5835796047aff2e057a1ec458287ba5515b"),
}


@pytest.mark.parametrize("name", sorted(_PINS))
def test_vendored_asset_matches_its_pin(name):
    """Each vendored file is byte-exactly the artifact it claims to be."""
    path = _TABULATOR_VENDOR_DIR / name
    assert path.exists(), f"vendored asset missing: {path}"
    size, digest = _PINS[name]
    raw = path.read_bytes()
    assert len(raw) == size, f"{name}: expected {size} bytes, found {len(raw)}"
    assert hashlib.sha256(raw).hexdigest() == digest, f"{name}: sha256 mismatch"


def test_vendor_dir_is_named_for_the_version_pin():
    """The vendored path and the CDN URLs derive from ONE constant, so they cannot
    silently disagree. A version bump that is not accompanied by re-vendoring must fail
    loudly at read time rather than serve one version from disk and another from a CDN."""
    assert _TABULATOR_VENDOR_DIR.name == _TABULATOR_VERSION


def test_license_ships_with_the_copy_and_reaches_the_rendered_page():
    """MIT permits redistribution provided the copyright and permission notice accompany
    the copy. Vendoring the LICENSE file satisfies the first half; emitting it into the
    inline payload satisfies the second, for a page that is itself redistributed."""
    text = (_TABULATOR_VENDOR_DIR / "LICENSE").read_text(encoding="utf-8")
    assert "MIT" in text and "Permission is hereby granted" in text

    head = tabulator_head_assets("inline")
    assert "Permission is hereby granted" in head, "the MIT notice must travel with the bundle"


def test_inline_mode_emits_no_network_reference_and_cdn_mode_still_does():
    """The whole point of the mode: an inline page fetches nothing at view time.

    Two arms rather than one — asserting only that inline contains the asset bytes would
    pass on an implementation that ALSO left the CDN tags in place, which would keep the
    network dependency the user's ruling was meant to remove.
    """
    inline = tabulator_head_assets("inline")
    assert "cdn.jsdelivr.net" not in inline, "inline mode must not reference a CDN"
    assert "<style>" in inline and "<script>" in inline

    cdn = tabulator_head_assets("cdn")
    assert "cdn.jsdelivr.net" in cdn, "cdn mode must still reference the CDN"
    assert cdn.count("cdn.jsdelivr.net") == 2, "one CSS link and one JS script"


def test_vendored_assets_are_declared_package_data():
    """The assets ship in the wheel. They sit under `report_templates/`, which pyproject
    already declares as package-data, so no packaging change was needed — this pins that
    the declaration still covers them rather than trusting it to stay true."""
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    patterns = cfg["tool"]["setuptools"]["package-data"]["hhemt"]
    assert "report_templates/**/*" in patterns
    # And the assets really are under that root, so the glob reaches them.
    src_root = Path(__file__).resolve().parents[1] / "src" / "hhemt"
    assert _TABULATOR_VENDOR_DIR.is_relative_to(src_root / "report_templates")

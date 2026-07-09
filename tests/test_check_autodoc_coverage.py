"""Regression tests for scripts/check_autodoc_coverage.py — the ADR-7 docs-accuracy
gate. Locks the classification + exit-code contract so a silent regression in the
release-floor gate is caught. Mirrors tests/test_check_anonymization.py."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_autodoc_coverage.py"

def _load_module():
    spec = importlib.util.spec_from_file_location("check_autodoc_coverage", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cac = _load_module()

def _make_fake_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)

    class SomeClass:
        pass

    def some_func():
        return None

    m.SomeClass = SomeClass
    m.some_func = some_func
    m.SOME_INT = 5
    m.SOME_STR = "x"
    m._private = some_func  # leading underscore -> filtered like mkdocs filters
    m.__all__ = ["SomeClass", "some_func", "SOME_INT", "SOME_STR", "_private"]
    return m

def test_expected_qualnames_classifies_by_kind(monkeypatch):
    monkeypatch.setitem(sys.modules, "fakepkg", _make_fake_module("fakepkg"))
    monkeypatch.setattr(cac, "PUBLIC_MODULES", ("fakepkg",))
    # class + function kept; int/str constants and _private dropped
    assert cac.expected_qualnames() == {"fakepkg.SomeClass", "fakepkg.some_func"}

def test_rendered_anchors_collects_ids(tmp_path):
    (tmp_path / "page.html").write_text(
        '<h2 id="fakepkg.SomeClass">C</h2>'
        '<span class="n">fakepkg.some_func</span>'  # incidental substring, no id
        '<h3 id="fakepkg.some_func">f</h3>',
        encoding="utf-8",
    )
    ids = cac.rendered_anchors(tmp_path)
    assert {"fakepkg.SomeClass", "fakepkg.some_func"} <= ids

def test_main_exit_2_when_site_dir_absent(tmp_path):
    assert cac.main(["--site-dir", str(tmp_path / "nope")]) == 2

def test_main_exit_1_when_symbol_unrendered(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fakepkg", _make_fake_module("fakepkg"))
    monkeypatch.setattr(cac, "PUBLIC_MODULES", ("fakepkg",))
    (tmp_path / "page.html").write_text('<h2 id="fakepkg.SomeClass">C</h2>', encoding="utf-8")
    assert cac.main(["--site-dir", str(tmp_path)]) == 1  # some_func unrendered

def test_main_exit_0_when_all_rendered(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fakepkg", _make_fake_module("fakepkg"))
    monkeypatch.setattr(cac, "PUBLIC_MODULES", ("fakepkg",))
    (tmp_path / "page.html").write_text(
        '<h2 id="fakepkg.SomeClass">C</h2><h3 id="fakepkg.some_func">f</h3>',
        encoding="utf-8",
    )
    assert cac.main(["--site-dir", str(tmp_path)]) == 0

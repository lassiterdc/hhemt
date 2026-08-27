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


def _make_fake_module(name: str, *, documented: bool = True) -> types.ModuleType:
    """A stand-in public module.

    ``documented`` controls whether its public symbols carry docstrings, which is
    the axis the docstring-presence assertion keys on. It defaults True so the
    anchor-assertion tests exercise anchors alone rather than failing for an
    unrelated reason.
    """
    m = types.ModuleType(name)

    class SomeClass:
        """A documented class."""

        def some_method(self):
            """A documented method."""

        @property
        def some_prop(self):
            """A documented property."""
            return None

        @classmethod
        def some_classmethod(cls):
            """A documented classmethod."""

    def some_func():
        """A documented function."""
        return None

    if not documented:
        SomeClass.__doc__ = None
        SomeClass.some_method.__doc__ = None
        some_func.__doc__ = None

    m.SomeClass = SomeClass
    m.some_func = some_func
    m.SOME_INT = 5
    m.SOME_STR = "x"
    m._private = some_func  # leading underscore -> filtered like mkdocs filters
    m.__all__ = ["SomeClass", "some_func", "SOME_INT", "SOME_STR", "_private"]
    return m


def _use_fake(monkeypatch, *, documented: bool = True) -> None:
    """Point the gate at a single fake module.

    Patches the `public_modules` FUNCTION rather than a module-level constant:
    the documented surface is derived from docs/reference/api.md's `:::`
    directives, so there is no tuple to patch.
    """
    monkeypatch.setitem(sys.modules, "fakepkg", _make_fake_module("fakepkg", documented=documented))
    monkeypatch.setattr(cac, "public_modules", lambda *a, **k: ("fakepkg",))


# --- the derived module list ------------------------------------------------

def test_public_modules_derives_from_api_reference_directives(tmp_path):
    page = tmp_path / "api.md"
    page.write_text(
        "# API Reference\n\n::: hhemt\n\n::: hhemt.analysis\n\nSome prose ::: not a directive\n",
        encoding="utf-8",
    )
    assert cac.public_modules(page) == ("hhemt", "hhemt.analysis")


def test_public_modules_matches_the_live_api_reference():
    """The gate's surface IS the rendering surface — no second hand-maintained list.

    This is the assertion that would have caught the divergence the derivation
    removed: a tuple declaring 6 modules while the page declared 8.
    """
    live = cac.public_modules()
    page_text = cac.API_REFERENCE_PAGE.read_text(encoding="utf-8")
    declared = [ln.split(":::", 1)[1].strip() for ln in page_text.splitlines() if ln.startswith(":::")]
    assert list(live) == declared


def test_public_modules_raises_when_page_declares_no_directives(tmp_path):
    """An empty derived set must be an error, never a vacuous pass."""
    page = tmp_path / "api.md"
    page.write_text("# API Reference\n\nNo directives here.\n", encoding="utf-8")
    try:
        cac.public_modules(page)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a page declaring no ::: directives")


# --- assertion 1: anchors ---------------------------------------------------

def test_expected_qualnames_classifies_by_kind(monkeypatch):
    _use_fake(monkeypatch)
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
    _use_fake(monkeypatch)
    (tmp_path / "page.html").write_text('<h2 id="fakepkg.SomeClass">C</h2>', encoding="utf-8")
    assert cac.main(["--site-dir", str(tmp_path)]) == 1  # some_func unrendered


def test_main_exit_0_when_all_rendered_and_documented(tmp_path, monkeypatch):
    _use_fake(monkeypatch)
    (tmp_path / "page.html").write_text(
        '<h2 id="fakepkg.SomeClass">C</h2><h3 id="fakepkg.some_func">f</h3>',
        encoding="utf-8",
    )
    assert cac.main(["--site-dir", str(tmp_path)]) == 0


# --- assertion 2: docstring presence ----------------------------------------

def test_undocumented_symbols_empty_when_all_documented(monkeypatch):
    _use_fake(monkeypatch)
    assert cac.undocumented_symbols() == []


def test_undocumented_symbols_flags_bare_names(monkeypatch):
    _use_fake(monkeypatch, documented=False)
    found = set(cac.undocumented_symbols())
    assert {"fakepkg.SomeClass", "fakepkg.some_func", "fakepkg.SomeClass.some_method"} <= found


def test_main_exit_1_when_rendered_but_undocumented(tmp_path, monkeypatch):
    """The defect the anchor assertion alone cannot see.

    Every symbol renders an anchor, so assertion 1 passes; the symbols carry no
    docstring, so they render as bare names. This is the state the gate was
    green on before the docstring assertion was added.
    """
    _use_fake(monkeypatch, documented=False)
    (tmp_path / "page.html").write_text(
        '<h2 id="fakepkg.SomeClass">C</h2><h3 id="fakepkg.some_func">f</h3>',
        encoding="utf-8",
    )
    assert cac.main(["--site-dir", str(tmp_path)]) == 1


def test_undocumented_symbols_sees_classmethods_and_properties(monkeypatch):
    """`inspect.isfunction` is False for a classmethod retrieved via `vars()`.

    Testing only `isfunction` silently undercounts — measured on this repository
    at 23 undocumented against a true 25, with both `Bundle.from_directory` and
    `CombinedBundle.from_directory` missed for exactly that reason.
    """
    m = _make_fake_module("fakepkg2")
    m.SomeClass.some_classmethod.__func__.__doc__ = None
    m.SomeClass.some_prop.fget.__doc__ = None
    monkeypatch.setitem(sys.modules, "fakepkg2", m)
    monkeypatch.setattr(cac, "public_modules", lambda *a, **k: ("fakepkg2",))
    found = set(cac.undocumented_symbols())
    assert "fakepkg2.SomeClass.some_classmethod" in found
    assert "fakepkg2.SomeClass.some_prop" in found

"""Regression tests for scripts/check_autodoc_coverage.py — the ADR-7 docs-accuracy
gate. Locks the classification + exit-code contract so a silent regression in the
release-floor gate is caught. Mirrors tests/test_check_anonymization.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

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


#: The twenty qualnames the ``__all__``-or-defs rule reaches and the old
#: ``getattr(mod, "__all__", ())`` rule did not. Pinned BY NAME, not by a count:
#: the count moved 47 -> 67 and a count assertion cannot say WHICH symbols it
#: gained, so it would pass on any repair that reached sixty-seven of anything.
#: These are the entire public surface of the two modules that declare no
#: ``__all__`` and were therefore read as declaring nothing.
RULE_GAINED_QUALNAMES = frozenset(
    {
        "hhemt.experiment_bundle.OverrideReport",
        "hhemt.experiment_bundle.build_case_from_bundle",
        "hhemt.experiment_bundle.expand_config_vars",
        "hhemt.experiment_bundle.format_override_gate",
        "hhemt.experiment_bundle.load_bundle",
        "hhemt.experiment_bundle.resolve_container_defs",
        "hhemt.experiment_bundle.resolve_def_recipe",
        "hhemt.experiment_bundle.resolve_hpc_system_config",
        "hhemt.experiment_bundle.resolve_overrides",
        "hhemt.experiment_bundle.run_experiment",
        "hhemt.synthetic_experiment.assert_coupling_nodes_distinct",
        "hhemt.synthetic_experiment.build_experiment_matrix",
        "hhemt.synthetic_experiment.dem_resolution_matrix_rows",
        "hhemt.synthetic_experiment.experiment_matrix_rows",
        "hhemt.synthetic_experiment.generate_synthetic_experiment",
        "hhemt.synthetic_experiment.model_arm_toggles",
        "hhemt.synthetic_experiment.size_resume_walltimes",
        "hhemt.synthetic_experiment.write_clean_matrix_csv",
        "hhemt.synthetic_experiment.write_resume_matrix_csv",
        "hhemt.synthetic_experiment.write_smoke_matrix_csv",
    }
)


def test_the_rule_reaches_modules_that_declare_no_dunder_all():
    """THE NON-VACUITY ARM. Red before the derivation repair, green after.

    ``expected - anchors`` is 0 in BOTH states -- 47 and 67 are each a subset of
    what renders -- so the gate's own assertion cannot tell the repair happened.
    This one can: every name below is absent from the old rule's output and
    present in the new one.
    """
    expected = cac.expected_qualnames()
    missing = sorted(RULE_GAINED_QUALNAMES - expected)
    assert not missing, (
        f"{len(missing)} symbol(s) the __all__-or-defs rule must reach are absent "
        f"from expected_qualnames(): {missing[:5]}"
    )


def test_the_rule_does_not_admit_imported_names():
    """The runtime-sweep failure this rule exists to avoid.

    ``dir(mod)`` on a module with no ``__all__`` returns imported names too, and
    the class/routine filter cannot strain them out. These four are stdlib or
    cross-module imports that render no anchor of their own.
    """
    expected = cac.expected_qualnames()
    for leaked in (
        "hhemt.experiment_bundle.Path",
        "hhemt.experiment_bundle.dataclass",
        "hhemt.synthetic_experiment.Path",
        "hhemt.experiment_bundle.ConfigurationError",
    ):
        assert leaked not in expected, f"{leaked} is an imported name and must not be expected"


# --- the population root: THIS checkout's src, not what the interpreter finds ----
#
# The seven ``sys.modules``-injected fixtures above pin classification and exit
# codes and pin NOTHING about resolution: ``import_module`` returns the injected
# entry before any finder runs. These arms construct both trees under ``tmp_path``
# and pin one half of the binder each: the prepend (cold, foreign copy first on
# ``sys.path``), the purge (warm, foreign module already imported), and the funnel's
# CALL to the provenance assert (a ``src`` that exists and supplies nothing).

_FAKE_PACKAGE = (
    '__all__ = ["{name}"]\n\n\n'
    "class {name}:\n"
    '    """Documented, so the docstring clause is never the reason a node is red."""\n'
)


@pytest.fixture
def clean_fakepkg():
    """Leave ``sys.modules`` clear of every ``fakepkg*`` entry a constructed arm loads,
    and put ``sys.path`` back to what it was before the arm ran.

    The cold arm imports a REAL ``fakepkg`` from a ``tmp_path`` pytest will delete, and
    every arm's funnel call prepends its ``src_root`` to ``sys.path``. ``monkeypatch``
    restores ``sys.path`` only for an arm that called ``syspath_prepend`` itself, which
    the warm arm does not, and it does nothing to ``sys.modules``. Without this teardown
    the warm arm would leave its ``repo/src`` first on ``sys.path`` for every later node,
    and would inherit a module that is not its decoy, so its precondition would be
    satisfied for the wrong reason.
    """

    def _pop() -> None:
        for name in [n for n in sys.modules if n == "fakepkg" or n.startswith("fakepkg.")]:
            del sys.modules[name]

    saved_path = sys.path[:]
    _pop()
    yield
    _pop()
    sys.path[:] = saved_path
    importlib.invalidate_caches()


def _two_trees(tmp_path: Path) -> tuple[Path, Path, Path]:
    """``repo/src/fakepkg`` exporting ``InRepo``, ``other/fakepkg`` exporting ``Elsewhere``.

    Returns ``(repo_src, other, api_page)``; the page declares ``::: fakepkg``.
    """
    repo_src = tmp_path / "repo" / "src"
    other = tmp_path / "other"
    for root, name in ((repo_src, "InRepo"), (other, "Elsewhere")):
        pkg = root / "fakepkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(_FAKE_PACKAGE.format(name=name), encoding="utf-8")
    api_page = tmp_path / "repo" / "docs" / "reference" / "api.md"
    api_page.parent.mkdir(parents=True)
    api_page.write_text("# API\n\n::: fakepkg\n", encoding="utf-8")
    return repo_src, other, api_page


def _load_decoy(path: Path) -> types.ModuleType:
    """A ``fakepkg`` loaded from ``path`` with NO finder consulted."""
    spec = importlib.util.spec_from_file_location("fakepkg", path)
    assert spec is not None and spec.loader is not None
    decoy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoy)
    return decoy


def _origin_of(name: str) -> Path:
    return Path(sys.modules[name].__file__).resolve()


def test_enumerator_binds_to_src_root_over_a_foreign_copy_first_on_sys_path(tmp_path, monkeypatch, clean_fakepkg):
    """The PREPEND half. Red today: the foreign copy is first on ``sys.path`` and nothing rebinds.

    Today this node is red for a reason other than the one its assertion states:
    ``expected_qualnames`` accepts no ``api_page``/``src_root`` keywords, so the call
    raises ``TypeError`` before any import. After the repair the assertion is what
    discriminates, and this arm alone cannot tell a prepend-only rebind from
    prepend-plus-purge; the warm arm below can.
    """
    repo_src, other, api_page = _two_trees(tmp_path)
    monkeypatch.syspath_prepend(str(other))
    expected = cac.expected_qualnames(api_page=api_page, src_root=repo_src)
    assert expected == {"fakepkg.InRepo"}
    assert _origin_of("fakepkg").is_relative_to(repo_src.resolve())


def test_enumerator_evicts_a_warm_foreign_module_before_importing(tmp_path, monkeypatch, clean_fakepkg):
    """The PURGE half: the only arm a prepend-only rebind fails.

    ``import_module`` returns the ``sys.modules`` entry before any finder runs, so a
    prepend changes nothing here and only an eviction restores the right tree. The
    decoy is loaded by path and installed with ``monkeypatch.setitem`` so no finder
    is consulted and the precondition, asserted before the enumerator runs, cannot
    be satisfied by a module another node left warm. Red today for the same
    ``TypeError`` reason as the cold arm.
    """
    repo_src, other, api_page = _two_trees(tmp_path)
    decoy = _load_decoy(other / "fakepkg" / "__init__.py")
    monkeypatch.setitem(sys.modules, "fakepkg", decoy)
    assert _origin_of("fakepkg").is_relative_to(other.resolve())
    expected = cac.expected_qualnames(api_page=api_page, src_root=repo_src)
    assert expected == {"fakepkg.InRepo"}
    assert _origin_of("fakepkg").is_relative_to(repo_src.resolve())


def test_enumerator_refuses_a_package_the_src_root_cannot_supply(tmp_path, monkeypatch, clean_fakepkg):
    """The ASSERT half's CALL SITE: a ``src`` that exists and holds no ``fakepkg``.

    The prepend lands on a directory that supplies nothing, the purge evicts, the
    import falls through to the foreign copy on ``sys.path`` (the stand-in for the
    editable ``.pth``), and the funnel's provenance assert is the only thing left
    that can distinguish that from success. With the assert call deleted this
    returns ``{"fakepkg.Elsewhere"}`` and raises nothing, so the both-states anchor
    is raise-versus-return, not wording. Today red for the wrong reason: the
    ``TypeError`` on the unknown keywords, not a missing raise.
    """
    _, other, api_page = _two_trees(tmp_path)
    empty_src = tmp_path / "empty" / "src"
    empty_src.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(other))
    with pytest.raises(RuntimeError) as excinfo:
        cac.expected_qualnames(api_page=api_page, src_root=empty_src)
    assert str(empty_src.resolve()) in str(excinfo.value)
    assert str(other.resolve()) in str(excinfo.value)

"""Regression tests for scripts/check_docs_content.py's rendered-docstring population.

The central assertion is a FIXTURE, not a count and not a recorded run. A count
passes when the population shrinks and something else grows; a recorded run
records a past state and cannot fail later. The pinned qualnames go red on the
exact symbol a STRICT derivation loses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "check_docs_content.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_docs_content", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_docs_content"] = mod
    spec.loader.exec_module(mod)
    return mod


cdc = _load()

#: Symbols whose docstrings carry a bare `path:line` citation at the pinned HEAD.
#: `assert_coupling_nodes_distinct` is the one a `__all__`-keyed (STRICT) derivation
#: LOSES -- its module declares no `__all__`. If this list stops being reached, the
#: population has silently narrowed and the gate is under-reporting, not clean.
CITATION_BEARING = {
    "hhemt.synthetic_experiment.assert_coupling_nodes_distinct",
}

#: Names the PAGE emits as anchors and that only a renderer-derived population
#: reaches. Each is a defect the `ast` traversal had, pinned as a name rather
#: than a count: a count passes when one thing shrinks and another grows.
#:
#:   `hhemt.bundle`                  -- a MODULE docstring, which the `ast` walk
#:                                      never read
#:   `hhemt.analysis.TestSubResult`  -- DEFINED in a module declaring `__all__`
#:                                      without listing it; the page renders it,
#:                                      an `__all__`-narrowing walk does not, and
#:                                      neither does `griffe`'s `is_public`
#:   `hhemt.Toolkit`                 -- a RE-EXPORT, keyed the way the page keys
#:                                      it rather than by its origin module
PAGE_KEYED = {
    "hhemt.bundle",
    "hhemt.analysis.TestSubResult",
    "hhemt.Toolkit",
}

#: Names the population must NOT contain. The first two are origin-module keys
#: for symbols the page presents under a shorter name; the third is an imported
#: alias that its module does not export, so the page renders it nowhere. All
#: three were in the population before it was derived from `griffe`.
NOT_THE_PAGE_S_NAMES = {
    "hhemt.toolkit.Toolkit",
    "hhemt.bundle._combine.CombinedBundle",
    "hhemt.config.experiment_bundle.ExperimentConfig",
}


def _population():
    return cdc.rendered_docstrings(cdc.SRC_ROOT, cdc.API_PAGE)


def test_population_reaches_the_symbols_a_strict_derivation_loses():
    """The fixture. Named symbols, not a count."""
    reached = {q for q, _home, _line, _doc in _population()}
    missing = CITATION_BEARING - reached
    assert not missing, (
        f"population no longer reaches {sorted(missing)}. A `__all__`-keyed derivation "
        f"loses exactly these; check the population rule, not the symbols."
    )


def test_population_reaches_the_symbols_only_the_renderer_s_own_model_finds():
    """Each pin is one defect an `ast` traversal had, expressed as a page anchor."""
    reached = {q for q, _home, _line, _doc in _population()}
    missing = PAGE_KEYED - reached
    assert not missing, (
        f"population no longer reaches {sorted(missing)}. These are names the built page "
        f"emits as anchors; losing one means the derivation has stopped tracking the "
        f"renderer -- check the derivation, not the symbols."
    )


def test_population_is_keyed_the_way_the_page_is():
    """Defect 3: the right bytes under the wrong name are still the wrong name.

    Asserted two ways, because the pinned trio is illustrative and the invariant
    is general: no name the page can emit contains a private path segment, since
    `filters: ["!^_"]` keeps private members off the page entirely. A derivation
    that keys by origin module reintroduces `_combine`, `_emit`, `_context` and
    sixteen more.
    """
    reached = {q for q, _home, _line, _doc in _population()}
    intruders = NOT_THE_PAGE_S_NAMES & reached
    assert not intruders, (
        f"{sorted(intruders)} are not names the page emits. The population is keyed by "
        f"origin module rather than by the module `api.md` declares the symbol through."
    )
    private = sorted(q for q in reached if any(seg.startswith("_") for seg in q.split(".")))
    assert not private, (
        f"{len(private)} qualname(s) carry a private path segment, e.g. {private[:3]}. "
        f"`filters: ['!^_']` keeps those off the page, so no rendered anchor can contain one."
    )


def test_each_docstring_is_reached_under_exactly_one_name():
    """The only pin here that catches OVER-reach, and it needs no built site.

    Every other pin tests whether a NAME is present or absent, so none of them
    fires when the population gains names nobody thought to list. That is the
    failure mode of applying `filters` without also requiring an alias to be
    exported: a re-exported symbol is then reached under every module that
    imports it, and `ConfigurationError` alone arrives four times. Measured, that
    rule yields 438 names over 271 docstring sites; the correct rule is a
    bijection, 183 over 183.
    """
    per_site: dict[tuple[str, int], list[str]] = {}
    for qualname, home, line, _doc in _population():
        per_site.setdefault((str(home), line), []).append(qualname)
    repeated = {site: names for site, names in per_site.items() if len(names) > 1}
    assert not repeated, (
        f"{len(repeated)} docstring(s) are reached under more than one name, e.g. "
        f"{sorted(repeated.values())[:2]}. An alias is being admitted without checking "
        f"that its module exports it, so the population carries names the page never emits."
    )


def test_unmodelled_handler_options_raise_rather_than_silently_narrowing(tmp_path):
    """A membership option this derivation cannot model must stop, not no-op.

    The `ast` traversal's defects were silent for exactly this reason: a rule the
    renderer applied and the gate did not made no noise.
    """
    yml = tmp_path / "mkdocs.yml"
    yml.write_text(
        "plugins:\n"
        "  - mkdocstrings:\n"
        "      handlers:\n"
        "        python:\n"
        "          options:\n"
        '            filters: ["!^_"]\n'
        "            show_submodules: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="show_submodules"):
        cdc._handler_options(yml)


def test_an_option_written_at_its_own_default_is_a_no_op(tmp_path):
    """The defaults are READ from `PythonOptions`, not transcribed.

    `extensions` and `preload_modules` carry `default_factory=list`, so their
    real default is `[]`. A transcribed `None` made writing the documented
    default explicitly RAISE -- the harmless value treated as dangerous, which is
    the shape of the bug this refusal was written to prevent.
    """
    for key, value in (
        ("extensions", "[]"),
        ("preload_modules", "[]"),
        ("show_submodules", "false"),
        ("allow_inspection", "true"),
    ):
        yml = tmp_path / f"mkdocs-{key}.yml"
        yml.write_text(
            "plugins:\n  - mkdocstrings:\n      handlers:\n        python:\n"
            '          options:\n            filters: ["!^_"]\n'
            f"            {key}: {value}\n",
            encoding="utf-8",
        )
        assert cdc._handler_options(yml)["filters"] == ["!^_"], f"{key}: {value} was refused"


def test_filters_are_read_from_mkdocs_yml_rather_than_written_here():
    """The population is a property of `mkdocs.yml` plus `api.md`, read not copied."""
    assert cdc._handler_options()["filters"] == ["!^_"], (
        "mkdocs.yml's python-handler `filters` is no longer what this gate reads; "
        "the population and the page have diverged at the configuration surface"
    )


def test_the_search_root_is_the_one_mkdocs_gives_the_renderer():
    """`handlers.python.paths` is a handler key, not an option, and it is READ.

    A refusal keyed on it could never fire, because it does not appear under
    `options`. The failure it guards is quieter than the ones the refusal list
    covers: the gate would scan a tree the page does not render from, and every
    count would look right.
    """
    assert cdc._handler_paths() == [cdc.SRC_ROOT], (
        f"mkdocs.yml points the python handler at {cdc._handler_paths()} while this gate's "
        f"default root is {cdc.SRC_ROOT}; the gate and the page are reading different trees"
    )


def test_reading_option_defaults_does_not_broadcast_upstream_deprecations():
    """Importing the handler emits 350 pydantic deprecations; the gate must not.

    Measured: `griffe` imports at 0 warnings, `mkdocstrings_handlers.python` at
    350, and an unsilenced module-scope import made this gate exit 1 under
    `-W error::DeprecationWarning`. The cache is reset first because a earlier
    test may already have paid the import, which would make this pass vacuously.
    """
    import warnings as _warnings

    cdc._PYTHON_OPTION_FIELDS = None
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        assert cdc._option_default("extensions") == []
    leaked = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not leaked, f"{len(leaked)} upstream deprecation(s) escaped, e.g. {leaked[0].message}"


def test_absent_paths_falls_back_where_the_renderer_looks(tmp_path):
    """`PythonConfig.paths` defaults to `['.']` -- the CONFIG dir, not `src`.

    The one branch that fires when configuration is absent is the one where a
    `src` fallback would reintroduce the hardcode `_handler_paths` removes.
    """
    yml = tmp_path / "mkdocs.yml"
    yml.write_text(
        "plugins:\n  - mkdocstrings:\n      handlers:\n        python:\n"
        '          options:\n            filters: ["!^_"]\n',
        encoding="utf-8",
    )
    assert cdc._handler_paths(yml) == [tmp_path.resolve()]


def test_multiple_declared_roots_are_all_read_in_order(tmp_path):
    """Multi-root is legal config and is supported, not refused.

    Declaration order is preserved because `search_paths` order is what breaks a
    tie when two roots hold the same module name.
    """
    yml = tmp_path / "mkdocs.yml"
    yml.write_text(
        "plugins:\n  - mkdocstrings:\n      handlers:\n        python:\n"
        "          paths: [src, vendor]\n"
        '          options:\n            filters: ["!^_"]\n',
        encoding="utf-8",
    )
    assert cdc._handler_paths(yml) == [(tmp_path / "src").resolve(), (tmp_path / "vendor").resolve()]


def test_every_manifested_module_contributes_at_least_one_member():
    """A module contributing zero members IS the STRICT signature.

    Exercised through the product's own guard rather than by re-deriving the
    owner from a qualname, which is ambiguous at different nesting depths.
    """
    import ast as _ast

    real = cdc.rendered_docstrings
    pop = real(cdc.SRC_ROOT, cdc.API_PAGE)
    assert pop, "population is empty"

    def strict_names(tree):
        for node in tree.body:
            if isinstance(node, _ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                return True
        return False

    modules = cdc.public_modules(cdc.API_PAGE)
    without_all = [
        m
        for m in modules
        if (p := cdc._module_file(m, cdc.SRC_ROOT)) is not None
        and not strict_names(_ast.parse(p.read_text(encoding="utf-8")))
    ]
    assert without_all, (
        "no manifested module lacks __all__, so this test can no longer "
        "discriminate RESOLVED from STRICT — re-pin it against a module that does"
    )
    reached = {q.rsplit(".", 1)[0] for q, _h, _l, _d in pop}
    for m in without_all:
        assert any(r == m or r.startswith(m + ".") for r in reached), (
            f"{m} declares no __all__ and contributed nothing. That is the STRICT "
            f"signature -- check the population derivation, not the module."
        )


def test_empty_population_raises_rather_than_passing_vacuously(tmp_path):
    """Non-vacuity: an unresolvable population is an error, never a clean scan."""
    api = tmp_path / "api.md"
    api.write_text("# API Reference\n\n::: nothing.here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nothing to check"):
        cdc.rendered_docstrings(tmp_path / "src", api)


def test_citations_in_docstrings_are_gated_not_advisory():
    """Layer 1's boundary, asserted on ROUTING rather than on a live defect.

    A synthetic docstring is used deliberately: asserting that the real
    population currently yields a citation is green only until the source is
    cleaned, which turns a repair into a red build.
    """
    synthetic = "See workflow.py:2326 for the call site.\n"
    codes = {c for c, _p, _l, _e in cdc._binary_findings(Path("<synthetic>"), synthetic)}
    assert "bare-line-citation" in codes, "the binary-class detector no longer fires"
    live = {c for c, _p, _l, _e in cdc.scan_rendered_docstrings()}
    assert live <= {
        "bare-line-citation",
        "self-declared-placeholder",
        "deferred-to-later-task",
        "deferred-release-content",
        "coming-soon",
        "bare-todo-marker",
        "stub-self-declaration",
        "unfilled-date-placeholder",
    }, f"a non-binary class reached the gated population: {sorted(live - {'bare-line-citation'})}"


def test_prose_classes_do_not_reach_the_gated_population():
    """Layer 2's boundary: `### D22b` prose classes stay out of the exit code."""
    codes = {c for c, _p, _l, _e in cdc.scan_rendered_docstrings()}
    assert "em-dash" not in codes and "banned-word-estate" not in codes


#: What each marker-carrying page DECLARES exempt, pinned BY NAME rather than
#: counted. `main()`'s skip line reports HOW MANY marker files exist and never
#: WHAT they declare, so a page adding `exempt=binary` would exempt itself from
#: the two classes this project gates on shipped metadata and on rendered
#: docstrings, and no existing mechanism would say so. A count cannot see it; a
#: named fixture goes red on the edit that does it.
DECLARED_EXEMPTIONS = {
    "docs/reference/config-schema.md": {"prose"},
    "docs/contributing.md": {"prose"},
}


def _marker_pages():
    docs = _REPO / "docs"
    return {
        md.relative_to(_REPO).as_posix(): set(cdc._exempt_groups(md.read_text(encoding="utf-8")))
        for md in cdc.generated_files(docs) + cdc.personal_voice_files(docs)
    }


def test_marker_pages_declare_exactly_the_pinned_exemptions():
    """A new marker page, or a changed declaration, goes red here.

    The generated page is gitignored build output: ABSENT on a fresh clone and
    present after `mkdocs build`. Its absence is not a failure; a mismatch is, and
    so is a marker page this fixture has never heard of.
    """
    seen = _marker_pages()
    assert "docs/contributing.md" in seen, "the tracked personal-voice page is not being detected"
    assert seen == {k: v for k, v in DECLARED_EXEMPTIONS.items() if k in seen}


def test_no_marker_page_exempts_the_binary_classes():
    """The omission the repair CONSISTS of, asserted rather than trusted.

    `binary` stays spellable so the class split lives in the pages rather than in
    the module -- which means nothing but this test stops a page declaring it.
    """
    seen = _marker_pages()
    assert seen, "no marker pages found: this assertion would pass vacuously"
    offenders = sorted(page for page, groups in seen.items() if "binary" in groups)
    assert offenders == [], (
        f"{offenders} declare `exempt=binary`. Placeholder leakage and bare line "
        f"citations are gated on shipped metadata and on rendered docstrings; a page "
        f"exempting them needs its own ruling, not a marker edit."
    )

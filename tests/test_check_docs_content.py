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

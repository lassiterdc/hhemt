"""Witness for the render-path `except` narrowing (S19).

No synthetic model: monkeypatching `resolve_active_reporting_set` in
`hhemt.config.report` and calling `resolve_render_path_category_order` directly
is the whole fixture, so this runs in the cheap pure-config tier.
"""

from __future__ import annotations

import pytest


def _stub_analysis(is_sensitivity: bool = True):
    """The two fields the category-order block reads before it resolves."""
    import types

    return types.SimpleNamespace(
        _cfg_report=types.SimpleNamespace(disabled_renderers=[], reporting_set=["benchmarking"]),
        cfg_analysis=types.SimpleNamespace(toggle_sensitivity_analysis=is_sensitivity),
    )


def _patch(monkeypatch, exc):
    import hhemt.config.report as report_mod

    def _raise(*_a, **_k):
        raise exc

    monkeypatch.setattr(report_mod, "resolve_active_reporting_set", _raise)


def test_soft_degrades_on_a_config_error(monkeypatch, caplog):
    """ARM A -- green BEFORE and AFTER. The differently-positioned satisfying arm:
    it catches an over-narrowing that would reverse F-I-3's fail-soft ruling, which
    no violating arm can."""
    from hhemt.exceptions import ConfigurationError
    from hhemt.render_category_order import resolve_render_path_category_order
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    _patch(monkeypatch, ConfigurationError(field="reporting_set", message="unknown set", config_path=None))
    with caplog.at_level("WARNING"):
        order = resolve_render_path_category_order(_stub_analysis())
    assert order == list(get_reporting_set("default").category_order)
    assert sum("reporting_set resolution failed" in r.message for r in caplog.records) == 1


def test_soft_degrades_on_a_composition_error(monkeypatch, caplog):
    """ARM A2 -- the POST-SET live user-origin class, which the first-drafted witness
    lacked.

    A config naming an incompatible pair (mixed shape, or two sets gating one renderer
    differently) raises here on the render-without-run path, because run-entry
    validation never ran. It is user-origin, so it soft-degrades like a
    ConfigurationError. If the refute round rules the other way, this arm inverts and
    the class comes out of the caught tuple -- a one-token change in both places.
    """
    from hhemt.render_category_order import resolve_render_path_category_order
    from hhemt.report_renderers._reporting_sets import (
        ReportingSetCompositionError,
        get_reporting_set,
    )

    _patch(monkeypatch, ReportingSetCompositionError("differing shapes"))
    with caplog.at_level("WARNING"):
        order = resolve_render_path_category_order(_stub_analysis())
    assert order == list(get_reporting_set("default").category_order)


def test_propagates_an_attribute_error(monkeypatch):
    """ARM B -- the defect class the narrowing exists to stop swallowing.

    A raw-dict _cfg_report has no .reporting_set attribute. That can only be an hhemt
    defect, and pre-narrowing the bare `except Exception` swallowed it: the report
    rendered, looked complete, and carried a sidebar order the user did not select.
    """
    from hhemt.render_category_order import resolve_render_path_category_order

    _patch(monkeypatch, AttributeError("'dict' object has no attribute 'reporting_set'"))
    with pytest.raises(AttributeError):
        resolve_render_path_category_order(_stub_analysis())


def test_propagates_a_type_error_regression_guard(monkeypatch):
    """ARM B2 -- a REGRESSION GUARD on the one-commit boundary, not a live class.

    After the full S19 set lands, spec 1's type dispatch and its isinstance(name, str)
    guard convert every non-string shape into a ConfigurationError, so a TypeError is
    nearly unreachable here. It IS reachable in the window where the field validator
    landed and the resolver widening did not -- the exact split the one-commit boundary
    forbids. This arm is red in that window and green on either side of it, which is
    what makes it a guard on the boundary rather than a test of a behaviour.
    """
    from hhemt.render_category_order import resolve_render_path_category_order

    _patch(monkeypatch, TypeError("unhashable type: 'list'"))
    with pytest.raises(TypeError):
        resolve_render_path_category_order(_stub_analysis())


def test_both_twins_delegate_to_the_shared_helper():
    """Neither twin may retain an inline copy of the resolution block.

    Before the extraction these two blocks were byte-identical over 2031 characters,
    which is the shape most likely to be repaired at one site only -- and the S19
    narrowing is a repair to exactly this region. The behavioural arms above test the
    shared helper once, so this is what keeps "both twins were repaired" checkable
    rather than assumed.

    Asserted on the SOURCE rather than by calling both render paths, because calling
    them needs a rendered report on disk and this must stay in the cheap tier.
    """
    import inspect

    import hhemt.analysis as analysis_mod
    import hhemt.sensitivity_analysis as sensitivity_mod

    for mod in (analysis_mod, sensitivity_mod):
        src = inspect.getsource(mod)
        assert "_category_order = resolve_render_path_category_order(self)" in src, (
            f"{mod.__name__} does not CALL the shared helper and bind its result; a bare "
            "import satisfies a name check and raises NameError at render"
        )
        assert "reporting_set resolution failed" not in src, (
            f"{mod.__name__} still carries an inline copy of the resolution block. "
            "NOTE: this greps the STABLE PREFIX of the log literal, not the closing "
            "clause -- A5's message change recases 'Falling back' and a lowercase grep "
            "would go vacuously green. It still greps a log literal, so a re-inlined "
            "copy with a fully reworded warning passes: a bounded residual, accepted "
            "because the extraction makes re-inlining deliberate rather than accidental."
        )

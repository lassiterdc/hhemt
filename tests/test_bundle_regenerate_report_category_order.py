"""A bundle re-render ranks its sidebar by the bundled config's reporting set.

``Bundle.regenerate_report`` re-renders a report from a bundle and applies the React
surgery, whose sort comparator ranks sidebar categories by a ``category_order``. The
analysis render paths pass the active reporting set's full order. If the bundle path
passes none, the surgery falls back to its own five-name default, which lacks
"Workflow performance" and "Metadata". Those two then rank 99 and sort alphabetically,
so the re-rendered sidebar ends "Metadata", "Workflow performance" while the original
report ends "Workflow performance", "Metadata".

Every step before the surgery call is replaced with a stub, and the two surgery entry
points are replaced with recorders. No solver runs and no Snakemake process starts.
``regenerate_report`` runs unbound against a stand-in carrying only what it reads.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import hhemt.bundle as bundle
import hhemt.bundle.snakefile_generator as snakefile_generator
import hhemt.provenance as provenance
import hhemt.report_plot_ids as report_plot_ids
import hhemt.report_renderers._react_surgery as react_surgery
import hhemt.workflow as workflow
from hhemt.config.report import DEFAULT_REPORT_CONFIG, report_config
from hhemt.report_renderers._reporting_sets import get_reporting_set

_FORMATS = ("html", "zip")


def _report_cfg(reporting_set):
    """A validated report_config naming ``reporting_set``; ``None`` omits the key."""
    data = DEFAULT_REPORT_CONFIG.model_dump()
    if reporting_set is None:
        data.pop("reporting_set")
    else:
        data["reporting_set"] = reporting_set
    return report_config.model_validate(data)


def _regenerate_and_capture(monkeypatch, tmp_path, fmt, cfg_analysis):
    """Run ``regenerate_report`` with everything before the surgery stubbed.

    Returns the ``category_order`` the surgery entry point for ``fmt`` received.
    """
    monkeypatch.setattr(provenance, "assert_plots_match_running_build", lambda *_a, **_kw: None)
    monkeypatch.setattr(report_plot_ids, "member_labels_from_status", lambda _d: {})
    monkeypatch.setattr(report_plot_ids, "event_sidebar_labels_from_status", lambda _d: {})
    monkeypatch.setattr(workflow, "_emit_report_artifacts", lambda *_a, **_kw: None)
    monkeypatch.setattr(workflow, "_brand_theme_css_map", lambda _t: {})
    monkeypatch.setattr(snakefile_generator, "write_regeneration_snakefile", lambda *_a, **_kw: None)
    monkeypatch.setattr(bundle, "run_subprocess_with_tee", lambda *_a, **_kw: SimpleNamespace(returncode=0))

    captured = {}

    def _html(text, **kwargs):
        captured.update(kwargs)
        return text

    def _zip(_path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(react_surgery, "apply_post_process_surgery", _html)
    monkeypatch.setattr(react_surgery, "apply_post_process_surgery_to_zip", _zip)

    (tmp_path / f"analysis_report.{fmt}").write_text("rendered")
    stand_in = SimpleNamespace(
        _root=tmp_path,
        _cfg_analysis=cfg_analysis,
        _read_static_backend=lambda: "plotly",
        _assert_render_inputs_present=lambda: None,
    )
    bundle.Bundle.regenerate_report(stand_in, format=fmt)
    # None when the call site passed no order, which is the surgery's own default.
    return captured.get("category_order")


def _cfg_analysis(reporting_set, *, sensitivity=False):
    return SimpleNamespace(
        report=_report_cfg(reporting_set),
        toggle_sensitivity_analysis=sensitivity,
        brand_theme=None,
        analysis_id="bundled-analysis",
    )


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("set_name", ["benchmarking", "combined"])
def test_rerender_passes_the_named_sets_full_order(monkeypatch, tmp_path, fmt, set_name):
    """Both surgery entry points receive the bundled reporting set's full category order.

    Property: when the bundled config names a reporting set, the ``category_order``
    passed to the surgery for that format equals the set's registered
    ``category_order``.

    Class: every bundled config that names a registered set, in each output format.
    ``benchmarking`` stands for the seven-name standard order. ``combined`` stands for
    an order that differs from every other set, so the test also fails if the bundle
    path passes a fixed order instead of the resolved one.

    Why it is required: the re-rendered sidebar must rank its categories the way the
    original render did. A missing order makes the surgery fall back to its five-name
    default, which reverses the last two categories.

    What kills it: passing no ``category_order`` (the surgery receives ``None``), or
    passing a fixed list such as the registry ``default`` set's order, which fails the
    ``combined`` cases.

    A second correct implementation under which it still passes: resolving the set with
    ``resolve_active_reporting_set`` directly and passing ``list(active.category_order)``,
    without the shared render-path helper.
    """
    order = _regenerate_and_capture(monkeypatch, tmp_path, fmt, _cfg_analysis(set_name))

    expected = list(get_reporting_set(set_name).category_order)
    assert list(order or []) == expected
    assert list(order or []) != list(react_surgery._DEFAULT_CATEGORY_ORDER)


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    ("sensitivity", "resolved_set"),
    [(False, "default"), (True, "sensitivity")],
    ids=["standard", "sensitivity"],
)
def test_rerender_without_a_named_set_uses_the_sentinel_resolution(
    monkeypatch, tmp_path, fmt, sensitivity, resolved_set
):
    """A bundled config with no ``reporting_set`` key resolves through the sentinel.

    Property: when the bundled report config omits ``reporting_set``, its default
    selection resolves by ``toggle_sensitivity_analysis`` (the standard set when False,
    ``sensitivity`` when True), and that set's full order is passed.

    Class: every bundled config that does not name a reporting set.

    Why it is required: most configs name no set, so this is the common bundle. Its
    re-render must still carry the full order rather than the surgery's default.

    What kills it: passing no ``category_order``. The ``sensitivity`` set's order is
    currently identical to the standard set's, so this test does not distinguish a
    resolution that ignores the sensitivity flag. Not established: which of the two
    sets was resolved, only that the full seven-name order was passed.

    A second correct implementation under which it still passes: calling
    ``resolve_reporting_set_names`` and composing the result by hand.
    """
    cfg = _cfg_analysis(None, sensitivity=sensitivity)

    order = _regenerate_and_capture(monkeypatch, tmp_path, fmt, cfg)

    assert list(order or []) == list(get_reporting_set(resolved_set).category_order)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_rerender_with_an_unknown_set_degrades_to_the_registry_default(monkeypatch, tmp_path, fmt, caplog):
    """A bundled config naming an unregistered set degrades to the registry default order.

    Property: when the bundled config names a set the registry does not hold, the order
    passed is the registry ``default`` set's full order, and one warning is logged.

    Class: every bundled config whose ``reporting_set`` cannot be resolved. Such a config
    loads, because the selector is validated at run entry rather than at field
    construction.

    Why it is required: the degraded render must keep all seven categories. The
    surgery's own default would drop two of them to the alphabetical tail.

    What kills it: passing ``None`` on a resolution failure, or letting the
    ``ConfigurationError`` escape the re-render.

    A second correct implementation under which it still passes: catching the
    resolution error at the bundle call site and passing the registry default order.
    """
    with caplog.at_level(logging.WARNING):
        order = _regenerate_and_capture(monkeypatch, tmp_path, fmt, _cfg_analysis("no-such-set"))

    assert list(order or []) == list(get_reporting_set("default").category_order)
    assert "Falling back to 'default' category order" in caplog.text

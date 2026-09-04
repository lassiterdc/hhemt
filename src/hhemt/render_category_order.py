"""Render-path category-order resolution, shared by the two report-rendering twins.

Extracted from `TRITONSWMM_analysis.render_report` and
`TRITONSWMM_sensitivity_analysis.render_report`, where it existed as two
byte-identical blocks before this module. It lives HERE rather than in either
twin because the two already form an import cycle -- `sensitivity_analysis`
imports `hhemt.analysis` at module level and `analysis` imports
`TRITONSWMM_sensitivity_analysis` back -- so a shared definition in either one
would convert a working lazy cycle into a hard one.

It is also deliberately NOT sited in `report_renderers/_react_surgery.py`, which
carries its own `_DEFAULT_CATEGORY_ORDER`. That constant is SHORTER than the
registry's `default` set (6 entries against 8; it omits "Workflow performance"
and "Metadata") and must not be reused as this fallback, or every degraded
render would silently drop two sidebar categories.
"""

from __future__ import annotations

import logging


def resolve_render_path_category_order(analysis) -> list[str]:
    """Return the sidebar category order for ``analysis``'s rendered report.

    `render_report()` is dominantly invoked from `render_report_runner.main()` on
    a FRESH analysis that never called `run()`, so `_active_reporting_set` may not
    exist. When it does not, resolve from config alone -- no CSV cross-validation
    is available at render time.

    Fails SOFT on a USER config error (SE F-I-3): the render path bypasses
    `validate_active_reporting_set`, so a stale, unknown or incompatible
    `reporting_set` would otherwise surface as an opaque Snakemake rule failure.
    Degrade to the registry's "default" order plus one warning instead.

    The caught tuple is NARROW by design (S19). An `AttributeError` here can only
    be an hhemt defect -- the only source on this path is `cfg.reporting_set`
    against something that is not a `report_config` -- and degrading to the wrong
    sidebar while the report renders and looks complete is the worst outcome for a
    defect. `ReportingSetCompositionError` is caught because it is user-origin: it
    means the config named an incompatible pair of sets.

    Parameters
    ----------
    analysis
        Duck-typed. Deliberately UNTYPED and deliberately not a Protocol: this module
        is sited outside both twins so it cannot import them, and a Protocol invites
        the next author to import it at runtime, which is the cycle the siting exists
        to avoid. The contract is these four attributes, and nothing else is read:

        * ``_active_reporting_set``  -- optional; the run-entry ReportingSet. Present
          only after ``run()``. When present it is returned directly and the remaining
          three are never touched.
        * ``_cfg_report``            -- optional; the run-entry report_config snapshot.
        * ``cfg_analysis.report``    -- the fallback report_config when ``_cfg_report``
          is absent.
        * ``cfg_analysis.toggle_sensitivity_analysis`` -- drives the sentinel branch.

        Both ``TRITONSWMM_analysis`` and ``TRITONSWMM_sensitivity_analysis`` satisfy it;
        a raw dict does NOT, and that is deliberate -- an AttributeError from a
        dict-shaped argument propagates as an hhemt defect rather than degrading.
    """
    from .config.report import resolve_active_reporting_set
    from .exceptions import ConfigurationError
    from .report_renderers._reporting_sets import (
        ReportingSetCompositionError,
        get_reporting_set,
    )

    active = getattr(analysis, "_active_reporting_set", None)
    if active is not None:
        return list(active.category_order)

    try:
        cfg_report = getattr(analysis, "_cfg_report", None)
        if cfg_report is None:
            cfg_report = analysis.cfg_analysis.report
        active = resolve_active_reporting_set(
            cfg_report,
            is_sensitivity=analysis.cfg_analysis.toggle_sensitivity_analysis,
        )
    except (ConfigurationError, ReportingSetCompositionError, KeyError) as exc:
        logging.getLogger(__name__).warning(
            "render-path reporting_set resolution failed (%s); this config did NOT pass "
            "analysis.run() entry validation, so the incompatibility was never reported "
            "there. Falling back to 'default' category order.",
            exc,
        )
        active = get_reporting_set("default")
    return list(active.category_order)

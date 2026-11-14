"""The `default` reporting-set sentinel resolves by analysis shape, and to a set that fits it.

`report.reporting_set` defaults to the sentinel `default`, which is not a set but a
request for "whichever set suits this analysis". On an event ensemble it resolves to
the registered `default` set; on a sensitivity analysis it resolves to the
`sensitivity` set. It does NOT resolve to `benchmarking`, because that set requires
compute-config axes a general sweep does not vary, so a default-configured sweep
would be refused at `analysis.run()` entry.

These tests read only the config layer. They compile no solver and run no simulation.
"""

import pytest

from hhemt.config.report import (
    report_config,
    resolve_reporting_set_names,
    validate_active_reporting_set,
)

_EXPECTED = {False: "default", True: "sensitivity"}


@pytest.mark.parametrize("is_sensitivity", [False, True], ids=["event-ensemble", "sensitivity"])
def test_default_sentinel_resolves_to_the_set_matching_the_analysis_shape(is_sensitivity):
    """Property: the sentinel resolves to `default` on an event ensemble and to
    `sensitivity` on a sensitivity analysis.

    Class: both values of the analysis-shape flag. The flag is the resolver's only
    input that varies the answer, so the two cases cover the whole class.

    Why required: a sensitivity analysis left on the default selection must get a
    sweep report, never the compute-config benchmarking report.

    What kills it: returning "benchmarking" for a sensitivity analysis (the previous
    rule) reds the sensitivity case; hardcoding either name for both cases reds
    exactly one case.

    A second correct implementation that still passes: looking the target up from a
    {shape: set name} mapping instead of a conditional expression.
    """
    assert resolve_reporting_set_names("default", is_sensitivity=is_sensitivity) == (_EXPECTED[is_sensitivity],)


@pytest.mark.parametrize("is_sensitivity", [False, True], ids=["event-ensemble", "sensitivity"])
def test_default_sentinel_passes_the_run_entry_shape_gate(is_sensitivity):
    """Property: an untouched report config is accepted by run-entry validation on
    both analysis shapes, and the accepted set is the one the sentinel names.

    Class: both analysis shapes, using the config's own default with no
    reporting_set given.

    Why required: run-entry validation refuses a set whose declared shape differs
    from the analysis. Resolving the sentinel to a set of the wrong shape, or to one
    whose required axes the sweep does not vary, would refuse every default-configured
    analysis of that shape before any compute is committed.

    What kills it: resolving to `default` on a sensitivity analysis (refused on
    shape) or to `benchmarking` (refused on required axes when none are varied).

    A second correct implementation that still passes: any resolution that returns a
    shape-matched, axis-free set object whose name is the expected one.
    """
    active = validate_active_reporting_set(
        report_config(),
        is_sensitivity=is_sensitivity,
        sensitivity_csv_path=None,
        varied_axes=frozenset(),
    )
    assert active.name == _EXPECTED[is_sensitivity]

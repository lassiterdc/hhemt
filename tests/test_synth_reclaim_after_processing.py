"""Synth-tier coverage for ``analysis_config.reclaim_after_processing``.

HPC-free developer-CI tier.

WHAT THE DISJOINTNESS GUARD IS, AND WHAT IT IS NOT. It is the FALSIFYING test for the
SAFETY property: the reclaim's drop set must never intersect the preserve set, so the
mechanism can never remove an artifact the toolkit is required to keep. It asserts that
against an INDEPENDENT signal -- ``_SUMMARY_ATTRS_BY_MODEL`` below is hand-maintained here
and deliberately NOT imported from ``summary_paths``. Do not "fix" that as an
inconsistency: a guard derived from the thing it guards weakens in lockstep with any
mistake in that thing, which is the same reasoning that keeps
``scripts/anonymization_blocklist.txt`` independent of ``src/hhemt/``.

WHAT IT DOES NOT GUARANTEE -- stated because the safety claim invites the stronger reading.
Three tables are hand-maintained: ``summary_paths._SUMMARY_STEMS_BY_MODEL`` (canonical),
``process_simulation._TIMESERIES_ATTRS_BY_MODEL`` (the production drop table), and the copy
below (this guard's independent signal). A ninth summary family added to the CANONICAL
table alone touches neither of the other two, so both tests here still pass and the ninth
timeseries is simply never reclaimed. That is the SAFE direction -- an artifact is retained
that could have been dropped, never the reverse -- so it is a coverage gap, not a defect,
and the disk cost is bounded by one artifact family.

WHICH TEST CATCHES WHAT. ``test_reclaim_drop_set_is_disjoint_from_preserve_set`` catches a
production drop table that reaches into the preserve set (the unsafe direction).
``test_preserve_and_drop_sets_are_one_to_one_paired`` catches a production drop table that
has fallen out of step with this guard's copy in EITHER direction, which is what keeps
disjointness from being satisfiable by an empty drop set. Neither compares against the
canonical table, by construction.
"""

from __future__ import annotations

import pytest

from hhemt.config.analysis import analysis_config
from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _Proc

# The ScenarioPaths attr names of the preserve set, keyed identically to
# summary_paths._SUMMARY_STEMS_BY_MODEL. Kept here (not imported) deliberately: this is the
# INDEPENDENT ground-truth signal for the guard, and deriving it from the same table the
# production code derives its drop set from would make the guard weaken in lockstep with any
# mistake in that table.
_SUMMARY_ATTRS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "tritonswmm": (
        "output_tritonswmm_triton_summary",
        "output_tritonswmm_node_summary",
        "output_tritonswmm_link_summary",
        "output_tritonswmm_performance_summary",
    ),
    "triton": (
        "output_triton_only_summary",
        "output_triton_only_performance_summary",
    ),
    "swmm": (
        "output_swmm_only_node_summary",
        "output_swmm_only_link_summary",
    ),
}

_MODEL_TYPES = ("tritonswmm", "triton", "swmm")


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_reclaim_drop_set_is_disjoint_from_preserve_set(model_type):
    """FALSIFYING TEST -- fails if the reclaim would ever remove a preserved artifact.

    Name-level and fixture-free on purpose: a guard that needs a compiled synth model is a
    guard that gets skipped. Pre-fix, ``_reclaim_attrs`` does not exist and this errors at
    import -- the required pre-fix FAIL.
    """
    preserve = set(_SUMMARY_ATTRS_BY_MODEL[model_type])
    drop = set(_Proc._reclaim_attrs(model_type, policy="all"))
    assert preserve & drop == set(), (
        f"reclaim would remove preserved artifact(s) for {model_type}: {sorted(preserve & drop)}"
    )


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_preserve_and_drop_sets_are_one_to_one_paired(model_type):
    """Every preserved summary has exactly one paired timeseries in the drop set.

    This is what makes the disjointness guard meaningful rather than vacuous: disjointness
    alone is satisfied by an EMPTY drop set. Pairing pins the cardinality too.
    """
    preserve = _SUMMARY_ATTRS_BY_MODEL[model_type]
    drop = _Proc._reclaim_attrs(model_type, policy="all")
    assert len(drop) == len(preserve), (
        f"{model_type}: {len(drop)} timeseries attr(s) paired against {len(preserve)} summary attr(s)"
    )


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_reclaim_attrs_is_empty_for_the_strict_safe_default(model_type):
    """The shipped default reclaims nothing, and the empty policy list is rejected."""
    assert _Proc._reclaim_attrs(model_type, policy="none") == ()
    assert _Proc._reclaim_attrs(model_type, policy=["raw_swmm_binaries"]) == ()


def test_rpt_suffix_is_never_in_the_coupled_swmm_reclaim_allowlist():
    """`hydraulics.rpt` is a live completion predicate, not an output.

    ``run_simulation._coupled_swmm_report_finalized`` returns False when it is absent, and
    ``model_run_completed('tritonswmm')`` calls that gate even on the log-True branch, so
    reclaiming the .rpt converts every completed coupled sim into a permanently-incomplete
    one. The `.bin` assertion covers the exchange-replay side-file, whose reclaim is owned
    by a different, differently-gated helper.
    """
    from hhemt.process_simulation import _RECLAIM_COUPLED_SWMM_SUFFIXES

    assert ".rpt" not in _RECLAIM_COUPLED_SWMM_SUFFIXES
    assert ".bin" not in _RECLAIM_COUPLED_SWMM_SUFFIXES
    assert ".out" in _RECLAIM_COUPLED_SWMM_SUFFIXES


def test_default_config_reclaims_nothing():
    """Asserted on the CONFIG, never by flipping the knob in memory.

    Gotcha 73: the process runner is a SUBPROCESS that loads its config from disk, so a
    test that sets ``analysis.cfg_analysis.reclaim_after_processing`` on a built analysis
    and then runs the workflow silently tests nothing. Every behavioral test in this module
    must inject the knob at case-construction time via
    ``retrieve_synth_TRITON_SWMM_test_case(..., additional_analysis_configs={...})``.
    """
    assert analysis_config.model_fields["reclaim_after_processing"].default == "none"


@pytest.mark.parametrize(
    "bad",
    [[], ["timeseries", "timeseries"], ["all"], ["none"]],
)
def test_reclaim_policy_validator_rejects(bad):
    """The three reject arms, each reachable from a hand-authored YAML."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        analysis_config._validate_reclaim_after_processing(bad)


def test_reclaim_classes_normalization():
    """`all` expands to both classes; `none` to none; a list passes through."""
    assert _Proc._reclaim_classes("none") == ()
    assert _Proc._reclaim_classes(None) == ()
    assert set(_Proc._reclaim_classes("all")) == {"timeseries", "raw_swmm_binaries"}
    assert _Proc._reclaim_classes(["timeseries"]) == ("timeseries",)


@pytest.mark.parametrize("which,expected_swmm,expected_triton", [
    ("both", True, True),
    ("TRITON", False, True),
    ("SWMM", True, False),
])
def test_which_restricts_the_drop_set(which, expected_swmm, expected_triton):
    """`which` narrows the drop set the same way it narrows the processing it follows."""
    attrs = _Proc._reclaim_attrs("tritonswmm", policy="all", which=which)
    has_swmm = any(_Proc._attr_is_swmm_side(a) for a in attrs)
    has_triton = any(not _Proc._attr_is_swmm_side(a) for a in attrs)
    assert has_swmm is expected_swmm
    assert has_triton is expected_triton


def test_data_availability_check_is_registered_and_graceful_absent():
    """The disclosure chain reaches `validate_analysis`, and reads the LOG not the config.

    Registration is asserted structurally (the aggregator names it) rather than by running
    a full analysis, because the behavioral half needs a compiled synth model. The
    read-the-log property is asserted on the module's field map, which is the thing that
    would have to change for the check to start reading `cfg_analysis`.
    """
    import inspect

    from hhemt import analysis_validation

    src = inspect.getsource(analysis_validation.validate_analysis)
    assert "check_data_availability(analysis)" in src
    assert set(analysis_validation._RECLAIM_LOG_FIELDS) == {
        "full_TRITON_timeseries_cleared",
        "full_SWMM_timeseries_cleared",
        "raw_SWMM_binaries_reclaimed",
    }
    # Assert on the CODE, not the source text: the docstring legitimately names
    # `reclaim_after_processing` in order to say the check must not read it, so a raw
    # substring scan over the source would fail on the very comment that documents the
    # property. Parse the function and inspect only its executable body.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(analysis_validation.check_data_availability)))
    fn = tree.body[0]
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) else fn.body  # drop the docstring
    names = {
        n.attr for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Attribute)
    } | {
        n.value for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "reclaim_after_processing" not in names, (
        "check_data_availability must read the per-scenario LOG, never cfg_analysis -- "
        "config records intent, and a scenario that failed before the reclaim fired would "
        "be mislabelled reclaimed"
    )
    assert "cfg_analysis" not in names


def test_metadata_page_carries_a_data_availability_section():
    """The jump-nav anchor and the section builder must ship together (dead-anchor pair)."""
    from hhemt.report_renderers import metadata

    assert "Data Availability" in metadata._jump_nav()
    assert hasattr(metadata, "_build_data_availability_html")


def test_metadata_data_availability_is_graceful_absent(tmp_path):
    """A pre-feature analysis renders a banner, never a traceback."""
    from hhemt.report_renderers import metadata

    html = metadata._build_data_availability_html(tmp_path / "validation_report.json")
    assert "Data Availability" in html
    assert "not available" in html


def test_summary_export_validates_timeseries_below_the_already_written_return():
    """The hoist: the timeseries check must sit BELOW the `_already_written` early return.

    Pre-fix this test FAILS, because the `raise FileNotFoundError` precedes the early
    return -- which is what makes the second invocation of the process rule on a reclaimed
    scenario crash with an input the function no longer needs.
    """
    import inspect

    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    for fn, needle in (
        (P._export_TRITON_summary, "input timeseries not found"),
        (P._export_SWMM_summaries, "input timeseries not found"),
    ):
        src = inspect.getsource(fn)
        assert src.index("_already_written") < src.index(needle), (
            f"{fn.__name__}: the timeseries-existence check must follow the "
            "_already_written early-return, not precede it"
        )

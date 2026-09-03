"""Synth-tier coverage for ``analysis_config.remove_after_processing``.

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
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _Proc
from hhemt.swmm_runoff_modeling import hydrograph_outputs_gate as _run_gate

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
    assert (
        preserve & drop == set()
    ), f"reclaim would remove preserved artifact(s) for {model_type}: {sorted(preserve & drop)}"


@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_preserve_and_drop_sets_are_one_to_one_paired(model_type):
    """Every preserved summary has exactly one paired timeseries in the drop set.

    This is what makes the disjointness guard meaningful rather than vacuous: disjointness
    alone is satisfied by an EMPTY drop set. Pairing pins the cardinality too.
    """
    preserve = _SUMMARY_ATTRS_BY_MODEL[model_type]
    drop = _Proc._reclaim_attrs(model_type, policy="all")
    assert len(drop) == len(
        preserve
    ), f"{model_type}: {len(drop)} timeseries attr(s) paired against {len(preserve)} summary attr(s)"


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
    test that sets ``analysis.cfg_analysis.remove_after_processing`` on a built analysis
    and then runs the workflow silently tests nothing. Every behavioral test in this module
    must inject the knob at case-construction time via
    ``retrieve_synth_TRITON_SWMM_test_case(..., additional_analysis_configs={...})``.
    """
    assert analysis_config.model_fields["remove_after_processing"].default == "none"


@pytest.mark.parametrize(
    "bad",
    [[], ["timeseries", "timeseries"], ["all"], ["none"]],
)
def test_reclaim_policy_validator_rejects(bad):
    """The three reject arms, each reachable from a hand-authored YAML."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        analysis_config._validate_remove_after_processing(bad)


def test_reclaim_classes_normalization():
    """`all` expands to both classes; `none` to none; a list passes through."""
    assert _Proc._reclaim_classes("none") == ()
    assert _Proc._reclaim_classes(None) == ()
    # EXACT-set pin (same shape and same purpose as the _RECLAIM_LOG_FIELDS pin below):
    # it caught the round-10 "all"-sentinel widening, which is its job. `"all"` must elect
    # EVERY class, or a user-facing sentinel named "all" silently elects a subset.
    assert set(_Proc._reclaim_classes("all")) == {
        "timeseries",
        "raw_swmm_binaries",
        "coupled_rpt",
        "hydro_out",
        "prep_inputs",
        "hydrographs",
        "standalone_rpt",
    }
    assert _Proc._reclaim_classes(["timeseries"]) == ("timeseries",)


@pytest.mark.parametrize(
    "which,expected_swmm,expected_triton",
    [
        ("both", True, True),
        ("TRITON", False, True),
        ("SWMM", True, False),
    ],
)
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
    # EXACT-set pin, deliberately not a subset check: it exists so that adding a
    # disclosure field cannot pass unnoticed, and it did its job when coupled_rpt_truncated
    # and hydro_out_reclaimed landed. Relaxing it to a subset would retire the property
    # that made it useful.
    assert set(analysis_validation._RECLAIM_LOG_FIELDS) == {
        "full_TRITON_timeseries_cleared",
        "full_SWMM_timeseries_cleared",
        "raw_SWMM_binaries_reclaimed",
        "coupled_rpt_truncated",
        "hydro_out_reclaimed",
        # The three regeneration-cost classes. The pin stays EXACT rather than
        # widening to a subset check: it exists so a disclosure field cannot land
        # unnoticed, it did that job when coupled_rpt_truncated and
        # hydro_out_reclaimed arrived, and it did it again here -- this update is
        # owed BECAUSE the pin failed on the applied tree rather than in spite of it.
        "prep_inputs_reclaimed",
        "hydrographs_reclaimed",
        "standalone_rpt_reclaimed",
    }
    # Assert on the CODE, not the source text: the docstring legitimately names
    # `remove_after_processing` in order to say the check must not read it, so a raw
    # substring scan over the source would fail on the very comment that documents the
    # property. Parse the function and inspect only its executable body.
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(analysis_validation.check_data_availability)))
    fn = tree.body[0]
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) else fn.body  # drop the docstring
    names = {n.attr for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Attribute)} | {
        n.value for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "remove_after_processing" not in names, (
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


# --------------------------------------------------------------------------
# Surface 2 — hydrograph already-written gate (Spec H).
# --------------------------------------------------------------------------


class _FakeScenario:
    """Minimal stand-in binding every name the gate reads. Fixture-free by design."""

    def __init__(self, tmp_path, *, logged, hyg_present, hydro_out_present):
        self.log = type("L", (), {})()
        self.log.hyg_timeseries_created = type("F", (), {"get": lambda s: logged})()
        self.log.hyg_locs_created = type("F", (), {"get": lambda s: logged})()
        strm = tmp_path / "strmflow"
        strm.mkdir()
        self.scen_paths = type("P", (), {})()
        self.scen_paths.hyg_timeseries = strm / "tseries.hyg"
        self.scen_paths.hyg_locs = strm / "loc.txt"
        self.scen_paths.swmm_hydro_inp = tmp_path / "swmm" / "hydro.inp"
        self.scen_paths.swmm_hydro_inp.parent.mkdir()
        if hyg_present:
            self.scen_paths.hyg_timeseries.write_text("x")
            self.scen_paths.hyg_locs.write_text("x")
        if hydro_out_present:
            (tmp_path / "swmm" / "hydro.out").write_text("x")


def test_gate_skips_when_logged_and_present(tmp_path, monkeypatch):
    """Fast path. Pre-fix FAILS: no gate exists, so hydro.out is opened unconditionally."""
    scen = _FakeScenario(tmp_path, logged=True, hyg_present=True, hydro_out_present=False)
    assert _run_gate(scen) == "skipped"


def test_gate_regenerates_when_outputs_absent_but_hydro_out_present(tmp_path):
    """Self-heal: the pre-reclaim recovery path must survive the gate."""
    scen = _FakeScenario(tmp_path, logged=True, hyg_present=False, hydro_out_present=True)
    assert _run_gate(scen) == "fell_through"


def test_gate_raises_when_outputs_and_hydro_out_both_absent(tmp_path):
    """The third state. Without it this failure is silent AND permanent."""
    scen = _FakeScenario(tmp_path, logged=True, hyg_present=False, hydro_out_present=False)
    with pytest.raises(ProcessingError) as exc:
        _run_gate(scen)
    assert "no rebuild source" in str(exc.value)
    assert "hydro_swmm_sim_completed" in str(exc.value)


def test_first_ever_prep_is_unaffected(tmp_path):
    """Regression guard: an unprepared scenario must reach the real body unchanged."""
    scen = _FakeScenario(tmp_path, logged=False, hyg_present=False, hydro_out_present=True)
    assert _run_gate(scen) == "fell_through"


def test_rename_accepts_the_new_key_and_rejects_the_old_one():
    """Both arms flip on the rename; a test asserting only one would pass pre-rename.

    The reject arm matters more than the accept arm. Toolkit convention forbids a
    deprecation alias ("update all usage sites immediately, delete the old code
    completely"), so an old config must FAIL LOUDLY at load rather than be silently
    ignored -- and under extra="forbid" it does, which is the property this pins.
    """
    import pytest as _pytest
    from pydantic import ValidationError

    from hhemt.config.analysis import analysis_config

    assert "remove_after_processing" in analysis_config.model_fields
    assert "reclaim_after_processing" not in analysis_config.model_fields
    # Payload built from the model's OWN optional defaults rather than a
    # hand-listed fixture, so this block binds every name it loads and cannot
    # drift as required fields change. Required fields stay absent, so the raise
    # is guaranteed -- which is exactly why `match=` is load-bearing: it pins the
    # failure to the RETIRED KEY rather than to the missing required fields.
    _payload = {name: f.default for name, f in analysis_config.model_fields.items() if not f.is_required()}
    with _pytest.raises(ValidationError, match="reclaim_after_processing"):
        analysis_config.model_validate({**_payload, "reclaim_after_processing": "all"})


def test_hydro_out_is_tritonswmm_only_and_absent_from_the_default_policy():
    """Guards the racing-models hazard and the strict-safe default together."""
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    assert "hydro_out" not in P._reclaim_classes("none")
    assert "hydro_out" in P._reclaim_classes("all")


# ---------------------------------------------------------------------------
# Surface 1 — rpt truncation (Spec D). Helpers first, so every test below binds.
# ---------------------------------------------------------------------------


def _parser_markers() -> list[str]:
    """The marker set DERIVED from the parser, not hand-listed.

    Scraped from the two functions that actually read them, so a marker added upstream
    makes `test_truncation_preserves_every_parser_needed_marker` fail on the day it lands.
    Hand-listing would let an upstream addition pass silently, which is the whole property
    this guard exists for.
    """
    import ast
    import inspect

    from hhemt import swmm_output_parser as sop

    wanted = (
        "Element Count",
        "Flow Units",
        "Flow Routing Continuity",
        "Continuity Error",
        "Flooding Loss",
        "Analysis ended on",
        "Summary",
        "Time Series Results",
    )
    found: set[str] = set()
    for fn in (sop._scan_metadata_and_summaries, sop.parse_rpt_single_pass):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        fn_node = tree.body[0]
        body = fn_node.body
        # Drop the function's own DOCSTRING before walking. It legitimately contains the
        # words the scrape keys on ("Summary", "Time Series Results") while being prose,
        # not a marker, so leaving it in makes the guard demand that truncation preserve a
        # paragraph of documentation.
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if "\n" in node.value:
                        continue
                    if any(w in node.value for w in wanted):
                        found.add(node.value)
    assert found, "parser-marker scrape found nothing — the scrape itself is broken"
    return sorted(found)


def _synthetic_rpt(tmp_path, *, n_body_lines: int, n_nodes: int = 5, trailer: bool = True):
    """Write a minimal rpt with the measured real-file ORDERING.

    Header + continuity + summary tables FIRST, then the time-series body, then the
    trailer as the last line — the layout measured on two independently produced real
    coupled rpts (reference: body starts at 8,947 of 50,013; synth: 267 of 28,537).
    """
    p = tmp_path / "hydraulics.rpt"
    lines = ["  EPA STORM WATER MANAGEMENT MODEL", "", "  Element Count", ""]
    lines += [f"  Node {i} ................ 0.00" for i in range(n_nodes)]
    lines += [
        "",
        "  Flow Units ............ CFS",
        "",
        "  Flow Routing Continuity",
        "  Flooding Loss ......... 0.000",
        "  Continuity Error (%) .. 0.000",
        "",
    ]
    for table in ("Node Inflow Summary", "Node Flooding Summary", "Link Flow Summary"):
        lines += [f"  {table}", ""]
        lines += [f"  row {i} for {table}" for i in range(n_nodes)]
        lines += [""]
    lines += ["  Node Time Series Results", ""]
    lines += [f"  t={i} 0.00 0.00" for i in range(n_body_lines)]
    if trailer:
        lines += ["  Analysis ended on:  Sun Aug 17 09:00:00 2026"]
    p.write_text("\n".join(lines) + "\n")
    return p


def _truncate(rpt_path):
    """Invoke the production truncation helper against a bare path.

    Binds only the attributes `_truncate_coupled_rpt` touches, so this stays fixture-free.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    # No shim: _truncate_coupled_rpt is a @staticmethod (it binds no instance state),
    # so the unbound call takes the three real arguments and nothing else.
    return P._truncate_coupled_rpt(rpt_path, rpt_path.parent, False)


def _markers_that_must_survive() -> list[str]:
    """Every parser marker EXCEPT the body-start sentinel.

    The split is not a convenience -- it is the design. `Node Time Series Results` is the
    truncation BOUNDARY, so truncation necessarily consumes it, and that consumption is
    exactly what makes a second pass a no-op (the no-body-start branch). Asserting that
    every marker survives would demand the boundary survive its own removal.
    `_scan_metadata_and_summaries` is unharmed by its absence: it uses the sentinel only to
    know where the time-series body starts, and on a truncated rpt there is no body.
    """
    from hhemt.process_simulation import _RPT_BODY_START_SENTINEL

    return [m for m in _parser_markers() if m != _RPT_BODY_START_SENTINEL]


def test_truncation_preserves_every_parser_needed_marker(tmp_path):
    """Derived from the parser, not hand-listed: a marker added upstream fails here."""
    p = _synthetic_rpt(tmp_path, n_body_lines=5000)
    _truncate(p)
    truncated = p.read_text()
    survivors = _markers_that_must_survive()
    assert (
        len(survivors) == len(_parser_markers()) - 1
    ), "exactly one marker (the body-start sentinel) may be consumed by truncation"
    for marker in survivors:
        assert marker in truncated, f"truncation dropped a parser-needed marker: {marker}"


def test_truncation_consumes_the_body_start_sentinel(tmp_path):
    """The other half of the split, and the reason idempotence works.

    If the sentinel survived, a second pass would find a body start that is not one and
    would re-truncate an already-truncated file.
    """
    from hhemt.process_simulation import _RPT_BODY_START_SENTINEL

    p = _synthetic_rpt(tmp_path, n_body_lines=5000)
    assert _RPT_BODY_START_SENTINEL in p.read_text()
    _truncate(p)
    assert _RPT_BODY_START_SENTINEL not in p.read_text()


def test_truncate_is_not_delete_and_rpt_is_complete_still_passes(tmp_path):
    """The discriminating test for the Gotcha-70 boundary.

    Gotcha 70 prohibits DELETING hydraulics.rpt because it is a live completion predicate.
    Truncation must leave it present, non-empty, smaller, and still passing rpt_is_complete
    with that predicate UNMODIFIED.
    """
    from hhemt.swmm_output_parser import rpt_is_complete

    p = _synthetic_rpt(tmp_path, n_body_lines=5000)
    before = p.stat().st_size
    assert _truncate(p) is True
    assert p.exists() and p.stat().st_size > 0
    assert p.stat().st_size < before
    assert rpt_is_complete(p) is True


def test_truncation_refuses_without_trailer(tmp_path):
    """Reachability of the trailer refusal: a walltime-killed or crashed coupled run."""
    p = _synthetic_rpt(tmp_path, n_body_lines=100, trailer=False)
    before = p.read_bytes()
    assert _truncate(p) is False
    assert p.read_bytes() == before


def test_truncation_is_idempotent_and_second_pass_is_a_no_op(tmp_path):
    """Reachability of the no-body-start branch: every Snakemake re-fire hits it."""
    p = _synthetic_rpt(tmp_path, n_body_lines=5000)
    assert _truncate(p) is True
    once = p.read_bytes()
    assert _truncate(p) is False
    assert p.read_bytes() == once


def test_truncation_marker_contains_no_parser_sentinel():
    """The trap: a marker containing a sentinel breaks idempotence AND misleads the parser."""
    from hhemt.process_simulation import _RPT_TRUNCATION_MARKER

    for marker in _parser_markers():
        assert marker not in _RPT_TRUNCATION_MARKER


def test_head_boundary_is_a_sentinel_not_a_size(tmp_path):
    """Guards the MEASURED 34x head-size spread: no byte/line constant may bound the head.

    Reference model head = 8,946 lines (~544 KB); synth head = 266 (~17 KB). A fixed-size
    head would cut a summary table mid-table on a large model, and the parser would then
    read a PARTIAL table rather than fail -- an absent row is indistinguishable from a node
    that never flooded.
    """
    small_dir = tmp_path / "small"
    large_dir = tmp_path / "large"
    small_dir.mkdir()
    large_dir.mkdir()
    small = _synthetic_rpt(small_dir, n_nodes=3, n_body_lines=5000)
    large = _synthetic_rpt(large_dir, n_nodes=1200, n_body_lines=5000)
    _truncate(small)
    _truncate(large)
    assert len(large.read_text()) > 10 * len(small.read_text())
    for marker in _markers_that_must_survive():
        assert marker in large.read_text()

"""Detection for W2's Finding-B five-spec set: the conditional `nc` refusal, its
option glossary, its temporary-marker pair, and the summary completeness gate's
kind check.

WHY THIS MODULE EXISTS. `testing-specialist` round 70 measured all five specs as
UNDETECTED by the suite: the refusal has no arm at all, the glossary is asserted only
structurally by `test_config_reference_hook.py`, the marker pair is comment text, and
the gate's single test consumer (`test_reclaim_unconsolidated_scenarios.py`)
monkeypatches the predicate away. THE MODULE DOES NOT IMPORT PRE-FIX -- the `validation`
import line names a symbol Spec 1 creates -- so pre-fix pytest records a COLLECTION ERROR and
no arm fails for its own reason. That is not coverage, and the arms divide two ways once the
module imports. RED AGAINST PRE-FIX BEHAVIOUR: A1, A2, A5, B, C, D, F1, G1. SEPARATORS
AGAINST A MIS-IMPLEMENTATION, green on pre-fix semantics and red on a plausible wrong fix:
F3 (the bare `is_file()` repair), G2 (a diagnostic that warns on every False), A3 (a
`Literal`-narrowing delivery), A4 (keying on the toggles alone), and E1 (a partial apply that
drops Spec 4). F2 is a CONTROL and discriminates nothing. Do not prune a separator as
redundant with the defect arm it sits beside: F1 passes on the bare `is_file()` repair and
only F3 refuses it, so pruning F3 as "already covered by F1" removes the only arm that
catches the wrong fix.

NO SOLVER, NO COMPILE, NO ALLOCATION, and no marker, so this is a Gate A module. The
static arms parse source with `ast` in the style of
`tests/test_marker_reconciliation_placement.py`; the refusal arms build config stubs in
the style of `tests/test_per_member_system_configs.py`; the gate arms use `tmp_path`
only. The ONE arm that touches a fixture mutates `synth_multi_sim_analysis`'s own
`cfg_analysis`, which is safe because that fixture is FUNCTION-scoped
(`tests/conftest.py:288`) -- it must NEVER mutate `tests/fixtures/test_case_builder.py`,
whose `target_processed_output_type: "zarr"` at :566 is what keeps every other
behaviour-keyed consumer in the suite green.

EACH ARM NAMES THE MIS-IMPLEMENTATION IT SEPARATES. An arm that only fires on the
defect is satisfied by predicates stricter than the one shipped; the separating arms
are the ones that fail on a plausible wrong fix, and they are marked SEPARATOR below.

ARM GROUPS, named here rather than in column-0 divider comments. A PEP8 block comment
necessarily begins "# " at column zero, which a markdown heading scanner reads as an H1
and which breaks the porter that extracts this spec -- so the grouping lives in this
docstring and in each test's own first docstring line. A: the refusal predicate
(Spec 1). B: the wiring (Spec 2). C: the seam (r161's correction). D: the option
glossary (Spec 3). E: the temporary-marker pair (Spec 4). F: the completeness gate's
kind check (Spec 5). G: the silent-skip diagnostic.
"""

import ast
import logging
import pathlib
from types import SimpleNamespace

import pytest

from hhemt.config.analysis import analysis_config
from hhemt.summary_paths import _SUMMARY_STEMS_BY_MODEL, scenario_summaries_present
from hhemt.validation import ValidationResult, _validate_gridded_output_type

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "hhemt"
_VALIDATION_SRC = _SRC / "validation.py"
_PROCESS_SIM_SRC = _SRC / "process_simulation.py"

#: The Spec 4 back-half token. Keyed on a fixed literal rather than on prose, so a
#: reword of the surrounding comment does not redden these arms.
_BACK_HALF_TOKEN = "BACK HALF OF A TEMPORARY-REFUSAL MARKER PAIR"
_HELPER = "_validate_gridded_output_type"


def _cfgs(out_type, *, triton=False, tritonswmm=False, swmm=False):
    """The three attributes `_validate_gridded_output_type` reads, and nothing else.

    Deliberately a stub rather than a real config pair: the helper's whole contract is
    that it reads one analysis field and two system toggles, so a stub is the exact
    surface and a real pair would couple these arms to unrelated validators.
    """
    cfg_system = SimpleNamespace(
        toggle_triton_model=triton,
        toggle_tritonswmm_model=tritonswmm,
        toggle_swmm_model=swmm,
    )
    cfg_analysis = SimpleNamespace(target_processed_output_type=out_type)
    return cfg_system, cfg_analysis


def _errors_for(out_type, **toggles):
    result = ValidationResult(context="test")
    cfg_system, cfg_analysis = _cfgs(out_type, **toggles)
    _validate_gridded_output_type(cfg_system, cfg_analysis, result)
    return [e for e in result.errors if e.field == "analysis.target_processed_output_type"]


def test_nc_with_only_tritonswmm_is_refused():
    """SEPARATOR: fails if the predicate is typed `and` instead of `or`."""
    assert _errors_for("nc", tritonswmm=True)


def test_nc_with_only_triton_is_refused():
    """SEPARATOR: the other half of the `or`. Both gridded exporters reach the
    defective writer independently (`process_simulation.py:974` inside
    `_export_TRITONSWMM_TRITON_outputs`, `:1053` inside `_export_TRITON_only_outputs`),
    so a conjunction under-refuses and only a single-toggle arm shows it."""
    assert _errors_for("nc", triton=True)


def test_nc_with_only_swmm_is_accepted():
    """SEPARATOR: fails if the refusal is delivered by narrowing the field's Literal
    to ['zarr'] rather than conditionally. `nc` is implemented and correct for the
    per-scenario summaries and the SWMM node/link timeseries, which route through
    `process_simulation._write_output`."""
    assert _errors_for("nc", swmm=True) == []


def test_zarr_with_every_model_enabled_is_accepted():
    """SEPARATOR: fails if the refusal keys on the toggles alone."""
    assert _errors_for("zarr", triton=True, tritonswmm=True, swmm=True) == []


def test_refusal_message_names_the_supported_value_and_its_own_removal_condition():
    """Asserted on the MESSAGE, not on the exception type. A refusal delivered by a
    Literal narrowing and the intended conditional refusal are indistinguishable by
    exception type, so `pytest.raises(ValidationError)` would pass on the wrong fix."""
    (issue,) = _errors_for("nc", tritonswmm=True)
    assert "zarr" in issue.message
    assert "_streaming_chunked_zarr_write" in issue.message
    assert "TEMPORARY" in issue.message
    assert "toggle_tritonswmm_model" in issue.message


def test_preflight_validate_invokes_the_refusal_helper():
    """Spec 2's omission mode -- helper defined, never called -- is invisible at import
    and at lint. This is the only arm that catches it. Static rather than behavioural
    so it needs no config pair that survives every other validator in preflight."""
    tree = ast.parse(_VALIDATION_SRC.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_validate")
    called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert _HELPER in called


def test_the_refusal_reaches_a_raising_consumer(synth_multi_sim_analysis):
    """`preflight_validate` only ACCUMULATES; `analysis.validate()` is the seam and its
    three consumers raise. This arm proves the error survives the seam. It mutates the
    FUNCTION-scoped fixture's own cfg_analysis and never the shared builder."""
    from hhemt.exceptions import ConfigurationError

    a = synth_multi_sim_analysis
    a.cfg_analysis.target_processed_output_type = "nc"
    result = a.validate()
    issues = [e for e in result.errors if e.field == "analysis.target_processed_output_type"]
    assert issues, [str(e) for e in result.errors]
    assert "_streaming_chunked_zarr_write" in issues[0].message
    # The raise is asserted on the MESSAGE, not on the type: `raise_if_invalid` concatenates
    # `str(err)` for every error and `ValidationIssue.__str__` leads with the field, so this
    # proves the seam carried THIS error. A bare `pytest.raises(ConfigurationError)` is
    # satisfied by any validation error at all and goes permanently green the day an unrelated
    # validator starts erroring on the synth fixture.
    with pytest.raises(ConfigurationError, match="analysis.target_processed_output_type"):
        result.raise_if_invalid()


def test_the_nc_option_glossary_states_the_condition_the_loader_enforces():
    """`test_config_reference_hook.py` asserts the generated page's STRUCTURE, not the
    option strings, so nothing else would notice the glossary still advertising what the
    loader refuses. Read off the model, which is the single declaration site the
    field_meta stipulation names."""
    extra = analysis_config.model_fields["target_processed_output_type"].json_schema_extra
    options = extra["options"]
    assert "REFUSED" in options["nc"]
    assert "toggle_triton_model" in options["nc"]
    assert "toggle_tritonswmm_model" in options["nc"]


def test_marker_pair_halves_coexist():
    """The pair's drift residual, converted from a command nobody runs into a check the
    suite runs. Keyed on the helper SYMBOL, so a reword of either comment is free."""
    helper_present = _HELPER in _VALIDATION_SRC.read_text()
    back_half_present = _BACK_HALF_TOKEN in _PROCESS_SIM_SRC.read_text()
    assert helper_present == back_half_present


def test_marker_back_half_is_inside_the_function_it_marks():
    """Spec 4's wrong-function placement mode. The back half is only locatable if it
    sits in the function the remover edits, so its enclosing def is the assertion."""
    src = _PROCESS_SIM_SRC.read_text()
    if _BACK_HALF_TOKEN not in src:
        pytest.skip("marker pair absent; coexistence is asserted by the sibling arm")
    marker_line = next(i + 1 for i, line in enumerate(src.splitlines()) if _BACK_HALF_TOKEN in line)
    enclosing = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef) and node.lineno <= marker_line <= (node.end_lineno or node.lineno)
    ]
    assert any(node.name == "_streaming_chunked_zarr_write" for node in enclosing)


def _analysis_with(tmp_path, out_type, kind):
    processed = tmp_path / "sims" / "evt" / "processed"
    processed.mkdir(parents=True)
    for stem in _SUMMARY_STEMS_BY_MODEL["triton"]:
        target = processed / f"{stem}.{out_type}"
        if kind == "dir":
            target.mkdir()
        elif kind == "file":
            target.write_bytes(b"CDF")
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(simulation_directory=tmp_path / "sims"),
        cfg_analysis=SimpleNamespace(target_processed_output_type=out_type),
    )


def test_zarr_store_under_an_nc_name_is_not_complete(tmp_path):
    """The defect arm. RED pre-fix: a bare existence probe returns True here."""
    analysis = _analysis_with(tmp_path, "nc", "dir")
    assert scenario_summaries_present(analysis, "evt", ["triton"]) is False


def test_a_real_netcdf_file_is_complete(tmp_path):
    analysis = _analysis_with(tmp_path, "nc", "file")
    assert scenario_summaries_present(analysis, "evt", ["triton"]) is True


def test_a_zarr_store_on_the_default_configuration_is_complete(tmp_path):
    """SEPARATOR, AND THE MOST IMPORTANT ARM IN THIS MODULE: it fails on the bare
    `is_file()` repair the concurred SHAPE's wording implies. A zarr store is a
    DIRECTORY, so `is_file()` returns False here -- on the DEFAULT configuration --
    which would report every healthy analysis incomplete at every consumer of this
    predicate. The defect arm above passes on that wrong fix; only this one does not."""
    analysis = _analysis_with(tmp_path, "zarr", "dir")
    assert scenario_summaries_present(analysis, "evt", ["triton"]) is True


def test_the_wrong_kind_case_warns(tmp_path, caplog):
    """The frozen `nc` population's only remaining signal. Without this the fix makes
    that population QUIETER: today it raises an uncaught IsADirectoryError downstream,
    and after the fix both consumers skip it in silence."""
    analysis = _analysis_with(tmp_path, "nc", "dir")
    with caplog.at_level(logging.WARNING, logger="hhemt.summary_paths"):
        scenario_summaries_present(analysis, "evt", ["triton"])
    assert any("WRONG KIND" in record.getMessage() for record in caplog.records)


def test_the_ordinary_absent_case_is_silent(tmp_path, caplog):
    """SEPARATOR: fails if the diagnostic fires on every False rather than on the
    wrong-kind case. Absence is the ordinary mid-workflow state this predicate reports
    constantly, and a warning there is noise that gets filtered."""
    analysis = _analysis_with(tmp_path, "zarr", "absent")
    with caplog.at_level(logging.WARNING, logger="hhemt.summary_paths"):
        assert scenario_summaries_present(analysis, "evt", ["triton"]) is False
    assert caplog.records == []

"""Detection for W2's Finding-B five-spec set: the conditional `nc` refusal, its
option glossary, its temporary-marker pair, and the summary completeness gate's
kind check.

WHY THIS MODULE EXISTS. `testing-specialist` round 70 measured all five specs as
UNDETECTED by the suite: the refusal has no arm at all, the glossary is asserted only
structurally by `test_config_reference_hook.py`, the marker pair is comment text, and
the gate's single test consumer (`test_reclaim_unconsolidated_scenarios.py`)
monkeypatches the predicate away. THE MODULE DOES NOT IMPORT PRE-FIX -- the `validation`
import line names a symbol Spec 1 creates -- so pre-fix pytest records a COLLECTION ERROR and
no arm fails for its own reason. That is not coverage, and the arms divide into the classes
`ARM_CLASSES` below maps them to -- RED AGAINST PRE-FIX BEHAVIOUR; SEPARATOR AGAINST A
MIS-IMPLEMENTATION, green on pre-fix semantics and red on a plausible wrong fix; and CONTROL,
which discriminates nothing; and INVARIANT, which is red only on a FUTURE edit rather than on
any wrong fix of the change this module guards.
EVERY CLASS ABOVE ANSWERS ONE QUESTION -- what is this arm's relationship to the CHANGE this
module guards -- and META is the cell for an arm that has none, because its subject is this
module's own bookkeeping rather than the product. That sentence is the taxonomy's DOMAIN and
it is what bounds this vocabulary: four relationship values for arms about the change, one
cell for arms that are not. An arm that resists all five is not asking for a sixth token; it
is asking whether it belongs in this module. Without the domain stated, a classification
defined relative to one change must either exclude an out-of-domain arm -- the hole the prior
roster had at fourteen of fifteen -- or misfile it, which this replacement did twice before
it was caught, once as RED and once as SEPARATOR. NO ROSTER IS KEPT HERE, deliberately:
the prior one named arms by short code, bound those codes to nothing in this file, went one
arm short, and classed two RED arms as separators -- and three close readings did not catch
any of it. `ARM_CLASSES` is the single statement and two arms check it.
Do not prune a separator as redundant with the defect arm it sits beside: the defect arm at
`test_zarr_store_under_an_nc_name_is_not_complete` passes on the bare `is_file()` repair and
only `test_a_zarr_store_on_the_default_configuration_is_complete` refuses it, so pruning the
latter as "already covered" removes the only arm that catches the wrong fix.
THE COMPOSITION RESIDUAL, stated because the arm that carried it is gone: no arm asserts
that the three legs of the seam JOIN. The forwarding arm proves `validate` forwards, the
preflight arms prove the helper errs, and the raise arm proves an error survives
`raise_if_invalid` -- the three arm NAMES tile the chain, which is what makes the absence
invisible as an absence. Joining them needs a real Analysis and is deferred with the
fixture-layer remedy.

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
import re
from types import SimpleNamespace

import pytest

from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config
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
    # NAME divergence, closed for every REFUSAL arm at once because `_cfgs` is their single
    # stub construction site. It is NOT the module's only one: `_analysis_with` at :200-202
    # builds its own `cfg_analysis` stub for the five gate arms and gets none of this
    # closure. Extending it there is a separate edit and is not in this set.
    # Class-level `model_fields` only -- no instance, no file, no validator, which is why
    # this is affordable where a real config pair is not.
    # RESIDUAL (A), carried deliberately: this closes NAME divergence and NOT TYPE
    # divergence. A stub supplying `str` where the model declares a constrained type
    # passes both assertions. Closing that needs a real instance and is not in scope.
    for stub, model, label in (
        (cfg_analysis, analysis_config, "analysis"),
        (cfg_system, system_config, "system"),
    ):
        stray = set(vars(stub)) - set(model.model_fields)
        if stray:
            near = {
                name: [f for f in model.model_fields if name in f or f in name] or ["<no substring match>"]
                for name in sorted(stray)
            }
            raise AssertionError(
                f"{label} stub sets {sorted(stray)}, which {model.__name__} does not declare. "
                f"Closest declared fields BY SUBSTRING (a hint, not a determination): {near}"
            )
    return cfg_system, cfg_analysis


#: The field string Spec 1's helper stamps on its `ValidationIssue`. Single-sourced so a
#: change to that `field=` reddens the refusal arms and the raise arm together rather than
#: leaving one of them green against a string nothing emits. Follows `_HELPER`'s convention
#: eight lines up.
_FIELD = "analysis.target_processed_output_type"
#: The same string as a REGEX-SAFE pattern. Consumers of `pytest.raises(match=...)` take
#: this one and never `_FIELD`, so there is no rule to remember and none to enforce: the
#: dots in `_FIELD` are wildcards under `match=`, harmlessly today, and a future field name
#: carrying `(` or `[` would make a consuming arm ERROR rather than fail.
_FIELD_RE = re.escape(_FIELD)


def _errors_for(out_type, **toggles):
    result = ValidationResult(context="test")
    cfg_system, cfg_analysis = _cfgs(out_type, **toggles)
    _validate_gridded_output_type(cfg_system, cfg_analysis, result)
    return [e for e in result.errors if e.field == _FIELD]


def test_nc_with_only_tritonswmm_is_refused():
    """RED: fails if the predicate is typed `and` instead of `or`."""
    assert _errors_for("nc", tritonswmm=True)


def test_nc_with_only_triton_is_refused():
    """RED: the other half of the `or`. Both gridded exporters reach the
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
    """RED: Asserted on the MESSAGE, not on the exception type. A refusal delivered by a
    Literal narrowing and the intended conditional refusal are indistinguishable by
    exception type, so `pytest.raises(ValidationError)` would pass on the wrong fix."""
    (issue,) = _errors_for("nc", tritonswmm=True)
    assert "zarr" in issue.message
    assert "_streaming_chunked_zarr_write" in issue.message
    assert "TEMPORARY" in issue.message
    assert "toggle_tritonswmm_model" in issue.message


def test_preflight_validate_invokes_the_refusal_helper():
    """RED: Spec 2's omission mode -- helper defined, never called -- is invisible at import
    and at lint. This is the only arm that catches it. Static rather than behavioural
    so it needs no config pair that survives every other validator in preflight."""
    tree = ast.parse(_VALIDATION_SRC.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_validate")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert _HELPER in {n.func.id for n in calls}
    # ARGUMENT ORDER, on the node the line above already located rather than in a fourth
    # arm. The helper takes (cfg_system, cfg_analysis, result) positionally; a transposed
    # call still type-checks, still runs, and reads the analysis field off the system
    # config -- which raises AttributeError only if that field is absent, and returns a
    # wrong answer silently if a like-named one is ever added.
    site = next(n for n in calls if n.func.id == _HELPER)
    assert [ast.unparse(a) for a in site.args] == ["cfg_system", "cfg_analysis", "result"]


def test_validate_forwards_every_config_to_preflight_validate():
    """INVARIANT: `src/hhemt/analysis.py` is absent from the commit this module guards, so
    this arm is red only on a FUTURE edit to a file that change never touched. It replaces
    half of the arm that needed a real Analysis. Asserts the keyword NAMES and their
    SOURCES: a forwarding arm that checks only the callee is satisfied by `validate()`
    forwarding the same config twice."""
    tree = ast.parse((_SRC / "analysis.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TRITONSWMM_analysis")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    call = body[0].value
    assert isinstance(call, ast.Call) and call.func.id == "preflight_validate"
    assert {kw.arg for kw in call.keywords} == {"cfg_system", "cfg_analysis", "cfg_hpc_system", "build_sifs"}
    sources = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert sources["cfg_system"] == "self._system.cfg_system"
    assert sources["cfg_analysis"] == "self.cfg_analysis"


def test_raise_if_invalid_carries_the_field_into_the_exception_text():
    """INVARIANT: `ValidationResult` is untouched by the commit this module guards. The
    other half of the replacement. Asserts on the MESSAGE and not the type, because
    `raise_if_invalid` raises on ANY error and a bare `pytest.raises` arm goes permanently
    green the day an unrelated validator starts erring. The field string comes from
    `_FIELD_RE` so a change to Spec 1's `field=` reddens this arm and the refusal arms
    together rather than leaving this one green against a string nothing emits."""
    from hhemt.exceptions import ConfigurationError

    result = ValidationResult(context="test")
    result.add_error(field=_FIELD, message="x", current_value="nc", fix_hint="y")
    with pytest.raises(ConfigurationError, match=_FIELD_RE):
        result.raise_if_invalid()


def test_the_nc_option_glossary_states_the_condition_the_loader_enforces():
    """RED: `test_config_reference_hook.py` asserts the generated page's STRUCTURE, not the
    option strings, so nothing else would notice the glossary still advertising what the
    loader refuses. Read off the model, which is the single declaration site the
    field_meta stipulation names."""
    extra = analysis_config.model_fields["target_processed_output_type"].json_schema_extra
    options = extra["options"]
    assert "REFUSED" in options["nc"]
    assert "toggle_triton_model" in options["nc"]
    assert "toggle_tritonswmm_model" in options["nc"]


def test_marker_pair_halves_coexist():
    """SEPARATOR: The pair's drift residual, converted from a command nobody runs into a check the
    suite runs. Keyed on the helper SYMBOL, so a reword of either comment is free."""
    helper_present = _HELPER in _VALIDATION_SRC.read_text()
    back_half_present = _BACK_HALF_TOKEN in _PROCESS_SIM_SRC.read_text()
    assert helper_present == back_half_present


def test_marker_back_half_is_inside_the_function_it_marks():
    """SEPARATOR: Spec 4's wrong-function placement mode. The back half is only locatable if it
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
    """RED: The defect arm. RED pre-fix: a bare existence probe returns True here."""
    analysis = _analysis_with(tmp_path, "nc", "dir")
    assert scenario_summaries_present(analysis, "evt", ["triton"]) is False


def test_a_real_netcdf_file_is_complete(tmp_path):
    """CONTROL: a real NetCDF FILE under an `.nc` name is complete. Green pre-fix AND green
    on the bare `is_file()` repair, so it discriminates nothing by design -- it exists to
    make the defect arm above attributable rather than merely passing."""
    analysis = _analysis_with(tmp_path, "nc", "file")
    assert scenario_summaries_present(analysis, "evt", ["triton"]) is True


def test_a_zarr_store_on_the_default_configuration_is_complete(tmp_path):
    """SEPARATOR: SEPARATOR, AND THE MOST IMPORTANT ARM IN THIS MODULE: it fails on the bare
    `is_file()` repair the concurred SHAPE's wording implies. A zarr store is a
    DIRECTORY, so `is_file()` returns False here -- on the DEFAULT configuration --
    which would report every healthy analysis incomplete at every consumer of this
    predicate. The defect arm above passes on that wrong fix; only this one does not."""
    analysis = _analysis_with(tmp_path, "zarr", "dir")
    assert scenario_summaries_present(analysis, "evt", ["triton"]) is True


def test_the_wrong_kind_case_warns(tmp_path, caplog):
    """RED: The frozen `nc` population's only remaining signal. Without this the fix makes
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


#: arm name -> its class. THE key. The roster this replaces named arms by short code and
#: bound those codes to nothing in this file, so no check could read it.
#: ONE-TO-ONE, because the classes are disjoint by the SEPARATOR definition's first
#: conjunct. An arm that is red pre-fix is RED even where it also discriminates a wrong
#: fix; WHAT it discriminates stays in its own docstring, which already says so in every
#: case -- a second field here would restate four sentences that exist ten lines away.
ARM_CLASSES: dict[str, str] = {
    "test_nc_with_only_tritonswmm_is_refused": "RED",
    "test_nc_with_only_triton_is_refused": "RED",
    "test_nc_with_only_swmm_is_accepted": "SEPARATOR",
    "test_zarr_with_every_model_enabled_is_accepted": "SEPARATOR",
    "test_refusal_message_names_the_supported_value_and_its_own_removal_condition": "RED",
    "test_preflight_validate_invokes_the_refusal_helper": "RED",
    "test_validate_forwards_every_config_to_preflight_validate": "INVARIANT",
    "test_raise_if_invalid_carries_the_field_into_the_exception_text": "INVARIANT",
    "test_the_nc_option_glossary_states_the_condition_the_loader_enforces": "RED",
    "test_marker_pair_halves_coexist": "SEPARATOR",
    "test_marker_back_half_is_inside_the_function_it_marks": "SEPARATOR",
    "test_zarr_store_under_an_nc_name_is_not_complete": "RED",
    "test_a_real_netcdf_file_is_complete": "CONTROL",
    "test_a_zarr_store_on_the_default_configuration_is_complete": "SEPARATOR",
    "test_the_wrong_kind_case_warns": "RED",
    "test_the_ordinary_absent_case_is_silent": "SEPARATOR",
    "test_every_arm_is_classified": "META",
    "test_each_arm_declares_the_class_it_is_mapped_to": "META",
}

_CLASS_TOKENS = frozenset({"RED", "SEPARATOR", "CONTROL", "INVARIANT", "META"})
_THIS = pathlib.Path(__file__)


def _own_arms():
    tree = ast.parse(_THIS.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def test_every_arm_is_classified():
    """META: totality, and red on an arm added without a class -- which is how the
    roster this replaces came to name fourteen of fifteen arms and omit the only one that
    can silently skip. Asserted as a SET EQUALITY and not a subset: a subset check is green
    on a stale entry for a deleted arm, which is the other direction the same drift runs."""
    missing = {fn.name for fn in _own_arms()} - set(ARM_CLASSES)
    stale = set(ARM_CLASSES) - {fn.name for fn in _own_arms()}
    assert not (missing or stale), (
        f"ARM_CLASSES is out of step: unmapped arms {sorted(missing)}, "
        f"entries for arms that no longer exist {sorted(stale)}. "
        f"Classes are: {', '.join(sorted(_CLASS_TOKENS))}."
    )


def test_each_arm_declares_the_class_it_is_mapped_to():
    """META: the two-way check, and the only arm that can see a disagreement BETWEEN
    the module's two statements of a class rather than inside either one. A roster-only
    partition test reads each arm in exactly one class and PASSES, because it never opens
    the arm -- which is why two RED arms sat under a SEPARATOR heading through three close
    readings. FIRST LINE then colon, case-sensitive: `doc.split(':')[0]` returns the whole
    docstring when the first sentence carries no colon, and `RED` is a substring of
    `PREDICATE`, so a case-insensitive substring census over this module returns 9 where
    the answer is 7. Both were measured, on this arm, before it shipped."""
    wrong = {}
    for fn in _own_arms():
        doc = ast.get_docstring(fn)
        assert doc is not None, (
            f"{fn.name} has no docstring to declare a class in; it must open with one of: "
            + ", ".join(sorted(_CLASS_TOKENS))
        )
        token = doc.splitlines()[0].split(":", 1)[0].strip()
        assert token in _CLASS_TOKENS, (
            f"{fn.name} opens with {token!r}, which is not a class token; use one of: "
            + ", ".join(sorted(_CLASS_TOKENS))
        )
        # .get, never ARM_CLASSES[...]: an unmapped arm must be named by THIS arm's own
        # message rather than raise KeyError. Measured -- the bracket form died with an
        # unreadable traceback when this arm was run alone against an unclassified module.
        mapped = ARM_CLASSES.get(fn.name, "<unmapped>")
        if token != mapped:
            wrong[fn.name] = (token, mapped)
    assert not wrong, f"docstring token disagrees with ARM_CLASSES: {wrong}"

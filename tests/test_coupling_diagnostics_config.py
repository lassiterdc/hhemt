"""The configuration surface of the coupling-diagnostics captures and of the solver-side coupling interval and lag.

This module guards the behaviour of ``hhemt.config.analysis`` (five new fields), ``hhemt.config.system`` (one new
field), ``hhemt.triton_cfg`` (the cfg lines those fields write) and ``hhemt.validation`` (three new preflight
checks). It is a chunk test: at the run's landing it is promoted into those modules' test files
or deleted, and it carries no stored blob and no absolute count, so the promotion gate finds nothing to refuse.

Everything here is pure text and schema: no compile, no solver run, no fixture that builds an analysis. Names that the
chunk creates are read lazily inside the test bodies so that on a tree without the chunk each test collects and
fails with an AssertionError from this file, not with an import error that hides the others.

The groups cover the capture items T1, T2, T5, T6 (the cfg keys and the toggle), T3 (the interval and the lag), H7
(the report step) and T7 (the extra C flags).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.test_triton_cfg_render import MAPPING, ONLY

NEW_ANALYSIS_FIELDS = (
    "toggle_coupling_diagnostics",
    "swmm_diagnostic_links",
    "swmm_diagnostic_node_path3_per_step",
    "swmm_coupling_interval",
    "swmm_coupling_lag",
    "swmm_report_step_s",
)
NEW_SYSTEM_FIELDS = ("TRITONSWMM_extra_c_flags",)

#: The four capture keys the single toggle writes. The spelling is load-bearing in the solver repository, which reads
#: each through ``argsd(key, argmap, "0")``: a key spelled differently silently unassigns a run from its arm.
CAPTURE_KEY_LINES = ["swmm_diag_steps=1", "diag_domain_balance=1", "diag_wet_cells=1", "diag_dt_limiter=1"]


def _stand_in(**extra) -> SimpleNamespace:
    """An analysis config as the cfg generators' older tests build one: the TRITON settings and nothing else."""
    return SimpleNamespace(
        TRITON_courant_number=0.5,
        TRITON_hextra_m=0.001,
        TRITON_fixed_timestep=False,
        TRITON_domain_decomposition="static",
        TRITON_domain_decomposition_check_factor=None,
        TRITON_progress_print_iterations=10000,
        **extra,
    )


def _real_defaults() -> dict:
    from hhemt.config.analysis import analysis_config

    return {name: analysis_config.model_fields[name].default for name in NEW_ANALYSIS_FIELDS}


def _refused_for_its_value(error: ValidationError, field: str) -> bool:
    """True when ``error`` refuses ``field`` for its VALUE, not as an unknown key.

    ``extra="forbid"`` refuses a field the model does not declare with a ``extra_forbidden`` error that names the field
    too, so a refusal test that matched only the name would be green on a tree where the field does not exist.
    """
    return any(e["loc"][-1:] == (field,) and e["type"] != "extra_forbidden" for e in error.errors())


def _render(coupled: bool, analysis) -> str:
    from hhemt.triton_cfg import render_triton_cfg, triton_cfg_settings

    mapping = {**(MAPPING if coupled else ONLY), **triton_cfg_settings(analysis, None)}
    return render_triton_cfg(mapping, coupled=coupled)


def _analysis(tmp_path, **over):
    from hhemt.config.analysis import analysis_config
    from tests.test_config_analysis_report_field import _minimum_valid_cfg_analysis_dict

    (tmp_path / "weather.nc").touch()
    (tmp_path / "events.csv").touch()
    base = _minimum_valid_cfg_analysis_dict(tmp_path)
    base["report"] = {"interactive": {"static_backend": "matplotlib"}}
    return analysis_config.model_validate({**base, **over})


# ----------------------------------------------------------------------------------------------- the fields


def test_the_fields_exist_and_default_to_the_values_that_leave_a_run_unchanged() -> None:
    """Property: the five analysis fields and the one system field are declared, each defaulting to the value that
    asks for nothing: toggle False, links None, interval 1, lag 0, report step None, extra C flags "".

    Class: NEW CAPABILITY. Plausible wrong implementation: a toggle that defaults on, or an interval that defaults
    to anything but 1, which changes every existing run. Reachability: ``model_fields`` is the schema itself.
    Both-states anchor: the assertions read declared defaults, present in any code that declares the fields.
    Why required: a non-neutral default changes every run that never mentions the field.
    Killed by: deleting a field or changing a default. A second correct implementation declares the same defaults
    through a ``default_factory``; the test reads ``.default`` and would need to follow, which is why the control
    below reads a neighbour field that is declared the plain way.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run and the green run
    after the fields exist.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    expected = {
        "toggle_coupling_diagnostics": False,
        "swmm_diagnostic_links": None,
        "swmm_diagnostic_node_path3_per_step": False,
        "swmm_coupling_interval": 1,
        "swmm_coupling_lag": 0,
        "swmm_report_step_s": None,
    }
    for name, default in expected.items():
        assert name in analysis_config.model_fields, f"{name} is not an analysis field"
        assert analysis_config.model_fields[name].default == default, name
    assert "TRITONSWMM_extra_c_flags" in system_config.model_fields
    assert system_config.model_fields["TRITONSWMM_extra_c_flags"].default == ""
    assert analysis_config.model_fields["toggle_coupling_exchange_flow"].default is False  # control: plain default


def test_every_new_field_is_an_experiment_field_of_the_reproducer_taxonomy() -> None:
    """Property: ``_FIELD_BUCKET`` buckets each of the six new fields as ``experiment``.

    Class: NEW CAPABILITY. Plausible wrong implementation: omitting an entry (the totality test of the taxonomy
    then reds, but only when that file runs) or bucketing one ``hpc``. Reachability: ``_FIELD_BUCKET`` is the table
    the totality tests compare every config field against. Both-states anchor: asserts a table value, not wording.
    Why required: each field changes what was modelled or how the solver was built, so a reproducer keeps it.
    Killed by: omitting an entry or using another bucket. A second correct implementation derives the bucket from
    an annotation; the test reads the resulting table value.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt.config.reprex_taxonomy import _FIELD_BUCKET

    for name in (*NEW_ANALYSIS_FIELDS, *NEW_SYSTEM_FIELDS):
        assert _FIELD_BUCKET.get(name) == "experiment", name
    assert _FIELD_BUCKET.get("toggle_coupling_exchange_flow") == "experiment"  # control: a neighbour's entry reads


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"swmm_coupling_interval": 0}, "swmm_coupling_interval"),
        ({"swmm_coupling_interval": -3}, "swmm_coupling_interval"),
        ({"swmm_coupling_lag": 2}, "swmm_coupling_lag"),
        ({"swmm_report_step_s": 0}, "swmm_report_step_s"),
        ({"swmm_report_step_s": -10}, "swmm_report_step_s"),
        ({"swmm_diagnostic_links": "   "}, "swmm_diagnostic_links"),
        ({"swmm_diagnostic_links": "links/relative.txt"}, "swmm_diagnostic_links"),
    ],
)
def test_a_value_outside_a_fields_domain_is_refused_naming_the_field(tmp_path, over, named) -> None:
    """Property: loading an analysis config with an out-of-domain value raises a ValidationError naming the field.

    Class: NEW CAPABILITY. Input class: an interval below 1 (SWMM would never step), a lag outside {0, 1}, a
    report step at or below 0 (SWMM rejects it), a blank links value, and a relative links path (the solver reads
    the file from its own working directory, so a relative name would resolve somewhere the user did not mean).
    Reachability: every row is a value a YAML can carry. Both-states anchor: asserts the refusal and the field name,
    not the wording of the message. Why required: each value would otherwise reach the solver or SWMM and fail
    hours into a run. Killed by: dropping a bound or the absolute-path rule. Second correct implementation: a
    ``model_validator(mode="after")`` raising the same error; the test reads the exception type and field name only.
    Control: the in-domain row below loads, so the refusals are not a property of the loader refusing everything.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    with pytest.raises(ValidationError, match=named) as raised:
        _analysis(tmp_path, **over)
    assert _refused_for_its_value(raised.value, named), "the field is refused as an unknown key, not for its value"


def test_an_in_domain_value_of_every_new_analysis_field_loads(tmp_path) -> None:
    """Property: the boundary values of each domain load, and ``ALL`` is accepted for the links.

    Class: NEW CAPABILITY. Plausible wrong implementation: bounds that are off by one (interval 1 refused, lag 1
    refused) or a links rule that refuses ``ALL``. Reachability: these are the values the experiment's arms use. Both-
    states anchor: asserts the loaded values, present in any implementation of the domains. Why required: the
    refusals above must not shut out the values the experiment sets. Killed by: tightening a bound by one. Second
    correct implementation: any validator accepting the same set.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    absolute = str(tmp_path / "links.txt")
    loaded = _analysis(
        tmp_path,
        toggle_coupling_diagnostics=True,
        swmm_diagnostic_links="ALL",
        swmm_coupling_interval=4,
        swmm_coupling_lag=1,
        swmm_report_step_s=10,
    )
    assert (loaded.swmm_diagnostic_links, loaded.swmm_coupling_interval, loaded.swmm_coupling_lag) == ("ALL", 4, 1)
    assert loaded.swmm_report_step_s == 10 and loaded.toggle_coupling_diagnostics is True
    assert _analysis(tmp_path, swmm_diagnostic_links=absolute).swmm_diagnostic_links == absolute
    assert _analysis(tmp_path, swmm_coupling_interval=1, swmm_coupling_lag=0).swmm_coupling_interval == 1


@pytest.mark.parametrize(
    ("value", "refused"),
    [("-O3", False), ("-O3 -march=x86-64", False), ("", False), ("-DX='1'", True), ("-O3\n-g", True)],
)
def test_the_extra_c_flags_field_refuses_text_that_breaks_the_quoting_it_is_spliced_into(
    tmp_path, value, refused
) -> None:
    """Property: a single quote or a line break in ``TRITONSWMM_extra_c_flags`` is refused at load; plain flags load.

    Class: NEW CAPABILITY. Input class: the flags string, which the compile appends inside the single-quoted
    ``-DCMAKE_C_FLAGS='-fopenmp ...'`` argument of a generated shell script. A quote closes that argument early and
    the rest of the string runs as shell text. Reachability: both refused rows are strings a YAML can carry.
    Both-states anchor: asserts accept versus refuse per row. Why required: the field is the first user-supplied
    text that reaches a generated compile script. Killed by: removing the check. Second correct implementation:
    escaping the quote instead of refusing it; the test would then need a different assertion, which is a change of
    contract and not of behaviour. Control: the plain rows load.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt.config.system import system_config
    from tests.test_config_validation import _minimal_system_config_dict

    raw = {**_minimal_system_config_dict(tmp_path), "TRITONSWMM_extra_c_flags": value}
    if refused:
        with pytest.raises(ValidationError, match="TRITONSWMM_extra_c_flags") as raised:
            system_config.model_validate(raw)
        assert _refused_for_its_value(raised.value, "TRITONSWMM_extra_c_flags")
    else:
        assert system_config.model_validate(raw).TRITONSWMM_extra_c_flags == value.strip()


# ------------------------------------------------------------------------------------------------- the cfg lines


@pytest.mark.parametrize("coupled", [True, False])
def test_an_analysis_at_the_defaults_renders_the_cfg_an_analysis_without_the_fields_renders(coupled) -> None:
    """Property: the cfg rendered for an analysis carrying every new field at its default equals, byte for byte, the
    cfg rendered for a stand-in that has none of the fields, and carries no diagnostic or coupling line.

    Class: NEW CAPABILITY. Plausible wrong implementation: writing ``swmm_coupling_interval=1`` or
    ``diag_wet_cells=0`` unconditionally, which changes the bytes of every existing run's cfg (the default-render
    digest test of ``test_triton_cfg_render`` reds too, but only by hash). Reachability: the stand-in is the form
    older tests hand the generators. Both-states anchor: a same-tree comparison of two renders, with no stored
    blob. Why required: the solver's defaults are the existing behaviour, and a changed cfg byte is a changed run.
    Killed by: any default-valued line. Second correct implementation: emitting through a post-process append in the
    scenario instead of the renderer; the renderer's output would still carry no line, so the test still passes.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    pre_change = _render(coupled, _stand_in())
    at_defaults = _render(coupled, _stand_in(**_real_defaults()))
    assert at_defaults == pre_change
    for token in ("diag_", "swmm_diag", "swmm_coupling_interval", "swmm_coupling_lag"):
        assert token not in at_defaults, token


def test_the_settings_of_an_analysis_at_the_defaults_are_the_seven_keys_of_an_older_analysis() -> None:
    """Property: ``triton_cfg_settings`` returns the same key set for the defaulted new fields as for none of them.

    Class: NEW CAPABILITY. Plausible wrong implementation: always adding the new keys, which breaks the exact-
    equality pin that older tests hold on this dict and leaks a placeholder into every mapping. Reachability: the
    function is the one place the placeholders are produced. Both-states anchor: compares two key sets of the same
    function. Why required: the default mapping must stay what the template filler expects. Killed by: adding a key
    at the defaults. Second correct implementation: building the dict in a helper; the returned keys are the same.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt.triton_cfg import triton_cfg_settings

    older = triton_cfg_settings(_stand_in(), None)
    assert set(triton_cfg_settings(_stand_in(**_real_defaults()), None)) == set(older)
    assert set(older) == {
        "COURANT",
        "HEXTRA",
        "FIXED_TSTEP",
        "DOMAIN_DECOMP",
        "DECOMP_CHECK_FACTOR",
        "IT_PRINT",
        "GPU_DIRECT",
    }


def test_the_toggle_writes_the_four_capture_keys_once_each_at_column_zero_with_value_one() -> None:
    """Property: with the toggle on, the coupled cfg carries each of the four capture lines exactly once, at column
    0, with the value 1; the cfg is the default cfg plus exactly those four lines.

    Class: NEW CAPABILITY. Plausible wrong implementation: ``=true``, ``=0``, an indented or commented line, a
    duplicate, or a fifth key. The solver reads each through ``atoi(argsd(key, argmap, "0"))``, so only ``1`` arms a
    capture and a stray line unassigns the run from its arm. Reachability: the written text is what the solver reads.
    Both-states anchor: compares the on and off renders of the same stand-in. Why required: a misspelt key never
    fails; it silently captures nothing. Killed by: changing any key's spelling or value. Second correct
    implementation: writing the lines in a different order; the test reads the sorted tail, so it still passes.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    off = _render(True, _stand_in(**{**_real_defaults(), "toggle_coupling_diagnostics": False}))
    on = _render(True, _stand_in(**{**_real_defaults(), "toggle_coupling_diagnostics": True}))
    assert on.startswith(off.rstrip("\n"))
    tail = on[len(off.rstrip("\n")) :].split("\n")
    assert tail[0] == "" and sorted(line for line in tail if line) == sorted(CAPTURE_KEY_LINES)
    for line in CAPTURE_KEY_LINES:
        assert on.splitlines().count(line) == 1, line


@pytest.mark.parametrize(
    ("over", "lines"),
    [
        ({"swmm_diagnostic_links": "ALL"}, ["swmm_diag_links=ALL"]),
        ({"swmm_diagnostic_node_path3_per_step": True}, ["swmm_diag_node_path3_per_step=1"]),
        ({"swmm_diagnostic_links": "/data/links_top20.txt"}, ['swmm_diag_links="/data/links_top20.txt"']),
        ({"swmm_coupling_interval": 2}, ["swmm_coupling_interval=2"]),
        ({"swmm_coupling_interval": 4}, ["swmm_coupling_interval=4"]),
        ({"swmm_coupling_lag": 1}, ["swmm_coupling_lag=1"]),
        ({"swmm_coupling_interval": 4, "swmm_coupling_lag": 1}, ["swmm_coupling_interval=4", "swmm_coupling_lag=1"]),
    ],
)
def test_a_setting_that_departs_from_its_default_writes_exactly_its_own_line_and_only_in_the_coupled_cfg(
    over, lines
) -> None:
    """Property: each departing value writes exactly its line, carrying the value; the TRITON-only cfg never does.

    Class: NEW CAPABILITY. Plausible wrong implementations: a hard-coded value (interval 2 and 4 would then give the
    same line), the links path unquoted or ``ALL`` quoted, the line written into the TRITON-only cfg (which has no
    SWMM coupling for the key to act on). Reachability: the rows are the arms the experiment sets. Both-states
    anchor: the departing render minus the default render is the line set, and the TRITON-only render is unchanged.
    Why required: the interval and lag change what SWMM receives, so a dropped or altered line is a different
    experiment with no error. Killed by: any of the wrong implementations. Second correct implementation: a
    different line order for the two-key row; the test compares sorted lines.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    default = _render(True, _stand_in(**_real_defaults()))
    changed = _render(True, _stand_in(**{**_real_defaults(), **over}))
    assert changed.startswith(default.rstrip("\n"))
    added = [line for line in changed[len(default.rstrip("\n")) :].split("\n") if line]
    assert sorted(added) == sorted(lines)
    assert _render(False, _stand_in(**{**_real_defaults(), **over})) == _render(False, _stand_in())


@pytest.mark.parametrize(
    "over",
    [
        {"swmm_coupling_interval": 1},
        {"swmm_coupling_lag": 0},
        {"toggle_coupling_diagnostics": False},
        {"swmm_diagnostic_node_path3_per_step": False},
    ],
)
def test_a_value_set_explicitly_to_its_default_writes_nothing(over) -> None:
    """Property: naming a default explicitly in the config renders the same cfg as leaving it unset.

    Class: NEW CAPABILITY. Plausible wrong implementation: testing whether the field was SET rather than whether it
    DIFFERS from the default, which writes a line for ``swmm_coupling_interval: 1`` and so changes the bytes of a
    run the experiment's baseline arm must keep identical. Reachability: a YAML naming the default is a normal
    input. Both-states anchor: compares two renders that differ only in whether the value was spelt out. Why
    required: the experiment's baseline arms set the keys to their defaults and expect the unchanged cfg. Killed by:
    keying on presence. Second correct implementation: comparing against ``model_fields`` defaults; same bytes.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    unset = _render(True, _stand_in(**_real_defaults()))
    assert _render(True, _stand_in(**{**_real_defaults(), **over})) == unset


# ---------------------------------------------------------------------------------------------- the preflight


def _result():
    from hhemt import validation

    return validation.ValidationResult(context="test")


def _errors(result, *names) -> list:
    return [e for e in result.errors if any(name in str(e) for name in names)]


@pytest.mark.parametrize(
    ("coupled", "over", "refused"),
    [
        (False, {"toggle_coupling_diagnostics": True}, True),
        (True, {"toggle_coupling_diagnostics": True}, False),
        (False, {"swmm_diagnostic_node_path3_per_step": True}, True),
        (True, {"swmm_diagnostic_node_path3_per_step": True}, False),
        (False, {"swmm_diagnostic_links": "ALL"}, True),
        (True, {"swmm_diagnostic_links": "ALL"}, False),
        (False, {"swmm_coupling_interval": 4}, True),
        (True, {"swmm_coupling_interval": 4}, False),
        (False, {"swmm_coupling_lag": 1}, True),
        (True, {"swmm_coupling_lag": 1}, False),
        (False, {}, False),
        (True, {}, False),
    ],
)
def test_preflight_refuses_a_coupling_setting_only_when_the_coupled_model_is_off(coupled, over, refused) -> None:
    """Property: ``_validate_coupling_diagnostics`` yields an error exactly when a setting that acts on the coupled
    model departs from its default and the coupled model is off.

    Class: NEW CAPABILITY. Plausible wrong implementations: never refusing (a run finishes with no capture and no
    sign why), or refusing whenever the setting is set. Reachability: the ten rows are the truth table of the rule
    over the four settings. Both-states anchor: asserts error presence per row. Why required: the captures and the
    T3 keys act on the SWMM coupling block, which a TRITON-only or SWMM-only system never runs. Killed by: either
    wrong implementation (rows 1 and 2 separate them). Second correct implementation: raising instead of
    accumulating; that is a change of contract.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt import validation

    check = getattr(validation, "_validate_coupling_diagnostics", None)
    assert check is not None, "validation._validate_coupling_diagnostics does not exist"
    result = _result()
    check(SimpleNamespace(toggle_tritonswmm_model=coupled), SimpleNamespace(**{**_real_defaults(), **over}), result)
    assert bool(_errors(result, "toggle_coupling_diagnostics", "swmm_diagnostic_", "swmm_coupling_")) is refused


def test_preflight_refuses_a_links_file_that_does_not_exist_and_admits_all_and_an_existing_file(tmp_path) -> None:
    """Property: a links value other than ``ALL`` must name an existing file; ``ALL`` needs no file.

    Class: NEW CAPABILITY. Plausible wrong implementation: skipping the existence check (the run then captures no
    link and says nothing) or demanding a file for ``ALL``. Reachability: a path the user mistyped, an existing file
    and the literal ``ALL`` are the three ordinary inputs. Both-states anchor: asserts error presence per input with
    the coupled model on in every row. Why required: the solver silently reads nothing from a missing list. Killed
    by: either wrong implementation. Second correct implementation: checking ``is_file`` through ``os.path``.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt import validation

    existing = tmp_path / "links.txt"
    existing.write_text("C1\nC2\n")
    system = SimpleNamespace(toggle_tritonswmm_model=True)
    for value, refused in (("ALL", False), (str(existing), False), (str(tmp_path / "typo.txt"), True)):
        result = _result()
        validation._validate_coupling_diagnostics(
            system, SimpleNamespace(**{**_real_defaults(), "swmm_diagnostic_links": value}), result
        )
        assert bool(_errors(result, "swmm_diagnostic_links")) is refused, value


def _inp(tmp_path, options: str) -> Path:
    path = tmp_path / "model.inp"
    path.write_text(f"[TITLE]\nx\n\n[OPTIONS]\n{options}\n\n[JUNCTIONS]\nJ1 0 1\n")
    return path


@pytest.mark.parametrize(
    ("options", "step", "refused"),
    [
        ("ROUTING_STEP 0:00:30", 10, True),
        ("ROUTING_STEP 0:00:30", 30, False),
        ("ROUTING_STEP 0:00:30", 60, False),
        ("FLOW_UNITS CMS", 10, True),
        ("FLOW_UNITS CMS", 20, False),
        ("ROUTING_STEP 45", 30, True),
        ("ROUTING_STEP 45", 45, False),
        ("ROUTING_STEP 0:01:00", 30, True),
        ("ROUTING_STEP 0:01:00", 60, False),
        ("ROUTING_STEP 0:02:00\nROUTING_STEP 0:00:10", 10, False),
        ("ROUTING_STEP 0:00:10\nROUTING_STEP 0:02:00", 60, True),
    ],
)
def test_preflight_refuses_a_report_step_below_the_models_routing_step(tmp_path, options, step, refused) -> None:
    """Property: ``_validate_swmm_report_step`` errors exactly when ``swmm_report_step_s`` is below the model's
    ROUTING_STEP, which is 20 s when the model declares none, is read as seconds when a bare number and as h:mm:ss
    otherwise (SWMM's own rule), and is the last declaring line when repeated.

    Class: NEW CAPABILITY. Plausible wrong implementations: ignoring the default of 20, reading a bare ``45`` as 45
    hours, or reading the first declaring line. Reachability: each row is a model SWMM accepts. Both-states anchor:
    asserts error presence against the step, with the equal-step rows (30 vs 30, 45 vs 45, 60 vs 60) as the
    boundary, where SWMM accepts a report step equal to the routing step. Why required: SWMM stops with an error at
    validation when the report step is below the routing step, after the allocation is spent. Killed by: each wrong
    implementation. Second correct implementation: parsing the option with SWMM's tokenizer; same verdicts.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt import validation

    check = getattr(validation, "_validate_swmm_report_step", None)
    assert check is not None, "validation._validate_swmm_report_step does not exist"
    system = SimpleNamespace(toggle_tritonswmm_model=True, SWMM_model=_inp(tmp_path, options))
    result = _result()
    check(system, SimpleNamespace(swmm_report_step_s=step), result)
    assert bool(_errors(result, "swmm_report_step_s")) is refused


def test_preflight_judges_the_report_step_only_when_it_is_set_and_refuses_it_without_the_coupled_model(
    tmp_path,
) -> None:
    """Property: an unset report step is never judged, even against a model whose routing step is large; a set one
    without the coupled model is refused.

    Class: NEW CAPABILITY. Plausible wrong implementation: judging the default (today's reporting step) against the
    routing step, which refuses runs that never asked for the field; or letting a set step pass on a system whose
    coupled model is off, where the field would be a silent no-op. Reachability: both inputs are ordinary configs.
    Both-states anchor: asserts error presence per input. Why required: the field changes nothing at None, and a
    silent no-op would let a run finish without the report step the analysis asked for. Killed by: either wrong
    implementation. Second correct implementation: returning early on None; same behaviour.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt import validation

    big = SimpleNamespace(toggle_tritonswmm_model=True, SWMM_model=_inp(tmp_path, "ROUTING_STEP 1:00:00"))
    unset = _result()
    validation._validate_swmm_report_step(big, SimpleNamespace(swmm_report_step_s=None), unset)
    assert not unset.errors
    off = _result()
    validation._validate_swmm_report_step(
        SimpleNamespace(toggle_tritonswmm_model=False, SWMM_model=big.SWMM_model),
        SimpleNamespace(swmm_report_step_s=3600),
        off,
    )
    assert _errors(off, "swmm_report_step_s")


@pytest.mark.parametrize(
    ("flags", "coupled", "environment", "refused"),
    [
        ("-O3", True, "native", False),
        ("-O3", True, "container", True),
        ("-O3", False, "native", True),
        ("", True, "container", False),
        ("", False, "native", False),
    ],
)
def test_preflight_refuses_extra_c_flags_under_a_container_or_without_the_coupled_model(
    flags, coupled, environment, refused
) -> None:
    """Property: ``_validate_extra_c_flags`` errors exactly when the flags are non-empty and either the run is a
    container run (no compile happens, so the flags would be inert) or the coupled model is off (no TRITON-SWMM
    compile happens).

    Class: NEW CAPABILITY. Plausible wrong implementations: never refusing (a container run records flags it never
    applied, and a reader of the bundle believes the build was optimised), or refusing the empty default under a
    container, which breaks every container run. Reachability: the five rows are the truth table over the three
    inputs that matter. Both-states anchor: asserts error presence per row. Why required: a silent no-op flag is a
    false record of how the solver was built. Killed by: each wrong implementation. Second correct implementation:
    one check per condition; the test reads the union.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    from hhemt import validation

    check = getattr(validation, "_validate_extra_c_flags", None)
    assert check is not None, "validation._validate_extra_c_flags does not exist"
    result = _result()
    check(
        SimpleNamespace(toggle_tritonswmm_model=coupled, TRITONSWMM_extra_c_flags=flags),
        SimpleNamespace(execution_environment=environment),
        result,
    )
    assert bool(_errors(result, "TRITONSWMM_extra_c_flags")) is refused


@pytest.mark.parametrize(
    "name", ["_validate_coupling_diagnostics", "_validate_swmm_report_step", "_validate_extra_c_flags"]
)
def test_the_preflight_path_calls_each_new_check_with_both_configs_and_the_result(name) -> None:
    """Property: ``validate_data_consistency`` contains one call ``{name}(cfg_system, cfg_analysis, result)``, and
    ``preflight_validate`` contains one call ``validate_data_consistency(cfg_system, cfg_analysis)``.

    Class: NEW CAPABILITY. Input class: the chain every run's preflight passes through. A check that is correct and
    never called refuses nothing. Plausible wrong implementations: a call missing, the two configs swapped, or the
    chain broken at its first link. This is a structural fitness check on two syntax trees, stated without
    reference to the rule. Reachability: the syntax trees of the two functions are the artifacts. Both-states
    anchor: asserts the call lists. Why required: see the first sentence. Killed by: deleting either call or
    reordering the arguments. Second correct implementation: calling the checks directly from
    ``preflight_validate``; this test would then need to follow, which is the price of a wiring pin.
    Invocation: not isolatable from this tree; discrimination shown by the red pre-change run.
    """
    import ast
    import inspect
    import textwrap

    from hhemt import validation

    def calls(function, callee):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        return [
            [arg.id for arg in node.args if isinstance(arg, ast.Name)]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee
        ]

    assert calls(validation.validate_data_consistency, name) == [["cfg_system", "cfg_analysis", "result"]]
    assert calls(validation.preflight_validate, "validate_data_consistency") == [["cfg_system", "cfg_analysis"]]


def test_the_path3_toggle_is_independent_of_the_capture_toggle_and_writes_only_its_own_line() -> None:
    """Property: ``swmm_diagnostic_node_path3_per_step`` writes its one line with ``toggle_coupling_diagnostics`` off,
    and turning both on writes the four capture lines plus it, five in all, each once.

    Class: NEW CAPABILITY. Plausible wrong implementation: folding the field into the capture toggle (the solver
    reads it independently and it needs no ``swmm_diag_steps``), which writes four lines for one request. Reachability:
    both fields are plain booleans a YAML sets. Both-states anchor: compares renders of the same stand-in with
    one and both fields on. Why required: the experiment's heaviest arm wants the per-step file and the uniform set
    together, other arms want the uniform set alone. Killed by: coupling the two. Second correct implementation: a
    different line order; the test compares sorted lines.
    """
    base = _real_defaults()
    default = _render(True, _stand_in(**base))
    alone = _render(True, _stand_in(**{**base, "swmm_diagnostic_node_path3_per_step": True}))
    assert [ln for ln in alone[len(default.rstrip("\n")) :].split("\n") if ln] == ["swmm_diag_node_path3_per_step=1"]
    both = _render(
        True,
        _stand_in(**{**base, "swmm_diagnostic_node_path3_per_step": True, "toggle_coupling_diagnostics": True}),
    )
    added = [ln for ln in both[len(default.rstrip("\n")) :].split("\n") if ln]
    assert sorted(added) == sorted([*CAPTURE_KEY_LINES, "swmm_diag_node_path3_per_step=1"])

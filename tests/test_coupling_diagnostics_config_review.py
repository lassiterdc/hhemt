"""Reviewer tests for the coupling-diagnostics config fields, cfg lines and preflight.

This module guards two properties of ``hhemt.validation`` and ``hhemt.config.analysis`` that the chunk's own tests
do not reach, because they call the new checks with stand-in configs and never put a configured path in front of
the SLURM node-local guard. It is a chunk test: at the run's landing it is promoted into
``tests/test_node_local_config_guard.py`` and ``tests/test_validation.py`` or deleted, and it carries no stored blob
and no absolute count.

Names the chunk creates are read lazily inside the test bodies, so on a tree without the chunk each test fails with
an AssertionError from this file and not with an import error.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hhemt.exceptions import ConfigurationError


def _configs(shared: Path, **analysis_over):
    """A real ``(system_config, analysis_config)`` pair whose own inputs all sit under ``shared``."""
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config
    from tests.test_config_analysis_report_field import _minimum_valid_cfg_analysis_dict
    from tests.test_config_validation import _minimal_system_config_dict

    shared.mkdir(parents=True, exist_ok=True)
    (shared / "weather.nc").touch()
    (shared / "events.csv").touch()
    base = _minimum_valid_cfg_analysis_dict(shared)
    base["report"] = {"interactive": {"static_backend": "matplotlib"}}
    analysis = analysis_config.model_validate({**base, **analysis_over})
    system = system_config.model_validate(_minimal_system_config_dict(shared / "sys"))
    return system, analysis


@pytest.mark.parametrize(
    ("field", "on_node_local_fs", "refused"),
    [
        ("weather_events_to_simulate", True, True),
        ("swmm_diagnostic_links", True, True),
        ("weather_events_to_simulate", False, False),
        ("swmm_diagnostic_links", False, False),
    ],
)
def test_the_cross_node_guard_refuses_a_slurm_submit_whose_configured_file_is_node_local(
    tmp_path, monkeypatch, field, on_node_local_fs, refused
) -> None:
    """Property: a SLURM submission is refused when a file the analysis config names sits under the system temp dir,
    whichever field names it; ``swmm_diagnostic_links`` is a field that names a file.

    Class: NEW CAPABILITY. The input class is every analysis field whose value, as configured, is the absolute path
    of a file the solver or the toolkit reads on a compute node: the guard's own docstring promises that a new path
    field is covered "the moment it is declared". Plausible wrong implementation: ``swmm_diagnostic_links`` as a plain
    ``str``, which ``_iter_config_paths`` skips because it tests ``isinstance(value, Path)``. Under that
    implementation the row ``("swmm_diagnostic_links", True, True)`` fails with DID NOT RAISE, and the rows around
    it pass, so the failure names the field and nothing else.
    Reachability: a links list a user keeps under ``/tmp`` and submits with ``mode="slurm"`` passes the field
    validator (absolute) and preflight (``is_file``), reaches the guard, and the guard finds no offender; the
    compute node cannot read it and the solver records no link and says nothing, which is the failure the field's
    own preflight check was written to prevent. Artifact: ``src/hhemt/validation.py`` ``_iter_config_paths``.
    Both-states anchor: the verdict turns on whether the guard names the configured file, a property present in the
    current code for ``weather_events_to_simulate`` (the control row, green today) and in any correct
    implementation; it does not turn on wording the chunk introduces.
    Why required: a refused submit costs nothing and an accepted one spends an allocation that returns an
    experiment arm with no link record.
    Killed by: typing the field ``str`` (the chunk's choice), which is the one row that fails today. A second
    correct implementation, materially different: type the field ``Path | None`` with ``ALL`` handled before the
    path conversion; the guard then sees it through the existing ``isinstance`` test. A third: keep ``str`` and
    extend ``_iter_config_paths`` to yield a path-valued string field. Both pass every row.
    Invocation: not isolatable from this tree; discrimination shown instead by the third implementation above,
    applied as a ``monkeypatch`` of ``validation._iter_config_paths`` in a throwaway script, which greens the row.
    """
    from hhemt import validation

    node_local = tmp_path / "node_local"
    node_local.mkdir()
    shared = tmp_path / "shared"
    target = (node_local if on_node_local_fs else shared) / "named_file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("C1\nC2\n")
    system, analysis = _configs(shared, **{field: str(target)})
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(node_local))
    monkeypatch.delenv("HHEMT_ALLOW_NODE_LOCAL_CONFIGS", raising=False)

    if refused:
        with pytest.raises(ConfigurationError) as raised:
            validation.assert_configs_visible_cross_node(system, analysis, {}, mode="slurm")
        assert str(target) in str(raised.value), "the refusal must name the offending file"
    else:
        validation.assert_configs_visible_cross_node(system, analysis, {}, mode="slurm")


@pytest.mark.parametrize(
    ("coupled", "refused"),
    [(False, True), (True, False)],
)
def test_preflight_on_real_configs_refuses_each_coupling_setting_only_when_the_coupled_model_is_off(
    tmp_path, coupled, refused
) -> None:
    """HELD (green before this review; must remain green). Property: ``preflight_validate`` over a REAL system and a
    REAL analysis config names each of ``analysis.toggle_coupling_diagnostics``, ``analysis.swmm_coupling_lag``,
    ``analysis.swmm_report_step_s`` and ``system.TRITONSWMM_extra_c_flags`` as an error exactly when the system's
    coupled model is off, and names none of them when it is on and the report step is at least the model's routing
    step.

    Class: REGRESSION (characterization). The chunk's checks read their fields with ``getattr(config, name,
    default)``, which turns a misspelt or removed attribute into the default and so into silence; the chunk's own
    tests feed those checks ``SimpleNamespace`` objects spelt the way the code spells them, so an attribute name
    that the code and its stand-in share but the real models do not carry (for example ``SWMM_modell``, which the
    report-step tests hand over as a literal) passes both. This test puts the real models in front of the same
    checks, through the public entry. Plausible wrong implementation: a check that reads ``swmm_coupling_lagg``;
    red here (shown below), and also red in the chunk's stand-in tests unless the stand-in repeats the misspelling,
    so this test is the only one that reds on the shared misspelling.
    Reachability: any real config pair with the coupled model off and one of these settings set.
    Both-states anchor: the verdict turns on the error list of ``preflight_validate``, which exists whether the
    checks are called from ``validate_data_consistency`` or directly. Why required: see the class sentence.
    Killed by: the misspelling above, in any of the three ``_validate_*`` bodies (node id and mutation are listed in
    the reviewer's round file). A second correct implementation, materially different: calling the three checks
    directly from ``preflight_validate`` instead of ``validate_data_consistency``; the test enters at the public
    function and still passes.
    Invocation: not isolatable from this tree; discrimination shown by the named mutation.
    """
    from hhemt.config.system import system_config
    from hhemt.validation import preflight_validate
    from tests.test_config_validation import _minimal_system_config_dict

    _, analysis = _configs(
        tmp_path / "a",
        toggle_coupling_diagnostics=True,
        swmm_coupling_lag=1,
        swmm_report_step_s=30,
    )
    system_dict = _minimal_system_config_dict(tmp_path / "s")
    system_dict["toggle_tritonswmm_model"] = coupled
    system_dict["toggle_triton_model"] = not coupled
    system_dict["TRITONSWMM_extra_c_flags"] = "-O3"
    system = system_config.model_validate(system_dict)

    result = preflight_validate(system, analysis)
    errors = " ".join(str(e) for e in result.errors)
    for name in (
        "toggle_coupling_diagnostics",
        "swmm_coupling_lag",
        "swmm_report_step_s",
        "TRITONSWMM_extra_c_flags",
    ):
        if name == "swmm_report_step_s" and coupled:
            # The stand-in model file is empty, so its ROUTING_STEP is SWMM's default of 20 s and 30 s clears it.
            assert name not in errors, errors
            continue
        assert (name in errors) is refused, (name, errors)

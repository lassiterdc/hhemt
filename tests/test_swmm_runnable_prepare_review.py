"""Review tests for the atomic cut-over: four classes the cut-over's own tests do not enter.

Each test names the input class it quantifies over, the production change that reds it, and a second
correct implementation under which it still passes. None runs a solver, and none builds more than a
stub scenario, a config dict or a short text.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "hhemt"

# The removed-token matcher of the grep line that guards the removed keys, as one Python alternation.
_RETIRED = re.compile(
    r"SWMM_hydraulics|SWMM_hydrology|SWMM_full\b|_model_template\.inp|create_swmm_inp_from_template"
    r"|build_templates|triton_swmm_configuration_template|TRITON_SWMM_definition_template"
)
_SUFFIXES = {".py", ".md", ".yaml"}


def _carriers(root: Path) -> set[str]:
    """Repo-relative paths under ``root`` whose text matches the removed-token matcher."""
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in _SUFFIXES and _RETIRED.search(p.read_text(encoding="utf-8", errors="replace"))
    }


def test_no_source_file_but_the_system_config_carries_a_retired_template_token(tmp_path):
    """Property: under ``src/hhemt`` the only file that names a removed template key, builder or file is
    ``config/system.py``, which must name the keys to refuse them.

    Class: every ``.py``, ``.md`` and ``.yaml`` file of ``src/hhemt``, read in full (the grep line that guards the
    removed keys, over ``src``, which is what that line returns for this chunk's tree).

    Why required: the guard line ``! grep -rln ... src docs test_data/... | grep -v -x
    src/hhemt/config/system.py`` succeeds only when no other source file matches, so a migration module whose
    docstring spells the removed keys makes the line fail whatever the code does.

    What kills it: spelling a removed key in a docstring, description or message of any ``src`` file other than
    ``config/system.py`` (the migration module's own docstring does).

    A second correct implementation: the migration module describing the retirement without the literal tokens,
    or ``config/system.py`` exporting the key table that the migration module imports.

    Control (must fire): the matcher flags ``config/system.py`` itself, and a planted file holding a token.
    """
    planted = tmp_path / "planted.py"
    planted.write_text('"""Refuses SWMM_hydraulics."""\n')
    assert _carriers(tmp_path) == {"planted.py"}
    found = _carriers(SRC)
    assert "config/system.py" in found, "control: the matcher no longer finds the one file that must carry the tokens"
    assert sorted(found - {"config/system.py"}) == []


def _analysis_cfg(tmp_path):
    from hhemt.config.analysis import analysis_config
    from tests.test_config_validation import _minimal_analysis_config_dict

    return analysis_config.model_validate(_minimal_analysis_config_dict(tmp_path))


def _flags_gpu_aware_mpi(cfg, hpc) -> bool:
    """True when validating ``cfg`` against ``hpc`` refuses, at validation or at construction, naming the field."""
    from hhemt.exceptions import ConfigurationError
    from hhemt.validation import validate_analysis_config

    try:
        result = validate_analysis_config(cfg, hpc)
    except (ConfigurationError, ValueError) as exc:
        return "gpu_aware_mpi" in str(exc)
    return any("gpu_aware_mpi" in f"{i.field} {i.message}" for i in result.errors)


def test_a_gpu_aware_mpi_machine_is_refused_for_a_run_that_uses_no_gpu_before_any_launch(tmp_path):
    """Property: with ``run_mode: serial`` and an HPC system config that declares ``gpu_aware_mpi: true``, the
    two configs are refused when they are validated, naming ``gpu_aware_mpi``, not when the solver is launched.

    Class: NEW CAPABILITY. Plausible wrong implementation: the refusal raised only inside the launch (the built
    form: ``run_simulation.triton_launch_env``), which a submitted job reaches after its queue wait.
    Reachability: a serial analysis config meets a GPU-aware HPC config at ``validate_analysis_config``, which
    receives both (``validation.py``, the ``cfg_hpc_system`` parameter); artifact: the list of
    load-time validators, "`gpu_aware_mpi` true on a CPU run mode", stated as a load-time refusal.
    Both-states anchor: the assertion reads whether any refusal names the field, present in the pre-fix and
    post-fix worlds, and not the wording.
    Invocation not isolatable from this tree; discrimination shown instead: a validation-time check on
    ``gpu_aware_mpi`` greens it, and the same call with ``gpu_aware_mpi: false`` stays green on the control arm.

    Why required: the shape names the check among the load-time validators, and a launch-time refusal costs a
    queue wait before it tells the user.

    A second correct implementation: raising ``ConfigurationError`` from the analysis constructor instead of
    returning a ``ValidationResult`` error; the test accepts either.
    """
    cfg = _analysis_cfg(tmp_path)
    assert cfg.run_mode == "serial"
    assert not _flags_gpu_aware_mpi(cfg, SimpleNamespace(gpu_aware_mpi=False)), "control: a CPU machine is not refused"
    assert _flags_gpu_aware_mpi(cfg, SimpleNamespace(gpu_aware_mpi=True))


@pytest.mark.parametrize(
    ("options", "refused"),
    [
        ("[OPTIONS]\nFLOW_UNITS LPS\n", True),
        ("[OPTIONS]\nFLOW_UNITS MLD\n", True),
        ("[OPTIONS]\nFLOW_UNITS CFS\n", True),
        ("[OPTIONS]\nFLOW_UNITS GPM\n", True),
        ("[OPTIONS]\nFLOW_UNITS MGD\n", True),
        ("[OPTIONS]\nINFILTRATION HORTON\n", True),
        ("[OPTIONS]\nFLOW_UNITS CFS\nFLOW_UNITS CMS\n", False),
        ("[OPTIONS]\nFLOW_UNITS CMS\n", False),
        ("[JUNCTIONS]\nJ1 1 2 0 0 0\n", False),
    ],
)
def test_the_single_swmm_model_must_declare_cms_by_the_last_flow_units_line(tmp_path, options, refused):
    """Property: ``SWMM_model`` is refused at config load when its last ``FLOW_UNITS`` line is not CMS, or when
    it declares none, and is admitted when that line is CMS; a file with no ``[OPTION`` section is skipped.

    Class: NEW CAPABILITY, over the SWMM flow-unit values (the five non-SI ones, none declared, CMS after CFS,
    CMS alone) and a file with no options section. The load-time check the cut-over replaced was pinned by a
    parametrised test over these arms; the retirement of the three template fields deleted that test and nothing
    states the rule for ``SWMM_model``.
    Reachability: every unit value is one a user model declares; artifact: ``flow_units_from_inp_text`` and
    the SWMM manual's FLOW_UNITS list.
    Both-states anchor: the assertion reads whether loading raises and whether the message names CMS, which
    the refusal carries before and after any rewording.
    Invocation not isolatable from this tree; discrimination shown instead: reading the first FLOW_UNITS line
    instead of the last reds the CFS-then-CMS arm, and admitting LPS (the old hydrology field's unit) reds
    the LPS arm.

    Why required: the toolkit reads the coupled report and the standalone output as m3/s with no conversion,
    so a model in another unit corrupts every flow the toolkit reports without an error.

    A second correct implementation: judging the unit in preparation only, provided load also refuses; the
    test reads only whether config load raises.
    """
    from pydantic import ValidationError

    from hhemt.config.system import system_config
    from tests.test_config_validation import _minimal_system_config_dict

    cfg = _minimal_system_config_dict(tmp_path)
    Path(cfg["SWMM_model"]).write_text(options)
    if refused:
        with pytest.raises(ValidationError, match="CMS"):
            system_config.model_validate(cfg)
    else:
        system_config.model_validate(cfg)


@pytest.mark.parametrize("model", ["coupled", "triton_only"])
def test_every_generated_cfg_carries_exactly_one_active_output_folder_naming_the_toolkits_folder(
    tmp_path, monkeypatch, model
):
    """Property: the cfg the toolkit writes for the coupled model and for the TRITON-only model holds exactly
    one active ``output_folder`` line, naming ``out_tritonswmm`` or ``out_triton``, directly before the first
    ``dem_filename`` line.

    Class: both cfg builders of the rendered cfg. Held (green before this review): the template-fill form of
    this property was pinned by parametrised tests over user templates, which the cut-over deleted with the
    template; the property itself survives, because the renderer writes no ``output_folder`` and only
    ``set_hhemt_output_folder`` adds one.
    Reachability: both builders run for every prepared scenario; artifact: ``scenario.py`` fill sites.
    Both-states anchor: the assertion reads the active lines of the written file.

    Named mutation that reds it: dropping the ``set_hhemt_output_folder`` call from either fill site leaves no
    active line, and TRITON then writes outputs to its default folder, where neither the container bind nor the
    resume step-back looks.

    Why required: those two consumers find outputs only under the toolkit's folder names.

    A second correct implementation: writing the folder line as part of the rendered text, provided the cfg
    still ends with exactly one such line before ``dem_filename``.
    """
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    folder = {"coupled": "out_tritonswmm", "triton_only": "out_triton"}[model]
    scen, cfg = _scenario_for_coupled_cfg(tmp_path, monkeypatch, rim_from_dem=False)
    scen.event_iloc = 3
    if model == "coupled":
        scen._generate_TRITON_SWMM_cfg()
    else:
        cfg = tmp_path / "sim" / "TRITON.cfg"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        scen._system.cfg_system.toggle_triton_model = True
        scen.scen_paths.triton_cfg = cfg
        scen._generate_TRITON_cfg()
    lines = cfg.read_text().splitlines()
    active = [i for i, line in enumerate(lines) if line.lstrip().startswith("output_folder")]
    assert [lines[i] for i in active] == [f'output_folder="{folder}"'], lines
    assert lines[active[0] + 1].lstrip().startswith("dem_filename"), lines

"""The TRITON cfg the toolkit renders (``hhemt.triton_cfg``), and the TRITON settings that are config fields.

Pure text, laptop tier. Two kinds of constant were recorded at base commit ``0a6b2dc0``, for the coupled and the
TRITON-ONLY cfg, each from the fixed mapping below, before any edit to ``scenario.py``: (1) the sha256 of the whole
text ``utils.create_from_template`` wrote from the synthetic ``_TEMPLATE`` (the renderer carries that text, so it
reproduces it byte for byte); (2) the sha256 of that text with comment and blank lines removed, which is the set of
lines TRITON reads. The Norfolk example's published cfg template holds the same keys and values with comments
removed (measured at ``0a6b2dc0``); its file is not in this repository, so its equality is carried by (2).
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from hhemt import triton_cfg
from hhemt.config.analysis import analysis_config, triton_extra_env_refusal
from hhemt.config.system import system_config
from hhemt.triton_cfg import render_triton_cfg, triton_cfg_settings

MAPPING = dict(
    CASE_DESC="sim_1",
    DEM="/d/dem.dem",
    SWMM="/s/hydraulics.inp",
    MH_DIAM=1.2,
    MH_LOSS=0.1,
    NUM_SOURCES=3,
    OUT_FORMAT="BIN",
    HYDROGRAPH="/h/hyg.csv",
    HYDO_SRC_LOC="/h/loc.csv",
    MANNINGS="/m/man.dem",
    CONST_MAN_TOGGLE="#",
    MAN_FILE_TOGGLE="",
    CONST_MAN=0.035,
    NUM_EXT_BC=2,
    EXTBC_DIR="/e",
    EXTBC_FILE="/e/bc.csv",
    SIM_DUR_S=1800,
    TSTEP_S=1.0,
    REPORTING_TSTEP_S=10.0,
    OPEN_BOUNDARIES=1,
    COURANT=0.5,
    HEXTRA=0.001,
)
ONLY = {**MAPPING, "CASE_DESC": "sim_1_triton_only", "SWMM": "#DISABLED_FOR_TRITON_ONLY", "MH_DIAM": 0, "MH_LOSS": 0}

# Recorded at base 0a6b2dc0 (see the module docstring): whole text, and the lines TRITON reads.
BASE_TEXT = {
    True: "2d2105876be132f08090784ba23862e79f19a454115cefee34250d47dcf404ff",
    False: "1c3a48ce61b47bbdb8164011f1a9c135c72121993e5cd20c0a51a5f79e864938",
}
BASE_KEYS = {
    True: "77ab3b9c9109a2722a822af693c4bc45547863df45ff87e2bfd5ac5841a4a808",
    False: "a6fa22e2289e0d39eb4cafbcdba463c4790d26185b37dbbd8b9544b4e5d32502",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_lines(text: str) -> str:
    """The lines TRITON reads: comment and blank lines dropped (the commented-out ``inp_filename`` stays)."""
    keep = [ln for ln in text.splitlines() if (ln.strip() and not ln.lstrip().startswith("#")) or "#inp_filename" in ln]
    return "\n".join(keep)


def _render(coupled: bool, **settings) -> str:
    return render_triton_cfg({**(MAPPING if coupled else ONLY), **settings}, coupled=coupled)


@pytest.mark.parametrize("coupled", [True, False])
def test_the_default_render_reproduces_the_retired_templates_fill(coupled):
    text = _render(coupled)
    assert _sha(text) == BASE_TEXT[coupled]
    assert _sha(_read_lines(text)) == BASE_KEYS[coupled]
    assert "${" not in text
    assert ("\ninp_filename=" in text) == coupled


def test_a_changed_literal_turns_the_pinned_constant_red(monkeypatch):
    """Control: the constants discriminate. One toolkit-owned literal changed, the read-lines hash moves."""
    monkeypatch.setattr(triton_cfg, "_TEMPLATE", triton_cfg._TEMPLATE.replace("print_option=huv", "print_option=h"))
    assert _sha(_read_lines(_render(True))) != BASE_KEYS[True]


@pytest.mark.parametrize("coupled", [True, False])
def test_each_cfg_key_field_arrives_on_its_rendered_line(coupled):
    text = _render(
        coupled,
        COURANT=0.45,
        HEXTRA=0.01,
        FIXED_TSTEP=1,
        DOMAIN_DECOMP="dynamic",
        DECOMP_CHECK_FACTOR=7,
        IT_PRINT=500,
        GPU_DIRECT=1,
    )
    lines = text.splitlines()
    for expected in (
        "courant=0.45",
        "hextra=0.01",
        "time_increment_fixed=1",
        "domain_decomposition=dynamic",
        "factor_interval_domain_decomposition=7",
        "it_print=500",
        "gpu_direct_flag=1",
    ):
        assert lines.count(expected) == 1, expected
    for default in ("courant=0.5", "hextra=0.001", "domain_decomposition=static", "time_increment_fixed=0"):
        assert default not in lines


def test_the_default_cfg_carries_neither_conditional_line():
    text = _render(True, DOMAIN_DECOMP="static", DECOMP_CHECK_FACTOR=None, IT_PRINT=10000)
    assert "it_print" not in text and "factor_interval_domain_decomposition" not in text
    # A factor under static decomposition is never written (TRITON reads it only under dynamic).
    assert "factor_interval" not in _render(True, DECOMP_CHECK_FACTOR=7)


def test_settings_come_from_the_analysis_and_hpc_fields():
    class A:
        TRITON_courant_number = 0.3
        TRITON_hextra_m = 0.002
        TRITON_fixed_timestep = True
        TRITON_domain_decomposition = "dynamic"
        TRITON_domain_decomposition_check_factor = 4
        TRITON_progress_print_iterations = 20

    class H:
        gpu_aware_mpi = True

    assert triton_cfg_settings(A, H) == {
        "COURANT": 0.3,
        "HEXTRA": 0.002,
        "FIXED_TSTEP": 1,
        "DOMAIN_DECOMP": "dynamic",
        "DECOMP_CHECK_FACTOR": 4,
        "IT_PRINT": 20,
        "GPU_DIRECT": 1,
    }
    assert triton_cfg_settings(A, SimpleNamespace(gpu_aware_mpi=False))["GPU_DIRECT"] == 0


def test_a_run_with_no_hpc_system_config_gets_the_settings_of_a_default_one():
    """A local run has no HPC system config; the cfg settings are those of an HPC config at its defaults.

    Input class: ``cfg_hpc_system`` None, against a default config (``gpu_aware_mpi`` False). The control, a config
    with ``gpu_aware_mpi`` True, must differ, so the equality is not one that holds for any value of the argument.
    """
    analysis = SimpleNamespace(
        TRITON_courant_number=0.5,
        TRITON_hextra_m=0.001,
        TRITON_fixed_timestep=False,
        TRITON_domain_decomposition="static",
        TRITON_domain_decomposition_check_factor=None,
        TRITON_progress_print_iterations=10000,
    )
    default = triton_cfg_settings(analysis, SimpleNamespace(gpu_aware_mpi=False))
    assert triton_cfg_settings(analysis, None) == default
    assert default["GPU_DIRECT"] == 0
    assert triton_cfg_settings(analysis, SimpleNamespace(gpu_aware_mpi=True)) != default


def _analysis(tmp_path, **over):
    from tests.test_config_analysis_report_field import _minimum_valid_cfg_analysis_dict

    (tmp_path / "weather.nc").touch()
    (tmp_path / "events.csv").touch()
    base = _minimum_valid_cfg_analysis_dict(tmp_path)
    base["report"] = {"interactive": {"static_backend": "matplotlib"}}
    return analysis_config.model_validate({**base, **over})


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"TRITON_courant_number": 0.0}, "TRITON_courant_number"),
        ({"TRITON_courant_number": 1.5}, "TRITON_courant_number"),
        ({"TRITON_hextra_m": -0.1}, "TRITON_hextra_m"),
        ({"TRITON_progress_print_iterations": 0}, "TRITON_progress_print_iterations"),
        ({"TRITON_domain_decomposition": "dynamic"}, "TRITON_domain_decomposition_check_factor"),
        ({"TRITON_domain_decomposition_check_factor": 3}, "TRITON_domain_decomposition_check_factor"),
        ({"TRITON_extra_env": {"OMP_NUM_THREADS": "4"}}, "n_omp_threads"),
        ({"TRITON_extra_env": {"KOKKOS_PRINT_CONFIGURATION": "1"}}, "TRITON_kokkos_print_configuration"),
        ({"TRITON_extra_env": {"HHEMT_X": "1"}}, "HHEMT_"),
        ({"TRITON_extra_env": {"CC": "gcc"}}, "CC"),
        ({"TRITON_extra_env": {"CXX": "g++"}}, "CXX"),
    ],
)
def test_a_bad_triton_setting_is_refused_naming_its_field(tmp_path, over, named):
    with pytest.raises(ValidationError, match=named):
        _analysis(tmp_path, **over)


def test_the_two_required_fields_have_no_default(tmp_path):
    from tests.test_config_analysis_report_field import _minimum_valid_cfg_analysis_dict

    (tmp_path / "weather.nc").touch()
    (tmp_path / "events.csv").touch()
    base = _minimum_valid_cfg_analysis_dict(tmp_path)
    base["report"] = {"interactive": {"static_backend": "matplotlib"}}
    for key in ("TRITON_courant_number", "TRITON_hextra_m"):
        with pytest.raises(ValidationError, match=key):
            analysis_config.model_validate({k: v for k, v in base.items() if k != key})
    good = _analysis(tmp_path, TRITON_domain_decomposition="dynamic", TRITON_domain_decomposition_check_factor=3)
    assert good.TRITON_domain_decomposition_check_factor == 3


@pytest.mark.parametrize("name", ["OPENBLAS_NUM_THREADS", "MY_VAR"])
def test_an_unowned_extra_env_name_is_admitted(name):
    assert triton_extra_env_refusal(name) is None
    assert triton_extra_env_refusal("APPTAINERENV_X") is not None


@pytest.mark.parametrize(
    "key", ["SWMM_hydraulics", "SWMM_hydrology", "SWMM_full", "triton_swmm_configuration_template"]
)
def test_a_retired_template_key_is_refused_with_the_teaching_message(key):
    text = yaml.safe_dump({key: "x.inp"})
    with pytest.raises(ValidationError, match="not a system config field"):
        system_config.model_validate(yaml.safe_load(text))
    with pytest.raises(ValidationError, match="SWMM_model|renders the TRITON cfg itself"):
        system_config.model_validate({key: "x.inp"})


@pytest.mark.parametrize("missing", ["COURANT", "HEXTRA"])
def test_the_renderer_has_no_default_for_the_two_required_settings(missing):
    from hhemt.exceptions import ProcessingError

    mapping = {k: v for k, v in MAPPING.items() if k != missing}
    with pytest.raises(ProcessingError, match=missing):
        render_triton_cfg(mapping, coupled=True)

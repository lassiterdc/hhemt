"""The coupled SWMM model's ``REPORT_STEP`` follows ``analysis.swmm_report_step_s`` and its series keep their own clock.

Chunk test of the report-step row H7. Two surfaces:

* ``hhemt.swmm_utils._swmm_fill_values`` writes ``REPORT_STEP`` of the COUPLED (hydraulics) role from the field when it
  is set and from ``TRITON_reporting_timestep_s`` otherwise; the hydrology and full roles never take the field, and a
  field left at ``None`` writes a role byte-identical to the one written before the field existed.
* ``hhemt.processing_analysis`` places the coupled SWMM node and link series of a scenario on their OWN ``elapsed_s``
  axis (built from the report step) when it differs from the TRITON step, and refuses them on the TRITON axis, as it
  always has, while the field is unset.

No compile, no solver, no Snakemake: the role arms use the stub scenario of ``tests/test_swmm_inp_file_path_length.py``
and the consolidation arms the committed multi-sim golden tree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.swmm_utils import _swmm_fill_values, write_swmm_role_model
from tests.fixtures.swmm_runnable_inputs import runnable_source_text
from tests.test_swmm_runnable_prepare import _stub
from tests.test_triton_timeseries_consolidation import _LINK, _NODE, _TRI, _group, _ready, _rewrite, _scen


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "model.inp"
    path.write_text(runnable_source_text())
    return path


def _report_step_line(text: str) -> str:
    """The value of the one ``REPORT_STEP`` option line of a written role."""
    lines = [ln.split()[1] for ln in text.splitlines() if ln.startswith("REPORT_STEP")]
    assert len(lines) == 1, lines
    return lines[0]


def _written(tmp_path: Path, model: Path, role: str, *, step) -> str:
    """The role text written for the stub scenario with ``swmm_report_step_s`` = ``step`` (absent when ``...``)."""
    scenario = _stub(tmp_path, model)
    if step is not ...:
        scenario._analysis.cfg_analysis.swmm_report_step_s = step
    destination = {
        "hydraulics": scenario.scen_paths.swmm_hydraulics_inp,
        "full": scenario.scen_paths.swmm_full_inp,
        "hydrology": scenario.scen_paths.swmm_hydro_inp,
    }[role]
    write_swmm_role_model(scenario, role, destination)
    return Path(destination).read_text()


@pytest.mark.parametrize(
    ("seconds", "hhmmss"), [(10, "00:00:10"), (90, "00:01:30"), (3600, "01:00:00")], ids=["10s", "90s", "1h"]
)
def test_the_coupled_role_takes_the_independent_report_step(tmp_path, model, seconds, hhmmss):
    """Property: with the field set, the hydraulics role's ``REPORT_STEP`` is the field and not the 60 s TRITON step.

    Control (varies the property): the same stub with the field left ``None`` writes the TRITON step.
    """
    assert _report_step_line(_written(tmp_path, model, "hydraulics", step=seconds)) == hhmmss
    assert _report_step_line(_written(tmp_path / "ctl", model, "hydraulics", step=None)) == "00:01:00"


@pytest.mark.parametrize("role", ["full", "hydrology"])
def test_no_other_role_takes_the_independent_report_step(tmp_path, model, role):
    """Property: the full and hydrology roles keep the TRITON step whatever the field says.

    Why: the hydrology capture's axis is built from ``hydro.inp``'s step and the SWMM-only model is not the coupled
    one. Control: the hydraulics role of the same scenario DOES move (the field is read, only the role gates it).
    """
    assert _report_step_line(_written(tmp_path, model, role, step=10)) == "00:01:00"
    assert _report_step_line(_written(tmp_path / "ctl", model, "hydraulics", step=10)) == "00:00:10"


@pytest.mark.parametrize("role", ["hydraulics", "full", "hydrology"])
def test_an_unset_field_writes_a_role_byte_identical_to_one_written_before_the_field_existed(tmp_path, model, role):
    """Property: ``None`` and an attribute absent altogether (the shape of a config that predates the field) write the
    same bytes. Control: a set field changes the hydraulics role's bytes, so equality is not vacuous.
    """
    before = _written(tmp_path / "absent", model, role, step=...)
    assert _written(tmp_path / "none", model, role, step=None) == before
    if role == "hydraulics":
        assert _written(tmp_path / "set", model, role, step=10) != before


def test_the_fill_values_differ_only_in_report_step_for_the_coupled_destination(tmp_path, model):
    """Property: of the whole forcing mapping, ``REPORT_STEP`` is the only key the field moves, and only for the
    hydraulics destination.
    """
    scenario = _stub(tmp_path, model)
    paths = scenario.scen_paths
    base = _swmm_fill_values(scenario, paths.swmm_hydraulics_inp)
    scenario._analysis.cfg_analysis.swmm_report_step_s = 5
    moved = _swmm_fill_values(scenario, paths.swmm_hydraulics_inp)
    assert {k for k in base if base[k] != moved[k]} == {"REPORT_STEP"}
    assert moved["REPORT_STEP"] == "00:00:05" and base["REPORT_STEP"] == "00:01:00"
    assert _swmm_fill_values(scenario, paths.swmm_hydro_inp) == base


# ---------------------------------------------------------------------------
# Ingestion: the coupled SWMM series land on their own elapsed_s axis
# ---------------------------------------------------------------------------

_SWMM_STEP = 5  # the golden's SWMM series and TRITON step are both 10 s


def _refine_to_5s(ds: xr.Dataset) -> xr.Dataset:
    """Every report period of the store twice, at half the step: 5 s, 10 s, 15 s, ... instead of 10 s, 20 s, ..."""
    n = ds.sizes["date_time"]
    start = np.datetime64("2000-01-01T00:00:00", "ns")
    doubled = ds.isel(date_time=np.repeat(np.arange(n), 2)).load()
    for variable in doubled.variables.values():
        variable.encoding = {}
    return doubled.assign_coords(date_time=start + np.arange(1, 2 * n + 1) * np.timedelta64(_SWMM_STEP, "s"))


def _refine_swmm_stores(a) -> None:
    for event_iloc in a.df_sims.index:
        scen = _scen(a, int(event_iloc))
        _rewrite(Path(scen.scen_paths.output_tritonswmm_node_time_series), _refine_to_5s)
        _rewrite(Path(scen.scen_paths.output_tritonswmm_link_time_series), _refine_to_5s)


def test_the_coupled_swmm_series_land_on_their_own_axis_while_the_triton_series_keep_theirs(golden_extractor):
    """Property: with ``swmm_report_step_s`` = 5 beside a 10 s TRITON step, the coupled SWMM node and link groups of
    the consolidated tree carry an ``elapsed_s`` axis of 5 s multiples that reaches the event end, and the gridded
    TRITON group keeps its 10 s axis.

    Class: NEW CAPABILITY. Plausible wrong implementations: one shared axis at the TRITON step (the 5 s store is then
    refused, as the control below shows), or one shared axis at the SWMM step (the gridded group then moves). Both-
    states anchor: both axes are read from the same tree. Control: ``_timeseries_axis`` with no argument is still
    the TRITON axis.
    """
    a = _ready(golden_extractor, clear=False, extra=f"swmm_report_step_s: {_SWMM_STEP}\n")
    _refine_swmm_stores(a)
    a.process.consolidate_timeseries()
    node_axis = _group(a, _NODE)["elapsed_s"].values
    link_axis = _group(a, _LINK)["elapsed_s"].values
    triton_axis = _group(a, _TRI)["elapsed_s"].values
    assert node_axis[0] == _SWMM_STEP and np.all(np.diff(node_axis) == _SWMM_STEP), node_axis
    np.testing.assert_array_equal(node_axis, link_axis)
    assert triton_axis[0] == 0 and np.all(np.diff(triton_axis) == 10), triton_axis
    assert node_axis[-1] == triton_axis[-1], "both axes reach the same event end"
    assert len(node_axis) == 2 * (len(triton_axis) - 1), "the SWMM axis is twice as fine"
    node_depth = _group(a, _NODE)["depth"]
    assert bool(node_depth.isel(elapsed_s=slice(0, 24)).notnull().any()), "the series carries values on its axis"
    assert a.process._timeseries_axis()[1] == float(a.cfg_analysis.TRITON_reporting_timestep_s)


def test_an_unset_field_still_refuses_swmm_series_off_the_triton_step(golden_extractor):
    """Property: without the field the SWMM series are placed on the TRITON axis only, so a 5 s store is refused
    naming its scenario. Control: this is the SAME store the arm above consolidates once the field is set, so the
    field, and nothing else, is what admits it.
    """
    a = _ready(golden_extractor, clear=False)
    _refine_swmm_stores(a)
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert "reporting step" in str(err.value)


def test_the_default_leaves_every_consolidated_axis_on_the_triton_step(golden_extractor):
    """Property: with the field unset, the coupled SWMM node, link and gridded groups share one axis (the report-step
    change is a no-op at the default). Control: the arm above, where the same assertion on the node axis fails once the
    field is set, so equality here is not vacuous.
    """
    unset = _ready(golden_extractor, clear=False)
    unset.process.consolidate_timeseries()
    node_unset = _group(unset, _NODE)["elapsed_s"].values
    assert np.all(np.diff(node_unset) == 10) and node_unset[0] == 10
    np.testing.assert_array_equal(_group(unset, _LINK)["elapsed_s"].values, node_unset)
    np.testing.assert_array_equal(_group(unset, _TRI)["elapsed_s"].values[1:], node_unset)

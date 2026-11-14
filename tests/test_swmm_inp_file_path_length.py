"""SWMM input file names against SWMM's 259-character file-name limit.

SWMM copies each file name it is handed into a fixed buffer of 259 characters and
truncates a longer one. A truncated time-series FILE name is opened as a different
path, so SWMM fails with "ERROR 361: could not open external file used for Time
Series", which names no length; a truncated binary output name is written somewhere
other than where the toolkit reads it back. ``create_swmm_inp_from_template`` refuses
both before it writes the input, naming the path, its byte length and the limit.

Every test here drives that function on a stub scenario: a two-placeholder template, a
two-step event dataset, and FILE paths that need not exist. Lengths are bytes, the
count SWMM's ``strlen`` sees, so a path holding a two-byte character is one byte longer
than its character count. No test runs a solver.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.swmm_utils import create_swmm_inp_from_template

LIMIT = 259
TWO_BYTE = "é"


def _path_of(root: Path, n_bytes: int, *, suffix: str = "", two_byte: bool = False) -> str:
    """An absolute path under ``root`` whose UTF-8 encoding is exactly ``n_bytes`` long."""
    base = f"{root}/p/"
    room = n_bytes - len(os.fsencode(base)) - len(os.fsencode(suffix))
    name = (TWO_BYTE + "a" * (room - 2)) if two_byte else "a" * room
    path = base + name + suffix
    assert len(os.fsencode(path)) == n_bytes, (path, len(os.fsencode(path)))
    return path


def _scenario(tmp_path: Path, *, rain=None, storm=None, scen_paths=None, env="native"):
    """The attributes ``create_swmm_inp_from_template`` reads, and nothing else."""
    times = pd.date_range("2020-01-01", periods=2, freq="5min")
    cfg = SimpleNamespace(
        weather_time_series_timestep_dimension_name="date_time",
        rainfall_units="mm/hr",
        toggle_storm_tide_boundary=storm is not None,
        TRITON_reporting_timestep_s=60,
        execution_environment=env,
    )
    rain_files = {"rg1": rain if rain is not None else str(tmp_path / "rain.dat")}
    if scen_paths is None:
        scen_paths = SimpleNamespace(
            swmm_hydro_inp=tmp_path / "hydro.inp",
            swmm_full_inp=tmp_path / "full.inp",
            swmm_full_out_file=tmp_path / "full.out",
            swmm_hydraulics_inp=tmp_path / "hydraulics.inp",
            swmm_hydraulics_rpt=tmp_path / "hydraulics.rpt",
        )

    def seconds_to_hhmmss(seconds):
        seconds = int(seconds)
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

    return SimpleNamespace(
        _analysis=SimpleNamespace(cfg_analysis=cfg, cfg_hpc_system=None),
        ds_event_ts=xr.Dataset(coords={"date_time": times}),
        log=SimpleNamespace(
            swmm_rainfall_dat_files=SimpleNamespace(get=lambda: rain_files),
            storm_tide_for_swmm=SimpleNamespace(get=lambda: storm),
        ),
        scen_paths=scen_paths,
        seconds_to_hhmmss=seconds_to_hhmmss,
        _resolve_event_window=lambda: (times[0], times[-1]),
    )


@pytest.fixture
def template(tmp_path: Path) -> Path:
    t = tmp_path / "template.inp"
    t.write_text("[TIMESERIES]\n${TIMESERIES}\n\n[RAINGAGES]\n${RAINGAGES}\n")
    return t


def _assert_refused(scenario, template: Path, destination: Path, path: str) -> None:
    """The call raises ProcessingError naming the path, its byte length and 259, and writes nothing."""
    with pytest.raises(ProcessingError) as excinfo:
        create_swmm_inp_from_template(scenario, template, destination)
    message = str(excinfo.value)
    assert path in message, message
    assert str(len(os.fsencode(path))) in message, message
    assert str(LIMIT) in message, message
    assert not destination.exists(), f"{destination} was written although the call refused"


# ---------------------------------------------------------------------------
# FILE paths written into the input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("which", "two_byte"),
    [("rain", False), ("storm_tide", False), ("rain", True)],
    ids=["rain_260_bytes", "storm_tide_260_bytes_beside_short_rain", "rain_259_chars_260_bytes"],
)
def test_an_over_long_file_path_is_refused_before_the_input_is_written(tmp_path, template, which, two_byte):
    """Property: a FILE path over 259 bytes raises ProcessingError and leaves no input.

    Class: every FILE line the function writes -- a rain-gauge line and the storm-tide
    line -- with the over-long path measured in bytes, so a path of 259 characters
    holding one two-byte character is refused.

    Why required: SWMM truncates the name and fails with an error naming no length.

    What kills it: writing the input without measuring, or measuring characters
    rather than bytes (the third case), or measuring only the rain-gauge lines (the
    second case).

    A second correct implementation under which it still passes: measuring each path
    as it is formatted into its FILE line rather than in a separate pass afterwards.
    """
    long_path = _path_of(tmp_path, LIMIT + 1, suffix=".dat", two_byte=two_byte)
    if which == "rain":
        scenario = _scenario(tmp_path, rain=long_path)
    else:
        scenario = _scenario(tmp_path, storm=long_path)
    destination = tmp_path / "out" / "model.inp"
    _assert_refused(scenario, template, destination, long_path)


@pytest.mark.parametrize("two_byte", [False, True], ids=["ascii_259_bytes", "258_chars_259_bytes"])
def test_control_a_file_path_of_exactly_259_bytes_is_written(tmp_path, template, two_byte):
    """Property: a FILE path of exactly 259 bytes is written into the input unchanged.

    Class: the boundary position the refusal tests do not occupy, in both encodings.

    Why required: without it an over-strict limit, or a check that counts characters
    and so passes the two-byte path while refusing a legal one, passes every refusal.

    What kills it: a limit of 258 or below, or refusing on the byte count plus one.

    A second correct implementation under which it still passes: a limit test written
    as ``>= 260`` instead of ``> 259``.
    """
    path = _path_of(tmp_path, LIMIT, suffix=".dat", two_byte=two_byte)
    destination = tmp_path / "out" / "model.inp"
    create_swmm_inp_from_template(_scenario(tmp_path, rain=path), template, destination)
    assert f'rg1 FILE "{path}"' in destination.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The binary output name SWMM is handed for the input
# ---------------------------------------------------------------------------


def _paths(tmp_path: Path, **overrides) -> SimpleNamespace:
    paths = dict(
        swmm_hydro_inp=tmp_path / "hydro.inp",
        swmm_full_inp=tmp_path / "full.inp",
        swmm_full_out_file=tmp_path / "full.out",
        swmm_hydraulics_inp=tmp_path / "hydraulics.inp",
        swmm_hydraulics_rpt=tmp_path / "hydraulics.rpt",
    )
    paths.update(overrides)
    return SimpleNamespace(**paths)


def _out_case(tmp_path: Path, case: str, n_bytes: int, *, two_byte: bool = False):
    """(scen_paths, destination, measured .out name) for one of the three output cases."""
    if case == "hydro":
        destination = Path(_path_of(tmp_path, n_bytes, suffix=".inp", two_byte=two_byte))
        return _paths(tmp_path, swmm_hydro_inp=destination), destination, str(destination)[:-4] + ".out"
    if case == "full":
        out = _path_of(tmp_path, n_bytes, suffix=".out", two_byte=two_byte)
        return _paths(tmp_path, swmm_full_out_file=Path(out)), tmp_path / "full.inp", out
    if case == "coupled":
        out = _path_of(tmp_path, n_bytes, suffix=".out", two_byte=two_byte)
        rpt = Path(out[:-4] + ".rpt")
        return _paths(tmp_path, swmm_hydraulics_rpt=rpt), tmp_path / "hydraulics.inp", out
    raise ValueError(case)


@pytest.mark.parametrize(
    ("case", "two_byte"),
    [("hydro", False), ("full", False), ("coupled", False), ("full", True)],
    ids=[
        "hydro_out_260",
        "full_out_260_beside_short_input",
        "coupled_out_260_beside_short_input",
        "full_out_259_chars",
    ],
)
def test_an_over_long_output_name_is_refused_before_the_input_is_written(tmp_path, template, case, two_byte):
    """Property: the .out name SWMM will be handed, if over 259 bytes, raises ProcessingError.

    Class: the three inputs whose output name the toolkit holds -- the runoff input
    (output derived from the input name), the full input (its ``swmm_full_out_file``)
    and the native coupled input (its report name with ``.out``).

    Why required: SWMM opens the binary output by its truncated copy of the name, so an
    over-long name is written under a shortened name with no error naming the length.

    What kills it: no output-name check; deriving every output name from the input
    name, which misses the full and coupled cases because their inputs are short here.

    A second correct implementation under which it still passes: selecting the output
    name through a lookup table keyed on the ``scen_paths`` attribute names.
    """
    scen_paths, destination, out = _out_case(tmp_path, case, LIMIT + 1, two_byte=two_byte)
    scenario = _scenario(tmp_path, scen_paths=scen_paths)
    _assert_refused(scenario, template, destination, out)


@pytest.mark.parametrize("case", ["hydro", "full", "coupled"])
def test_control_an_output_name_of_exactly_259_bytes_is_accepted(tmp_path, template, case):
    """Property: an output name of exactly 259 bytes does not stop the input being written.

    Why required: without it an over-strict output limit passes the refusal tests.

    What kills it: a limit below 259 on the output name.

    A second correct implementation under which it still passes: comparing the name's
    byte length with ``>= 260``.
    """
    scen_paths, destination, _ = _out_case(tmp_path, case, LIMIT)
    create_swmm_inp_from_template(_scenario(tmp_path, scen_paths=scen_paths), template, destination)
    assert destination.exists()


def test_control_a_coupled_output_name_in_a_container_run_is_not_measured(tmp_path, template, monkeypatch):
    """Property: in a container run the coupled output name is not measured.

    Why required: TRITON then hands SWMM a name under the image's install root, so a
    host path over 259 bytes is only a bind source and must not refuse the input.

    What kills it: a check that measures the coupled output name whatever the run mode.

    A second correct implementation under which it still passes: measuring the in-image
    name instead, which is short.
    """
    monkeypatch.setattr("hhemt.config.hpc_system.resolve_container_spec", lambda cfg: SimpleNamespace())
    scen_paths, destination, _ = _out_case(tmp_path, "coupled", LIMIT + 1)
    scenario = _scenario(tmp_path, scen_paths=scen_paths, env="container")
    create_swmm_inp_from_template(scenario, template, destination)
    assert destination.exists()


def test_control_a_coupled_output_name_is_not_measured_when_the_coupled_model_is_off(tmp_path, template):
    """Property: with no coupled report name (coupled model off) nothing is measured.

    Why required: the coupled output name exists only when the coupled model runs.

    What kills it: a check that measures the hydraulics input's output whatever the toggle.

    A second correct implementation under which it still passes: testing the coupled
    toggle on the system config instead of the absent report name.
    """
    scen_paths, destination, _ = _out_case(tmp_path, "coupled", LIMIT + 1)
    scen_paths.swmm_hydraulics_rpt = None
    create_swmm_inp_from_template(_scenario(tmp_path, scen_paths=scen_paths), template, destination)
    assert destination.exists()


def test_control_an_over_long_input_name_is_not_measured(tmp_path, template):
    """Property: the input name itself is not measured; only its output name is.

    Why required: SWMM opens the input by the caller's full string, so a long input name
    with a short output name is valid.

    What kills it: a check that measures the input name.

    A second correct implementation under which it still passes: measuring the input
    name but only warning.
    """
    destination = Path(_path_of(tmp_path, LIMIT + 1, suffix=".inp"))
    scen_paths = _paths(tmp_path, swmm_full_inp=destination, swmm_full_out_file=tmp_path / "full.out")
    create_swmm_inp_from_template(_scenario(tmp_path, scen_paths=scen_paths), template, destination)
    assert destination.exists()

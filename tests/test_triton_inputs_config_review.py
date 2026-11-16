"""Review of the optional TRITON inputs, probed with inputs the coder's tests do not position.

Laptop tier, no solver and no compile. Each test is held to 0.15 s over setup, call and teardown.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hhemt.system  # noqa: F401  (imported at collection so the lazy import inside the entry costs no test time)
from hhemt.exceptions import ConfigurationError
from hhemt.triton_cfg import OBSERVATION_FILE_NAME, observation_point_problems, triton_input_settings

# A two-cell DEM in the writer's layout, so a point can sit on it.
_DEM = (
    "ncols         2\nnrows         1\nxllcorner     1000.0\nyllcorner     2000.0\ncellsize      10.0\n"
    "NODATA_value  -9999\n1.00000 2.00000\n"
)


@pytest.mark.parametrize("entry", ["preflight_file_check", "scenario_preparation"])
def test_bytes_that_are_not_utf8_never_escape_either_reader_of_the_observation_file_as_a_decode_error(tmp_path, entry):
    """Property: no ``UnicodeDecodeError`` escapes either reader of the observation file.

    Class: an ``id,x,y`` CSV saved by a spreadsheet as Windows-1252, the default of Excel's "CSV (comma
    delimited)", whose ``id`` holds one accented letter. Why required: today both readers call
    ``read_text(encoding="utf-8-sig")``, so the decode error escapes ``preflight_validate`` (which exists to
    accumulate every problem and returns them together) and the user sees a traceback naming no file and no
    point. Reachability: a site name with an accent is an ordinary point id, and the decode sits in
    ``observation_point_problems`` and in ``triton_input_settings``. Class of the finding: REGRESSION on a
    named input, red on the chunk's tip with the decode error pasted in the reviewer's file. Both-states
    anchor: the assertion turns on the exception type that escapes, never on the wording of any refusal, and
    it holds for every implementation that returns, refuses with a problem, or raises ``ConfigurationError``.
    What kills it: the current decode with no handling. Second correct implementation: sniffing the encoding
    and accepting the file, which returns normally and passes this test as well.
    """
    csv_path = tmp_path / "points.csv"
    csv_path.write_bytes("id,x,y\nS\xe9vign\xe9,1005,2005\n".encode("cp1252"))
    try:
        if entry == "preflight_file_check":
            observation_point_problems(SimpleNamespace(observation_points=csv_path))
        else:
            dem = tmp_path / "dem.dem"
            dem.write_text(_DEM)
            cfg_system = SimpleNamespace(
                observation_points=csv_path, dem_outside_watershed_height=None, dem_building_height=None
            )
            cfg_analysis = SimpleNamespace(
                TRITON_observation_interval_s=None,
                TRITON_initial_depth_raster=None,
                TRITON_initial_qx_raster=None,
                TRITON_initial_qy_raster=None,
            )
            triton_input_settings(
                cfg_system, cfg_analysis, dem_path=dem, observation_file=tmp_path / "sim" / OBSERVATION_FILE_NAME
            )
    except ConfigurationError:
        pass  # a refusal is a correct outcome; only a decode traceback is the defect
    except UnicodeDecodeError as error:
        pytest.fail(f"{entry}: a non-UTF-8 observation file escaped as a decode traceback: {error}")

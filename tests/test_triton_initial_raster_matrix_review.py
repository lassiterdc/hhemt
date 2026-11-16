"""Review of the initial-condition rasters reach TRITON in the form it reads.

The inputs are ones the coder's tests do not position. Laptop tier, no solver and no compile. Each test is held to
0.15 s over setup, call and teardown. The tests reuse the coder's helpers (``_inputs``, ``_raster``) so a fixture
change lands in one place.

One test is RED on the reviewed tip by design (the decode test) and three are held green: a characterization each,
with the change that reds it named in its own docstring.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt.system  # noqa: F401  (imported at collection so the lazy import inside the entry costs no test time)
from hhemt.exceptions import ConfigurationError
from hhemt.run_simulation import TRITONSWMM_run
from tests.test_triton_inputs_config import _inputs, _raster

_TIFF_LIKE = b"II*\x00\x08\x00\x00\x00\xff\xfe\x00\x01 a GeoTIFF handed over where an ASCII grid was expected"


def _cp1252_header(tmp_path):
    """A valid grid whose header line carries one cp1252 byte, as a spreadsheet or a Windows tool leaves one."""
    path = _raster(tmp_path, "grid.asc")
    path.write_bytes(path.read_bytes().replace(b"NODATA_value -9999", b"NODATA_value -9999 \xe9"))
    return path


def _tiff_like(tmp_path):
    path = tmp_path / "grid.asc"
    path.write_bytes(_TIFF_LIKE)
    return path


@pytest.mark.parametrize("make", [_tiff_like, _cp1252_header], ids=["binary_raster", "cp1252_byte"])
@pytest.mark.parametrize("field", ["depth", "qx", "qy"])
def test_bytes_that_are_not_utf8_never_escape_a_raster_read_as_a_decode_error(tmp_path, field, make):
    """Property: no ``UnicodeDecodeError`` escapes ``triton_input_settings`` for any of the three raster fields.

    Class: REGRESSION on a named input, red on the chunk's tip with the decode error pasted in the reviewer's
    file. The input class is a file that is not UTF-8 text at the path of an initial-condition raster: a GeoTIFF
    or any other binary raster handed over where the field says "ESRI ASCII grid", and an ASCII grid with one
    cp1252 byte, over each of the three raster fields. Reachability: the field description names an ESRI ASCII
    grid and a GeoTIFF is the common GIS raster, so a wrong file reaches the field; ``parse_ascii_grid`` then
    calls ``Path.read_text()`` with no encoding handling (``src/hhemt/triton_cfg.py`` ``parse_ascii_grid``, its
    first statement), which raises before any problem is collected, so ``triton_input_settings`` (whose job is to
    return every problem together) raises a traceback that names neither the field nor the file. The observation
    file reader beside it already handles this. Why required: a user told "the raster is
    refused" can act on it, and a user handed ``'utf-8' codec can't decode byte 0xff`` cannot.

    Both-states anchor: the assertion turns on the exception type that escapes and never on a refusal's wording;
    it holds for an implementation that returns, refuses with ``ConfigurationError``, or reads the bytes
    tolerantly. What kills it: the current ``read_text()`` with no decode handling. Second correct
    implementation: reading with ``errors="replace"`` so the cp1252 grid parses or is refused as a non-number,
    which passes this test as well. Invocation: not isolatable from this tree - discrimination shown instead:
    catching ``UnicodeDecodeError`` around the read and returning it as a problem greens this test (run
    forward in the reviewer's file).
    """
    path = make(tmp_path)
    kind = {"kind": "depth"} if field == "depth" else {}
    try:
        _inputs(tmp_path, **{field: path}, **kind)
    except ConfigurationError:
        pass  # a refusal is a correct outcome; only a decode traceback is the defect


def test_a_cell_where_the_processed_dem_has_no_data_never_becomes_a_flood_depth_of_the_nodata_magnitude(tmp_path):
    """Held test (green on the tip). Property: under the elevation kind no cell's depth exceeds the raster's own
    highest level minus the DEM's lowest valid ground.

    Class: NEW CAPABILITY, a characterization. The coder's judgement call (processed DEM no-data cell set dry) is
    "not pinned by a test". The input class is a water-surface raster over a processed DEM that holds its own
    no-data value (the helper DEM has -9999 at row 1, col 3): ``level - dem`` there is level plus 9999, a depth of
    about ten kilometers if the cell is not excluded. Reachability: the processed DEM carries ``NODATA_value
    -9999`` in its header (``tests/test_triton_inputs_config.py`` ``_dem``, which mirrors the writer's layout)
    and ``_depth_values`` subtracts the DEM array cell by cell. Both-states anchor: the bound is a property of
    any correct result (dry, or the raster refused), never of the wording of a message.

    What kills it: dropping the ``dem_missing`` term from the ``dry`` mask in ``triton_cfg._depth_values``, which
    reads that cell as 10004 m. Second correct implementation, materially different: refusing the raster with a
    ``ConfigurationError`` that names the DEM cell (the coder names this alternative); the test passes under it.
    Invocation: not isolatable from this tree - discrimination shown instead: the named change to what the code
    does greens it.
    """
    level = 5.0
    path = _raster(tmp_path, "h.asc", values=[[level] * 4] * 3)
    try:
        settings = _inputs(tmp_path, depth=path, kind="water_surface_elevation", units="m", offset=0.0)
    except ConfigurationError:
        return
    rows = [line.split(" ") for line in Path(settings["H_INFILE"]).read_text().split("\n") if line]
    lowest_valid_ground = 1.0  # the DEM's smallest cell other than the -9999 hole
    bound = level - lowest_valid_ground
    deepest = max(float(token) for row in rows for token in row)
    assert deepest <= bound + 1e-9, f"a cell is {deepest} m deep; no valid ground is lower than {lowest_valid_ground}"


@pytest.mark.parametrize("magnitude", [1e-12, 3.7e-6, -2.4e-9, 123456789.123456, -1e15], ids=str)
def test_every_written_token_is_fixed_point_and_the_grid_is_the_shape_triton_counts(tmp_path, magnitude):
    """Held test (green on the tip). Property: the file a raster field names reads under TRITON's own rules.

    Class: NEW CAPABILITY, a characterization with a differently-positioned input from the coder's (which pins
    one ``1e-05`` cell on a 3 by 4 grid). The input class is a discharge raster whose one cell takes each
    magnitude the formatter can meet: below the five-decimal resolution, tiny negative, large and negative huge.
    The reader is modelled from ``src/matrix.h`` at faba2232 (``get_dims_2d``: columns are the words of the first
    line split on one space, rows are the lines; ``load_from_ascii_file``: a word with a ``.`` is read with
    ``atof`` and any other word with ``atoi``): so the file must hold ``ncols`` single-space words on every line,
    ``nrows`` lines, and no word without a ``.`` or with an exponent. Reachability: ``write_triton_ascii_grid``
    formats with ``{:.5f}``, so an exponent form is unreachable today, and the held test is what stops a later
    change of that format string (to ``{:g}`` or ``repr``) from reaching TRITON as ``atoi("1e-12")``.

    What kills it: a format that prints an exponent or an integer for a whole number. Second correct
    implementation: a different fixed width or more decimal places; the test passes under it. Both-states
    anchor: it turns on the presence of a point and the absence of an exponent, not on a digit count.
    Invocation: not isolatable from this tree - discrimination shown instead: ``{:g}`` reds it for every row of
    the parameter list, and restoring ``{:.5f}`` greens it.
    """
    values = [[0, 0, 0, 0], [0, magnitude, 0, 0], [0, 0, 0, 0]]
    path = _raster(tmp_path, "qx.asc", values=values)
    settings = _inputs(tmp_path, qx=path)
    lines = Path(settings["QX_INFILE"]).read_text().split("\n")
    assert lines[-1] == "", "the file must end with one newline and carry no trailing blank line"
    rows = [line.split(" ") for line in lines[:-1]]
    assert len(rows) == 3
    for row in rows:
        assert len(row) == 4, f"TRITON counts {len(row)} columns on this line; the DEM has 4"
        for word in row:
            assert word != "", "an empty word means a doubled or a leading space"
            assert "." in word and "e" not in word.lower(), f"'{word}' would be read by atoi or as an exponent"


@pytest.mark.parametrize("model_log_says", ["no_model_log", "recorded_complete"])
@pytest.mark.parametrize("model_type", ["triton", "tritonswmm"])
def test_a_run_whose_log_has_no_simulation_ends_marker_is_not_complete_whatever_else_the_tree_holds(
    tmp_path, model_type, model_log_says
):
    """Held test (green on the tip). Property: ``model_run_completed`` is False on a tree whose run log carries no
    ``Simulation ends`` marker, even with a ``cfg/config_*.cfg`` dump and a recorded-complete model log present.

    Class: NEW CAPABILITY, the instrument the chunk's single-simulation assertion now stands on. The shape says
    the run is proven by the marker and not by a cfg dump (TRITON writes its dump from ``write_output``, and a
    dump found under ``cfg/`` can come from an earlier run of the same tree), and that this chunk's reviewer test
    pins the instrument on a tree whose log carries no marker (``run_simulation.py`` records that stale logs once
    carried the marker). Input class: both TRITON model types, with the model log absent and with it recorded
    complete (the latch the method's own comment names). Reachability: a run that exits at a raster read prints
    no marker and still leaves the dump of an earlier run beside it. Both-states anchor: it turns on the verdict,
    not on the log's wording.

    What kills it: ``model_run_completed`` reading the cfg dump, or a positive model-log record, as sufficient.
    Second correct implementation: any other completion proof that still needs the marker; the test passes
    under it. Invocation: not isolatable from this tree - discrimination shown instead: accepting a cfg dump as
    completion greens-to-red this test, and the marker plus ``performance.txt`` control below must read True so
    a broken stub cannot pass for a true False.
    """

    def verdict(log_text, with_performance):
        out = tmp_path / f"out_{len(list(tmp_path.glob('out_*')))}"
        (out / "cfg").mkdir(parents=True)
        (out / "cfg" / "config_0.cfg").write_text("courant=0.5\n")
        if with_performance:
            (out / "performance.txt").write_text("1.0\n")
        log = out / "run.log"
        if log_text is not None:
            log.write_text(log_text)

        class _ModelLog:
            simulation_completed = SimpleNamespace(get=lambda: True)

        def get_log(_model_type):
            if model_log_says == "recorded_complete":
                return _ModelLog()
            raise FileNotFoundError

        scenario = SimpleNamespace(
            _removed=False,
            get_log=get_log,
            scen_paths=SimpleNamespace(out_triton=out, out_tritonswmm=out, out_swmm=out, swmm_hydraulics_rpt=None),
        )
        stub = SimpleNamespace(_scenario=scenario, _analysis_level_model_logfile=lambda _mt: log)
        stub.performance_file = lambda model_type: TRITONSWMM_run.performance_file(stub, model_type=model_type)
        stub._coupled_swmm_report_finalized = lambda mt: TRITONSWMM_run._coupled_swmm_report_finalized(stub, mt)
        return TRITONSWMM_run.model_run_completed(stub, model_type)

    assert verdict("[IN] Reading file h.asc\nInvalid dimension of h.asc.\n", with_performance=True) is False
    assert verdict(None, with_performance=True) is False
    if model_type == "triton":
        # The control: the same tree with the marker present must read True, so the False above is the marker's
        # absence and not a stub that cannot answer True.
        assert verdict("[IN] Reading file h.asc\nSimulation ends\n", with_performance=True) is True

"""The DEM-rim check of ``swmm_node_rim_from_dem``, and the opt-in fallback that keeps SWMM's depth for a failing node.

The problem this file pins. With ``swmm_node_rim_from_dem`` true the solver sets each coupled node's rim to the DEM
cell elevation minus the node's invert, in place of the depth SWMM assigns the node (its ``MaxDepth`` raised to the
crown of its highest conduit). The solver refuses a cell at or below the invert, two minutes into a GPU job, and it
accepts a cell above the invert but below SWMM's depth, which leaves the node shallower than its own pipe. Scenario
preparation therefore judges every coupled node before any cfg line is written, refuses with every failing node
named, and offers one deliberate override, ``swmm_node_rim_swmm_fallback``, that lists the failing nodes in the
solver's per-node exemption file so the solver keeps SWMM's depth for them. With the key off or the field unset
nothing is read and nothing is written.

Every test runs the pre-existing coupled-cfg generator on stand-ins over a 10 by 10 DEM of 10 m cells and a small
SWMM model whose nodes differ in the rule that sets their depth. None compiles or runs a solver, so the file runs on
a laptop. It is a chunk test: it proves the change once and belongs with the scenario preparation tests at the
landing (module guarded: ``hhemt.scenario.check_dem_rim_against_swmm_depth``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import ConfigurationError

# Node: invert, MaxDepth, the conduit ends that raise it, and so the depth SWMM assigns.
#   A  1.0  0     one 0.5 pipe, offset 0                         -> 0.5
#   B  1.0  3.0   one 0.5 pipe                                   -> 0.5  (a MaxDepth above the crown is not a floor)
#   C  1.0  0     out pipe 0.6 at offset 0.2; in pipe 0.3 at 0.1 -> 0.8  (offset and two diameters)
#   E  1.0  0.3   one 0.5 pipe                                   -> 0.5  (MaxDepth under the crown: the crown)
#   K  1.0  0     a crown_rim outfall, exempt already            -> not judged
_MODEL = """[OPTIONS]
FLOW_UNITS CMS
LINK_OFFSETS {offsets}

[JUNCTIONS]
A {a_invert} 0 0 0 0
B 1.0 3.0 0 0 0
C 1.0 0 0 0 0
E 1.0 0.3 0 0 0
K 1.0 0 0 0 0
X 5.0 5 0 0 0

[OUTFALLS]
OUT 0.0 FREE NO

[CONDUITS]
cA A OUT 10 0.013 0 0
cB B OUT 10 0.013 0 0
cCout C OUT 10 0.013 {c_out_offset} 0
cXC X C 10 0.013 0 {c_in_offset}
cE E OUT 10 0.013 0 0
cK K OUT 10 0.013 0 0

[XSECTIONS]
cA CIRCULAR {a_dia} 0 0 0 1
cB CIRCULAR 0.5 0 0 0 1
cCout CIRCULAR 0.6 0 0 0 1
cXC CIRCULAR 0.3 0 0 0 1
cE CIRCULAR 0.5 0 0 0 1
cK CIRCULAR 0.5 0 0 0 1

[COORDINATES]
A 5 95
B 15 95
C 25 95
E 35 95
K 45 95
X 55 95
OUT 65 95
"""

#: SWMM's depth for each judged node, in the units of the model, and the cell (row 0, column) that holds it.
_DEPTH = {"A": 0.5, "B": 0.5, "C": 0.8, "E": 0.5}
_COLUMN = {"A": 0, "B": 1, "C": 2, "E": 3, "K": 4}
_INVERT = 1.0

_READS = (
    'argsd("swmm_coupling_nodes", argmap, "");\nargsd("swmm_manhole_table", argmap, "");\n'
    'argsd("swmm_dem_rim_exempt_nodes", argmap, "");\nargsd("swmm_node_rim_from_dem", argmap, "0");\n'
)
_READS_NO_EXEMPT = _READS.replace('argsd("swmm_dem_rim_exempt_nodes", argmap, "");\n', "")


class _AnyField:
    """A stand-in log whose every field reads False and accepts any value."""

    def __getattr__(self, name):
        return SimpleNamespace(get=lambda: False, set=lambda value: None)


def _write_dem(system_dir: Path, cells: dict[str, float], *, default: float = 100.0) -> None:
    """The processed DEM TRITON loads: 10 cells by 10, 10 m each, ``default`` everywhere but the node cells."""
    grid = [[default] * 10 for _ in range(10)]
    for node, z in cells.items():
        grid[0][_COLUMN[node]] = z
    header = ["ncols 10", "nrows 10", "xllcorner 0.0", "yllcorner 0.0", "cellsize 10.0", "NODATA_value -9999"]
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "elevation_10.00m.dem").write_text(
        "\n".join(header + [" ".join(repr(v) for v in row) for row in grid]) + "\n"
    )


def _ok_cells(extra: float = 0.0) -> dict[str, float]:
    """Every judged node's cell exactly at SWMM's depth above its invert, plus ``extra``; K far below its invert."""
    return {**{n: _INVERT + d + extra for n, d in _DEPTH.items()}, "K": -50.0}


def _scenario(
    tmp_path: Path,
    monkeypatch,
    cells: dict[str, float],
    *,
    rim: bool = True,
    resolution: str | None = "processed",
    fallback=None,
    clone_reads: str = _READS,
    offsets: str = "DEPTH",
    c_out_offset: str = "0.2",
    c_in_offset: str = "0.1",
    a_invert: float = 1.0,
    a_dia: float = 0.5,
):
    """A scenario whose coupled cfg generator runs on stand-ins; returns (scenario, cfg path, scenario directory)."""
    from hhemt import scenario as scenario_module
    from hhemt.scenario import TRITONSWMM_scenario

    tmp_path.mkdir(parents=True, exist_ok=True)
    inp = tmp_path / "hydraulics.inp"
    inp.write_text(
        _MODEL.format(
            offsets=offsets, c_out_offset=c_out_offset, c_in_offset=c_in_offset, a_invert=a_invert, a_dia=a_dia
        )
    )
    (tmp_path / "coupling_nodes.txt").write_text("A\nB\nC\nE\nK\n")
    (tmp_path / "dem_rim_exempt_nodes.txt").write_text("K\n")
    system_dir = tmp_path / "system"
    _write_dem(system_dir, cells)
    clone = tmp_path / "triton"
    (clone / "src").mkdir(parents=True, exist_ok=True)
    (clone / "src" / "config_utils.h").write_text(clone_reads)
    extbc = tmp_path / "extbc_loc.csv"
    extbc.write_text("x,y\n")
    (tmp_path / "TRITONSWMM.cfg").write_text("sim_start_time=0\n")
    options = SimpleNamespace(
        Value=SimpleNamespace(
            START_DATE="01/01/2020", START_TIME="00:00:00", END_DATE="01/01/2020", END_TIME="01:00:00"
        )
    )
    monkeypatch.setattr(
        scenario_module.swmmio, "Model", lambda path: SimpleNamespace(inp=SimpleNamespace(options=options))
    )
    fields = dict(
        manhole_diameter=1.2,
        manhole_loss_coefficient=0.1,
        manhole_table=None,
        manhole_table_fill_with_constants=False,
        TRITON_raw_output_type="bin",
        hydraulic_timestep_s=1.0,
        TRITON_courant_number=0.5,
        TRITON_hextra_m=0.001,
        TRITON_fixed_timestep=False,
        TRITON_domain_decomposition="static",
        TRITON_domain_decomposition_check_factor=None,
        TRITON_progress_print_iterations=10000,
        TRITON_reporting_timestep_s=60,
        open_boundaries=1,
        swmm_snapshot_disable=False,
        swmm_snapshot_keep_all=False,
        swmm_node_rim_from_dem=rim,
        dem_rim_resolution=resolution,
        swmm_open_water_outfalls=["K"],
        open_water_outfall_representation="crown_rim",
        execution_environment="native",
        one_coupled_inflow_node_per_dem_cell=False,
    )
    if fallback is not None:
        fields["swmm_node_rim_swmm_fallback"] = fallback
    cfg_system = SimpleNamespace(
        toggle_use_constant_mannings=True,
        constant_mannings=0.05,
        TRITONSWMM_software_directory=clone,
        TRITONSWMM_branch_key=None,
        system_directory=system_dir,
        target_dem_resolution=10.0,
        DEM_fullres=tmp_path / "absent_fullres.tif",
    )
    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen._analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(**fields), cfg_hpc_system=SimpleNamespace(gpu_aware_mpi=False)
    )
    scen._system = SimpleNamespace(
        cfg_system=cfg_system,
        sys_paths=SimpleNamespace(dem_processed=system_dir / "elevation_10.00m.dem", mannings_processed=tmp_path / "m"),
        log=SimpleNamespace(triton_head_sha=SimpleNamespace(get=lambda: None)),
    )
    cfg = tmp_path / "sim" / "TRITONSWMM.cfg"
    scen.scen_paths = SimpleNamespace(
        swmm_hydraulics_inp=inp,
        hyg_timeseries=tmp_path / "t.hyg",
        hyg_locs=tmp_path / "l.txt",
        extbc_loc=extbc,
        triton_swmm_cfg=cfg,
    )
    scen.sim_id_str = "evt"
    scen.log = _AnyField()
    scen._swmm_inp_for_sim_duration = lambda: inp
    scen._n_hydrograph_sources = lambda: 0
    return scen, cfg, tmp_path


def _failing_names(error: ConfigurationError) -> list[str]:
    """The node names an error lists, one per ``- NAME:`` row of its message, in order."""
    return re.findall(r"^    - (\w+):", str(error), flags=re.M)


def _refusal(tmp_path, monkeypatch, cells, **kwargs) -> ConfigurationError:
    scen, cfg, _ = _scenario(tmp_path, monkeypatch, cells, **kwargs)
    with pytest.raises(ConfigurationError) as caught:
        scen._generate_TRITON_SWMM_cfg()
    assert not cfg.exists(), "a refusal leaves no cfg line written"
    return caught.value


def test_a_cell_at_or_above_every_nodes_swmm_depth_passes_and_a_cell_below_one_names_only_it(tmp_path, monkeypatch):
    """Property: preparation refuses a coupled node whose DEM rim is below SWMM's depth, and passes one at or above.

    Class: the four judged nodes, each with the cell exactly at SWMM's depth above its invert (passes), 0.5 m above it
    (passes), and 0.01 m below it (refused), one node lowered at a time. The four depths come from four different
    rules: a plain crown (A), a crown under an authored MaxDepth of 3.0 (B, not failing at 1.5), the larger of an
    offset out pipe and a smaller in pipe (C) and a crown above an authored MaxDepth of 0.3 (E), so a reader that
    takes MaxDepth or only one pipe is wrong on at least one.
    Why this is required: a node whose DEM rim sits below its own pipe's crown is
    the shallow rim the solver accepts silently. What kills it: no check; comparing with MaxDepth alone; taking the
    sum of the offset and the diameter of one pipe only. A second correct implementation: reading the depths from a
    SWMM engine run instead of the text; the assertions read only whether preparation refuses and which nodes it
    names. Must-fire control: the lowered arms are the control of the passing arms.
    """
    scen, cfg, _ = _scenario(tmp_path / "exact", monkeypatch, _ok_cells())
    scen._generate_TRITON_SWMM_cfg()
    assert cfg.exists()
    scen, cfg, _ = _scenario(tmp_path / "above", monkeypatch, _ok_cells(0.5))
    scen._generate_TRITON_SWMM_cfg()
    for node in _DEPTH:
        cells = {**_ok_cells(), node: _INVERT + _DEPTH[node] - 0.01}
        error = _refusal(tmp_path / f"low_{node}", monkeypatch, cells)
        assert _failing_names(error) == [node]
        assert "0.0100 m" in str(error)  # the shortfall, as a number
        assert error.field == "swmm_node_rim_from_dem"


def test_a_cell_between_the_invert_and_swmms_depth_is_refused_as_is_one_at_or_below_the_invert(tmp_path, monkeypatch):
    """Property: the band the solver accepts silently, and the cells it refuses, are all refused at preparation.

    Class: node A (invert 1.0, SWMM depth 0.5) with its cell at 1.2 (above the invert, below the rim 1.5), at 1.0
    (the invert), at 0.9 (below it) and with no value (not a number). Why: a DEM rim of 0.2 m for a node whose own
    pipe is 0.5 m is a node full of water at 0.2 m, and the solver does not say so; the other three the solver
    refuses at run start, after the queue wait. What kills it: comparing only against the invert. A second correct
    implementation: one refusal reason for all four; the assertions read the nodes named and one reason word each.
    """
    for label, z, word in (
        ("band", 1.2, "below the depth"),
        ("invert", 1.0, "at or below"),
        ("under", 0.9, "at or below"),
    ):
        error = _refusal(tmp_path / label, monkeypatch, {**_ok_cells(), "A": z})
        assert _failing_names(error) == ["A"] and word in str(error), label
    error = _refusal(tmp_path / "nan", monkeypatch, {**_ok_cells(), "A": float("nan")})
    assert _failing_names(error) == ["A"] and "no value" in str(error)


def test_every_failing_node_is_named_with_its_numbers_and_the_three_remedies_in_order(tmp_path, monkeypatch):
    """Property: one refusal lists every failing node of the population, with invert, depth, cell and shortfall.

    Class: all four judged nodes 0.01 m below SWMM's depth at once. Why: a seat or a user who sees one node per
    prepare cycle repairs a 200-node model in 200 cycles, and the three remedies are the developer's ruling in its
    stated order (override, key off, amend the DEM or nodes). What kills it: raising on the first failing node.
    A second correct implementation: a table instead of rows; the assertions read the node names, the numbers
    and the order of the remedies in the hint.
    """
    cells = {**{n: _INVERT + d - 0.01 for n, d in _DEPTH.items()}, "K": -50.0}
    error = _refusal(tmp_path, monkeypatch, cells)
    assert _failing_names(error) == ["A", "B", "C", "E"]
    assert "SWMM depth 0.5000 m" in str(error) and "DEM rim 0.4900 m" in str(error) and "invert 1.0000 m" in str(error)
    hint = error.fix_hint
    assert hint.index("swmm_node_rim_swmm_fallback") < hint.index("swmm_node_rim_from_dem") < hint.index("amend")


def test_offsets_under_elevation_mode_and_a_crown_rim_outfall_are_read_as_swmm_reads_them(tmp_path, monkeypatch):
    """Property: under LINK_OFFSETS ELEVATION the offset is an elevation above sea level, and an exempt node is skipped.

    Class: node C's pipe offsets written as elevations (invert 1.0 plus 0.2 and plus 0.1), and K, the listed
    crown_rim outfall, with a cell far below its invert. Why: SWMM subtracts the node's invert from an elevation
    offset before it adds the diameter, so reading 1.2 as a height would put C's depth at 1.8 and refuse a node
    SWMM itself leaves at 0.8; K is exempt in the solver and judging it would refuse a run the solver accepts.
    What kills it: adding the elevation offset unchanged; judging the exempt nodes. A second correct implementation:
    converting the model to DEPTH offsets first; the assertions read whether preparation passes.
    """
    scen, cfg, _ = _scenario(
        tmp_path / "elev", monkeypatch, _ok_cells(), offsets="ELEVATION", c_out_offset="1.2", c_in_offset="1.1"
    )
    scen._generate_TRITON_SWMM_cfg()
    assert cfg.exists()  # K sat at -50 m with an invert of 1.0 and was not refused
    low = {**_ok_cells(), "C": _INVERT + 0.8 - 0.01}
    error = _refusal(
        tmp_path / "elev_low", monkeypatch, low, offsets="ELEVATION", c_out_offset="1.2", c_in_offset="1.1"
    )
    assert _failing_names(error) == ["C"] and "SWMM depth 0.8000 m" in str(error)


def test_the_override_lists_exactly_the_failing_nodes_with_the_crown_outfalls_and_records_them(tmp_path, monkeypatch):
    """Property: with the fallback on, preparation passes, lists the failing nodes after the crown_rim ones, records.

    Class: nodes A and E failing (A in the band, E below the invert), B and C passing, K the crown_rim outfall already
    in the exemption file, with a TRITON clone that reads the exemption key. Why: the solver, which is unchanged,
    keeps SWMM's depth for exactly the nodes in the file, so the file must be the union of the crown outfalls and the
    failing nodes and no more; the record is how a reader of the results learns that a rim is not from the DEM.
    What kills it: listing every node, listing only the new nodes (dropping K), or writing no record. A second correct
    implementation: the failing nodes in another order; the assertions compare sets and one order.
    """
    cells = {**_ok_cells(), "A": 1.2, "E": 0.5}
    scen, cfg, scen_dir = _scenario(tmp_path, monkeypatch, cells, fallback=True)
    with pytest.warns(UserWarning, match="swmm_node_rim_swmm_fallback keeps SWMM's depth"):
        scen._generate_TRITON_SWMM_cfg()
    assert (scen_dir / "dem_rim_exempt_nodes.txt").read_text().split() == ["K", "A", "E"]
    record = json.loads((scen_dir / "dem_rim_swmm_fallback.json").read_text())
    assert [n["name"] for n in record["nodes"]] == ["A", "E"]
    by_name = {n["name"]: n for n in record["nodes"]}
    assert by_name["A"]["swmm_depth_m"] == pytest.approx(0.5) and by_name["A"]["shortfall_m"] == pytest.approx(0.3)
    assert 'swmm_dem_rim_exempt_nodes="' in cfg.read_text() and "swmm_node_rim_from_dem=1" in cfg.read_text()
    # Control: the same cells with the fallback off refuse, naming the same two nodes.
    error = _refusal(tmp_path / "off", monkeypatch, cells, fallback=False)
    assert _failing_names(error) == ["A", "E"]


def test_the_override_alone_writes_the_exemption_file_and_a_build_that_cannot_read_it_is_refused(tmp_path, monkeypatch):
    """Property: with no crown_rim list the override still creates the exemption file, and needs a build that reads it.

    Class: no listed outfall and no pre-existing exemption file, node A failing; a clone that reads the exemption key
    and one that does not. Why: a build that ignores the key takes the DEM rim for A, the shallow rim the check
    found, while the configuration says SWMM's depth. What kills it: requiring a crown_rim list for the file; skipping
    the build guard. A second correct implementation: refusing earlier, at run entry; the assertions read the
    file and the refusal's field.
    """
    cells = {**_ok_cells(), "A": 1.2}
    scen, cfg, scen_dir = _scenario(tmp_path / "ok", monkeypatch, cells, fallback=True)
    scen._analysis.cfg_analysis.swmm_open_water_outfalls = []
    (scen_dir / "dem_rim_exempt_nodes.txt").unlink()
    (scen_dir / "coupling_nodes.txt").write_text("A\nB\nC\nE\n")
    with pytest.warns(UserWarning):
        scen._generate_TRITON_SWMM_cfg()
    assert (scen_dir / "dem_rim_exempt_nodes.txt").read_text() == "A\n"
    scen, cfg, scen_dir = _scenario(tmp_path / "blind", monkeypatch, cells, fallback=True, clone_reads=_READS_NO_EXEMPT)
    scen._analysis.cfg_analysis.swmm_open_water_outfalls = []
    (scen_dir / "coupling_nodes.txt").write_text("A\nB\nC\nE\n")
    with pytest.raises(ConfigurationError) as caught:
        scen._generate_TRITON_SWMM_cfg()
    assert caught.value.field == "swmm_node_rim_swmm_fallback"
    assert not cfg.exists()


@pytest.mark.parametrize(
    ("rim", "fallback", "refused"),
    [(False, True, True), (True, True, False), (False, False, False), (False, None, False)],
)
def test_the_fallback_with_no_dem_rim_written_is_a_refused_configuration(rim, fallback, refused):
    """Property: the preflight refuses ``swmm_node_rim_swmm_fallback`` unless the cfg will carry a DEM rim.

    Class: the field set with the key off (refused), with the key on and the processed DEM (accepted), unset with the
    key off (accepted), and the field absent from the stand-in (accepted). Why: a field that does nothing is a
    configuration error, as ``dem_rim_resolution`` is. What kills it: no refusal, or refusing the field when it is
    False. A second correct implementation: the refusal in a model validator; the assertions read the preflight's
    problem list for the field.
    """
    from hhemt.water_level_boundary import preflight_problems

    cfg_system = SimpleNamespace(
        SWMM_model=Path(__file__),
        swmm_outfalls_to_couple=[],
        toggle_tritonswmm_model=True,
        system_directory=Path("."),
        target_dem_resolution=10.0,
        DEM_fullres=Path("absent.tif"),
    )
    fields = dict(
        toggle_external_boundaries=False,
        swmm_outfalls_to_keep=[],
        swmm_open_water_outfalls=[],
        open_water_outfall_representation="driven_stage",
        swmm_node_rim_from_dem=rim,
        dem_rim_resolution="processed" if rim else None,
        water_level_boundary_series_source=None,
    )
    if fallback is not None:
        fields["swmm_node_rim_swmm_fallback"] = fallback
    # preflight_problems returns early for a model it cannot read, so give it a readable one.
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        model = Path(raw) / "m.inp"
        model.write_text("[OUTFALLS]\nOUT 0 FREE NO\n[COORDINATES]\nOUT 1 1\n")
        cfg_system.SWMM_model = model
        report = preflight_problems(cfg_system, SimpleNamespace(**fields))
    hit = [p for p in report.problems if p.field == "analysis.swmm_node_rim_swmm_fallback"]
    assert bool(hit) is refused


@pytest.mark.parametrize(
    ("rim", "resolution", "fallback"),
    [(False, None, None), (False, None, False), (True, "full", None), (True, "processed", None)],
)
def test_with_the_key_off_or_the_field_unset_nothing_is_read_and_no_file_is_written(
    tmp_path, monkeypatch, rim, resolution, fallback
):
    """Property: a configuration that does not use the check writes today's cfg and today's files and reads no DEM.

    Class: the key off, the key on with the full-resolution DEM (the toolkit writes depths and no key), and the key on
    with the processed DEM and the field unset over cells that all pass; the DEM reader is replaced by one that fails
    the test when called in the first two. Why: the default behaviour of every configuration that does not set the
    field is byte-identical, and reading a DEM for a run that never uses one costs seconds per scenario on a
    Norfolk-size raster. What kills it: running the check when the key is not written; writing the record file
    whenever the field is read. A second correct implementation: caching the DEM read; the assertions read the
    files and the call count.
    """
    import hhemt.swmm_utils as swmm_utils

    reads: list[str] = []
    real = swmm_utils.processed_dem_grid
    monkeypatch.setattr(swmm_utils, "processed_dem_grid", lambda cfg: reads.append("dem") or real(cfg))
    scen, cfg, scen_dir = _scenario(
        tmp_path, monkeypatch, _ok_cells(), rim=rim, resolution=resolution, fallback=fallback
    )
    scen._generate_TRITON_SWMM_cfg()
    assert cfg.exists()
    assert not (scen_dir / "dem_rim_swmm_fallback.json").exists()
    assert (scen_dir / "dem_rim_exempt_nodes.txt").read_text() == "K\n"  # the one the test wrote, untouched
    assert bool(reads) is (rim and resolution == "processed")


@pytest.mark.parametrize(
    ("node", "invert", "crown", "cell"),
    [("E144026", 1.542, 0.381, 1.910770058631897), ("F143940", 2.542, 0.457, 2.9585800170898438)],
)
def test_the_measured_silent_band_nodes_of_the_3_5_m_raster_are_refused_and_their_crown_is_not(
    tmp_path, monkeypatch, node, invert, crown, cell
):
    """Property: a node with its cell above the invert but below the crown floor is refused, at the crown it passes.

    Class: the two Norfolk nodes measured at 3.5 m on Rivanna (invert, crown and cell as measured, shortfalls 0.0122
    and 0.0404 m), rebuilt as node A of the fixture with its pipe diameter set to the crown; control: the cell raised to
    invert plus crown. Why: this is the band the solver accepts silently. What kills it: comparing with the invert only.
    A second correct implementation: another reason wording; the assertions read the node and the shortfall.
    """
    cells = {**_ok_cells(), "A": cell}
    error = _refusal(tmp_path / "low", monkeypatch, cells, a_invert=invert, a_dia=crown)
    assert _failing_names(error) == ["A"] and f"SWMM depth {crown:.4f} m" in str(error)
    ok = {**_ok_cells(), "A": invert + crown}
    scen, cfg, _ = _scenario(tmp_path / "ok", monkeypatch, ok, a_invert=invert, a_dia=crown)
    scen._generate_TRITON_SWMM_cfg()
    assert cfg.exists(), node

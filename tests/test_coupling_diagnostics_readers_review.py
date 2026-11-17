"""Reviewer tests for the hhemt readers of the coupling-diagnostics capture files.

Each test names the input class it quantifies over, why the property is required, and the change
to production code that kills it. These are the reviewing seat's tests; the coder greens them.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from hhemt import coupling_diagnostics as cd
from hhemt.analysis import PERF_VARS
from hhemt.cf_conventions import apply_cf_attributes
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import parse_performance_file


def _header(magic: int, n_items: int) -> bytes:
    return struct.pack("<4i", magic, 1, 8, n_items)


def _bodyless_cases(tmp_path: Path):
    """One valid file per reader that holds a header (or header row) and zero records."""
    nodes = tmp_path / "swmm_node_diag.nodes"
    nodes.write_text("N" + "1\nN" + "2\n")
    links = tmp_path / "swmm_link_flow_diag.links"
    links.write_text("L1\n")
    files = {
        "step_diag": (tmp_path / "swmm_step_diag.bin", _header(cd.MAGIC_SWMM_STEP_DIAG, 0)),
        "node_path3": (tmp_path / "swmm_node_path3.bin", _header(cd.MAGIC_SWMM_NODE_PATH3, 2)),
        "node_path3_step": (tmp_path / "swmm_node_path3_step.bin", _header(cd.MAGIC_SWMM_NODE_PATH3_STEP, 2)),
        "link_flow": (tmp_path / "swmm_link_flow_diag.bin", _header(cd.MAGIC_SWMM_LINK_FLOW_DIAG, 1)),
        "edge_flux": (tmp_path / "edge_flux.bin", _header(cd.MAGIC_EDGE_FLUX, 4)),
        "dt_limiter": (tmp_path / "dt_limiter.bin", _header(cd.MAGIC_DT_LIMITER, 0)),
    }
    for path, blob in files.values():
        path.write_bytes(blob)
    wet = tmp_path / "wet_cells.csv"
    wet.write_text("%print_id, simtime, rank, wet_mean, wet_at_print\n")
    balance = tmp_path / "domain_balance.csv"
    balance.write_text("%" + ", ".join(cd._DOMAIN_BALANCE_COLUMNS) + "\n")
    return {
        "step_diag": lambda: cd.read_swmm_step_diag(files["step_diag"][0]),
        "node_path3": lambda: cd.read_swmm_node_path3(files["node_path3"][0], nodes),
        "node_path3_step": lambda: cd.read_swmm_node_path3_step(files["node_path3_step"][0], nodes),
        "link_flow": lambda: cd.read_swmm_link_flow_diag(files["link_flow"][0], links),
        "edge_flux": lambda: cd.read_edge_flux(files["edge_flux"][0]),
        "dt_limiter": lambda: cd.read_dt_limiter(files["dt_limiter"][0]),
        "wet_cells": lambda: cd.read_wet_cells(wet),
        "domain_balance": lambda: cd.read_domain_balance(balance),
    }


@pytest.mark.parametrize(
    "reader",
    [
        "step_diag",
        "node_path3",
        "node_path3_step",
        "link_flow",
        "edge_flux",
        "dt_limiter",
        "wet_cells",
        "domain_balance",
    ],
)
def test_a_valid_file_with_no_records_never_escapes_as_a_non_processing_error(tmp_path, reader):
    """Property: every reader, handed a file whose header is valid and whose body holds zero
    records, returns a Dataset or raises ``ProcessingError``; no other exception type escapes.

    Class: the eight readers x the body-less valid file. The solver's file format states a text header row is
    "always written", and the capture writers open every file before the first print interval, so
    a run that ends or is cut before its first interval leaves exactly this file; the chunk's own
    ``test_step_diag_of_a_header_only_file_is_empty_not_an_error`` already pins the empty arm for
    one reader.
    Why: ingestion and the validation layer catch ``ProcessingError`` as the typed
    refusal; an ``IndexError`` out of ``read_wet_cells`` is neither a Dataset nor that refusal, so
    it surfaces as an unexplained crash of the whole processing step.
    What kills it: indexing ``simtime[0]`` of an empty grid in ``read_wet_cells`` without a guard
    (the present code).
    Second correct implementation: ``read_wet_cells`` raising ``ProcessingError`` for an empty
    body, or returning an empty Dataset; either passes.
    """
    call = _bodyless_cases(tmp_path)[reader]
    try:
        result = call()
    except ProcessingError:
        return
    assert hasattr(result, "data_vars"), reader


def _populated_outputs(tmp_path: Path) -> dict:
    """One non-empty Dataset per reader, for the property over variables."""
    nodes = tmp_path / "n.nodes"
    nodes.write_text("N" + "1\n")
    links = tmp_path / "l.links"
    links.write_text("L1\n")
    out = {}

    step = tmp_path / "s.bin"
    step.write_bytes(
        _header(cd.MAGIC_SWMM_STEP_DIAG, 0)
        + struct.pack("<ddiiddd", 1.0, 0.5, 3, 1, 1.0, -1.0, 2.0)
        + struct.pack("<i", 0)
    )
    out["step_diag"] = cd.read_swmm_step_diag(step)

    p3 = tmp_path / "p3.bin"
    p3.write_bytes(_header(cd.MAGIC_SWMM_NODE_PATH3, 1) + struct.pack("<dddi", 120.0, 1.0, -1.0, 2))
    out["node_path3"] = cd.read_swmm_node_path3(p3, nodes)

    p3s = tmp_path / "p3s.bin"
    p3s.write_bytes(_header(cd.MAGIC_SWMM_NODE_PATH3_STEP, 1) + struct.pack("<ddd", 1.0, 0.5, 3.0))
    out["node_path3_step"] = cd.read_swmm_node_path3_step(p3s, nodes)

    lf = tmp_path / "lf.bin"
    lf.write_bytes(_header(cd.MAGIC_SWMM_LINK_FLOW_DIAG, 1) + struct.pack("<dd", 1.0, 2.0))
    out["link_flow"] = cd.read_swmm_link_flow_diag(lf, links)

    ef = tmp_path / "ef.bin"
    ef.write_bytes(_header(cd.MAGIC_EDGE_FLUX, 4) + struct.pack("<ddddd", 1.0, 1.0, 2.0, 3.0, 4.0))
    out["edge_flux"] = cd.read_edge_flux(ef)

    dtl = tmp_path / "dtl.bin"
    dtl.write_bytes(_header(cd.MAGIC_DT_LIMITER, 0) + struct.pack("<ddiiiidd", 1.0, 0.1, 0, 3, 4, 0, 0.5, 2.0))
    out["dt_limiter"] = cd.read_dt_limiter(dtl)

    wet = tmp_path / "w.csv"
    wet.write_text("%print_id, simtime, rank, wet_mean, wet_at_print\n0, 120.0, 0, 5.0, 6\n")
    out["wet_cells"] = cd.read_wet_cells(wet)

    bal = tmp_path / "b.csv"
    bal.write_text("%" + ", ".join(cd._DOMAIN_BALANCE_COLUMNS) + "\n" + ", ".join(["120"] + ["1.0"] * 11) + "\n")
    out["domain_balance"] = cd.read_domain_balance(bal)
    return out


def test_every_reader_data_variable_resolves_units_after_the_cf_stamp(tmp_path):
    """Property: after ``apply_cf_attributes`` (the stamp the consolidation stage applies), every
    data variable of every reader's Dataset carries ``units``, except a flag-valued categorical.

    Class: all data variables of the eight readers' outputs, derived from the Datasets
    themselves, never a count. ``read_swmm_node_path3`` converts ``path3_volume_pos`` and
    ``path3_volume_neg`` from ft3 to m3 and returns them, and ``nonconverged_steps``, with no
    attribute at all, while its T1a sibling ``read_swmm_step_diag`` labels the same quantity "m3".
    Why: the module converts a volume to m3 on read, so a consumer of the converted array has no
    way to tell m3 from the engine's ft3 or from a count; ``cf_conventions._CF_VARIABLE_MAP`` has
    no entry for any of these names (``apply_cf_attributes`` then only adds an auto ``long_name``),
    so the reader's own attributes are the only source today.
    What kills it: stamping no ``units`` on the three node-path-3 variables (the present code).
    Second correct implementation: the readers return bare variables and ``_CF_VARIABLE_MAP``
    gains entries for the names; ``apply_cf_attributes`` then supplies the units and this passes.
    Reachability: ``read_swmm_node_path3`` on any valid T1b file (the one the chunk's own
    ``test_node_path3_maps_each_value_to_its_node_and_interval`` reads).
    """
    unitless: dict[str, list[str]] = {}
    for reader, ds in _populated_outputs(tmp_path).items():
        stamped = apply_cf_attributes(ds.copy(deep=False), "tritonswmm_triton")
        missing = [
            str(name)
            for name, da in stamped.data_vars.items()
            if "flag_values" not in da.attrs and "units" not in da.attrs
        ]
        if missing:
            unitless[reader] = missing
    assert not unitless, f"variables with no units after the CF stamp, by reader: {unitless}"


def test_the_16_column_header_in_the_e2_file_order_parses_every_perf_name_by_name(tmp_path):
    """Property (HELD, green before this turn): a ``performance.txt`` whose header follows the file
    order the solver's file format fixes (``SWMM_SOLVE`` and ``SWMM_LOG`` right after ``SWMM_OTHER``,
    then ``DIAG``, before ``Other``) is parsed by name, and every ``PERF_VARS`` name is a column.

    Class: the header order the solver chunk will emit; hhemt's list order differs from it (DIAG
    after Resize), so this is the case where order-dependence would show.
    Why: the coder's round report rests "does not matter" on the parse being header-driven; this
    runs the real parser on the real order rather than trusting that.
    What kills it: a positional parse of the header, or a PERF_VARS name the parser drops.
    Second correct implementation: any header-driven parse.
    """
    cols = [
        "Compute", "MPI", "IO", "Resize", "SWMM", "SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER",
        "SWMM_SOLVE", "SWMM_LOG", "DIAG", "Other", "Simulation", "Init", "Total",
    ]  # fmt: skip
    path = tmp_path / "performance.txt"
    rows = "\n".join(f"{r}, " + ", ".join(str(float(i + r)) for i in range(len(cols))) for r in (0, 1))
    path.write_text(
        "%" + ", ".join(["Rank", *cols]) + "\n" + rows + "\nAverage, " + ", ".join(["0"] * len(cols)) + "\n"
    )
    df_ranks, _ = parse_performance_file(path)
    assert set(PERF_VARS) <= set(df_ranks.columns)
    assert float(df_ranks.loc[0, "DIAG"]) == 11.0
    assert float(df_ranks.loc[1, "SWMM_LOG"]) == 11.0

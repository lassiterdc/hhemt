"""Reviewer test: the hydrology reader returns every report period of a real `hydro.out`.

Problem this file solves: the per-node inflow capture and `tseries.hyg` are built from `read_hydrology_node_inflow`,
and under pyswmm 2.1.0 the accessor the reader used before took one period too few of every series (11 of 12 on the
committed goldens), so the capture and the hydrograph file were short by the final period while pyswmm 2.0.1 read the
same file whole. The coder's tests drive the reader through constructed doubles; this one drives it over the real
`swmm/hydro.out` of the committed multi-sim golden, so it fails on the installed pyswmm that drops the period.
No solver, no compile, no Snakemake: the golden is extracted into `tmp_path` and `hydro.out` is only read.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pyswmm import Output
from swmm.toolkit import output as swmm_output
from swmm.toolkit.shared_enum import Time

from hhemt.swmm_runoff_modeling import read_hydrology_node_inflow
from tests.fixtures.golden_trees import _hydro_inp_nodes, extract_golden


@pytest.mark.parametrize("kind", ["multisim", "sensitivity"])
def test_the_reader_returns_every_report_period_of_a_real_hydro_out(tmp_path, kind):
    """Property: for every scenario of an extracted golden, `read_hydrology_node_inflow` returns, for each node, a
    series with exactly `NUM_PERIODS` values, and a time axis of the same length.

    Class: every scenario of both committed goldens, every non-outfall node of its `hydro.inp`. Reachability: the
    golden `hydro.out` has 12 periods and the reader is the one `write_hydrograph_files` and the capture writer call.
    Plausible wrong implementation: reading the series through `Output.node_series`, which on pyswmm 2.1.0 returns 11
    of 12. Must-fire control: the period count is read from the file through the toolkit (`get_times(NUM_PERIODS)`)
    and asserted greater than one, and the node set is non-empty, so a reader that returns nothing fails here.
    Why required: a short series drops the last period of the capture and of `tseries.hyg` without any error.
    Killed by: the unfixed reader on pyswmm 2.1.0. Passes under a second correct implementation that reads each
    series through the toolkit over the window (0, NUM_PERIODS). Not established: on pyswmm 2.0.1 the unfixed reader
    also passes, because that version reads the file whole, so this test discriminates only where the accessor drops
    the period.
    """
    root = extract_golden(kind, tmp_path / kind)
    inputs = sorted(root.rglob("swmm/hydro.inp"))
    assert inputs
    for inp in inputs:
        nodes = list(_hydro_inp_nodes(inp))
        assert nodes, inp
        with Output(str(inp.with_suffix(".out"))) as out:
            n_periods = int(swmm_output.get_times(out.handle, Time.NUM_PERIODS))
        assert n_periods > 1, inp
        frame = pd.DataFrame({"node_key": nodes, "dem_x_coord": range(len(nodes)), "dem_y_coord": 0})
        result = read_hydrology_node_inflow(str(inp.with_suffix(".out")), frame, [])
        assert set(result.inflow_by_node) == set(nodes), inp
        assert {len(v) for v in result.inflow_by_node.values()} == {n_periods}, inp
        assert len(result.times_hr) == n_periods, inp

"""Reviewer tests for the golden hydrology capture and the routes that delete a scenario's processed outputs.

Problem this file solves: with the package defaults the SWMM hydrology output is reclaimed, so the per-node inflow
capture in ``processed/hydrology_inflow_summary.zarr`` is the only source the hydrology consolidation has left. Two
things must hold for that to work. The extractor of the committed goldens must write a capture that agrees with the
golden's own ``hydro.out`` on everything the consolidation reads except the flow values, and every route that deletes
``processed/`` for a process-stage rebuild must keep it. The coder's tests pin one route and compare the extractor
with its own constants; these cover the class each of them samples at one point.

No solver, no compile, no Snakemake: the goldens are extracted into ``tmp_path`` and ``hydro.out`` is only read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.golden_trees import _hydro_inp_nodes, extract_golden

_CAPTURE = "hydrology_inflow_summary.zarr"


def _seed_processed(sim: Path, *, capture: bool) -> None:
    """A scenario directory whose ``processed/`` holds two summary stores, the capture when asked, and a raw sibling."""
    (sim / "processed" / "a_summary.zarr").mkdir(parents=True)
    (sim / "processed" / "a_tseries.zarr").mkdir()
    if capture:
        (sim / "processed" / _CAPTURE).mkdir()
        (sim / "processed" / _CAPTURE / "zarr.json").write_text("{}")
    (sim / "out_triton").mkdir()
    (sim / "out_triton" / "h_0001.bin").write_text("raw")


def _run_delete_processed_runner(analysis_dir: Path, event_ids: list[str]) -> None:
    """Drive ``delete_processed_runner`` once per event, as the per-event Snakemake rule does."""
    from hhemt import delete_processed_runner as runner

    for event_id in event_ids:
        assert runner.main(["--event-id", event_id, "--analysis-dir", str(analysis_dir)]) == 0


def _run_delete_member_reprocess_runner(analysis_dir: Path, event_ids: list[str]) -> None:
    """Drive ``delete_member_reprocess_runner`` once for the member, as the per-member Snakemake rule does."""
    from hhemt import delete_member_reprocess_runner as runner

    rc = runner.main(["--member-id", "m0", "--analysis-dir", str(analysis_dir), "--delete-processed"])
    assert rc == 0


@pytest.mark.parametrize(
    "drive",
    [_run_delete_processed_runner, _run_delete_member_reprocess_runner],
    ids=["delete_processed_runner", "delete_member_reprocess_runner"],
)
def test_every_runner_that_deletes_processed_outputs_keeps_the_per_node_inflow_capture(tmp_path, monkeypatch, drive):
    """Property: after a process-stage delete runner has run, a scenario that held the capture still holds exactly the
    capture under ``processed/``, and a scenario that held none has no ``processed/`` directory; the raw ``out_*``
    sibling of both survives.

    Class: the SLURM-offloaded delete routes of a process-stage rebuild (one runner per analysis event, one per
    sensitivity member), over a scenario with the capture and a scenario without it, in that order and in the other.
    Plausible wrong implementation: a runner that removes the whole ``processed/`` directory, which deletes the capture
    that the hydrology consolidation reads once ``hydro.out`` is reclaimed. Reachability: ``workflow.py`` emits
    ``--delete-processed`` for the member rule whenever ``start_with`` is ``process``.
    Must-fire control: the scenario without a capture loses its whole directory, so a runner that keeps everything
    also fails.

    Why required: a rebuild that deletes the capture leaves the consolidation with no hydrology source and it refuses
    the scenario, naming the capture. Killed by: ``delete_and_account([processed_dir], ...)`` in either runner. Passes
    under a second correct implementation that moves the capture aside and back, or that lists the doomed children.
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    analysis_dir = tmp_path / "analysis"
    _seed_processed(analysis_dir / "sims" / "evt_with", capture=True)
    _seed_processed(analysis_dir / "sims" / "evt_without", capture=False)
    _seed_processed(analysis_dir / "sims" / "evt_z_with", capture=True)

    drive(analysis_dir, ["evt_with", "evt_without", "evt_z_with"])

    for event_id in ("evt_with", "evt_z_with"):
        kept = sorted(p.name for p in (analysis_dir / "sims" / event_id / "processed").iterdir())
        assert kept == [_CAPTURE], (event_id, kept)
    assert not (analysis_dir / "sims" / "evt_without" / "processed").exists()
    for event_id in ("evt_with", "evt_without", "evt_z_with"):
        assert (analysis_dir / "sims" / event_id / "out_triton" / "h_0001.bin").exists()


def _golden_scenarios(root: Path) -> list[Path]:
    """Every scenario directory of an extracted golden that carries a hydrology input."""
    return [inp.parent.parent for inp in sorted(root.rglob("swmm/hydro.inp"))]


@pytest.mark.parametrize("kind", ["multisim", "sensitivity"])
def test_the_extracted_capture_agrees_with_the_golden_hydro_out_on_nodes_and_report_steps(tmp_path, kind):
    """Property: for every scenario of an extracted golden, the capture holds an ``inflow`` series for exactly the
    nodes the golden's own ``hydro.out`` stores, on as many report steps as that file has and at the same elapsed
    hours.

    Class: every scenario of both committed goldens. Plausible wrong implementation: a capture whose step count or step
    length is a constant of the extractor that the golden's ``hydro.out`` does not share (the coder's test compares the
    capture with that same constant, so it cannot fail on it). Must-fire control: ``hydro.out`` is read here through the
    toolkit's own output reader, and the node and step counts are asserted non-zero first.

    Why required: the consolidation appends the capture in place of ``hydro.out``, so a capture on a different time axis
    consolidates into a group that no longer describes the scenario the golden stands for. Killed by: changing
    ``HYDROLOGY_CAPTURE_STEPS`` or ``HYDROLOGY_CAPTURE_STEP_S``, or building the node set from anything but the model.
    Passes under a second correct implementation that copies the series of ``hydro.out`` into the capture.
    Not established: the flow VALUES are the extractor's, by design, and are not compared with ``hydro.out``.
    """
    import xarray as xr
    from pyswmm import Output
    from swmm.toolkit import output as swmm_output
    from swmm.toolkit.shared_enum import NodeAttribute, Time

    root = extract_golden(kind, tmp_path / kind)
    scenarios = _golden_scenarios(root)
    assert scenarios
    for sim in scenarios:
        capture = xr.open_zarr(sim / "processed" / _CAPTURE, consolidated=False).load()
        nodes = sorted(capture["node_id"].values.tolist())
        assert nodes and nodes == sorted(_hydro_inp_nodes(sim / "swmm" / "hydro.inp"))
        with Output(str(sim / "swmm" / "hydro.out")) as out:
            # The reference reads the toolkit's output reader over the window (0, NUM_PERIODS), and not
            # Output.node_series: pyswmm 2.1.0 resolves that accessor's default end one period short, so a
            # reference built on it would hold 11 steps for a file of 12 and disagree with a correct capture.
            n_periods = int(swmm_output.get_times(out.handle, Time.NUM_PERIODS))
            times = list(out.times)
            assert len(times) == n_periods, sim
            series = {
                n: pd.Series(
                    swmm_output.get_node_series(out.handle, out.nodes[n], NodeAttribute.TOTAL_INFLOW, 0, n_periods),
                    index=times,
                )
                for n in nodes
            }
        first = series[nodes[0]]
        assert len(first) > 0
        hours = (first.index - first.index[0]).total_seconds().to_numpy() / 3600.0
        assert capture.sizes["timestep"] == len(first), sim
        assert np.allclose(capture["timestep"].values, hours), sim


@pytest.mark.parametrize("kind", ["multisim", "sensitivity"])
def test_two_extractions_of_one_golden_write_byte_identical_captures(tmp_path, kind):
    """Property: extracting a golden twice into different directories writes, for every scenario, a capture whose
    every file (metadata and chunks) has the same bytes.

    Class: every scenario of both committed goldens. Plausible wrong implementation: a capture carrying a clock reading,
    a random draw or a path (an attribute stamped at write time), which makes two runs of one test see different
    stores. Must-fire control: the file list is asserted non-empty and equal between the two trees before the bytes are
    compared, so an empty comparison cannot pass.

    Why required: the goldens are the fixed input the reprocess tests assert against; an input that varies between
    runs turns a real failure into one that reproduces only sometimes. Killed by: any value in the capture that depends
    on the moment or the directory of the extraction. Passes under a second correct implementation that writes the
    stores from a different closed formula, provided it is a function of node, event and step alone.
    """

    def digests(root: Path) -> dict[str, str]:
        """The sha256 of every file of every scenario's capture, keyed by its path relative to ``root``."""
        found = {}
        for sim in _golden_scenarios(root):
            for path in sorted((sim / "processed" / _CAPTURE).rglob("*")):
                if path.is_file():
                    found[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return found

    first = digests(extract_golden(kind, tmp_path / "first" / kind))
    second = digests(extract_golden(kind, tmp_path / "second" / kind))
    assert first and sorted(first) == sorted(second)
    assert first == second


def test_the_keeping_fixture_changes_the_clearing_switch_and_nothing_else(
    golden_sensitivity_analysis_keeping_scenarios, tmp_path
):
    """Property: the resolved analysis configuration of ``golden_sensitivity_analysis_keeping_scenarios`` equals that of
    the plain sensitivity golden in every field except ``toggle_clear_scenario_timeseries_after_consolidation``, which
    is False in the first and True in the second.

    Class: every field of the analysis configuration. Plausible wrong implementation: a fixture that also pins
    ``clear_raw`` or ``remove_after_processing`` to a keeping value, which would make the rows that use it silent about
    the defaults they are meant to run under. Must-fire control: the plain golden resolves the switch True (the
    default), so the one permitted difference is asserted to exist.

    Why required: the rows that use this fixture exercise a reprocess of a kept scenario; naming more than the one
    switch they need would hide behaviour under the defaults. Killed by: any added key in the fixture's
    ``analysis_config.yaml`` other than the switch. Passes under a second correct implementation that sets the switch
    through ``additional_analysis_configs`` or an override, provided the resolved configuration is the same.
    """
    from tests.fixtures.golden_trees import construct_golden_analysis

    keeping = golden_sensitivity_analysis_keeping_scenarios
    keeping_root = tmp_path / "golden_sensitivity"
    plain_dir = extract_golden("sensitivity", tmp_path / "plain")
    plain = construct_golden_analysis(plain_dir)

    def resolved(analysis, root: Path) -> dict:
        """The analysis configuration as plain data, with the extraction root replaced by a token."""
        return json.loads(json.dumps(analysis.cfg_analysis.model_dump(mode="json")).replace(str(root), "@ROOT@"))

    left, right = resolved(keeping, keeping_root), resolved(plain, plain_dir)
    switch = "toggle_clear_scenario_timeseries_after_consolidation"
    assert (left.pop(switch), right.pop(switch)) == (False, True)
    assert left == right

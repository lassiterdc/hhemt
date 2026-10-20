"""The gridded TRITON timeseries under ``target_processed_output_type: nc``.

THE PROBLEM. Until this change both gridded exporters published their chapter set as a
zarr STORE at the ``.nc``-named path, which the reader then opened with the ``h5netcdf``
engine and raised on. ``process_simulation.publish_chapters_to_netcdf`` is the ``nc`` arm
that now publishes one NetCDF FILE there. These tests pin the three things that arm must
do: write a file the production reader opens, leave nothing behind but recoverable chapters
when the write fails, and be reached from BOTH gridded exports.

TWO TIERS IN ONE FILE, so select by NODE ID and never run the file whole. The first two
tests are solver-free: they write flagged chapter stores directly with xarray and run the
publisher on them, and they run anywhere. The third runs the synthetic models end to end,
which compiles and executes a solver, so it runs on Rivanna scratch only and never on a
laptop.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import hhemt.process_simulation as process_simulation
from hhemt.process_simulation import TRITONSWMM_sim_post_processing, publish_chapters_to_netcdf
from hhemt.utils import chapter_flag_for, chapter_store_for, completed_chapters, unified_flag_for

_VARS = ("H", "QX")


def _write_chapters(root, lengths, ny: int = 5, nx: int = 6) -> None:
    """One flagged chapter store per entry of ``lengths``, with distinct values per timestep.

    The values encode the timestep, row and column, so a concatenation in the wrong order or
    a dropped chapter changes the data and not only the shape.
    """
    root.mkdir(parents=True, exist_ok=True)
    first = 0
    for index, n_t in enumerate(lengths):
        t = np.arange(first, first + n_t)
        data = (t[:, None, None] * 1000.0 + np.arange(ny)[None, :, None] * 10.0 + np.arange(nx)[None, None, :]).astype(
            "float64"
        )
        ds = xr.Dataset(
            {name: (("timestep_min", "y", "x"), data + offset) for offset, name in enumerate(_VARS)},
            coords={"timestep_min": t, "y": np.arange(ny), "x": np.arange(nx)},
        )
        ds.to_zarr(chapter_store_for(root, index), mode="w", consolidated=False)
        chapter_flag_for(root, index).write_text("ok", encoding="utf-8")
        first += n_t


def _open_like_production(path):
    """Open ``path`` through the production reader, ``TRITONSWMM_sim_post_processing._open``,
    with the engine its ``_open_engine`` selects for ``nc``."""
    reader = SimpleNamespace(_analysis=SimpleNamespace(cfg_analysis=SimpleNamespace(target_processed_output_type="nc")))
    reader._open_engine = lambda: TRITONSWMM_sim_post_processing._open_engine(reader)
    return TRITONSWMM_sim_post_processing._open(reader, path)


def test_nc_tail_publishes_a_netcdf_file_equal_to_the_chapter_concatenation(tmp_path):
    """Property: under ``nc`` the published artifact is a NetCDF FILE at the ``.nc`` path
    that the production reader opens, and its values and timestep coordinate equal the
    chapters concatenated along ``timestep_min``; the completion flag is written and the
    chapters are gone.

    Class: NEW CAPABILITY. The plausible wrong implementation is the pre-change tail,
    ``utils.merge_chapters_to_unified`` called on the ``.nc`` path, which publishes a zarr
    DIRECTORY there; the production reader then raises ``IsADirectoryError`` from h5netcdf.

    Input class: any contiguous flagged chapter set, here three chapters of unequal length
    (4, 3, 2 timesteps) so an off-by-one at a chapter seam changes the timestep coordinate.

    Why it is required: the summary step reads this artifact through ``_open`` with the
    ``h5netcdf`` engine under ``nc``, so an artifact that is not a NetCDF file fails every
    ``nc`` analysis at the summary.

    Kills it: dispatching ``nc`` to ``merge_chapters_to_unified``; concatenating
    ``sorted(parts)`` in any other order; dropping the last chapter.

    A second correct implementation under which it still passes: merge the chapters to a
    temporary zarr store, convert that store with ``utils.write_zarr_then_netcdf`` onto a
    temporary file, and ``os.replace`` it into place -- the test asserts the published file,
    its content and the flag, never how the bytes were produced.
    """
    chapters = tmp_path / "TRITON_only_tseries.nc.chapters"
    _write_chapters(chapters, [4, 3, 2])
    expected = xr.concat(
        [xr.open_zarr(chapter_store_for(chapters, k), consolidated=False) for k in range(3)],
        dim="timestep_min",
    ).load()
    final = tmp_path / "TRITON_only_tseries.nc"

    publish_chapters_to_netcdf(chapters, final, scenario_dir=tmp_path, compression_level=5)

    assert final.is_file(), f"{final.name} is not a regular file"
    assert unified_flag_for(final).exists()
    assert not chapters.exists(), "chapters must be deleted once the flag certifies the file"
    with _open_like_production(final) as got:
        got = got.load()
    np.testing.assert_array_equal(got["timestep_min"].values, expected["timestep_min"].values)
    for name in _VARS:
        np.testing.assert_array_equal(got[name].values, expected[name].values)


def test_nc_tail_leaves_no_file_no_flag_and_every_chapter_when_the_write_fails(tmp_path, monkeypatch):
    """Property: when the NetCDF write raises part-way, the final ``.nc`` path holds no file,
    no completion flag is written, and every flagged chapter survives, so the next invocation
    can publish again from the chapters.

    Class: NEW CAPABILITY. The plausible wrong implementation writes the final path directly
    (``write_netcdf(ds, fname_out)``) and then the flag; a write that dies part-way then
    leaves a truncated file at the published path.

    Input class: any write failure after bytes have reached the target, simulated by a
    writer that writes a partial header to the path it is handed and then raises.

    Why it is required: raw TRITON outputs are cleared chapter by chapter, so after the
    chapters are deleted nothing can regenerate the data; and the gridded writer returns
    early whenever the flag exists, so a flag written before a complete file would make a
    truncated file permanent.

    Kills it: writing the final path directly; writing the flag before ``os.replace``;
    deleting the chapters before the flag; not removing the temporary on failure.

    A second correct implementation under which it still passes: write the NetCDF file into
    a temporary directory beside the chapters and move it into place, deleting that directory
    on failure -- the test asserts the three post-failure states, not the temporary's name.
    """
    chapters = tmp_path / "TRITONSWMM_TRITON_tseries.nc.chapters"
    _write_chapters(chapters, [3, 3])
    before = {k: p.name for k, p in completed_chapters(chapters).items()}
    final = tmp_path / "TRITONSWMM_TRITON_tseries.nc"

    def _dies_part_way(ds, path, *args, **kwargs):
        with open(path, "wb") as fh:
            fh.write(b"\x89HDF\r\n\x1a\n")
        raise RuntimeError("simulated failure part-way through the NetCDF write")

    monkeypatch.setattr(process_simulation, "write_netcdf", _dies_part_way)
    with pytest.raises(RuntimeError, match="part-way"):
        publish_chapters_to_netcdf(chapters, final, scenario_dir=tmp_path, compression_level=5)

    assert not final.exists(), "a failed write must not leave a file at the published path"
    assert not unified_flag_for(final).exists(), "a failed write must not be flagged complete"
    assert {k: p.name for k, p in completed_chapters(chapters).items()} == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


# The synthetic end-to-end invocation, matching tests/test_synth_timeseries_consolidation.py.
_WORKFLOW_KWARGS = dict(
    mode="local",
    process_system_level_inputs=True,
    overwrite_system_inputs=True,
    compile_TRITON_SWMM=True,
    recompile_if_already_done_successfully=False,
    prepare_scenarios=True,
    overwrite_scenario_if_already_set_up=True,
    rerun_swmm_hydro_if_outputs_exist=True,
    process_timeseries=True,
    which="both",
    override_clear_raw="all",
    compression_level=5,
    pickup_where_leftoff=False,
    verbose=True,
)


@pytest.mark.usefixtures("triton_only_cpu_compiled")
def test_nc_synthetic_run_publishes_netcdf_from_both_gridded_exports(tmp_path, monkeypatch):
    """Property: a synthetic ``nc`` run of BOTH the TRITON-only and the TRITON-SWMM model
    publishes each gridded timeseries as a NetCDF FILE the production reader opens, and the
    summary step completes for both.

    RIVANNA ONLY: this compiles and runs the solvers. Run it by node id on Rivanna scratch,
    never on a laptop.

    Class: NEW CAPABILITY. The plausible wrong implementation adds the ``nc`` arm to only
    one of the two gridded exports, ``_export_TRITONSWMM_TRITON_outputs`` or
    ``_export_TRITON_only_outputs``; the other model's ``.nc`` path is then a zarr directory
    and its summary step raises. The solver-free tests above call the publisher directly and
    cannot see which exports reach it.

    Input class: any analysis enabling both gridded models under ``nc``.

    Why it is required: each export resolves its own path and consolidates its own metadata,
    so the publisher being correct says nothing about whether each export reaches it.

    Kills it: either export calling ``merge_chapters_to_unified`` under ``nc``, or calling
    ``zarr.consolidate_metadata`` on the NetCDF path.

    A second correct implementation under which it still passes: branch on the output type
    in each export instead of inside the shared chapter writer.
    """
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_nc_second_arm",
        toggle_tritonswmm_model=True,
        toggle_triton_model=True,
        toggle_swmm_model=False,
        start_from_scratch=True,
        additional_analysis_configs={"target_processed_output_type": "nc"},
    )
    a = case.analysis
    assert a.cfg_analysis.target_processed_output_type == "nc"
    result = a.submit_workflow(**_WORKFLOW_KWARGS)
    assert result["success"], f"Workflow failed: {result.get('message', '')}"

    processed_dirs = sorted(a.analysis_paths.simulation_directory.glob("*/processed"))
    assert processed_dirs, "no scenario processed directories were written"
    for processed in processed_dirs:
        for tseries, summary in (
            ("TRITONSWMM_TRITON_tseries.nc", "TRITONSWMM_TRITON_summary.nc"),
            ("TRITON_only_tseries.nc", "TRITON_only_summary.nc"),
        ):
            path = processed / tseries
            assert path.is_file(), f"{path} is not a NetCDF file"
            with _open_like_production(path) as ds:
                assert ds.sizes["timestep_min"] > 0
            assert (processed / summary).is_file(), f"{processed / summary} was not written"

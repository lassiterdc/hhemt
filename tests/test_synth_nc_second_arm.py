"""The gridded TRITON timeseries under ``target_processed_output_type: nc``.

THE PROBLEM. ``utils.merge_chapters_to_unified`` publishes a chapter set as a zarr STORE, which is a directory,
and at a ``.nc``-named path the production reader opens that directory with the ``h5netcdf`` engine and raises
``IsADirectoryError``. ``process_simulation.publish_chapters_to_netcdf`` is the ``nc`` arm that publishes one
NetCDF FILE there instead. These tests pin the three things that arm must
do: write a file the production reader opens, leave nothing behind but recoverable chapters
when the write fails, and be reached from BOTH gridded exports.

The first two tests are solver-free: they write flagged chapter stores directly with xarray and run
the publisher on them. The third starts from the RAW layer of the committed multi-sim golden
(one committed zip holding a finished
analysis tree and the files hhemt needs to construct an analysis object from it, described in section 1 of
``tests/goldens/SPEC.md``; the raw layer is the binary TRITON
outputs under ``out_triton/`` and ``out_tritonswmm/``), switches the persisted config to ``nc``, discards the golden's
processed layer, and runs the processing path in-process. No solver is compiled or executed and no Snakemake process
starts.

A reprocess of a golden that runs through a workflow would need a Snakefile dry run, the interpreter directory on
``PATH`` and a child-interpreter identity shim; this file sets up none of them. The third test calls the
post-processing object directly, so it has no Snakefile, no ``python`` placed on ``PATH``, no child-interpreter
identity shim and no build-gate declaration.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import hhemt.process_simulation as process_simulation
from hhemt.process_simulation import TRITONSWMM_sim_post_processing, publish_chapters_to_netcdf
from hhemt.utils import chapter_flag_for, chapter_store_for, completed_chapters, unified_flag_for
from tests.fixtures.golden_trees import construct_golden_analysis

_VARS = ("H", "QX")


def _write_chapters(root, lengths, ny: int = 5, nx: int = 6) -> None:
    """One flagged chapter store per entry of ``lengths``, with distinct values per timestep.

    The values encode the timestep, row and column, so a concatenation in the wrong order or
    a dropped chapter changes the data and not only the shape.

    A chapter store counts as present only if its sibling flag file exists
    (``utils.completed_chapters`` lists the flagged chapters whose store still exists).
    What a chapter flag attests is stated in the docstring of ``utils.verify_and_flag_chapter``.
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

    The plausible wrong implementation is the tail that calls
    ``utils.merge_chapters_to_unified`` on the ``.nc`` path, which publishes a zarr
    DIRECTORY there; the production reader then raises ``IsADirectoryError`` from h5netcdf.

    Input class: any contiguous flagged chapter set, here three chapters of unequal length
    (4, 3, 2 timesteps) so an off-by-one at a chapter seam changes the timestep coordinate.

    Why it is required: the summary step reads this artifact through ``_open`` with the
    ``h5netcdf`` engine under ``nc``, so an artifact that is not a NetCDF file fails every
    ``nc`` analysis at the summary.

    Kills it: dispatching ``nc`` to ``merge_chapters_to_unified``; concatenating the chapters
    in any order other than ascending chapter index (``completed_chapters`` maps chapter index
    to store path); dropping the last chapter.

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

    The plausible wrong implementation writes the final path directly
    (``write_netcdf(ds, fname_out)``) and then the flag; a write that dies part-way then
    leaves a truncated file at the published path.

    Input class: any write failure after bytes have reached the target, simulated by a
    writer that writes a partial header to the path it is handed and then raises.

    Why it is required: raw TRITON outputs are cleared chapter by chapter, so after the
    chapters are deleted nothing can regenerate the data; and the gridded writer
    (``TRITONSWMM_sim_post_processing._streaming_chunked_zarr_write`` in ``process_simulation.py``)
    returns early whenever the flag exists, so a flag written before a complete file would make a
    truncated file permanent.

    Kills it: writing the final path directly; writing the flag before ``os.replace``;
    deleting the chapters before the flag; not removing a ``.tmp``-suffixed temporary on failure.

    A second correct implementation under which it still passes: write the NetCDF file into
    a temporary directory beside the chapters and move it into place, deleting that directory
    on failure -- the test asserts the three post-failure states, plus that no entry directly
    under ``tmp_path`` ends in ``.tmp``. That last assertion constrains a temporary only when
    it is a ``.tmp``-suffixed entry directly under ``tmp_path``: a leftover named ``.part``, or
    placed inside another directory, is not seen and passes.
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


def _prepare_golden_for_nc_processing(golden_extractor, event_iloc: int = 0):
    """Extract the multi-sim golden with ``nc`` as the persisted output type and no processed layer for one event.

    The raw layer is left exactly as the golden carries it. Returns the constructed analysis and
    the path of the event's ``processed`` directory, which no longer exists."""
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    text = cfg.read_text()
    assert 'target_processed_output_type: "zarr"' in text
    cfg.write_text(text.replace('target_processed_output_type: "zarr"', 'target_processed_output_type: "nc"'))
    sim_dir = system_dir / "synth_multi_sim" / "sims" / f"event_index.{event_iloc}"
    shutil.rmtree(sim_dir / "processed")
    return construct_golden_analysis(system_dir), sim_dir / "processed"


def test_nc_synthetic_run_publishes_netcdf_from_both_gridded_exports(golden_extractor):
    """Property: processing the raw TRITON outputs of BOTH the TRITON-only and the TRITON-SWMM model under ``nc``
    publishes each gridded timeseries as a NetCDF FILE the production reader opens, and the summary step completes
    for both.

    The plausible wrong implementation adds the ``nc`` arm to only one of the two gridded
    exports, ``_export_TRITONSWMM_TRITON_outputs`` or ``_export_TRITON_only_outputs``; the other model's ``.nc`` path
    is then a zarr directory and its summary step raises. The solver-free tests above call the publisher directly and
    cannot see which exports reach it.

    Input class: any analysis enabling both gridded models under ``nc``; here one event of the golden, whose raw layer
    is the binary rasters of a reduced-size run.

    Why it is required: each export resolves its own path and consolidates its own metadata, so the publisher being
    correct says nothing about whether each export reaches it.

    Kills it: either export calling ``merge_chapters_to_unified`` under ``nc``, or calling
    ``zarr.consolidate_metadata`` on the NetCDF path.

    A second correct implementation under which it still passes: branch on the output type in each export instead of
    inside ``_streaming_chunked_zarr_write``, the writer both exports call (it calls
    ``publish_chapters_to_netcdf`` under ``nc``).
    """
    from hhemt.scenario import TRITONSWMM_scenario

    a, processed = _prepare_golden_for_nc_processing(golden_extractor)
    assert a.cfg_analysis.target_processed_output_type == "nc"
    assert not processed.exists(), "precondition: the processed layer is gone, so every output below was just written"

    scenario = TRITONSWMM_scenario(0, a)
    for model_type in ("tritonswmm", "triton"):
        proc = TRITONSWMM_sim_post_processing(scenario.run, model_log=scenario.get_log(model_type))
        proc.write_timeseries_outputs(
            which="TRITON", model_type=model_type, override_clear_raw="none", compression_level=5
        )
        proc.write_summary_outputs(which="TRITON", model_type=model_type, compression_level=5)

    for tseries, summary in (
        ("TRITONSWMM_TRITON_tseries.nc", "TRITONSWMM_TRITON_summary.nc"),
        ("TRITON_only_tseries.nc", "TRITON_only_summary.nc"),
    ):
        path = processed / tseries
        assert path.is_file(), f"{path} is not a NetCDF file"
        with _open_like_production(path) as ds:
            assert ds.sizes["timestep_min"] > 0
        assert (processed / summary).is_file(), f"{processed / summary} was not written"

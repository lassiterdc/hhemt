"""Consolidation refuses a summary store that exists but never completed.

Compile-INDEPENDENT (no GPU, no solver, no real simulation pipeline): uses the
compile-free ``norfolk_sensitivity_analysis`` object fixture and writes its own
tiny zarr stores. The discriminating arm is
``test_incomplete_summary_is_refused``: PRE-FIX ``_retrieve_combined_output``
consults no completion signal at all, so it opens the store and returns; POST-FIX
it raises. The assertion is on BEHAVIOUR (raise vs return), not on a message.
"""

import numpy as np
import pytest
import xarray as xr

from hhemt.log import ProcessingEntry
from hhemt.scenario import TRITONSWMM_scenario


def _model_type_for(mode: str) -> str:
    if mode.startswith("tritonswmm"):
        return "tritonswmm"
    if mode.startswith("triton"):
        return "triton"
    return "swmm"


def _seed_one_sub(sub, mode):
    """Write a minimal summary store + a successful log record for every scenario."""
    attr = sub.process._MODE_CONFIG[mode][0]
    model_type = _model_type_for(mode)
    seeded = []
    for event_iloc in sub.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, sub)
        f_out = getattr(scen.scen_paths, attr)
        if f_out is None:
            pytest.skip(f"mode {mode!r} not enabled for this fixture")
        f_out.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"v": (("event_iloc",), np.array([float(event_iloc)]))},
            coords={"event_iloc": [event_iloc]},
        ).to_zarr(f_out, mode="w", consolidated=False)
        log = scen.get_log(model_type)
        log.processing_log.update(ProcessingEntry(filepath=f_out, size_MiB=0.1, time_elapsed_s=0.1, success=True))
        seeded.append((scen, f_out, model_type))
    return seeded


def _first_mode(sub):
    for mode, cfg in sub.process._MODE_CONFIG.items():
        if "performance" in mode:
            continue
        attr = cfg[0]
        scen = TRITONSWMM_scenario(sub.df_sims.index[0], sub)
        if getattr(scen.scen_paths, attr, None) is not None:
            return mode
    pytest.skip("no non-performance summary mode enabled for this fixture")


def test_complete_summary_is_accepted(norfolk_sensitivity_analysis):
    """Guard arm: a seeded, record-carrying summary set still consolidates."""
    sub = next(iter(norfolk_sensitivity_analysis.sensitivity.members.values()))
    mode = _first_mode(sub)
    _seed_one_sub(sub, mode)
    ds = sub.process._retrieve_combined_output(mode)
    assert ds is not None


def test_incomplete_summary_is_refused(norfolk_sensitivity_analysis):
    """DISCRIMINATING arm: store present, completion record absent -> refuse.

    Deleting the record is the kill analogue: ``add_sim_processing_entry`` runs
    only after ``write_zarr`` RETURNS, so a killed write leaves exactly this state.
    Pre-fix this returns a dataset; post-fix it raises.
    """
    sub = next(iter(norfolk_sensitivity_analysis.sensitivity.members.values()))
    mode = _first_mode(sub)
    seeded = _seed_one_sub(sub, mode)
    scen, f_out, model_type = seeded[0]
    log = scen.get_log(model_type)
    log.processing_log.outputs.pop(f_out.name)
    # PERSIST it: _retrieve_combined_output builds a FRESH scenario, which reloads
    # the log from disk. `dict.pop` mutates only memory -- `Processing.update` is
    # what normally calls `_log.write()` -- so without this the on-disk record
    # survives and the fresh reader still sees success.
    log.write()

    with pytest.raises(FileNotFoundError, match="present but incomplete"):
        sub.process._retrieve_combined_output(mode)

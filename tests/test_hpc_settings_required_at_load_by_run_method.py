"""Three HPC settings are required at config load under the run methods that read them.

`hpc_time_min_per_sim` (the simulation rules' walltime) is required for
`multi_sim_run_method='batch_job'`; `hpc_ensemble_partition` and
`hpc_setup_and_analysis_processing_partition` are required for both HPC methods,
`batch_job` and `1_job_many_srun_tasks`, because the Snakefile generator raises
"hpc partition must be set when generating SLURM resources" for any method other
than `local` when either is unset. Each requirement is DECLARED on the field
(`field_meta(required_when=...)`), which is what the generic loader enforces and what
the Reproduction Guide's "Required" cell reads. A requirement enforced only by a
hand-written check, or only at Snakefile generation, renders as "Optional" and fails
late, after the operator has submitted.

No test here constructs a `TRITONSWMM_analysis`; each validates an analysis-config
dict whose input files are empty placeholders in a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hhemt.config.analysis import analysis_config
from hhemt.config.base import declared, when

_TIME = "hpc_time_min_per_sim"
_ENSEMBLE = "hpc_ensemble_partition"
_SETUP = "hpc_setup_and_analysis_processing_partition"
_HPC_METHODS = ("batch_job", "1_job_many_srun_tasks")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    return path


def _config(tmp_path: Path, method: str) -> dict:
    """A complete analysis config for `method`, with all three settings present."""
    return {
        "analysis_id": "analysis_01",
        "hhemt_sha": "0" * 40,
        "weather_event_indices": ["event_id"],
        "weather_timeseries": str(_touch(tmp_path / "inputs" / "weather.nc")),
        "weather_time_series_timestep_dimension_name": "timestep",
        "weather_time_series_spatial_mean_rainfall_datavar": "RG_synth",
        "rainfall_units": "mm/hr",
        "run_mode": "serial",
        "multi_sim_run_method": method,
        "toggle_sensitivity_analysis": False,
        "toggle_storm_tide_boundary": False,
        "weather_events_to_simulate": str(_touch(tmp_path / "inputs" / "events.csv")),
        "manhole_diameter": 1.0,
        "manhole_loss_coefficient": 0.1,
        "hydraulic_timestep_s": 1.0,
        "TRITON_reporting_timestep_s": 60,
        "open_boundaries": 1,
        "report": {},
        "clear_raw": "none",
        "force_rerun": "none",
        "hpc_total_job_duration_min": 120,
        _TIME: 60,
        _ENSEMBLE: "gpu",
        _SETUP: "standard",
    }


_REQUIRED_CASES = [
    ("batch_job", _TIME),
    ("batch_job", _ENSEMBLE),
    ("batch_job", _SETUP),
    ("1_job_many_srun_tasks", _ENSEMBLE),
    ("1_job_many_srun_tasks", _SETUP),
]


@pytest.mark.parametrize(("method", "field"), _REQUIRED_CASES)
def test_missing_setting_is_rejected_at_load_by_its_declaration(tmp_path, method, field) -> None:
    """Under the method that reads it, an unset setting is rejected by the declared requirement.

    Property: the config with `field` set to None fails to load, with the generic
    `required_when` message ("{field} is required when ..."), and the complete config loads.
    Class: every (run method, setting) pair the ruling names, one parametrized case each.
    Why: the declaration is the one enforcement site the rendered "Required" cell reads;
    a requirement raised from anywhere else renders "Optional", and a partition missing
    until Snakefile generation fails after submission rather than at load.
    What kills it: removing or narrowing the field's `required_when`, or enforcing the
    requirement only in a hand-written check with its own message.
    Second correct implementation: a declaration listing the two HPC methods in the
    other order still passes.
    """
    cfg = _config(tmp_path, method)
    assert analysis_config.model_validate(cfg)

    cfg[field] = None
    with pytest.raises(ValidationError, match=rf"{field} is required when"):
        analysis_config.model_validate(cfg)


def test_each_setting_declares_its_condition_on_the_field() -> None:
    """The three fields carry exactly the ruled `required_when` declarations.

    Property: `hpc_time_min_per_sim` declares batch_job; both partition selectors declare
    batch_job and 1_job_many_srun_tasks.
    Class: the three fields of ruling D-52.
    Why: the declaration, not the error text, is what the report and the loader share.
    What kills it: a declaration naming a different trigger or a different method set.
    Second correct implementation: the same methods in another order (compared as sets).
    """
    expected = {
        _TIME: {"batch_job"},
        _ENSEMBLE: set(_HPC_METHODS),
        _SETUP: set(_HPC_METHODS),
    }
    for field, methods in expected.items():
        clauses = declared(analysis_config.model_fields[field], "required_when")
        assert clauses, field
        assert [c["field"] for c in clauses] == ["multi_sim_run_method"], (field, clauses)
        assert set(clauses[0]["in"]) == methods, (field, clauses)
    assert when("multi_sim_run_method", "batch_job")["in"] == ["batch_job"]


def test_all_three_unset_load_under_local(tmp_path) -> None:
    """Control: `local` requires none of the three.

    Property: with all three settings None, a `local` config loads.
    Class: the one method that reads none of them.
    Why: the ruled condition is "every method other than local"; a requirement leaking
    into local would break every desktop run.
    What kills it: a declaration that also names `local`, or an unconditional requirement.
    Second correct implementation: any condition set excluding `local`.
    """
    cfg = _config(tmp_path, "local")
    for field in (_TIME, _ENSEMBLE, _SETUP):
        cfg[field] = None
    assert analysis_config.model_validate(cfg)


def test_walltime_unset_loads_under_1_job_many_srun_tasks(tmp_path) -> None:
    """Control: `hpc_time_min_per_sim` is not required for `1_job_many_srun_tasks`.

    Property: a `1_job_many_srun_tasks` config with `hpc_time_min_per_sim` None loads.
    Class: the method under which the walltime is not required.
    Why: the ruling requires it for batch_job only.
    What kills it: a walltime declaration that also names 1_job_many_srun_tasks.
    Second correct implementation: any declaration naming batch_job alone.
    """
    cfg = _config(tmp_path, "1_job_many_srun_tasks")
    cfg[_TIME] = None
    assert analysis_config.model_validate(cfg)


@pytest.mark.parametrize("bad", [0, -5])
def test_batch_job_walltime_value_bound_still_holds(tmp_path, bad) -> None:
    """Control: the kept `>= 1` check still rejects a present walltime below 1.

    Property: under batch_job, `hpc_time_min_per_sim` of 0 or -5 fails to load.
    Class: present integer walltimes below 1, which the declaration grammar cannot express.
    Why: only the presence check moved; the value bound stays in the hand-written validator.
    What kills it: removing the `>= 1` check along with the presence check.
    Second correct implementation: a pydantic `ge=1` constraint applied under batch_job.
    The kept NaN check is covered by the next test, which can observe it: the check runs
    in a `mode="before"` validator, so it sees the raw NaN before pydantic coerces the field.
    """
    cfg = _config(tmp_path, "batch_job")
    cfg[_TIME] = bad
    with pytest.raises(ValidationError):
        analysis_config.model_validate(cfg)


def test_batch_job_walltime_nan_is_rejected_by_the_kept_check(tmp_path) -> None:
    """Control: the kept NaN check still rejects a NaN walltime, with its own message.

    Property: under batch_job, `hpc_time_min_per_sim` of NaN fails to load with the kept
    check's "(NaN detected)" message.
    Class: a present NaN walltime, which the declaration grammar cannot express.
    Why: the check lives in `check_consistency`, a `mode="before"` validator that sees the
    raw value; pydantic would also reject NaN for an `int` field, but with its own message,
    so only the message shows the kept check is the one that fired.
    What kills it: removing the NaN branch along with the presence check.
    Second correct implementation: a NaN check with a reworded message that still names
    NaN, matched on the word "NaN".
    """
    cfg = _config(tmp_path, "batch_job")
    cfg[_TIME] = float("nan")
    with pytest.raises(ValidationError, match="NaN detected"):
        analysis_config.model_validate(cfg)

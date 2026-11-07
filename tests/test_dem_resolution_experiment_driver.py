"""The DEM-resolution experiment's route: the `dem` subcommand of the synthetic-experiment driver and the CLI dry run.

VENUE: HPC-free and solver-free, so the laptop runs this file. No test here compiles or executes a
solver, directly or through a fixture. `retrieve_synth_TRITON_SWMM_test_case` is replaced by a
recorder, so no model is materialized and no TRITON source is cloned, and `_emit_bundle` is replaced
by a recorder, so no analysis is constructed. The CLI arm runs `hhemt synth-experiment --dry-run`,
which validates the configuration and builds the matrix in memory. The file name does not start with
`test_synth_`, and `pytest tests/test_dem_resolution_experiment_driver.py -m compile_tier
--collect-only -q` selects no test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import scripts.experiments.synth_compute_config as driver
from hhemt.cli import app
from hhemt.config.synthetic_experiment import synthetic_experiment_config
from hhemt.exceptions import ConfigurationError
from hhemt.synthetic_experiment import (
    _CLEAN_WALLTIME_MIN,
    _configs,
    _rows,
    build_experiment_matrix,
    dem_resolution_matrix_rows,
)

#: The four renderer keys of the DEM-resolution EDA family, written out so the expectation does not
#: derive from the driver constant under test.
_DEM_KINDS = {
    "dem_resolution_cost_error",
    "dem_resolution_diff_maps",
    "dem_resolution_error_ecdf",
    "dem_resolution_coupling_table",
}
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _write_hpc_yaml(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            hpc_name: test_synth_cluster
            partitions:
              gpu-a6000:
                max_runtime: 4320
                gpus_per_node: 8
                cpus_per_node: 40
                max_gpu: 8
                gpu_hardware: a6000
                gpu_compilation_backend: CUDA
              gpu-a100-80:
                max_runtime: 4320
                gpus_per_node: 8
                cpus_per_node: 40
                max_gpu: 8
                gpu_hardware: a100
                gpu_compilation_backend: CUDA
              standard:
                max_runtime: 4320
                cpus_per_node: 40
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path: Path, ladder: list[float], **extra) -> Path:
    """A `synthetic_experiment_config` YAML at the shipped grid defaults carrying `ladder`."""
    hpc = _write_hpc_yaml(tmp_path / "hpc.yaml")
    body = {
        "hpc_system_config_yaml": str(hpc),
        "ensemble_partition": "gpu-a6000",
        "setup_partition": "standard",
        "dem_resolution_ladder": ladder,
        **extra,
    }
    path = tmp_path / "synth_experiment.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _load(path: Path) -> synthetic_experiment_config:
    return synthetic_experiment_config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    """Replace the case builder and the bundle step with recorders; return what they saw."""
    seen: dict = {"build": None, "bundle_calls": 0}

    def fake_build(**kwargs):
        seen["build"] = kwargs
        return SimpleNamespace(
            analysis=SimpleNamespace(),
            system=SimpleNamespace(cfg_system=SimpleNamespace(system_directory="x")),
        )

    def fake_bundle(case):
        seen["bundle_calls"] += 1
        return Path("bundle_path")

    monkeypatch.setattr(driver, "retrieve_synth_TRITON_SWMM_test_case", fake_build)
    monkeypatch.setattr(driver, "_emit_bundle", fake_bundle)
    monkeypatch.setattr(driver, "_GENERATED", tmp_path / "generated")
    return seen


def _run_dem(monkeypatch, config: Path, *extra: str) -> None:
    argv = [
        "synth_compute_config",
        "dem",
        "--config",
        str(config),
        "--system-directory",
        "/path/to/dem",
        "--tritonswmm-sha",
        _SHA,
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    driver._cli()


def test_dem_subcommand_writes_the_matrix_and_selects_the_four_kinds(monkeypatch, tmp_path, recorder):
    """`dem` writes `build_experiment_matrix(cfg)` as the CSV and persists the four DEM kinds.

    Class: NEW CAPABILITY. Plausible wrong implementation: the subcommand writing the
    compute-configuration clean matrix, or leaving the compute-configuration kinds in
    `eda.enabled_plots`; red because the CSV then differs from the DEM frame and the kind set
    differs from the four. Reachability: `_cli` reaches `dem_case` and `_build_case` with the
    recorder as the case builder; artifacts: the CSV path in `sensitivity_analysis` and the
    recorded analysis configuration. Both-states property: the default `_build_case` call, with
    no DEM arguments, records the five compute-configuration kinds (second test below), so the
    two states differ in the recorded kind set. Second correct implementation: writing the CSV
    with `DataFrame.to_csv` from a frame built outside `dem_case` passes, because the assertion
    compares CSV content.
    """
    config = _write_config(tmp_path, [3.5, 7.0, 14.0])
    _run_dem(monkeypatch, config)

    kwargs = recorder["build"]
    analysis_cfg = kwargs["additional_analysis_configs"]
    assert set(analysis_cfg["eda"]["enabled_plots"]) == _DEM_KINDS
    assert len(analysis_cfg["eda"]["enabled_plots"]) == len(_DEM_KINDS)
    assert analysis_cfg["report"]["reporting_set"] == "dem_resolution"
    assert analysis_cfg["report"]["sensitivity"]["independent_vars"] == ["system.target_dem_resolution"]

    written = pd.read_csv(analysis_cfg["sensitivity_analysis"])
    expected = build_experiment_matrix(_load(config))
    pd.testing.assert_frame_equal(written, expected, check_dtype=False)

    assert kwargs["additional_system_configs"]["target_dem_resolution"] == 3.5
    assert kwargs["additional_system_configs"]["TRITONSWMM_branch_key"] == _SHA
    assert kwargs["additional_system_configs"]["system_directory"] == "/path/to/dem"
    assert recorder["bundle_calls"] == 0


def test_dem_subcommand_runs_the_eda_step_only_with_the_flag(monkeypatch, tmp_path, recorder):
    """`--eda` runs the bundle step the other arms run, and the bare subcommand does not.

    Class: NEW CAPABILITY. Plausible wrong implementation: ignoring the flag, or running the step
    unconditionally; red on the bare call (one call recorded) or the flagged call (none).
    Invocation isolation: the recorder counter is read after each call, with the second call
    made after the first, so the flagged call's increment is the difference.
    """
    config = _write_config(tmp_path, [3.5, 7.0, 14.0])
    _run_dem(monkeypatch, config)
    assert recorder["bundle_calls"] == 0
    _run_dem(monkeypatch, config, "--eda")
    assert recorder["bundle_calls"] == 1


def test_the_default_build_case_call_keeps_the_compute_configuration_values(monkeypatch, tmp_path, recorder):
    """A `_build_case` call that names no DEM argument persists the compute-configuration values.

    Class: PRESERVATION, green on the chunk base and on the changed tree by design. The recorded
    kinds, reporting set and independent variables are the values
    the clean, resume, cross-solver and smoke arms persisted before the DEM arguments existed.
    Plausible wrong implementation: changing a default to the DEM values; red because the
    recorded kinds then name the DEM family.
    """
    csv = tmp_path / "m.csv"
    csv.write_text("member_id\n0\n")
    driver._build_case(
        analysis_name="a", sensitivity_csv=csv, start_from_scratch=False, resume=False, system_directory=None
    )
    analysis_cfg = recorder["build"]["additional_analysis_configs"]
    assert analysis_cfg["eda"]["enabled_plots"] == [
        "config_diff_maps",
        "raw_byte_identity",
        "b4b_clean_vs_resume",
        "eda_rank_sensitivity",
        "eda_cross_hardware_magnitude",
    ]
    assert analysis_cfg["report"]["reporting_set"] == "compute_config_byte_identity"
    assert analysis_cfg["report"]["sensitivity"]["independent_vars"] == ["n_devices"]


@pytest.mark.parametrize(
    ("overrides", "ladder", "field"),
    [
        ({}, [], "dem_resolution_ladder"),
        ({"cell_size_m": 7.0, "n_cols": 32, "n_rows": 60}, [3.5, 7.0], "dem_resolution_ladder"),
        ({"n_cols": 32, "n_rows": 60}, [3.5, 7.0], "n_cols"),
        ({"model_arms": ["tritonswmm", "triton"]}, [3.5, 7.0], "model_arms"),
    ],
    ids=["empty-ladder", "finest-rung-off-the-grid", "domain-off-the-driver", "two-arms"],
)
def test_dem_case_refuses_a_configuration_whose_grid_the_case_would_not_use(
    tmp_path, recorder, overrides, ladder, field
):
    """`dem_case` names the field to change when the case would not use the grid of the configuration.

    Class: NEW CAPABILITY. Plausible wrong implementation: building the case at the first rung
    whatever the grid; red because no ConfigurationError is raised. Each parameter is a
    differently positioned violating input (empty ladder, finest rung against `cell_size_m`,
    extent against the driver domain, two arms); the satisfying input is the shipped grid, which the
    first test builds.
    """
    cfg = _load(_write_config(tmp_path, ladder, **overrides))
    with pytest.raises(ConfigurationError) as excinfo:
        driver.dem_case(cfg, system_directory="/path/to/dem", tritonswmm_branch_key=_SHA)
    assert excinfo.value.field == field
    assert recorder["build"] is None


def test_dem_subcommand_fails_closed_without_a_sha():
    """`dem` carries no default solver version, as `clean` and `resume` carry none.

    Class: NEW CAPABILITY. Plausible wrong implementation: a `--tritonswmm-sha` default; red
    because the parser then accepts the invocation and the return code is not 2.
    """
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.experiments.synth_compute_config",
            "dem",
            "--config",
            "c.yaml",
            "--system-directory",
            "/x",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, r.stderr
    assert "--tritonswmm-sha" in r.stderr


def _dry_run_matrix_rows(config: Path) -> int:
    result = CliRunner().invoke(app, ["synth-experiment", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    marker = " matrix rows;"
    assert marker in flat, flat
    return int(flat.split(marker)[0].rsplit("(", 1)[1])


def test_synth_experiment_dry_run_prints_the_row_count_of_the_set_experiment(tmp_path):
    """`hhemt synth-experiment --dry-run` prints the DEM row count for a ladder config.

    Class: REGRESSION. Red on the pre-change tree, where the printed count is the
    compute-configuration count for a configuration that sets a ladder. Plausible wrong
    implementation: counting the DEM rows only for the CSV and not for the printed matrix;
    red because the printed count then equals the compute-configuration count. The satisfying
    input in a different position is the empty-ladder configuration, which prints the
    compute-configuration count.
    """
    ladder_cfg = _load(_write_config(tmp_path, [3.5, 7.0, 14.0]))
    n_dem = len(dem_resolution_matrix_rows(ladder_cfg, walltime_min=_CLEAN_WALLTIME_MIN))
    n_compute = len(_rows(_configs(tuple(ladder_cfg.rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN))
    assert n_dem != n_compute

    assert _dry_run_matrix_rows(_write_config(tmp_path, [3.5, 7.0, 14.0])) == n_dem

    other = tmp_path / "compute"
    other.mkdir()
    assert _dry_run_matrix_rows(_write_config(other, [])) == n_compute


def test_dem_case_accepts_a_grid_whose_extent_rounds_off_the_driver_domain(tmp_path, recorder):
    """`dem_case` builds a configuration at 0.56 m, where 400 * 0.56 is 224.00000000000003.

    Class: NEW CAPABILITY (a wrong refusal removed). Plausible wrong implementation: comparing the
    extents with `!=`; red because the 0.56 m grid is refused with a ConfigurationError. The
    satisfying input in a different position is the shipped 3.5 m grid, built by the first test, and
    the violating input is the 32 x 60 grid refused by the arm above, so a tolerance wide enough to
    accept a grid that differs by a whole cell turns that arm red.
    """
    cfg = _load(_write_config(tmp_path, [0.56, 1.12], cell_size_m=0.56, n_cols=400, n_rows=750))
    driver.dem_case(cfg, system_directory="/path/to/dem", tritonswmm_branch_key=_SHA)
    assert recorder["build"] is not None
    assert recorder["build"]["additional_system_configs"]["target_dem_resolution"] == 0.56


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"setup_partition": "gpu-a6000"}, "setup_partition"),
        ({"multi_sim_run_method": "local"}, "multi_sim_run_method"),
    ],
    ids=["setup-partition", "run-method"],
)
def test_dem_case_refuses_a_field_the_case_builder_fixes(tmp_path, recorder, overrides, field):
    """`dem_case` names the field when the configuration sets a value that `_build_case` would drop.

    Class: NEW CAPABILITY. Plausible wrong implementation: accepting the field and persisting the
    fixed value, so the configuration reads as honoured; red because no ConfigurationError is
    raised. Control in a different position: the values `_build_case` fixes (`standard` and
    `batch_job`), written explicitly, build and reach the recorder.
    """
    cfg = _load(_write_config(tmp_path, [3.5, 7.0, 14.0], **overrides))
    with pytest.raises(ConfigurationError) as excinfo:
        driver.dem_case(cfg, system_directory="/path/to/dem", tritonswmm_branch_key=_SHA)
    assert excinfo.value.field == field
    assert recorder["build"] is None

    accepted = _load(
        _write_config(tmp_path, [3.5, 7.0, 14.0], setup_partition="standard", multi_sim_run_method="batch_job")
    )
    driver.dem_case(accepted, system_directory="/path/to/dem", tritonswmm_branch_key=_SHA)
    persisted = recorder["build"]["additional_analysis_configs"]
    assert persisted["hpc_setup_and_analysis_processing_partition"] == "standard"
    assert persisted["multi_sim_run_method"] == "batch_job"


def _dry_run_line(config: Path) -> str:
    result = CliRunner().invoke(app, ["synth-experiment", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    return flat.split("validated", 1)[1].split("--dry-run", 1)[0]


def test_synth_experiment_dry_run_prints_the_rungs_for_a_ladder_and_the_ranks_otherwise(tmp_path):
    """The dry-run line names the swept axis: `ladder=` for a set ladder, `ranks=` for an empty one.

    Class: NEW CAPABILITY. Plausible wrong implementation: printing `ranks=` for a ladder
    configuration, as the line did, where `rank_sweep` has no role in the DEM rows; red because the
    line carries `ranks=`. The empty-ladder configuration is the differently positioned input and
    keeps `ranks=`.
    """
    with_ladder = _dry_run_line(_write_config(tmp_path, [3.5, 7.0, 14.0]))
    assert "ladder=(3.5, 7.0, 14.0)" in with_ladder
    assert "ranks=" not in with_ladder

    other = tmp_path / "compute"
    other.mkdir()
    without = _dry_run_line(_write_config(other, []))
    assert "ranks=(2, 4, 8)" in without
    assert "ladder=" not in without

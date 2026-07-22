"""Phase-1 synthetic-experiment framework tests (HPC-free, fast).

Covers the DoD assertions: config cap/coupling validators reject bad tuples, the
experiment matrix is partition-as-axis (hpc.partition, no retired system.gpu_*),
the lifted synthetic_model imports without tests, and src never imports tests/scripts.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from hhemt.config.synthetic_experiment import synthetic_experiment_config
from hhemt.exceptions import ConfigurationError
from hhemt.synthetic_experiment import build_experiment_matrix


def _write_hpc_yaml(path: Path, *, max_gpu: int = 8) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""\
            system_name: test_synth_cluster
            partitions:
              gpu-a6000:
                max_runtime: 4320
                gpus_per_node: 8
                cpus_per_node: 40
                max_gpu: {max_gpu}
                gpu_hardware: a6000
                gpu_compilation_backend: CUDA
              gpu-a100-80:
                max_runtime: 4320
                gpus_per_node: 8
                cpus_per_node: 40
                max_gpu: {max_gpu}
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


def _cfg(hpc_yaml: Path, **overrides) -> synthetic_experiment_config:
    return synthetic_experiment_config(
        hpc_system_config_yaml=hpc_yaml,
        ensemble_partition="gpu-a6000",
        setup_partition="standard",
        **overrides,
    )


def test_default_config_valid(tmp_path):
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"))
    assert cfg.n_rows == 120
    assert cfg.n_cols == 64
    assert cfg.cell_size_m == 3.5
    assert cfg.rank_sweep == (2, 4, 8)


def test_over_cap_tuple_rejected(tmp_path):
    """A partition whose max_gpu is below the 3-GPU matrix rows must reject at load."""
    with pytest.raises(ConfigurationError):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml", max_gpu=2))


def test_coupling_divisibility_rejected(tmp_path):
    """n_rows not divisible by a rank in rank_sweep -> uneven strips -> deadlock guard."""
    with pytest.raises(ConfigurationError):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), n_rows=30, rank_sweep=(8,))


def test_coupling_strip_ownership_rejected(tmp_path):
    """n_rows=16 IS divisible by 8, but the top row-strip is pure wall (no coupling
    node) -> the strip-ownership guard (not the divisibility guard) must reject."""
    with pytest.raises(ConfigurationError):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), n_rows=16, rank_sweep=(8,))


def test_matrix_is_partition_as_axis(tmp_path):
    """build_experiment_matrix emits hpc.partition, NO retired system.gpu_* columns,
    and every column is a valid sensitivity-CSV column (no Unknown-column error)."""
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config
    from hhemt.sensitivity_analysis import (
        _ANALYSIS_COLUMN_PREFIX,
        _HPC_ALIAS_TO_ANALYSIS_FIELD,
        _HPC_COLUMN_PREFIX,
        _SYSTEM_COLUMN_PREFIX,
    )

    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"))
    df = build_experiment_matrix(cfg)

    assert len(df) == 28  # 14 configs x 2 replicates (default rank_sweep reproduces baseline)
    assert "hpc.partition" in df.columns
    assert "system.gpu_hardware" not in df.columns
    assert "system.gpu_compilation_backend" not in df.columns

    # Mirror _retrieve_df_setup's valid-column allowlist -> no Unknown column.
    valid = (
        {"system_config_yaml"}
        | set(analysis_config.model_fields)
        | {_SYSTEM_COLUMN_PREFIX + f for f in system_config.model_fields}
        | {_ANALYSIS_COLUMN_PREFIX + f for f in analysis_config.model_fields}
        | {_HPC_COLUMN_PREFIX + k for k in _HPC_ALIAS_TO_ANALYSIS_FIELD}
    )
    unknown = (set(df.columns) - {"sa_id"}) - valid
    assert not unknown, f"matrix has Unknown sensitivity-CSV columns: {sorted(unknown)}"


def test_rank_sweep_generates_mpi_rows(tmp_path):
    """The mpi rows are generated from rank_sweep at baseline enumerate indices."""
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"))
    df = build_experiment_matrix(cfg)
    mpi = df[df.run_mode == "mpi"]
    assert sorted(mpi.n_mpi_procs.tolist()) == [2, 2, 4, 4, 8, 8]
    # baseline global-enumerate indices 9/10/11 (mpi spliced between openmp and hybrid)
    assert set(mpi.sa_id) == {"mpi_9_r1", "mpi_9_r2", "mpi_10_r1", "mpi_10_r2", "mpi_11_r1", "mpi_11_r2"}


def test_synth_model_importable_without_tests():
    import hhemt.synthetic_model as sm

    assert set(sm.__all__) == {"SyntheticCaseArtifacts", "SyntheticModelParams", "build_synthetic_case"}


def test_src_has_no_tests_or_scripts_import():
    """CR5 import-direction guard (extended to scripts): no src/hhemt module imports
    tests/ or scripts/ (which would break `pip install -e .`)."""
    src = Path(__file__).resolve().parents[1] / "src" / "hhemt"
    r = subprocess.run(
        [
            "grep",
            "-rn",
            "-e",
            "import tests",
            "-e",
            "from tests",
            "-e",
            "import scripts",
            "-e",
            "from scripts",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, f"src/hhemt imports tests/ or scripts/:\n{r.stdout}"


def test_resume_matrix_uses_generous_walltime(tmp_path):
    """Option D: every resume row gets the generous clean walltime (no short
    T/3 sizing) — the kill is deterministic, not walltime-driven."""
    import pandas as pd

    from hhemt.synthetic_experiment import _CLEAN_WALLTIME_MIN, write_resume_matrix_csv

    csv = tmp_path / "resume_matrix.csv"
    write_resume_matrix_csv(csv)
    df = pd.read_csv(csv)
    assert (df["hpc_time_min_per_sim"] == _CLEAN_WALLTIME_MIN).all()
    assert len(df) == 28  # 14 configs x 2 replicates (rank_sweep=(2,4,8))

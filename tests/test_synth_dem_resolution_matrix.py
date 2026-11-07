"""The DEM-resolution matrix builder, the coupling-node precheck and their wiring into the experiment matrix.

VENUE: HPC-free and solver-free, so the laptop runs this file. No test here reaches a compile or a
solver run, directly or through a fixture: the only fixture is ``tmp_path``, and the code under test
is configuration validation, row enumeration and a geometric precheck over the synthetic-model
placement helpers. The file name starts with ``test_synth_``, and
``pytest tests/test_synth_dem_resolution_matrix.py -m compile_tier --collect-only -q`` selects no test.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hhemt.config.synthetic_experiment import synthetic_experiment_config
from hhemt.exceptions import ConfigurationError, ProcessingError
from hhemt.synthetic_experiment import (
    _CLEAN_WALLTIME_MIN,
    _configs,
    _rows,
    assert_coupling_nodes_distinct,
    build_experiment_matrix,
    dem_resolution_matrix_rows,
    experiment_matrix_rows,
)


def _write_hpc_yaml(path: Path, *, max_gpu: int = 8) -> Path:
    """A minimal per-cluster hpc_system_config YAML (the config's required
    hpc_system_config_yaml). Mirrors tests/test_synth_experiment_framework.py."""
    path.write_text(
        textwrap.dedent(
            f"""\
            hpc_name: test_synth_cluster
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


def _cfg(hpc_yaml: Path, ladder: tuple[float, ...], **overrides) -> synthetic_experiment_config:
    """A synthetic_experiment_config carrying ``ladder`` at the shipped grid defaults
    unless an ``overrides`` key (n_cols/n_rows/rank_sweep) says otherwise. Runs every
    config validator, so a bad ``ladder`` raises here."""
    return synthetic_experiment_config(
        hpc_system_config_yaml=hpc_yaml,
        ensemble_partition="gpu-a6000",
        setup_partition="standard",
        dem_resolution_ladder=ladder,
        **overrides,
    )


def test_ladder_rejects_non_divisor(tmp_path):
    """A non-divisor rung inflates area-integrated metrics +7% to +30% (measured), and
    every run completes, so the error is silent. Reject at config load."""
    with pytest.raises(ValueError, match="not an integer divisor"):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 15.0))


def test_ladder_rejects_non_constant_ratio(tmp_path):
    """(3.5, 7.0, 28.0) is divisor-valid at every rung (clears the divisor gate) but
    its ratios are 2 then 4 -- the constant-ratio gate must reject it. (3.5,7,21 would
    fail the divisor gate first and never reach the ratio check.)"""
    with pytest.raises(ValueError, match="CONSTANT refinement ratio"):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 28.0))


def test_ladder_rejects_coarsest_first(tmp_path):
    with pytest.raises(ValueError, match="FINEST FIRST"):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (14.0, 7.0, 3.5))


def test_rows_carry_the_resolution_and_a_fixed_compute_config(tmp_path):
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 14.0))
    rows = dem_resolution_matrix_rows(cfg, replicates=2)
    assert len(rows) == 3 * 2
    assert {r["system.target_dem_resolution"] for r in rows} == {3.5, 7.0, 14.0}
    # The control: ONE compute config across the whole sweep.
    assert {
        (r["analysis.run_mode"], r["analysis.n_mpi_procs"], r["analysis.n_omp_threads"], r["analysis.n_gpus"])
        for r in rows
    } == {("serial", 1, 1, 0)}
    assert len({r["member_id"] for r in rows}) == len(rows)
    assert all(r["member_id"].startswith("dem_") for r in rows)


def test_coupling_precheck_returns_retained_counts_at_a_safe_ladder(tmp_path):
    """The DoD's canonical divisor-valid ladder 3.5/7/14 must PASS on the default
    64x120 grid -- every rung keeps all _N_COUPLING_NODES in-line coupling nodes in
    distinct DEM cells. (Iterating all of ``_nodes`` instead of the in-line set would
    wrongly reject 14 m, where the disconnected dummy_outfall bins into J1's cell.)"""
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 14.0))
    retained = assert_coupling_nodes_distinct(cfg)
    assert set(retained) == {3.5, 7.0, 14.0}
    assert retained == {3.5: 8, 7.0: 8, 14.0: 8}
    # Monotone: coarsening can only merge cells, never split them.
    assert retained[7.0] <= retained[3.5]
    assert retained[14.0] <= retained[7.0]


def test_coupling_precheck_raises_on_a_colliding_rung(tmp_path):
    """The deadlock this converts into a plan-time error is a HANG, not a crash -- the
    worst failure shape on HPC -- so a coarse-enough rung must fail LOUDLY here. A 32x32
    grid packs the 8 in-line nodes tightly enough that the divisor-valid 14 m rung merges
    two of them (J5+J6); the default grid's first in-line collision (56 m) is a non-divisor
    of the y-extent and would be rejected by the ladder validator before the guard runs.
    rank_sweep=(2,) keeps the small grid valid against the coupling invariant."""
    cfg = _cfg(
        _write_hpc_yaml(tmp_path / "hpc.yaml"),
        (3.5, 7.0, 14.0),
        n_cols=32,
        n_rows=32,
        rank_sweep=(2,),
    )
    with pytest.raises(ProcessingError, match="share DEM cell"):
        assert_coupling_nodes_distinct(cfg)


#: The member ids of the ladder (3.5, 7.0, 14.0) at two replicates, written out so the test
#: does not derive its expectation from the formatting code it checks.
_LADDER_MEMBER_IDS = {
    "dem_3p5_r1",
    "dem_3p5_r2",
    "dem_7p0_r1",
    "dem_7p0_r2",
    "dem_14p0_r1",
    "dem_14p0_r2",
}


def test_matrix_of_a_ladder_config_is_one_row_per_rung_and_replicate(tmp_path):
    """`build_experiment_matrix` emits the DEM-resolution rows when a ladder is set.

    Class: REGRESSION. Red on the pre-change tree, where `experiment_matrix_rows` ignores the
    ladder and the matrix holds compute-configuration rows with member ids such as `serial_6_r1`.
    Plausible wrong implementation: appending the DEM rows to the compute-configuration rows;
    red because the member-id set then holds the compute rows as well. Satisfying inputs in
    different positions: a ladder config (DEM rows) and an empty-ladder config (compute rows)
    in one test body, each building its own matrix.
    """
    hpc = _write_hpc_yaml(tmp_path / "hpc.yaml")
    ladder_df = build_experiment_matrix(_cfg(hpc, (3.5, 7.0, 14.0)))
    assert set(ladder_df["member_id"]) == _LADDER_MEMBER_IDS
    assert len(ladder_df) == len(_LADDER_MEMBER_IDS)
    for _, row in ladder_df.iterrows():
        rung = float(row["member_id"].split("_")[1].replace("p", "."))
        assert row["system.target_dem_resolution"] == rung
    assert {int(v) for v in ladder_df["analysis.hpc_time_min_per_sim"]} == {_CLEAN_WALLTIME_MIN}

    compute_df = build_experiment_matrix(_cfg(hpc, ()))
    assert not any(m.startswith("dem_") for m in compute_df["member_id"])
    assert compute_df["system.target_dem_resolution"].isna().all()


def test_the_config_cap_validator_and_the_emitted_matrix_read_one_enumeration(tmp_path):
    """The cap validator iterates `experiment_matrix_rows`, so a ladder config validates the DEM rows.

    Class: NEW CAPABILITY. Plausible wrong implementation: `build_experiment_matrix` switching to
    the DEM rows while `experiment_matrix_rows` keeps the compute rows; red because the ladder
    config then still validates the GPU rows against an hpc file that declares no GPU partition.
    Both-states property: against a file that declares only `standard`, the ladder config
    constructs and the empty-ladder config raises, because the compute rows target the GPU
    partitions. Second correct implementation: the ladder branch in `build_experiment_matrix`
    reading `experiment_matrix_rows(cfg)` passes the same assertions.
    """
    standard_only = tmp_path / "hpc_standard.yaml"
    standard_only.write_text(
        textwrap.dedent(
            """\
            hpc_name: test_standard_only
            partitions:
              standard:
                max_runtime: 4320
                cpus_per_node: 40
            """
        ),
        encoding="utf-8",
    )

    def build(ladder):
        return synthetic_experiment_config(
            hpc_system_config_yaml=standard_only,
            ensemble_partition="standard",
            setup_partition="standard",
            dem_resolution_ladder=ladder,
        )

    cfg = build((3.5, 7.0, 14.0))
    assert experiment_matrix_rows(cfg) == dem_resolution_matrix_rows(cfg, walltime_min=_CLEAN_WALLTIME_MIN)
    assert [r["member_id"] for r in experiment_matrix_rows(cfg)] == list(build_experiment_matrix(cfg)["member_id"])
    with pytest.raises(ConfigurationError, match="undeclared partition"):
        build(())


def test_an_empty_ladder_matrix_is_the_compute_configuration_enumeration(tmp_path):
    """Without a ladder the matrix is the compute-configuration enumeration, row for row.

    Class: PRESERVATION, green on the chunk base and on the changed tree by design. Plausible wrong
    implementation: routing the empty ladder through the DEM branch; red because the rows then carry
    `dem_` member ids.
    """
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), ())
    expected = _rows(_configs(tuple(cfg.rank_sweep)), walltime_min=_CLEAN_WALLTIME_MIN)
    assert experiment_matrix_rows(cfg) == expected


def test_matrix_of_a_colliding_ladder_raises_from_build_experiment_matrix(tmp_path):
    """A ladder with a rung that merges in-line coupling nodes raises from `build_experiment_matrix`.

    Class: NEW CAPABILITY. Plausible wrong implementation: the precheck wired into a path the
    matrix builder does not reach; red because `build_experiment_matrix` then returns a frame.
    The 32x32 grid and `rank_sweep=(2,)` are the colliding-rung inputs of the precheck arm above;
    the satisfying input in a different position is the default grid with the safe ladder, which
    builds. Second correct implementation: calling the precheck from `experiment_matrix_rows`
    instead passes, because the assertion reads the raised error and not the call site.
    """
    hpc = _write_hpc_yaml(tmp_path / "hpc.yaml")
    colliding = _cfg(hpc, (3.5, 7.0, 14.0), n_cols=32, n_rows=32, rank_sweep=(2,))
    with pytest.raises(ProcessingError, match="share DEM cell"):
        build_experiment_matrix(colliding)
    assert len(build_experiment_matrix(_cfg(hpc, (3.5, 7.0, 14.0)))) == len(_LADDER_MEMBER_IDS)

"""Pydantic config model for the synthetic compute-config sensitivity experiment.

``synthetic_experiment_config`` is the single user-facing knob surface for the
synthetic-experiment framework (``hhemt.synthetic_experiment``): the synthetic
model grid, the forcing, the mpi-rank sweep, and the HPC targeting. Two
``model_validator`` guards fail FAST at config-load rather than hours into an
HPC allocation:

* ``_validate_coupling_invariant`` — rejects any ``(n_rows, rank_sweep)`` combo
  that would leave a TRITON row-strip rank owning zero SWMM coupling junctions
  (an ``ENSIFY_COMM_WORLD`` coupling-collective deadlock). Three-part guard:
  (1) ``max(rank_sweep) <= _N_COUPLING_NODES``; (2) ``n_rows % rank == 0`` for
  every rank (so the row-strip boundaries are exactly ``k*n_rows/rank``,
  independent of TRITON's remainder distribution); (3) every equal strip owns
  >= 1 coupling node, computed from the solver-authoritative placement helper
  ``swmm_template._node_matrix_rows`` (which itself asserts the interior span is
  large enough for the nodes).
* ``_validate_caps`` — rejects any experiment-matrix row whose requested
  ``(n_nodes, n_gpus, n_mpi*n_omp, runtime)`` tuple exceeds the resolved
  ``PartitionSpec`` caps of its ``hpc.partition``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from hhemt.config.base import cfgBaseModel
from hhemt.exceptions import ConfigurationError


class synthetic_experiment_config(cfgBaseModel):
    # --- synthetic model knobs (grid + forcing) ---
    cell_size_m: float = Field(
        default=3.5, description="Synthetic DEM cell size (m). Finer -> more cells -> longer per-sim wallclock."
    )
    n_cols: int = Field(default=64, description="Synthetic grid columns (E-W).")
    n_rows: int = Field(
        default=120,
        description=(
            "Synthetic grid rows (N-S). MUST be divisible by every rank in rank_sweep "
            "(coupling-deadlock invariant); default 120 covers ranks {2,4,8}."
        ),
    )
    rainfall_peak_mm_per_hr: float = Field(
        default=100.0,
        description="Constant rainfall forcing (mm/hr) threaded into SyntheticModelParams.rainfall_peak_mm_per_hr.",
    )

    # --- experiment matrix ---
    rank_sweep: tuple[int, ...] = Field(
        default=(2, 4, 8),
        description=(
            "FUNCTIONAL mpi-rank axis: build_experiment_matrix generates the mpi rows from this. "
            "The default (2,4,8) reproduces the fixed baseline mpi rows byte-for-byte."
        ),
    )
    include_resume: bool = Field(
        default=True, description="Emit the clean+resume (kill-and-hotstart) combined experiment."
    )

    # --- HPC targeting (partition-as-axis) ---
    hpc_system_config_yaml: Path = Field(
        description="Path to the per-cluster hpc_system_config YAML (partition specs)."
    )
    ensemble_partition: str = Field(
        description="analysis.hpc_ensemble_partition selector — the master-default per-sim partition."
    )
    setup_partition: str = Field(
        description="analysis.hpc_setup_and_analysis_processing_partition — setup/consolidate/process partition."
    )
    multi_sim_run_method: Literal["batch_job", "1_job_many_srun_tasks", "local"] = Field(
        default="batch_job",
        description="Execution strategy; per-row partition variation requires batch_job (Gotcha 54).",
    )

    @model_validator(mode="after")
    def _validate_coupling_invariant(self) -> synthetic_experiment_config:
        """Reject (n_rows, rank_sweep) combos that would deadlock the TRITON-SWMM
        coupling collective (a rank owning zero coupling junctions)."""
        from hhemt.synthetic_model import SyntheticModelParams
        from hhemt.synthetic_model.swmm_template import _N_COUPLING_NODES, _node_matrix_rows

        ranks = sorted({int(r) for r in self.rank_sweep})
        if not ranks:
            return self

        # (1) There must be at least as many coupling junctions as the largest rank,
        # else a top rank owns no node -> coupling-collective deadlock (triton.h:2363-2404).
        if max(ranks) > _N_COUPLING_NODES:
            raise ConfigurationError(
                field="rank_sweep",
                message=(
                    f"max rank {max(ranks)} exceeds the fixed number of SWMM coupling "
                    f"junctions (_N_COUPLING_NODES={_N_COUPLING_NODES}); a rank would own "
                    f"zero coupling nodes and deadlock the coupling collective. Reduce the "
                    f"largest rank in rank_sweep to <= {_N_COUPLING_NODES}."
                ),
                config_path=None,
            )

        # (2) n_rows divisible by every rank -> TRITON row-strips are exactly equal
        # (k*n_rows/rank), independent of TRITON's remainder distribution.
        bad_div = [r for r in ranks if self.n_rows % r != 0]
        if bad_div:
            raise ConfigurationError(
                field="n_rows",
                message=(
                    f"n_rows={self.n_rows} is not divisible by rank(s) {bad_div} in rank_sweep; "
                    f"TRITON's row-strip decomposition would be uneven and a rank could own "
                    f"zero coupling nodes -> ENSIFY_COMM_WORLD deadlock. Choose an n_rows "
                    f"divisible by every rank in rank_sweep."
                ),
                config_path=None,
            )

        # (3) Every equal row-strip must own >= 1 coupling node. Reuse the
        # solver-authoritative placement helper (it also asserts the interior span
        # is large enough for _N_COUPLING_NODES nodes).
        params = SyntheticModelParams(n_cols=self.n_cols, n_rows=self.n_rows, cell_size_m=self.cell_size_m)
        try:
            node_mrs = sorted(_node_matrix_rows(params))
        except AssertionError as exc:
            raise ConfigurationError(
                field="n_rows",
                message=(
                    f"synthetic grid n_rows={self.n_rows} is too small to place "
                    f"{_N_COUPLING_NODES} coupling nodes: {exc}"
                ),
                config_path=None,
            ) from exc
        for r in ranks:
            strip = self.n_rows // r
            for k in range(r):
                lo, hi = k * strip, (k + 1) * strip
                if not any(lo <= mr < hi for mr in node_mrs):
                    raise ConfigurationError(
                        field="n_rows",
                        message=(
                            f"rank {r}: TRITON row-strip [{lo},{hi}) owns no coupling node "
                            f"(coupling nodes at matrix-rows {node_mrs}); this rank would "
                            f"deadlock the coupling collective. Increase n_rows or drop rank {r}."
                        ),
                        config_path=None,
                    )
        return self

    @model_validator(mode="after")
    def _validate_caps(self) -> synthetic_experiment_config:
        """Reject any experiment-matrix row whose requested resources exceed the
        resolved PartitionSpec caps of its partition."""
        from hhemt.config.loaders import load_hpc_system_config
        from hhemt.synthetic_experiment import experiment_matrix_rows

        hpc = load_hpc_system_config(self.hpc_system_config_yaml)

        # The selectors must name declared partitions.
        for field_name, part in (
            ("ensemble_partition", self.ensemble_partition),
            ("setup_partition", self.setup_partition),
        ):
            if part not in hpc.partitions:
                raise ConfigurationError(
                    field=field_name,
                    message=(
                        f"{field_name}='{part}' is not a declared partition in "
                        f"{self.hpc_system_config_yaml}; declared: {sorted(hpc.partitions)}."
                    ),
                    config_path=None,
                )

        for row in experiment_matrix_rows(self):
            part_name = row["hpc.partition"]
            spec = hpc.partitions.get(part_name)
            if spec is None:
                raise ConfigurationError(
                    field="hpc.partition",
                    message=(
                        f"matrix row '{row['sa_id']}' targets undeclared partition "
                        f"'{part_name}'; declared: {sorted(hpc.partitions)}."
                    ),
                    config_path=None,
                )
            n_nodes = int(row["n_nodes"])
            n_gpus = int(row["n_gpus"])
            n_tasks = int(row["n_mpi_procs"]) * max(int(row["n_omp_threads"]), 1)

            if spec.max_nodes is not None and n_nodes > spec.max_nodes:
                _raise_cap(row, part_name, "n_nodes", n_nodes, "max_nodes", spec.max_nodes)
            if spec.max_gpu is not None and n_gpus > spec.max_gpu:
                _raise_cap(row, part_name, "n_gpus", n_gpus, "max_gpu", spec.max_gpu)
            if spec.gpus_per_node is not None and n_gpus > spec.gpus_per_node * n_nodes:
                _raise_cap(
                    row,
                    part_name,
                    "n_gpus",
                    n_gpus,
                    "gpus_per_node*n_nodes",
                    spec.gpus_per_node * n_nodes,
                )
            if spec.cpus_per_node is not None and n_tasks > spec.cpus_per_node * n_nodes:
                _raise_cap(
                    row,
                    part_name,
                    "n_mpi_procs*n_omp_threads",
                    n_tasks,
                    "cpus_per_node*n_nodes",
                    spec.cpus_per_node * n_nodes,
                )
            runtime = row.get("hpc_time_min_per_sim")
            if runtime is not None:
                hpc.check_runtime_within_cap(part_name, int(runtime))
        return self


def _raise_cap(row, part_name, req_field, req_val, cap_field, cap_val) -> None:
    raise ConfigurationError(
        field="rank_sweep/partition caps",
        message=(
            f"matrix row '{row['sa_id']}' requests {req_field}={req_val} on partition "
            f"'{part_name}', exceeding its {cap_field}={cap_val}."
        ),
        config_path=None,
    )

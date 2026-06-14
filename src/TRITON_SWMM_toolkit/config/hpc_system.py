"""Pydantic models for the per-HPC-system configuration (`hpc_system_config.yaml`).

A user authors ONE file per cluster from its public docs: a ``partitions:`` block
whose field names mirror the ``snakemake-executor-plugin-slurm`` ``PartitionLimits``
schema (native vocabulary) plus toolkit-only extension fields (node topology,
gres-vs-gpus flavor, execution-mode default, login node, modules, and a guarded
``executor_profile_extras`` escape hatch). Realizes ADR-6 of the
``public-release-reproducibility`` system design.

These models use plain ``BaseModel`` (NOT ``cfgBaseModel``) — they carry no local
``Path`` fields requiring on-disk existence, and HPC-side values describe a remote
cluster, not the local machine (mirrors the ``config/globus.py`` precedent, Gotcha 1).

The native ``partitions:`` field names are transcribed against the INSTALLED
``snakemake-executor-plugin-slurm`` 2.0.3 ``PartitionLimits`` (numeric caps mirrored
as ``int | None`` rather than the plugin's ``float`` — a deliberate toolkit-side
simplification, users author integer minutes/MB/cpus). The toolkit model is a
hand-mirror, not an import of the plugin class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from TRITON_SWMM_toolkit.exceptions import ConfigurationError

# Resources the toolkit emits per-rule whose override via a profile
# set-resources/default-resources entry would silently clobber a
# correctness-critical emission (Gotcha 32: tasks_per_gpu=0 / mpi=True
# gres-multi-GPU duplication invariants).
_TOOLKIT_EMITTED_RESOURCES: frozenset[str] = frozenset(
    {
        "tasks",
        "tasks_per_gpu",
        "mpi",
        "gres",
        "slurm_extra",
        "slurm_partition",
        "runtime",
        "mem_mb",
        "mem_mb_per_cpu",
        "cpus_per_task",
        "nodes",
    }
)


def _guard_executor_profile_extras(extras: dict) -> None:
    """Raise ConfigurationError if executor_profile_extras tries to override
    a toolkit-emitted per-rule resource via set-resources/default-resources.

    Scans both the top-level default-resources list and every per-rule
    set-resources mapping. default-resources entries are "name=value"
    strings; set-resources is {rule: {resource: value}}.
    """
    # default-resources: list[str] of "name=value"
    default_res = extras.get("default-resources", []) or []
    for entry in default_res:
        name = str(entry).split("=", 1)[0].strip()
        if name in _TOOLKIT_EMITTED_RESOURCES:
            raise ConfigurationError(
                field="executor_profile_extras.default-resources",
                message=(
                    f"'{name}' is a toolkit-emitted per-rule resource and may "
                    f"not be overridden via executor_profile_extras "
                    f"(Gotcha 32 clobber hazard). Remove it; the toolkit emits "
                    f"this resource correctly per-rule."
                ),
                config_path=None,
            )
    # set-resources: {rule: {resource: value}}
    set_res = extras.get("set-resources", {}) or {}
    for rule, res_map in set_res.items():
        for name in res_map or {}:
            if str(name).strip() in _TOOLKIT_EMITTED_RESOURCES:
                raise ConfigurationError(
                    field=f"executor_profile_extras.set-resources.{rule}",
                    message=(
                        f"'{name}' is a toolkit-emitted per-rule resource and "
                        f"may not be overridden via executor_profile_extras "
                        f"(Gotcha 32 clobber hazard). Remove it from rule "
                        f"'{rule}'."
                    ),
                    config_path=None,
                )


class PartitionSpec(BaseModel):
    """A single SLURM partition's capability profile.

    Native fields mirror ``snakemake-executor-plugin-slurm`` 2.0.3
    ``PartitionLimits``; extension fields carry toolkit-only metadata with no
    native ``PartitionLimits`` analogue.
    """

    model_config = ConfigDict(extra="forbid")

    # --- NATIVE (mirror snakemake-executor-plugin-slurm 2.0.3 PartitionLimits) ---
    max_runtime: int | None = None  # minutes; per-partition wall-clock cap
    max_mem_mb: int | None = None
    max_mem_mb_per_cpu: int | None = None
    max_cpus_per_task: int | None = None
    max_nodes: int | None = None
    max_gpu: int | None = None  # per-job GPU cap
    available_gpu_models: list[str] | None = None
    supports_mpi: bool = False
    # --- EXTENSION (no native PartitionLimits field) ---
    gpus_per_node: int | None = None  # node topology
    cpus_per_node: int | None = None  # node topology
    gpu_hardware: str | None = None  # D1 Option-A (VMS-4a): the specific arch string
    #                                  _resolve_cuda_arch_flags keys on (a6000/a100/h100/...),
    #                                  distinct from available_gpu_models (capability allowlist)
    gpu_compilation_backend: Literal["HIP", "CUDA"] | None = None  # D1: per-partition GPU backend


class hpc_system_config(BaseModel):
    """Per-HPC-system configuration loaded from ``hpc_system_config.yaml``.

    One file per cluster (ADR-6). Threaded through the toolkit as the third
    config path (``--hpc-system-config`` CLI / ``hpc_system_config=`` Python).
    """

    model_config = ConfigDict(extra="forbid")

    system_name: str
    default_account: str | None = None  # NATIVE value -> slurm_account default-resource
    login_node: str | None = None  # EXTENSION (tmux orchestration; batch_job only)
    default_execution_mode: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None
    gpu_allocation_flavor: Literal["gres", "gpus"] | None = None  # EXTENSION (chooses native channel)
    additional_modules: list[str] | None = None
    partitions: dict[str, PartitionSpec]
    executor_profile_extras: dict = Field(
        default_factory=dict,
        description=(
            "Advanced escape hatch: keys merged VERBATIM into the generated "
            ".snakemake_profile/{mode}/config.yaml. Use ONLY for plugin-level "
            "snakemake-executor-plugin-slurm settings the toolkit does not model "
            "(e.g. slurm-init-seconds-before-status-checks, slurm-requeue, "
            "slurm-reservation, max-jobs-per-second). A reject-guard FORBIDS "
            "set-resources / default-resources entries that name toolkit-emitted "
            "resources (tasks, tasks_per_gpu, mpi, gres, slurm_extra, "
            "slurm_partition, runtime, mem_mb, cpus_per_task, nodes) — those are "
            "correctness-critical per-rule emissions (see Gotcha 32) and may not "
            "be overridden via the profile."
        ),
    )

    @model_validator(mode="after")
    def _reject_clobbering_extras(self) -> hpc_system_config:
        # D4/R6 guard: reject set-resources/default-resources keys naming
        # toolkit-emitted resources. Adopt the snakemake-specialist VMS (Option B)
        # verbatim — raise ConfigurationError on a clobber attempt at config-load.
        _guard_executor_profile_extras(self.executor_profile_extras)
        return self

    def check_runtime_within_cap(self, partition: str, requested_runtime_min: int) -> None:
        """Raise ConfigurationError if a requested wall-clock runtime exceeds the
        partition's ``max_runtime`` cap (R5 preflight bound helper).

        Fail-fast at config/preflight time rather than surfacing as a cryptic
        SLURM rejection hours into a submitted job. A partition with no declared
        ``max_runtime`` imposes no cap (returns without error).
        """
        spec = self.partitions.get(partition)
        if spec is None:
            raise ConfigurationError(
                field="partitions",
                message=(
                    f"Unknown partition '{partition}'. "
                    f"Declared partitions: {sorted(self.partitions)}"
                ),
                config_path=None,
            )
        if spec.max_runtime is not None and requested_runtime_min > spec.max_runtime:
            raise ConfigurationError(
                field=f"partitions.{partition}.max_runtime",
                message=(
                    f"Requested runtime {requested_runtime_min} min exceeds the "
                    f"'{partition}' partition wall-clock cap of {spec.max_runtime} min. "
                    f"Reduce the requested runtime or choose a partition with a higher cap."
                ),
                config_path=None,
            )

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

from hhemt.exceptions import ConfigurationError

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
    max_concurrent_jobs: int | None = None  # EXTENSION (D-D): cluster-throughput cap; new home
    #                                          for the retired analysis_config.hpc_max_simultaneous_sims
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

    container: ContainerSpec | None = None  # ADR-1/2/3: the per-cluster Apptainer
    #   exec parameters; None on a cluster with no container support (native-only).

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


def resolve_gpu_target(
    cfg_hpc_system: hpc_system_config | None,
    partition_name: str | None,
) -> tuple[str | None, str | None]:
    """Resolve ``(gpu_hardware, gpu_compilation_backend)`` for a partition.

    Phase-4 (D1 Option-A / VMS-2a): GPU hardware + compilation backend live on
    ``PartitionSpec`` (per-partition), no longer on ``system_config``. This free
    function is the single resolution point used at ``TRITONSWMM_system``
    construction + the GPU-compile runners to inject those values.

    Returns ``(None, None)`` when no HPC-system config is present, no partition
    is named, the partition is undeclared, or the partition declares no GPU —
    i.e. the CPU / render-only path. Partition existence (for non-GPU concerns)
    is validated separately by ``check_runtime_within_cap`` / preflight.
    """
    if cfg_hpc_system is None or partition_name is None:
        return (None, None)
    spec = cfg_hpc_system.partitions.get(partition_name)
    if spec is None:
        return (None, None)
    return (spec.gpu_hardware, spec.gpu_compilation_backend)


def resolve_gpus_per_node(
    cfg_hpc_system: hpc_system_config | None,
    partition_name: str | None,
) -> int | None:
    """Resolve a partition's per-node GPU count from its PartitionSpec.

    Phase-4 (4d): per-node GPU topology moved off analysis_config.hpc_gpus_per_node
    to PartitionSpec.gpus_per_node. This free function is the resolution point for
    sites that have no SnakemakeWorkflowBuilder instance (analysis.py + the
    sensitivity builder's per-sub read). Returns None when no config / no partition /
    undeclared / no gpus_per_node on the spec; callers apply their own `or 0`/`or 1`.
    """
    if cfg_hpc_system is None or partition_name is None:
        return None
    spec = cfg_hpc_system.partitions.get(partition_name)
    if spec is None:
        return None
    return spec.gpus_per_node


def resolve_additional_modules(
    cfg_hpc_system: hpc_system_config | None,
) -> str | None:
    """Resolve the space-joined `module load` string from the HPC-system config.

    Phase-4 (4c, D-E / OE-1): `additional_modules` moved off `system_config` to
    `hpc_system_config.additional_modules` (a `list[str]`), but the compile/runtime
    consumers emit `f"module load {modules}"` expecting a space-joined `str`. This
    free function is the single join site, usable at the GPU-compile runner
    construction sites that have no `SnakemakeWorkflowBuilder` instance. Module
    names are space-free, so the join is lossless. Returns `None` (no `module load`
    line) when no config is present or no modules are declared.
    """
    if cfg_hpc_system is not None and cfg_hpc_system.additional_modules:
        return " ".join(cfg_hpc_system.additional_modules)
    return None


class ContainerSpec(BaseModel):
    """Per-cluster Apptainer container execution parameters (ADR-1/ADR-3).

    Cluster-coupled (C-HPC-FIELD-PLACEMENT): lives on hpc_system_config, NOT
    analysis_config. The native|container SELECTOR is the experiment-scoped
    analysis_config field; THIS model carries the cluster-bound 'how to exec'.
    Consumed only when analysis_config.execution_environment == "container".
    """

    model_config = ConfigDict(extra="forbid")

    sif_path: str  # absolute on-cluster path to the TRANSFERRED, signed SIF
    gpu_flag: Literal["--rocm", "--nv"] | None = None  # None => CPU-only cluster
    binds: list[str] = Field(default_factory=list)  # APPTAINER_BIND entries
    #   (e.g. "/opt/cray", "/var/spool/slurmd"); analysis_dir same:same is
    #   appended by the seam, never hand-authored here (it is experiment-scoped).
    containlibs: list[str] = Field(default_factory=list)  # APPTAINER_CONTAINLIBS
    apptainerenv_ld_library_path: str | None = None  # the Frontier Cray-MPICH
    #   recipe as a SHELL-TEMPLATE string (references ${CRAY_MPICH_DIR} etc. that
    #   expand at runtime AFTER `module load cray-mpich-abi`); None on UVA-CUDA.
    cray_mpich_abi_module: bool = False  # True on Frontier => the seam emits
    #   `module load cray-mpich-abi` ahead of the APPTAINERENV export so the
    #   ${CRAY_*} vars exist when the template expands. False on UVA.
    pre_exec_modules: list[str] = Field(default_factory=list)  # container-only Lmod
    #   modules emitted as `module load {m} 2>/dev/null` at the TOP of the container
    #   host-env segment — BEFORE `module load cray-mpich-abi` and the APPTAINERENV
    #   exports. Frontier production multi-rank: the OLCF helper set
    #   ["olcf-container-tools","apptainer-enable-mpi","apptainer-enable-gpu"], which
    #   bind the open-ended host MPI+ROCm+compiler-runtime closure (libpgmath/libflang/
    #   …) so it need NOT be hand-enumerated into containlibs (validated probe job
    #   4898044: size=16/2 nodes). Empty on UVA and on the single-rank validation fallback.
    srun_mpi: str | None = None  # UVA: "pmix" => the seam emits `srun --mpi=pmix`
    #   for the container-own OpenMPI; None on Frontier (Cray-PALS path, never pmix).
    exe_in_sif: dict[str, str] = Field(default_factory=dict)  # OD-A: per-model
    #   in-SIF absolute exe path, keyed by model_type ("triton"/"tritonswmm"/
    #   "swmm"). Empty => fall back to the convention /opt/hhemt/bin/{name}.
    extra_exec_args: list[str] = Field(default_factory=list)  # shared escape
    #   hatch for an unforeseen per-cluster `apptainer exec` flag; applied to
    #   EVERY class. NEVER put `--cleanenv` here for an MPI cluster (NQ-11).
    apptainer_module: str | None = None  # cluster Lmod module providing the
    #   `apptainer` binary. REQUIRED where apptainer is module-only (UVA Rivanna:
    #   "apptainer/1.5.0") — without it `srun … apptainer exec` dies execve
    #   (apptainer not on PATH). None on clusters where apptainer is on the default
    #   PATH (Frontier). The seam emits `module load {apptainer_module}` in container
    #   mode ONLY (native rows never load it -> native byte-identical).

    @model_validator(mode="after")
    def _check_mpi_flavor_exclusive(self):
        # FI4 (ADR-3 / R7): srun --mpi=pmix is UVA-only (container-own OpenMPI);
        # the host Cray-MPICH-ABI bind is Frontier-only. The two MPI flavors are
        # mutually exclusive — fail fast rather than silently emit a forbidden
        # Frontier+pmix combo at the seam (the guard the plan claimed "lives in
        # code" but nothing enforced).
        if self.cray_mpich_abi_module and self.srun_mpi:
            raise ValueError(
                "ContainerSpec: cray_mpich_abi_module=True (Frontier host-Cray-MPICH bind) "
                "is mutually exclusive with srun_mpi (UVA container-OpenMPI pmix). Set exactly one."
            )
        if self.cray_mpich_abi_module and not self.apptainerenv_ld_library_path:
            raise ValueError(
                "ContainerSpec: cray_mpich_abi_module=True requires apptainerenv_ld_library_path "
                "(the Cray-MPICH ABI LD_LIBRARY_PATH shell-template)."
            )
        return self


def resolve_container_spec(
    cfg_hpc_system: hpc_system_config | None,
) -> ContainerSpec | None:
    """Single resolution point for the per-cluster ContainerSpec (mirrors
    resolve_gpu_target / resolve_additional_modules). Returns None when no
    hpc-system config is present or no container block is declared — the
    native path. Container-mode preflight raises if this returns None while
    execution_environment == "container"."""
    if cfg_hpc_system is None:
        return None
    return cfg_hpc_system.container

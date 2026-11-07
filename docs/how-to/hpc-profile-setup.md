# Set up an HPC system profile

## Goal

Write an HPC system profile that describes one cluster, and point an analysis at the file with `--hpc-system-config` or with `Toolkit.from_configs(..., hpc_system_config=...)`.

## Prerequisites

- A working [installation](installation.md).
- The partition names, caps, and module names from the documentation of your cluster.

## Key references

- [Configuration schema](../reference/config-schema.md#hpc-system-config) gives the meaning, type, and requiredness of every field. The schema derives from the models themselves, so the schema stays current.
- [Config filling](config-filling.md) describes the path through the system, analysis, and HPC configs.
- [Building container images](building-container-images.md) covers the images for container mode.
- [Compile the solver](compiling-the-solver.md) covers the build for a partition.

## Start from a working profile

Below is an example HPC profile. The profile shows every accepted field. The optional fields that the example cluster leaves unused carry their default values. Anonymized versions of the profile and of an AMD-GPU counterpart are in [`test_data/norfolk_coastal_flooding/`](https://github.com/lassiterdc/hhemt/tree/main/test_data/norfolk_coastal_flooding) on the repository. Where a repository file and this page differ, this page is current.

```yaml
hpc_name: uva_rivanna
default_account: "{your-allocation}"   # CHANGE THIS: your SLURM account
login_node: login1.hpc.virginia.edu
gpu_allocation_flavor: gres            # this cluster requests GPUs with --gres=gpu:N
max_concurrent_jobs: 1000              # how many jobs to keep queued at once
max_concurrent_cpus: null              # optional CPU budget; measure your QOS first (below)
halt_sims_below_free_bytes: null       # optional disk floor; measure what statvfs sees first (below)
additional_modules:
  - gcc/12.4.0

partitions:
  standard:                            # CPU partition
    max_runtime: 4320                  # minutes
    max_mem_mb: 768000
    max_mem_mb_per_cpu: null           # no per-CPU memory cap declared
    max_cpus_per_task: 96
    max_nodes: 1
    supports_mpi: true
  gpu-a6000:
    max_runtime: 4320
    max_gpu: 8                         # per-job cap
    available_gpu_models: ["a6000"]    # what the scheduler will place here
    gpu_hardware: a6000                # what the solver is compiled for
    gpus_per_node: 8
    cpus_per_node: 48
    gpu_compilation_backend: CUDA
  gpu-a100-80:
    max_runtime: 4320
    max_gpu: 8
    available_gpu_models: ["a100"]
    gpu_hardware: a100
    gpus_per_node: 8
    cpus_per_node: 64
    gpu_compilation_backend: CUDA

container:                             # only read when execution_environment is container
  sif_root: "/scratch/{your-allocation}/sifs"   # CHANGE THIS: where `hhemt build-sifs` writes; images resolve by identity
  builds_containers: true              # this host may run `hhemt build-sifs` (fakeroot-capable)
  apptainer_module: "apptainer/1.5.0"  # the module that puts `apptainer` on PATH here
  gpu_flag: "--nv"
  srun_mpi: "pmix"
  binds: ["/scratch", "/sfs"]
  containlibs: []                      # the image carries its own MPI, so no host libraries are bound
  pre_exec_modules: []
  cray_mpich_abi_module: false         # Cray host-MPI bind; a setting for Cray systems
  apptainerenv_ld_library_path: null   # set only together with cray_mpich_abi_module
  exe_in_sif:
    triton: "/opt/hhemt/bin/triton_only.exe"
    tritonswmm: "/opt/hhemt/bin/triton.exe"
    swmm: "/opt/hhemt/bin/runswmm"
  python_in_sif: "python"              # the image's own interpreter
  extra_exec_args: []

executor_profile_extras: {}            # advanced escape hatch; leave empty
```

### Required changes

Change the values marked `CHANGE THIS` above.

- **`default_account`** is your SLURM account. The account is charged for every job that the toolkit submits.
- **`container.sif_root`** is the directory that `hhemt build-sifs` writes into and the only place where the toolkit looks for an image, resolved by identity. The field matters only with `execution_environment: container` on the analysis config. A native analysis ignores the whole `container:` block, so you can leave the block untouched.

### Changes for a different cluster

Everything under `partitions:` describes real hardware and real scheduler caps. A profile for another cluster takes the values of that site from the documentation of the cluster. Preflight validates a requested analysis against the caps, so a wrong value surfaces as a rejection of an analysis that the cluster would allow, or as a job that the scheduler refuses after the toolkit accepted the analysis.

### Fields to leave alone

`executor_profile_extras` stays empty unless you pass a Snakemake SLURM plugin setting that is missing from the toolkit models. The keys of `container:` other than `sif_root` and `builds_containers` describe how the cluster runs the container and the contents of the image. Most keys describe the cluster, meaning how the cluster launches a container and exposes GPUs and fabric to the container. `exe_in_sif` and `python_in_sif` describe the image, and both must resolve inside the image and not on the host.

`exe_in_sif` gives the absolute in-image path of each simulation executable. `python_in_sif` is the interpreter command for the processing steps of the container, and a path is not required. The bare name `python` works because every image recipe that the toolkit ships for `hhemt build-sifs` puts the image's own Python first on the `PATH`. Set the field to an absolute in-image path only for an image whose `PATH` leads to a different Python, such as an image built from a recipe of your own. The values in the example are correct for the example cluster.

!!! tip "Start native, add containers later"
    Container mode is opt-in, and [Building container images](building-container-images.md) covers the images. Get a native analysis working first.

## Verify the site caps before you rely on them

The optional fields `max_concurrent_cpus` and `halt_sims_below_free_bytes` hold a cap that tracks the real ceiling of your cluster only when the cluster meters the quantity that the field counts. Measure the quantity before you set either field.

### `max_concurrent_cpus` counts CPUs

The field tracks a site ceiling only where the ceiling also meters `cpu`. Check the quantities that your QOS meters.

```bash
sacctmgr -nP show qos standard,normal format=Name,MaxTRESPU,GrpTRES,MaxTRES
```

On the example cluster the command returns `standard|cpu=1500||cpu=1500`, and no QOS on the cluster meters `billing`, so the CPU budget tracks the real ceiling there. Where a QOS meters `billing` instead, the budget holds the CPUs that the field names, and the hold falls short of the ceiling of the cluster.

### `halt_sims_below_free_bytes` reads `statvfs`

The free space that `statvfs` reports equals your quota only where the quota is a GPFS fileset with `filesetdf` enabled. Otherwise a quota-exhausted analysis gets `EDQUOT` on write while `statvfs` still reports free space, and the guard never fires. Check the value that `statvfs` reports, substituting your own analysis directory.

```bash
python3 -c "import os,sys; s=os.statvfs(sys.argv[1]); print(s.f_blocks*s.f_frsize/1e12,'TB total')" /path/to/analysis_dir
```

On `/scratch` of the example cluster the command reported 13.19 TB total. The value is the real filesystem ceiling and no smaller soft limit, so the guard covers the binding limit there.

## Choose a partition

Pick a partition name on your **analysis** config. Use `hpc_ensemble_partition` for the simulations and `hpc_setup_and_analysis_processing_partition` for setup and post-processing. The name must be a key under `partitions:` in the profile. The profile above offers `standard`, `gpu-a6000`, and `gpu-a100-80`.

!!! warning "GPU hardware derives from the partition"
    You set no `gpu_hardware` on the analysis config. The partition you choose determines the GPU architecture that the solver is compiled for and the GPU backend that builds the solver, because the toolkit reads `gpu_hardware` and `gpu_compilation_backend` from the entry of that partition above.

    The GPU partitions in the profile differ only in hardware values for this reason, and selecting `gpu-a100-80` in place of `gpu-a6000` changes the GPU hardware of an analysis.

Compile the solver for a partition before the first analysis on the partition. [Compile the solver](compiling-the-solver.md#on-a-cluster) gives the command-line form.

## Differences on an AMD-GPU cluster

A profile for an AMD-GPU cluster uses the same schema with different values.

??? example "AMD-GPU profile compared with the UVA profile above"
    Apart from `hpc_name`, the top-level fields below differ.

    | Field | UVA Rivanna | OLCF Frontier |
    |---|---|---|
    | `gpu_allocation_flavor` | `gres`, so GPUs are requested with `--gres=gpu:N` | `gpus`, so GPUs are requested with `--gpus-per-node` |
    | `additional_modules` | `gcc/12.4.0` | `PrgEnv-amd` and `rocm` |
    | `login_node` | set, because `batch_job` runs an orchestrator there | absent, because that dispatch method uses no orchestrator |

    The AMD-GPU cluster has a single `batch` partition where the UVA profile has more. The partition uses `gpu_compilation_backend: HIP` and `gpu_hardware: mi250x`, the partition-derives-hardware rule above producing an AMD build in place of an NVIDIA build.

    The `container:` block carries more, because a container against the Cray MPI and Slingshot fabric needs library paths and bind mounts that a container on the UVA profile avoids. `containlibs`, `pre_exec_modules`, `cray_mpich_abi_module`, and `apptainerenv_ld_library_path` sit at their defaults in the profile above, and the AMD-GPU profile fills them. Take that block unchanged.

    The full file is in [`test_data/norfolk_coastal_flooding/`](https://github.com/lassiterdc/hhemt/tree/main/test_data/norfolk_coastal_flooding) on the repository.

# Set up an HPC system profile

--8<-- "hpc-system-config-role.md"

One file describes one cluster. Point a run at it with `--hpc-system-config`, or
with `Toolkit.from_configs(..., hpc_system_config=...)`.

This page shows a complete working profile you can copy. For the meaning, type and
requiredness of every individual field, see the generated
[configuration schema](../reference/config-schema.md#hpc-system-config), which is
derived from the models themselves and cannot fall behind them.

## Start from a working profile

This is the UVA Rivanna profile, complete and ready to copy. Anonymized copies of
this file and its Frontier counterpart also ship in the repository under
`test_data/norfolk_coastal_flooding/`.

```yaml
system_name: uva_rivanna
default_account: "{your-allocation}"   # CHANGE THIS: your SLURM account
login_node: login1.hpc.virginia.edu
default_execution_mode: batch_job      # one sbatch per simulation
gpu_allocation_flavor: gres            # this cluster requests GPUs with --gres=gpu:N
max_concurrent_jobs: 1000              # how many jobs to keep queued at once
additional_modules:
  - gcc/12.4.0

partitions:
  standard:                            # CPU partition
    max_runtime: 4320                  # minutes
    max_mem_mb: 768000
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
  sif_path: "/scratch/{your-allocation}/hhemt_uva_cuda.sif"   # CHANGE THIS if you use containers
  gpu_flag: "--nv"
  srun_mpi: "pmix"
  binds: ["/scratch", "/sfs"]
  exe_in_sif:
    triton: "/opt/hhemt/bin/triton_only.exe"
    tritonswmm: "/opt/hhemt/bin/triton.exe"
    swmm: "/opt/hhemt/bin/runswmm"

executor_profile_extras: {}            # advanced escape hatch; leave empty
```

### What you must change

Two values, both marked `CHANGE THIS` above:

- **`default_account`** is your SLURM account. Every job the toolkit submits is
  charged to it.
- **`container.sif_path`** is where your Apptainer image lives, and it matters only
  if you set `execution_environment: container` on the analysis config. A native
  run ignores the whole `container:` block, so you can leave it as it stands.

### What you should change if your cluster differs

Everything under `partitions:` describes real hardware and real scheduler caps. If
you are writing a profile for a cluster that is not Rivanna, these are the values to
replace with your own site's, taken from your cluster's documentation. The caps are
what preflight validates a requested run against, so a wrong value here surfaces
either as a rejected run that should have been allowed, or as a job the scheduler
refuses after the toolkit accepted it.

### What you can leave alone

`executor_profile_extras` stays empty unless you are passing a Snakemake SLURM
plugin setting the toolkit does not model. Inside `container:`, everything except
`sif_path` describes how this cluster exposes its GPUs and fabric to a container,
and those values are correct for Rivanna as written.

!!! tip "Start native, add containers later"
    Container mode is opt-in and needs a transferred, signed Apptainer image. Get a
    native run working first, then see `containers/README.md`.

## Verify two site caps before you rely on them

Two optional fields on the HPC-system config hold a cap that tracks your
cluster's real ceiling only if your cluster meters the same thing the field
counts. Measure both before you set either.

### `max_concurrent_cpus` counts CPUs

It tracks a site ceiling only where that ceiling also meters `cpu`. Check what
your QOS meters:

```bash
sacctmgr -nP show qos standard,normal format=Name,MaxTRESPU,GrpTRES,MaxTRES
```

On Rivanna this returns `standard|cpu=1500||cpu=1500`, and no QOS on that
cluster meters `billing`, so the CPU budget tracks the real ceiling there. Where
a QOS meters `billing` instead, the budget still holds exactly the CPUs it
names, but that hold will not track the cluster's own ceiling.

### `halt_sims_below_free_bytes` reads `statvfs`

`statvfs` reports the filesystem's free space, which equals your quota only
where the quota is a GPFS fileset with `filesetdf` enabled. Otherwise a
quota-exhausted run gets `EDQUOT` on write while `statvfs` still reports free
space, and the guard never fires. Check what `statvfs` sees, substituting your
own analysis directory:

```bash
python3 -c "import os,sys; s=os.statvfs(sys.argv[1]); print(s.f_blocks*s.f_frsize/1e12,'TB total')" /path/to/analysis_dir
```

On Rivanna `/scratch` this reported 13.19 TB total, which is the real filesystem
ceiling rather than a smaller soft limit, so the guard covers the binding one
there.

## Choose a partition

Pick a partition name on your **analysis** config, using
`hpc_ensemble_partition` for the simulations and
`hpc_setup_and_analysis_processing_partition` for setup and post-processing. The
name must be a key under `partitions:` in this profile.

!!! warning "GPU hardware derives from the partition"
    You do not set `gpu_hardware` on the analysis config. The partition you choose
    determines both the GPU architecture the solver is compiled for and the GPU
    backend it is built with, by reading `gpu_hardware` and
    `gpu_compilation_backend` off that partition's entry above.

    This is why the two GPU partitions in the profile differ only in a handful of
    values: selecting `gpu-a100-80` instead of `gpu-a6000` is how you change GPU
    hardware for a run.

On the profile above that gives you `standard`, `gpu-a6000` and `gpu-a100-80`.

The same partition entry decides what the solver is built for, and a run checks
for that build rather than making it. Before the first run on a partition, compile
for it with the command-line form on
[Compile the solver](compiling-the-solver.md#on-a-cluster).

## A second cluster, and what actually differs

A Frontier profile is not a different kind of file. It is the same schema with
different values, and the differences are worth knowing because they are the ones
that vary most between sites.

??? example "How the OLCF Frontier profile differs from the UVA one above"
    Four things differ at the top level:

    | Field | UVA Rivanna | OLCF Frontier |
    |---|---|---|
    | `gpu_allocation_flavor` | `gres`, so GPUs are requested with `--gres=gpu:N` | `gpus`, so GPUs are requested with `--gpus-per-node` |
    | `default_execution_mode` | `batch_job`, one sbatch per simulation | `1_job_many_srun_tasks`, one allocation with a pool of steps |
    | `additional_modules` | `gcc/12.4.0` | `PrgEnv-amd` and `rocm` |
    | `login_node` | set, because `batch_job` runs an orchestrator there | absent, because that dispatch method does not use one |

    The partition set differs too: Frontier has a single `batch` partition where
    Rivanna has three. That partition uses `gpu_compilation_backend: HIP` and
    `gpu_hardware: mi250x`, which is the same partition-derives-hardware rule above
    producing an AMD build instead of an NVIDIA one.

    Its `container:` block is substantially larger, because running a container
    against the Cray MPI and Slingshot fabric needs library paths and bind mounts
    that a Rivanna container does not. Copy that block as it stands rather than
    reconstructing it.

    The full file is at
    `test_data/norfolk_coastal_flooding/hpc_system_config_frontier.yaml`.

## See also

- [Configuration schema](../reference/config-schema.md#hpc-system-config): every field, with its type, requiredness and default.
- [Config-filling guide](config-filling.md): the task-oriented path through all three configs.
- [Norfolk end-to-end tutorial](../tutorials/norfolk-end-to-end.md)

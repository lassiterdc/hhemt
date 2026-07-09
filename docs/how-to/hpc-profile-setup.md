# Set up an HPC system profile

hhemt describes each cluster in an `hpc_system_config.yaml` (one per cluster), threaded via `--hpc-system-config` or `Toolkit.from_configs(..., hpc_system_config=...)`. Anonymized example profiles ship in-repo: `test_data/norfolk_coastal_flooding/hpc_system_config_uva.yaml` and `hpc_system_config_frontier.yaml`.

## Fill in your allocation

Copy an example profile and fill exactly two fields:

- `default_account: "{your-allocation}"` → your SLURM account (`slurm_account`).
- `container.sif_path` → only if you run in container mode (`execution_environment: container`); the native default ignores the whole `container:` block.

!!! tip
    Start with native (no SIF). Container mode is opt-in and needs a transferred, signed Apptainer SIF — see `containers/README.md`.

## Choose a partition

Pick a partition name on your analysis config (`hpc_ensemble_partition`, `hpc_setup_and_analysis_processing_partition`) that keys into `partitions:` in the profile. GPU hardware and compilation backend DERIVE from the partition — you do not set `gpu_hardware` directly. (UVA: `standard` / `gpu-a6000` / `gpu-a100-80`; Frontier: `batch`.)

## See also
- [Config-filling guide](config-filling.md)
- [Norfolk end-to-end tutorial](../tutorials/norfolk-end-to-end.md)

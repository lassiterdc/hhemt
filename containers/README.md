<!-- hhemt:repo-internal exempt=prose -->

# hhemt container SIFs — built by `hhemt build-sifs` from the family recipes under `src/hhemt/sif/recipes/`

This directory no longer holds recipes. The four FAMILY recipes are toolkit package
data (`src/hhemt/sif/recipes/{openmpi-cpu,openmpi-cuda,cray-mpich-cpu,cray-mpich-rocm}.def`,
ADR-21) and every image is created by ONE verb:

```bash
hhemt build-sifs --build-hpc-config hpc_system_config_uva.yaml \
                 --sif-build-config sif_build.yaml \
                 --experiment experiments/norfolk/stochastic --dry-run   # the plan table
hhemt build-sifs --build-hpc-config hpc_system_config_uva.yaml \
                 --sif-build-config sif_build.yaml \
                 --experiment experiments/norfolk/stochastic             # detached build DAG
```

`--dry-run` prints one row per IDENTITY the named experiments need (family, gpu hardware,
Kokkos arch, TRITON sha, hhemt sha, the resolved `.sif` path, and whether it already
exists under `sif_root`) and writes nothing. Without `--dry-run` the build DAG (one
Snakemake rule per identity, one SLURM job each on the build partition named in
`sif_build.yaml`) runs DETACHED and never holds your login shell; `--foreground` keeps it
in the shell, `--only KEY` restricts it, `--force` rebuilds. Measured build times on
Rivanna: `openmpi-cpu` 2h21m, `openmpi-cuda` (a100) 1h13m, `openmpi-cuda` (a6000) 5h45m.

## Identity, not filename

A SIF is addressed by an IDENTITY derived from the experiment's own configuration —
`(mpi_family, accel, gpu_hardware, gpu_compilation_backend, kokkos_arch, triton_url,
triton_sha, swmm_tag, hhemt_sha, base_digest, recipe_sha256)` — never by a path an operator
types. `hpc_system_config.container` carries only `sif_root` (where images live) and
`builds_containers` (whether this host may build); `sif_path`, `sif_paths_by_arch` and
`sif_sha256` are retired and REFUSED at config load. At run time the toolkit recomputes the
identity from the running git checkout (`git rev-parse HEAD`, 40-hex) and resolves
`{sif_root}/{family}/{stem}.sif`; preflight fails fast if the image or its manifest is
absent, or if the image's labels disagree with the identity. A `pip install` of hhemt is
refused in container mode: the driver must be a git checkout, so the image it resolves is the
one built from that exact commit.

## What a recipe does

Every identity value enters a recipe as an Apptainer `{{ VAR }}` template variable rendered
from the identity by `hhemt.sif.args.render_build_args` (parity is checked both ways before
any build). `%files` copies a `git archive` of the identity's `hhemt_sha` (so the image
carries no `.git`, no `.venv`, no test data), and `%post` MEASURES what landed — the TRITON
checkout sha, the git-export-substituted `HHEMT_SHA`, the standalone and coupled SWMM
versions, the MPI version, the toolkit version — appending them to `$APPTAINER_LABELS` and
FAILING the build on any mismatch with the request. The CUDA and ROCm families additionally
assert that the compiled `triton.exe` carries the identity's `sm_NN` / `gfx` code object, so
a mis-keyed arch dies at the end of the compile step rather than at the first GPU job.

After the build, `hhemt.sif.transaction` reads the labels back (`apptainer inspect --json`)
and REJECTS an image whose labels differ from the requested identity (the rejected image is
kept beside the target as `*.rejected.*` for forensics); on success it writes the sidecar
`{stem}.manifest.json` (identity, sha256 digest, build host, durations) which is what the
resolver, preflight, and the bundle emitter read.

## Provenance and bundles

The bundle carries the resolved images' `.manifest.json` set
(`bundle_manifest.json["sif_manifests"]`, schema 6); `from_doi` places each carried image at
its identity path under the reproducer's `sif_root` or rebuilds it through the same
transaction from a checkout at the carried `hhemt_sha`. The RO-Crate references the image by
its SHA-256 digest; `bundle.reprex()` verifies a fetched image against that digest
fail-closed. SIFs are NOT PGP-signed by the toolkit and are never embedded in a bundle
(digest-manifested and DOI-referenced, ADR-2 as amended by ADR-21).

## Build hosts

- **UVA Rivanna:** `apptainer build --fakeroot` works on the login node (root-mapped
  namespace fallback; no `/etc/subuid` entry needed) and both `docker.io` and
  `code.ornl.gov` are reachable, so images are built in place under `sif_root` on
  `/scratch` by the build DAG's SLURM jobs.
- **OLCF Frontier:** Frontier login nodes do not grant fakeroot and ORNL Harbor blocks the
  in-job CPE pull. Build the `cray-mpich-*` identities on a host that has fakeroot (a Linux
  box, or Rivanna with `builds_containers: true`), then place the `.sif` AND its
  `.manifest.json` under Frontier's `sif_root` at the same relative path; preflight verifies
  the labels and the digest, not the filename.

## Selecting container mode

```yaml
# analysis_config.yaml
execution_environment: container        # default is "native" (byte-identical to today)
```

```yaml
# hpc_system_config.yaml  (see test_data/norfolk_coastal_flooding/hpc_system_config_*.yaml)
container:
  sif_root: "/scratch/{your-user}/sifs"
  builds_containers: true               # false on a host that only RUNS images
  gpu_flag: "--nv"
  # … cluster-specific MPI-bind fields (see the example profiles)
```

Flip back to `execution_environment: native` at any time — the native source build is never
removed (C-NONCONTAINER), so it is always the fallback.

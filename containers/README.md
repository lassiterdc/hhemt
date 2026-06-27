# hhemt container SIFs — off-site build, sign, and transfer

This directory holds the two cluster-specific Apptainer definition files for the
hhemt reproducibility-system container subsystem (component C1; ADR-1/2/3/4):

| `.def` | Cluster | GPU / MPI model | Status |
|---|---|---|---|
| `frontier-rocm.def` | OLCF Frontier | Kokkos-HIP (`gfx90a`/MI250X); host Cray-MPICH-ABI bind | validated probe lineage (job 4898044) |
| `uva-cuda.def` | UVA Rivanna | Kokkos-CUDA; container-own OpenMPI + `srun --mpi=pmix` | design-complete, pending Phase-5 validation |

Each SIF carries a pre-built, SWMM-coupled `triton.exe` plus a standalone
`runswmm` and the hhemt Python toolkit, so the on-cluster source compile is
**skipped** in container mode (this is what dissolves the M-7 libstdc++/MPI
reconciliation bug class). The signed, fetched SIF's **SHA-256 is the
within-family identity carrier** — a byte-identical SquashFS rebuild is
foreclosed (ADR-4), so the SIF is referenced by DOI + SHA-256, never embedded.

> **Why off-site?** ADR-2: the build needs `root`/`--fakeroot`, which the
> clusters' login nodes do not grant, and ORNL Harbor's vuln-severity policy
> blocks the in-job CPE pull. Build where root exists (a Linux box / laptop),
> sign, then transfer the signed SIF to the cluster.

---

## 0. Prerequisites (off-site build host)

- A Linux box with **`apptainer`** installed and either `--fakeroot` (rootless;
  needs `uidmap` + `/etc/sub{u,g}id` mappings) or `sudo`.
- A **PGP key** for signing (`apptainer key newpair`, or an existing key).
- Tens of GB free disk — the ROCm `-complete` base alone is large; point
  `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` at a roomy filesystem.
- `skopeo` (or `docker buildx`) to capture the base-image digest, and `jq`.

## 1. Assemble the build context (pin capture)

The `.def` files reference values that must be pinned at build time (ADR-2 pin
contract; ADR-4 within-family identity). Capture them into the build context
**before** building:

### 1a. Base-image digest pin

```bash
# Frontier-ROCm:
skopeo inspect docker://rocm/dev-ubuntu-24.04:6.4.3-complete | jq -r '.Digest'
# UVA-CUDA (matches the proven Rivanna stack: cuda/12.8.0 + gompi/14.2.0_5.0.7):
skopeo inspect docker://nvidia/cuda:12.8.0-devel-ubuntu24.04 | jq -r '.Digest'
# -> sha256:XXXX ; substitute into each .def's `From: …@sha256:<PINNED_BASE_DIGEST>`
#    and record (tag, sha256) in the SIF lockfile alongside the conda lockfile.
```

### 1b. Pinned conda explicit lockfile

The `.def` `%files`-copies `hhemt.conda-lock.yml` into the build context and
`micromamba create -f` it with **no re-solve**. Generate it from the working
hhemt environment and commit it here (must pin `libstdcxx-ng`/`libgcc-ng` at or
above the base gcc-13 libstdc++ floor — the FQ2 single-libstdc++ guarantee):

```bash
# conda-lock (preferred, multi-platform), OR `conda list --explicit > hhemt.conda-lock.yml`
conda-lock --kind explicit -p linux-64 -f environment.yml --filename-template 'containers/hhemt.conda-lock.yml'
```

### 1c. Pinned source git-shas

- Set each `.def`'s `<PINNED_TRITON_GIT_URL>` + `<PINNED_TRITON_COMMIT_SHA>` to
  the experiment's TRITON-SWMM repo at a **fixed commit** (not the moving
  `TRITONSWMM_branch_key` — OE-3). A known-working commit is
  `02438b60613a7d913d884e7b836f9f5ff421fe7d`.
- SWMM is pinned to the public tag `v5.2.4` (USEPA) inside the `.def`.
- `git submodule --recursive` pins Kokkos + bundled SWMM transitively.

### 1d. MPICH source hash (Frontier only)

```bash
sha256sum mpich-3.4.3.tar.gz   # -> substitute into <PINNED_MPICH_SHA256>
```

### 1e. Host-floor verification (one-time, on a cluster GPU compute node)

```bash
# Frontier:  ldd --version | head -1   (host glibc; expect >= 2.38)
# UVA:       ldd --version | head -1   (host glibc; RHEL/Rocky-8 family ~2.28)
#            nvidia-smi | grep "CUDA Version"   (host driver CUDA major.minor; must be >= 12)
#            srun --mpi=list                    (confirm `pmix` is present)
#            module spider openmpi              (confirm container OpenMPI major.minor matches)
```

## 2. Build

```bash
export APPTAINER_CACHEDIR="$PWD/.apptainer_cache"
export APPTAINER_TMPDIR="$PWD/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Rootless first; fall back to sudo if --fakeroot is unavailable.
apptainer build --fakeroot containers/hhemt_frontier_rocm.sif containers/frontier-rocm.def \
  || sudo -E apptainer build containers/hhemt_frontier_rocm.sif containers/frontier-rocm.def
# (and likewise for uva-cuda.def -> hhemt_uva_cuda.sif)
```

> The single highest-risk integration point in `frontier-rocm.def` is whether
> Kokkos-HIP's cmake compiler detection accepts `mpicxx`-wrapping-`hipcc` as a
> HIP compiler — confirm `Built target triton.exe` in the build log.

## 3. Sign

```bash
apptainer key newpair            # once, if you have no signing key
apptainer sign containers/hhemt_frontier_rocm.sif
apptainer verify containers/hhemt_frontier_rocm.sif
```

## 4. Compute SHA-256 + record the DOI

```bash
sha256sum containers/hhemt_frontier_rocm.sif
```

Archive the signed SIF to a persistent store (HydroShare / Zenodo — see the
sibling `reproducibility-system_bundle-reprex-roundtrip` plan), mint a **DOI**,
and record `(DOI, SHA-256, base digest, triton sha, conda lockfile hash)` in the
SIF lockfile. The bundle/RO-Crate references the SIF by **DOI + SHA-256**, never
by embedding the 3–8 GB file.

## 5. Transfer to the cluster

```bash
# Frontier ($MEMBERWORK, readable from a compute node):
scp containers/hhemt_frontier_rocm.sif \
    YOUR_OLCF_USER@dtn.olcf.ornl.gov:'$MEMBERWORK/{your-allocation}/hhemt_frontier_rocm.sif'
# UVA Rivanna (/scratch):
scp containers/hhemt_uva_cuda.sif \
    rivanna:'/scratch/{your-allocation}/hhemt_uva_cuda.sif'
# (or use Globus for the large transfer; destination must be readable from a compute node)
```

## 6. Point the profile at the transferred SIF

Set `container.sif_path` in your `hpc_system_config` to the transferred path,
and select container mode in your `analysis_config`:

```yaml
# analysis_config.yaml
execution_environment: container        # default is "native" (byte-identical to today)
```

```yaml
# hpc_system_config.yaml  (see test_data/norfolk_coastal_flooding/hpc_system_config_*.yaml
#                          for the full anonymized container: blocks)
container:
  sif_path: "${MEMBERWORK}/{your-allocation}/hhemt_frontier_rocm.sif"
  gpu_flag: "--rocm"
  # … cluster-specific MPI-bind fields (see the example profiles)
```

Flip back to `execution_environment: native` at any time — the native source
build is never removed (C-NONCONTAINER), so it is always the fallback.

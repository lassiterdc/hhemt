---
type: integration
title: Solver builds and container identity
description: Native compilation boundaries, partition-resolved GPU targets, identity-derived Apptainer images, and transactional SIF verification.
tags: [solver, compilation, containers, apptainer, identity]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-9b00d480292dea607c832ef8
    resource: repo://containers/README.md
  - id: openwiki-source-89268c2c76dccde01e70ed85
    resource: repo://docs/how-to/compiling-the-solver.md
  - id: openwiki-source-8c5027a2ab5d04823b61b767
    resource: repo://src/hhemt/config/hpc_system.py
  - id: openwiki-source-76507eb2d2ecd573f79f8fd2
    resource: repo://src/hhemt/run_simulation.py
  - id: openwiki-source-244df0efdde1c09a688f9ce6
    resource: repo://src/hhemt/setup_workflow.py
  - id: openwiki-source-2ab907e5933670879507da45
    resource: repo://src/hhemt/sif/identity.py
  - id: openwiki-source-66329bacc5e41fc7c3fb155e
    resource: repo://src/hhemt/sif/plan.py
  - id: openwiki-source-87fde414d99b546f46eb9f4a
    resource: repo://src/hhemt/sif/preflight.py
  - id: openwiki-source-8bbca6700397a3d803641bab
    resource: repo://src/hhemt/sif/transaction.py
  - id: openwiki-source-e026674a4e4c40d1fa882dd1
    resource: repo://src/hhemt/system.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Solver builds and container identity

## Native builds are a prerequisite

Workflow setup asserts the native compiled tier rather than automatically building a shared solver tree. Direct `python -m hhemt.setup_workflow` invocations can request `--compile-triton-swmm`, `--compile-triton-only`, or `--compile-swmm`; without those flags, an enabled native model with no successful build stops setup. Container mode skips native compilation and its native-build verification, but system input preprocessing remains independent: the image supplies binaries, not the DEM.

GPU hardware and compilation backend belong to the selected HPC partition. `resolve_gpu_target` returns that pair or `(None, None)` when no configured GPU target resolves. Setup explicitly refuses requested GPUs without a resolved backend. A workstation invocation without an HPC profile takes the CPU path. See the [compilation guide](../../docs/how-to/compiling-the-solver.md) for commands and [HPC execution](../operations/hpc-execution.md) for scheduler placement.

The TRITON-only compiler downloads missing source, verifies the source pin, captures solver provenance, and builds the requested CPU/GPU backends. Per-build-directory locking surrounds the backend build/reuse boundary. Existing successful-build markers are not a general content-addressed invalidation system: the operator guide requires deliberate recompilation after relevant source/module changes and warns that remote-URL changes are not independently checked. Native build reuse is therefore weaker than the SIF identity contract below.

## Images are resolved from identity

`SifIdentity` carries MPI family, accelerator, GPU hardware/backend and Kokkos architecture, TRITON URL and resolved SHA, optional standalone SWMM tag, toolkit SHA, base-image digest, and recipe digest. Its key hashes the full identity; its filename stem contains family/hardware and abbreviated TRITON/toolkit SHAs. The resolver uses `{sif_root}/{family}/{stem}.sif`. Because the filename is less complete than the identity, the manifest and validation gates matter: a plausible filename alone is not proof.

`derive_identity` requires a full running toolkit SHA, derives the image family and partition architecture, resolves the solver pin, and hashes the selected recipe. A wheel-only driver without that Git identity is refused. Four packaged recipe families cover OpenMPI CPU/CUDA and Cray-MPICH CPU/ROCm; `containers/` is operator guidance, not the recipe owner.

`plan_sif_set` deduplicates identities over all required experiment partitions. It refuses dirty toolkit trees, non-container experiments, incompatible toolkit pins, and identities with queued/submitted producer sentinels. Without force it regards a SIF with its adjacent manifest as already built; that planning shortcut is not the runtime integrity check.

## Transactional publication

`build_transaction` requires `SLURMD_NODENAME`, including when a user already holds an allocation: the build must execute in a job step. It stages a `git archive` of the exact toolkit commit and runs `apptainer build --fakeroot` into a temporary image. Cache, temporary files, and staged source are isolated beneath the identity's `_build` directory.

After a successful build, it reads labels back with `apptainer inspect --json` and compares identity plus expected measured labels. Mismatches retain a `.rejected.*` image for diagnosis instead of publishing it. Build failure removes the temporary image and leaves any prior final image intact.

A passing image is hashed and accompanied by a manifest containing identity, read-back labels, digest, size, host/job information, recipe/build-argument digests, and duration. Publication replaces the image, then its digest sidecar, then its manifest **last**. This is an ordered multi-file transaction, not a single atomic replacement of all three artifacts. A rename failure preserves the built image and raises with a recovery message. Successful publication removes the working directory.

## Runtime preflight and binding

Container preflight recomputes identity for each required partition. It requires the image and manifest, compares manifest identity, rehashes image bytes, and checks recorded identity/TRITON/toolkit labels plus standalone SWMM when declared. The ordinary path refuses missing images; `build_sifs=True` can downgrade that particular absence to a warning so a build pre-step can supply it. Existing corrupt or mismatched images are not thereby accepted. Runtime preflight uses the transaction's recorded label set; the build transaction is where fresh image label inspection occurs.

The simulation launcher resolves the partition-specific image, substitutes the in-image executable, adds the configured GPU flag only for GPU run modes, and binds the model-specific host output directory into the executable's expected image project directory. This avoids directing writes into the read-only SIF. Host pre-exec modules, optional Cray ABI setup, library injection, configured binds, system-directory coverage, and the analysis directory form the runtime environment. These cluster ABI requirements remain host-profile responsibilities rather than disappearing inside an image.

## Reproduction and verification limits

[Experiment reproduction](../workflows/experiment-reproduction.md) covers carried identities and bundles; [sensitivity experiments](../workflows/sensitivity-and-synthetic-experiments.md) explains why a matrix can need multiple partition images. [Testing and development](../operations/testing-and-development.md) covers permitted compile venues.

No compilation, Apptainer build/inspection, GPU run, or SLURM submission was exercised for this page. The concrete gates described above are grounded in implementation, not a live compatibility certification for a cluster or image.

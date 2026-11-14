# Building container images

## Goal

Build the Apptainer images that container mode needs for an experiment.

## Prerequisites

- A build host, meaning a machine where `apptainer build --fakeroot` works, declared on the HPC system config of that cluster.
- An experiment definition, plus the HPC system config for the cluster that the images target.

## Key references

- [Configuration schema](../reference/config-schema.md) lists the fields of the build config.
- [Running an experiment definition](running-an-experiment-definition.md) describes the experiment definition.

## Declare the build host

Set `builds_containers` on the HPC system config of the build cluster.

```yaml
container:
  builds_containers: true
```

`hhemt build-sifs` refuses to run on a config without the declaration and names the key in the refusal. A target-only cluster leaves the key at the default `false`. Some clusters grant no fakeroot on the login nodes. Build the images for such a cluster elsewhere and place them under the cluster's `sif_root`.

## See what would be built

`build-sifs` plans the images of one experiment. Name an experiment definition with `--experiment` and plan first. A dry run writes no file.

```bash
hhemt build-sifs --build-hpc-config hpc/uva.yaml --sif-build-config sif_build.yaml \
  --experiment experiments/my_experiment --dry-run
```

`--experiment` is repeatable. The system config, the analysis config, and the HPC map of the experiment definition determine the targets. Without an experiment definition, name the system config, the analysis config, and the target HPC config as a positional triplet with `--system-config`, `--analysis-config`, and `--target-hpc-config`. The command refuses to run with neither form.

The output is the identity table, with a row for each image that the experiment matrix requires. The toolkit recomputes the identity from your configs. Each row names a family and a hardware target. The table shows the images the experiment needs and the images that exist already, so read the table before building.

## Build

Drop `--dry-run` to build. The options shape the build.

- `--only KEY` builds a subset. Take the KEY or a `FAMILY:HW` pair from the dry-run table.
- `--force` rebuilds identities with an existing image. A rebuild replaces an image only on success, so a failed rebuild leaves the working image in place.

By default the build starts as a detached driver and returns. Pass `--foreground` to build in the current shell. Use that form for a first build or to capture the output.

## Image location

Each image is written under the `sif_root` of your build config, at a path derived from the identity of the image, beside a `.manifest.json` that records the digest. Every reader resolves an image by identity. An analysis recomputes the identity from the configs of the analysis and resolves the same path, so after the toolkit moves on, the analysis finds no image built for an earlier commit.

A cluster without a build host receives an image by a copy of the `.sif` and the `.manifest.json` to the same relative path under the `sif_root` of that cluster. Preflight verifies the labels and the digest of the image and ignores the filename.

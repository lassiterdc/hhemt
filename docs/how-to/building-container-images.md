# Building container images

Container mode runs each simulation inside an Apptainer image. `hhemt build-sifs`
creates every image an experiment needs, addressed by identity rather than by
filename, so you never name an image file in a config.

For the fields of the build config itself, see
[Configuration schema](../reference/config-schema.md). This page is the workflow.

## Before you start: declare the build host

A build host is a machine where `apptainer build --fakeroot` works. Declare it on
that cluster's HPC-system config:

```yaml
container:
  builds_containers: true
```

`hhemt build-sifs` refuses to run on a config that does not declare it, naming the
key in the refusal. A target-only cluster leaves it `false`, which is the default:
Frontier login nodes grant no fakeroot, so images for that cluster are built
elsewhere and placed under its `sif_root`.

## See what would be built

`build-sifs` needs to know which experiment's images to plan, so name a bundle with
`--experiment`. Run the plan first; it writes nothing:

```bash
hhemt build-sifs --build-hpc-config hpc/uva.yaml --sif-build-config sif_build.yaml \
  --experiment experiments/my_experiment --dry-run
```

`--experiment` is repeatable, and the bundle's system and analysis configs plus its
HPC map are what derive the targets. With no bundle, name the three configs directly
instead, with `--system-config`, `--analysis-config` and `--target-hpc-config` as a
positional triplet. Naming neither is refused.

The output is the identity table: one row per image the experiment's matrix
requires, keyed by the identity the toolkit recomputes from your configs. Each row
names a family and a hardware target. Read it before building, because it tells you
how many images the run needs and which of them already exist.

## Build

Drop `--dry-run` to build. Two options shape what happens:

- `--only KEY` builds a subset. Take the KEY, or a `FAMILY:HW` pair, from the
  dry-run table.
- `--force` rebuilds identities whose images already exist. Images are replaced
  only on success, so a failed rebuild leaves the working image in place.

By default the build runs as a detached driver and returns. Pass `--foreground` to
run it in the current shell instead, which is what you want when you are watching a
first build or capturing its output.

## Where the images go

Each image is written under the `sif_root` your build config names, at a path
derived from its identity, beside a `.manifest.json` recording the digest. Nothing
reads an image by filename. A run recomputes the identity from its own configs and
resolves the same path, which is why an image built for one commit is not found
after the toolkit moves on.

To place an image on a cluster that cannot build, copy both the `.sif` and its
`.manifest.json` to the same relative path under that cluster's `sif_root`.
Preflight verifies the labels and the digest, not the filename.

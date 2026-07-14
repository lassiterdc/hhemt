# H&H Ensemble Modeling Toolkit (hhemt)

[![PyPI version](https://img.shields.io/pypi/v/hhemt.svg)](https://pypi.org/project/hhemt/)
[![Documentation Status](https://readthedocs.org/projects/hhemt/badge/?version=latest)](https://hhemt.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21359151.svg)](https://doi.org/10.5281/zenodo.21359151)

**hhemt** orchestrates coupled TRITON–SWMM flood-ensemble simulations from a
single Python interface, across both a local workstation and HPC (NVIDIA and
AMD GPUs).

- **Coupled TRITON–SWMM flood-ensemble orchestration** over the full
  preprocessing → compile → run → process → consolidate → report lifecycle.
- **Local + HPC execution** (NVIDIA/AMD), driven by a Snakemake workflow with a
  SLURM executor for cluster runs.
- **Interactive analysis report** plus a **portable render-bundle** so results
  travel without the source tree.

* PyPI: https://pypi.org/project/hhemt/
* Documentation: https://hhemt.readthedocs.io
* License: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)

## Installation

The full toolkit — including SWMM hydrology execution — is validated only against
the conda environment shipped in this repo. This is the supported install:

```bash
conda env create -n hhemt --file environment.yaml
conda activate hhemt
pip install --no-deps "swmmio==0.8.5"
pip install -e . --no-deps
```

Both `--no-deps` steps are required. The validated SWMM engine (`swmm-toolkit`
0.15.x + `pyswmm` 2.0.1) is only available from conda-forge and cannot be
expressed as pip metadata, so a plain `pip install hhemt` installs an unvalidated
SWMM stack: everything except SWMM *execution* works, and SWMM execution fails
closed with an actionable error pointing here. See
[`docs/how-to/installation.md`](docs/how-to/installation.md) for details.

## Usage

```bash
hhemt --help          # CLI entry point
```

See the [documentation](https://hhemt.readthedocs.io) for the end-to-end
analysis workflow.

## How to cite

If you use this software, please cite it via its Zenodo DOI. Citation metadata is
maintained in [`CITATION.cff`](CITATION.cff), which GitHub's "Cite this
repository" resolves. The DOI badge above always resolves to the latest version.

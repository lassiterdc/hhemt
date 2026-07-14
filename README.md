# H&H Ensemble Modeling Toolkit

![PyPI version](https://img.shields.io/pypi/v/hhemt.svg)
[![Documentation Status](https://readthedocs.org/projects/hhemt/badge/?version=latest)](https://hhemt.readthedocs.io/en/latest/?version=latest)

Tools for running and processing TRITON-SWMM models.

* PyPI package: https://pypi.org/project/hhemt/
* License: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
* Documentation: https://hhemt.readthedocs.io.

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

If you use this software, please cite it. Citation metadata is maintained in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" resolves it). Once the first release is published, cite the specific version via its Zenodo DOI:

<!-- The Zenodo concept-DOI badge is added when the first release DOI is minted (first-public-release):
[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER) -->

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

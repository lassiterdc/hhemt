# H&H Ensemble Modeling Toolkit, hhemt

[![PyPI version](https://img.shields.io/pypi/v/hhemt.svg)](https://pypi.org/project/hhemt/)
[![Documentation Status](https://readthedocs.org/projects/hhemt/badge/?version=latest)](https://hhemt.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21359151.svg)](https://doi.org/10.5281/zenodo.21359151)

**hhemt** orchestrates coupled TRITON-SWMM flood-ensemble simulations from a
single Python interface, on a local workstation and on HPC with NVIDIA and AMD
GPUs.

- **Lifecycle.** The workflow covers preprocessing, compilation, execution, processing, consolidation, and reporting.
- **Workflow engine.** Snakemake drives the workflow, with a SLURM executor for cluster execution.
- **Interactive analysis report** plus a **portable render-bundle** so results
  travel without the source tree.

* [PyPI](https://pypi.org/project/hhemt/)
* [Documentation](https://hhemt.readthedocs.io)
* [AGPL-3.0-or-later license](https://www.gnu.org/licenses/agpl-3.0.html)

<!-- hhemt:maturity-disclosure -->
> **On this release.** The software is documented and archived on Zenodo. General outside use falls outside the
> scope of this release.
<!-- /hhemt:maturity-disclosure -->

## Installation

The full toolkit, including SWMM hydrology execution, is validated only against
the conda environment shipped in this repo, and that environment is the supported
install. [Installation](https://hhemt.readthedocs.io/en/latest/how-to/installation/) gives the commands, states why both
`--no-deps` steps are required inside a conda environment, and covers the pip-only route.

## Usage

```bash
hhemt --help                                              # every command
hhemt run --system-config CFG_SYS --analysis-config CFG_ANA --dry-run
```

An analysis is driven by a [*system*](https://hhemt.readthedocs.io/en/latest/reference/glossary/#system) config and an
[*analysis*](https://hhemt.readthedocs.io/en/latest/reference/glossary/#analysis) config. `--dry-run` checks both before anything
runs. The [CLI reference](https://hhemt.readthedocs.io/en/latest/reference/cli/#running-an-analysis) describes the flag.

Pick the row that matches the job:

| You want to | Start at |
|---|---|
| Get something running | [Quickstart](https://hhemt.readthedocs.io/en/latest/quickstart/) |
| Run simulations and explore results in Python | [Running a simulation interactively in Python](https://hhemt.readthedocs.io/en/latest/tutorials/interactive-python/) |
| Look up a config field | [Configuration schema](https://hhemt.readthedocs.io/en/latest/reference/config-schema/) |
| Look up a command | [CLI reference](https://hhemt.readthedocs.io/en/latest/reference/cli/) |
| Read the contents of the outputs | [Output data model](https://hhemt.readthedocs.io/en/latest/reference/output-data-model/) |
| Work out why an analysis failed | [Diagnosing a failed analysis](https://hhemt.readthedocs.io/en/latest/how-to/diagnosing-a-failed-run/) |
| Know where the toolkit stops | [Limitations](https://hhemt.readthedocs.io/en/latest/explanation/limitations/) |
| Contribute a change | [Contributing](https://hhemt.readthedocs.io/en/latest/contributing/) |

## Citation

If you use this software, please cite the software through the Zenodo DOI. Citation metadata is
maintained in [`CITATION.cff`](CITATION.cff), and GitHub's "Cite this
repository" button reads that file.

**Cite the version you ran.** The [Citation](https://hhemt.readthedocs.io/en/latest/#how-to-cite) section of the
documentation explains the concept DOI and the version DOI.
